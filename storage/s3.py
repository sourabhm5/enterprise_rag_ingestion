"""
MODULE 3 — S3 STORAGE ABSTRACTION
=================================
Raw + derived asset lifecycle consistency for the Enterprise RAG Ingestion Pipeline.

Features:
- Version-aware paths: {tenant}/{document_id}/v{version}/raw|derived/...
- Deletion hooks for document lifecycle
- Content hash for idempotency
- Async operations with aioboto3
- Support for both AWS S3 and MinIO

Path Structure:
    {tenant_id}/
        {document_id}/
            v{version}/
                raw/
                    original.pdf
                derived/
                    page_1.png
                    page_2.png
                    extracted_image_001.png
"""

import hashlib
import io
import mimetypes
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, BinaryIO, Dict, List, Optional, Union

import aioboto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config.settings import S3Config, get_settings


class S3PathType(str, Enum):
    """Types of S3 paths for document storage."""
    RAW = "raw"
    DERIVED = "derived"


@dataclass
class S3ObjectInfo:
    """Information about an S3 object."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class UploadResult:
    """Result of an S3 upload operation."""
    key: str
    bucket: str
    etag: str
    version_id: Optional[str] = None
    content_hash: str = ""


class S3StorageError(Exception):
    """Base exception for S3 storage errors."""
    pass


class S3NotFoundError(S3StorageError):
    """Raised when an S3 object is not found."""
    pass


class S3UploadError(S3StorageError):
    """Raised when an S3 upload fails."""
    pass


class S3DeleteError(S3StorageError):
    """Raised when an S3 delete operation fails."""
    pass


class S3Storage:
    """
    S3 Storage abstraction for document lifecycle management.
    
    Provides version-aware storage with support for:
    - Raw document storage
    - Derived asset storage (images, extracted content)
    - Lifecycle management (create, update, delete)
    - Content hashing for idempotency
    """
    
    def __init__(self, config: Optional[S3Config] = None):
        """
        Initialize S3 storage.
        
        Args:
            config: S3 configuration. If None, uses settings from config module.
        """
        self.config = config or get_settings().s3
        self._session = aioboto3.Session()
        
        # Configure boto client
        self._client_config = Config(
            max_pool_connections=self.config.max_pool_connections,
            retries={"max_attempts": 3, "mode": "adaptive"}
        )
    
    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[Any, None]:
        """Get an async S3 client context manager."""
        async with self._session.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region,
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            use_ssl=self.config.use_ssl,
            config=self._client_config,
        ) as client:
            yield client
    
    # =========================================================================
    # Path Generation
    # =========================================================================
    
    def generate_path(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        path_type: S3PathType,
        filename: str
    ) -> str:
        """
        Generate a version-aware S3 path.
        
        Format: {tenant_id}/{document_id}/v{version}/{raw|derived}/{filename}
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Document version number
            path_type: Raw or derived path type
            filename: File name
            
        Returns:
            Full S3 key path
        """
        return f"{tenant_id}/{document_id}/v{version}/{path_type.value}/{filename}"
    
    def generate_prefix(
        self,
        tenant_id: str,
        document_id: str,
        version: Optional[int] = None,
        path_type: Optional[S3PathType] = None
    ) -> str:
        """
        Generate an S3 prefix for listing/deletion operations.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Optional version number (if None, matches all versions)
            path_type: Optional path type (if None, matches raw and derived)
            
        Returns:
            S3 prefix string
        """
        prefix = f"{tenant_id}/{document_id}/"
        if version is not None:
            prefix += f"v{version}/"
            if path_type is not None:
                prefix += f"{path_type.value}/"
        return prefix
    
    # =========================================================================
    # Content Hashing
    # =========================================================================
    
    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        """
        Compute SHA-256 hash of content for idempotency.
        
        Args:
            content: Binary content to hash
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    async def compute_stream_hash(stream: BinaryIO, chunk_size: int = 8192) -> str:
        """
        Compute SHA-256 hash of a stream.
        
        Args:
            stream: Binary stream to hash
            chunk_size: Size of chunks to read
            
        Returns:
            Hex-encoded SHA-256 hash
        """
        hasher = hashlib.sha256()
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
        stream.seek(0)  # Reset stream position
        return hasher.hexdigest()
    
    # =========================================================================
    # Upload Operations
    # =========================================================================
    
    async def upload_raw_document(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """
        Upload a raw document to S3.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Document version
            filename: Original filename
            content: File content as bytes
            content_type: MIME type (auto-detected if None)
            metadata: Additional metadata to store
            
        Returns:
            UploadResult with key, etag, and content hash
        """
        key = self.generate_path(
            tenant_id=tenant_id,
            document_id=document_id,
            version=version,
            path_type=S3PathType.RAW,
            filename=filename
        )
        
        return await self._upload(
            key=key,
            content=content,
            content_type=content_type or self._guess_content_type(filename),
            metadata=metadata
        )
    
    async def upload_derived_asset(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """
        Upload a derived asset (extracted image, etc.) to S3.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Document version
            filename: Asset filename
            content: File content as bytes
            content_type: MIME type (auto-detected if None)
            metadata: Additional metadata to store
            
        Returns:
            UploadResult with key, etag, and content hash
        """
        key = self.generate_path(
            tenant_id=tenant_id,
            document_id=document_id,
            version=version,
            path_type=S3PathType.DERIVED,
            filename=filename
        )
        
        return await self._upload(
            key=key,
            content=content,
            content_type=content_type or self._guess_content_type(filename),
            metadata=metadata
        )
    
    async def _upload(
        self,
        key: str,
        content: bytes,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """
        Internal upload method.
        
        Args:
            key: S3 object key
            content: File content
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            UploadResult
        """
        content_hash = self.compute_content_hash(content)
        
        # Prepare metadata
        upload_metadata = metadata or {}
        upload_metadata["content-hash"] = content_hash
        upload_metadata["upload-timestamp"] = datetime.utcnow().isoformat()
        
        try:
            async with self._get_client() as client:
                response = await client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=content,
                    ContentType=content_type,
                    Metadata=upload_metadata
                )
                
                return UploadResult(
                    key=key,
                    bucket=self.config.bucket_name,
                    etag=response.get("ETag", "").strip('"'),
                    version_id=response.get("VersionId"),
                    content_hash=content_hash
                )
        except ClientError as e:
            raise S3UploadError(f"Failed to upload {key}: {e}")
    
    async def upload_stream(
        self,
        key: str,
        stream: BinaryIO,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> UploadResult:
        """
        Upload a stream to S3 using multipart upload.
        
        Args:
            key: S3 object key
            stream: Binary stream
            content_type: MIME type
            metadata: Optional metadata
            
        Returns:
            UploadResult
        """
        content = stream.read()
        stream.seek(0)
        return await self._upload(key, content, content_type, metadata)
    
    # =========================================================================
    # Download Operations
    # =========================================================================
    
    async def download(self, key: str) -> bytes:
        """
        Download an object from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            Object content as bytes
        """
        try:
            async with self._get_client() as client:
                response = await client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
                return await response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise S3NotFoundError(f"Object not found: {key}")
            raise S3StorageError(f"Failed to download {key}: {e}")
    
    async def download_raw_document(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        filename: str
    ) -> bytes:
        """
        Download a raw document from S3.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Document version
            filename: Original filename
            
        Returns:
            Document content as bytes
        """
        key = self.generate_path(
            tenant_id=tenant_id,
            document_id=document_id,
            version=version,
            path_type=S3PathType.RAW,
            filename=filename
        )
        return await self.download(key)
    
    async def get_object_info(self, key: str) -> S3ObjectInfo:
        """
        Get metadata about an S3 object.
        
        Args:
            key: S3 object key
            
        Returns:
            S3ObjectInfo with object metadata
        """
        try:
            async with self._get_client() as client:
                response = await client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
                return S3ObjectInfo(
                    key=key,
                    size=response["ContentLength"],
                    last_modified=response["LastModified"],
                    etag=response["ETag"].strip('"'),
                    content_type=response.get("ContentType"),
                    metadata=response.get("Metadata")
                )
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise S3NotFoundError(f"Object not found: {key}")
            raise S3StorageError(f"Failed to get info for {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        """
        Check if an object exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            await self.get_object_info(key)
            return True
        except S3NotFoundError:
            return False
    
    # =========================================================================
    # List Operations
    # =========================================================================
    
    async def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000
    ) -> List[S3ObjectInfo]:
        """
        List objects with a given prefix.
        
        Args:
            prefix: S3 key prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of S3ObjectInfo
        """
        objects = []
        
        try:
            async with self._get_client() as client:
                paginator = client.get_paginator("list_objects_v2")
                
                async for page in paginator.paginate(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    PaginationConfig={"MaxItems": max_keys}
                ):
                    for obj in page.get("Contents", []):
                        objects.append(S3ObjectInfo(
                            key=obj["Key"],
                            size=obj["Size"],
                            last_modified=obj["LastModified"],
                            etag=obj["ETag"].strip('"')
                        ))
                        
            return objects
        except ClientError as e:
            raise S3StorageError(f"Failed to list objects with prefix {prefix}: {e}")
    
    async def list_document_versions(
        self,
        tenant_id: str,
        document_id: str
    ) -> List[int]:
        """
        List all versions of a document.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            
        Returns:
            List of version numbers
        """
        prefix = self.generate_prefix(tenant_id, document_id)
        objects = await self.list_objects(prefix)
        
        versions = set()
        for obj in objects:
            # Extract version from path: tenant/doc/v{N}/...
            parts = obj.key.split("/")
            if len(parts) >= 3:
                version_part = parts[2]  # v{N}
                if version_part.startswith("v"):
                    try:
                        versions.add(int(version_part[1:]))
                    except ValueError:
                        continue
        
        return sorted(versions)
    
    # =========================================================================
    # Delete Operations
    # =========================================================================
    
    async def delete_object(self, key: str) -> bool:
        """
        Delete a single object from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if deleted successfully
        """
        try:
            async with self._get_client() as client:
                await client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
                return True
        except ClientError as e:
            raise S3DeleteError(f"Failed to delete {key}: {e}")
    
    async def delete_objects(self, keys: List[str]) -> Dict[str, Any]:
        """
        Delete multiple objects from S3.
        
        Args:
            keys: List of S3 object keys
            
        Returns:
            Dict with 'deleted' and 'errors' lists
        """
        if not keys:
            return {"deleted": [], "errors": []}
        
        # S3 delete_objects has a limit of 1000 keys per request
        deleted = []
        errors = []
        
        try:
            async with self._get_client() as client:
                # Process in batches of 1000
                for i in range(0, len(keys), 1000):
                    batch = keys[i:i + 1000]
                    response = await client.delete_objects(
                        Bucket=self.config.bucket_name,
                        Delete={
                            "Objects": [{"Key": k} for k in batch],
                            "Quiet": False
                        }
                    )
                    
                    deleted.extend([
                        obj["Key"] for obj in response.get("Deleted", [])
                    ])
                    errors.extend(response.get("Errors", []))
                    
            return {"deleted": deleted, "errors": errors}
        except ClientError as e:
            raise S3DeleteError(f"Failed to delete objects: {e}")
    
    async def delete_document_version(
        self,
        tenant_id: str,
        document_id: str,
        version: int
    ) -> Dict[str, Any]:
        """
        Delete all objects for a specific document version.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            version: Document version to delete
            
        Returns:
            Dict with 'deleted' and 'errors' lists
        """
        prefix = self.generate_prefix(tenant_id, document_id, version)
        objects = await self.list_objects(prefix)
        keys = [obj.key for obj in objects]
        
        if not keys:
            return {"deleted": [], "errors": []}
        
        return await self.delete_objects(keys)
    
    async def delete_all_document_versions(
        self,
        tenant_id: str,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Delete all versions of a document (complete removal).
        
        This is called when a document is permanently deleted.
        
        Args:
            tenant_id: Tenant identifier
            document_id: Document identifier
            
        Returns:
            Dict with 'deleted' and 'errors' lists
        """
        prefix = self.generate_prefix(tenant_id, document_id)
        objects = await self.list_objects(prefix, max_keys=10000)
        keys = [obj.key for obj in objects]
        
        if not keys:
            return {"deleted": [], "errors": []}
        
        return await self.delete_objects(keys)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    @staticmethod
    def _guess_content_type(filename: str) -> str:
        """Guess MIME type from filename."""
        content_type, _ = mimetypes.guess_type(filename)
        return content_type or "application/octet-stream"
    
    async def ensure_bucket_exists(self) -> bool:
        """
        Ensure the configured bucket exists, create if not.
        
        Returns:
            True if bucket exists or was created
        """
        try:
            async with self._get_client() as client:
                try:
                    await client.head_bucket(Bucket=self.config.bucket_name)
                    return True
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        # Create bucket
                        await client.create_bucket(
                            Bucket=self.config.bucket_name,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.config.region
                            } if self.config.region != "us-east-1" else {}
                        )
                        return True
                    raise
        except ClientError as e:
            raise S3StorageError(f"Failed to ensure bucket exists: {e}")
    
    async def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        http_method: str = "GET"
    ) -> str:
        """
        Generate a presigned URL for an object.
        
        Args:
            key: S3 object key
            expiration: URL expiration in seconds
            http_method: HTTP method (GET or PUT)
            
        Returns:
            Presigned URL string
        """
        try:
            async with self._get_client() as client:
                client_method = "get_object" if http_method == "GET" else "put_object"
                url = await client.generate_presigned_url(
                    client_method,
                    Params={
                        "Bucket": self.config.bucket_name,
                        "Key": key
                    },
                    ExpiresIn=expiration
                )
                return url
        except ClientError as e:
            raise S3StorageError(f"Failed to generate presigned URL: {e}")


# ============================================================================
# Global Instance & Dependency
# ============================================================================

_s3_storage: Optional[S3Storage] = None


def get_s3_storage() -> S3Storage:
    """Get or create the global S3 storage instance."""
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage()
    return _s3_storage


async def get_s3_storage_dependency() -> S3Storage:
    """FastAPI dependency for S3 storage injection."""
    return get_s3_storage()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "S3Storage",
    "S3PathType",
    "S3ObjectInfo",
    "UploadResult",
    "S3StorageError",
    "S3NotFoundError",
    "S3UploadError",
    "S3DeleteError",
    "get_s3_storage",
    "get_s3_storage_dependency",
]
