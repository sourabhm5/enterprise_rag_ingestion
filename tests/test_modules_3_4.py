"""
Test suite for Module 3 (S3 Storage) and Module 4 (Ingestion API).

Run with: pytest tests/test_modules_3_4.py -v
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestModule3S3Storage:
    """Tests for Module 3 - S3 Storage Abstraction."""
    
    def test_s3_path_type_enum(self):
        """Test S3PathType enumeration."""
        from storage.s3 import S3PathType
        
        assert S3PathType.RAW.value == "raw"
        assert S3PathType.DERIVED.value == "derived"
    
    def test_s3_storage_instantiation(self):
        """Test S3Storage can be instantiated."""
        from storage.s3 import S3Storage
        
        storage = S3Storage()
        assert storage is not None
        assert storage.config is not None
    
    def test_generate_path(self):
        """Test version-aware path generation."""
        from storage.s3 import S3Storage, S3PathType
        
        storage = S3Storage()
        
        # Raw path
        path = storage.generate_path(
            tenant_id="tenant-123",
            document_id="doc-456",
            version=1,
            path_type=S3PathType.RAW,
            filename="document.pdf"
        )
        assert path == "tenant-123/doc-456/v1/raw/document.pdf"
        
        # Derived path
        path = storage.generate_path(
            tenant_id="tenant-123",
            document_id="doc-456",
            version=2,
            path_type=S3PathType.DERIVED,
            filename="page_1.png"
        )
        assert path == "tenant-123/doc-456/v2/derived/page_1.png"
    
    def test_generate_prefix(self):
        """Test S3 prefix generation for listing."""
        from storage.s3 import S3Storage, S3PathType
        
        storage = S3Storage()
        
        # Document-level prefix
        prefix = storage.generate_prefix("tenant-123", "doc-456")
        assert prefix == "tenant-123/doc-456/"
        
        # Version-level prefix
        prefix = storage.generate_prefix("tenant-123", "doc-456", version=1)
        assert prefix == "tenant-123/doc-456/v1/"
        
        # Path-type level prefix
        prefix = storage.generate_prefix(
            "tenant-123", "doc-456", version=1, path_type=S3PathType.RAW
        )
        assert prefix == "tenant-123/doc-456/v1/raw/"
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        from storage.s3 import S3Storage
        
        content = b"Hello, World!"
        hash1 = S3Storage.compute_content_hash(content)
        
        # Should be deterministic
        hash2 = S3Storage.compute_content_hash(content)
        assert hash1 == hash2
        
        # Should be 64 characters (SHA-256 hex)
        assert len(hash1) == 64
        
        # Different content should produce different hash
        hash3 = S3Storage.compute_content_hash(b"Different content")
        assert hash3 != hash1
    
    def test_s3_object_info_dataclass(self):
        """Test S3ObjectInfo dataclass."""
        from storage.s3 import S3ObjectInfo
        
        info = S3ObjectInfo(
            key="test/path/file.pdf",
            size=1024,
            last_modified=datetime.utcnow(),
            etag="abc123",
            content_type="application/pdf"
        )
        
        assert info.key == "test/path/file.pdf"
        assert info.size == 1024
        assert info.etag == "abc123"
    
    def test_upload_result_dataclass(self):
        """Test UploadResult dataclass."""
        from storage.s3 import UploadResult
        
        result = UploadResult(
            key="test/path/file.pdf",
            bucket="test-bucket",
            etag="abc123",
            content_hash="def456"
        )
        
        assert result.key == "test/path/file.pdf"
        assert result.bucket == "test-bucket"
        assert result.version_id is None
    
    def test_s3_error_classes(self):
        """Test S3 exception classes."""
        from storage.s3 import (
            S3StorageError,
            S3NotFoundError,
            S3UploadError,
            S3DeleteError
        )
        
        # Test inheritance
        assert issubclass(S3NotFoundError, S3StorageError)
        assert issubclass(S3UploadError, S3StorageError)
        assert issubclass(S3DeleteError, S3StorageError)
        
        # Test instantiation
        error = S3NotFoundError("Object not found: test/path")
        assert "Object not found" in str(error)
    
    def test_guess_content_type(self):
        """Test MIME type guessing."""
        from storage.s3 import S3Storage
        
        assert S3Storage._guess_content_type("document.pdf") == "application/pdf"
        assert S3Storage._guess_content_type("image.png") == "image/png"
        assert S3Storage._guess_content_type("image.jpg") == "image/jpeg"
        assert S3Storage._guess_content_type("unknown.xyz") == "application/octet-stream"


class TestModule4IngestionAPI:
    """Tests for Module 4 - Ingestion API."""
    
    def test_rbac_config_model(self):
        """Test RBACConfig model."""
        from api.routes.documents import RBACConfig
        
        # From lists
        config = RBACConfig(
            allowed_roles=["admin", "editor"],
            allowed_users=["user-1", "user-2"]
        )
        assert "admin" in config.allowed_roles
        assert "user-1" in config.allowed_users
        
        # From comma-separated string
        config = RBACConfig(
            allowed_roles="admin, editor",
            allowed_users="user-1, user-2"
        )
        assert "admin" in config.allowed_roles
        assert "user-1" in config.allowed_users
    
    def test_document_response_model(self):
        """Test DocumentResponse model."""
        from api.routes.documents import DocumentResponse
        
        response = DocumentResponse(
            id=str(uuid.uuid4()),
            document_id="doc-123",
            tenant_id="tenant-456",
            version=1,
            status="ACTIVE",
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            content_hash="abc123",
            classification="INTERNAL",
            s3_raw_path="tenant-456/doc-123/v1/raw/test.pdf",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            allowed_roles=["admin"],
            allowed_users=["user-1"]
        )
        
        assert response.document_id == "doc-123"
        assert response.version == 1
    
    def test_delete_response_model(self):
        """Test DeleteResponse model."""
        from api.routes.documents import DeleteResponse
        
        response = DeleteResponse(
            document_id="doc-123",
            deleted_versions=[1, 2, 3],
            message="Successfully deleted 3 versions"
        )
        
        assert len(response.deleted_versions) == 3
    
    def test_ingestion_job_response_model(self):
        """Test IngestionJobResponse model."""
        from api.routes.documents import IngestionJobResponse
        
        response = IngestionJobResponse(
            job_id="job-123",
            document_id="doc-456",
            status="RUNNING",
            current_stage="text_pipeline",
            completed_stages=["routing", "layout_parsing"],
            error_log=[],
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            completed_at=None
        )
        
        assert response.status == "RUNNING"
        assert len(response.completed_stages) == 2
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        from api.routes.documents import compute_file_hash
        
        content = b"Test file content"
        hash1 = compute_file_hash(content)
        hash2 = compute_file_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_document_list_response_model(self):
        """Test DocumentListResponse model."""
        from api.routes.documents import DocumentListResponse, DocumentResponse
        
        response = DocumentListResponse(
            documents=[],
            total=100,
            page=1,
            page_size=20,
            has_more=True
        )
        
        assert response.total == 100
        assert response.has_more is True


class TestAPIAuthContext:
    """Tests for API authentication context."""
    
    def test_auth_context_dataclass(self):
        """Test AuthContext dataclass."""
        from api.dependencies import AuthContext
        
        ctx = AuthContext(
            tenant_id="tenant-123",
            user_id="user-456",
            roles=["admin", "editor"],
            is_admin=False
        )
        
        assert ctx.tenant_id == "tenant-123"
        assert ctx.has_role("admin") is True
        assert ctx.has_role("viewer") is False
    
    def test_auth_context_admin_override(self):
        """Test AuthContext admin privileges."""
        from api.dependencies import AuthContext
        
        ctx = AuthContext(
            tenant_id="tenant-123",
            user_id="user-456",
            roles=[],
            is_admin=True
        )
        
        # Admin should have any role
        assert ctx.has_role("any_role") is True
    
    def test_auth_context_classification_access(self):
        """Test AuthContext classification level access."""
        from api.dependencies import AuthContext
        
        # Regular user with internal access
        ctx = AuthContext(
            tenant_id="tenant-123",
            user_id="user-456",
            roles=["internal_access"],
            is_admin=False
        )
        
        assert ctx.can_access_classification("PUBLIC") is True
        assert ctx.can_access_classification("INTERNAL") is True
        assert ctx.can_access_classification("CONFIDENTIAL") is False
        
        # Admin should access everything
        admin_ctx = AuthContext(
            tenant_id="tenant-123",
            user_id="admin-user",
            roles=[],
            is_admin=True
        )
        
        assert admin_ctx.can_access_classification("RESTRICTED") is True


class TestAPIRequestContext:
    """Tests for API request context."""
    
    def test_request_context_dataclass(self):
        """Test RequestContext dataclass."""
        from api.dependencies import RequestContext, AuthContext
        
        auth = AuthContext(
            tenant_id="tenant-123",
            user_id="user-456",
            roles=["admin"],
            is_admin=False
        )
        
        ctx = RequestContext(
            db=MagicMock(),
            storage=MagicMock(),
            settings=MagicMock(),
            auth=auth
        )
        
        assert ctx.auth.tenant_id == "tenant-123"


class TestModule3And4Integration:
    """Integration tests for Modules 3 and 4."""
    
    def test_s3_path_in_document_response(self):
        """Test S3 path format consistency."""
        from storage.s3 import S3Storage, S3PathType
        from api.routes.documents import DocumentResponse
        
        storage = S3Storage()
        
        # Generate path
        path = storage.generate_path(
            tenant_id="tenant-123",
            document_id="doc-456",
            version=1,
            path_type=S3PathType.RAW,
            filename="test.pdf"
        )
        
        # Use in response
        response = DocumentResponse(
            id=str(uuid.uuid4()),
            document_id="doc-456",
            tenant_id="tenant-123",
            version=1,
            status="ACTIVE",
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            content_hash="abc123",
            classification="INTERNAL",
            s3_raw_path=path,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert response.s3_raw_path == "tenant-123/doc-456/v1/raw/test.pdf"
    
    def test_content_hash_consistency(self):
        """Test content hash is consistent between S3 and API."""
        from storage.s3 import S3Storage
        from api.routes.documents import compute_file_hash
        
        content = b"Test document content"
        
        s3_hash = S3Storage.compute_content_hash(content)
        api_hash = compute_file_hash(content)
        
        assert s3_hash == api_hash


class TestFastAPIApp:
    """Tests for FastAPI application setup."""
    
    def test_app_creation(self):
        """Test application can be created."""
        from api.main import create_app
        
        app = create_app()
        assert app is not None
        assert app.title == "Enterprise RAG Ingestion Pipeline"
    
    def test_app_routes_registered(self):
        """Test that routes are registered."""
        from api.main import app
        
        routes = [route.path for route in app.routes]
        
        # Check health routes
        assert "/health" in routes
        assert "/health/ready" in routes
        assert "/health/live" in routes
        
        # Check document routes
        assert "/api/v1/documents" in routes or any(
            "/api/v1/documents" in r for r in routes
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
