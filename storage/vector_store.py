"""
MODULE 15 — VECTOR STORE (RBAC-ENFORCED)
========================================
Vector store abstraction for the Enterprise RAG Ingestion Pipeline.

Requirements:
- Store ACL metadata with vectors
- Enforce filters at query time
- Support deletion by document_id + version

Uses Qdrant as the primary vector database.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pipelines.base import BasePipeline
from schema.intermediate_representation import (
    Chunk,
    IngestionDocument,
)
from config.settings import get_settings, VectorDBType
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Vector Store Types
# ============================================================================

@dataclass
class VectorPoint:
    """A point to store in the vector database."""
    id: str
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


@dataclass
class SearchFilter:
    """
    Search filter for RBAC enforcement.
    
    Filters are AND-ed together.
    """
    tenant_id: Optional[str] = None
    allowed_roles: Optional[List[str]] = None
    allowed_users: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    classifications: Optional[List[str]] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_qdrant_filter(self) -> Dict[str, Any]:
        """Convert to Qdrant filter format."""
        must_conditions = []
        
        if self.tenant_id:
            must_conditions.append({
                "key": "tenant_id",
                "match": {"value": self.tenant_id}
            })
        
        if self.document_ids:
            must_conditions.append({
                "key": "document_id",
                "match": {"any": self.document_ids}
            })
        
        if self.classifications:
            must_conditions.append({
                "key": "classification",
                "match": {"any": self.classifications}
            })
        
        # RBAC: User must have access via role OR direct user permission
        if self.allowed_roles or self.allowed_users:
            should_conditions = []
            
            if self.allowed_roles:
                for role in self.allowed_roles:
                    should_conditions.append({
                        "key": "allowed_roles",
                        "match": {"any": [role]}
                    })
            
            if self.allowed_users:
                for user in self.allowed_users:
                    should_conditions.append({
                        "key": "allowed_users",
                        "match": {"any": [user]}
                    })
            
            if should_conditions:
                must_conditions.append({
                    "should": should_conditions
                })
        
        # Custom metadata filters
        for key, value in self.metadata_filters.items():
            if isinstance(value, list):
                must_conditions.append({
                    "key": key,
                    "match": {"any": value}
                })
            else:
                must_conditions.append({
                    "key": key,
                    "match": {"value": value}
                })
        
        if not must_conditions:
            return {}
        
        return {"must": must_conditions}


# ============================================================================
# Vector Store Interface
# ============================================================================

class VectorStoreBase(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: str = "cosine"
    ) -> bool:
        """Create a collection."""
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass
    
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        collection_name: str,
        points: List[VectorPoint]
    ) -> int:
        """Upsert points into collection."""
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[SearchFilter] = None,
        with_vectors: bool = False
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_by_filter(
        self,
        collection_name: str,
        filter: SearchFilter
    ) -> int:
        """Delete points matching filter."""
        pass
    
    @abstractmethod
    async def get_by_ids(
        self,
        collection_name: str,
        ids: List[str],
        with_vectors: bool = False
    ) -> List[VectorPoint]:
        """Get points by IDs."""
        pass


# ============================================================================
# Qdrant Vector Store
# ============================================================================

class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant vector store implementation.
    
    Features:
    - RBAC-enforced filtering
    - Efficient batch operations
    - Support for document versioning
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False,
        grpc_port: Optional[int] = None,
        prefer_grpc: bool = False
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: API key for authentication
            https: Use HTTPS
            grpc_port: gRPC port (optional)
            prefer_grpc: Prefer gRPC over HTTP
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.https = https
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self._client = None
    
    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                https=self.https,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc
            )
        
        return self._client
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: str = "cosine"
    ) -> bool:
        """Create a Qdrant collection."""
        from qdrant_client.models import Distance, VectorParams
        
        client = self._get_client()
        
        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        qdrant_distance = distance_map.get(distance.lower(), Distance.COSINE)
        
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=qdrant_distance
                )
            )
            
            # Create payload indexes for efficient filtering
            self._create_payload_indexes(collection_name)
            
            return True
        except Exception as e:
            # Collection might already exist
            if "already exists" in str(e).lower():
                return True
            raise
    
    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create indexes for common filter fields."""
        from qdrant_client.models import PayloadSchemaType
        
        client = self._get_client()
        
        # Index fields used for RBAC filtering
        index_fields = [
            ("tenant_id", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("document_version", PayloadSchemaType.INTEGER),
            ("classification", PayloadSchemaType.KEYWORD),
            ("allowed_roles", PayloadSchemaType.KEYWORD),
            ("allowed_users", PayloadSchemaType.KEYWORD),
        ]
        
        for field_name, field_type in index_fields:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
            except Exception:
                # Index might already exist
                pass
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection."""
        client = self._get_client()
        
        try:
            client.delete_collection(collection_name)
            return True
        except Exception:
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        client = self._get_client()
        
        try:
            collections = client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception:
            return False
    
    async def upsert(
        self,
        collection_name: str,
        points: List[VectorPoint]
    ) -> int:
        """Upsert points into Qdrant."""
        from qdrant_client.models import PointStruct
        
        if not points:
            return 0
        
        client = self._get_client()
        
        # Convert to Qdrant format
        qdrant_points = [
            PointStruct(
                id=p.id,
                vector=p.vector,
                payload=p.payload
            )
            for p in points
        ]
        
        # Upsert in batches
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(qdrant_points), batch_size):
            batch = qdrant_points[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            total_upserted += len(batch)
        
        return total_upserted
    
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter: Optional[SearchFilter] = None,
        with_vectors: bool = False
    ) -> List[SearchResult]:
        """Search for similar vectors with RBAC filtering."""
        from qdrant_client.models import Filter
        
        client = self._get_client()
        
        # Build filter
        qdrant_filter = None
        if filter:
            filter_dict = filter.to_qdrant_filter()
            if filter_dict:
                qdrant_filter = Filter(**filter_dict)
        
        # Search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=with_vectors
        )
        
        # Convert results
        return [
            SearchResult(
                id=str(r.id),
                score=r.score,
                payload=r.payload or {},
                vector=r.vector if with_vectors else None
            )
            for r in results
        ]
    
    async def delete_by_filter(
        self,
        collection_name: str,
        filter: SearchFilter
    ) -> int:
        """Delete points matching filter."""
        from qdrant_client.models import Filter, FilterSelector
        
        client = self._get_client()
        
        filter_dict = filter.to_qdrant_filter()
        if not filter_dict:
            return 0
        
        qdrant_filter = Filter(**filter_dict)
        
        # Count before deletion (for return value)
        count_before = client.count(
            collection_name=collection_name,
            count_filter=qdrant_filter
        ).count
        
        # Delete
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=qdrant_filter)
        )
        
        return count_before
    
    async def delete_by_document(
        self,
        collection_name: str,
        document_id: str,
        version: Optional[int] = None,
        tenant_id: Optional[str] = None
    ) -> int:
        """
        Delete all vectors for a document.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            version: Specific version (optional)
            tenant_id: Tenant ID (optional, for safety)
            
        Returns:
            Number of deleted points
        """
        filter = SearchFilter(
            document_ids=[document_id],
            tenant_id=tenant_id
        )
        
        if version is not None:
            filter.metadata_filters["document_version"] = version
        
        return await self.delete_by_filter(collection_name, filter)
    
    async def get_by_ids(
        self,
        collection_name: str,
        ids: List[str],
        with_vectors: bool = False
    ) -> List[VectorPoint]:
        """Get points by IDs."""
        client = self._get_client()
        
        results = client.retrieve(
            collection_name=collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=with_vectors
        )
        
        return [
            VectorPoint(
                id=str(r.id),
                vector=r.vector if with_vectors else [],
                payload=r.payload or {}
            )
            for r in results
        ]
    
    async def get_collection_info(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """Get collection information."""
        client = self._get_client()
        
        info = client.get_collection(collection_name)
        
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value
            }
        }


# ============================================================================
# Vector Store Service
# ============================================================================

class VectorStoreService:
    """
    High-level vector store service.
    
    Provides:
    - Document indexing
    - RBAC-aware search
    - Document deletion
    """
    
    def __init__(
        self,
        store: Optional[VectorStoreBase] = None,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize vector store service.
        
        Args:
            store: Vector store implementation
            collection_name: Collection name
            dimension: Vector dimension
        """
        settings = get_settings()
        
        # Initialize store
        if store:
            self.store = store
        else:
            self.store = QdrantVectorStore(
                host=settings.vector_db.host,
                port=settings.vector_db.port,
                api_key=settings.vector_db.api_key
            )
        
        self.collection_name = collection_name or settings.vector_db.collection_name
        self.dimension = dimension or settings.get_embedding_model().dimensions
    
    async def ensure_collection(self) -> bool:
        """Ensure collection exists."""
        exists = await self.store.collection_exists(self.collection_name)
        if not exists:
            return await self.store.create_collection(
                self.collection_name,
                self.dimension
            )
        return True
    
    async def index_chunks(
        self,
        chunks: List[Chunk],
        document_id: str,
        tenant_id: str,
        document_version: int = 1
    ) -> int:
        """
        Index chunks into vector store.
        
        Args:
            chunks: Chunks to index
            document_id: Document ID
            tenant_id: Tenant ID
            document_version: Document version
            
        Returns:
            Number of indexed chunks
        """
        await self.ensure_collection()
        
        points = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            
            # Build payload with ACL metadata
            payload = {
                "chunk_id": chunk.id,
                "document_id": document_id,
                "document_version": document_version,
                "tenant_id": tenant_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "image_refs": chunk.image_refs,
                "source_node_ids": chunk.source_node_ids,
                "indexed_at": datetime.utcnow().isoformat(),
            }
            
            # Add ACL fields
            if chunk.acl:
                payload["allowed_roles"] = chunk.acl.get("allowed_roles", [])
                payload["allowed_users"] = chunk.acl.get("allowed_users", [])
                payload["classification"] = chunk.acl.get("classification", "INTERNAL")
            
            points.append(VectorPoint(
                id=chunk.id,
                vector=chunk.embedding,
                payload=payload
            ))
        
        if not points:
            return 0
        
        return await self.store.upsert(self.collection_name, points)
    
    async def search(
        self,
        query_vector: List[float],
        tenant_id: str,
        user_id: Optional[str] = None,
        user_roles: Optional[List[str]] = None,
        limit: int = 10,
        document_ids: Optional[List[str]] = None,
        classifications: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks with RBAC enforcement.
        
        Args:
            query_vector: Query embedding
            tenant_id: Tenant ID (required)
            user_id: User ID for access control
            user_roles: User roles for access control
            limit: Number of results
            document_ids: Filter by document IDs
            classifications: Filter by classifications
            
        Returns:
            Search results
        """
        # Build RBAC filter
        filter = SearchFilter(
            tenant_id=tenant_id,
            allowed_roles=user_roles,
            allowed_users=[user_id] if user_id else None,
            document_ids=document_ids,
            classifications=classifications
        )
        
        return await self.store.search(
            self.collection_name,
            query_vector,
            limit=limit,
            filter=filter
        )
    
    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
        version: Optional[int] = None
    ) -> int:
        """
        Delete document vectors.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            version: Specific version (deletes all if not specified)
            
        Returns:
            Number of deleted vectors
        """
        if isinstance(self.store, QdrantVectorStore):
            return await self.store.delete_by_document(
                self.collection_name,
                document_id,
                version=version,
                tenant_id=tenant_id
            )
        
        # Fallback for other stores
        filter = SearchFilter(
            tenant_id=tenant_id,
            document_ids=[document_id]
        )
        if version is not None:
            filter.metadata_filters["document_version"] = version
        
        return await self.store.delete_by_filter(self.collection_name, filter)
    
    async def get_document_chunks(
        self,
        document_id: str,
        tenant_id: str,
        version: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            version: Specific version
            
        Returns:
            List of chunk payloads
        """
        # This requires scrolling through all matching points
        # For Qdrant, we use the scroll endpoint
        if isinstance(self.store, QdrantVectorStore):
            from qdrant_client.models import Filter
            
            client = self.store._get_client()
            
            filter_dict = SearchFilter(
                tenant_id=tenant_id,
                document_ids=[document_id]
            ).to_qdrant_filter()
            
            if version is not None:
                filter_dict["must"].append({
                    "key": "document_version",
                    "match": {"value": version}
                })
            
            qdrant_filter = Filter(**filter_dict)
            
            results = []
            offset = None
            
            while True:
                response = client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True
                )
                
                points, next_offset = response
                results.extend([p.payload for p in points])
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return results
        
        return []


# ============================================================================
# Vector Store Pipeline
# ============================================================================

@register_pipeline("vector_store")
class VectorStorePipeline(BasePipeline):
    """
    Pipeline stage for indexing into vector store.
    
    Indexes embeddings with:
    - ACL metadata
    - Document metadata
    - Chunk information
    """
    
    stage_name = "vector_store"
    
    def __init__(
        self,
        store: Optional[VectorStoreBase] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize vector store pipeline.
        
        Args:
            store: Vector store implementation
            collection_name: Collection name
        """
        super().__init__()
        self.service = VectorStoreService(
            store=store,
            collection_name=collection_name
        )
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document to index vectors.
        
        Args:
            document: IngestionDocument with embedded chunks
            
        Returns:
            Document (unchanged, indexing is side effect)
        """
        if not document.chunks:
            document.add_error(
                self.stage_name,
                "No chunks to index"
            )
            return document
        
        # Check that chunks have embeddings
        chunks_with_embeddings = [c for c in document.chunks if c.embedding]
        
        if not chunks_with_embeddings:
            document.add_error(
                self.stage_name,
                "No chunks have embeddings"
            )
            return document
        
        # Index chunks
        indexed_count = await self.service.index_chunks(
            chunks=chunks_with_embeddings,
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            document_version=document.version
        )
        
        # Statistics
        document.metadata["vector_store_stats"] = {
            "chunks_indexed": indexed_count,
            "collection": self.service.collection_name,
            "dimension": self.service.dimension,
        }
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if vector indexing can be skipped."""
        return document.is_stage_completed(self.stage_name)


# ============================================================================
# Factory Functions
# ============================================================================

def get_vector_store_service(
    collection_name: Optional[str] = None
) -> VectorStoreService:
    """Get a vector store service instance."""
    return VectorStoreService(collection_name=collection_name)


def get_qdrant_store() -> QdrantVectorStore:
    """Get a Qdrant store instance."""
    settings = get_settings()
    return QdrantVectorStore(
        host=settings.vector_db.host,
        port=settings.vector_db.port,
        api_key=settings.vector_db.api_key
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "VectorPoint",
    "SearchResult",
    "SearchFilter",
    "VectorStoreBase",
    "QdrantVectorStore",
    "VectorStoreService",
    "VectorStorePipeline",
    "get_vector_store_service",
    "get_qdrant_store",
]
