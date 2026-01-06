"""
MODULE 2 — DATABASE SCHEMA (METADATA + RBAC + LIFECYCLE)
========================================================
PostgreSQL schema using SQLAlchemy for the Enterprise RAG Ingestion Pipeline.

Key Features:
- Document versioning with soft delete support
- RBAC enforcement via allowed_roles and allowed_users (JSONB)
- Ingestion job tracking with DAG pipeline plan
- Resume-safe job state management
- Full audit trail for compliance

Tables:
- documents: Core document metadata with RBAC
- content_nodes: Parsed content blocks (text, images)
- chunks: Chunked content for retrieval
- embeddings: Vector embedding metadata
- ingestion_jobs: Job tracking with stage persistence
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from config.settings import DocumentStatus, JobStatus, get_settings


# ============================================================================
# Base Classes
# ============================================================================

class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    
    type_annotation_map = {
        dict: JSONB,
        Dict[str, Any]: JSONB,
        List[str]: ARRAY(String),
    }


class TimestampMixin:
    """Mixin for automatic timestamp management."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


# ============================================================================
# Enums (Database Level)
# ============================================================================

class ContentNodeType(str, Enum):
    """Types of content nodes in the IR."""
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    TABLE = "TABLE"
    HEADING = "HEADING"


class ClassificationLevel(str, Enum):
    """Document classification levels."""
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


# ============================================================================
# Models
# ============================================================================

class Document(Base, TimestampMixin):
    """
    Core document table with RBAC and versioning support.
    
    Each document can have multiple versions. On re-ingestion:
    1. Old version is soft-deleted
    2. New version is created with incremented version number
    """
    
    __tablename__ = "documents"
    
    # Primary identification
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="External document identifier"
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Tenant/organization identifier"
    )
    
    # Versioning
    document_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Document version (incremented on re-ingestion)"
    )
    
    # Lifecycle status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=DocumentStatus.ACTIVE.value,
        index=True,
        comment="Document status (ACTIVE, SOFT_DELETED)"
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of soft deletion"
    )
    
    # RBAC - Access Control Lists
    allowed_roles: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Roles allowed to access this document"
    )
    allowed_users: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Users allowed to access this document"
    )
    
    # Classification
    classification: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=ClassificationLevel.INTERNAL.value,
        comment="Security classification level"
    )
    
    # Document metadata
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="Original filename"
    )
    mime_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="MIME type of the document"
    )
    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="File size in bytes"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of content for idempotency"
    )
    
    # S3 storage references
    s3_raw_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="S3 path to raw document"
    )
    s3_derived_prefix: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 prefix for derived assets"
    )
    
    # Enriched metadata (populated by metadata enrichment pipeline)
    enriched_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="LLM-extracted metadata (department, doc_type, fiscal_year)"
    )
    
    # Source tracking
    source_system: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Source system identifier"
    )
    source_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Original source URL if applicable"
    )
    
    # Relationships
    content_nodes: Mapped[List["ContentNode"]] = relationship(
        "ContentNode",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    ingestion_jobs: Mapped[List["IngestionJob"]] = relationship(
        "IngestionJob",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "document_id", "document_version",
            name="uq_document_version"
        ),
        Index("ix_document_tenant_status", "tenant_id", "status"),
        Index("ix_document_classification", "tenant_id", "classification"),
        Index("ix_document_content_hash", "content_hash"),
        # GIN indexes for JSONB RBAC queries
        Index(
            "ix_document_allowed_roles",
            "allowed_roles",
            postgresql_using="gin"
        ),
        Index(
            "ix_document_allowed_users",
            "allowed_users",
            postgresql_using="gin"
        ),
    )
    
    def soft_delete(self) -> None:
        """Mark document as soft-deleted."""
        self.status = DocumentStatus.SOFT_DELETED.value
        self.deleted_at = datetime.utcnow()
    
    def is_accessible_by_user(self, user_id: str, user_roles: List[str]) -> bool:
        """Check if user has access based on RBAC."""
        # Check user-level access
        if user_id in self.allowed_users.get("users", []):
            return True
        
        # Check role-level access
        allowed = set(self.allowed_roles.get("roles", []))
        if allowed.intersection(set(user_roles)):
            return True
        
        return False


class ContentNode(Base, TimestampMixin):
    """
    Parsed content blocks from documents.
    
    Represents elements extracted during layout parsing:
    - TEXT: Paragraphs, sections
    - IMAGE: Embedded images
    - TABLE: Extracted tables
    - HEADING: Section headers
    """
    
    __tablename__ = "content_nodes"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Parent document reference
    document_db_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Node identification within document
    node_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Unique node ID within the document"
    )
    node_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Node type (TEXT, IMAGE, TABLE, HEADING)"
    )
    
    # Reading order (for proper sequencing)
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Position in reading order"
    )
    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Page number (for PDFs)"
    )
    
    # Content
    text_content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Text content (for TEXT/HEADING nodes)"
    )
    
    # Bounding box (normalized 0-1)
    bbox_x0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y0: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_x1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Image-specific fields
    image_s3_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 path for image nodes"
    )
    image_caption: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Vision model generated caption"
    )
    ocr_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="OCR extracted text from image"
    )
    
    # Linkage (graph relationships)
    parent_node_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Parent node ID (for hierarchy)"
    )
    linked_node_ids: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of linked node IDs"
    )
    
    # Metadata
    metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional node metadata"
    )
    
    # Heading level (for HEADING nodes)
    heading_level: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Heading level (1-6)"
    )
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="content_nodes"
    )
    
    __table_args__ = (
        UniqueConstraint("document_db_id", "node_id", name="uq_node_in_document"),
        Index("ix_content_node_type", "document_db_id", "node_type"),
        Index("ix_content_node_sequence", "document_db_id", "sequence_number"),
        Index("ix_content_node_page", "document_db_id", "page_number"),
    )


class Chunk(Base, TimestampMixin):
    """
    Chunked content for retrieval.
    
    Each chunk contains:
    - Text content for embedding
    - References to supporting images
    - Inherited RBAC metadata
    """
    
    __tablename__ = "chunks"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Parent document reference
    document_db_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Chunk identification
    chunk_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Unique chunk ID within the document"
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Chunk sequence index"
    )
    
    # Content
    text_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Chunk text content"
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Token count for the chunk"
    )
    
    # Source tracking
    source_node_ids: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Source content node IDs"
    )
    
    # Image references (supporting evidence)
    image_refs: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="References to linked image nodes"
    )
    
    # RBAC (denormalized for query efficiency)
    acl_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="ACL metadata (denormalized from document)"
    )
    
    # Page reference
    start_page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Starting page number"
    )
    end_page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Ending page number"
    )
    
    # Content hash for deduplication
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="SHA-256 hash of chunk content"
    )
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks"
    )
    embeddings: Mapped[List["Embedding"]] = relationship(
        "Embedding",
        back_populates="chunk",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    
    __table_args__ = (
        UniqueConstraint("document_db_id", "chunk_id", name="uq_chunk_in_document"),
        Index("ix_chunk_document_index", "document_db_id", "chunk_index"),
        Index("ix_chunk_content_hash", "content_hash"),
        # GIN index for ACL queries
        Index(
            "ix_chunk_acl",
            "acl_metadata",
            postgresql_using="gin"
        ),
    )


class Embedding(Base, TimestampMixin):
    """
    Vector embedding metadata and storage reference.
    
    Actual vectors are stored in the vector database.
    This table tracks metadata and enables lifecycle management.
    """
    
    __tablename__ = "embeddings"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Parent chunk reference
    chunk_db_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Embedding identification
    embedding_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Unique embedding ID (used in vector DB)"
    )
    
    # Model information
    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Embedding model name"
    )
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0",
        comment="Embedding model version"
    )
    dimensions: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Vector dimensions"
    )
    
    # Content hash for cache lookup
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Hash of embedded content"
    )
    
    # Embedding type
    embedding_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="text",
        comment="Type: text, image_caption, combined"
    )
    
    # Vector DB reference
    vector_db_collection: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Vector DB collection name"
    )
    vector_db_point_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Point ID in vector DB"
    )
    
    # Relationships
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="embeddings"
    )
    
    __table_args__ = (
        Index("ix_embedding_model", "model_name", "model_version"),
        Index("ix_embedding_content_hash", "content_hash"),
        Index("ix_embedding_vector_db", "vector_db_collection", "vector_db_point_id"),
    )


class IngestionJob(Base, TimestampMixin):
    """
    Ingestion job tracking with DAG pipeline plan.
    
    Features:
    - Pipeline plan storage (DAG)
    - Current stage tracking for resume
    - Error logging
    - Heartbeat for zombie detection
    """
    
    __tablename__ = "ingestion_jobs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Parent document reference
    document_db_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Job identification
    job_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Unique job identifier"
    )
    
    # Job status
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=JobStatus.PENDING.value,
        index=True,
        comment="Job status"
    )
    
    # Pipeline execution plan (DAG)
    pipeline_plan: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Ordered list of pipeline stages"
    )
    
    # Stage tracking for resume
    current_stage: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Currently executing stage"
    )
    completed_stages: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of completed stages"
    )
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Job start timestamp"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Job completion timestamp"
    )
    
    # Heartbeat for zombie detection
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last heartbeat timestamp"
    )
    
    # Error tracking
    error_log: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of errors encountered"
    )
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of retry attempts"
    )
    max_retries: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        comment="Maximum retry attempts"
    )
    
    # Worker information
    worker_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Celery worker ID"
    )
    
    # Metrics
    metrics: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Job metrics (latency, costs, etc.)"
    )
    
    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="ingestion_jobs"
    )
    
    __table_args__ = (
        Index("ix_job_status_heartbeat", "status", "last_heartbeat"),
        Index("ix_job_document", "document_db_id", "status"),
    )
    
    def update_heartbeat(self) -> None:
        """Update job heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
    
    def mark_stage_complete(self, stage: str) -> None:
        """Mark a pipeline stage as complete."""
        if isinstance(self.completed_stages, list):
            if stage not in self.completed_stages:
                self.completed_stages = self.completed_stages + [stage]
        else:
            self.completed_stages = [stage]
    
    def add_error(self, stage: str, error: str, details: Optional[Dict] = None) -> None:
        """Add an error to the error log."""
        error_entry = {
            "stage": stage,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        if isinstance(self.error_log, list):
            self.error_log = self.error_log + [error_entry]
        else:
            self.error_log = [error_entry]
    
    def is_zombie(self, threshold_seconds: int = 300) -> bool:
        """Check if job is a zombie (no heartbeat within threshold)."""
        if self.status != JobStatus.RUNNING.value:
            return False
        if self.last_heartbeat is None:
            return True
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed > threshold_seconds


# ============================================================================
# Database Session Management
# ============================================================================

class DatabaseManager:
    """Async database session manager."""
    
    def __init__(self, connection_url: Optional[str] = None):
        settings = get_settings()
        self.connection_url = connection_url or settings.postgres.connection_url
        self.engine = create_async_engine(
            self.connection_url,
            pool_size=settings.postgres.pool_size,
            max_overflow=settings.postgres.max_overflow,
            echo=settings.postgres.echo,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def create_tables(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self) -> None:
        """Drop all tables (use with caution!)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def get_session(self) -> AsyncSession:
        """Get a new async session."""
        return self.async_session()
    
    async def close(self) -> None:
        """Close the database engine."""
        await self.engine.dispose()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_db_session() -> AsyncSession:
    """FastAPI dependency for database session."""
    db_manager = get_db_manager()
    async with db_manager.async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ============================================================================
# Utility Functions
# ============================================================================

async def get_active_document(
    session: AsyncSession,
    tenant_id: str,
    document_id: str
) -> Optional[Document]:
    """Get the active (non-deleted) version of a document."""
    from sqlalchemy import select
    
    stmt = (
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.document_id == document_id)
        .where(Document.status == DocumentStatus.ACTIVE.value)
        .order_by(Document.document_version.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_latest_version_number(
    session: AsyncSession,
    tenant_id: str,
    document_id: str
) -> int:
    """Get the latest version number for a document (including soft-deleted)."""
    from sqlalchemy import select, func as sql_func
    
    stmt = (
        select(sql_func.coalesce(sql_func.max(Document.document_version), 0))
        .where(Document.tenant_id == tenant_id)
        .where(Document.document_id == document_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def soft_delete_old_versions(
    session: AsyncSession,
    tenant_id: str,
    document_id: str,
    keep_version: int
) -> int:
    """Soft delete all versions except the specified one."""
    from sqlalchemy import update
    
    stmt = (
        update(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.document_id == document_id)
        .where(Document.document_version != keep_version)
        .where(Document.status == DocumentStatus.ACTIVE.value)
        .values(
            status=DocumentStatus.SOFT_DELETED.value,
            deleted_at=datetime.utcnow()
        )
    )
    result = await session.execute(stmt)
    return result.rowcount


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Models
    "Base",
    "Document",
    "ContentNode",
    "Chunk",
    "Embedding",
    "IngestionJob",
    
    # Enums
    "ContentNodeType",
    "ClassificationLevel",
    
    # Database management
    "DatabaseManager",
    "get_db_manager",
    "get_db_session",
    
    # Utility functions
    "get_active_document",
    "get_latest_version_number",
    "soft_delete_old_versions",
]
