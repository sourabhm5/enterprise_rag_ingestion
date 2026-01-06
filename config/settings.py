"""
MODULE 1 — CONFIGURATION & FEATURE FLAGS
=========================================
Centralized, deterministic runtime behavior for the Enterprise RAG Ingestion Pipeline.

This module provides:
- Environment-based configuration (dev/stage/prod)
- Database connections (PostgreSQL, Redis, S3)
- Vector DB configuration
- Embedding model registry with dimension validation
- Feature flags for conditional pipeline behavior
- Versioning policy for document lifecycle
"""

from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment enumeration."""
    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""
    ACTIVE = "ACTIVE"
    SOFT_DELETED = "SOFT_DELETED"


class JobStatus(str, Enum):
    """Ingestion job status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RESUMING = "RESUMING"


class EmbeddingModelConfig(BaseSettings):
    """Configuration for an embedding model."""
    model_name: str = Field(..., description="Name/identifier of the embedding model")
    provider: str = Field(..., description="Provider (openai, cohere, huggingface, etc.)")
    dimensions: int = Field(..., gt=0, description="Expected embedding dimensions")
    max_tokens: int = Field(default=8192, gt=0, description="Maximum input tokens")
    batch_size: int = Field(default=32, gt=0, description="Batch size for embedding requests")
    endpoint_url: Optional[str] = Field(default=None, description="Custom endpoint URL if applicable")
    api_key_env_var: str = Field(default="EMBEDDING_API_KEY", description="Environment variable for API key")
    
    model_config = SettingsConfigDict(extra="forbid")


class PostgresConfig(BaseSettings):
    """PostgreSQL database configuration."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="rag_pipeline", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    echo: bool = Field(default=False, description="Echo SQL statements (debug)")
    
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        extra="forbid"
    )
    
    @property
    def connection_url(self) -> str:
        """Generate SQLAlchemy connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def sync_connection_url(self) -> str:
        """Generate synchronous SQLAlchemy connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis configuration for caching and job queues."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL connection")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout in seconds")
    embedding_cache_ttl: int = Field(default=86400, ge=0, description="Embedding cache TTL in seconds")
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="forbid"
    )
    
    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class S3Config(BaseSettings):
    """S3/MinIO storage configuration."""
    endpoint_url: Optional[str] = Field(default=None, description="S3 endpoint URL (for MinIO/localstack)")
    bucket_name: str = Field(default="rag-documents", description="Primary bucket name")
    region: str = Field(default="us-east-1", description="AWS region")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    use_ssl: bool = Field(default=True, description="Use SSL for S3 connections")
    max_pool_connections: int = Field(default=50, ge=1, description="Max pool connections")
    
    model_config = SettingsConfigDict(
        env_prefix="S3_",
        extra="forbid"
    )
    
    def get_versioned_path(self, tenant_id: str, document_id: str, version: int, path_type: str, filename: str) -> str:
        """
        Generate version-aware S3 path.
        
        Format: {tenant}/{document_id}/v{version}/raw|derived/{filename}
        """
        if path_type not in ("raw", "derived"):
            raise ValueError(f"path_type must be 'raw' or 'derived', got '{path_type}'")
        return f"{tenant_id}/{document_id}/v{version}/{path_type}/{filename}"


class VectorDBConfig(BaseSettings):
    """Vector database configuration (supports multiple backends)."""
    provider: str = Field(default="qdrant", description="Vector DB provider (qdrant, pinecone, weaviate, milvus)")
    host: str = Field(default="localhost", description="Vector DB host")
    port: int = Field(default=6333, ge=1, le=65535, description="Vector DB port")
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    collection_name: str = Field(default="rag_embeddings", description="Default collection/index name")
    grpc_port: Optional[int] = Field(default=6334, description="gRPC port (for Qdrant)")
    prefer_grpc: bool = Field(default=True, description="Prefer gRPC over HTTP")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    
    # Indexing configuration
    distance_metric: str = Field(default="cosine", description="Distance metric (cosine, euclidean, dot)")
    hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW M parameter")
    hnsw_ef_construct: int = Field(default=100, ge=10, le=500, description="HNSW ef_construct parameter")
    
    model_config = SettingsConfigDict(
        env_prefix="VECTOR_DB_",
        extra="forbid"
    )
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"qdrant", "pinecone", "weaviate", "milvus", "pgvector"}
        if v.lower() not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v.lower()
    
    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        allowed = {"cosine", "euclidean", "dot"}
        if v.lower() not in allowed:
            raise ValueError(f"distance_metric must be one of {allowed}")
        return v.lower()


class FeatureFlags(BaseSettings):
    """Feature flags for conditional pipeline behavior."""
    enable_layout_parsing: bool = Field(
        default=True, 
        description="Enable PDF layout parsing (Docling/Unstructured)"
    )
    enable_image_pipeline: bool = Field(
        default=True, 
        description="Enable image extraction and processing"
    )
    enable_llm_metadata_enrichment: bool = Field(
        default=True, 
        description="Enable LLM-based metadata extraction"
    )
    enable_ocr: bool = Field(
        default=True, 
        description="Enable OCR for images"
    )
    enable_vision_captioning: bool = Field(
        default=True, 
        description="Enable vision model captioning for images"
    )
    enable_embedding_cache: bool = Field(
        default=True, 
        description="Enable Redis-based embedding caching"
    )
    enable_soft_delete: bool = Field(
        default=True, 
        description="Enable soft delete instead of hard delete"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_",
        extra="forbid"
    )


class VersioningPolicy(BaseSettings):
    """Document versioning policy configuration."""
    soft_delete_old_versions: bool = Field(
        default=True, 
        description="Soft delete old versions on re-ingestion"
    )
    max_versions_to_retain: int = Field(
        default=5, 
        ge=1, 
        description="Maximum versions to retain per document"
    )
    retention_days: int = Field(
        default=90, 
        ge=1, 
        description="Days to retain soft-deleted versions"
    )
    auto_cleanup_enabled: bool = Field(
        default=True, 
        description="Enable automatic cleanup of old versions"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="VERSIONING_",
        extra="forbid"
    )


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""
    default_chunk_size: int = Field(default=512, ge=100, le=4096, description="Default chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks in tokens")
    min_chunk_size: int = Field(default=100, ge=10, description="Minimum chunk size")
    respect_sentence_boundaries: bool = Field(default=True, description="Respect sentence boundaries when chunking")
    include_metadata_in_chunk: bool = Field(default=True, description="Include metadata in chunk text")
    
    model_config = SettingsConfigDict(
        env_prefix="CHUNKING_",
        extra="forbid"
    )


class CeleryConfig(BaseSettings):
    """Celery job queue configuration."""
    broker_url: str = Field(default="redis://localhost:6379/1", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379/2", description="Celery result backend")
    task_serializer: str = Field(default="json", description="Task serializer")
    result_serializer: str = Field(default="json", description="Result serializer")
    accept_content: List[str] = Field(default=["json"], description="Accepted content types")
    task_acks_late: bool = Field(default=True, description="Acknowledge tasks after completion")
    worker_prefetch_multiplier: int = Field(default=1, ge=1, description="Worker prefetch multiplier")
    task_time_limit: int = Field(default=3600, ge=60, description="Task time limit in seconds")
    task_soft_time_limit: int = Field(default=3300, ge=60, description="Soft time limit in seconds")
    
    # Job health monitoring
    heartbeat_interval: int = Field(default=30, ge=10, description="Heartbeat interval in seconds")
    zombie_threshold: int = Field(default=300, ge=60, description="Zombie job threshold in seconds")
    
    model_config = SettingsConfigDict(
        env_prefix="CELERY_",
        extra="forbid"
    )


class Settings(BaseSettings):
    """
    Main application settings.
    
    Aggregates all configuration modules into a single, validated settings object.
    """
    # Environment
    environment: Environment = Field(default=Environment.DEV, description="Deployment environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Application metadata
    app_name: str = Field(default="Enterprise RAG Ingestion Pipeline", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    
    # Sub-configurations
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    s3: S3Config = Field(default_factory=S3Config)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    celery: CeleryConfig = Field(default_factory=CeleryConfig)
    
    # Feature flags and policies
    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    versioning: VersioningPolicy = Field(default_factory=VersioningPolicy)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    
    # Embedding model registry
    embedding_models: Dict[str, EmbeddingModelConfig] = Field(
        default_factory=lambda: {
            "text-embedding-3-small": EmbeddingModelConfig(
                model_name="text-embedding-3-small",
                provider="openai",
                dimensions=1536,
                max_tokens=8191
            ),
            "text-embedding-3-large": EmbeddingModelConfig(
                model_name="text-embedding-3-large",
                provider="openai",
                dimensions=3072,
                max_tokens=8191
            ),
        },
        description="Registry of available embedding models"
    )
    
    # Default embedding model
    default_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model to use"
    )
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, description="Number of API workers")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1, description="Rate limit requests per minute")
    max_upload_size_mb: int = Field(default=100, ge=1, description="Maximum upload size in MB")
    
    # Supported file types (V1 scope: Text + Image + PDF only)
    supported_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/html",
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        ],
        description="Supported MIME types for ingestion"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False
    )
    
    @model_validator(mode="after")
    def validate_default_embedding_model(self) -> "Settings":
        """Ensure default embedding model exists in registry."""
        if self.default_embedding_model not in self.embedding_models:
            raise ValueError(
                f"default_embedding_model '{self.default_embedding_model}' "
                f"not found in embedding_models registry"
            )
        return self
    
    def get_embedding_model(self, model_name: Optional[str] = None) -> EmbeddingModelConfig:
        """Get embedding model configuration by name or return default."""
        name = model_name or self.default_embedding_model
        if name not in self.embedding_models:
            raise ValueError(f"Embedding model '{name}' not found in registry")
        return self.embedding_models[name]
    
    def validate_embedding_dimensions(self, model_name: str, actual_dimensions: int) -> bool:
        """Validate that embedding dimensions match expected dimensions."""
        model = self.get_embedding_model(model_name)
        return model.dimensions == actual_dimensions
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PROD
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEV


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure singleton behavior.
    """
    return Settings()


# Convenience function for dependency injection
def get_settings_dependency() -> Settings:
    """FastAPI dependency for settings injection."""
    return get_settings()


# Export commonly used enums
__all__ = [
    "Settings",
    "get_settings",
    "get_settings_dependency",
    "Environment",
    "DocumentStatus",
    "JobStatus",
    "PostgresConfig",
    "RedisConfig",
    "S3Config",
    "VectorDBConfig",
    "FeatureFlags",
    "VersioningPolicy",
    "ChunkingConfig",
    "CeleryConfig",
    "EmbeddingModelConfig",
]
