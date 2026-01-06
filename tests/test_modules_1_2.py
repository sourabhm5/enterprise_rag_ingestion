"""
Test suite for Module 1 (Configuration) and Module 2 (Database Schema).

Run with: pytest tests/test_modules_1_2.py -v
"""

import pytest
import uuid
from datetime import datetime


class TestModule1Configuration:
    """Tests for Module 1 - Configuration & Feature Flags."""
    
    def test_settings_instantiation(self):
        """Test that settings can be instantiated with defaults."""
        from config.settings import Settings
        
        settings = Settings()
        assert settings is not None
        assert settings.environment.value == "dev"
        assert settings.app_name == "Enterprise RAG Ingestion Pipeline"
    
    def test_environment_enum(self):
        """Test environment enumeration."""
        from config.settings import Environment
        
        assert Environment.DEV.value == "dev"
        assert Environment.STAGE.value == "stage"
        assert Environment.PROD.value == "prod"
    
    def test_document_status_enum(self):
        """Test document status enumeration."""
        from config.settings import DocumentStatus
        
        assert DocumentStatus.ACTIVE.value == "ACTIVE"
        assert DocumentStatus.SOFT_DELETED.value == "SOFT_DELETED"
    
    def test_job_status_enum(self):
        """Test job status enumeration."""
        from config.settings import JobStatus
        
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.RUNNING.value == "RUNNING"
        assert JobStatus.COMPLETED.value == "COMPLETED"
        assert JobStatus.FAILED.value == "FAILED"
    
    def test_postgres_config(self):
        """Test PostgreSQL configuration."""
        from config.settings import PostgresConfig
        
        config = PostgresConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "rag_pipeline"
        assert "postgresql" in config.connection_url
    
    def test_redis_config(self):
        """Test Redis configuration."""
        from config.settings import RedisConfig
        
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert "redis://" in config.connection_url
    
    def test_s3_config_versioned_path(self):
        """Test S3 versioned path generation."""
        from config.settings import S3Config
        
        config = S3Config()
        path = config.get_versioned_path(
            tenant_id="tenant-123",
            document_id="doc-456",
            version=1,
            path_type="raw",
            filename="document.pdf"
        )
        assert path == "tenant-123/doc-456/v1/raw/document.pdf"
    
    def test_s3_config_invalid_path_type(self):
        """Test S3 path validation."""
        from config.settings import S3Config
        
        config = S3Config()
        with pytest.raises(ValueError, match="path_type must be"):
            config.get_versioned_path(
                tenant_id="tenant-123",
                document_id="doc-456",
                version=1,
                path_type="invalid",
                filename="document.pdf"
            )
    
    def test_vector_db_config_provider_validation(self):
        """Test vector DB provider validation."""
        from config.settings import VectorDBConfig
        from pydantic import ValidationError
        
        # Valid provider
        config = VectorDBConfig(provider="qdrant")
        assert config.provider == "qdrant"
        
        # Invalid provider
        with pytest.raises(ValidationError):
            VectorDBConfig(provider="invalid_provider")
    
    def test_feature_flags_defaults(self):
        """Test feature flags default values."""
        from config.settings import FeatureFlags
        
        flags = FeatureFlags()
        assert flags.enable_layout_parsing is True
        assert flags.enable_image_pipeline is True
        assert flags.enable_llm_metadata_enrichment is True
    
    def test_versioning_policy(self):
        """Test versioning policy configuration."""
        from config.settings import VersioningPolicy
        
        policy = VersioningPolicy()
        assert policy.soft_delete_old_versions is True
        assert policy.max_versions_to_retain == 5
        assert policy.retention_days == 90
    
    def test_embedding_model_config(self):
        """Test embedding model configuration."""
        from config.settings import EmbeddingModelConfig
        
        config = EmbeddingModelConfig(
            model_name="text-embedding-3-small",
            provider="openai",
            dimensions=1536
        )
        assert config.dimensions == 1536
        assert config.max_tokens == 8192
    
    def test_get_embedding_model(self):
        """Test getting embedding model from settings."""
        from config.settings import Settings
        
        settings = Settings()
        model = settings.get_embedding_model("text-embedding-3-small")
        assert model.dimensions == 1536
        assert model.provider == "openai"
    
    def test_get_embedding_model_invalid(self):
        """Test getting invalid embedding model."""
        from config.settings import Settings
        
        settings = Settings()
        with pytest.raises(ValueError, match="not found in registry"):
            settings.get_embedding_model("invalid-model")
    
    def test_validate_embedding_dimensions(self):
        """Test embedding dimension validation."""
        from config.settings import Settings
        
        settings = Settings()
        assert settings.validate_embedding_dimensions("text-embedding-3-small", 1536) is True
        assert settings.validate_embedding_dimensions("text-embedding-3-small", 512) is False
    
    def test_settings_is_production(self):
        """Test production environment check."""
        from config.settings import Settings, Environment
        
        dev_settings = Settings(environment=Environment.DEV)
        assert dev_settings.is_production is False
        assert dev_settings.is_development is True
        
        prod_settings = Settings(environment=Environment.PROD)
        assert prod_settings.is_production is True
        assert prod_settings.is_development is False
    
    def test_get_settings_singleton(self):
        """Test settings singleton behavior."""
        from config.settings import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2


class TestModule2DatabaseSchema:
    """Tests for Module 2 - Database Schema."""
    
    def test_document_model_structure(self):
        """Test Document model has required fields."""
        from storage.postgres import Document, DocumentStatus
        
        # Check required columns exist
        columns = Document.__table__.columns.keys()
        required = [
            "id", "document_id", "tenant_id", "document_version",
            "status", "allowed_roles", "allowed_users", "classification",
            "filename", "mime_type", "file_size_bytes", "content_hash",
            "s3_raw_path", "created_at", "updated_at"
        ]
        for col in required:
            assert col in columns, f"Missing column: {col}"
    
    def test_document_soft_delete(self):
        """Test Document soft delete functionality."""
        from storage.postgres import Document
        from config.settings import DocumentStatus
        
        doc = Document(
            document_id="test-doc",
            tenant_id="test-tenant",
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            content_hash="abc123",
            s3_raw_path="test/path"
        )
        
        assert doc.status == DocumentStatus.ACTIVE.value
        assert doc.deleted_at is None
        
        doc.soft_delete()
        
        assert doc.status == DocumentStatus.SOFT_DELETED.value
        assert doc.deleted_at is not None
    
    def test_document_rbac_check(self):
        """Test Document RBAC access check."""
        from storage.postgres import Document
        
        doc = Document(
            document_id="test-doc",
            tenant_id="test-tenant",
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            content_hash="abc123",
            s3_raw_path="test/path",
            allowed_users={"users": ["user-1", "user-2"]},
            allowed_roles={"roles": ["admin", "manager"]}
        )
        
        # User-level access
        assert doc.is_accessible_by_user("user-1", []) is True
        assert doc.is_accessible_by_user("user-3", []) is False
        
        # Role-level access
        assert doc.is_accessible_by_user("user-3", ["admin"]) is True
        assert doc.is_accessible_by_user("user-3", ["viewer"]) is False
    
    def test_content_node_model_structure(self):
        """Test ContentNode model has required fields."""
        from storage.postgres import ContentNode
        
        columns = ContentNode.__table__.columns.keys()
        required = [
            "id", "document_db_id", "node_id", "node_type",
            "sequence_number", "text_content", "bbox_x0", "bbox_y0",
            "bbox_x1", "bbox_y1", "linked_node_ids"
        ]
        for col in required:
            assert col in columns, f"Missing column: {col}"
    
    def test_content_node_type_enum(self):
        """Test ContentNodeType enumeration."""
        from storage.postgres import ContentNodeType
        
        assert ContentNodeType.TEXT.value == "TEXT"
        assert ContentNodeType.IMAGE.value == "IMAGE"
        assert ContentNodeType.TABLE.value == "TABLE"
        assert ContentNodeType.HEADING.value == "HEADING"
    
    def test_chunk_model_structure(self):
        """Test Chunk model has required fields."""
        from storage.postgres import Chunk
        
        columns = Chunk.__table__.columns.keys()
        required = [
            "id", "document_db_id", "chunk_id", "chunk_index",
            "text_content", "token_count", "source_node_ids",
            "image_refs", "acl_metadata", "content_hash"
        ]
        for col in required:
            assert col in columns, f"Missing column: {col}"
    
    def test_embedding_model_structure(self):
        """Test Embedding model has required fields."""
        from storage.postgres import Embedding
        
        columns = Embedding.__table__.columns.keys()
        required = [
            "id", "chunk_db_id", "embedding_id", "model_name",
            "dimensions", "content_hash", "vector_db_collection",
            "vector_db_point_id"
        ]
        for col in required:
            assert col in columns, f"Missing column: {col}"
    
    def test_ingestion_job_model_structure(self):
        """Test IngestionJob model has required fields."""
        from storage.postgres import IngestionJob
        
        columns = IngestionJob.__table__.columns.keys()
        required = [
            "id", "document_db_id", "job_id", "status",
            "pipeline_plan", "current_stage", "completed_stages",
            "error_log", "last_heartbeat"
        ]
        for col in required:
            assert col in columns, f"Missing column: {col}"
    
    def test_ingestion_job_heartbeat(self):
        """Test IngestionJob heartbeat functionality."""
        from storage.postgres import IngestionJob
        
        job = IngestionJob(
            job_id="test-job",
            pipeline_plan={"stages": ["stage1", "stage2"]}
        )
        
        assert job.last_heartbeat is None
        job.update_heartbeat()
        assert job.last_heartbeat is not None
    
    def test_ingestion_job_stage_completion(self):
        """Test IngestionJob stage completion tracking."""
        from storage.postgres import IngestionJob
        
        job = IngestionJob(
            job_id="test-job",
            pipeline_plan={"stages": ["stage1", "stage2"]},
            completed_stages=[]
        )
        
        job.mark_stage_complete("stage1")
        assert "stage1" in job.completed_stages
        
        # Adding same stage again should not duplicate
        job.mark_stage_complete("stage1")
        assert job.completed_stages.count("stage1") == 1
    
    def test_ingestion_job_error_logging(self):
        """Test IngestionJob error logging."""
        from storage.postgres import IngestionJob
        
        job = IngestionJob(
            job_id="test-job",
            pipeline_plan={"stages": ["stage1"]},
            error_log=[]
        )
        
        job.add_error("stage1", "Test error", {"key": "value"})
        
        assert len(job.error_log) == 1
        assert job.error_log[0]["stage"] == "stage1"
        assert job.error_log[0]["error"] == "Test error"
        assert "timestamp" in job.error_log[0]
    
    def test_ingestion_job_zombie_detection(self):
        """Test IngestionJob zombie detection."""
        from storage.postgres import IngestionJob
        from config.settings import JobStatus
        from datetime import timedelta
        
        job = IngestionJob(
            job_id="test-job",
            pipeline_plan={"stages": ["stage1"]},
            status=JobStatus.RUNNING.value
        )
        
        # No heartbeat = zombie
        assert job.is_zombie() is True
        
        # Fresh heartbeat = not zombie
        job.update_heartbeat()
        assert job.is_zombie() is False
        
        # Old heartbeat = zombie
        job.last_heartbeat = datetime.utcnow() - timedelta(seconds=400)
        assert job.is_zombie(threshold_seconds=300) is True
    
    def test_classification_level_enum(self):
        """Test ClassificationLevel enumeration."""
        from storage.postgres import ClassificationLevel
        
        assert ClassificationLevel.PUBLIC.value == "PUBLIC"
        assert ClassificationLevel.INTERNAL.value == "INTERNAL"
        assert ClassificationLevel.CONFIDENTIAL.value == "CONFIDENTIAL"
        assert ClassificationLevel.RESTRICTED.value == "RESTRICTED"
    
    def test_database_manager_instantiation(self):
        """Test DatabaseManager can be instantiated."""
        from storage.postgres import DatabaseManager
        
        # Test with explicit connection URL
        manager = DatabaseManager(
            connection_url="postgresql+asyncpg://test:test@localhost:5432/test"
        )
        assert manager is not None
        assert manager.engine is not None


class TestModule1And2Integration:
    """Integration tests for Modules 1 and 2."""
    
    def test_settings_postgres_url_format(self):
        """Test that settings produce valid PostgreSQL URLs."""
        from config.settings import Settings
        
        settings = Settings()
        url = settings.postgres.connection_url
        
        assert url.startswith("postgresql+asyncpg://")
        assert settings.postgres.host in url
        assert str(settings.postgres.port) in url
    
    def test_document_status_consistency(self):
        """Test DocumentStatus is consistent between config and storage."""
        from config.settings import DocumentStatus as ConfigStatus
        from storage.postgres import Document
        
        doc = Document(
            document_id="test",
            tenant_id="test",
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=100,
            content_hash="abc",
            s3_raw_path="test"
        )
        
        assert doc.status == ConfigStatus.ACTIVE.value
        doc.soft_delete()
        assert doc.status == ConfigStatus.SOFT_DELETED.value
    
    def test_job_status_consistency(self):
        """Test JobStatus is consistent between config and storage."""
        from config.settings import JobStatus as ConfigStatus
        from storage.postgres import IngestionJob
        
        job = IngestionJob(
            job_id="test",
            pipeline_plan={}
        )
        
        assert job.status == ConfigStatus.PENDING.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
