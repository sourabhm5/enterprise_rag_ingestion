"""
Test suite for Module 5 (Job Manager) and Module 6 (Ingestion Router).

Run with: pytest tests/test_modules_5_6.py -v
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestModule5JobManager:
    """Tests for Module 5 - Job Manager."""
    
    def test_celery_app_creation(self):
        """Test Celery app can be created."""
        from jobs.celery_app import create_celery_app, celery_app
        
        app = create_celery_app()
        assert app is not None
        assert app.main == "enterprise_rag_ingestion"
        
        # Check global instance
        assert celery_app is not None
    
    def test_celery_app_configuration(self):
        """Test Celery app has correct configuration."""
        from jobs.celery_app import celery_app
        
        # Check task serialization
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"
        
        # Check task tracking
        assert celery_app.conf.task_track_started is True
        assert celery_app.conf.task_acks_late is True
    
    def test_celery_beat_schedule(self):
        """Test Celery beat schedule is configured."""
        from jobs.celery_app import celery_app
        
        schedule = celery_app.conf.beat_schedule
        assert "cleanup-zombie-jobs" in schedule
        assert schedule["cleanup-zombie-jobs"]["task"] == "jobs.job_manager.cleanup_zombie_jobs"
    
    def test_heartbeat_monitor_instantiation(self):
        """Test HeartbeatMonitor can be instantiated."""
        from jobs.heartbeat_monitor import HeartbeatMonitor
        
        async def callback(job_id):
            pass
        
        monitor = HeartbeatMonitor(
            job_id="test-job",
            heartbeat_callback=callback,
            interval=30
        )
        
        assert monitor.job_id == "test-job"
        assert monitor.interval == 30
    
    def test_sync_heartbeat_monitor_instantiation(self):
        """Test SyncHeartbeatMonitor can be instantiated."""
        from jobs.heartbeat_monitor import SyncHeartbeatMonitor
        
        def callback(job_id):
            pass
        
        monitor = SyncHeartbeatMonitor(
            job_id="test-job",
            heartbeat_callback=callback,
            interval=30
        )
        
        assert monitor.job_id == "test-job"
        assert monitor.interval == 30
    
    @pytest.mark.asyncio
    async def test_job_manager_create_job(self):
        """Test JobManager can create a job."""
        from jobs.job_manager import JobManager
        from config.settings import JobStatus
        
        # Mock database session
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        
        job_manager = JobManager(mock_session)
        
        job = await job_manager.create_job(
            document_db_id=uuid.uuid4(),
            pipeline_plan={"stages": ["routing", "text_pipeline"]},
            max_retries=3
        )
        
        assert job is not None
        assert job.status == JobStatus.PENDING.value
        assert "routing" in job.pipeline_plan["stages"]
        assert job.max_retries == 3
        mock_session.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_manager_get_next_stage(self):
        """Test JobManager can determine next stage."""
        from jobs.job_manager import JobManager
        from storage.postgres import IngestionJob
        
        mock_session = AsyncMock()
        
        # Create mock job
        mock_job = MagicMock(spec=IngestionJob)
        mock_job.pipeline_plan = {"stages": ["routing", "text_pipeline", "chunking"]}
        mock_job.completed_stages = ["routing"]
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        job_manager = JobManager(mock_session)
        
        next_stage = await job_manager.get_next_stage("test-job")
        assert next_stage == "text_pipeline"
    
    @pytest.mark.asyncio
    async def test_job_manager_get_remaining_stages(self):
        """Test JobManager can get remaining stages."""
        from jobs.job_manager import JobManager
        from storage.postgres import IngestionJob
        
        mock_session = AsyncMock()
        
        mock_job = MagicMock(spec=IngestionJob)
        mock_job.pipeline_plan = {"stages": ["routing", "text_pipeline", "chunking", "embedding"]}
        mock_job.completed_stages = ["routing", "text_pipeline"]
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        job_manager = JobManager(mock_session)
        
        remaining = await job_manager.get_remaining_stages("test-job")
        assert remaining == ["chunking", "embedding"]


class TestModule6IngestionRouter:
    """Tests for Module 6 - Ingestion Router."""
    
    def test_content_type_enum(self):
        """Test ContentType enumeration."""
        from routing.ingestion_router import ContentType
        
        assert ContentType.PDF.value == "pdf"
        assert ContentType.IMAGE.value == "image"
        assert ContentType.TEXT.value == "text"
        assert ContentType.MARKDOWN.value == "markdown"
        assert ContentType.HTML.value == "html"
    
    def test_image_format_enum(self):
        """Test ImageFormat enumeration."""
        from routing.ingestion_router import ImageFormat
        
        assert ImageFormat.PNG.value == "png"
        assert ImageFormat.JPEG.value == "jpeg"
        assert ImageFormat.GIF.value == "gif"
        assert ImageFormat.WEBP.value == "webp"
    
    def test_content_prober_instantiation(self):
        """Test ContentProber can be instantiated."""
        from routing.ingestion_router import ContentProber
        
        prober = ContentProber()
        assert prober is not None
    
    def test_content_prober_detect_pdf(self):
        """Test PDF detection."""
        from routing.ingestion_router import ContentProber, ContentType
        
        prober = ContentProber()
        
        # PDF magic bytes
        pdf_content = b"%PDF-1.4 test content"
        result = prober.probe(pdf_content, "test.pdf")
        
        assert result.content_type == ContentType.PDF
        assert result.is_pdf is True
        assert result.mime_type == "application/pdf"
    
    def test_content_prober_detect_png(self):
        """Test PNG detection."""
        from routing.ingestion_router import ContentProber, ContentType, ImageFormat
        
        prober = ContentProber()
        
        # PNG magic bytes with minimal header
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = prober.probe(png_content, "test.png")
        
        assert result.content_type == ContentType.IMAGE
        assert result.is_image is True
        assert result.image_format == ImageFormat.PNG
    
    def test_content_prober_detect_jpeg(self):
        """Test JPEG detection."""
        from routing.ingestion_router import ContentProber, ContentType, ImageFormat
        
        prober = ContentProber()
        
        # JPEG magic bytes
        jpeg_content = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        result = prober.probe(jpeg_content, "test.jpg")
        
        assert result.content_type == ContentType.IMAGE
        assert result.is_image is True
        assert result.image_format == ImageFormat.JPEG
    
    def test_content_prober_detect_text(self):
        """Test plain text detection."""
        from routing.ingestion_router import ContentProber, ContentType
        
        prober = ContentProber()
        
        text_content = b"This is plain text content without any special formatting."
        result = prober.probe(text_content, "test.txt")
        
        assert result.content_type == ContentType.TEXT
        assert result.is_text is True
        assert result.text_encoding == "utf-8"
    
    def test_content_prober_detect_markdown(self):
        """Test Markdown detection."""
        from routing.ingestion_router import ContentProber, ContentType
        
        prober = ContentProber()
        
        markdown_content = b"""# Heading
        
## Subheading

- List item 1
- List item 2

**Bold text** and `code`
"""
        result = prober.probe(markdown_content, "test.md")
        
        assert result.content_type == ContentType.MARKDOWN
        assert result.is_markdown is True
    
    def test_content_prober_detect_html(self):
        """Test HTML detection."""
        from routing.ingestion_router import ContentProber, ContentType
        
        prober = ContentProber()
        
        html_content = b"""<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><p>Content</p></body>
</html>"""
        result = prober.probe(html_content, "test.html")
        
        assert result.content_type == ContentType.HTML
        assert result.is_html is True
    
    def test_ingestion_router_instantiation(self):
        """Test IngestionRouter can be instantiated."""
        from routing.ingestion_router import IngestionRouter
        
        router = IngestionRouter()
        assert router is not None
        assert router.feature_flags is not None
    
    def test_ingestion_router_route_pdf(self):
        """Test routing for PDF document."""
        from routing.ingestion_router import IngestionRouter, ContentType
        
        router = IngestionRouter()
        
        pdf_content = b"%PDF-1.4 test content with /XObject /Image"
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=pdf_content,
            filename="test.pdf"
        )
        
        assert route.content_type == ContentType.PDF
        assert "layout_parsing" in route.pipeline_stages
        assert "text_pipeline" in route.pipeline_stages
        assert "chunking" in route.pipeline_stages
        assert "embedding" in route.pipeline_stages
        assert "vector_store" in route.pipeline_stages
    
    def test_ingestion_router_route_image(self):
        """Test routing for image document."""
        from routing.ingestion_router import IngestionRouter, ContentType
        
        router = IngestionRouter()
        
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=png_content,
            filename="test.png"
        )
        
        assert route.content_type == ContentType.IMAGE
        assert "image_pipeline" in route.pipeline_stages
        assert "layout_parsing" not in route.pipeline_stages  # Not a PDF
    
    def test_ingestion_router_route_text(self):
        """Test routing for text document."""
        from routing.ingestion_router import IngestionRouter, ContentType
        
        router = IngestionRouter()
        
        text_content = b"This is a plain text document."
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=text_content,
            filename="test.txt"
        )
        
        assert route.content_type == ContentType.TEXT
        assert "text_pipeline" in route.pipeline_stages
        assert "image_pipeline" not in route.pipeline_stages
        assert "layout_parsing" not in route.pipeline_stages
    
    def test_ingestion_router_quick_route(self):
        """Test quick routing from MIME type only."""
        from routing.ingestion_router import IngestionRouter, ContentType
        
        router = IngestionRouter()
        
        route = router.route_from_mime_type(
            document_id="doc-123",
            tenant_id="tenant-456",
            mime_type="application/pdf",
            file_size=1024,
            has_images=True
        )
        
        assert route.content_type == ContentType.PDF
        assert route.routing_metadata.get("quick_route") is True
    
    def test_pipeline_route_to_plan(self):
        """Test PipelineRoute conversion to pipeline plan."""
        from routing.ingestion_router import PipelineRoute, ContentType
        
        route = PipelineRoute(
            document_id="doc-123",
            tenant_id="tenant-456",
            mime_type="application/pdf",
            content_type=ContentType.PDF,
            pipeline_stages=["routing", "text_pipeline", "chunking"],
            features_enabled={"layout_parsing": True}
        )
        
        plan = route.to_pipeline_plan()
        
        assert plan["document_id"] == "doc-123"
        assert plan["tenant_id"] == "tenant-456"
        assert plan["stages"] == ["routing", "text_pipeline", "chunking"]
        assert plan["content_type"] == "pdf"
    
    def test_ingestion_router_stage_order(self):
        """Test that pipeline stages are in correct order."""
        from routing.ingestion_router import IngestionRouter
        
        router = IngestionRouter()
        
        # PDF with images should have all stages in order
        pdf_content = b"%PDF-1.4 /XObject /Image content"
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=pdf_content
        )
        
        stages = route.pipeline_stages
        
        # Routing should be first
        assert stages.index("routing") == 0
        
        # Embedding should be before vector_store
        assert stages.index("embedding") < stages.index("vector_store")
        
        # Chunking should be before embedding
        assert stages.index("chunking") < stages.index("embedding")
    
    def test_ingestion_router_validate_pipeline(self):
        """Test pipeline validation."""
        from routing.ingestion_router import IngestionRouter
        
        router = IngestionRouter()
        
        # Valid pipeline
        valid, error = router.validate_pipeline([
            "routing", "text_pipeline", "chunking", "embedding", "vector_store"
        ])
        assert valid is True
        assert error is None
        
        # Invalid stage
        valid, error = router.validate_pipeline(["routing", "invalid_stage"])
        assert valid is False
        assert "Unknown stage" in error
    
    def test_ingestion_router_get_dependencies(self):
        """Test getting stage dependencies."""
        from routing.ingestion_router import IngestionRouter
        
        router = IngestionRouter()
        
        deps = router.get_stage_dependencies()
        
        # Routing has no dependencies
        assert deps["routing"] == []
        
        # Vector store depends on embedding
        assert "embedding" in deps["vector_store"]
    
    def test_content_probe_result_dataclass(self):
        """Test ContentProbeResult dataclass."""
        from routing.ingestion_router import ContentProbeResult, ContentType
        
        result = ContentProbeResult(
            content_type=ContentType.PDF,
            mime_type="application/pdf",
            file_extension="pdf",
            file_size_bytes=1024,
            is_pdf=True,
            pdf_has_images=True,
            pdf_page_count=5
        )
        
        assert result.is_pdf is True
        assert result.pdf_page_count == 5
        assert result.file_size_bytes == 1024


class TestModule5And6Integration:
    """Integration tests for Modules 5 and 6."""
    
    def test_router_produces_valid_job_plan(self):
        """Test that router output can be used for job creation."""
        from routing.ingestion_router import IngestionRouter
        
        router = IngestionRouter()
        
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=b"%PDF-1.4 test",
            filename="test.pdf"
        )
        
        plan = route.to_pipeline_plan()
        
        # Plan should have required fields for job
        assert "stages" in plan
        assert "document_id" in plan
        assert "tenant_id" in plan
        assert isinstance(plan["stages"], list)
        assert len(plan["stages"]) > 0
    
    @pytest.mark.asyncio
    async def test_job_with_routed_pipeline(self):
        """Test creating a job with routed pipeline plan."""
        from jobs.job_manager import JobManager
        from routing.ingestion_router import IngestionRouter
        
        # Route document
        router = IngestionRouter()
        route = router.route(
            document_id="doc-123",
            tenant_id="tenant-456",
            content=b"Plain text document",
            filename="test.txt"
        )
        
        # Create job with routed plan
        mock_session = AsyncMock()
        mock_session.flush = AsyncMock()
        
        job_manager = JobManager(mock_session)
        job = await job_manager.create_job(
            document_db_id=uuid.uuid4(),
            pipeline_plan=route.to_pipeline_plan()
        )
        
        assert job is not None
        assert job.pipeline_plan["document_id"] == "doc-123"
        assert "text_pipeline" in job.pipeline_plan["stages"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
