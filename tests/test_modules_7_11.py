"""
Test suite for Modules 7-11.

Module 7: Layout Parser
Module 8: Pipeline DAG Executor
Module 9: Text Pipeline
Module 10: Image Pipeline
Module 11: Intermediate Representation

Run with: pytest tests/test_modules_7_11.py -v
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestModule11IntermediateRepresentation:
    """Tests for Module 11 - Intermediate Representation (THE CONTRACT)."""
    
    def test_node_type_enum(self):
        """Test NodeType enumeration."""
        from schema.intermediate_representation import NodeType
        
        assert NodeType.TEXT.value == "text"
        assert NodeType.IMAGE.value == "image"
        assert NodeType.HEADING.value == "heading"
        assert NodeType.TABLE.value == "table"
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enumeration."""
        from schema.intermediate_representation import ProcessingStatus
        
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
    
    def test_bounding_box_creation(self):
        """Test BoundingBox creation and validation."""
        from schema.intermediate_representation import BoundingBox
        
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.8, y1=0.9, page=0)
        
        assert bbox.x0 == 0.1
        assert bbox.y0 == 0.2
        assert bbox.x1 == 0.8
        assert bbox.y1 == 0.9
        assert bbox.page == 0
    
    def test_bounding_box_clamping(self):
        """Test BoundingBox clamps values to 0-1."""
        from schema.intermediate_representation import BoundingBox
        
        bbox = BoundingBox(x0=-0.1, y0=1.5, x1=0.8, y1=0.9)
        
        assert bbox.x0 == 0.0
        assert bbox.y0 == 1.0
    
    def test_bounding_box_swap(self):
        """Test BoundingBox swaps reversed coordinates."""
        from schema.intermediate_representation import BoundingBox
        
        bbox = BoundingBox(x0=0.8, y0=0.9, x1=0.1, y1=0.2)
        
        assert bbox.x0 == 0.1
        assert bbox.x1 == 0.8
        assert bbox.y0 == 0.2
        assert bbox.y1 == 0.9
    
    def test_bounding_box_properties(self):
        """Test BoundingBox computed properties."""
        from schema.intermediate_representation import BoundingBox
        
        bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        
        assert bbox.width == 0.5
        assert bbox.height == 0.5
        assert bbox.area == 0.25
        assert bbox.center == (0.25, 0.25)
    
    def test_bounding_box_contains(self):
        """Test BoundingBox contains method."""
        from schema.intermediate_representation import BoundingBox
        
        outer = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, page=0)
        inner = BoundingBox(x0=0.2, y0=0.2, x1=0.8, y1=0.8, page=0)
        other_page = BoundingBox(x0=0.2, y0=0.2, x1=0.8, y1=0.8, page=1)
        
        assert outer.contains(inner) is True
        assert inner.contains(outer) is False
        assert outer.contains(other_page) is False
    
    def test_bounding_box_overlaps(self):
        """Test BoundingBox overlap detection."""
        from schema.intermediate_representation import BoundingBox
        
        box1 = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        box2 = BoundingBox(x0=0.3, y0=0.3, x1=0.8, y1=0.8)
        box3 = BoundingBox(x0=0.6, y0=0.6, x1=1.0, y1=1.0)
        
        assert box1.overlaps(box2) is True
        assert box1.overlaps(box3) is False
    
    def test_bounding_box_from_absolute(self):
        """Test BoundingBox creation from absolute coordinates."""
        from schema.intermediate_representation import BoundingBox
        
        bbox = BoundingBox.from_absolute(
            x0=100, y0=200, x1=400, y1=600,
            page_width=800, page_height=1000,
            page=0
        )
        
        assert bbox.x0 == 0.125
        assert bbox.y0 == 0.2
        assert bbox.x1 == 0.5
        assert bbox.y1 == 0.6
    
    def test_content_node_creation(self):
        """Test ContentNode creation."""
        from schema.intermediate_representation import ContentNode, NodeType
        
        node = ContentNode(
            node_type=NodeType.TEXT,
            sequence=1,
            text="Hello, World!"
        )
        
        assert node.node_type == NodeType.TEXT
        assert node.text == "Hello, World!"
        assert node.id.startswith("node-")
    
    def test_content_node_links(self):
        """Test ContentNode link management."""
        from schema.intermediate_representation import ContentNode, NodeType
        
        node = ContentNode(node_type=NodeType.TEXT)
        
        node.add_link("node-123")
        assert "node-123" in node.linked_ids
        
        node.add_link("node-123")  # Duplicate
        assert node.linked_ids.count("node-123") == 1
        
        node.remove_link("node-123")
        assert "node-123" not in node.linked_ids
    
    def test_content_node_full_text(self):
        """Test ContentNode get_full_text method."""
        from schema.intermediate_representation import ContentNode, NodeType
        
        node = ContentNode(
            node_type=NodeType.IMAGE,
            text="",
            ocr_text="OCR text",
            image_caption="Image description"
        )
        
        full_text = node.get_full_text()
        assert "OCR text" in full_text
        assert "Image description" in full_text
    
    def test_chunk_creation(self):
        """Test Chunk creation."""
        from schema.intermediate_representation import Chunk
        
        chunk = Chunk(
            text="This is a chunk of text.",
            token_count=5,
            source_node_ids=["node-1", "node-2"]
        )
        
        assert chunk.text == "This is a chunk of text."
        assert chunk.token_count == 5
        assert len(chunk.source_node_ids) == 2
    
    def test_rbac_context(self):
        """Test RBACContext creation."""
        from schema.intermediate_representation import RBACContext
        
        rbac = RBACContext(
            tenant_id="tenant-123",
            allowed_roles=["admin", "editor"],
            allowed_users=["user-1"],
            classification="CONFIDENTIAL"
        )
        
        assert rbac.tenant_id == "tenant-123"
        assert "admin" in rbac.allowed_roles
        assert rbac.classification == "CONFIDENTIAL"
    
    def test_ingestion_document_creation(self):
        """Test IngestionDocument creation."""
        from schema.intermediate_representation import IngestionDocument
        
        doc = IngestionDocument(
            document_id="doc-123",
            tenant_id="tenant-456",
            filename="test.pdf",
            mime_type="application/pdf"
        )
        
        assert doc.document_id == "doc-123"
        assert doc.tenant_id == "tenant-456"
        assert doc.rbac.tenant_id == "tenant-456"
    
    def test_ingestion_document_add_nodes(self):
        """Test adding nodes to IngestionDocument."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        doc = IngestionDocument(
            document_id="doc-123",
            tenant_id="tenant-456"
        )
        
        node1 = ContentNode(node_type=NodeType.TEXT, text="Text 1")
        node2 = ContentNode(node_type=NodeType.IMAGE)
        
        doc.add_node(node1)
        doc.add_node(node2)
        
        assert len(doc.nodes) == 2
        assert doc.get_node(node1.id) == node1
    
    def test_ingestion_document_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        
        doc.add_node(ContentNode(node_type=NodeType.TEXT, text="Text"))
        doc.add_node(ContentNode(node_type=NodeType.IMAGE))
        doc.add_node(ContentNode(node_type=NodeType.TEXT, text="More text"))
        
        text_nodes = doc.get_nodes_by_type(NodeType.TEXT)
        assert len(text_nodes) == 2
        
        image_nodes = doc.get_image_nodes()
        assert len(image_nodes) == 1
    
    def test_ingestion_document_stage_management(self):
        """Test stage management in IngestionDocument."""
        from schema.intermediate_representation import IngestionDocument
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        
        doc.start_stage("text_pipeline")
        assert doc.current_stage == "text_pipeline"
        
        doc.complete_stage("text_pipeline", {"nodes_created": 5})
        assert "text_pipeline" in doc.completed_stages
        assert doc.current_stage is None
        
        assert doc.is_stage_completed("text_pipeline") is True
        assert doc.is_stage_completed("image_pipeline") is False
    
    def test_ingestion_document_serialization(self):
        """Test IngestionDocument to_dict/from_dict."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType, create_ingestion_document
        )
        
        doc = create_ingestion_document(
            document_id="doc-123",
            tenant_id="tenant-456",
            filename="test.pdf",
            mime_type="application/pdf",
            content=b"test content"
        )
        
        doc.add_node(ContentNode(node_type=NodeType.TEXT, text="Hello"))
        
        # Serialize
        data = doc.to_dict()
        
        assert data["document_id"] == "doc-123"
        assert len(data["nodes"]) == 1
        
        # Deserialize
        doc2 = IngestionDocument.from_dict(data)
        
        assert doc2.document_id == "doc-123"
        assert len(doc2.nodes) == 1


class TestModule7LayoutParser:
    """Tests for Module 7 - Layout Parser."""
    
    def test_layout_block_type_enum(self):
        """Test LayoutBlockType enumeration."""
        from processing.layout_parser import LayoutBlockType
        
        assert LayoutBlockType.TEXT.value == "text"
        assert LayoutBlockType.HEADING.value == "heading"
        assert LayoutBlockType.IMAGE.value == "image"
    
    def test_layout_block_to_content_node(self):
        """Test LayoutBlock to ContentNode conversion."""
        from processing.layout_parser import LayoutBlock, LayoutBlockType
        from schema.intermediate_representation import NodeType
        
        block = LayoutBlock(
            block_type=LayoutBlockType.TEXT,
            text="Hello, World!",
            page=0,
            bbox=(100, 200, 400, 300),
            sequence=1
        )
        
        node = block.to_content_node(page_width=800, page_height=600)
        
        assert node.node_type == NodeType.TEXT
        assert node.text == "Hello, World!"
        assert node.bbox is not None
        assert node.bbox.page == 0
    
    def test_layout_parser_instantiation(self):
        """Test LayoutParser instantiation."""
        from processing.layout_parser import LayoutParser
        
        parser = LayoutParser()
        assert parser is not None
    
    def test_simple_text_parser(self):
        """Test SimpleTextParser for plain text."""
        from processing.layout_parser import SimpleTextParser, LayoutBlockType
        
        parser = SimpleTextParser()
        
        content = b"# Heading\n\nParagraph one.\n\nParagraph two."
        blocks = parser.parse(content, "test.txt", "text/plain")
        
        assert len(blocks) > 0
        # First block should be heading
        assert blocks[0].block_type == LayoutBlockType.HEADING


class TestModule8PipelineExecutor:
    """Tests for Module 8 - Pipeline DAG Executor."""
    
    def test_pipeline_registry(self):
        """Test PipelineRegistry registration."""
        from jobs.pipeline_executor import PipelineRegistry, register_pipeline
        from pipelines.base import BasePipeline
        
        # Clear registry
        PipelineRegistry.clear()
        
        @register_pipeline("test_stage")
        class TestPipeline(BasePipeline):
            async def process(self, doc):
                return doc
        
        assert PipelineRegistry.is_registered("test_stage")
        
        pipeline = PipelineRegistry.get("test_stage")
        assert pipeline is not None
        
        # Cleanup
        PipelineRegistry.clear()
    
    def test_stage_result_dataclass(self):
        """Test StageResult dataclass."""
        from jobs.pipeline_executor import StageResult
        
        result = StageResult(
            stage="text_pipeline",
            success=True,
            duration_seconds=1.5,
            metrics={"nodes_created": 10}
        )
        
        assert result.stage == "text_pipeline"
        assert result.success is True
        assert result.duration_seconds == 1.5
    
    def test_execution_result_dataclass(self):
        """Test ExecutionResult dataclass."""
        from jobs.pipeline_executor import ExecutionResult
        
        result = ExecutionResult(
            job_id="job-123",
            document_id="doc-456",
            success=True,
            total_duration_seconds=5.0,
            stages_executed=["routing", "text_pipeline"]
        )
        
        assert result.job_id == "job-123"
        assert result.success is True
        assert len(result.stages_executed) == 2
    
    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        from jobs.pipeline_executor import ExecutionResult, StageResult
        
        result = ExecutionResult(
            job_id="job-123",
            document_id="doc-456",
            success=True,
            total_duration_seconds=5.0,
            stage_results=[
                StageResult(stage="routing", success=True, duration_seconds=0.1)
            ]
        )
        
        data = result.to_dict()
        
        assert data["job_id"] == "job-123"
        assert len(data["stage_results"]) == 1


class TestModule9TextPipeline:
    """Tests for Module 9 - Text Pipeline."""
    
    def test_markdown_processor(self):
        """Test MarkdownProcessor."""
        from pipelines.text_pipeline import MarkdownProcessor
        from schema.intermediate_representation import NodeType
        
        processor = MarkdownProcessor()
        
        markdown = """# Heading 1

This is a paragraph.

## Heading 2

- List item 1
- List item 2

```python
code block
```
"""
        
        nodes = processor.process(markdown)
        
        assert len(nodes) > 0
        
        # Check for heading
        headings = [n for n in nodes if n.node_type == NodeType.HEADING]
        assert len(headings) >= 2
        assert headings[0].heading_level == 1
    
    def test_html_processor(self):
        """Test HTMLProcessor."""
        from pipelines.text_pipeline import HTMLProcessor
        from schema.intermediate_representation import NodeType
        
        processor = HTMLProcessor()
        
        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>Paragraph text.</p>
            <h2>Subtitle</h2>
        </body>
        </html>
        """
        
        nodes = processor.process(html)
        
        # Should extract headings and paragraphs
        assert len(nodes) > 0
    
    def test_plain_text_processor(self):
        """Test PlainTextProcessor."""
        from pipelines.text_pipeline import PlainTextProcessor
        
        processor = PlainTextProcessor()
        
        text = """CHAPTER ONE

This is the first paragraph of text.

This is the second paragraph."""
        
        nodes = processor.process(text)
        
        assert len(nodes) == 3  # 1 heading + 2 paragraphs
    
    @pytest.mark.asyncio
    async def test_text_pipeline_process(self):
        """Test TextPipeline.process()."""
        from pipelines.text_pipeline import TextPipeline
        from schema.intermediate_representation import IngestionDocument
        
        pipeline = TextPipeline()
        
        doc = IngestionDocument(
            document_id="doc-123",
            tenant_id="tenant-456",
            filename="test.md",
            mime_type="text/markdown"
        )
        doc.raw_content = b"# Hello\n\nWorld"
        
        result = await pipeline.process(doc)
        
        assert len(result.nodes) > 0


class TestModule10ImagePipeline:
    """Tests for Module 10 - Image Pipeline."""
    
    def test_image_utils_detect_format(self):
        """Test ImageUtils format detection."""
        from pipelines.image_pipeline import ImageUtils
        
        # PNG
        png_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert ImageUtils.detect_image_format(png_data) == 'png'
        
        # JPEG
        jpeg_data = b'\xff\xd8\xff' + b'\x00' * 100
        assert ImageUtils.detect_image_format(jpeg_data) == 'jpeg'
        
        # GIF
        gif_data = b'GIF89a' + b'\x00' * 100
        assert ImageUtils.detect_image_format(gif_data) == 'gif'
    
    def test_image_utils_compute_hash(self):
        """Test ImageUtils hash computation."""
        from pipelines.image_pipeline import ImageUtils
        
        data = b'test image data'
        hash1 = ImageUtils.compute_image_hash(data)
        hash2 = ImageUtils.compute_image_hash(data)
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_ocr_service_instantiation(self):
        """Test OCRService instantiation."""
        from pipelines.image_pipeline import OCRService
        
        service = OCRService()
        assert service is not None
        # availability depends on tesseract being installed
    
    def test_vision_service_instantiation(self):
        """Test VisionCaptionService instantiation."""
        from pipelines.image_pipeline import VisionCaptionService
        
        service = VisionCaptionService()
        assert service is not None
    
    @pytest.mark.asyncio
    async def test_image_pipeline_standalone(self):
        """Test ImagePipeline with standalone image."""
        from pipelines.image_pipeline import ImagePipeline
        from schema.intermediate_representation import IngestionDocument
        from unittest.mock import AsyncMock
        
        # Create minimal PNG
        png_header = b'\x89PNG\r\n\x1a\n'
        ihdr_chunk = b'\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6'
        png_data = png_header + ihdr_chunk + b'\x00' * 100
        
        # Mock S3
        mock_s3 = AsyncMock()
        mock_s3.generate_path.return_value = "test/path/image.png"
        mock_s3.upload_derived_asset = AsyncMock(return_value=MagicMock(key="test/path/image.png"))
        
        pipeline = ImagePipeline(
            s3_storage=mock_s3,
            enable_ocr=False,
            enable_captioning=False
        )
        
        doc = IngestionDocument(
            document_id="doc-123",
            tenant_id="tenant-456",
            filename="test.png",
            mime_type="image/png"
        )
        doc.raw_content = png_data
        
        result = await pipeline.process(doc)
        
        # Should have created an image node
        image_nodes = result.get_image_nodes()
        assert len(image_nodes) == 1


class TestPipelineIntegration:
    """Integration tests for pipeline components."""
    
    def test_pipeline_base_class(self):
        """Test BasePipeline abstract class."""
        from pipelines.base import BasePipeline, PipelineError
        
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            BasePipeline()
        
        # PipelineError
        error = PipelineError("test_stage", "Test error", {"key": "value"})
        assert error.stage == "test_stage"
        assert "Test error" in str(error)
    
    def test_ir_flows_through_pipelines(self):
        """Test that IR can flow through multiple pipelines."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        # Create document
        doc = IngestionDocument(
            document_id="doc-123",
            tenant_id="tenant-456",
            filename="test.pdf",
            mime_type="application/pdf"
        )
        
        # Simulate layout parser adding nodes
        doc.add_node(ContentNode(node_type=NodeType.TEXT, text="Text 1", sequence=0))
        doc.add_node(ContentNode(node_type=NodeType.IMAGE, sequence=1))
        doc.add_node(ContentNode(node_type=NodeType.TEXT, text="Text 2", sequence=2))
        doc.complete_stage("layout_parsing")
        
        # Verify state
        assert len(doc.nodes) == 3
        assert doc.is_stage_completed("layout_parsing")
        
        # Text pipeline would enhance text nodes
        for node in doc.get_text_nodes():
            node.source_pipeline = "text_pipeline"
        doc.complete_stage("text_pipeline")
        
        # Image pipeline would process images
        for node in doc.get_image_nodes():
            node.image_path = "s3://bucket/path"
        doc.complete_stage("image_pipeline")
        
        # Verify final state
        assert len(doc.completed_stages) == 3
        assert doc.get_image_nodes()[0].image_path == "s3://bucket/path"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
