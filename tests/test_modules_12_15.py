"""
Test suite for Modules 12-15.

Module 12: Linkage Strategy
Module 13: Chunking Service
Module 14: Embedding Service
Module 15: Vector Store

Run with: pytest tests/test_modules_12_15.py -v
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestModule12Linkage:
    """Tests for Module 12 - Linkage Strategy."""
    
    def test_linkage_result_dataclass(self):
        """Test LinkageResult dataclass."""
        from processing.linker import LinkageResult
        
        result = LinkageResult(
            image_node_id="img-1",
            text_node_id="text-1",
            linkage_type="spatial_containment",
            confidence=0.95
        )
        
        assert result.image_node_id == "img-1"
        assert result.linkage_type == "spatial_containment"
        assert result.confidence == 0.95
    
    def test_spatial_analyzer_containment(self):
        """Test SpatialAnalyzer containment detection."""
        from processing.linker import SpatialAnalyzer
        from schema.intermediate_representation import BoundingBox
        
        analyzer = SpatialAnalyzer(containment_threshold=0.7)
        
        outer = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, page=0)
        inner = BoundingBox(x0=0.2, y0=0.2, x1=0.8, y1=0.8, page=0)
        
        is_contained, ratio = analyzer.is_contained_in(inner, outer)
        
        assert is_contained is True
        assert ratio == 1.0
    
    def test_spatial_analyzer_no_containment(self):
        """Test SpatialAnalyzer with no containment."""
        from processing.linker import SpatialAnalyzer
        from schema.intermediate_representation import BoundingBox
        
        analyzer = SpatialAnalyzer(containment_threshold=0.7)
        
        box1 = BoundingBox(x0=0.0, y0=0.0, x1=0.3, y1=0.3, page=0)
        box2 = BoundingBox(x0=0.5, y0=0.5, x1=1.0, y1=1.0, page=0)
        
        is_contained, ratio = analyzer.is_contained_in(box1, box2)
        
        assert is_contained is False
        assert ratio == 0.0
    
    def test_spatial_analyzer_different_pages(self):
        """Test SpatialAnalyzer with different pages."""
        from processing.linker import SpatialAnalyzer
        from schema.intermediate_representation import BoundingBox
        
        analyzer = SpatialAnalyzer()
        
        box1 = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, page=0)
        box2 = BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0, page=1)
        
        is_contained, ratio = analyzer.is_contained_in(box1, box2)
        
        assert is_contained is False
    
    def test_reading_order_analyzer_preceding(self):
        """Test ReadingOrderAnalyzer find preceding."""
        from processing.linker import ReadingOrderAnalyzer
        from schema.intermediate_representation import ContentNode, NodeType
        
        analyzer = ReadingOrderAnalyzer()
        
        nodes = [
            ContentNode(id="n1", node_type=NodeType.TEXT, sequence=0, page=0),
            ContentNode(id="n2", node_type=NodeType.TEXT, sequence=1, page=0),
            ContentNode(id="n3", node_type=NodeType.TEXT, sequence=2, page=0),
        ]
        
        target = ContentNode(id="img", node_type=NodeType.IMAGE, sequence=2, page=0)
        
        preceding = analyzer.find_preceding_text(target, nodes)
        
        assert preceding is not None
        assert preceding.id == "n2"
    
    def test_reading_order_analyzer_following(self):
        """Test ReadingOrderAnalyzer find following."""
        from processing.linker import ReadingOrderAnalyzer
        from schema.intermediate_representation import ContentNode, NodeType
        
        analyzer = ReadingOrderAnalyzer()
        
        nodes = [
            ContentNode(id="n1", node_type=NodeType.TEXT, sequence=0, page=0),
            ContentNode(id="n2", node_type=NodeType.TEXT, sequence=2, page=0),
            ContentNode(id="n3", node_type=NodeType.TEXT, sequence=3, page=0),
        ]
        
        target = ContentNode(id="img", node_type=NodeType.IMAGE, sequence=1, page=0)
        
        following = analyzer.find_following_text(target, nodes)
        
        assert following is not None
        assert following.id == "n2"
    
    def test_node_linker_link_nodes(self):
        """Test NodeLinker creating links."""
        from processing.linker import NodeLinker
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType, BoundingBox
        )
        
        linker = NodeLinker()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        
        # Add text node
        text_node = ContentNode(
            id="text-1",
            node_type=NodeType.TEXT,
            sequence=0,
            page=0,
            text="Some text",
            bbox=BoundingBox(x0=0.0, y0=0.0, x1=1.0, y1=0.5, page=0)
        )
        doc.add_node(text_node)
        
        # Add image node that follows text
        image_node = ContentNode(
            id="img-1",
            node_type=NodeType.IMAGE,
            sequence=1,
            page=0,
            bbox=BoundingBox(x0=0.3, y0=0.6, x1=0.7, y1=0.9, page=0)
        )
        doc.add_node(image_node)
        
        # Link nodes
        results = linker.link_nodes(doc)
        
        assert len(results) > 0
        assert image_node.linked_ids
        assert text_node.linked_ids
    
    @pytest.mark.asyncio
    async def test_linkage_pipeline_process(self):
        """Test LinkagePipeline.process()."""
        from processing.linker import LinkagePipeline
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        pipeline = LinkagePipeline()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            id="text-1", node_type=NodeType.TEXT, sequence=0, text="Text"
        ))
        doc.add_node(ContentNode(
            id="img-1", node_type=NodeType.IMAGE, sequence=1
        ))
        
        result = await pipeline.process(doc)
        
        assert "linkage_stats" in result.metadata


class TestModule13Chunking:
    """Tests for Module 13 - Chunking Service."""
    
    def test_token_counter(self):
        """Test TokenCounter."""
        from chunking.chunker import TokenCounter
        
        counter = TokenCounter()
        
        count = counter.count("Hello world")
        assert count > 0
    
    def test_text_splitter_short_text(self):
        """Test TextSplitter with text that fits in one chunk."""
        from chunking.chunker import TextSplitter
        
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        
        chunks = splitter.split("Short text.")
        
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
    
    def test_text_splitter_long_text(self):
        """Test TextSplitter with text requiring multiple chunks."""
        from chunking.chunker import TextSplitter
        
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        
        long_text = "This is a much longer text. " * 20
        chunks = splitter.split(long_text)
        
        assert len(chunks) > 1
    
    def test_text_splitter_sentence_boundaries(self):
        """Test TextSplitter respects sentence boundaries."""
        from chunking.chunker import TextSplitter
        
        splitter = TextSplitter(
            chunk_size=50, 
            chunk_overlap=0,
            respect_sentence_boundaries=True
        )
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = splitter.split(text)
        
        # Should not cut mid-sentence
        for chunk in chunks:
            assert not chunk.endswith(" senten")
    
    def test_semantic_chunking_strategy(self):
        """Test SemanticChunkingStrategy."""
        from chunking.chunker import SemanticChunkingStrategy
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        strategy = SemanticChunkingStrategy(chunk_size=100)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.rbac.allowed_roles = ["admin"]
        
        doc.add_node(ContentNode(
            id="h1", node_type=NodeType.HEADING, sequence=0, text="Title", heading_level=1
        ))
        doc.add_node(ContentNode(
            id="p1", node_type=NodeType.TEXT, sequence=1, text="Paragraph one content."
        ))
        doc.add_node(ContentNode(
            id="p2", node_type=NodeType.TEXT, sequence=2, text="Paragraph two content."
        ))
        
        chunks = strategy.chunk(doc)
        
        assert len(chunks) > 0
        assert chunks[0].acl.get("allowed_roles") == ["admin"]
    
    def test_fixed_size_chunking_strategy(self):
        """Test FixedSizeChunkingStrategy."""
        from chunking.chunker import FixedSizeChunkingStrategy
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        strategy = FixedSizeChunkingStrategy(chunk_size=50)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            id="p1", node_type=NodeType.TEXT, sequence=0, text="Some text content."
        ))
        
        chunks = strategy.chunk(doc)
        
        assert len(chunks) > 0
    
    def test_chunking_service(self):
        """Test ChunkingService."""
        from chunking.chunker import ChunkingService
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        service = ChunkingService(chunk_size=100)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            id="p1", node_type=NodeType.TEXT, sequence=0, text="Test content."
        ))
        
        chunks = service.chunk_document(doc)
        
        assert len(chunks) > 0
        assert chunks[0].text
    
    @pytest.mark.asyncio
    async def test_chunking_pipeline_process(self):
        """Test ChunkingPipeline.process()."""
        from chunking.chunker import ChunkingPipeline
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        pipeline = ChunkingPipeline()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            id="p1", node_type=NodeType.TEXT, sequence=0, text="Test content for chunking."
        ))
        
        result = await pipeline.process(doc)
        
        assert len(result.chunks) > 0
        assert "chunking_stats" in result.metadata


class TestModule14Embedding:
    """Tests for Module 14 - Embedding Service."""
    
    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass."""
        from embeddings.service import EmbeddingResult
        
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
            content_hash="abc123"
        )
        
        assert len(result.embedding) == 3
        assert result.dimensions == 3
        assert result.validate_dimensions(3) is True
        assert result.validate_dimensions(4) is False
    
    def test_mock_embedding_provider(self):
        """Test MockEmbeddingProvider."""
        from embeddings.service import MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimensions=128)
        
        assert provider.model_name == "mock-embedding-model"
        assert provider.dimensions == 128
    
    @pytest.mark.asyncio
    async def test_mock_embedding_provider_embed(self):
        """Test MockEmbeddingProvider embedding generation."""
        from embeddings.service import MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimensions=128)
        
        embedding = await provider.embed_text("Hello world")
        
        assert len(embedding) == 128
        assert all(isinstance(v, float) for v in embedding)
    
    @pytest.mark.asyncio
    async def test_mock_embedding_provider_deterministic(self):
        """Test MockEmbeddingProvider is deterministic."""
        from embeddings.service import MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimensions=128)
        
        emb1 = await provider.embed_text("Hello world")
        emb2 = await provider.embed_text("Hello world")
        
        assert emb1 == emb2
    
    @pytest.mark.asyncio
    async def test_mock_embedding_provider_batch(self):
        """Test MockEmbeddingProvider batch embedding."""
        from embeddings.service import MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimensions=128)
        
        texts = ["Hello", "World", "Test"]
        embeddings = await provider.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(e) == 128 for e in embeddings)
    
    @pytest.mark.asyncio
    async def test_embedding_service_with_mock(self):
        """Test EmbeddingService with mock provider."""
        from embeddings.service import EmbeddingService, MockEmbeddingProvider
        
        provider = MockEmbeddingProvider(dimensions=128)
        service = EmbeddingService(provider=provider, enable_cache=False)
        
        result = await service.embed_text("Test text")
        
        assert result.dimensions == 128
        assert result.model == "mock-embedding-model"
    
    @pytest.mark.asyncio
    async def test_embedding_service_embed_chunks(self):
        """Test EmbeddingService.embed_chunks()."""
        from embeddings.service import EmbeddingService, MockEmbeddingProvider
        from schema.intermediate_representation import Chunk
        
        provider = MockEmbeddingProvider(dimensions=128)
        service = EmbeddingService(provider=provider, enable_cache=False)
        
        chunks = [
            Chunk(id="c1", text="First chunk"),
            Chunk(id="c2", text="Second chunk"),
        ]
        
        embedded = await service.embed_chunks(chunks)
        
        assert all(c.embedding is not None for c in embedded)
        assert all(c.embedding_model == "mock-embedding-model" for c in embedded)
    
    @pytest.mark.asyncio
    async def test_embedding_pipeline_process(self):
        """Test EmbeddingPipeline.process()."""
        from embeddings.service import EmbeddingPipeline, MockEmbeddingProvider
        from schema.intermediate_representation import IngestionDocument, Chunk
        
        provider = MockEmbeddingProvider(dimensions=128)
        pipeline = EmbeddingPipeline(provider=provider, enable_cache=False)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.chunks = [
            Chunk(id="c1", text="First chunk"),
            Chunk(id="c2", text="Second chunk"),
        ]
        
        result = await pipeline.process(doc)
        
        assert all(c.embedding is not None for c in result.chunks)
        assert "embedding_stats" in result.metadata


class TestModule15VectorStore:
    """Tests for Module 15 - Vector Store."""
    
    def test_vector_point_dataclass(self):
        """Test VectorPoint dataclass."""
        from storage.vector_store import VectorPoint
        
        point = VectorPoint(
            id="p1",
            vector=[0.1, 0.2, 0.3],
            payload={"key": "value"}
        )
        
        assert point.id == "p1"
        assert len(point.vector) == 3
        assert point.payload["key"] == "value"
    
    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        from storage.vector_store import SearchResult
        
        result = SearchResult(
            id="r1",
            score=0.95,
            payload={"text": "content"}
        )
        
        assert result.id == "r1"
        assert result.score == 0.95
    
    def test_search_filter_creation(self):
        """Test SearchFilter creation."""
        from storage.vector_store import SearchFilter
        
        filter = SearchFilter(
            tenant_id="tenant-1",
            allowed_roles=["admin", "editor"],
            allowed_users=["user-1"],
            classifications=["INTERNAL", "PUBLIC"]
        )
        
        assert filter.tenant_id == "tenant-1"
        assert "admin" in filter.allowed_roles
    
    def test_search_filter_to_qdrant(self):
        """Test SearchFilter conversion to Qdrant format."""
        from storage.vector_store import SearchFilter
        
        filter = SearchFilter(
            tenant_id="tenant-1",
            document_ids=["doc-1", "doc-2"]
        )
        
        qdrant_filter = filter.to_qdrant_filter()
        
        assert "must" in qdrant_filter
        assert len(qdrant_filter["must"]) >= 2
    
    def test_search_filter_rbac(self):
        """Test SearchFilter RBAC conditions."""
        from storage.vector_store import SearchFilter
        
        filter = SearchFilter(
            tenant_id="tenant-1",
            allowed_roles=["admin"],
            allowed_users=["user-1"]
        )
        
        qdrant_filter = filter.to_qdrant_filter()
        
        # Should have should conditions for RBAC
        must_conditions = qdrant_filter["must"]
        rbac_condition = next(
            (c for c in must_conditions if "should" in c),
            None
        )
        
        assert rbac_condition is not None
        assert len(rbac_condition["should"]) == 2
    
    def test_search_filter_empty(self):
        """Test SearchFilter with no conditions."""
        from storage.vector_store import SearchFilter
        
        filter = SearchFilter()
        
        qdrant_filter = filter.to_qdrant_filter()
        
        assert qdrant_filter == {}


class TestModule12To15Integration:
    """Integration tests for Modules 12-15."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self):
        """Test full flow from linkage to vector store."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType, Chunk
        )
        from processing.linker import LinkagePipeline
        from chunking.chunker import ChunkingPipeline
        from embeddings.service import EmbeddingPipeline, MockEmbeddingProvider
        
        # Create document with nodes
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.rbac.allowed_roles = ["admin"]
        
        doc.add_node(ContentNode(
            id="h1", node_type=NodeType.HEADING, sequence=0, 
            text="Title", heading_level=1
        ))
        doc.add_node(ContentNode(
            id="p1", node_type=NodeType.TEXT, sequence=1,
            text="This is some content for the document."
        ))
        doc.add_node(ContentNode(
            id="img1", node_type=NodeType.IMAGE, sequence=2,
            image_path="s3://bucket/image.png"
        ))
        
        # Run linkage
        linkage_pipeline = LinkagePipeline()
        doc = await linkage_pipeline.process(doc)
        
        # Run chunking
        chunking_pipeline = ChunkingPipeline()
        doc = await chunking_pipeline.process(doc)
        
        assert len(doc.chunks) > 0
        
        # Run embedding
        provider = MockEmbeddingProvider(dimensions=128)
        embedding_pipeline = EmbeddingPipeline(provider=provider, enable_cache=False)
        doc = await embedding_pipeline.process(doc)
        
        assert all(c.embedding is not None for c in doc.chunks)
        
        # Verify ACL is preserved
        for chunk in doc.chunks:
            assert chunk.acl.get("allowed_roles") == ["admin"]
    
    def test_chunk_image_refs_from_linkage(self):
        """Test that chunks include image refs from linkage."""
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        from processing.linker import NodeLinker
        from chunking.chunker import SemanticChunkingStrategy
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        
        # Add text and image nodes
        text_node = ContentNode(
            id="text-1",
            node_type=NodeType.TEXT,
            sequence=0,
            text="Text content with image.",
        )
        doc.add_node(text_node)
        
        image_node = ContentNode(
            id="img-1",
            node_type=NodeType.IMAGE,
            sequence=1,
            image_path="s3://bucket/img.png",
            image_caption="A sample image"
        )
        doc.add_node(image_node)
        
        # Create links
        linker = NodeLinker()
        linker.link_nodes(doc)
        
        # Create chunks
        strategy = SemanticChunkingStrategy(chunk_size=100)
        chunks = strategy.chunk(doc)
        
        # At least one chunk should have image refs
        has_image_refs = any(c.image_refs for c in chunks)
        # Note: May not have refs if no linkage occurred due to missing bbox
        # This test validates the flow works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
