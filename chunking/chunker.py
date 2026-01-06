"""
MODULE 13 — CHUNKING SERVICE (MULTI-MODAL SAFE)
===============================================
Chunking over IR for the Enterprise RAG Ingestion Pipeline.

Rules:
- Chunk TEXT nodes
- Attach linked IMAGE nodes as supporting evidence

Output chunks must include:
- text
- image_refs
- acl metadata

The chunker creates retrieval-ready chunks that preserve
multi-modal relationships and carry RBAC metadata.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pipelines.base import BasePipeline
from schema.intermediate_representation import (
    Chunk,
    ContentNode,
    IngestionDocument,
    NodeType,
    ProcessingStatus,
)
from config.settings import get_settings
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Token Counter
# ============================================================================

class TokenCounter:
    """
    Counts tokens in text.
    
    Uses tiktoken if available, falls back to word-based estimation.
    """
    
    def __init__(self, model: str = "cl100k_base"):
        """
        Initialize token counter.
        
        Args:
            model: Tiktoken encoding model name
        """
        self.model = model
        self._encoder = None
        self._tiktoken_available = self._check_tiktoken()
    
    def _check_tiktoken(self) -> bool:
        """Check if tiktoken is available."""
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.model)
            return True
        except (ImportError, Exception):
            return False
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self._tiktoken_available and self._encoder:
            return len(self._encoder.encode(text))
        
        # Fallback: estimate ~4 characters per token
        return len(text) // 4
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to maximum token count.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        if self._tiktoken_available and self._encoder:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoder.decode(tokens[:max_tokens])
        
        # Fallback: estimate
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]


# ============================================================================
# Text Splitter
# ============================================================================

class TextSplitter:
    """
    Splits text into chunks while respecting boundaries.
    
    Supports:
    - Sentence boundary detection
    - Paragraph boundary detection
    - Overlap between chunks
    """
    
    # Sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
    PARAGRAPH_ENDINGS = re.compile(r'\n\n+')
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        respect_sentence_boundaries: bool = True
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            respect_sentence_boundaries: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.token_counter = TokenCounter()
    
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Check if text fits in single chunk
        token_count = self.token_counter.count(text)
        if token_count <= self.chunk_size:
            return [text.strip()]
        
        # Split into sentences first
        if self.respect_sentence_boundaries:
            sentences = self._split_into_sentences(text)
            return self._combine_sentences_into_chunks(sentences)
        else:
            return self._split_by_tokens(text)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # First split by paragraphs
        paragraphs = self.PARAGRAPH_ENDINGS.split(text)
        
        sentences = []
        for para in paragraphs:
            if not para.strip():
                continue
            
            # Split paragraph into sentences
            para_sentences = self.SENTENCE_ENDINGS.split(para)
            sentences.extend([s.strip() for s in para_sentences if s.strip()])
        
        return sentences
    
    def _combine_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Combine sentences into chunks respecting size limits."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by tokens
                sub_chunks = self._split_by_tokens(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk,
                        self.chunk_overlap
                    )
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(
                        self.token_counter.count(s) for s in current_chunk
                    )
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int
    ) -> List[str]:
        """Get sentences from end for overlap."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count(sentence)
            if tokens + sentence_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by token count (fallback for long text)."""
        chunks = []
        
        words = text.split()
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.token_counter.count(word)
            
            if current_tokens + word_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


# ============================================================================
# Chunking Strategy
# ============================================================================

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(
        self,
        document: IngestionDocument
    ) -> List[Chunk]:
        """
        Create chunks from document.
        
        Args:
            document: IngestionDocument to chunk
            
        Returns:
            List of Chunk objects
        """
        pass


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Semantic chunking strategy.
    
    Groups text nodes by semantic structure (headings, sections)
    and creates chunks that preserve document hierarchy.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        include_headings: bool = True
    ):
        """
        Initialize semantic chunking strategy.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            include_headings: Include heading context in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.include_headings = include_headings
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.token_counter = TokenCounter()
    
    def chunk(self, document: IngestionDocument) -> List[Chunk]:
        """Create chunks using semantic strategy."""
        chunks = []
        
        # Get nodes in reading order
        nodes = document.get_nodes_in_reading_order()
        
        # Track current section context
        current_heading = None
        current_section_nodes = []
        
        for node in nodes:
            # Skip non-text nodes (images handled via linkage)
            if node.node_type == NodeType.IMAGE:
                continue
            
            # Check for heading
            if node.node_type == NodeType.HEADING:
                # Process accumulated section
                if current_section_nodes:
                    section_chunks = self._chunk_section(
                        current_section_nodes,
                        current_heading,
                        document
                    )
                    chunks.extend(section_chunks)
                
                # Start new section
                current_heading = node
                current_section_nodes = []
            else:
                current_section_nodes.append(node)
        
        # Process final section
        if current_section_nodes:
            section_chunks = self._chunk_section(
                current_section_nodes,
                current_heading,
                document
            )
            chunks.extend(section_chunks)
        
        # Assign indices
        for i, chunk in enumerate(chunks):
            chunk.index = i
        
        return chunks
    
    def _chunk_section(
        self,
        nodes: List[ContentNode],
        heading: Optional[ContentNode],
        document: IngestionDocument
    ) -> List[Chunk]:
        """Chunk a section of nodes."""
        if not nodes:
            return []
        
        # Combine text from nodes
        combined_text = ""
        if self.include_headings and heading and heading.text:
            combined_text = f"{heading.text}\n\n"
        
        combined_text += " ".join(n.text for n in nodes if n.text)
        
        if not combined_text.strip():
            return []
        
        # Split into chunks
        text_chunks = self.splitter.split(combined_text)
        
        # Create Chunk objects
        chunks = []
        source_node_ids = [n.id for n in nodes]
        if heading:
            source_node_ids.insert(0, heading.id)
        
        # Collect linked images from all source nodes
        image_refs = self._collect_image_refs(nodes, document)
        
        # Determine page range
        pages = [n.page for n in nodes if n.page is not None]
        start_page = min(pages) if pages else None
        end_page = max(pages) if pages else None
        
        for text in text_chunks:
            chunk = Chunk(
                text=text,
                token_count=self.token_counter.count(text),
                source_node_ids=source_node_ids.copy(),
                image_refs=image_refs.copy(),
                start_page=start_page,
                end_page=end_page,
                content_hash=self._compute_hash(text),
                acl=document.rbac.to_dict()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _collect_image_refs(
        self,
        nodes: List[ContentNode],
        document: IngestionDocument
    ) -> List[Dict[str, Any]]:
        """Collect image references from linked nodes."""
        image_refs = []
        seen_ids = set()
        
        for node in nodes:
            for linked_id in node.linked_ids:
                if linked_id in seen_ids:
                    continue
                
                linked_node = document.get_node(linked_id)
                if linked_node and linked_node.node_type == NodeType.IMAGE:
                    seen_ids.add(linked_id)
                    image_refs.append({
                        "node_id": linked_node.id,
                        "image_path": linked_node.image_path,
                        "caption": linked_node.image_caption,
                        "ocr_text": linked_node.ocr_text
                    })
        
        return image_refs
    
    def _compute_hash(self, text: str) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Fixed-size chunking strategy.
    
    Simple strategy that creates chunks of fixed token size
    with optional overlap.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize fixed-size chunking strategy.
        
        Args:
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.token_counter = TokenCounter()
    
    def chunk(self, document: IngestionDocument) -> List[Chunk]:
        """Create fixed-size chunks."""
        # Combine all text
        text_nodes = document.get_text_nodes()
        full_text = " ".join(n.text for n in text_nodes if n.text)
        
        if not full_text.strip():
            return []
        
        # Split
        text_chunks = self.splitter.split(full_text)
        
        # Collect all image refs
        image_refs = []
        for node in document.get_image_nodes():
            image_refs.append({
                "node_id": node.id,
                "image_path": node.image_path,
                "caption": node.image_caption,
                "ocr_text": node.ocr_text
            })
        
        # Create chunks
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk = Chunk(
                index=i,
                text=text,
                token_count=self.token_counter.count(text),
                source_node_ids=[n.id for n in text_nodes],
                image_refs=image_refs,
                content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
                acl=document.rbac.to_dict()
            )
            chunks.append(chunk)
        
        return chunks


# ============================================================================
# Chunking Service
# ============================================================================

class ChunkingService:
    """
    Main chunking service.
    
    Coordinates chunking strategies and produces
    retrieval-ready chunks with multi-modal support.
    """
    
    def __init__(
        self,
        strategy: Optional[ChunkingStrategy] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize chunking service.
        
        Args:
            strategy: Chunking strategy (default: SemanticChunkingStrategy)
            chunk_size: Override chunk size from settings
            chunk_overlap: Override chunk overlap from settings
        """
        settings = get_settings()
        
        self.chunk_size = chunk_size or settings.chunking.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        
        self.strategy = strategy or SemanticChunkingStrategy(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_document(self, document: IngestionDocument) -> List[Chunk]:
        """
        Chunk a document.
        
        Args:
            document: IngestionDocument to chunk
            
        Returns:
            List of Chunk objects
        """
        return self.strategy.chunk(document)
    
    def set_strategy(self, strategy: ChunkingStrategy) -> None:
        """Set chunking strategy."""
        self.strategy = strategy


# ============================================================================
# Chunking Pipeline
# ============================================================================

@register_pipeline("chunking")
class ChunkingPipeline(BasePipeline):
    """
    Pipeline stage for chunking documents.
    
    Creates retrieval-ready chunks with:
    - Text content
    - Image references (from linkage)
    - ACL metadata
    """
    
    stage_name = "chunking"
    
    def __init__(
        self,
        strategy: str = "semantic",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize chunking pipeline.
        
        Args:
            strategy: Strategy name ("semantic" or "fixed")
            chunk_size: Override chunk size
            chunk_overlap: Override chunk overlap
        """
        super().__init__()
        
        settings = get_settings()
        size = chunk_size or settings.chunking.default_chunk_size
        overlap = chunk_overlap or settings.chunking.chunk_overlap
        
        if strategy == "fixed":
            chunking_strategy = FixedSizeChunkingStrategy(size, overlap)
        else:
            chunking_strategy = SemanticChunkingStrategy(size, overlap)
        
        self.service = ChunkingService(strategy=chunking_strategy)
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document to create chunks.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Document with chunks populated
        """
        # Create chunks
        chunks = self.service.chunk_document(document)
        
        # Add to document
        document.add_chunks(chunks)
        
        # Statistics
        total_tokens = sum(c.token_count for c in chunks)
        chunks_with_images = len([c for c in chunks if c.image_refs])
        
        document.metadata["chunking_stats"] = {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(chunks) if chunks else 0,
            "chunks_with_images": chunks_with_images,
        }
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if chunking can be skipped."""
        # Skip if already has chunks
        if document.chunks:
            return True
        
        return document.is_stage_completed(self.stage_name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TokenCounter",
    "TextSplitter",
    "ChunkingStrategy",
    "SemanticChunkingStrategy",
    "FixedSizeChunkingStrategy",
    "ChunkingService",
    "ChunkingPipeline",
]
