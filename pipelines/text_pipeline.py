"""
MODULE 9 — TEXT PIPELINE (PDF-FIRST)
====================================
Text extraction and processing pipeline for the Enterprise RAG Ingestion Pipeline.

Inputs:
- Layout blocks (from layout parser)
- OR raw content (for non-PDF documents)

Responsibilities:
- Produce TEXT nodes
- Preserve headings with levels
- Attach bounding boxes
- Handle various text formats (plain text, markdown, HTML)

Output:
- Mutated IngestionDocument with TEXT nodes populated
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pipelines.base import BasePipeline, PipelineError
from schema.intermediate_representation import (
    BoundingBox,
    ContentNode,
    IngestionDocument,
    NodeType,
    ProcessingStatus,
)
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Text Processors
# ============================================================================

class MarkdownProcessor:
    """Process markdown text into structured nodes."""
    
    # Regex patterns
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    LIST_ITEM_PATTERN = re.compile(r'^(\s*)[-*+]\s+(.+)$', re.MULTILINE)
    NUMBERED_LIST_PATTERN = re.compile(r'^(\s*)\d+\.\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    INLINE_CODE_PATTERN = re.compile(r'`([^`]+)`')
    BOLD_PATTERN = re.compile(r'\*\*(.+?)\*\*')
    ITALIC_PATTERN = re.compile(r'\*(.+?)\*')
    LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    
    def process(self, text: str, start_sequence: int = 0) -> List[ContentNode]:
        """
        Process markdown text into content nodes.
        
        Args:
            text: Markdown text
            start_sequence: Starting sequence number
            
        Returns:
            List of ContentNode objects
        """
        nodes = []
        sequence = start_sequence
        
        # Extract code blocks first (to avoid processing their contents)
        code_blocks = []
        def store_code_block(match):
            code_blocks.append({
                'language': match.group(1),
                'code': match.group(2)
            })
            return f'\n__CODE_BLOCK_{len(code_blocks) - 1}__\n'
        
        text = self.CODE_BLOCK_PATTERN.sub(store_code_block, text)
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check for code block placeholder
            code_match = re.match(r'__CODE_BLOCK_(\d+)__', para)
            if code_match:
                idx = int(code_match.group(1))
                if idx < len(code_blocks):
                    block = code_blocks[idx]
                    nodes.append(ContentNode(
                        node_type=NodeType.CODE,
                        sequence=sequence,
                        text=block['code'],
                        metadata={'language': block['language']},
                        source_pipeline='text_pipeline'
                    ))
                    sequence += 1
                continue
            
            # Check for heading
            heading_match = self.HEADING_PATTERN.match(para)
            if heading_match:
                level = len(heading_match.group(1))
                text_content = heading_match.group(2).strip()
                nodes.append(ContentNode(
                    node_type=NodeType.HEADING,
                    sequence=sequence,
                    text=text_content,
                    heading_level=level,
                    source_pipeline='text_pipeline'
                ))
                sequence += 1
                continue
            
            # Check for list items in paragraph
            if self.LIST_ITEM_PATTERN.search(para) or self.NUMBERED_LIST_PATTERN.search(para):
                # Process each line as list item
                for line in para.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    list_match = self.LIST_ITEM_PATTERN.match(line)
                    if not list_match:
                        list_match = self.NUMBERED_LIST_PATTERN.match(line)
                    
                    if list_match:
                        item_text = list_match.group(2)
                        nodes.append(ContentNode(
                            node_type=NodeType.LIST_ITEM,
                            sequence=sequence,
                            text=self._clean_inline_formatting(item_text),
                            source_pipeline='text_pipeline'
                        ))
                        sequence += 1
                continue
            
            # Regular paragraph
            clean_text = self._clean_inline_formatting(para)
            nodes.append(ContentNode(
                node_type=NodeType.TEXT,
                sequence=sequence,
                text=clean_text,
                source_pipeline='text_pipeline'
            ))
            sequence += 1
        
        return nodes
    
    def _clean_inline_formatting(self, text: str) -> str:
        """Remove markdown formatting, keeping plain text."""
        # Remove links but keep text
        text = self.LINK_PATTERN.sub(r'\1', text)
        # Remove bold/italic markers
        text = self.BOLD_PATTERN.sub(r'\1', text)
        text = self.ITALIC_PATTERN.sub(r'\1', text)
        # Remove inline code markers
        text = self.INLINE_CODE_PATTERN.sub(r'\1', text)
        return text


class HTMLProcessor:
    """Process HTML text into structured nodes."""
    
    def __init__(self):
        """Initialize HTML processor."""
        self._bs4_available = self._check_bs4()
    
    def _check_bs4(self) -> bool:
        """Check if BeautifulSoup is available."""
        try:
            from bs4 import BeautifulSoup
            return True
        except ImportError:
            return False
    
    def process(self, html: str, start_sequence: int = 0) -> List[ContentNode]:
        """
        Process HTML into content nodes.
        
        Args:
            html: HTML content
            start_sequence: Starting sequence number
            
        Returns:
            List of ContentNode objects
        """
        if self._bs4_available:
            return self._process_with_bs4(html, start_sequence)
        else:
            return self._process_simple(html, start_sequence)
    
    def _process_with_bs4(self, html: str, start_sequence: int) -> List[ContentNode]:
        """Process HTML using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        nodes = []
        sequence = start_sequence
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Process elements
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'pre', 'code']):
            text = element.get_text(strip=True)
            if not text:
                continue
            
            tag = element.name.lower()
            
            if tag.startswith('h') and len(tag) == 2:
                # Heading
                level = int(tag[1])
                nodes.append(ContentNode(
                    node_type=NodeType.HEADING,
                    sequence=sequence,
                    text=text,
                    heading_level=level,
                    source_pipeline='text_pipeline'
                ))
            elif tag == 'li':
                # List item
                nodes.append(ContentNode(
                    node_type=NodeType.LIST_ITEM,
                    sequence=sequence,
                    text=text,
                    source_pipeline='text_pipeline'
                ))
            elif tag in ('pre', 'code'):
                # Code
                nodes.append(ContentNode(
                    node_type=NodeType.CODE,
                    sequence=sequence,
                    text=text,
                    source_pipeline='text_pipeline'
                ))
            else:
                # Paragraph
                nodes.append(ContentNode(
                    node_type=NodeType.TEXT,
                    sequence=sequence,
                    text=text,
                    source_pipeline='text_pipeline'
                ))
            
            sequence += 1
        
        return nodes
    
    def _process_simple(self, html: str, start_sequence: int) -> List[ContentNode]:
        """Simple HTML processing without BeautifulSoup."""
        import re
        
        # Remove tags but try to preserve structure
        # Remove script/style content
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        nodes = []
        sequence = start_sequence
        
        # Extract headings
        for match in re.finditer(r'<h([1-6])[^>]*>(.*?)</h\1>', html, re.DOTALL | re.IGNORECASE):
            level = int(match.group(1))
            text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            if text:
                nodes.append(ContentNode(
                    node_type=NodeType.HEADING,
                    sequence=sequence,
                    text=text,
                    heading_level=level,
                    source_pipeline='text_pipeline'
                ))
                sequence += 1
        
        # Extract paragraphs
        for match in re.finditer(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE):
            text = re.sub(r'<[^>]+>', '', match.group(1)).strip()
            if text:
                nodes.append(ContentNode(
                    node_type=NodeType.TEXT,
                    sequence=sequence,
                    text=text,
                    source_pipeline='text_pipeline'
                ))
                sequence += 1
        
        return nodes


class PlainTextProcessor:
    """Process plain text into structured nodes."""
    
    def process(self, text: str, start_sequence: int = 0) -> List[ContentNode]:
        """
        Process plain text into content nodes.
        
        Args:
            text: Plain text content
            start_sequence: Starting sequence number
            
        Returns:
            List of ContentNode objects
        """
        nodes = []
        sequence = start_sequence
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if it looks like a heading (short, no period, possibly uppercase)
            lines = para.split('\n')
            if len(lines) == 1 and len(para) < 100 and not para.endswith('.'):
                # Might be a heading
                if para.isupper() or (para[0].isupper() and para.count(' ') < 10):
                    nodes.append(ContentNode(
                        node_type=NodeType.HEADING,
                        sequence=sequence,
                        text=para,
                        heading_level=2,  # Default to H2
                        source_pipeline='text_pipeline'
                    ))
                    sequence += 1
                    continue
            
            # Regular text
            nodes.append(ContentNode(
                node_type=NodeType.TEXT,
                sequence=sequence,
                text=para,
                source_pipeline='text_pipeline'
            ))
            sequence += 1
        
        return nodes


# ============================================================================
# Text Pipeline
# ============================================================================

@register_pipeline("text_pipeline")
class TextPipeline(BasePipeline):
    """
    Text extraction and processing pipeline.
    
    Handles:
    - PDF documents (uses layout parser output)
    - Plain text
    - Markdown
    - HTML
    """
    
    stage_name = "text_pipeline"
    
    def __init__(self):
        """Initialize text pipeline."""
        super().__init__()
        self.markdown_processor = MarkdownProcessor()
        self.html_processor = HTMLProcessor()
        self.plain_text_processor = PlainTextProcessor()
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document text content.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Updated IngestionDocument with TEXT nodes
        """
        # Check if layout parser already created nodes (PDF flow)
        existing_text_nodes = document.get_text_nodes()
        
        if existing_text_nodes:
            # Layout parser already created nodes - just enhance them
            document = await self._enhance_text_nodes(document)
        else:
            # Need to extract text from raw content
            document = await self._extract_text_from_raw(document)
        
        # Post-process all text nodes
        document = await self._post_process_text(document)
        
        return document
    
    async def _enhance_text_nodes(self, document: IngestionDocument) -> IngestionDocument:
        """
        Enhance existing text nodes from layout parser.
        
        Args:
            document: Document with existing nodes
            
        Returns:
            Enhanced document
        """
        for node in document.get_text_nodes():
            # Clean up text
            if node.text:
                node.text = self._clean_text(node.text)
            
            # Mark as processed by text pipeline
            if node.source_pipeline != 'text_pipeline':
                node.metadata['original_source'] = node.source_pipeline
            node.source_pipeline = 'text_pipeline'
            node.status = ProcessingStatus.COMPLETED
        
        return document
    
    async def _extract_text_from_raw(self, document: IngestionDocument) -> IngestionDocument:
        """
        Extract text nodes from raw document content.
        
        Args:
            document: Document with raw_content
            
        Returns:
            Document with extracted text nodes
        """
        if not document.raw_content:
            document.add_error(
                self.stage_name,
                "No raw content available for text extraction"
            )
            return document
        
        # Decode content
        try:
            text = document.raw_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = document.raw_content.decode('latin-1')
            except Exception as e:
                document.add_error(
                    self.stage_name,
                    f"Failed to decode text content: {e}"
                )
                return document
        
        # Process based on MIME type
        mime_type = document.mime_type.lower()
        
        # Get current max sequence
        max_seq = max([n.sequence for n in document.nodes], default=-1)
        start_sequence = max_seq + 1
        
        if 'markdown' in mime_type or document.filename.endswith('.md'):
            nodes = self.markdown_processor.process(text, start_sequence)
        elif 'html' in mime_type:
            nodes = self.html_processor.process(text, start_sequence)
        else:
            nodes = self.plain_text_processor.process(text, start_sequence)
        
        # Add nodes to document
        document.add_nodes(nodes)
        
        return document
    
    async def _post_process_text(self, document: IngestionDocument) -> IngestionDocument:
        """
        Post-process all text nodes.
        
        Args:
            document: Document to post-process
            
        Returns:
            Post-processed document
        """
        for node in document.get_text_nodes():
            # Clean text
            if node.text:
                node.text = self._clean_text(node.text)
            
            # Skip empty nodes
            if not node.text or not node.text.strip():
                node.status = ProcessingStatus.SKIPPED
                continue
            
            # Mark as completed
            node.status = ProcessingStatus.COMPLETED
        
        # Calculate statistics
        text_nodes = [n for n in document.get_text_nodes() if n.status == ProcessingStatus.COMPLETED]
        total_chars = sum(len(n.text) for n in text_nodes)
        
        document.metadata['text_pipeline_stats'] = {
            'text_node_count': len(text_nodes),
            'total_characters': total_chars,
            'heading_count': len([n for n in text_nodes if n.node_type == NodeType.HEADING]),
        }
        
        return document
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters (except newlines)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def validate_input(self, document: IngestionDocument) -> bool:
        """Validate input document."""
        # Need either existing nodes or raw content
        return bool(document.nodes) or bool(document.raw_content)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "TextPipeline",
    "MarkdownProcessor",
    "HTMLProcessor",
    "PlainTextProcessor",
]
