"""
MODULE 7 — LAYOUT PARSING (WRAPPER, NOT DIY)
============================================
Thin wrapper around document layout parsing libraries.

MUST use:
- Docling OR Unstructured

Responsibilities:
- Parse PDF into layout blocks
- Preserve reading order
- Extract bounding boxes

DO NOT:
- Implement geometry heuristics
- Build custom layout algorithms

Output:
- Structured blocks for IR (IngestionDocument)
"""

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from schema.intermediate_representation import (
    BoundingBox,
    ContentNode,
    IngestionDocument,
    NodeType,
    ProcessingStatus,
)


# ============================================================================
# Layout Block Types
# ============================================================================

class LayoutBlockType(str, Enum):
    """Types of layout blocks from parsers."""
    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    CODE = "code"
    FORMULA = "formula"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


# ============================================================================
# Layout Block
# ============================================================================

@dataclass
class LayoutBlock:
    """
    A single block extracted from document layout.
    
    This is the intermediate format between the parser output
    and the IR ContentNode.
    """
    block_type: LayoutBlockType
    text: str = ""
    
    # Position
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1) absolute
    
    # Reading order
    sequence: int = 0
    
    # Heading level (for headings)
    level: Optional[int] = None
    
    # Table data
    table_data: Optional[List[List[str]]] = None
    table_html: Optional[str] = None
    
    # Image data
    image_data: Optional[bytes] = None
    image_format: Optional[str] = None
    
    # Metadata from parser
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_content_node(
        self,
        page_width: float = 1.0,
        page_height: float = 1.0
    ) -> ContentNode:
        """
        Convert to IR ContentNode.
        
        Args:
            page_width: Page width for bbox normalization
            page_height: Page height for bbox normalization
            
        Returns:
            ContentNode instance
        """
        # Map block type to node type
        type_mapping = {
            LayoutBlockType.TEXT: NodeType.TEXT,
            LayoutBlockType.TITLE: NodeType.HEADING,
            LayoutBlockType.HEADING: NodeType.HEADING,
            LayoutBlockType.LIST: NodeType.LIST,
            LayoutBlockType.LIST_ITEM: NodeType.LIST_ITEM,
            LayoutBlockType.TABLE: NodeType.TABLE,
            LayoutBlockType.FIGURE: NodeType.IMAGE,
            LayoutBlockType.IMAGE: NodeType.IMAGE,
            LayoutBlockType.CAPTION: NodeType.CAPTION,
            LayoutBlockType.HEADER: NodeType.HEADER,
            LayoutBlockType.FOOTER: NodeType.FOOTER,
            LayoutBlockType.FOOTNOTE: NodeType.FOOTNOTE,
            LayoutBlockType.CODE: NodeType.CODE,
            LayoutBlockType.FORMULA: NodeType.FORMULA,
        }
        
        node_type = type_mapping.get(self.block_type, NodeType.TEXT)
        
        # Create normalized bounding box
        bbox = None
        if self.bbox:
            x0, y0, x1, y1 = self.bbox
            bbox = BoundingBox.from_absolute(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                page_width=page_width,
                page_height=page_height,
                page=self.page
            )
        
        # Create node
        node = ContentNode(
            node_type=node_type,
            sequence=self.sequence,
            page=self.page,
            bbox=bbox,
            text=self.text,
            heading_level=self.level if self.block_type in (LayoutBlockType.HEADING, LayoutBlockType.TITLE) else None,
            table_data=self.table_data,
            table_html=self.table_html,
            image_data=self.image_data,
            image_format=self.image_format,
            confidence=self.confidence,
            source_pipeline="layout_parser",
            metadata=self.metadata,
        )
        
        return node


# ============================================================================
# Abstract Parser Interface
# ============================================================================

class LayoutParserBase(ABC):
    """Abstract base class for layout parsers."""
    
    @abstractmethod
    def parse(
        self,
        content: bytes,
        filename: str,
        mime_type: str
    ) -> List[LayoutBlock]:
        """
        Parse document content into layout blocks.
        
        Args:
            content: Raw document bytes
            filename: Original filename
            mime_type: Document MIME type
            
        Returns:
            List of LayoutBlock objects in reading order
        """
        pass
    
    @abstractmethod
    def get_page_dimensions(self) -> List[Tuple[float, float]]:
        """
        Get page dimensions from last parsed document.
        
        Returns:
            List of (width, height) tuples for each page
        """
        pass
    
    @abstractmethod
    def get_page_count(self) -> int:
        """Get page count from last parsed document."""
        pass


# ============================================================================
# Docling Parser Implementation
# ============================================================================

class DoclingParser(LayoutParserBase):
    """
    Layout parser using IBM Docling.
    
    Docling provides excellent PDF parsing with:
    - Layout analysis
    - Table extraction
    - Figure detection
    - Reading order
    """
    
    def __init__(self):
        """Initialize Docling parser."""
        self._page_dimensions: List[Tuple[float, float]] = []
        self._page_count: int = 0
        self._docling_available = self._check_docling()
    
    def _check_docling(self) -> bool:
        """Check if Docling is available."""
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False
    
    def parse(
        self,
        content: bytes,
        filename: str,
        mime_type: str
    ) -> List[LayoutBlock]:
        """Parse document using Docling."""
        if not self._docling_available:
            raise RuntimeError(
                "Docling is not installed. Install with: pip install docling"
            )
        
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        
        # Determine input format
        if mime_type == "application/pdf":
            input_format = InputFormat.PDF
        else:
            # Docling primarily handles PDFs
            raise ValueError(f"Docling does not support MIME type: {mime_type}")
        
        # Create temporary file-like object
        doc_stream = io.BytesIO(content)
        
        # Initialize converter
        converter = DocumentConverter()
        
        # Convert document
        result = converter.convert(doc_stream, input_format=input_format)
        
        # Extract page dimensions
        self._page_dimensions = []
        self._page_count = 0
        
        if hasattr(result, 'pages'):
            self._page_count = len(result.pages)
            for page in result.pages:
                width = getattr(page, 'width', 612)  # Default letter width
                height = getattr(page, 'height', 792)  # Default letter height
                self._page_dimensions.append((width, height))
        
        # Extract blocks
        blocks = []
        sequence = 0
        
        if hasattr(result, 'document') and hasattr(result.document, 'elements'):
            for element in result.document.elements:
                block = self._element_to_block(element, sequence)
                if block:
                    blocks.append(block)
                    sequence += 1
        
        return blocks
    
    def _element_to_block(self, element: Any, sequence: int) -> Optional[LayoutBlock]:
        """Convert Docling element to LayoutBlock."""
        # Get element type
        elem_type = getattr(element, 'type', 'text').lower()
        
        # Map Docling types to our types
        type_mapping = {
            'text': LayoutBlockType.TEXT,
            'title': LayoutBlockType.TITLE,
            'heading': LayoutBlockType.HEADING,
            'section-header': LayoutBlockType.HEADING,
            'list': LayoutBlockType.LIST,
            'list-item': LayoutBlockType.LIST_ITEM,
            'table': LayoutBlockType.TABLE,
            'figure': LayoutBlockType.FIGURE,
            'picture': LayoutBlockType.IMAGE,
            'caption': LayoutBlockType.CAPTION,
            'header': LayoutBlockType.HEADER,
            'footer': LayoutBlockType.FOOTER,
            'footnote': LayoutBlockType.FOOTNOTE,
            'code': LayoutBlockType.CODE,
            'formula': LayoutBlockType.FORMULA,
        }
        
        block_type = type_mapping.get(elem_type, LayoutBlockType.TEXT)
        
        # Extract text
        text = getattr(element, 'text', '') or ''
        
        # Extract bbox
        bbox = None
        if hasattr(element, 'bbox'):
            b = element.bbox
            bbox = (b.x0, b.y0, b.x1, b.y1)
        elif hasattr(element, 'bounding_box'):
            b = element.bounding_box
            bbox = (b.x0, b.y0, b.x1, b.y1)
        
        # Extract page
        page = getattr(element, 'page', 0) or 0
        
        # Extract level for headings
        level = None
        if block_type in (LayoutBlockType.HEADING, LayoutBlockType.TITLE):
            level = getattr(element, 'level', 1) or 1
        
        # Extract table data
        table_data = None
        table_html = None
        if block_type == LayoutBlockType.TABLE and hasattr(element, 'table'):
            table = element.table
            if hasattr(table, 'data'):
                table_data = table.data
            if hasattr(table, 'to_html'):
                table_html = table.to_html()
        
        # Extract image data
        image_data = None
        image_format = None
        if block_type in (LayoutBlockType.FIGURE, LayoutBlockType.IMAGE):
            if hasattr(element, 'image_data'):
                image_data = element.image_data
            if hasattr(element, 'image_format'):
                image_format = element.image_format
        
        return LayoutBlock(
            block_type=block_type,
            text=text,
            page=page,
            bbox=bbox,
            sequence=sequence,
            level=level,
            table_data=table_data,
            table_html=table_html,
            image_data=image_data,
            image_format=image_format,
            confidence=getattr(element, 'confidence', 1.0),
            metadata={
                'original_type': elem_type,
                'parser': 'docling'
            }
        )
    
    def get_page_dimensions(self) -> List[Tuple[float, float]]:
        """Get page dimensions."""
        return self._page_dimensions
    
    def get_page_count(self) -> int:
        """Get page count."""
        return self._page_count


# ============================================================================
# Unstructured Parser Implementation
# ============================================================================

class UnstructuredParser(LayoutParserBase):
    """
    Layout parser using Unstructured.
    
    Unstructured provides:
    - Multi-format support
    - Element classification
    - Layout detection
    """
    
    def __init__(self):
        """Initialize Unstructured parser."""
        self._page_dimensions: List[Tuple[float, float]] = []
        self._page_count: int = 0
        self._unstructured_available = self._check_unstructured()
    
    def _check_unstructured(self) -> bool:
        """Check if Unstructured is available."""
        try:
            from unstructured.partition.auto import partition
            return True
        except ImportError:
            return False
    
    def parse(
        self,
        content: bytes,
        filename: str,
        mime_type: str
    ) -> List[LayoutBlock]:
        """Parse document using Unstructured."""
        if not self._unstructured_available:
            raise RuntimeError(
                "Unstructured is not installed. Install with: pip install unstructured"
            )
        
        from unstructured.partition.auto import partition
        
        # Create temporary file-like object
        doc_stream = io.BytesIO(content)
        
        # Partition document
        elements = partition(
            file=doc_stream,
            metadata_filename=filename,
            content_type=mime_type,
            strategy="hi_res",  # Use high-resolution strategy for better layout
        )
        
        # Track pages
        pages_seen = set()
        
        # Convert elements to blocks
        blocks = []
        for i, element in enumerate(elements):
            block = self._element_to_block(element, i)
            if block:
                blocks.append(block)
                pages_seen.add(block.page)
        
        # Set page count
        self._page_count = max(pages_seen) + 1 if pages_seen else 1
        
        # Default page dimensions (Unstructured doesn't always provide these)
        self._page_dimensions = [(612, 792)] * self._page_count
        
        return blocks
    
    def _element_to_block(self, element: Any, sequence: int) -> Optional[LayoutBlock]:
        """Convert Unstructured element to LayoutBlock."""
        # Get element category
        category = element.category if hasattr(element, 'category') else 'NarrativeText'
        
        # Map Unstructured categories to our types
        type_mapping = {
            'Title': LayoutBlockType.TITLE,
            'NarrativeText': LayoutBlockType.TEXT,
            'Text': LayoutBlockType.TEXT,
            'UncategorizedText': LayoutBlockType.TEXT,
            'ListItem': LayoutBlockType.LIST_ITEM,
            'BulletedText': LayoutBlockType.LIST_ITEM,
            'Table': LayoutBlockType.TABLE,
            'Image': LayoutBlockType.IMAGE,
            'Figure': LayoutBlockType.FIGURE,
            'FigureCaption': LayoutBlockType.CAPTION,
            'Caption': LayoutBlockType.CAPTION,
            'Header': LayoutBlockType.HEADER,
            'Footer': LayoutBlockType.FOOTER,
            'Footnote': LayoutBlockType.FOOTNOTE,
            'CodeSnippet': LayoutBlockType.CODE,
            'Formula': LayoutBlockType.FORMULA,
            'PageBreak': LayoutBlockType.PAGE_BREAK,
        }
        
        block_type = type_mapping.get(category, LayoutBlockType.TEXT)
        
        # Skip page breaks
        if block_type == LayoutBlockType.PAGE_BREAK:
            return None
        
        # Extract text
        text = str(element) if element else ''
        
        # Extract metadata
        metadata = element.metadata if hasattr(element, 'metadata') else {}
        
        # Extract bbox
        bbox = None
        if hasattr(metadata, 'coordinates') and metadata.coordinates:
            coords = metadata.coordinates
            if hasattr(coords, 'points') and coords.points:
                points = coords.points
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Extract page
        page = 0
        if hasattr(metadata, 'page_number'):
            page = (metadata.page_number or 1) - 1  # Convert to 0-indexed
        
        # Extract heading level
        level = None
        if block_type == LayoutBlockType.TITLE:
            level = 1
        elif hasattr(metadata, 'header_level'):
            level = metadata.header_level
        
        # Extract table data
        table_data = None
        table_html = None
        if block_type == LayoutBlockType.TABLE:
            if hasattr(element, 'text_as_html'):
                table_html = element.text_as_html
            # Try to parse table structure
            if hasattr(metadata, 'table_data'):
                table_data = metadata.table_data
        
        return LayoutBlock(
            block_type=block_type,
            text=text,
            page=page,
            bbox=bbox,
            sequence=sequence,
            level=level,
            table_data=table_data,
            table_html=table_html,
            metadata={
                'original_category': category,
                'parser': 'unstructured'
            }
        )
    
    def get_page_dimensions(self) -> List[Tuple[float, float]]:
        """Get page dimensions."""
        return self._page_dimensions
    
    def get_page_count(self) -> int:
        """Get page count."""
        return self._page_count


# ============================================================================
# Fallback Simple Parser (for testing/development)
# ============================================================================

class SimpleTextParser(LayoutParserBase):
    """
    Simple fallback parser for plain text documents.
    
    This is used when no layout parsing is needed or as a fallback.
    """
    
    def __init__(self):
        """Initialize simple parser."""
        self._page_dimensions: List[Tuple[float, float]] = [(612, 792)]
        self._page_count: int = 1
    
    def parse(
        self,
        content: bytes,
        filename: str,
        mime_type: str
    ) -> List[LayoutBlock]:
        """Parse text content into blocks."""
        # Decode content
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        blocks = []
        sequence = 0
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detect headings (lines starting with #)
            if para.startswith('#'):
                level = 0
                while para.startswith('#'):
                    level += 1
                    para = para[1:]
                para = para.strip()
                
                blocks.append(LayoutBlock(
                    block_type=LayoutBlockType.HEADING,
                    text=para,
                    page=0,
                    sequence=sequence,
                    level=min(level, 6)
                ))
            else:
                blocks.append(LayoutBlock(
                    block_type=LayoutBlockType.TEXT,
                    text=para,
                    page=0,
                    sequence=sequence
                ))
            
            sequence += 1
        
        self._page_count = 1
        return blocks
    
    def get_page_dimensions(self) -> List[Tuple[float, float]]:
        """Get page dimensions."""
        return self._page_dimensions
    
    def get_page_count(self) -> int:
        """Get page count."""
        return self._page_count


# ============================================================================
# Layout Parser Factory
# ============================================================================

class LayoutParser:
    """
    Main layout parser that delegates to appropriate backend.
    
    Automatically selects between Docling, Unstructured, or fallback
    based on availability and document type.
    """
    
    def __init__(self, preferred_backend: Optional[str] = None):
        """
        Initialize layout parser.
        
        Args:
            preferred_backend: Preferred backend ('docling', 'unstructured', or None for auto)
        """
        self.preferred_backend = preferred_backend
        self._parser: Optional[LayoutParserBase] = None
        self._backend_name: str = ""
    
    def _get_parser(self, mime_type: str) -> LayoutParserBase:
        """Get appropriate parser for document type."""
        # For PDFs, prefer Docling
        if mime_type == "application/pdf":
            if self.preferred_backend == "unstructured":
                parser = UnstructuredParser()
                if parser._unstructured_available:
                    self._backend_name = "unstructured"
                    return parser
            
            # Try Docling first
            parser = DoclingParser()
            if parser._docling_available:
                self._backend_name = "docling"
                return parser
            
            # Fall back to Unstructured
            parser = UnstructuredParser()
            if parser._unstructured_available:
                self._backend_name = "unstructured"
                return parser
            
            raise RuntimeError(
                "No PDF parser available. Install docling or unstructured."
            )
        
        # For text/markdown/html, use simple parser
        if mime_type.startswith("text/"):
            self._backend_name = "simple"
            return SimpleTextParser()
        
        # Try Unstructured for other formats
        parser = UnstructuredParser()
        if parser._unstructured_available:
            self._backend_name = "unstructured"
            return parser
        
        # Fallback to simple parser
        self._backend_name = "simple"
        return SimpleTextParser()
    
    def parse(
        self,
        content: bytes,
        filename: str,
        mime_type: str
    ) -> Tuple[List[LayoutBlock], str]:
        """
        Parse document into layout blocks.
        
        Args:
            content: Raw document bytes
            filename: Original filename
            mime_type: Document MIME type
            
        Returns:
            Tuple of (blocks, backend_name)
        """
        self._parser = self._get_parser(mime_type)
        blocks = self._parser.parse(content, filename, mime_type)
        return blocks, self._backend_name
    
    def get_page_dimensions(self) -> List[Tuple[float, float]]:
        """Get page dimensions from last parsed document."""
        if self._parser:
            return self._parser.get_page_dimensions()
        return [(612, 792)]
    
    def get_page_count(self) -> int:
        """Get page count from last parsed document."""
        if self._parser:
            return self._parser.get_page_count()
        return 1
    
    def parse_to_ir(
        self,
        document: IngestionDocument
    ) -> IngestionDocument:
        """
        Parse document and populate IR with nodes.
        
        This is the main entry point for the layout parsing pipeline stage.
        
        Args:
            document: IngestionDocument with raw_content
            
        Returns:
            Updated IngestionDocument with nodes populated
        """
        if not document.raw_content:
            document.add_error(
                "layout_parser",
                "No raw content available for parsing"
            )
            return document
        
        try:
            # Parse content
            blocks, backend = self.parse(
                content=document.raw_content,
                filename=document.filename,
                mime_type=document.mime_type
            )
            
            # Get page info
            page_dims = self.get_page_dimensions()
            page_count = self.get_page_count()
            
            # Update document metadata
            document.page_count = page_count
            document.page_dimensions = page_dims
            
            # Convert blocks to nodes
            for block in blocks:
                # Get page dimensions for normalization
                page = block.page
                if page < len(page_dims):
                    page_width, page_height = page_dims[page]
                else:
                    page_width, page_height = 612, 792
                
                # Convert to node
                node = block.to_content_node(
                    page_width=page_width,
                    page_height=page_height
                )
                
                document.add_node(node)
            
            # Add metadata
            document.metadata["layout_parser_backend"] = backend
            document.metadata["layout_block_count"] = len(blocks)
            
        except Exception as e:
            document.add_error(
                "layout_parser",
                str(e),
                {"exception_type": type(e).__name__}
            )
        
        return document


# ============================================================================
# Factory Function
# ============================================================================

def get_layout_parser(preferred_backend: Optional[str] = None) -> LayoutParser:
    """Get a layout parser instance."""
    return LayoutParser(preferred_backend=preferred_backend)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "LayoutBlockType",
    "LayoutBlock",
    "LayoutParserBase",
    "DoclingParser",
    "UnstructuredParser",
    "SimpleTextParser",
    "LayoutParser",
    "get_layout_parser",
]
