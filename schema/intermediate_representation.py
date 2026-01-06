"""
MODULE 11 — INTERMEDIATE REPRESENTATION (THE CONTRACT)
=====================================================
The core data contract for the Enterprise RAG Ingestion Pipeline.

This module is LOCKED — no redesign later.

Key Properties:
- Flat node list (no deep nesting)
- Graph relationships via IDs
- Bounding boxes normalized (0-1)
- RBAC carried end-to-end

The IR is the single source of truth passed between all pipeline stages.
All pipelines ONLY mutate IngestionDocument - they never call each other.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ============================================================================
# Enums
# ============================================================================

class NodeType(str, Enum):
    """Types of content nodes in the IR."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    HEADING = "heading"
    LIST = "list"
    LIST_ITEM = "list_item"
    CODE = "code"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    HEADER = "header"  # Page header
    FOOTER = "footer"  # Page footer
    CAPTION = "caption"


class HeadingLevel(int, Enum):
    """Heading levels."""
    H1 = 1
    H2 = 2
    H3 = 3
    H4 = 4
    H5 = 5
    H6 = 6


class ProcessingStatus(str, Enum):
    """Processing status for nodes and documents."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# Bounding Box
# ============================================================================

@dataclass
class BoundingBox:
    """
    Normalized bounding box (0-1 coordinates).
    
    Origin is top-left corner.
    All values are normalized to page dimensions.
    """
    x0: float  # Left edge (0-1)
    y0: float  # Top edge (0-1)
    x1: float  # Right edge (0-1)
    y1: float  # Bottom edge (0-1)
    page: int = 0  # Page number (0-indexed)
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        # Clamp values to 0-1 range
        self.x0 = max(0.0, min(1.0, self.x0))
        self.y0 = max(0.0, min(1.0, self.y0))
        self.x1 = max(0.0, min(1.0, self.x1))
        self.y1 = max(0.0, min(1.0, self.y1))
        
        # Ensure x0 <= x1 and y0 <= y1
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.y1 - self.y0
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def contains(self, other: "BoundingBox") -> bool:
        """Check if this bbox contains another bbox."""
        if self.page != other.page:
            return False
        return (
            self.x0 <= other.x0 and
            self.y0 <= other.y0 and
            self.x1 >= other.x1 and
            self.y1 >= other.y1
        )
    
    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bbox overlaps with another."""
        if self.page != other.page:
            return False
        return not (
            self.x1 < other.x0 or
            self.x0 > other.x1 or
            self.y1 < other.y0 or
            self.y0 > other.y1
        )
    
    def overlap_area(self, other: "BoundingBox") -> float:
        """Calculate overlap area with another bbox."""
        if not self.overlaps(other):
            return 0.0
        
        x_overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        y_overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        
        return max(0.0, x_overlap) * max(0.0, y_overlap)
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bbox."""
        if self.page != other.page:
            return 0.0
        
        intersection = self.overlap_area(other)
        union = self.area + other.area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        """Create from dictionary."""
        return cls(
            x0=data["x0"],
            y0=data["y0"],
            x1=data["x1"],
            y1=data["y1"],
            page=data.get("page", 0)
        )
    
    @classmethod
    def from_absolute(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        page_width: float,
        page_height: float,
        page: int = 0
    ) -> "BoundingBox":
        """Create normalized bbox from absolute coordinates."""
        return cls(
            x0=x0 / page_width if page_width > 0 else 0,
            y0=y0 / page_height if page_height > 0 else 0,
            x1=x1 / page_width if page_width > 0 else 0,
            y1=y1 / page_height if page_height > 0 else 0,
            page=page
        )


# ============================================================================
# Content Node
# ============================================================================

@dataclass
class ContentNode:
    """
    A single content element extracted from a document.
    
    Nodes are the atomic units of the IR. They represent:
    - Text blocks (paragraphs, headings)
    - Images
    - Tables
    - Other structural elements
    
    All relationships are expressed via IDs (flat graph structure).
    """
    # Identification
    id: str = field(default_factory=lambda: f"node-{uuid.uuid4().hex[:12]}")
    node_type: NodeType = NodeType.TEXT
    
    # Position in document
    sequence: int = 0  # Reading order position
    page: int = 0  # Page number (0-indexed)
    bbox: Optional[BoundingBox] = None
    
    # Content
    text: str = ""  # Text content (for text nodes)
    
    # Image-specific fields
    image_path: Optional[str] = None  # S3 path to image
    image_data: Optional[bytes] = None  # Raw image bytes (temporary, not persisted)
    image_caption: Optional[str] = None  # Generated caption
    ocr_text: Optional[str] = None  # OCR extracted text
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None  # png, jpeg, etc.
    
    # Table-specific fields
    table_data: Optional[List[List[str]]] = None  # 2D table data
    table_html: Optional[str] = None  # HTML representation
    
    # Heading-specific fields
    heading_level: Optional[int] = None  # 1-6
    
    # Relationships (graph via IDs)
    parent_id: Optional[str] = None  # Parent node ID
    children_ids: List[str] = field(default_factory=list)  # Child node IDs
    linked_ids: List[str] = field(default_factory=list)  # Related node IDs (e.g., image linked to text)
    
    # Processing metadata
    status: ProcessingStatus = ProcessingStatus.PENDING
    source_pipeline: Optional[str] = None  # Which pipeline created this node
    confidence: float = 1.0  # Extraction confidence (0-1)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure proper initialization."""
        if self.children_ids is None:
            self.children_ids = []
        if self.linked_ids is None:
            self.linked_ids = []
        if self.metadata is None:
            self.metadata = {}
    
    def add_link(self, node_id: str) -> None:
        """Add a link to another node."""
        if node_id not in self.linked_ids:
            self.linked_ids.append(node_id)
    
    def remove_link(self, node_id: str) -> None:
        """Remove a link to another node."""
        if node_id in self.linked_ids:
            self.linked_ids.remove(node_id)
    
    def add_child(self, node_id: str) -> None:
        """Add a child node."""
        if node_id not in self.children_ids:
            self.children_ids.append(node_id)
    
    def get_full_text(self) -> str:
        """Get all text content from this node."""
        parts = []
        if self.text:
            parts.append(self.text)
        if self.ocr_text:
            parts.append(self.ocr_text)
        if self.image_caption:
            parts.append(f"[Image: {self.image_caption}]")
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "sequence": self.sequence,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "text": self.text,
            "image_path": self.image_path,
            "image_caption": self.image_caption,
            "ocr_text": self.ocr_text,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_format": self.image_format,
            "table_data": self.table_data,
            "table_html": self.table_html,
            "heading_level": self.heading_level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "linked_ids": self.linked_ids,
            "status": self.status.value,
            "source_pipeline": self.source_pipeline,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentNode":
        """Create from dictionary."""
        return cls(
            id=data.get("id", f"node-{uuid.uuid4().hex[:12]}"),
            node_type=NodeType(data.get("node_type", "text")),
            sequence=data.get("sequence", 0),
            page=data.get("page", 0),
            bbox=BoundingBox.from_dict(data["bbox"]) if data.get("bbox") else None,
            text=data.get("text", ""),
            image_path=data.get("image_path"),
            image_caption=data.get("image_caption"),
            ocr_text=data.get("ocr_text"),
            image_width=data.get("image_width"),
            image_height=data.get("image_height"),
            image_format=data.get("image_format"),
            table_data=data.get("table_data"),
            table_html=data.get("table_html"),
            heading_level=data.get("heading_level"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            linked_ids=data.get("linked_ids", []),
            status=ProcessingStatus(data.get("status", "pending")),
            source_pipeline=data.get("source_pipeline"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Chunk (for retrieval)
# ============================================================================

@dataclass
class Chunk:
    """
    A chunk of content ready for embedding and retrieval.
    
    Chunks are created from nodes during the chunking stage.
    They include text content and references to supporting images.
    """
    # Identification
    id: str = field(default_factory=lambda: f"chunk-{uuid.uuid4().hex[:12]}")
    index: int = 0  # Chunk sequence index
    
    # Content
    text: str = ""
    token_count: int = 0
    
    # Source tracking
    source_node_ids: List[str] = field(default_factory=list)
    
    # Image references (supporting evidence)
    image_refs: List[Dict[str, Any]] = field(default_factory=list)
    # Each image_ref: {"node_id": str, "image_path": str, "caption": str}
    
    # Location
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    
    # Content hash for deduplication
    content_hash: str = ""
    
    # Embedding (populated by embedding service)
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # ACL metadata (denormalized from document)
    acl: Dict[str, Any] = field(default_factory=dict)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "token_count": self.token_count,
            "source_node_ids": self.source_node_ids,
            "image_refs": self.image_refs,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "content_hash": self.content_hash,
            "embedding_model": self.embedding_model,
            "acl": self.acl,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(
            id=data.get("id", f"chunk-{uuid.uuid4().hex[:12]}"),
            index=data.get("index", 0),
            text=data.get("text", ""),
            token_count=data.get("token_count", 0),
            source_node_ids=data.get("source_node_ids", []),
            image_refs=data.get("image_refs", []),
            start_page=data.get("start_page"),
            end_page=data.get("end_page"),
            content_hash=data.get("content_hash", ""),
            embedding_model=data.get("embedding_model"),
            acl=data.get("acl", {}),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# RBAC Context
# ============================================================================

@dataclass
class RBACContext:
    """
    RBAC context carried through the pipeline.
    
    This ensures access control is enforced end-to-end.
    """
    tenant_id: str
    allowed_roles: List[str] = field(default_factory=list)
    allowed_users: List[str] = field(default_factory=list)
    classification: str = "INTERNAL"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ACL embedding."""
        return {
            "tenant_id": self.tenant_id,
            "allowed_roles": self.allowed_roles,
            "allowed_users": self.allowed_users,
            "classification": self.classification,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RBACContext":
        """Create from dictionary."""
        return cls(
            tenant_id=data.get("tenant_id", ""),
            allowed_roles=data.get("allowed_roles", []),
            allowed_users=data.get("allowed_users", []),
            classification=data.get("classification", "INTERNAL"),
        )


# ============================================================================
# Ingestion Document (THE IR)
# ============================================================================

@dataclass
class IngestionDocument:
    """
    The Intermediate Representation for document ingestion.
    
    This is THE CONTRACT - all pipelines read from and write to this object.
    
    Key Design Principles:
    1. Flat node list - no deep nesting
    2. Graph relationships via IDs
    3. Bounding boxes normalized (0-1)
    4. RBAC carried end-to-end
    5. Immutable identification, mutable content
    """
    # Identification (immutable after creation)
    document_id: str
    tenant_id: str
    version: int = 1
    
    # Source information
    filename: str = ""
    mime_type: str = ""
    file_size_bytes: int = 0
    content_hash: str = ""
    
    # Raw content (may be cleared after processing)
    raw_content: Optional[bytes] = None
    
    # S3 paths
    s3_raw_path: str = ""
    s3_derived_prefix: str = ""
    
    # Page information (for multi-page documents)
    page_count: int = 1
    page_dimensions: List[Tuple[float, float]] = field(default_factory=list)  # [(width, height), ...]
    
    # Content nodes (flat list)
    nodes: List[ContentNode] = field(default_factory=list)
    
    # Node index for quick lookup
    _node_index: Dict[str, ContentNode] = field(default_factory=dict, repr=False)
    
    # Chunks (populated by chunking pipeline)
    chunks: List[Chunk] = field(default_factory=list)
    
    # RBAC context
    rbac: RBACContext = field(default_factory=lambda: RBACContext(tenant_id=""))
    
    # Enriched metadata
    enriched_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing state
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_stage: Optional[str] = None
    completed_stages: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing and metrics
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize node index."""
        self._rebuild_node_index()
        if not self.rbac.tenant_id:
            self.rbac = RBACContext(tenant_id=self.tenant_id)
    
    # =========================================================================
    # Node Management
    # =========================================================================
    
    def _rebuild_node_index(self) -> None:
        """Rebuild the node index from the nodes list."""
        self._node_index = {node.id: node for node in self.nodes}
    
    def add_node(self, node: ContentNode) -> None:
        """Add a node to the document."""
        self.nodes.append(node)
        self._node_index[node.id] = node
    
    def add_nodes(self, nodes: List[ContentNode]) -> None:
        """Add multiple nodes to the document."""
        for node in nodes:
            self.add_node(node)
    
    def get_node(self, node_id: str) -> Optional[ContentNode]:
        """Get a node by ID."""
        return self._node_index.get(node_id)
    
    def remove_node(self, node_id: str) -> Optional[ContentNode]:
        """Remove a node by ID."""
        node = self._node_index.pop(node_id, None)
        if node:
            self.nodes = [n for n in self.nodes if n.id != node_id]
        return node
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ContentNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.node_type == node_type]
    
    def get_nodes_by_page(self, page: int) -> List[ContentNode]:
        """Get all nodes on a specific page."""
        return [n for n in self.nodes if n.page == page]
    
    def get_text_nodes(self) -> List[ContentNode]:
        """Get all text-containing nodes."""
        text_types = {NodeType.TEXT, NodeType.HEADING, NodeType.LIST_ITEM, NodeType.CODE}
        return [n for n in self.nodes if n.node_type in text_types]
    
    def get_image_nodes(self) -> List[ContentNode]:
        """Get all image nodes."""
        return self.get_nodes_by_type(NodeType.IMAGE)
    
    def get_nodes_in_reading_order(self) -> List[ContentNode]:
        """Get nodes sorted by reading order."""
        return sorted(self.nodes, key=lambda n: (n.page, n.sequence))
    
    def get_linked_nodes(self, node_id: str) -> List[ContentNode]:
        """Get all nodes linked to a specific node."""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.get_node(lid) for lid in node.linked_ids if self.get_node(lid)]
    
    # =========================================================================
    # Stage Management
    # =========================================================================
    
    def start_stage(self, stage: str) -> None:
        """Mark a stage as started."""
        self.current_stage = stage
        self.status = ProcessingStatus.PROCESSING
        self.stage_metrics[stage] = {
            "started_at": datetime.utcnow().isoformat()
        }
    
    def complete_stage(self, stage: str, metrics: Optional[Dict] = None) -> None:
        """Mark a stage as completed."""
        if stage not in self.completed_stages:
            self.completed_stages.append(stage)
        self.current_stage = None
        
        if stage in self.stage_metrics:
            self.stage_metrics[stage]["completed_at"] = datetime.utcnow().isoformat()
            if metrics:
                self.stage_metrics[stage]["metrics"] = metrics
    
    def add_error(self, stage: str, error: str, details: Optional[Dict] = None) -> None:
        """Add an error to the document."""
        self.errors.append({
            "stage": stage,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        })
    
    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed."""
        return stage in self.completed_stages
    
    # =========================================================================
    # Chunk Management
    # =========================================================================
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the document."""
        # Inherit ACL from document
        chunk.acl = self.rbac.to_dict()
        self.chunks.append(chunk)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks to the document."""
        for chunk in chunks:
            self.add_chunk(chunk)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document statistics."""
        node_type_counts = {}
        for node in self.nodes:
            t = node.node_type.value
            node_type_counts[t] = node_type_counts.get(t, 0) + 1
        
        total_text_length = sum(len(n.text) for n in self.nodes if n.text)
        
        return {
            "document_id": self.document_id,
            "page_count": self.page_count,
            "node_count": len(self.nodes),
            "node_type_counts": node_type_counts,
            "chunk_count": len(self.chunks),
            "total_text_length": total_text_length,
            "completed_stages": self.completed_stages,
            "error_count": len(self.errors),
        }
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "version": self.version,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "file_size_bytes": self.file_size_bytes,
            "content_hash": self.content_hash,
            "s3_raw_path": self.s3_raw_path,
            "s3_derived_prefix": self.s3_derived_prefix,
            "page_count": self.page_count,
            "page_dimensions": self.page_dimensions,
            "nodes": [n.to_dict() for n in self.nodes],
            "chunks": [c.to_dict() for c in self.chunks],
            "rbac": self.rbac.to_dict(),
            "enriched_metadata": self.enriched_metadata,
            "status": self.status.value,
            "completed_stages": self.completed_stages,
            "errors": self.errors,
            "created_at": self.created_at.isoformat(),
            "stage_metrics": self.stage_metrics,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionDocument":
        """Create from dictionary."""
        doc = cls(
            document_id=data["document_id"],
            tenant_id=data["tenant_id"],
            version=data.get("version", 1),
            filename=data.get("filename", ""),
            mime_type=data.get("mime_type", ""),
            file_size_bytes=data.get("file_size_bytes", 0),
            content_hash=data.get("content_hash", ""),
            s3_raw_path=data.get("s3_raw_path", ""),
            s3_derived_prefix=data.get("s3_derived_prefix", ""),
            page_count=data.get("page_count", 1),
            page_dimensions=data.get("page_dimensions", []),
            rbac=RBACContext.from_dict(data.get("rbac", {})),
            enriched_metadata=data.get("enriched_metadata", {}),
            status=ProcessingStatus(data.get("status", "pending")),
            completed_stages=data.get("completed_stages", []),
            errors=data.get("errors", []),
            stage_metrics=data.get("stage_metrics", {}),
            metadata=data.get("metadata", {}),
        )
        
        # Load nodes
        for node_data in data.get("nodes", []):
            doc.add_node(ContentNode.from_dict(node_data))
        
        # Load chunks
        for chunk_data in data.get("chunks", []):
            doc.chunks.append(Chunk.from_dict(chunk_data))
        
        return doc


# ============================================================================
# Factory Functions
# ============================================================================

def create_ingestion_document(
    document_id: str,
    tenant_id: str,
    filename: str,
    mime_type: str,
    content: Optional[bytes] = None,
    s3_raw_path: str = "",
    rbac: Optional[RBACContext] = None,
    version: int = 1,
) -> IngestionDocument:
    """
    Factory function to create an IngestionDocument.
    
    Args:
        document_id: External document identifier
        tenant_id: Tenant identifier
        filename: Original filename
        mime_type: Document MIME type
        content: Raw document content (optional)
        s3_raw_path: S3 path to raw document
        rbac: RBAC context
        version: Document version
        
    Returns:
        Initialized IngestionDocument
    """
    import hashlib
    
    content_hash = ""
    file_size = 0
    
    if content:
        content_hash = hashlib.sha256(content).hexdigest()
        file_size = len(content)
    
    return IngestionDocument(
        document_id=document_id,
        tenant_id=tenant_id,
        version=version,
        filename=filename,
        mime_type=mime_type,
        file_size_bytes=file_size,
        content_hash=content_hash,
        raw_content=content,
        s3_raw_path=s3_raw_path,
        rbac=rbac or RBACContext(tenant_id=tenant_id),
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "NodeType",
    "HeadingLevel",
    "ProcessingStatus",
    
    # Data classes
    "BoundingBox",
    "ContentNode",
    "Chunk",
    "RBACContext",
    "IngestionDocument",
    
    # Factory functions
    "create_ingestion_document",
]
