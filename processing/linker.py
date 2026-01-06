"""
MODULE 12 — LINKAGE STRATEGY (V1-SAFE)
======================================
Node linkage logic for the Enterprise RAG Ingestion Pipeline.

Rules (V1):
1. If image bbox is spatially inside text block → link
2. Else → link to immediately preceding text in reading order

DO NOT:
- Use semantic similarity
- Over-engineer

This module creates relationships between image nodes and text nodes
to enable multi-modal retrieval where images are linked to their
contextual text.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pipelines.base import BasePipeline
from schema.intermediate_representation import (
    BoundingBox,
    ContentNode,
    IngestionDocument,
    NodeType,
    ProcessingStatus,
)
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Linkage Result
# ============================================================================

@dataclass
class LinkageResult:
    """Result of linkage between two nodes."""
    image_node_id: str
    text_node_id: str
    linkage_type: str  # "spatial_containment" or "reading_order"
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Spatial Analysis
# ============================================================================

class SpatialAnalyzer:
    """
    Analyzes spatial relationships between nodes.
    
    Uses bounding box geometry to determine containment
    and proximity relationships.
    """
    
    def __init__(self, containment_threshold: float = 0.7):
        """
        Initialize spatial analyzer.
        
        Args:
            containment_threshold: Minimum overlap ratio for containment (0-1)
        """
        self.containment_threshold = containment_threshold
    
    def is_contained_in(
        self,
        inner: BoundingBox,
        outer: BoundingBox
    ) -> Tuple[bool, float]:
        """
        Check if inner bbox is contained within outer bbox.
        
        Args:
            inner: Potential inner bounding box
            outer: Potential outer bounding box
            
        Returns:
            Tuple of (is_contained, overlap_ratio)
        """
        if inner.page != outer.page:
            return False, 0.0
        
        # Calculate overlap area
        overlap = inner.overlap_area(outer)
        
        # Calculate containment ratio (how much of inner is inside outer)
        inner_area = inner.area
        if inner_area == 0:
            return False, 0.0
        
        containment_ratio = overlap / inner_area
        
        is_contained = containment_ratio >= self.containment_threshold
        
        return is_contained, containment_ratio
    
    def find_containing_node(
        self,
        target: ContentNode,
        candidates: List[ContentNode]
    ) -> Optional[Tuple[ContentNode, float]]:
        """
        Find a node that contains the target node spatially.
        
        Args:
            target: Node to find container for
            candidates: List of potential container nodes
            
        Returns:
            Tuple of (containing_node, containment_ratio) or None
        """
        if not target.bbox:
            return None
        
        best_match = None
        best_ratio = 0.0
        
        for candidate in candidates:
            if not candidate.bbox:
                continue
            
            is_contained, ratio = self.is_contained_in(target.bbox, candidate.bbox)
            
            if is_contained and ratio > best_ratio:
                best_match = candidate
                best_ratio = ratio
        
        if best_match:
            return best_match, best_ratio
        
        return None
    
    def calculate_distance(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox
    ) -> float:
        """
        Calculate distance between two bounding boxes.
        
        Uses center-to-center Euclidean distance.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Distance (normalized 0-1 scale)
        """
        if bbox1.page != bbox2.page:
            return float('inf')
        
        c1 = bbox1.center
        c2 = bbox2.center
        
        distance = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
        
        return distance
    
    def find_nearest_node(
        self,
        target: ContentNode,
        candidates: List[ContentNode]
    ) -> Optional[Tuple[ContentNode, float]]:
        """
        Find the nearest node to target by spatial distance.
        
        Args:
            target: Node to find nearest neighbor for
            candidates: List of candidate nodes
            
        Returns:
            Tuple of (nearest_node, distance) or None
        """
        if not target.bbox:
            return None
        
        nearest = None
        min_distance = float('inf')
        
        for candidate in candidates:
            if not candidate.bbox:
                continue
            
            distance = self.calculate_distance(target.bbox, candidate.bbox)
            
            if distance < min_distance:
                nearest = candidate
                min_distance = distance
        
        if nearest:
            return nearest, min_distance
        
        return None


# ============================================================================
# Reading Order Analyzer
# ============================================================================

class ReadingOrderAnalyzer:
    """
    Analyzes reading order relationships between nodes.
    
    Uses sequence numbers and page positions to determine
    preceding/following relationships.
    """
    
    def find_preceding_text(
        self,
        target: ContentNode,
        text_nodes: List[ContentNode]
    ) -> Optional[ContentNode]:
        """
        Find the text node immediately preceding target in reading order.
        
        Args:
            target: Target node
            text_nodes: List of text nodes sorted by reading order
            
        Returns:
            Preceding text node or None
        """
        # Filter to same page first, then all pages
        same_page_nodes = [
            n for n in text_nodes 
            if n.page == target.page and n.sequence < target.sequence
        ]
        
        if same_page_nodes:
            # Return the closest preceding node on same page
            return max(same_page_nodes, key=lambda n: n.sequence)
        
        # Fall back to any preceding node
        preceding_nodes = [
            n for n in text_nodes
            if (n.page < target.page) or 
               (n.page == target.page and n.sequence < target.sequence)
        ]
        
        if preceding_nodes:
            return max(preceding_nodes, key=lambda n: (n.page, n.sequence))
        
        return None
    
    def find_following_text(
        self,
        target: ContentNode,
        text_nodes: List[ContentNode]
    ) -> Optional[ContentNode]:
        """
        Find the text node immediately following target in reading order.
        
        Args:
            target: Target node
            text_nodes: List of text nodes sorted by reading order
            
        Returns:
            Following text node or None
        """
        # Filter to same page first
        same_page_nodes = [
            n for n in text_nodes
            if n.page == target.page and n.sequence > target.sequence
        ]
        
        if same_page_nodes:
            return min(same_page_nodes, key=lambda n: n.sequence)
        
        # Fall back to any following node
        following_nodes = [
            n for n in text_nodes
            if (n.page > target.page) or
               (n.page == target.page and n.sequence > target.sequence)
        ]
        
        if following_nodes:
            return min(following_nodes, key=lambda n: (n.page, n.sequence))
        
        return None


# ============================================================================
# Node Linker
# ============================================================================

class NodeLinker:
    """
    Links nodes based on V1 rules.
    
    V1 Rules:
    1. If image bbox is spatially inside text block → link
    2. Else → link to immediately preceding text in reading order
    """
    
    def __init__(
        self,
        containment_threshold: float = 0.7,
        link_to_following: bool = False
    ):
        """
        Initialize node linker.
        
        Args:
            containment_threshold: Minimum overlap for spatial containment
            link_to_following: Also link to following text (in addition to preceding)
        """
        self.spatial_analyzer = SpatialAnalyzer(containment_threshold)
        self.reading_order_analyzer = ReadingOrderAnalyzer()
        self.link_to_following = link_to_following
    
    def link_nodes(self, document: IngestionDocument) -> List[LinkageResult]:
        """
        Create links between image and text nodes.
        
        Args:
            document: IngestionDocument with nodes
            
        Returns:
            List of LinkageResult objects
        """
        results = []
        
        # Get image and text nodes
        image_nodes = document.get_image_nodes()
        text_nodes = document.get_text_nodes()
        
        if not image_nodes or not text_nodes:
            return results
        
        # Sort text nodes by reading order
        text_nodes_sorted = sorted(text_nodes, key=lambda n: (n.page, n.sequence))
        
        # Process each image node
        for image_node in image_nodes:
            linkage = self._link_image_to_text(
                image_node,
                text_nodes_sorted,
                document
            )
            if linkage:
                results.extend(linkage)
        
        return results
    
    def _link_image_to_text(
        self,
        image_node: ContentNode,
        text_nodes: List[ContentNode],
        document: IngestionDocument
    ) -> List[LinkageResult]:
        """
        Link a single image node to text nodes.
        
        Args:
            image_node: Image node to link
            text_nodes: Sorted list of text nodes
            document: Parent document
            
        Returns:
            List of LinkageResult objects
        """
        results = []
        
        # Rule 1: Check spatial containment
        if image_node.bbox:
            containment = self.spatial_analyzer.find_containing_node(
                image_node,
                text_nodes
            )
            
            if containment:
                containing_node, ratio = containment
                
                # Create bidirectional link
                image_node.add_link(containing_node.id)
                containing_node.add_link(image_node.id)
                
                results.append(LinkageResult(
                    image_node_id=image_node.id,
                    text_node_id=containing_node.id,
                    linkage_type="spatial_containment",
                    confidence=ratio,
                    metadata={
                        "containment_ratio": ratio,
                        "page": image_node.page
                    }
                ))
                
                return results
        
        # Rule 2: Link to preceding text in reading order
        preceding = self.reading_order_analyzer.find_preceding_text(
            image_node,
            text_nodes
        )
        
        if preceding:
            image_node.add_link(preceding.id)
            preceding.add_link(image_node.id)
            
            results.append(LinkageResult(
                image_node_id=image_node.id,
                text_node_id=preceding.id,
                linkage_type="reading_order_preceding",
                confidence=0.8,  # Lower confidence than spatial
                metadata={
                    "sequence_diff": image_node.sequence - preceding.sequence,
                    "same_page": image_node.page == preceding.page
                }
            ))
        
        # Optionally link to following text as well
        if self.link_to_following:
            following = self.reading_order_analyzer.find_following_text(
                image_node,
                text_nodes
            )
            
            if following:
                image_node.add_link(following.id)
                following.add_link(image_node.id)
                
                results.append(LinkageResult(
                    image_node_id=image_node.id,
                    text_node_id=following.id,
                    linkage_type="reading_order_following",
                    confidence=0.6,  # Even lower confidence
                    metadata={
                        "sequence_diff": following.sequence - image_node.sequence,
                        "same_page": image_node.page == following.page
                    }
                ))
        
        return results


# ============================================================================
# Linkage Pipeline
# ============================================================================

@register_pipeline("linkage")
class LinkagePipeline(BasePipeline):
    """
    Pipeline stage for linking image and text nodes.
    
    Implements V1 linkage rules:
    1. Spatial containment
    2. Reading order fallback
    """
    
    stage_name = "linkage"
    
    def __init__(
        self,
        containment_threshold: float = 0.7,
        link_to_following: bool = False
    ):
        """
        Initialize linkage pipeline.
        
        Args:
            containment_threshold: Minimum overlap for spatial containment
            link_to_following: Also link to following text
        """
        super().__init__()
        self.linker = NodeLinker(
            containment_threshold=containment_threshold,
            link_to_following=link_to_following
        )
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document to create node links.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Document with linked nodes
        """
        # Create links
        results = self.linker.link_nodes(document)
        
        # Store statistics
        spatial_links = len([r for r in results if r.linkage_type == "spatial_containment"])
        reading_order_links = len([r for r in results if "reading_order" in r.linkage_type])
        
        document.metadata["linkage_stats"] = {
            "total_links": len(results),
            "spatial_containment_links": spatial_links,
            "reading_order_links": reading_order_links,
            "image_nodes_processed": len(document.get_image_nodes()),
        }
        
        # Store linkage results for debugging
        document.metadata["linkage_results"] = [
            {
                "image_id": r.image_node_id,
                "text_id": r.text_node_id,
                "type": r.linkage_type,
                "confidence": r.confidence
            }
            for r in results
        ]
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if linkage can be skipped."""
        # Skip if no images to link
        if not document.get_image_nodes():
            return True
        
        # Skip if no text to link to
        if not document.get_text_nodes():
            return True
        
        return document.is_stage_completed(self.stage_name)


# ============================================================================
# Utility Functions
# ============================================================================

def get_linked_images(
    document: IngestionDocument,
    text_node: ContentNode
) -> List[ContentNode]:
    """
    Get all images linked to a text node.
    
    Args:
        document: Parent document
        text_node: Text node to get linked images for
        
    Returns:
        List of linked image nodes
    """
    linked_images = []
    
    for node_id in text_node.linked_ids:
        node = document.get_node(node_id)
        if node and node.node_type == NodeType.IMAGE:
            linked_images.append(node)
    
    return linked_images


def get_linked_text(
    document: IngestionDocument,
    image_node: ContentNode
) -> List[ContentNode]:
    """
    Get all text nodes linked to an image.
    
    Args:
        document: Parent document
        image_node: Image node to get linked text for
        
    Returns:
        List of linked text nodes
    """
    linked_text = []
    text_types = {NodeType.TEXT, NodeType.HEADING, NodeType.LIST_ITEM}
    
    for node_id in image_node.linked_ids:
        node = document.get_node(node_id)
        if node and node.node_type in text_types:
            linked_text.append(node)
    
    return linked_text


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "LinkageResult",
    "SpatialAnalyzer",
    "ReadingOrderAnalyzer",
    "NodeLinker",
    "LinkagePipeline",
    "get_linked_images",
    "get_linked_text",
]
