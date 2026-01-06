"""
MODULE 6 — INGESTION ROUTER (MERGED MODULE)
===========================================
One deterministic "brain" for routing in the Enterprise RAG Ingestion Pipeline.

This module merges the responsibilities of:
- MIME type detection
- Content probing
- PDF detection
- Routing decision making

Responsibilities:
- Detect document MIME type
- Probe content characteristics
- Determine if PDF needs layout parsing
- Check for embedded images
- Generate ordered pipeline list (DAG)

Routing Rules (V1):
- PDFs → layout parser enabled
- Images present → image pipeline enabled
- All documents → text pipeline
- All documents → chunking
- All documents → embedding
"""

import io
import mimetypes
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Set, Tuple

from config.settings import FeatureFlags, get_settings


# ============================================================================
# Content Type Detection
# ============================================================================

class ContentType(str, Enum):
    """Detected content types."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    """Detected image formats."""
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"
    TIFF = "tiff"
    BMP = "bmp"
    UNKNOWN = "unknown"


# Magic bytes for file type detection
MAGIC_BYTES = {
    b"%PDF": ContentType.PDF,
    b"\x89PNG\r\n\x1a\n": ContentType.IMAGE,
    b"\xff\xd8\xff": ContentType.IMAGE,  # JPEG
    b"GIF87a": ContentType.IMAGE,
    b"GIF89a": ContentType.IMAGE,
    b"RIFF": ContentType.IMAGE,  # WEBP (need to check further)
    b"<!DOCTYPE": ContentType.HTML,
    b"<html": ContentType.HTML,
    b"<HTML": ContentType.HTML,
}

IMAGE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ImageFormat.PNG,
    b"\xff\xd8\xff": ImageFormat.JPEG,
    b"GIF87a": ImageFormat.GIF,
    b"GIF89a": ImageFormat.GIF,
    b"II*\x00": ImageFormat.TIFF,  # Little-endian TIFF
    b"MM\x00*": ImageFormat.TIFF,  # Big-endian TIFF
    b"BM": ImageFormat.BMP,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ContentProbeResult:
    """Result of content probing."""
    content_type: ContentType
    mime_type: str
    file_extension: str
    file_size_bytes: int
    
    # PDF-specific
    is_pdf: bool = False
    pdf_has_images: bool = False
    pdf_page_count: Optional[int] = None
    pdf_is_scanned: bool = False  # Likely image-only PDF
    
    # Image-specific
    is_image: bool = False
    image_format: Optional[ImageFormat] = None
    image_dimensions: Optional[Tuple[int, int]] = None
    
    # Text-specific
    is_text: bool = False
    is_markdown: bool = False
    is_html: bool = False
    text_encoding: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRoute:
    """Routing decision for a document."""
    document_id: str
    tenant_id: str
    mime_type: str
    content_type: ContentType
    
    # Pipeline stages in execution order
    pipeline_stages: List[str] = field(default_factory=list)
    
    # Feature flags applied
    features_enabled: Dict[str, bool] = field(default_factory=dict)
    
    # Routing metadata
    routing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_pipeline_plan(self) -> Dict[str, Any]:
        """Convert to pipeline plan format for job storage."""
        return {
            "stages": self.pipeline_stages,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "mime_type": self.mime_type,
            "content_type": self.content_type.value,
            "features_enabled": self.features_enabled,
            "routing_metadata": self.routing_metadata,
        }


# ============================================================================
# Content Prober
# ============================================================================

class ContentProber:
    """
    Probes document content to determine type and characteristics.
    
    Uses magic bytes and content analysis to detect:
    - File type (PDF, image, text, etc.)
    - PDF characteristics (page count, images, scanned)
    - Image format and dimensions
    - Text encoding
    """
    
    def __init__(self):
        """Initialize content prober."""
        self._magic_bytes = MAGIC_BYTES
        self._image_magic = IMAGE_MAGIC
    
    def probe(
        self,
        content: bytes,
        filename: Optional[str] = None,
        declared_mime_type: Optional[str] = None
    ) -> ContentProbeResult:
        """
        Probe content to determine type and characteristics.
        
        Args:
            content: File content as bytes
            filename: Original filename (optional)
            declared_mime_type: MIME type declared by client (optional)
            
        Returns:
            ContentProbeResult with detected information
        """
        file_size = len(content)
        
        # Detect content type from magic bytes
        detected_type = self._detect_content_type(content)
        
        # Determine MIME type
        mime_type = self._determine_mime_type(
            content, filename, declared_mime_type, detected_type
        )
        
        # Get file extension
        file_extension = self._get_extension(filename, mime_type)
        
        # Create base result
        result = ContentProbeResult(
            content_type=detected_type,
            mime_type=mime_type,
            file_extension=file_extension,
            file_size_bytes=file_size,
        )
        
        # Probe specific types
        if detected_type == ContentType.PDF:
            self._probe_pdf(content, result)
        elif detected_type == ContentType.IMAGE:
            self._probe_image(content, result)
        elif detected_type in (ContentType.TEXT, ContentType.MARKDOWN, ContentType.HTML):
            self._probe_text(content, result, detected_type)
        
        return result
    
    def _detect_content_type(self, content: bytes) -> ContentType:
        """Detect content type from magic bytes."""
        if not content:
            return ContentType.UNKNOWN
        
        # Check magic bytes
        for magic, content_type in self._magic_bytes.items():
            if content.startswith(magic):
                return content_type
        
        # Check for WEBP specifically (RIFF....WEBP)
        if content[:4] == b"RIFF" and len(content) >= 12 and content[8:12] == b"WEBP":
            return ContentType.IMAGE
        
        # Try to detect text
        try:
            # Try UTF-8 decode
            content[:1000].decode("utf-8")
            
            # Check for markdown indicators
            text_sample = content[:2000].decode("utf-8", errors="ignore")
            if self._looks_like_markdown(text_sample):
                return ContentType.MARKDOWN
            elif self._looks_like_html(text_sample):
                return ContentType.HTML
            else:
                return ContentType.TEXT
        except (UnicodeDecodeError, AttributeError):
            pass
        
        return ContentType.UNKNOWN
    
    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text looks like markdown."""
        markdown_indicators = [
            "# ",  # Headers
            "## ",
            "### ",
            "```",  # Code blocks
            "- ",  # Lists
            "* ",
            "1. ",
            "[",  # Links
            "**",  # Bold
            "__",
        ]
        
        indicator_count = sum(1 for ind in markdown_indicators if ind in text)
        return indicator_count >= 2
    
    def _looks_like_html(self, text: str) -> bool:
        """Check if text looks like HTML."""
        text_lower = text.lower()
        return (
            "<!doctype html" in text_lower or
            "<html" in text_lower or
            ("<head" in text_lower and "<body" in text_lower)
        )
    
    def _determine_mime_type(
        self,
        content: bytes,
        filename: Optional[str],
        declared_mime: Optional[str],
        detected_type: ContentType
    ) -> str:
        """Determine the best MIME type."""
        # Trust detection for certain types
        type_to_mime = {
            ContentType.PDF: "application/pdf",
            ContentType.MARKDOWN: "text/markdown",
            ContentType.HTML: "text/html",
            ContentType.TEXT: "text/plain",
        }
        
        if detected_type in type_to_mime:
            return type_to_mime[detected_type]
        
        if detected_type == ContentType.IMAGE:
            # Determine specific image type
            image_format = self._detect_image_format(content)
            format_to_mime = {
                ImageFormat.PNG: "image/png",
                ImageFormat.JPEG: "image/jpeg",
                ImageFormat.GIF: "image/gif",
                ImageFormat.WEBP: "image/webp",
                ImageFormat.TIFF: "image/tiff",
                ImageFormat.BMP: "image/bmp",
            }
            return format_to_mime.get(image_format, "image/unknown")
        
        # Fall back to declared or guessed from filename
        if declared_mime and declared_mime != "application/octet-stream":
            return declared_mime
        
        if filename:
            guessed, _ = mimetypes.guess_type(filename)
            if guessed:
                return guessed
        
        return "application/octet-stream"
    
    def _get_extension(self, filename: Optional[str], mime_type: str) -> str:
        """Get file extension."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext:
                return ext.lstrip(".")
        
        # Guess from MIME type
        ext = mimetypes.guess_extension(mime_type)
        if ext:
            return ext.lstrip(".")
        
        return ""
    
    def _detect_image_format(self, content: bytes) -> ImageFormat:
        """Detect specific image format."""
        for magic, fmt in self._image_magic.items():
            if content.startswith(magic):
                return fmt
        
        # Check WEBP
        if content[:4] == b"RIFF" and len(content) >= 12 and content[8:12] == b"WEBP":
            return ImageFormat.WEBP
        
        return ImageFormat.UNKNOWN
    
    def _probe_pdf(self, content: bytes, result: ContentProbeResult) -> None:
        """Probe PDF-specific characteristics."""
        result.is_pdf = True
        
        try:
            # Simple PDF analysis without heavy dependencies
            content_str = content.decode("latin-1", errors="ignore")
            
            # Count pages (rough estimate)
            page_count = content_str.count("/Type /Page") - content_str.count("/Type /Pages")
            if page_count <= 0:
                page_count = content_str.count("/Type/Page")
            result.pdf_page_count = max(1, page_count)
            
            # Check for images
            has_images = (
                "/XObject" in content_str or
                "/Image" in content_str or
                "/DCTDecode" in content_str or  # JPEG
                "/FlateDecode" in content_str  # Compressed (often images)
            )
            result.pdf_has_images = has_images
            
            # Check if likely scanned (image-heavy, little text)
            has_text_content = (
                "/Font" in content_str or
                "/Text" in content_str or
                "BT" in content_str  # Begin text
            )
            result.pdf_is_scanned = has_images and not has_text_content
            
            result.metadata["pdf_version"] = self._extract_pdf_version(content_str)
            
        except Exception as e:
            result.metadata["probe_error"] = str(e)
    
    def _extract_pdf_version(self, content_str: str) -> Optional[str]:
        """Extract PDF version from header."""
        if content_str.startswith("%PDF-"):
            return content_str[5:8]
        return None
    
    def _probe_image(self, content: bytes, result: ContentProbeResult) -> None:
        """Probe image-specific characteristics."""
        result.is_image = True
        result.image_format = self._detect_image_format(content)
        
        try:
            dimensions = self._get_image_dimensions(content, result.image_format)
            result.image_dimensions = dimensions
        except Exception as e:
            result.metadata["dimension_error"] = str(e)
    
    def _get_image_dimensions(
        self,
        content: bytes,
        image_format: ImageFormat
    ) -> Optional[Tuple[int, int]]:
        """Get image dimensions without PIL."""
        try:
            if image_format == ImageFormat.PNG:
                # PNG dimensions are at bytes 16-24
                if len(content) >= 24:
                    width = struct.unpack(">I", content[16:20])[0]
                    height = struct.unpack(">I", content[20:24])[0]
                    return (width, height)
            
            elif image_format == ImageFormat.JPEG:
                # JPEG requires parsing markers
                return self._get_jpeg_dimensions(content)
            
            elif image_format == ImageFormat.GIF:
                # GIF dimensions at bytes 6-10
                if len(content) >= 10:
                    width = struct.unpack("<H", content[6:8])[0]
                    height = struct.unpack("<H", content[8:10])[0]
                    return (width, height)
            
        except Exception:
            pass
        
        return None
    
    def _get_jpeg_dimensions(self, content: bytes) -> Optional[Tuple[int, int]]:
        """Parse JPEG to get dimensions."""
        try:
            stream = io.BytesIO(content)
            stream.read(2)  # Skip SOI marker
            
            while True:
                marker = stream.read(2)
                if len(marker) < 2:
                    break
                
                if marker[0] != 0xFF:
                    break
                
                # SOF markers contain dimensions
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    stream.read(3)  # Skip length and precision
                    height = struct.unpack(">H", stream.read(2))[0]
                    width = struct.unpack(">H", stream.read(2))[0]
                    return (width, height)
                
                # Skip other markers
                if marker[1] != 0xD8 and marker[1] != 0xD9:
                    length = struct.unpack(">H", stream.read(2))[0]
                    stream.read(length - 2)
        except Exception:
            pass
        
        return None
    
    def _probe_text(
        self,
        content: bytes,
        result: ContentProbeResult,
        detected_type: ContentType
    ) -> None:
        """Probe text-specific characteristics."""
        result.is_text = True
        result.is_markdown = detected_type == ContentType.MARKDOWN
        result.is_html = detected_type == ContentType.HTML
        
        # Detect encoding
        encodings = ["utf-8", "utf-16", "latin-1", "ascii"]
        for encoding in encodings:
            try:
                content.decode(encoding)
                result.text_encoding = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue


# ============================================================================
# Ingestion Router
# ============================================================================

class IngestionRouter:
    """
    Determines the processing pipeline for documents.
    
    Routes documents to appropriate pipelines based on:
    - Content type (PDF, image, text)
    - Feature flags
    - Content characteristics
    """
    
    # Pipeline stage definitions
    STAGE_ROUTING = "routing"
    STAGE_LAYOUT_PARSING = "layout_parsing"
    STAGE_TEXT_PIPELINE = "text_pipeline"
    STAGE_IMAGE_PIPELINE = "image_pipeline"
    STAGE_LINKAGE = "linkage"
    STAGE_CHUNKING = "chunking"
    STAGE_METADATA_ENRICHMENT = "metadata_enrichment"
    STAGE_EMBEDDING = "embedding"
    STAGE_VECTOR_STORE = "vector_store"
    
    def __init__(self, feature_flags: Optional[FeatureFlags] = None):
        """
        Initialize ingestion router.
        
        Args:
            feature_flags: Feature flags (default from settings)
        """
        self.feature_flags = feature_flags or get_settings().feature_flags
        self.prober = ContentProber()
    
    def route(
        self,
        document_id: str,
        tenant_id: str,
        content: bytes,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> PipelineRoute:
        """
        Determine the processing pipeline for a document.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            content: Document content
            filename: Original filename
            mime_type: Declared MIME type
            
        Returns:
            PipelineRoute with ordered pipeline stages
        """
        # Probe content
        probe_result = self.prober.probe(content, filename, mime_type)
        
        # Determine pipeline stages
        stages = self._determine_stages(probe_result)
        
        # Track enabled features
        features = {
            "layout_parsing": self.feature_flags.enable_layout_parsing,
            "image_pipeline": self.feature_flags.enable_image_pipeline,
            "llm_metadata_enrichment": self.feature_flags.enable_llm_metadata_enrichment,
            "ocr": self.feature_flags.enable_ocr,
            "vision_captioning": self.feature_flags.enable_vision_captioning,
        }
        
        # Build routing metadata
        routing_metadata = {
            "probe_result": {
                "content_type": probe_result.content_type.value,
                "is_pdf": probe_result.is_pdf,
                "pdf_has_images": probe_result.pdf_has_images,
                "pdf_page_count": probe_result.pdf_page_count,
                "pdf_is_scanned": probe_result.pdf_is_scanned,
                "is_image": probe_result.is_image,
                "image_format": probe_result.image_format.value if probe_result.image_format else None,
                "is_text": probe_result.is_text,
                "is_markdown": probe_result.is_markdown,
                "is_html": probe_result.is_html,
            },
            "file_size_bytes": probe_result.file_size_bytes,
        }
        
        return PipelineRoute(
            document_id=document_id,
            tenant_id=tenant_id,
            mime_type=probe_result.mime_type,
            content_type=probe_result.content_type,
            pipeline_stages=stages,
            features_enabled=features,
            routing_metadata=routing_metadata,
        )
    
    def route_from_mime_type(
        self,
        document_id: str,
        tenant_id: str,
        mime_type: str,
        file_size: int = 0,
        has_images: bool = False
    ) -> PipelineRoute:
        """
        Quick routing based on MIME type only (without content probing).
        
        Useful when content isn't available or for quick routing decisions.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            mime_type: Document MIME type
            file_size: File size in bytes
            has_images: Whether document has images (for PDFs)
            
        Returns:
            PipelineRoute with ordered pipeline stages
        """
        # Map MIME type to content type
        content_type = self._mime_to_content_type(mime_type)
        
        # Create synthetic probe result
        probe_result = ContentProbeResult(
            content_type=content_type,
            mime_type=mime_type,
            file_extension=mimetypes.guess_extension(mime_type) or "",
            file_size_bytes=file_size,
            is_pdf=content_type == ContentType.PDF,
            pdf_has_images=has_images,
            is_image=content_type == ContentType.IMAGE,
            is_text=content_type in (ContentType.TEXT, ContentType.MARKDOWN, ContentType.HTML),
        )
        
        stages = self._determine_stages(probe_result)
        
        return PipelineRoute(
            document_id=document_id,
            tenant_id=tenant_id,
            mime_type=mime_type,
            content_type=content_type,
            pipeline_stages=stages,
            features_enabled={
                "layout_parsing": self.feature_flags.enable_layout_parsing,
                "image_pipeline": self.feature_flags.enable_image_pipeline,
                "llm_metadata_enrichment": self.feature_flags.enable_llm_metadata_enrichment,
            },
            routing_metadata={"quick_route": True},
        )
    
    def _mime_to_content_type(self, mime_type: str) -> ContentType:
        """Map MIME type to content type."""
        mime_lower = mime_type.lower()
        
        if mime_lower == "application/pdf":
            return ContentType.PDF
        elif mime_lower.startswith("image/"):
            return ContentType.IMAGE
        elif mime_lower == "text/markdown":
            return ContentType.MARKDOWN
        elif mime_lower == "text/html":
            return ContentType.HTML
        elif mime_lower.startswith("text/"):
            return ContentType.TEXT
        else:
            return ContentType.UNKNOWN
    
    def _determine_stages(self, probe_result: ContentProbeResult) -> List[str]:
        """
        Determine pipeline stages based on probe result.
        
        Args:
            probe_result: Content probe result
            
        Returns:
            Ordered list of pipeline stage names
        """
        stages = []
        
        # 1. Always start with routing (already done, but tracked)
        stages.append(self.STAGE_ROUTING)
        
        # 2. Layout parsing for PDFs (if enabled)
        if probe_result.is_pdf and self.feature_flags.enable_layout_parsing:
            stages.append(self.STAGE_LAYOUT_PARSING)
        
        # 3. Text pipeline (always for text-containing documents)
        if not probe_result.is_image or probe_result.is_pdf:
            stages.append(self.STAGE_TEXT_PIPELINE)
        
        # 4. Image pipeline (if enabled and document has images)
        if self.feature_flags.enable_image_pipeline:
            needs_image_pipeline = (
                probe_result.is_image or
                probe_result.pdf_has_images or
                probe_result.pdf_is_scanned
            )
            if needs_image_pipeline:
                stages.append(self.STAGE_IMAGE_PIPELINE)
        
        # 5. Linkage (if we have both text and images)
        has_text = self.STAGE_TEXT_PIPELINE in stages
        has_images = self.STAGE_IMAGE_PIPELINE in stages
        if has_text and has_images:
            stages.append(self.STAGE_LINKAGE)
        
        # 6. Chunking (always needed for retrieval)
        stages.append(self.STAGE_CHUNKING)
        
        # 7. Metadata enrichment (if enabled)
        if self.feature_flags.enable_llm_metadata_enrichment:
            stages.append(self.STAGE_METADATA_ENRICHMENT)
        
        # 8. Embedding (always needed)
        stages.append(self.STAGE_EMBEDDING)
        
        # 9. Vector store (always needed)
        stages.append(self.STAGE_VECTOR_STORE)
        
        return stages
    
    def get_stage_dependencies(self) -> Dict[str, List[str]]:
        """
        Get stage dependency graph.
        
        Returns:
            Dict mapping stage to its dependencies
        """
        return {
            self.STAGE_ROUTING: [],
            self.STAGE_LAYOUT_PARSING: [self.STAGE_ROUTING],
            self.STAGE_TEXT_PIPELINE: [self.STAGE_ROUTING, self.STAGE_LAYOUT_PARSING],
            self.STAGE_IMAGE_PIPELINE: [self.STAGE_ROUTING, self.STAGE_LAYOUT_PARSING],
            self.STAGE_LINKAGE: [self.STAGE_TEXT_PIPELINE, self.STAGE_IMAGE_PIPELINE],
            self.STAGE_CHUNKING: [self.STAGE_TEXT_PIPELINE, self.STAGE_LINKAGE],
            self.STAGE_METADATA_ENRICHMENT: [self.STAGE_CHUNKING],
            self.STAGE_EMBEDDING: [self.STAGE_CHUNKING, self.STAGE_METADATA_ENRICHMENT],
            self.STAGE_VECTOR_STORE: [self.STAGE_EMBEDDING],
        }
    
    def validate_pipeline(self, stages: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that a pipeline stage list is valid.
        
        Args:
            stages: List of stage names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        dependencies = self.get_stage_dependencies()
        seen = set()
        
        for stage in stages:
            if stage not in dependencies:
                return False, f"Unknown stage: {stage}"
            
            # Check dependencies are satisfied
            for dep in dependencies[stage]:
                # Dependencies are optional if they're not in the pipeline
                if dep in stages and dep not in seen:
                    return False, f"Stage {stage} depends on {dep} which hasn't been executed"
            
            seen.add(stage)
        
        return True, None


# ============================================================================
# Factory Functions
# ============================================================================

def get_ingestion_router() -> IngestionRouter:
    """Get an ingestion router instance."""
    return IngestionRouter()


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ContentType",
    "ImageFormat",
    "ContentProbeResult",
    "PipelineRoute",
    "ContentProber",
    "IngestionRouter",
    "get_ingestion_router",
]
