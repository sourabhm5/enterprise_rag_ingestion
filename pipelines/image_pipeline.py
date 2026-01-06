"""
MODULE 10 — IMAGE PIPELINE (NO COMPROMISE)
==========================================
Image extraction and processing pipeline for the Enterprise RAG Ingestion Pipeline.

Responsibilities:
- Extract embedded images from documents
- Store original images in S3
- Perform OCR text extraction
- Generate vision captions
- Emit IMAGE nodes

Image nodes MUST have:
- Bounding box (bbox)
- Be linkable to text nodes
- S3 path to stored image
- OCR text (if applicable)
- Vision caption (if applicable)
"""

import base64
import hashlib
import io
import struct
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pipelines.base import BasePipeline, PipelineError
from schema.intermediate_representation import (
    BoundingBox,
    ContentNode,
    IngestionDocument,
    NodeType,
    ProcessingStatus,
)
from storage.s3 import S3Storage, S3PathType, get_s3_storage
from config.settings import get_settings
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Image Utilities
# ============================================================================

class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def get_image_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
        """
        Get image dimensions without PIL.
        
        Args:
            data: Image bytes
            
        Returns:
            (width, height) tuple or None
        """
        if not data or len(data) < 24:
            return None
        
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            try:
                width = struct.unpack('>I', data[16:20])[0]
                height = struct.unpack('>I', data[20:24])[0]
                return (width, height)
            except Exception:
                pass
        
        # JPEG
        if data[:2] == b'\xff\xd8':
            try:
                return ImageUtils._get_jpeg_dimensions(data)
            except Exception:
                pass
        
        # GIF
        if data[:6] in (b'GIF87a', b'GIF89a'):
            try:
                width = struct.unpack('<H', data[6:8])[0]
                height = struct.unpack('<H', data[8:10])[0]
                return (width, height)
            except Exception:
                pass
        
        # WebP
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            try:
                if data[12:16] == b'VP8 ':
                    width = struct.unpack('<H', data[26:28])[0] & 0x3fff
                    height = struct.unpack('<H', data[28:30])[0] & 0x3fff
                    return (width, height)
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def _get_jpeg_dimensions(data: bytes) -> Optional[Tuple[int, int]]:
        """Parse JPEG to get dimensions."""
        stream = io.BytesIO(data)
        stream.read(2)  # Skip SOI
        
        while True:
            marker = stream.read(2)
            if len(marker) < 2:
                break
            
            if marker[0] != 0xFF:
                break
            
            # SOF markers
            if marker[1] in (0xC0, 0xC1, 0xC2):
                stream.read(3)  # Skip length and precision
                height = struct.unpack('>H', stream.read(2))[0]
                width = struct.unpack('>H', stream.read(2))[0]
                return (width, height)
            
            # Skip other markers
            if marker[1] not in (0xD8, 0xD9):
                length = struct.unpack('>H', stream.read(2))[0]
                stream.read(length - 2)
        
        return None
    
    @staticmethod
    def detect_image_format(data: bytes) -> Optional[str]:
        """
        Detect image format from magic bytes.
        
        Args:
            data: Image bytes
            
        Returns:
            Format string (png, jpeg, gif, webp) or None
        """
        if not data or len(data) < 12:
            return None
        
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'
        if data[:2] == b'\xff\xd8':
            return 'jpeg'
        if data[:6] in (b'GIF87a', b'GIF89a'):
            return 'gif'
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return 'webp'
        if data[:2] in (b'BM',):
            return 'bmp'
        if data[:4] in (b'II*\x00', b'MM\x00*'):
            return 'tiff'
        
        return None
    
    @staticmethod
    def compute_image_hash(data: bytes) -> str:
        """Compute SHA-256 hash of image data."""
        return hashlib.sha256(data).hexdigest()


# ============================================================================
# OCR Service
# ============================================================================

class OCRService:
    """
    OCR text extraction service.
    
    Uses Tesseract via pytesseract if available.
    """
    
    def __init__(self):
        """Initialize OCR service."""
        self._tesseract_available = self._check_tesseract()
        self._pil_available = self._check_pil()
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            # Try to get version to confirm it's working
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _check_pil(self) -> bool:
        """Check if PIL is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._tesseract_available and self._pil_available
    
    async def extract_text(self, image_data: bytes) -> Optional[str]:
        """
        Extract text from image using OCR.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Extracted text or None
        """
        if not self.is_available:
            return None
        
        try:
            import pytesseract
            from PIL import Image
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Extract text
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Clean up
            text = text.strip()
            
            return text if text else None
            
        except Exception as e:
            return None


# ============================================================================
# Vision Captioning Service
# ============================================================================

class VisionCaptionService:
    """
    Vision model captioning service.
    
    Uses OpenAI Vision API or similar for generating image descriptions.
    """
    
    def __init__(self):
        """Initialize vision service."""
        self.settings = get_settings()
        self._openai_available = self._check_openai()
    
    def _check_openai(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return True
        except ImportError:
            return False
    
    @property
    def is_available(self) -> bool:
        """Check if vision captioning is available."""
        return (
            self._openai_available and
            self.settings.feature_flags.enable_vision_captioning
        )
    
    async def generate_caption(
        self,
        image_data: bytes,
        image_format: str = 'png',
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a caption for an image.
        
        Args:
            image_data: Image bytes
            image_format: Image format (png, jpeg, etc.)
            context: Optional context about the document
            
        Returns:
            Generated caption or None
        """
        if not self.is_available:
            return None
        
        try:
            import openai
            
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Determine media type
            media_type_map = {
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'jpg': 'image/jpeg',
                'gif': 'image/gif',
                'webp': 'image/webp',
            }
            media_type = media_type_map.get(image_format, 'image/png')
            
            # Build prompt
            prompt = "Describe this image in detail for document retrieval purposes. Focus on any text, diagrams, charts, or important visual information."
            if context:
                prompt += f"\n\nDocument context: {context}"
            
            # Call OpenAI Vision
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_b64}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            caption = response.choices[0].message.content
            return caption.strip() if caption else None
            
        except Exception as e:
            # Log but don't fail
            return None


# ============================================================================
# Image Pipeline
# ============================================================================

@register_pipeline("image_pipeline")
class ImagePipeline(BasePipeline):
    """
    Image extraction and processing pipeline.
    
    Handles:
    - Extracting embedded images from PDFs (from layout parser)
    - Processing standalone image documents
    - OCR text extraction
    - Vision captioning
    - S3 storage
    """
    
    stage_name = "image_pipeline"
    
    def __init__(
        self,
        s3_storage: Optional[S3Storage] = None,
        enable_ocr: Optional[bool] = None,
        enable_captioning: Optional[bool] = None
    ):
        """
        Initialize image pipeline.
        
        Args:
            s3_storage: S3 storage instance
            enable_ocr: Override OCR feature flag
            enable_captioning: Override captioning feature flag
        """
        super().__init__()
        self.s3 = s3_storage or get_s3_storage()
        self.settings = get_settings()
        
        self.enable_ocr = enable_ocr if enable_ocr is not None else self.settings.feature_flags.enable_ocr
        self.enable_captioning = enable_captioning if enable_captioning is not None else self.settings.feature_flags.enable_vision_captioning
        
        self.ocr_service = OCRService()
        self.vision_service = VisionCaptionService()
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process images in the document.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Updated IngestionDocument with IMAGE nodes
        """
        # Check if this is a standalone image document
        if document.mime_type.startswith('image/'):
            document = await self._process_standalone_image(document)
        else:
            # Process embedded images from layout parser
            document = await self._process_embedded_images(document)
        
        # Calculate statistics
        image_nodes = document.get_image_nodes()
        document.metadata['image_pipeline_stats'] = {
            'image_count': len(image_nodes),
            'images_with_ocr': len([n for n in image_nodes if n.ocr_text]),
            'images_with_caption': len([n for n in image_nodes if n.image_caption]),
        }
        
        return document
    
    async def _process_standalone_image(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process a standalone image document.
        
        Args:
            document: Document with image as raw_content
            
        Returns:
            Updated document
        """
        if not document.raw_content:
            document.add_error(
                self.stage_name,
                "No raw content available for image processing"
            )
            return document
        
        image_data = document.raw_content
        
        # Detect format
        image_format = ImageUtils.detect_image_format(image_data)
        if not image_format:
            # Try to get from MIME type
            format_map = {
                'image/png': 'png',
                'image/jpeg': 'jpeg',
                'image/gif': 'gif',
                'image/webp': 'webp',
            }
            image_format = format_map.get(document.mime_type, 'png')
        
        # Get dimensions
        dimensions = ImageUtils.get_image_dimensions(image_data)
        
        # Create image node
        node = ContentNode(
            node_type=NodeType.IMAGE,
            sequence=0,
            page=0,
            image_data=image_data,
            image_format=image_format,
            image_width=dimensions[0] if dimensions else None,
            image_height=dimensions[1] if dimensions else None,
            source_pipeline=self.stage_name,
        )
        
        # Process the image
        node = await self._process_image_node(node, document)
        
        # Add to document
        document.add_node(node)
        
        return document
    
    async def _process_embedded_images(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process embedded images from layout parser.
        
        Args:
            document: Document with image nodes from layout parser
            
        Returns:
            Updated document
        """
        # Find existing image nodes
        image_nodes = document.get_image_nodes()
        
        for node in image_nodes:
            # Skip if already processed
            if node.status == ProcessingStatus.COMPLETED and node.image_path:
                continue
            
            # Need image data
            if not node.image_data:
                node.status = ProcessingStatus.SKIPPED
                node.metadata['skip_reason'] = 'no_image_data'
                continue
            
            # Process the image
            node = await self._process_image_node(node, document)
        
        return document
    
    async def _process_image_node(
        self,
        node: ContentNode,
        document: IngestionDocument
    ) -> ContentNode:
        """
        Process a single image node.
        
        Args:
            node: Image node to process
            document: Parent document
            
        Returns:
            Processed node
        """
        if not node.image_data:
            node.status = ProcessingStatus.FAILED
            return node
        
        try:
            # Detect format if not set
            if not node.image_format:
                node.image_format = ImageUtils.detect_image_format(node.image_data) or 'png'
            
            # Get dimensions if not set
            if not node.image_width or not node.image_height:
                dimensions = ImageUtils.get_image_dimensions(node.image_data)
                if dimensions:
                    node.image_width, node.image_height = dimensions
            
            # Generate filename
            image_hash = ImageUtils.compute_image_hash(node.image_data)[:12]
            filename = f"image_{node.sequence}_{image_hash}.{node.image_format}"
            
            # Upload to S3
            s3_path = await self._upload_image(
                document=document,
                filename=filename,
                image_data=node.image_data,
                image_format=node.image_format
            )
            node.image_path = s3_path
            
            # OCR
            if self.enable_ocr and self.ocr_service.is_available:
                ocr_text = await self.ocr_service.extract_text(node.image_data)
                if ocr_text:
                    node.ocr_text = ocr_text
            
            # Vision captioning
            if self.enable_captioning and self.vision_service.is_available:
                # Get context from nearby text
                context = self._get_context_for_image(document, node)
                caption = await self.vision_service.generate_caption(
                    node.image_data,
                    node.image_format,
                    context
                )
                if caption:
                    node.image_caption = caption
            
            # Clear raw data to save memory
            node.image_data = None
            
            # Mark as completed
            node.status = ProcessingStatus.COMPLETED
            node.source_pipeline = self.stage_name
            
        except Exception as e:
            node.status = ProcessingStatus.FAILED
            node.metadata['error'] = str(e)
        
        return node
    
    async def _upload_image(
        self,
        document: IngestionDocument,
        filename: str,
        image_data: bytes,
        image_format: str
    ) -> str:
        """
        Upload image to S3.
        
        Args:
            document: Parent document
            filename: Image filename
            image_data: Image bytes
            image_format: Image format
            
        Returns:
            S3 path
        """
        # Generate S3 path
        s3_path = self.s3.generate_path(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            version=document.version,
            path_type=S3PathType.DERIVED,
            filename=filename
        )
        
        # Determine content type
        content_type_map = {
            'png': 'image/png',
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg',
            'gif': 'image/gif',
            'webp': 'image/webp',
        }
        content_type = content_type_map.get(image_format, 'image/png')
        
        # Upload
        result = await self.s3.upload_derived_asset(
            tenant_id=document.tenant_id,
            document_id=document.document_id,
            version=document.version,
            filename=filename,
            content=image_data,
            content_type=content_type,
            metadata={
                'source_document': document.document_id,
                'image_format': image_format
            }
        )
        
        return result.key
    
    def _get_context_for_image(
        self,
        document: IngestionDocument,
        image_node: ContentNode
    ) -> Optional[str]:
        """
        Get context text for image captioning.
        
        Args:
            document: Parent document
            image_node: Image node
            
        Returns:
            Context string or None
        """
        context_parts = []
        
        # Document info
        context_parts.append(f"Document: {document.filename}")
        
        # Find nearby text nodes
        text_nodes = document.get_text_nodes()
        
        # Get nodes on same page
        same_page_nodes = [
            n for n in text_nodes
            if n.page == image_node.page
        ]
        
        # Sort by sequence
        same_page_nodes.sort(key=lambda n: n.sequence)
        
        # Get nodes before and after
        before = [n for n in same_page_nodes if n.sequence < image_node.sequence]
        after = [n for n in same_page_nodes if n.sequence > image_node.sequence]
        
        # Take closest text
        if before:
            context_parts.append(f"Preceding text: {before[-1].text[:200]}")
        if after:
            context_parts.append(f"Following text: {after[0].text[:200]}")
        
        return " | ".join(context_parts) if context_parts else None
    
    def validate_input(self, document: IngestionDocument) -> bool:
        """Validate input document."""
        # Need either image MIME type or existing image nodes
        is_image = document.mime_type.startswith('image/')
        has_image_nodes = bool(document.get_image_nodes())
        return is_image or has_image_nodes


# ============================================================================
# Layout Parser Integration Pipeline
# ============================================================================

@register_pipeline("layout_parsing")
class LayoutParsingPipeline(BasePipeline):
    """
    Layout parsing pipeline stage.
    
    Wraps the layout parser module for use in the pipeline executor.
    """
    
    stage_name = "layout_parsing"
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Run layout parsing on the document.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Document with nodes from layout parsing
        """
        from processing.layout_parser import LayoutParser
        
        settings = get_settings()
        
        # Skip if layout parsing disabled
        if not settings.feature_flags.enable_layout_parsing:
            return document
        
        # Skip if not a PDF
        if document.mime_type != 'application/pdf':
            return document
        
        # Parse document
        parser = LayoutParser()
        document = parser.parse_to_ir(document)
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if layout parsing can be skipped."""
        # Skip if already has nodes from layout parsing
        if document.is_stage_completed(self.stage_name):
            return True
        
        # Skip if not a PDF
        if document.mime_type != 'application/pdf':
            return True
        
        return False


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ImagePipeline",
    "LayoutParsingPipeline",
    "ImageUtils",
    "OCRService",
    "VisionCaptionService",
]
