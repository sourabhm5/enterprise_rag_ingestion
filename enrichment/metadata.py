"""
MODULE 16 — METADATA ENRICHMENT (RBAC-RELEVANT ONLY)
====================================================
Metadata extraction and enrichment for the Enterprise RAG Ingestion Pipeline.

Extract:
- department
- document_type
- fiscal_year

Constraints:
- Deterministic
- Strict JSON schema

This module extracts structured metadata from documents to enable
better filtering, routing, and access control decisions.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pipelines.base import BasePipeline
from schema.intermediate_representation import (
    ContentNode,
    IngestionDocument,
    NodeType,
)
from config.settings import get_settings
from jobs.pipeline_executor import register_pipeline


# ============================================================================
# Metadata Schema
# ============================================================================

class DocumentType(str, Enum):
    """Standard document types."""
    REPORT = "report"
    PRESENTATION = "presentation"
    MEMO = "memo"
    CONTRACT = "contract"
    INVOICE = "invoice"
    POLICY = "policy"
    PROCEDURE = "procedure"
    MANUAL = "manual"
    SPECIFICATION = "specification"
    PROPOSAL = "proposal"
    MEETING_NOTES = "meeting_notes"
    EMAIL = "email"
    LETTER = "letter"
    FORM = "form"
    SPREADSHEET = "spreadsheet"
    OTHER = "other"


class Department(str, Enum):
    """Standard departments."""
    ENGINEERING = "engineering"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    HR = "hr"
    LEGAL = "legal"
    OPERATIONS = "operations"
    IT = "it"
    EXECUTIVE = "executive"
    PRODUCT = "product"
    SUPPORT = "support"
    RESEARCH = "research"
    COMPLIANCE = "compliance"
    UNKNOWN = "unknown"


@dataclass
class EnrichedMetadata:
    """
    Enriched metadata extracted from document.
    
    All fields are optional and extracted on best-effort basis.
    """
    # Core classification
    document_type: Optional[str] = None
    department: Optional[str] = None
    
    # Temporal
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[str] = None
    document_date: Optional[str] = None  # ISO format
    
    # Business context
    project_name: Optional[str] = None
    client_name: Optional[str] = None
    product_name: Optional[str] = None
    
    # Document properties
    author: Optional[str] = None
    language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Confidence scores
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Extraction metadata
    extraction_method: str = "rule_based"
    extracted_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_type": self.document_type,
            "department": self.department,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "document_date": self.document_date,
            "project_name": self.project_name,
            "client_name": self.client_name,
            "product_name": self.product_name,
            "author": self.author,
            "language": self.language,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "extracted_at": self.extracted_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedMetadata":
        """Create from dictionary."""
        return cls(
            document_type=data.get("document_type"),
            department=data.get("department"),
            fiscal_year=data.get("fiscal_year"),
            fiscal_quarter=data.get("fiscal_quarter"),
            document_date=data.get("document_date"),
            project_name=data.get("project_name"),
            client_name=data.get("client_name"),
            product_name=data.get("product_name"),
            author=data.get("author"),
            language=data.get("language"),
            keywords=data.get("keywords", []),
            confidence=data.get("confidence", {}),
            extraction_method=data.get("extraction_method", "rule_based"),
            extracted_at=data.get("extracted_at"),
        )


# ============================================================================
# Extraction Strategies
# ============================================================================

class MetadataExtractor(ABC):
    """Abstract base class for metadata extractors."""
    
    @abstractmethod
    def extract(self, document: IngestionDocument) -> EnrichedMetadata:
        """
        Extract metadata from document.
        
        Args:
            document: IngestionDocument to extract from
            
        Returns:
            EnrichedMetadata
        """
        pass


class RuleBasedExtractor(MetadataExtractor):
    """
    Rule-based metadata extraction.
    
    Uses pattern matching and heuristics for deterministic extraction.
    """
    
    # Document type patterns
    DOC_TYPE_PATTERNS = {
        DocumentType.REPORT: [
            r'\breport\b', r'\banalysis\b', r'\bstudy\b', r'\bfindings\b'
        ],
        DocumentType.PRESENTATION: [
            r'\bpresentation\b', r'\bslides?\b', r'\bdeck\b'
        ],
        DocumentType.MEMO: [
            r'\bmemo\b', r'\bmemorandum\b'
        ],
        DocumentType.CONTRACT: [
            r'\bcontract\b', r'\bagreement\b', r'\bterms\b'
        ],
        DocumentType.INVOICE: [
            r'\binvoice\b', r'\bbill\b', r'\bpayment\b'
        ],
        DocumentType.POLICY: [
            r'\bpolicy\b', r'\bguideline\b', r'\bstandard\b'
        ],
        DocumentType.PROCEDURE: [
            r'\bprocedure\b', r'\bprocess\b', r'\bworkflow\b'
        ],
        DocumentType.MANUAL: [
            r'\bmanual\b', r'\bguide\b', r'\bhandbook\b'
        ],
        DocumentType.SPECIFICATION: [
            r'\bspec\b', r'\bspecification\b', r'\brequirements?\b'
        ],
        DocumentType.PROPOSAL: [
            r'\bproposal\b', r'\brfp\b', r'\bbid\b'
        ],
        DocumentType.MEETING_NOTES: [
            r'\bmeeting\s+notes?\b', r'\bminutes\b', r'\bagenda\b'
        ],
    }
    
    # Department patterns
    DEPT_PATTERNS = {
        Department.ENGINEERING: [
            r'\bengineering\b', r'\bdevelopment\b', r'\btechnical\b', r'\bdev\b'
        ],
        Department.SALES: [
            r'\bsales\b', r'\brevenue\b', r'\bdeals?\b'
        ],
        Department.MARKETING: [
            r'\bmarketing\b', r'\bbrand\b', r'\bcampaign\b'
        ],
        Department.FINANCE: [
            r'\bfinance\b', r'\baccounting\b', r'\bbudget\b', r'\bfiscal\b'
        ],
        Department.HR: [
            r'\bhr\b', r'\bhuman\s+resources?\b', r'\bpeople\b', r'\brecruiting\b'
        ],
        Department.LEGAL: [
            r'\blegal\b', r'\bcompliance\b', r'\bregulatory\b'
        ],
        Department.OPERATIONS: [
            r'\boperations?\b', r'\blogs?istics?\b', r'\bsupply\s+chain\b'
        ],
        Department.IT: [
            r'\bit\b', r'\binformation\s+technology\b', r'\binfrastructure\b'
        ],
        Department.EXECUTIVE: [
            r'\bexecutive\b', r'\bc-suite\b', r'\bboard\b', r'\bceo\b', r'\bcfo\b'
        ],
        Department.PRODUCT: [
            r'\bproduct\b', r'\bpm\b', r'\broadmap\b'
        ],
    }
    
    # Fiscal year patterns
    FISCAL_YEAR_PATTERNS = [
        r'\bfy\s*(\d{2,4})\b',
        r'\bfiscal\s+year\s+(\d{4})\b',
        r'\b(20\d{2})\s+fiscal\b',
        r'\bq[1-4]\s+(20\d{2})\b',
    ]
    
    # Fiscal quarter patterns
    FISCAL_QUARTER_PATTERNS = [
        r'\b(q[1-4])\s*(?:20)?\d{2}\b',
        r'\b(q[1-4])\b',
        r'\b(first|second|third|fourth)\s+quarter\b',
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
        r'\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b',
    ]
    
    def __init__(self):
        """Initialize rule-based extractor."""
        # Compile patterns
        self._doc_type_compiled = {
            dt: [re.compile(p, re.IGNORECASE) for p in patterns]
            for dt, patterns in self.DOC_TYPE_PATTERNS.items()
        }
        
        self._dept_compiled = {
            dept: [re.compile(p, re.IGNORECASE) for p in patterns]
            for dept, patterns in self.DEPT_PATTERNS.items()
        }
        
        self._fy_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.FISCAL_YEAR_PATTERNS
        ]
        
        self._fq_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.FISCAL_QUARTER_PATTERNS
        ]
        
        self._date_compiled = [
            re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS
        ]
    
    def extract(self, document: IngestionDocument) -> EnrichedMetadata:
        """Extract metadata using rules."""
        # Get text content
        text = self._get_document_text(document)
        title = self._get_document_title(document)
        
        # Combined text for analysis
        full_text = f"{title}\n{text}"
        
        # Extract each field
        doc_type, doc_type_conf = self._extract_document_type(full_text, document.filename)
        dept, dept_conf = self._extract_department(full_text)
        fy, fy_conf = self._extract_fiscal_year(full_text)
        fq = self._extract_fiscal_quarter(full_text)
        doc_date = self._extract_date(full_text)
        keywords = self._extract_keywords(full_text)
        
        return EnrichedMetadata(
            document_type=doc_type,
            department=dept,
            fiscal_year=fy,
            fiscal_quarter=fq,
            document_date=doc_date,
            keywords=keywords,
            confidence={
                "document_type": doc_type_conf,
                "department": dept_conf,
                "fiscal_year": fy_conf,
            },
            extraction_method="rule_based",
            extracted_at=datetime.utcnow().isoformat(),
        )
    
    def _get_document_text(self, document: IngestionDocument) -> str:
        """Get combined text from document."""
        text_parts = []
        for node in document.get_text_nodes():
            if node.text:
                text_parts.append(node.text)
        return " ".join(text_parts)
    
    def _get_document_title(self, document: IngestionDocument) -> str:
        """Get document title from headings or filename."""
        # Try to find first heading
        for node in document.nodes:
            if node.node_type == NodeType.HEADING and node.heading_level == 1:
                return node.text or ""
        
        # Fall back to filename
        return document.filename
    
    def _extract_document_type(
        self,
        text: str,
        filename: str
    ) -> Tuple[Optional[str], float]:
        """Extract document type."""
        scores: Dict[DocumentType, int] = {}
        
        # Check filename first (higher weight)
        for doc_type, patterns in self._doc_type_compiled.items():
            for pattern in patterns:
                if pattern.search(filename):
                    scores[doc_type] = scores.get(doc_type, 0) + 2
        
        # Check text content
        for doc_type, patterns in self._doc_type_compiled.items():
            for pattern in patterns:
                matches = pattern.findall(text[:5000])  # First 5000 chars
                scores[doc_type] = scores.get(doc_type, 0) + len(matches)
        
        if not scores:
            return None, 0.0
        
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / 5.0, 1.0)  # Normalize
        
        return best_type.value, confidence
    
    def _extract_department(self, text: str) -> Tuple[Optional[str], float]:
        """Extract department."""
        scores: Dict[Department, int] = {}
        
        for dept, patterns in self._dept_compiled.items():
            for pattern in patterns:
                matches = pattern.findall(text[:5000])
                scores[dept] = scores.get(dept, 0) + len(matches)
        
        if not scores:
            return None, 0.0
        
        best_dept = max(scores, key=scores.get)
        confidence = min(scores[best_dept] / 3.0, 1.0)
        
        return best_dept.value, confidence
    
    def _extract_fiscal_year(self, text: str) -> Tuple[Optional[int], float]:
        """Extract fiscal year."""
        for pattern in self._fy_compiled:
            match = pattern.search(text)
            if match:
                year_str = match.group(1)
                year = int(year_str)
                
                # Handle 2-digit years
                if year < 100:
                    year = 2000 + year if year < 50 else 1900 + year
                
                # Validate reasonable range
                current_year = datetime.now().year
                if 2000 <= year <= current_year + 2:
                    return year, 0.9
        
        return None, 0.0
    
    def _extract_fiscal_quarter(self, text: str) -> Optional[str]:
        """Extract fiscal quarter."""
        quarter_map = {
            "first": "Q1",
            "second": "Q2",
            "third": "Q3",
            "fourth": "Q4",
        }
        
        for pattern in self._fq_compiled:
            match = pattern.search(text)
            if match:
                quarter = match.group(1).lower()
                if quarter in quarter_map:
                    return quarter_map[quarter]
                return quarter.upper()
        
        return None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract document date."""
        for pattern in self._date_compiled:
            match = pattern.search(text[:2000])  # Check beginning
            if match:
                # Return raw match - could be normalized in future
                return match.group(1)
        
        return None
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        # Simple frequency-based extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common words
        stopwords = {
            'that', 'this', 'with', 'from', 'have', 'will', 'been',
            'they', 'their', 'there', 'which', 'when', 'what', 'were',
            'would', 'could', 'should', 'about', 'into', 'more', 'some',
            'than', 'them', 'then', 'these', 'other', 'only', 'also',
        }
        
        word_counts: Dict[str, int] = {}
        for word in words:
            if word not in stopwords and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:max_keywords]]


class LLMExtractor(MetadataExtractor):
    """
    LLM-based metadata extraction.
    
    Uses language model for more accurate extraction with strict JSON schema.
    """
    
    EXTRACTION_PROMPT = """Extract the following metadata from the document text. 
Respond with a JSON object containing these fields:
- document_type: one of [report, presentation, memo, contract, invoice, policy, procedure, manual, specification, proposal, meeting_notes, email, letter, form, spreadsheet, other]
- department: one of [engineering, sales, marketing, finance, hr, legal, operations, it, executive, product, support, research, compliance, unknown]
- fiscal_year: integer (e.g., 2024) or null
- fiscal_quarter: string (e.g., "Q1") or null
- project_name: string or null
- client_name: string or null
- keywords: list of up to 5 relevant keywords

Document text (first 3000 characters):
{text}

Respond with ONLY valid JSON, no other text."""

    def __init__(self):
        """Initialize LLM extractor."""
        self.settings = get_settings()
        self._openai_available = self._check_openai()
    
    def _check_openai(self) -> bool:
        """Check if OpenAI is available."""
        try:
            import openai
            return True
        except ImportError:
            return False
    
    def extract(self, document: IngestionDocument) -> EnrichedMetadata:
        """Extract metadata using LLM."""
        if not self._openai_available:
            # Fall back to rule-based
            return RuleBasedExtractor().extract(document)
        
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._extract_async(document)
        )
    
    async def _extract_async(self, document: IngestionDocument) -> EnrichedMetadata:
        """Async extraction using LLM."""
        import openai
        
        # Get text content
        text_parts = []
        for node in document.get_text_nodes():
            if node.text:
                text_parts.append(node.text)
        text = " ".join(text_parts)[:3000]
        
        try:
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": self.EXTRACTION_PROMPT.format(text=text)}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0  # Deterministic
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return EnrichedMetadata(
                document_type=result.get("document_type"),
                department=result.get("department"),
                fiscal_year=result.get("fiscal_year"),
                fiscal_quarter=result.get("fiscal_quarter"),
                project_name=result.get("project_name"),
                client_name=result.get("client_name"),
                keywords=result.get("keywords", []),
                confidence={"overall": 0.85},
                extraction_method="llm",
                extracted_at=datetime.utcnow().isoformat(),
            )
            
        except Exception as e:
            # Fall back to rule-based on error
            metadata = RuleBasedExtractor().extract(document)
            metadata.extraction_method = "rule_based_fallback"
            return metadata


# ============================================================================
# Metadata Enrichment Service
# ============================================================================

class MetadataEnrichmentService:
    """
    Service for metadata enrichment.
    
    Coordinates extraction strategies and applies enriched metadata.
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        extractor: Optional[MetadataExtractor] = None
    ):
        """
        Initialize enrichment service.
        
        Args:
            use_llm: Use LLM-based extraction
            extractor: Custom extractor
        """
        if extractor:
            self.extractor = extractor
        elif use_llm:
            self.extractor = LLMExtractor()
        else:
            self.extractor = RuleBasedExtractor()
    
    def enrich(self, document: IngestionDocument) -> EnrichedMetadata:
        """
        Enrich document with metadata.
        
        Args:
            document: Document to enrich
            
        Returns:
            EnrichedMetadata
        """
        return self.extractor.extract(document)
    
    def apply_enrichment(
        self,
        document: IngestionDocument,
        metadata: EnrichedMetadata
    ) -> IngestionDocument:
        """
        Apply enriched metadata to document.
        
        Args:
            document: Document to update
            metadata: Metadata to apply
            
        Returns:
            Updated document
        """
        document.enriched_metadata = metadata.to_dict()
        return document


# ============================================================================
# Metadata Enrichment Pipeline
# ============================================================================

@register_pipeline("metadata_enrichment")
class MetadataEnrichmentPipeline(BasePipeline):
    """
    Pipeline stage for metadata enrichment.
    
    Extracts structured metadata for RBAC and filtering.
    """
    
    stage_name = "metadata_enrichment"
    
    def __init__(self, use_llm: Optional[bool] = None):
        """
        Initialize metadata enrichment pipeline.
        
        Args:
            use_llm: Override LLM usage from settings
        """
        super().__init__()
        
        settings = get_settings()
        use_llm_flag = use_llm if use_llm is not None else settings.feature_flags.enable_llm_metadata_enrichment
        
        self.service = MetadataEnrichmentService(use_llm=use_llm_flag)
    
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process document to extract metadata.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Document with enriched_metadata populated
        """
        # Extract metadata
        metadata = self.service.enrich(document)
        
        # Apply to document
        document = self.service.apply_enrichment(document, metadata)
        
        # Statistics
        document.metadata["enrichment_stats"] = {
            "extraction_method": metadata.extraction_method,
            "fields_extracted": sum(
                1 for v in [
                    metadata.document_type,
                    metadata.department,
                    metadata.fiscal_year,
                ] if v is not None
            ),
            "keywords_count": len(metadata.keywords),
        }
        
        return document
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """Check if enrichment can be skipped."""
        # Skip if already enriched
        if document.enriched_metadata:
            return True
        
        return document.is_stage_completed(self.stage_name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "DocumentType",
    "Department",
    "EnrichedMetadata",
    "MetadataExtractor",
    "RuleBasedExtractor",
    "LLMExtractor",
    "MetadataEnrichmentService",
    "MetadataEnrichmentPipeline",
]
