"""
MODULE 18 — OBSERVABILITY & AUDIT
=================================
Observability infrastructure for the Enterprise RAG Ingestion Pipeline.

Must track:
- Per-stage latency
- Cost per document
- ACL changes

This module provides comprehensive monitoring, metrics collection,
and audit logging for the ingestion pipeline.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import logging


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up structured logging.
    
    Args:
        level: Log level
        format_string: Custom format string
        json_format: Use JSON formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("enterprise_rag")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        formatter = JsonFormatter()
    else:
        format_str = format_string or (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        formatter = logging.Formatter(format_str)
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


# ============================================================================
# Metrics Types
# ============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    stage_name: str
    document_id: str
    tenant_id: str
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Counts
    items_processed: int = 0
    items_failed: int = 0
    
    # Resource usage
    tokens_used: int = 0
    api_calls: int = 0
    
    # Cost tracking
    estimated_cost_usd: float = 0.0
    
    # Custom metrics
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "tokens_used": self.tokens_used,
            "api_calls": self.api_calls,
            "estimated_cost_usd": self.estimated_cost_usd,
            "custom": self.custom,
        }


@dataclass
class DocumentMetrics:
    """Aggregate metrics for a document ingestion."""
    document_id: str
    tenant_id: str
    
    # Overall timing
    total_duration_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Stage breakdown
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    
    # Resource totals
    total_tokens: int = 0
    total_api_calls: int = 0
    total_cost_usd: float = 0.0
    
    # Document stats
    page_count: int = 0
    node_count: int = 0
    chunk_count: int = 0
    file_size_bytes: int = 0
    
    # Status
    success: bool = False
    error_count: int = 0
    
    def add_stage_metrics(self, metrics: StageMetrics) -> None:
        """Add metrics for a stage."""
        self.stage_metrics[metrics.stage_name] = metrics
        self.total_tokens += metrics.tokens_used
        self.total_api_calls += metrics.api_calls
        self.total_cost_usd += metrics.estimated_cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "total_duration_ms": self.total_duration_ms,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "stage_metrics": {
                k: v.to_dict() for k, v in self.stage_metrics.items()
            },
            "total_tokens": self.total_tokens,
            "total_api_calls": self.total_api_calls,
            "total_cost_usd": self.total_cost_usd,
            "page_count": self.page_count,
            "node_count": self.node_count,
            "chunk_count": self.chunk_count,
            "file_size_bytes": self.file_size_bytes,
            "success": self.success,
            "error_count": self.error_count,
        }


# ============================================================================
# Metrics Collector
# ============================================================================

class MetricsCollector:
    """
    Collects and stores metrics.
    
    In production, this would push to:
    - Prometheus
    - DataDog
    - CloudWatch
    - Custom time-series DB
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            buffer_size: Maximum metrics to buffer in memory
        """
        self.buffer_size = buffer_size
        self._metrics: List[MetricPoint] = []
        self._document_metrics: Dict[str, DocumentMetrics] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
    
    # =========================================================================
    # Counter Methods
    # =========================================================================
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value
        
        self._record(MetricPoint(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {}
        ))
    
    def decrement(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Decrement a counter."""
        self.increment(name, -value, tags)
    
    # =========================================================================
    # Gauge Methods
    # =========================================================================
    
    def set_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self._gauges[key] = value
        
        self._record(MetricPoint(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {}
        ))
    
    # =========================================================================
    # Timer Methods
    # =========================================================================
    
    @contextmanager
    def timer(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Context manager for timing operations.
        
        Usage:
            with metrics.timer("stage.duration", {"stage": "embedding"}):
                do_work()
        """
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self._record(MetricPoint(
                name=name,
                value=duration_ms,
                metric_type=MetricType.TIMER,
                tags=tags or {}
            ))
    
    def record_timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timing value directly."""
        self._record(MetricPoint(
            name=name,
            value=duration_ms,
            metric_type=MetricType.TIMER,
            tags=tags or {}
        ))
    
    # =========================================================================
    # Document Metrics
    # =========================================================================
    
    def start_document(
        self,
        document_id: str,
        tenant_id: str
    ) -> DocumentMetrics:
        """Start tracking metrics for a document."""
        metrics = DocumentMetrics(
            document_id=document_id,
            tenant_id=tenant_id,
            start_time=datetime.utcnow()
        )
        self._document_metrics[document_id] = metrics
        
        self.increment(
            "documents.started",
            tags={"tenant_id": tenant_id}
        )
        
        return metrics
    
    def end_document(
        self,
        document_id: str,
        success: bool = True
    ) -> Optional[DocumentMetrics]:
        """End tracking metrics for a document."""
        metrics = self._document_metrics.get(document_id)
        if not metrics:
            return None
        
        metrics.end_time = datetime.utcnow()
        metrics.success = success
        
        if metrics.start_time:
            metrics.total_duration_ms = (
                metrics.end_time - metrics.start_time
            ).total_seconds() * 1000
        
        # Record aggregate metrics
        status = "success" if success else "failed"
        self.increment(
            f"documents.{status}",
            tags={"tenant_id": metrics.tenant_id}
        )
        
        self.record_timing(
            "documents.duration_ms",
            metrics.total_duration_ms,
            tags={"tenant_id": metrics.tenant_id, "status": status}
        )
        
        if metrics.total_cost_usd > 0:
            self._record(MetricPoint(
                name="documents.cost_usd",
                value=metrics.total_cost_usd,
                metric_type=MetricType.GAUGE,
                tags={"tenant_id": metrics.tenant_id}
            ))
        
        return metrics
    
    def get_document_metrics(
        self,
        document_id: str
    ) -> Optional[DocumentMetrics]:
        """Get metrics for a document."""
        return self._document_metrics.get(document_id)
    
    def record_stage(
        self,
        document_id: str,
        stage_metrics: StageMetrics
    ) -> None:
        """Record metrics for a pipeline stage."""
        doc_metrics = self._document_metrics.get(document_id)
        if doc_metrics:
            doc_metrics.add_stage_metrics(stage_metrics)
        
        # Record stage-level metrics
        self.record_timing(
            "stage.duration_ms",
            stage_metrics.duration_ms,
            tags={
                "stage": stage_metrics.stage_name,
                "tenant_id": stage_metrics.tenant_id
            }
        )
        
        if stage_metrics.items_processed > 0:
            self.increment(
                "stage.items_processed",
                stage_metrics.items_processed,
                tags={"stage": stage_metrics.stage_name}
            )
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _make_key(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a unique key for a metric."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"
    
    def _record(self, point: MetricPoint) -> None:
        """Record a metric point."""
        self._metrics.append(point)
        
        # Trim buffer if needed
        if len(self._metrics) > self.buffer_size:
            self._metrics = self._metrics[-self.buffer_size:]
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_metrics(
        self,
        name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[MetricPoint]:
        """
        Query recorded metrics.
        
        Args:
            name: Filter by metric name
            metric_type: Filter by type
            since: Filter by timestamp
            limit: Maximum results
            
        Returns:
            List of MetricPoint
        """
        results = self._metrics
        
        if name:
            results = [m for m in results if m.name == name]
        
        if metric_type:
            results = [m for m in results if m.metric_type == metric_type]
        
        if since:
            results = [m for m in results if m.timestamp >= since]
        
        return results[-limit:]
    
    def get_counter(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)
    
    def get_gauge(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> float:
        """Get current gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key, 0)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        return {
            "total_metrics": len(self._metrics),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "documents_tracked": len(self._document_metrics),
        }


# ============================================================================
# ACL Audit
# ============================================================================

@dataclass
class ACLAuditEntry:
    """Audit entry for ACL changes."""
    document_id: str
    tenant_id: str
    user_id: str
    action: str  # "create", "update", "delete"
    timestamp: datetime
    
    # Previous values
    old_roles: Optional[List[str]] = None
    old_users: Optional[List[str]] = None
    old_classification: Optional[str] = None
    
    # New values
    new_roles: Optional[List[str]] = None
    new_users: Optional[List[str]] = None
    new_classification: Optional[str] = None
    
    # Metadata
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "old_roles": self.old_roles,
            "old_users": self.old_users,
            "old_classification": self.old_classification,
            "new_roles": self.new_roles,
            "new_users": self.new_users,
            "new_classification": self.new_classification,
            "reason": self.reason,
        }


class ACLAuditor:
    """
    Audits ACL changes for compliance.
    
    Tracks all changes to document access control for:
    - Compliance reporting
    - Security auditing
    - Change history
    """
    
    def __init__(self, max_entries: int = 100000):
        """
        Initialize ACL auditor.
        
        Args:
            max_entries: Maximum entries to retain
        """
        self.max_entries = max_entries
        self._entries: List[ACLAuditEntry] = []
        self._logger = logging.getLogger("enterprise_rag.acl_audit")
    
    def log_change(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        action: str,
        old_roles: Optional[List[str]] = None,
        old_users: Optional[List[str]] = None,
        old_classification: Optional[str] = None,
        new_roles: Optional[List[str]] = None,
        new_users: Optional[List[str]] = None,
        new_classification: Optional[str] = None,
        reason: Optional[str] = None
    ) -> ACLAuditEntry:
        """
        Log an ACL change.
        
        Args:
            document_id: Document ID
            tenant_id: Tenant ID
            user_id: User making the change
            action: Type of action
            old_*: Previous ACL values
            new_*: New ACL values
            reason: Reason for change
            
        Returns:
            ACLAuditEntry
        """
        entry = ACLAuditEntry(
            document_id=document_id,
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            timestamp=datetime.utcnow(),
            old_roles=old_roles,
            old_users=old_users,
            old_classification=old_classification,
            new_roles=new_roles,
            new_users=new_users,
            new_classification=new_classification,
            reason=reason,
        )
        
        self._entries.append(entry)
        
        # Trim if needed
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        
        # Log
        self._logger.info(
            f"ACL change: {action} on {document_id} by {user_id}",
            extra={"extra_data": entry.to_dict()}
        )
        
        return entry
    
    def get_history(
        self,
        document_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ACLAuditEntry]:
        """
        Get ACL audit history.
        
        Args:
            document_id: Filter by document
            tenant_id: Filter by tenant
            user_id: Filter by user
            since: Filter by timestamp
            limit: Maximum results
            
        Returns:
            List of ACLAuditEntry
        """
        results = self._entries
        
        if document_id:
            results = [e for e in results if e.document_id == document_id]
        
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if since:
            results = [e for e in results if e.timestamp >= since]
        
        # Sort by timestamp descending
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        
        return results[:limit]


# ============================================================================
# Cost Tracker
# ============================================================================

class CostTracker:
    """
    Tracks costs for document processing.
    
    Estimates costs for:
    - Embedding API calls
    - Vision API calls
    - LLM API calls
    - Storage usage
    """
    
    # Cost estimates (USD per unit)
    COST_RATES = {
        # OpenAI embeddings
        "embedding_text-embedding-3-small": 0.00002 / 1000,  # per token
        "embedding_text-embedding-3-large": 0.00013 / 1000,
        "embedding_text-embedding-ada-002": 0.0001 / 1000,
        
        # OpenAI vision
        "vision_gpt-4-vision": 0.01 / 1000,  # per token (approx)
        
        # LLM
        "llm_gpt-4o-mini_input": 0.00015 / 1000,
        "llm_gpt-4o-mini_output": 0.0006 / 1000,
        "llm_gpt-4o_input": 0.005 / 1000,
        "llm_gpt-4o_output": 0.015 / 1000,
        
        # Storage (per GB per month, prorated to per byte)
        "storage_s3": 0.023 / (1024 ** 3),
        "storage_qdrant": 0.05 / (1024 ** 3),
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self._costs: Dict[str, float] = {}
    
    def track(
        self,
        document_id: str,
        cost_type: str,
        units: float,
        rate_key: Optional[str] = None
    ) -> float:
        """
        Track a cost.
        
        Args:
            document_id: Document ID
            cost_type: Type of cost
            units: Number of units (tokens, bytes, etc.)
            rate_key: Key in COST_RATES (default: cost_type)
            
        Returns:
            Estimated cost in USD
        """
        rate = self.COST_RATES.get(rate_key or cost_type, 0)
        cost = units * rate
        
        key = f"{document_id}:{cost_type}"
        self._costs[key] = self._costs.get(key, 0) + cost
        
        return cost
    
    def get_document_cost(self, document_id: str) -> float:
        """Get total cost for a document."""
        total = 0.0
        prefix = f"{document_id}:"
        
        for key, cost in self._costs.items():
            if key.startswith(prefix):
                total += cost
        
        return total
    
    def get_cost_breakdown(self, document_id: str) -> Dict[str, float]:
        """Get cost breakdown for a document."""
        breakdown = {}
        prefix = f"{document_id}:"
        
        for key, cost in self._costs.items():
            if key.startswith(prefix):
                cost_type = key[len(prefix):]
                breakdown[cost_type] = cost
        
        return breakdown


# ============================================================================
# Global Instances
# ============================================================================

# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None

# Global ACL auditor
_acl_auditor: Optional[ACLAuditor] = None

# Global cost tracker
_cost_tracker: Optional[CostTracker] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_acl_auditor() -> ACLAuditor:
    """Get the global ACL auditor."""
    global _acl_auditor
    if _acl_auditor is None:
        _acl_auditor = ACLAuditor()
    return _acl_auditor


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# ============================================================================
# Decorators
# ============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def timed(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to time function execution.
    
    Usage:
        @timed("my_function.duration")
        def my_function():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.timer(metric_name, tags):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.record_timing(metric_name, duration_ms, tags)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


def counted(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.
    
    Usage:
        @counted("my_function.calls")
        def my_function():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            metrics.increment(metric_name, tags=tags)
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics()
            metrics.increment(metric_name, tags=tags)
            return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Logging
    "setup_logging",
    "JsonFormatter",
    
    # Metrics
    "MetricType",
    "MetricPoint",
    "StageMetrics",
    "DocumentMetrics",
    "MetricsCollector",
    
    # ACL Audit
    "ACLAuditEntry",
    "ACLAuditor",
    
    # Cost Tracking
    "CostTracker",
    
    # Global accessors
    "get_metrics",
    "get_acl_auditor",
    "get_cost_tracker",
    
    # Decorators
    "timed",
    "counted",
]
