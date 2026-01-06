"""
Test suite for Modules 16, 17, 18.

Module 16: Metadata Enrichment
Module 17: Delete & Update Handlers
Module 18: Observability & Audit

Run with: pytest tests/test_modules_16_18.py -v
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestModule16MetadataEnrichment:
    """Tests for Module 16 - Metadata Enrichment."""
    
    def test_document_type_enum(self):
        """Test DocumentType enumeration."""
        from enrichment.metadata import DocumentType
        
        assert DocumentType.REPORT.value == "report"
        assert DocumentType.CONTRACT.value == "contract"
        assert DocumentType.INVOICE.value == "invoice"
    
    def test_department_enum(self):
        """Test Department enumeration."""
        from enrichment.metadata import Department
        
        assert Department.ENGINEERING.value == "engineering"
        assert Department.FINANCE.value == "finance"
        assert Department.HR.value == "hr"
    
    def test_enriched_metadata_dataclass(self):
        """Test EnrichedMetadata dataclass."""
        from enrichment.metadata import EnrichedMetadata
        
        metadata = EnrichedMetadata(
            document_type="report",
            department="engineering",
            fiscal_year=2024,
            fiscal_quarter="Q1",
            keywords=["ai", "machine learning"]
        )
        
        assert metadata.document_type == "report"
        assert metadata.fiscal_year == 2024
        assert len(metadata.keywords) == 2
    
    def test_enriched_metadata_serialization(self):
        """Test EnrichedMetadata to_dict/from_dict."""
        from enrichment.metadata import EnrichedMetadata
        
        metadata = EnrichedMetadata(
            document_type="memo",
            department="hr",
            fiscal_year=2024
        )
        
        data = metadata.to_dict()
        restored = EnrichedMetadata.from_dict(data)
        
        assert restored.document_type == "memo"
        assert restored.department == "hr"
        assert restored.fiscal_year == 2024
    
    def test_rule_based_extractor_document_type(self):
        """Test RuleBasedExtractor document type detection."""
        from enrichment.metadata import RuleBasedExtractor
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        extractor = RuleBasedExtractor()
        
        doc = IngestionDocument(
            document_id="doc",
            tenant_id="tenant",
            filename="Q3_Sales_Report.pdf"
        )
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="This quarterly report summarizes our sales performance."
        ))
        
        metadata = extractor.extract(doc)
        
        assert metadata.document_type == "report"
    
    def test_rule_based_extractor_department(self):
        """Test RuleBasedExtractor department detection."""
        from enrichment.metadata import RuleBasedExtractor
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        extractor = RuleBasedExtractor()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="The engineering team has completed the development sprint."
        ))
        
        metadata = extractor.extract(doc)
        
        assert metadata.department == "engineering"
    
    def test_rule_based_extractor_fiscal_year(self):
        """Test RuleBasedExtractor fiscal year detection."""
        from enrichment.metadata import RuleBasedExtractor
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        extractor = RuleBasedExtractor()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="FY2024 budget allocation for Q3."
        ))
        
        metadata = extractor.extract(doc)
        
        assert metadata.fiscal_year == 2024
    
    def test_rule_based_extractor_fiscal_quarter(self):
        """Test RuleBasedExtractor fiscal quarter detection."""
        from enrichment.metadata import RuleBasedExtractor
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        extractor = RuleBasedExtractor()
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="Q3 2024 performance review"
        ))
        
        metadata = extractor.extract(doc)
        
        assert metadata.fiscal_quarter in ("Q3", "q3")
    
    def test_metadata_enrichment_service(self):
        """Test MetadataEnrichmentService."""
        from enrichment.metadata import MetadataEnrichmentService
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        service = MetadataEnrichmentService(use_llm=False)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="Finance department budget report for FY2024."
        ))
        
        metadata = service.enrich(doc)
        
        assert metadata.extraction_method == "rule_based"
        assert metadata.extracted_at is not None
    
    @pytest.mark.asyncio
    async def test_metadata_enrichment_pipeline(self):
        """Test MetadataEnrichmentPipeline.process()."""
        from enrichment.metadata import MetadataEnrichmentPipeline
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        pipeline = MetadataEnrichmentPipeline(use_llm=False)
        
        doc = IngestionDocument(document_id="doc", tenant_id="tenant")
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="Legal contract review document."
        ))
        
        result = await pipeline.process(doc)
        
        assert result.enriched_metadata is not None
        assert "enrichment_stats" in result.metadata


class TestModule17Lifecycle:
    """Tests for Module 17 - Delete & Update Handlers."""
    
    def test_lifecycle_action_enum(self):
        """Test LifecycleAction enumeration."""
        from lifecycle.document_manager import LifecycleAction
        
        assert LifecycleAction.CREATE.value == "create"
        assert LifecycleAction.SOFT_DELETE.value == "soft_delete"
        assert LifecycleAction.HARD_DELETE.value == "hard_delete"
    
    def test_lifecycle_status_enum(self):
        """Test LifecycleStatus enumeration."""
        from lifecycle.document_manager import LifecycleStatus
        
        assert LifecycleStatus.SUCCESS.value == "success"
        assert LifecycleStatus.PARTIAL.value == "partial"
        assert LifecycleStatus.FAILED.value == "failed"
    
    def test_lifecycle_result_dataclass(self):
        """Test LifecycleResult dataclass."""
        from lifecycle.document_manager import (
            LifecycleResult, LifecycleAction, LifecycleStatus
        )
        
        result = LifecycleResult(
            action=LifecycleAction.SOFT_DELETE,
            status=LifecycleStatus.SUCCESS,
            document_id="doc-123",
            tenant_id="tenant-456",
            postgres_affected=1,
            vectors_deleted=10
        )
        
        assert result.action == LifecycleAction.SOFT_DELETE
        assert result.postgres_affected == 1
        assert result.vectors_deleted == 10
    
    def test_lifecycle_result_to_dict(self):
        """Test LifecycleResult serialization."""
        from lifecycle.document_manager import (
            LifecycleResult, LifecycleAction, LifecycleStatus
        )
        
        result = LifecycleResult(
            action=LifecycleAction.HARD_DELETE,
            status=LifecycleStatus.PARTIAL,
            document_id="doc",
            tenant_id="tenant",
            errors=[{"component": "s3", "error": "timeout"}]
        )
        
        data = result.to_dict()
        
        assert data["action"] == "hard_delete"
        assert data["status"] == "partial"
        assert len(data["errors"]) == 1
    
    def test_audit_entry_dataclass(self):
        """Test AuditEntry dataclass."""
        from lifecycle.document_manager import AuditEntry, LifecycleAction
        
        entry = AuditEntry(
            action=LifecycleAction.UPDATE,
            document_id="doc",
            tenant_id="tenant",
            user_id="user",
            version=2,
            timestamp=datetime.utcnow().isoformat()
        )
        
        assert entry.action == LifecycleAction.UPDATE
        assert entry.version == 2
    
    @pytest.mark.asyncio
    async def test_document_manager_instantiation(self):
        """Test DocumentManager instantiation."""
        from lifecycle.document_manager import DocumentManager
        from unittest.mock import AsyncMock
        
        mock_session = AsyncMock()
        mock_s3 = AsyncMock()
        mock_vector = AsyncMock()
        
        manager = DocumentManager(
            db_session=mock_session,
            s3_storage=mock_s3,
            vector_service=mock_vector
        )
        
        assert manager.db == mock_session
        assert manager.s3 == mock_s3


class TestModule18Observability:
    """Tests for Module 18 - Observability & Audit."""
    
    def test_metric_type_enum(self):
        """Test MetricType enumeration."""
        from monitoring.metrics import MetricType
        
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
    
    def test_metric_point_dataclass(self):
        """Test MetricPoint dataclass."""
        from monitoring.metrics import MetricPoint, MetricType
        
        point = MetricPoint(
            name="test.metric",
            value=42.0,
            metric_type=MetricType.COUNTER,
            tags={"env": "test"}
        )
        
        assert point.name == "test.metric"
        assert point.value == 42.0
        assert point.tags["env"] == "test"
    
    def test_metrics_collector_increment(self):
        """Test MetricsCollector.increment()."""
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.increment("requests.count")
        collector.increment("requests.count")
        collector.increment("requests.count", value=3)
        
        count = collector.get_counter("requests.count")
        assert count == 5
    
    def test_metrics_collector_gauge(self):
        """Test MetricsCollector.set_gauge()."""
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.set_gauge("queue.size", 100)
        collector.set_gauge("queue.size", 150)
        
        value = collector.get_gauge("queue.size")
        assert value == 150
    
    def test_metrics_collector_timer_context(self):
        """Test MetricsCollector.timer() context manager."""
        import time
        from monitoring.metrics import MetricsCollector, MetricType
        
        collector = MetricsCollector()
        
        with collector.timer("operation.duration"):
            time.sleep(0.01)  # 10ms
        
        metrics = collector.get_metrics(
            name="operation.duration",
            metric_type=MetricType.TIMER
        )
        
        assert len(metrics) == 1
        assert metrics[0].value >= 10  # At least 10ms
    
    def test_metrics_collector_document_tracking(self):
        """Test MetricsCollector document tracking."""
        from monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector()
        
        # Start tracking
        doc_metrics = collector.start_document("doc-1", "tenant-1")
        assert doc_metrics.document_id == "doc-1"
        
        # End tracking
        result = collector.end_document("doc-1", success=True)
        assert result.success is True
        assert result.total_duration_ms > 0
    
    def test_stage_metrics_dataclass(self):
        """Test StageMetrics dataclass."""
        from monitoring.metrics import StageMetrics
        
        metrics = StageMetrics(
            stage_name="embedding",
            document_id="doc-1",
            tenant_id="tenant-1",
            duration_ms=150.5,
            items_processed=10,
            tokens_used=5000,
            estimated_cost_usd=0.001
        )
        
        assert metrics.stage_name == "embedding"
        assert metrics.tokens_used == 5000
    
    def test_document_metrics_add_stage(self):
        """Test DocumentMetrics.add_stage_metrics()."""
        from monitoring.metrics import DocumentMetrics, StageMetrics
        
        doc_metrics = DocumentMetrics(
            document_id="doc-1",
            tenant_id="tenant-1"
        )
        
        stage1 = StageMetrics(
            stage_name="text_pipeline",
            document_id="doc-1",
            tenant_id="tenant-1",
            tokens_used=1000,
            estimated_cost_usd=0.001
        )
        
        stage2 = StageMetrics(
            stage_name="embedding",
            document_id="doc-1",
            tenant_id="tenant-1",
            tokens_used=2000,
            estimated_cost_usd=0.002
        )
        
        doc_metrics.add_stage_metrics(stage1)
        doc_metrics.add_stage_metrics(stage2)
        
        assert doc_metrics.total_tokens == 3000
        assert doc_metrics.total_cost_usd == 0.003
    
    def test_acl_auditor_log_change(self):
        """Test ACLAuditor.log_change()."""
        from monitoring.metrics import ACLAuditor
        
        auditor = ACLAuditor()
        
        entry = auditor.log_change(
            document_id="doc-1",
            tenant_id="tenant-1",
            user_id="user-1",
            action="update",
            old_roles=["viewer"],
            new_roles=["viewer", "editor"],
            reason="Promoted user access"
        )
        
        assert entry.action == "update"
        assert entry.old_roles == ["viewer"]
        assert entry.new_roles == ["viewer", "editor"]
    
    def test_acl_auditor_get_history(self):
        """Test ACLAuditor.get_history()."""
        from monitoring.metrics import ACLAuditor
        
        auditor = ACLAuditor()
        
        # Log multiple changes
        auditor.log_change("doc-1", "tenant", "user", "create")
        auditor.log_change("doc-1", "tenant", "user", "update")
        auditor.log_change("doc-2", "tenant", "user", "create")
        
        # Get all
        all_history = auditor.get_history()
        assert len(all_history) == 3
        
        # Filter by document
        doc1_history = auditor.get_history(document_id="doc-1")
        assert len(doc1_history) == 2
    
    def test_cost_tracker_track(self):
        """Test CostTracker.track()."""
        from monitoring.metrics import CostTracker
        
        tracker = CostTracker()
        
        # Track embedding cost
        cost = tracker.track(
            document_id="doc-1",
            cost_type="embedding",
            units=1000,  # tokens
            rate_key="embedding_text-embedding-3-small"
        )
        
        assert cost > 0
    
    def test_cost_tracker_document_cost(self):
        """Test CostTracker.get_document_cost()."""
        from monitoring.metrics import CostTracker
        
        tracker = CostTracker()
        
        tracker.track("doc-1", "embedding", 1000, "embedding_text-embedding-3-small")
        tracker.track("doc-1", "storage", 1024 * 1024, "storage_s3")  # 1MB
        
        total = tracker.get_document_cost("doc-1")
        assert total > 0
        
        breakdown = tracker.get_cost_breakdown("doc-1")
        assert "embedding" in breakdown
        assert "storage" in breakdown
    
    def test_timed_decorator(self):
        """Test @timed decorator."""
        from monitoring.metrics import timed, get_metrics, MetricType
        
        @timed("test.function.duration")
        def slow_function():
            import time
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        assert result == "done"
        
        metrics = get_metrics().get_metrics(
            name="test.function.duration",
            metric_type=MetricType.TIMER
        )
        assert len(metrics) >= 1
    
    def test_counted_decorator(self):
        """Test @counted decorator."""
        from monitoring.metrics import counted, get_metrics
        
        # Reset for clean test
        collector = get_metrics()
        
        @counted("test.function.calls")
        def counted_function():
            return "called"
        
        counted_function()
        counted_function()
        counted_function()
        
        count = collector.get_counter("test.function.calls")
        assert count >= 3
    
    def test_setup_logging(self):
        """Test setup_logging function."""
        from monitoring.metrics import setup_logging
        
        logger = setup_logging(level="DEBUG", json_format=False)
        
        assert logger.name == "enterprise_rag"
        assert logger.level == 10  # DEBUG
    
    def test_json_formatter(self):
        """Test JsonFormatter."""
        import logging
        from monitoring.metrics import JsonFormatter
        import json
        
        formatter = JsonFormatter()
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        data = json.loads(output)
        
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert "timestamp" in data


class TestModule16To18Integration:
    """Integration tests for Modules 16-18."""
    
    @pytest.mark.asyncio
    async def test_enrichment_with_metrics(self):
        """Test metadata enrichment with metrics tracking."""
        from enrichment.metadata import MetadataEnrichmentPipeline
        from monitoring.metrics import get_metrics
        from schema.intermediate_representation import (
            IngestionDocument, ContentNode, NodeType
        )
        
        metrics = get_metrics()
        
        # Start document tracking
        doc_metrics = metrics.start_document("doc-1", "tenant-1")
        
        pipeline = MetadataEnrichmentPipeline(use_llm=False)
        
        doc = IngestionDocument(
            document_id="doc-1",
            tenant_id="tenant-1"
        )
        doc.add_node(ContentNode(
            node_type=NodeType.TEXT,
            text="Engineering quarterly report FY2024 Q3"
        ))
        
        result = await pipeline.process(doc)
        
        # End tracking
        final_metrics = metrics.end_document("doc-1", success=True)
        
        assert result.enriched_metadata is not None
        assert final_metrics.success is True
    
    def test_acl_audit_with_lifecycle(self):
        """Test ACL audit integration concept."""
        from monitoring.metrics import get_acl_auditor
        from lifecycle.document_manager import LifecycleAction
        
        auditor = get_acl_auditor()
        
        # Simulate document creation with ACL
        auditor.log_change(
            document_id="doc-1",
            tenant_id="tenant-1",
            user_id="admin",
            action="create",
            new_roles=["viewer"],
            new_users=["user-1"],
            new_classification="INTERNAL"
        )
        
        # Simulate ACL update
        auditor.log_change(
            document_id="doc-1",
            tenant_id="tenant-1",
            user_id="admin",
            action="update",
            old_roles=["viewer"],
            old_classification="INTERNAL",
            new_roles=["viewer", "editor"],
            new_classification="CONFIDENTIAL",
            reason="Security upgrade"
        )
        
        history = auditor.get_history(document_id="doc-1")
        
        assert len(history) == 2
        assert history[0]["action"] == "update"  # Most recent first


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
