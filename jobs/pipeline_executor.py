"""
MODULE 8 — PIPELINE DAG EXECUTOR (CORE ENGINE)
==============================================
The core execution engine for the Enterprise RAG Ingestion Pipeline.

Responsibilities:
- Execute ordered pipelines
- Pass IR (IngestionDocument) between stages
- Persist current_stage for resume capability
- Fail fast with logging
- Handle heartbeat updates during execution

All pipelines implement:
    async process(doc: IngestionDocument) -> IngestionDocument

The executor owns orchestration - pipelines never call each other.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from schema.intermediate_representation import (
    IngestionDocument,
    ProcessingStatus,
    RBACContext,
    create_ingestion_document,
)
from storage.postgres import Document, IngestionJob
from storage.s3 import S3Storage, get_s3_storage
from pipelines.base import BasePipeline, PipelineError, PipelineSkipped


# ============================================================================
# Execution Result
# ============================================================================

@dataclass
class StageResult:
    """Result of a single stage execution."""
    stage: str
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    skipped: bool = False


@dataclass
class ExecutionResult:
    """Result of full pipeline execution."""
    job_id: str
    document_id: str
    success: bool
    total_duration_seconds: float
    stages_executed: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    stage_results: List[StageResult] = field(default_factory=list)
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "document_id": self.document_id,
            "success": self.success,
            "total_duration_seconds": self.total_duration_seconds,
            "stages_executed": self.stages_executed,
            "stages_skipped": self.stages_skipped,
            "stage_results": [
                {
                    "stage": r.stage,
                    "success": r.success,
                    "duration_seconds": r.duration_seconds,
                    "metrics": r.metrics,
                    "error": r.error,
                    "skipped": r.skipped,
                }
                for r in self.stage_results
            ],
            "error": self.error,
            "error_stage": self.error_stage,
        }


# ============================================================================
# Pipeline Registry
# ============================================================================

class PipelineRegistry:
    """
    Registry of available pipeline stages.
    
    Maps stage names to pipeline classes.
    """
    
    _pipelines: Dict[str, Type[BasePipeline]] = {}
    _instances: Dict[str, BasePipeline] = {}
    
    @classmethod
    def register(cls, stage_name: str, pipeline_class: Type[BasePipeline]) -> None:
        """
        Register a pipeline class for a stage.
        
        Args:
            stage_name: Name of the stage
            pipeline_class: Pipeline class to register
        """
        cls._pipelines[stage_name] = pipeline_class
    
    @classmethod
    def get(cls, stage_name: str) -> Optional[BasePipeline]:
        """
        Get a pipeline instance for a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Pipeline instance or None
        """
        if stage_name not in cls._instances:
            pipeline_class = cls._pipelines.get(stage_name)
            if pipeline_class:
                cls._instances[stage_name] = pipeline_class()
        return cls._instances.get(stage_name)
    
    @classmethod
    def get_all_stages(cls) -> List[str]:
        """Get all registered stage names."""
        return list(cls._pipelines.keys())
    
    @classmethod
    def is_registered(cls, stage_name: str) -> bool:
        """Check if a stage is registered."""
        return stage_name in cls._pipelines
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._pipelines.clear()
        cls._instances.clear()


def register_pipeline(stage_name: str):
    """
    Decorator to register a pipeline class.
    
    Usage:
        @register_pipeline("text_pipeline")
        class TextPipeline(BasePipeline):
            ...
    """
    def decorator(cls: Type[BasePipeline]) -> Type[BasePipeline]:
        cls.stage_name = stage_name
        PipelineRegistry.register(stage_name, cls)
        return cls
    return decorator


# ============================================================================
# Pipeline Executor
# ============================================================================

class PipelineExecutor:
    """
    Executes pipeline stages for document ingestion.
    
    The executor:
    1. Creates IngestionDocument (IR) from database Document
    2. Executes each stage in order
    3. Passes IR between stages
    4. Updates job status after each stage
    5. Handles errors and resume
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        job_manager: Any,
        s3_storage: Optional[S3Storage] = None,
        heartbeat_interval: int = 30
    ):
        """
        Initialize pipeline executor.
        
        Args:
            db_session: Database session
            job_manager: Job manager instance
            s3_storage: S3 storage instance (optional)
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.db = db_session
        self.job_manager = job_manager
        self.s3 = s3_storage or get_s3_storage()
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._stop_heartbeat = False
    
    async def execute(
        self,
        job: IngestionJob,
        document: Document
    ) -> ExecutionResult:
        """
        Execute the full pipeline for a document.
        
        Args:
            job: Ingestion job with pipeline plan
            document: Database document record
            
        Returns:
            ExecutionResult with status and metrics
        """
        start_time = time.time()
        stages = job.pipeline_plan.get("stages", [])
        completed_stages = job.completed_stages or []
        
        result = ExecutionResult(
            job_id=job.job_id,
            document_id=document.document_id,
            success=False,
            total_duration_seconds=0,
        )
        
        try:
            # Start heartbeat
            await self._start_heartbeat(job.job_id)
            
            # Create IR from document
            ir_doc = await self._create_ir_document(document)
            
            # Mark completed stages in IR
            ir_doc.completed_stages = list(completed_stages)
            
            # Execute each stage
            for stage in stages:
                # Skip already completed stages (for resume)
                if stage in completed_stages:
                    result.stages_skipped.append(stage)
                    result.stage_results.append(StageResult(
                        stage=stage,
                        success=True,
                        duration_seconds=0,
                        skipped=True
                    ))
                    continue
                
                # Execute stage
                stage_result = await self._execute_stage(
                    stage=stage,
                    ir_doc=ir_doc,
                    job=job
                )
                
                result.stage_results.append(stage_result)
                
                if stage_result.success:
                    result.stages_executed.append(stage)
                else:
                    # Stage failed - stop execution
                    result.error = stage_result.error
                    result.error_stage = stage
                    break
            
            # Check if all stages completed
            if not result.error:
                result.success = True
                await self.job_manager.complete_job(
                    job.job_id,
                    metrics={"stages_executed": len(result.stages_executed)}
                )
            else:
                await self.job_manager.fail_job(
                    job.job_id,
                    error=result.error,
                    stage=result.error_stage
                )
            
        except Exception as e:
            result.error = str(e)
            result.error_stage = job.current_stage
            await self.job_manager.fail_job(
                job.job_id,
                error=str(e),
                stage=job.current_stage,
                details={"exception_type": type(e).__name__}
            )
        
        finally:
            # Stop heartbeat
            await self._stop_heartbeat_task()
            
            # Record total duration
            result.total_duration_seconds = time.time() - start_time
        
        return result
    
    async def _execute_stage(
        self,
        stage: str,
        ir_doc: IngestionDocument,
        job: IngestionJob
    ) -> StageResult:
        """
        Execute a single pipeline stage.
        
        Args:
            stage: Stage name
            ir_doc: IngestionDocument (IR)
            job: Ingestion job
            
        Returns:
            StageResult
        """
        start_time = time.time()
        
        # Update job status
        await self.job_manager.start_stage(job.job_id, stage)
        
        try:
            # Get pipeline for stage
            pipeline = PipelineRegistry.get(stage)
            
            if pipeline is None:
                # No pipeline registered - skip with warning
                duration = time.time() - start_time
                await self.job_manager.complete_stage(
                    job.job_id, 
                    stage,
                    {"skipped": True, "reason": "no_pipeline_registered"}
                )
                return StageResult(
                    stage=stage,
                    success=True,
                    duration_seconds=duration,
                    skipped=True,
                    metrics={"reason": "no_pipeline_registered"}
                )
            
            # Check if pipeline can be skipped
            if pipeline.can_skip(ir_doc):
                duration = time.time() - start_time
                await self.job_manager.complete_stage(
                    job.job_id,
                    stage,
                    {"skipped": True, "reason": "already_completed"}
                )
                return StageResult(
                    stage=stage,
                    success=True,
                    duration_seconds=duration,
                    skipped=True
                )
            
            # Execute pipeline
            ir_doc = await pipeline.execute(ir_doc)
            
            duration = time.time() - start_time
            metrics = pipeline.get_metrics()
            
            # Update job status
            await self.job_manager.complete_stage(job.job_id, stage, metrics)
            
            return StageResult(
                stage=stage,
                success=True,
                duration_seconds=duration,
                metrics=metrics
            )
            
        except PipelineSkipped as e:
            duration = time.time() - start_time
            await self.job_manager.complete_stage(
                job.job_id,
                stage,
                {"skipped": True, "reason": e.reason}
            )
            return StageResult(
                stage=stage,
                success=True,
                duration_seconds=duration,
                skipped=True,
                metrics={"skip_reason": e.reason}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=duration,
                error=str(e)
            )
    
    async def _create_ir_document(self, document: Document) -> IngestionDocument:
        """
        Create IngestionDocument (IR) from database Document.
        
        Args:
            document: Database document record
            
        Returns:
            IngestionDocument
        """
        # Download raw content from S3
        raw_content = None
        try:
            raw_content = await self.s3.download(document.s3_raw_path)
        except Exception as e:
            # Content might not be needed for all stages
            pass
        
        # Create RBAC context
        rbac = RBACContext(
            tenant_id=document.tenant_id,
            allowed_roles=document.allowed_roles.get("roles", []),
            allowed_users=document.allowed_users.get("users", []),
            classification=document.classification
        )
        
        # Create IR document
        ir_doc = create_ingestion_document(
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            filename=document.filename,
            mime_type=document.mime_type,
            content=raw_content,
            s3_raw_path=document.s3_raw_path,
            rbac=rbac,
            version=document.document_version
        )
        
        # Set derived prefix
        ir_doc.s3_derived_prefix = document.s3_derived_prefix or ""
        
        # Copy enriched metadata if exists
        if document.enriched_metadata:
            ir_doc.enriched_metadata = dict(document.enriched_metadata)
        
        return ir_doc
    
    async def _start_heartbeat(self, job_id: str) -> None:
        """Start heartbeat background task."""
        self._stop_heartbeat = False
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(job_id)
        )
    
    async def _heartbeat_loop(self, job_id: str) -> None:
        """Background heartbeat loop."""
        while not self._stop_heartbeat:
            try:
                await self.job_manager.update_heartbeat(job_id)
            except Exception:
                pass  # Don't fail on heartbeat errors
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _stop_heartbeat_task(self) -> None:
        """Stop heartbeat background task."""
        self._stop_heartbeat = True
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None


# ============================================================================
# Convenience Functions
# ============================================================================

async def execute_pipeline(
    db_session: AsyncSession,
    job_manager: Any,
    job: IngestionJob,
    document: Document
) -> ExecutionResult:
    """
    Convenience function to execute a pipeline.
    
    Args:
        db_session: Database session
        job_manager: Job manager instance
        job: Ingestion job
        document: Database document
        
    Returns:
        ExecutionResult
    """
    executor = PipelineExecutor(db_session, job_manager)
    return await executor.execute(job, document)


# ============================================================================
# Register Built-in Pipelines
# ============================================================================

def register_builtin_pipelines() -> None:
    """
    Register all built-in pipeline stages.
    
    This is called at application startup.
    """
    # Import pipelines to trigger registration
    try:
        from pipelines.text_pipeline import TextPipeline
    except ImportError:
        pass
    
    try:
        from pipelines.image_pipeline import ImagePipeline
    except ImportError:
        pass
    
    # Register placeholder for stages without full implementation yet
    from pipelines.base import BasePipeline
    
    class RoutingPipeline(BasePipeline):
        stage_name = "routing"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Routing is handled by IngestionRouter before executor
            return document
    
    class LinkagePipeline(BasePipeline):
        stage_name = "linkage"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Will be implemented in Module 12
            return document
    
    class ChunkingPipeline(BasePipeline):
        stage_name = "chunking"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Will be implemented in Module 13
            return document
    
    class MetadataEnrichmentPipeline(BasePipeline):
        stage_name = "metadata_enrichment"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Will be implemented in Module 16
            return document
    
    class EmbeddingPipeline(BasePipeline):
        stage_name = "embedding"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Will be implemented in Module 14
            return document
    
    class VectorStorePipeline(BasePipeline):
        stage_name = "vector_store"
        async def process(self, document: IngestionDocument) -> IngestionDocument:
            # Will be implemented in Module 15
            return document
    
    # Register placeholders
    PipelineRegistry.register("routing", RoutingPipeline)
    PipelineRegistry.register("linkage", LinkagePipeline)
    PipelineRegistry.register("chunking", ChunkingPipeline)
    PipelineRegistry.register("metadata_enrichment", MetadataEnrichmentPipeline)
    PipelineRegistry.register("embedding", EmbeddingPipeline)
    PipelineRegistry.register("vector_store", VectorStorePipeline)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "StageResult",
    "ExecutionResult",
    "PipelineRegistry",
    "register_pipeline",
    "PipelineExecutor",
    "execute_pipeline",
    "register_builtin_pipelines",
]
