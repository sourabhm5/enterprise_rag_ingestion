"""
Base Pipeline
=============
Abstract base class for all pipeline stages.

All pipelines implement:
    async process(doc: IngestionDocument) -> IngestionDocument

Pipelines ONLY mutate IngestionDocument - they never call each other directly.
The DAG Executor owns orchestration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from schema.intermediate_representation import IngestionDocument, ProcessingStatus


class BasePipeline(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Every pipeline stage must:
    1. Implement the process() method
    2. Only mutate the IngestionDocument
    3. NOT call other pipelines directly
    4. Handle errors gracefully
    """
    
    # Pipeline stage name (used for tracking)
    stage_name: str = "base"
    
    def __init__(self):
        """Initialize the pipeline."""
        self._metrics: Dict[str, Any] = {}
    
    @abstractmethod
    async def process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Process the document.
        
        This is the main entry point for the pipeline stage.
        
        Args:
            document: IngestionDocument to process
            
        Returns:
            Updated IngestionDocument
        """
        pass
    
    async def pre_process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Pre-processing hook (optional override).
        
        Called before process(). Use for validation, setup, etc.
        """
        return document
    
    async def post_process(self, document: IngestionDocument) -> IngestionDocument:
        """
        Post-processing hook (optional override).
        
        Called after process(). Use for cleanup, validation, etc.
        """
        return document
    
    async def execute(self, document: IngestionDocument) -> IngestionDocument:
        """
        Execute the full pipeline stage.
        
        This wraps process() with pre/post hooks and error handling.
        """
        import time
        
        start_time = time.time()
        
        try:
            # Mark stage as started
            document.start_stage(self.stage_name)
            
            # Pre-processing
            document = await self.pre_process(document)
            
            # Main processing
            document = await self.process(document)
            
            # Post-processing
            document = await self.post_process(document)
            
            # Record metrics
            elapsed = time.time() - start_time
            self._metrics = {
                "duration_seconds": elapsed,
                "success": True
            }
            
            # Mark stage as completed
            document.complete_stage(self.stage_name, self._metrics)
            
        except Exception as e:
            # Record error
            elapsed = time.time() - start_time
            self._metrics = {
                "duration_seconds": elapsed,
                "success": False,
                "error": str(e)
            }
            
            document.add_error(
                self.stage_name,
                str(e),
                {"exception_type": type(e).__name__}
            )
            
            # Re-raise to let executor handle
            raise
        
        return document
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from last execution."""
        return self._metrics
    
    def validate_input(self, document: IngestionDocument) -> bool:
        """
        Validate input document (optional override).
        
        Returns:
            True if document is valid for this stage
        """
        return True
    
    def can_skip(self, document: IngestionDocument) -> bool:
        """
        Check if this stage can be skipped (optional override).
        
        Returns:
            True if stage can be skipped
        """
        return document.is_stage_completed(self.stage_name)


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(self, stage: str, message: str, details: Optional[Dict] = None):
        self.stage = stage
        self.message = message
        self.details = details or {}
        super().__init__(f"[{stage}] {message}")


class PipelineSkipped(Exception):
    """Raised when a pipeline stage is skipped."""
    
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] Skipped: {reason}")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BasePipeline",
    "PipelineError",
    "PipelineSkipped",
]
