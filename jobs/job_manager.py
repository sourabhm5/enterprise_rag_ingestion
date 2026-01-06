"""
MODULE 5 — JOB MANAGER (CELERY + RESUME SAFE)
=============================================
Reliable job execution under failure for the Enterprise RAG Ingestion Pipeline.

Features:
- Job heartbeat for health monitoring
- Zombie job detection and recovery
- Resume from last completed stage
- Pipeline plan persistence
- Error logging and retry management

Job Lifecycle:
1. PENDING - Job created, waiting for worker
2. RUNNING - Job picked up by worker, executing stages
3. COMPLETED - All stages finished successfully
4. FAILED - Job failed after max retries
5. CANCELLED - Job cancelled by user
6. RESUMING - Previously failed job being resumed
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import JobStatus, get_settings
from jobs.celery_app import celery_app
from storage.postgres import (
    Document,
    IngestionJob,
    get_db_manager,
)


# ============================================================================
# Job Manager Class
# ============================================================================

class JobManager:
    """
    Manages ingestion job lifecycle with resume capability.
    
    Responsibilities:
    - Create and track ingestion jobs
    - Update job status and progress
    - Handle heartbeats for zombie detection
    - Resume failed jobs from last checkpoint
    - Clean up stale jobs
    """
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize JobManager.
        
        Args:
            db_session: SQLAlchemy async session
        """
        self.db = db_session
        self.settings = get_settings()
    
    # =========================================================================
    # Job Creation
    # =========================================================================
    
    async def create_job(
        self,
        document_db_id: uuid.UUID,
        pipeline_plan: Dict[str, Any],
        max_retries: int = 3
    ) -> IngestionJob:
        """
        Create a new ingestion job.
        
        Args:
            document_db_id: Database ID of the document
            pipeline_plan: Pipeline execution plan (stages, metadata)
            max_retries: Maximum retry attempts
            
        Returns:
            Created IngestionJob instance
        """
        job = IngestionJob(
            document_db_id=document_db_id,
            job_id=f"job-{uuid.uuid4().hex[:16]}",
            status=JobStatus.PENDING.value,
            pipeline_plan=pipeline_plan,
            current_stage=None,
            completed_stages=[],
            error_log=[],
            retry_count=0,
            max_retries=max_retries,
            metrics={}
        )
        
        self.db.add(job)
        await self.db.flush()
        
        return job
    
    # =========================================================================
    # Job Status Management
    # =========================================================================
    
    async def get_job(self, job_id: str) -> Optional[IngestionJob]:
        """Get job by ID."""
        stmt = select(IngestionJob).where(IngestionJob.job_id == job_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_job_by_db_id(self, db_id: uuid.UUID) -> Optional[IngestionJob]:
        """Get job by database ID."""
        stmt = select(IngestionJob).where(IngestionJob.id == db_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def start_job(
        self,
        job_id: str,
        worker_id: str
    ) -> Optional[IngestionJob]:
        """
        Mark a job as started.
        
        Args:
            job_id: Job identifier
            worker_id: Celery worker ID
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.worker_id = worker_id
        job.last_heartbeat = datetime.utcnow()
        
        await self.db.flush()
        return job
    
    async def complete_job(self, job_id: str, metrics: Optional[Dict] = None) -> Optional[IngestionJob]:
        """
        Mark a job as completed.
        
        Args:
            job_id: Job identifier
            metrics: Final job metrics
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.status = JobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        job.current_stage = None
        
        if metrics:
            job.metrics = {**job.metrics, **metrics}
        
        # Calculate total duration
        if job.started_at:
            duration = (job.completed_at - job.started_at).total_seconds()
            job.metrics["total_duration_seconds"] = duration
        
        await self.db.flush()
        return job
    
    async def fail_job(
        self,
        job_id: str,
        error: str,
        stage: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> Optional[IngestionJob]:
        """
        Mark a job as failed.
        
        Args:
            job_id: Job identifier
            error: Error message
            stage: Stage where failure occurred
            details: Additional error details
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.status = JobStatus.FAILED.value
        job.completed_at = datetime.utcnow()
        job.add_error(
            stage=stage or job.current_stage or "unknown",
            error=error,
            details=details
        )
        
        await self.db.flush()
        return job
    
    async def cancel_job(self, job_id: str, reason: str = "User cancelled") -> Optional[IngestionJob]:
        """
        Cancel a running or pending job.
        
        Args:
            job_id: Job identifier
            reason: Cancellation reason
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        if job.status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
            return job  # Already finished
        
        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        job.add_error(stage="cancellation", error=reason)
        
        await self.db.flush()
        return job
    
    # =========================================================================
    # Stage Management
    # =========================================================================
    
    async def start_stage(self, job_id: str, stage: str) -> Optional[IngestionJob]:
        """
        Mark a stage as started.
        
        Args:
            job_id: Job identifier
            stage: Stage name
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.current_stage = stage
        job.last_heartbeat = datetime.utcnow()
        
        # Track stage start time in metrics
        if "stage_timings" not in job.metrics:
            job.metrics = {**job.metrics, "stage_timings": {}}
        job.metrics["stage_timings"][stage] = {
            "started_at": datetime.utcnow().isoformat()
        }
        
        await self.db.flush()
        return job
    
    async def complete_stage(
        self,
        job_id: str,
        stage: str,
        stage_metrics: Optional[Dict] = None
    ) -> Optional[IngestionJob]:
        """
        Mark a stage as completed.
        
        Args:
            job_id: Job identifier
            stage: Stage name
            stage_metrics: Metrics for this stage
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.mark_stage_complete(stage)
        job.last_heartbeat = datetime.utcnow()
        
        # Track stage completion time
        if "stage_timings" in job.metrics and stage in job.metrics["stage_timings"]:
            job.metrics["stage_timings"][stage]["completed_at"] = datetime.utcnow().isoformat()
            if stage_metrics:
                job.metrics["stage_timings"][stage]["metrics"] = stage_metrics
        
        await self.db.flush()
        return job
    
    async def get_next_stage(self, job_id: str) -> Optional[str]:
        """
        Get the next stage to execute for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Next stage name or None if all stages complete
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        stages = job.pipeline_plan.get("stages", [])
        completed = set(job.completed_stages) if isinstance(job.completed_stages, list) else set()
        
        for stage in stages:
            if stage not in completed:
                return stage
        
        return None
    
    async def get_remaining_stages(self, job_id: str) -> List[str]:
        """
        Get all remaining stages for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of remaining stage names
        """
        job = await self.get_job(job_id)
        if not job:
            return []
        
        stages = job.pipeline_plan.get("stages", [])
        completed = set(job.completed_stages) if isinstance(job.completed_stages, list) else set()
        
        return [s for s in stages if s not in completed]
    
    # =========================================================================
    # Heartbeat Management
    # =========================================================================
    
    async def update_heartbeat(self, job_id: str) -> Optional[IngestionJob]:
        """
        Update job heartbeat timestamp.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Updated job or None if not found
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        job.update_heartbeat()
        await self.db.flush()
        return job
    
    async def check_job_health(self, job_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a job is healthy (not a zombie).
        
        Args:
            job_id: Job identifier
            
        Returns:
            Tuple of (is_healthy, reason_if_unhealthy)
        """
        job = await self.get_job(job_id)
        if not job:
            return False, "Job not found"
        
        if job.status != JobStatus.RUNNING.value:
            return True, None  # Only check running jobs
        
        threshold = self.settings.celery.zombie_threshold
        if job.is_zombie(threshold_seconds=threshold):
            return False, f"No heartbeat for {threshold} seconds"
        
        return True, None
    
    # =========================================================================
    # Zombie Detection & Recovery
    # =========================================================================
    
    async def find_zombie_jobs(self) -> List[IngestionJob]:
        """
        Find all zombie jobs (running but no heartbeat).
        
        Returns:
            List of zombie jobs
        """
        threshold = self.settings.celery.zombie_threshold
        cutoff = datetime.utcnow() - timedelta(seconds=threshold)
        
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.status == JobStatus.RUNNING.value)
            .where(
                (IngestionJob.last_heartbeat < cutoff) |
                (IngestionJob.last_heartbeat.is_(None))
            )
        )
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def recover_zombie_job(self, job_id: str) -> Optional[IngestionJob]:
        """
        Attempt to recover a zombie job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Recovered job or None
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        # Check if we can retry
        if job.retry_count >= job.max_retries:
            job.status = JobStatus.FAILED.value
            job.completed_at = datetime.utcnow()
            job.add_error(
                stage=job.current_stage or "unknown",
                error="Max retries exceeded after zombie recovery",
                details={"retry_count": job.retry_count}
            )
        else:
            # Mark for resumption
            job.status = JobStatus.RESUMING.value
            job.retry_count += 1
            job.worker_id = None
            job.add_error(
                stage=job.current_stage or "unknown",
                error="Job detected as zombie, scheduling for resume",
                details={"retry_attempt": job.retry_count}
            )
        
        await self.db.flush()
        return job
    
    async def resume_job(
        self,
        job_id: str,
        worker_id: str
    ) -> Optional[IngestionJob]:
        """
        Resume a previously failed or zombie job.
        
        Args:
            job_id: Job identifier
            worker_id: New worker ID
            
        Returns:
            Resumed job or None
        """
        job = await self.get_job(job_id)
        if not job:
            return None
        
        if job.status not in [JobStatus.RESUMING.value, JobStatus.FAILED.value]:
            return None
        
        job.status = JobStatus.RUNNING.value
        job.worker_id = worker_id
        job.last_heartbeat = datetime.utcnow()
        # current_stage is preserved from where it failed
        
        await self.db.flush()
        return job
    
    # =========================================================================
    # Job Queries
    # =========================================================================
    
    async def get_jobs_by_document(
        self,
        document_db_id: uuid.UUID,
        limit: int = 10
    ) -> List[IngestionJob]:
        """Get all jobs for a document."""
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.document_db_id == document_db_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_pending_jobs(self, limit: int = 100) -> List[IngestionJob]:
        """Get all pending jobs."""
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.status == JobStatus.PENDING.value)
            .order_by(IngestionJob.created_at.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_jobs_to_resume(self, limit: int = 100) -> List[IngestionJob]:
        """Get all jobs marked for resumption."""
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.status == JobStatus.RESUMING.value)
            .order_by(IngestionJob.created_at.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())


# ============================================================================
# Celery Tasks
# ============================================================================

class IngestionTask(Task):
    """Base task class with database session management."""
    
    _db_manager = None
    
    @property
    def db_manager(self):
        if self._db_manager is None:
            self._db_manager = get_db_manager()
        return self._db_manager


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="jobs.job_manager.execute_ingestion_job",
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=3300,
    time_limit=3600
)
def execute_ingestion_job(self, job_id: str) -> Dict[str, Any]:
    """
    Execute an ingestion job.
    
    This is the main Celery task that orchestrates document ingestion.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job result dictionary
    """
    # Run async code in sync context
    return asyncio.get_event_loop().run_until_complete(
        _execute_ingestion_job_async(self, job_id)
    )


async def _execute_ingestion_job_async(task: IngestionTask, job_id: str) -> Dict[str, Any]:
    """Async implementation of job execution."""
    from jobs.pipeline_executor import PipelineExecutor
    
    async with task.db_manager.async_session() as session:
        job_manager = JobManager(session)
        
        # Start the job
        job = await job_manager.start_job(job_id, task.request.id)
        if not job:
            return {"status": "error", "message": f"Job not found: {job_id}"}
        
        try:
            # Get the document
            stmt = select(Document).where(Document.id == job.document_db_id)
            result = await session.execute(stmt)
            document = result.scalar_one_or_none()
            
            if not document:
                await job_manager.fail_job(job_id, "Document not found")
                return {"status": "error", "message": "Document not found"}
            
            # Create pipeline executor
            executor = PipelineExecutor(session, job_manager)
            
            # Execute pipeline
            result = await executor.execute(job, document)
            
            await session.commit()
            return result
            
        except SoftTimeLimitExceeded:
            await job_manager.fail_job(
                job_id,
                "Job exceeded soft time limit",
                details={"time_limit": task.soft_time_limit}
            )
            await session.commit()
            raise
            
        except Exception as e:
            await job_manager.fail_job(
                job_id,
                str(e),
                details={"exception_type": type(e).__name__}
            )
            await session.commit()
            
            # Retry if possible
            if task.request.retries < task.max_retries:
                raise task.retry(exc=e)
            
            return {"status": "error", "message": str(e)}


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="jobs.job_manager.resume_failed_job",
    max_retries=1
)
def resume_failed_job(self, job_id: str) -> Dict[str, Any]:
    """
    Resume a failed or zombie job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job result dictionary
    """
    return asyncio.get_event_loop().run_until_complete(
        _resume_failed_job_async(self, job_id)
    )


async def _resume_failed_job_async(task: IngestionTask, job_id: str) -> Dict[str, Any]:
    """Async implementation of job resumption."""
    async with task.db_manager.async_session() as session:
        job_manager = JobManager(session)
        
        # Resume the job
        job = await job_manager.resume_job(job_id, task.request.id)
        if not job:
            return {"status": "error", "message": f"Cannot resume job: {job_id}"}
        
        await session.commit()
        
        # Dispatch to main execution task
        execute_ingestion_job.delay(job_id)
        
        return {"status": "resumed", "job_id": job_id}


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="jobs.job_manager.cleanup_zombie_jobs"
)
def cleanup_zombie_jobs(self) -> Dict[str, Any]:
    """
    Periodic task to find and recover zombie jobs.
    
    Should be scheduled via Celery Beat.
    
    Returns:
        Cleanup result dictionary
    """
    return asyncio.get_event_loop().run_until_complete(
        _cleanup_zombie_jobs_async(self)
    )


async def _cleanup_zombie_jobs_async(task: IngestionTask) -> Dict[str, Any]:
    """Async implementation of zombie cleanup."""
    async with task.db_manager.async_session() as session:
        job_manager = JobManager(session)
        
        zombies = await job_manager.find_zombie_jobs()
        recovered = []
        failed = []
        
        for zombie in zombies:
            job = await job_manager.recover_zombie_job(zombie.job_id)
            if job:
                if job.status == JobStatus.RESUMING.value:
                    recovered.append(zombie.job_id)
                    # Schedule resumption
                    resume_failed_job.delay(zombie.job_id)
                else:
                    failed.append(zombie.job_id)
        
        await session.commit()
        
        return {
            "status": "completed",
            "zombies_found": len(zombies),
            "recovered": recovered,
            "failed": failed
        }


@celery_app.task(
    bind=True,
    base=IngestionTask,
    name="jobs.job_manager.send_heartbeat"
)
def send_heartbeat(self, job_id: str) -> Dict[str, Any]:
    """
    Update heartbeat for a running job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Heartbeat result
    """
    return asyncio.get_event_loop().run_until_complete(
        _send_heartbeat_async(self, job_id)
    )


async def _send_heartbeat_async(task: IngestionTask, job_id: str) -> Dict[str, Any]:
    """Async implementation of heartbeat."""
    async with task.db_manager.async_session() as session:
        job_manager = JobManager(session)
        job = await job_manager.update_heartbeat(job_id)
        await session.commit()
        
        if job:
            return {"status": "ok", "job_id": job_id}
        return {"status": "error", "message": "Job not found"}


# ============================================================================
# Celery Beat Schedule (for periodic tasks)
# ============================================================================

celery_app.conf.beat_schedule = {
    "cleanup-zombie-jobs": {
        "task": "jobs.job_manager.cleanup_zombie_jobs",
        "schedule": 60.0,  # Every minute
    },
}


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "JobManager",
    "execute_ingestion_job",
    "resume_failed_job",
    "cleanup_zombie_jobs",
    "send_heartbeat",
]
