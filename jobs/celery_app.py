"""
Celery Application Configuration
================================
Celery application setup for the Enterprise RAG Ingestion Pipeline.

Provides:
- Celery app instance with Redis broker
- Task routing configuration
- Retry policies
- Serialization settings
"""

from celery import Celery

from config.settings import get_settings


def create_celery_app() -> Celery:
    """
    Create and configure the Celery application.
    
    Returns:
        Configured Celery application instance
    """
    settings = get_settings()
    celery_config = settings.celery
    
    app = Celery(
        "enterprise_rag_ingestion",
        broker=celery_config.broker_url,
        backend=celery_config.result_backend,
        include=[
            "jobs.job_manager",
            "jobs.pipeline_executor",
        ]
    )
    
    # Configure Celery
    app.conf.update(
        # Serialization
        task_serializer=celery_config.task_serializer,
        result_serializer=celery_config.result_serializer,
        accept_content=celery_config.accept_content,
        
        # Task execution
        task_acks_late=celery_config.task_acks_late,
        worker_prefetch_multiplier=celery_config.worker_prefetch_multiplier,
        task_time_limit=celery_config.task_time_limit,
        task_soft_time_limit=celery_config.task_soft_time_limit,
        
        # Task tracking
        task_track_started=True,
        result_extended=True,
        
        # Retry policy
        task_default_retry_delay=60,  # 1 minute
        task_max_retries=3,
        
        # Task routing
        task_routes={
            "jobs.job_manager.*": {"queue": "ingestion"},
            "jobs.pipeline_executor.*": {"queue": "ingestion"},
        },
        
        # Queue configuration
        task_queues={
            "ingestion": {
                "exchange": "ingestion",
                "routing_key": "ingestion",
            },
        },
        task_default_queue="ingestion",
        
        # Result expiration (24 hours)
        result_expires=86400,
        
        # Timezone
        timezone="UTC",
        enable_utc=True,
    )
    
    return app


# Global Celery app instance
celery_app = create_celery_app()


# ============================================================================
# Exports
# ============================================================================

__all__ = ["celery_app", "create_celery_app"]
