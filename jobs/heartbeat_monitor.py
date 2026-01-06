"""
Heartbeat Monitor
=================
Continuous health monitoring for running ingestion jobs.

Provides:
- Background heartbeat updates during job execution
- Health status reporting
- Integration with job manager for zombie detection
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable, Optional

from config.settings import get_settings


class HeartbeatMonitor:
    """
    Background heartbeat monitor for long-running jobs.
    
    Usage:
        async with HeartbeatMonitor(job_id, heartbeat_callback) as monitor:
            # Do long-running work
            # Heartbeats are sent automatically
            pass
    """
    
    def __init__(
        self,
        job_id: str,
        heartbeat_callback: Callable[[str], None],
        interval: Optional[int] = None
    ):
        """
        Initialize heartbeat monitor.
        
        Args:
            job_id: Job identifier
            heartbeat_callback: Async function to call for heartbeat updates
            interval: Heartbeat interval in seconds (default from settings)
        """
        self.job_id = job_id
        self.heartbeat_callback = heartbeat_callback
        self.interval = interval or get_settings().celery.heartbeat_interval
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._last_heartbeat: Optional[datetime] = None
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                await self.heartbeat_callback(self.job_id)
                self._last_heartbeat = datetime.utcnow()
            except Exception as e:
                # Log but don't fail the job
                print(f"Heartbeat failed for job {self.job_id}: {e}")
            
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval
                )
            except asyncio.TimeoutError:
                # Expected - continue heartbeat loop
                pass
    
    async def start(self):
        """Start the heartbeat monitor."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """Stop the heartbeat monitor."""
        self._stop_event.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
    
    @property
    def last_heartbeat(self) -> Optional[datetime]:
        """Get the timestamp of the last successful heartbeat."""
        return self._last_heartbeat
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        return False


class SyncHeartbeatMonitor:
    """
    Synchronous heartbeat monitor using a background thread.
    
    Useful for non-async job execution contexts.
    """
    
    def __init__(
        self,
        job_id: str,
        heartbeat_callback: Callable[[str], None],
        interval: Optional[int] = None
    ):
        """
        Initialize sync heartbeat monitor.
        
        Args:
            job_id: Job identifier
            heartbeat_callback: Sync function to call for heartbeat updates
            interval: Heartbeat interval in seconds
        """
        self.job_id = job_id
        self.heartbeat_callback = heartbeat_callback
        self.interval = interval or get_settings().celery.heartbeat_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat: Optional[datetime] = None
    
    def _heartbeat_loop(self):
        """Background heartbeat loop."""
        while not self._stop_event.is_set():
            try:
                self.heartbeat_callback(self.job_id)
                self._last_heartbeat = datetime.utcnow()
            except Exception as e:
                print(f"Heartbeat failed for job {self.job_id}: {e}")
            
            self._stop_event.wait(timeout=self.interval)
    
    def start(self):
        """Start the heartbeat monitor."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the heartbeat monitor."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
    
    @property
    def last_heartbeat(self) -> Optional[datetime]:
        """Get the timestamp of the last successful heartbeat."""
        return self._last_heartbeat
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "HeartbeatMonitor",
    "SyncHeartbeatMonitor",
]
