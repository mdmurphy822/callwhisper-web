"""
Job Queue Service for Batch Transcription

Provides async job queue for processing multiple audio files sequentially.

Features:
- FIFO with priority support
- Thread-safe with asyncio lock
- WebSocket notifications for UI updates
- Persistent state tracking
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
import asyncio
import uuid

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedJob:
    """Represents a job in the transcription queue."""

    job_id: str
    audio_path: Path
    original_filename: str
    ticket_id: Optional[str] = None
    priority: int = 0  # Higher = more urgent
    status: str = "queued"  # queued, processing, complete, failed
    error_message: Optional[str] = None
    progress: int = 0  # 0-100 progress percentage
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "original_filename": self.original_filename,
            "ticket_id": self.ticket_id,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class JobQueue:
    """
    Async job queue for batch transcription.

    Features:
    - FIFO with priority support
    - Thread-safe with asyncio lock
    - WebSocket notifications for UI updates
    """

    def __init__(self):
        self._queue: List[QueuedJob] = []
        self._processing: Optional[QueuedJob] = None
        self._completed: List[QueuedJob] = []
        self._failed: List[QueuedJob] = []
        self._lock = asyncio.Lock()
        self._worker_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._status_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = (
            None
        )

    def set_status_callback(
        self, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """Set callback for status updates (used for WebSocket notifications)."""
        self._status_callback = callback

    async def _notify_status_change(self):
        """Notify listeners of queue status change."""
        if self._status_callback:
            try:
                await self._status_callback(self.get_status())
            except Exception as e:
                logger.error("queue_status_callback_error", error=str(e))

    async def add_job(
        self,
        audio_path: Path,
        original_filename: str,
        ticket_id: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """Add a job to the queue. Returns job_id."""
        job_id = str(uuid.uuid4())[:8]
        job = QueuedJob(
            job_id=job_id,
            audio_path=audio_path,
            original_filename=original_filename,
            ticket_id=ticket_id,
            priority=priority,
        )

        async with self._lock:
            self._queue.append(job)
            # Sort by priority (higher first), then by created_at (older first)
            self._queue.sort(key=lambda j: (-j.priority, j.created_at))

        logger.info(
            "job_queued",
            job_id=job_id,
            filename=original_filename,
            priority=priority,
            queue_length=len(self._queue),
        )
        await self._notify_status_change()
        return job_id

    async def get_next_job(self) -> Optional[QueuedJob]:
        """Get next job from queue (does not remove it)."""
        async with self._lock:
            if self._queue:
                return self._queue[0]
        return None

    async def start_processing(self, job_id: str) -> Optional[QueuedJob]:
        """Mark a job as processing and remove from queue."""
        async with self._lock:
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    job.status = "processing"
                    job.started_at = datetime.now().timestamp()
                    self._processing = job
                    self._queue.pop(i)
                    logger.info(
                        "job_processing_started",
                        job_id=job_id,
                        filename=job.original_filename,
                    )
                    await self._notify_status_change()
                    return job
        return None

    async def update_progress(self, job_id: str, progress: int):
        """Update progress for current job."""
        async with self._lock:
            if self._processing and self._processing.job_id == job_id:
                self._processing.progress = min(100, max(0, progress))
                await self._notify_status_change()

    async def complete_job(self, job_id: str, success: bool, error: str = None):
        """Mark current job as complete or failed."""
        async with self._lock:
            if self._processing and self._processing.job_id == job_id:
                self._processing.completed_at = datetime.now().timestamp()
                self._processing.progress = (
                    100 if success else self._processing.progress
                )
                if success:
                    self._processing.status = "complete"
                    self._completed.append(self._processing)
                    logger.info(
                        "job_completed",
                        job_id=job_id,
                        filename=self._processing.original_filename,
                        duration_seconds=self._processing.completed_at
                        - self._processing.started_at,
                    )
                else:
                    self._processing.status = "failed"
                    self._processing.error_message = error
                    self._failed.append(self._processing)
                    logger.error(
                        "job_failed",
                        job_id=job_id,
                        filename=self._processing.original_filename,
                        error=error,
                    )
                self._processing = None
        await self._notify_status_change()

    async def cancel_job(self, job_id: str) -> bool:
        """Remove a queued job. Returns True if found and removed."""
        async with self._lock:
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    removed_job = self._queue.pop(i)
                    # Clean up the temp file
                    if removed_job.audio_path.exists():
                        try:
                            removed_job.audio_path.unlink()
                        except Exception as e:
                            logger.warning(
                                "job_cleanup_failed", job_id=job_id, error=str(e)
                            )
                    logger.info("job_cancelled", job_id=job_id)
                    await self._notify_status_change()
                    return True
        return False

    async def clear_completed(self):
        """Clear completed and failed job history."""
        async with self._lock:
            self._completed.clear()
            self._failed.clear()
        await self._notify_status_change()

    def get_status(self) -> Dict[str, Any]:
        """Get full queue status."""
        return {
            "queued": [j.to_dict() for j in self._queue],
            "processing": self._processing.to_dict() if self._processing else None,
            "completed": [j.to_dict() for j in self._completed[-10:]],  # Last 10
            "failed": [j.to_dict() for j in self._failed[-10:]],
            "counts": {
                "queued": len(self._queue),
                "processing": 1 if self._processing else 0,
                "completed": len(self._completed),
                "failed": len(self._failed),
            },
        }

    def get_job(self, job_id: str) -> Optional[QueuedJob]:
        """Get a job by ID from any list."""
        if self._processing and self._processing.job_id == job_id:
            return self._processing
        for job in self._queue:
            if job.job_id == job_id:
                return job
        for job in self._completed:
            if job.job_id == job_id:
                return job
        for job in self._failed:
            if job.job_id == job_id:
                return job
        return None

    async def start_worker(self, process_func: Callable[[QueuedJob], Awaitable[None]]):
        """Start background worker to process queue."""
        if self._worker_task and not self._worker_task.done():
            logger.debug("queue_worker_already_running")
            return

        self._stop_event.clear()
        self._worker_task = asyncio.create_task(self._worker_loop(process_func))
        logger.info("queue_worker_started")

    async def stop_worker(self, wait: bool = True):
        """Stop background worker."""
        self._stop_event.set()
        if self._worker_task and wait:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("queue_worker_stop_timeout")
                self._worker_task.cancel()
        logger.info("queue_worker_stopped")

    def is_worker_running(self) -> bool:
        """Check if worker is currently running."""
        return self._worker_task is not None and not self._worker_task.done()

    async def _worker_loop(self, process_func: Callable[[QueuedJob], Awaitable[None]]):
        """Background loop that processes jobs."""
        logger.debug("queue_worker_loop_started")
        while not self._stop_event.is_set():
            job = await self.get_next_job()
            if job:
                await self.start_processing(job.job_id)
                try:
                    await process_func(job)
                    await self.complete_job(job.job_id, success=True)
                except Exception as e:
                    logger.exception(
                        "job_processing_error", job_id=job.job_id, error=str(e)
                    )
                    await self.complete_job(job.job_id, success=False, error=str(e))
            else:
                # No jobs, wait before checking again
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass  # Continue loop
        logger.debug("queue_worker_loop_ended")


# Global queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get or create the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
        logger.info("job_queue_initialized")
    return _job_queue


def reset_job_queue():
    """Reset the global job queue (for testing)."""
    global _job_queue
    _job_queue = None
