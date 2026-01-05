"""
Job Store Module for Crash Recovery

Based on LibV2 orchestrator-architecture course (Temporal patterns):
- Durable execution with event sourcing
- Checkpoint progress for crash recovery
- Resume from last known good state
"""

import json
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class JobCheckpoint:
    """
    Checkpoint for a transcription job.

    Stores the current state of a job so it can be resumed
    after a crash or restart.
    """
    job_id: str
    audio_path: str
    status: str  # "recording", "processing", "chunk_N", "complete", "failed"
    chunks_completed: int = 0
    total_chunks: int = 0
    partial_transcript: str = ""
    error_message: Optional[str] = None
    device_name: Optional[str] = None
    ticket_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobCheckpoint":
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_incomplete(self) -> bool:
        return self.status not in ("complete", "failed")

    @property
    def progress_percent(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.chunks_completed / self.total_chunks) * 100


class JobStore:
    """
    Persistent storage for job checkpoints.

    Enables crash recovery by saving job state to disk
    after each processing step.
    """

    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize job store.

        Args:
            store_path: Directory for storing checkpoints.
                       Defaults to ~/.callwhisper/jobs/
        """
        if store_path is None:
            store_path = Path.home() / ".callwhisper" / "jobs"

        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._archive_path = self.store_path / "archive"
        self._archive_path.mkdir(exist_ok=True)

        logger.debug("job_store_initialized", path=str(self.store_path))

    def _get_checkpoint_path(self, job_id: str) -> Path:
        """Get path for a job's checkpoint file."""
        return self.store_path / f"{job_id}.json"

    def save_checkpoint(self, checkpoint: JobCheckpoint) -> None:
        """
        Save job checkpoint to disk.

        This is the critical operation for crash recovery.
        Called after each processing stage.
        """
        checkpoint.updated_at = time.time()
        path = self._get_checkpoint_path(checkpoint.job_id)

        try:
            # Write to temp file first, then rename (atomic on most filesystems)
            temp_path = path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            temp_path.rename(path)

            logger.debug(
                "checkpoint_saved",
                job_id=checkpoint.job_id,
                status=checkpoint.status,
                progress=f"{checkpoint.chunks_completed}/{checkpoint.total_chunks}"
            )

        except Exception as e:
            logger.error(
                "checkpoint_save_failed",
                job_id=checkpoint.job_id,
                error=str(e)
            )
            raise

    def load_checkpoint(self, job_id: str) -> Optional[JobCheckpoint]:
        """
        Load job checkpoint from disk.

        Returns None if checkpoint doesn't exist.
        """
        path = self._get_checkpoint_path(job_id)

        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return JobCheckpoint.from_dict(data)

        except Exception as e:
            logger.error(
                "checkpoint_load_failed",
                job_id=job_id,
                error=str(e)
            )
            return None

    def get_incomplete_jobs(self) -> List[JobCheckpoint]:
        """
        Find all jobs that didn't complete.

        Used on startup to offer recovery options.
        """
        incomplete = []

        for path in self.store_path.glob("*.json"):
            if path.name == "archive":
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                checkpoint = JobCheckpoint.from_dict(data)

                if checkpoint.is_incomplete:
                    incomplete.append(checkpoint)

            except Exception as e:
                logger.warning(
                    "checkpoint_parse_error",
                    path=str(path),
                    error=str(e)
                )

        # Sort by created_at (most recent first)
        incomplete.sort(key=lambda c: c.created_at, reverse=True)

        logger.info(
            "incomplete_jobs_found",
            count=len(incomplete)
        )

        return incomplete

    def mark_complete(self, job_id: str) -> None:
        """
        Mark job as complete and archive the checkpoint.

        Moves checkpoint to archive directory for history.
        """
        path = self._get_checkpoint_path(job_id)

        if not path.exists():
            logger.warning("checkpoint_not_found", job_id=job_id)
            return

        try:
            # Load and update status
            checkpoint = self.load_checkpoint(job_id)
            if checkpoint:
                checkpoint.status = "complete"
                checkpoint.updated_at = time.time()

                # Save updated checkpoint
                self.save_checkpoint(checkpoint)

            # Move to archive
            archive_path = self._archive_path / path.name
            shutil.move(str(path), str(archive_path))

            logger.info(
                "job_completed",
                job_id=job_id,
                archived_to=str(archive_path)
            )

        except Exception as e:
            logger.error(
                "mark_complete_failed",
                job_id=job_id,
                error=str(e)
            )

    def mark_failed(self, job_id: str, error_message: str) -> None:
        """
        Mark job as failed with error message.

        Keeps checkpoint in main directory for potential manual recovery.
        """
        checkpoint = self.load_checkpoint(job_id)

        if checkpoint:
            checkpoint.status = "failed"
            checkpoint.error_message = error_message
            checkpoint.updated_at = time.time()
            self.save_checkpoint(checkpoint)

            logger.warning(
                "job_failed",
                job_id=job_id,
                error=error_message
            )

    def delete_checkpoint(self, job_id: str) -> bool:
        """
        Delete a job checkpoint (used when discarding incomplete jobs).

        Returns True if deleted, False if not found.
        """
        path = self._get_checkpoint_path(job_id)

        if path.exists():
            path.unlink()
            logger.info("checkpoint_deleted", job_id=job_id)
            return True

        return False

    def cleanup_old_checkpoints(self, max_age_days: int = 7) -> int:
        """
        Remove old completed job checkpoints from archive.

        Args:
            max_age_days: Maximum age of archived checkpoints

        Returns:
            Number of checkpoints removed
        """
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)
        removed = 0

        for path in self._archive_path.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("updated_at", 0) < cutoff:
                    path.unlink()
                    removed += 1

            except Exception as e:
                logger.warning(
                    "cleanup_error",
                    path=str(path),
                    error=str(e)
                )

        if removed > 0:
            logger.info(
                "old_checkpoints_cleaned",
                count=removed,
                max_age_days=max_age_days
            )

        return removed

    def get_job_history(self, limit: int = 50) -> List[JobCheckpoint]:
        """
        Get recent job history from archive.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of completed job checkpoints
        """
        jobs = []

        for path in self._archive_path.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                jobs.append(JobCheckpoint.from_dict(data))
            except json.JSONDecodeError as e:
                logger.warning(
                    "corrupted_checkpoint_skipped",
                    path=str(path),
                    error=str(e)
                )
                continue
            except KeyError as e:
                logger.warning(
                    "malformed_checkpoint_skipped",
                    path=str(path),
                    missing_field=str(e)
                )
                continue
            except Exception as e:
                logger.warning(
                    "checkpoint_load_error",
                    path=str(path),
                    error=str(e),
                    error_type=type(e).__name__
                )
                continue

        # Sort by updated_at (most recent first)
        jobs.sort(key=lambda c: c.updated_at, reverse=True)

        return jobs[:limit]


# Global job store instance
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get or create global job store instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store


def create_checkpoint(
    job_id: str,
    audio_path: str,
    device_name: Optional[str] = None,
    ticket_id: Optional[str] = None
) -> JobCheckpoint:
    """
    Create and save a new job checkpoint.

    Convenience function for starting a new job.
    """
    checkpoint = JobCheckpoint(
        job_id=job_id,
        audio_path=audio_path,
        status="recording",
        device_name=device_name,
        ticket_id=ticket_id
    )

    store = get_job_store()
    store.save_checkpoint(checkpoint)

    return checkpoint


def update_checkpoint(
    job_id: str,
    status: Optional[str] = None,
    chunks_completed: Optional[int] = None,
    total_chunks: Optional[int] = None,
    partial_transcript: Optional[str] = None
) -> Optional[JobCheckpoint]:
    """
    Update an existing job checkpoint.

    Convenience function for updating job progress.
    """
    store = get_job_store()
    checkpoint = store.load_checkpoint(job_id)

    if checkpoint is None:
        logger.warning("checkpoint_not_found_for_update", job_id=job_id)
        return None

    if status is not None:
        checkpoint.status = status
    if chunks_completed is not None:
        checkpoint.chunks_completed = chunks_completed
    if total_chunks is not None:
        checkpoint.total_chunks = total_chunks
    if partial_transcript is not None:
        checkpoint.partial_transcript = partial_transcript

    store.save_checkpoint(checkpoint)
    return checkpoint
