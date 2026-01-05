"""
Idempotency Manager Module

Based on LibV2 orchestrator-architecture patterns:
- Pipeline Execution Models (chunks 5029-5050)
- Prevent duplicate processing on retries
- Guarantee exactly-once semantics for transcription

Key use cases:
- Audio upload retries don't re-transcribe
- Network failures during upload don't cause duplicates
- Crash recovery doesn't re-process completed work
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from .logging_config import get_core_logger

logger = get_core_logger()

T = TypeVar("T")


@dataclass
class IdempotencyConfig:
    """Configuration for idempotency manager."""

    cache_ttl: int = 86400  # 24 hours default
    max_entries: int = 1000
    enabled: bool = True


@dataclass
class IdempotencyRecord:
    """
    Record of an idempotent operation.

    Uses __slots__ for memory efficiency (~40% savings per entry).
    """

    __slots__ = ("result", "created_at", "completed", "error", "metadata")

    result: Any
    created_at: float
    completed: bool
    error: Optional[str]
    metadata: Dict[str, Any]

    def __init__(
        self,
        result: Any = None,
        created_at: Optional[float] = None,
        completed: bool = False,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.result = result
        self.created_at = created_at or time.time()
        self.completed = completed
        self.error = error
        self.metadata = metadata or {}


class IdempotencyManager:
    """
    Prevent duplicate transcription processing on retries.

    Based on LibV2 orchestrator-architecture (Pipeline Execution Models):
    - Uses idempotency keys to detect duplicate requests
    - Caches results for TTL period
    - Returns cached result instead of re-executing
    - Thread-safe with Lock()

    Key generation strategies:
    - Audio file: SHA256 of file content + size
    - Session-based: session_id + operation_name
    - Request-based: client-provided idempotency token

    Example:
        manager = IdempotencyManager(IdempotencyConfig())

        # Generate key from audio file
        key = manager.generate_audio_key(audio_path)

        # Get cached or execute
        result = manager.get_or_execute(
            key,
            transcribe_audio,
            audio_path, model
        )
    """

    def __init__(self, config: IdempotencyConfig):
        self.config = config
        self._lock = threading.Lock()
        self._records: Dict[str, IdempotencyRecord] = {}
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "duplicates_prevented": 0,
            "expirations": 0,
            "in_progress_blocks": 0,
        }

    @staticmethod
    def generate_audio_key(audio_path: Path) -> str:
        """
        Generate idempotency key from audio file.

        Uses full SHA256 hash for collision resistance.
        Includes file size and modification time for accuracy.
        """
        stat = audio_path.stat()
        hasher = hashlib.sha256()

        # Add file metadata
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime).encode())

        # Hash file contents
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)

        return f"audio_{hasher.hexdigest()}"

    @staticmethod
    def generate_session_key(session_id: str, operation: str) -> str:
        """Generate idempotency key from session and operation."""
        hasher = hashlib.sha256()
        hasher.update(session_id.encode())
        hasher.update(operation.encode())
        return f"session_{hasher.hexdigest()[:32]}"

    @staticmethod
    def generate_request_key(client_token: str) -> str:
        """Generate key from client-provided idempotency token."""
        # Validate and normalize client token
        if not client_token or len(client_token) > 128:
            raise ValueError("Invalid idempotency token: must be 1-128 chars")
        return f"request_{client_token}"

    def _is_expired(self, record: IdempotencyRecord) -> bool:
        """Check if a record has expired."""
        return time.time() - record.created_at > self.config.cache_ttl

    def _evict_oldest(self) -> None:
        """Evict the oldest completed record."""
        if not self._records:
            return

        # Only evict completed records
        completed_records = [(k, v) for k, v in self._records.items() if v.completed]

        if not completed_records:
            return

        oldest_key = min(completed_records, key=lambda x: x[1].created_at)[0]

        del self._records[oldest_key]
        logger.debug("idempotency_eviction", key=oldest_key[:16])

    def _cleanup_expired(self) -> int:
        """Remove all expired records. Returns count removed."""
        now = time.time()
        expired_keys = [
            key
            for key, record in self._records.items()
            if now - record.created_at > self.config.cache_ttl
        ]

        for key in expired_keys:
            del self._records[key]
            self._stats["expirations"] += 1

        return len(expired_keys)

    def check(self, idempotency_key: str) -> Optional[Tuple[bool, Any]]:
        """
        Check if operation was already processed.

        Returns:
            None if not found (should execute)
            (True, result) if completed successfully
            (False, error) if completed with error
        """
        if not self.config.enabled:
            return None

        with self._lock:
            if idempotency_key not in self._records:
                return None

            record = self._records[idempotency_key]

            # Check expiration
            if self._is_expired(record):
                del self._records[idempotency_key]
                self._stats["expirations"] += 1
                return None

            # Operation in progress - block duplicate
            if not record.completed:
                self._stats["in_progress_blocks"] += 1
                logger.warning(
                    "idempotency_in_progress",
                    key=idempotency_key[:16],
                    age_seconds=time.time() - record.created_at,
                )
                raise OperationInProgressError(
                    f"Operation {idempotency_key[:16]} already in progress"
                )

            self._stats["cache_hits"] += 1
            self._stats["duplicates_prevented"] += 1

            logger.info(
                "idempotency_hit",
                key=idempotency_key[:16],
                had_error=record.error is not None,
            )

            if record.error:
                return (False, record.error)
            return (True, record.result)

    def start(
        self, idempotency_key: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark operation as started (in progress).

        Should be called before beginning the operation.
        Prevents concurrent duplicate executions.
        """
        if not self.config.enabled:
            return

        with self._lock:
            # Evict if at capacity
            while len(self._records) >= self.config.max_entries:
                self._evict_oldest()

            self._records[idempotency_key] = IdempotencyRecord(
                result=None,
                created_at=time.time(),
                completed=False,
                error=None,
                metadata=metadata or {},
            )

            logger.debug(
                "idempotency_start",
                key=idempotency_key[:16],
                records_count=len(self._records),
            )

    def complete(
        self,
        idempotency_key: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark operation as completed successfully.

        Stores result for future duplicate detection.
        """
        if not self.config.enabled:
            return

        with self._lock:
            if idempotency_key in self._records:
                record = self._records[idempotency_key]
                record.result = result
                record.completed = True
                if metadata:
                    record.metadata.update(metadata)
            else:
                # Operation wasn't started with start() - still record it
                self._records[idempotency_key] = IdempotencyRecord(
                    result=result,
                    created_at=time.time(),
                    completed=True,
                    error=None,
                    metadata=metadata or {},
                )

            logger.debug("idempotency_complete", key=idempotency_key[:16])

    def fail(
        self,
        idempotency_key: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark operation as failed.

        Note: Failed operations are still cached to prevent
        retrying operations that will always fail (deterministic errors).
        For transient errors, use cancel() instead.
        """
        if not self.config.enabled:
            return

        with self._lock:
            if idempotency_key in self._records:
                record = self._records[idempotency_key]
                record.error = error
                record.completed = True
                if metadata:
                    record.metadata.update(metadata)
            else:
                self._records[idempotency_key] = IdempotencyRecord(
                    result=None,
                    created_at=time.time(),
                    completed=True,
                    error=error,
                    metadata=metadata or {},
                )

            logger.warning("idempotency_fail", key=idempotency_key[:16], error=error)

    def cancel(self, idempotency_key: str) -> bool:
        """
        Cancel an in-progress operation.

        Use for transient errors that should be retried.
        Returns True if record was found and removed.
        """
        with self._lock:
            if idempotency_key in self._records:
                del self._records[idempotency_key]
                logger.debug("idempotency_cancel", key=idempotency_key[:16])
                return True
            return False

    def get_or_execute(
        self, idempotency_key: str, func: Callable[..., T], *args, **kwargs
    ) -> T:
        """
        Get cached result or execute function.

        High-level API that combines check/start/complete/fail.

        Args:
            idempotency_key: Unique key for this operation
            func: Function to execute if not cached
            *args, **kwargs: Arguments to pass to func

        Returns:
            Cached or newly computed result

        Raises:
            OperationInProgressError: If operation is already running
            Original exception: If func raises (after recording failure)
        """
        # Check for existing result
        check_result = self.check(idempotency_key)
        if check_result is not None:
            success, value = check_result
            if success:
                return value
            else:
                # Re-raise recorded error
                raise IdempotencyRecordedError(value)

        # Mark as started
        self.start(idempotency_key)
        self._stats["cache_misses"] += 1

        try:
            result = func(*args, **kwargs)
            self.complete(idempotency_key, result)
            return result
        except Exception as e:
            # Record failure for deterministic errors
            self.fail(idempotency_key, str(e))
            raise

    async def get_or_execute_async(
        self, idempotency_key: str, func: Callable[..., T], *args, **kwargs
    ) -> T:
        """
        Async version of get_or_execute.

        Same semantics but awaits the function.
        """
        # Check for existing result
        check_result = self.check(idempotency_key)
        if check_result is not None:
            success, value = check_result
            if success:
                return value
            else:
                raise IdempotencyRecordedError(value)

        # Mark as started
        self.start(idempotency_key)
        self._stats["cache_misses"] += 1

        try:
            result = await func(*args, **kwargs)
            self.complete(idempotency_key, result)
            return result
        except Exception as e:
            self.fail(idempotency_key, str(e))
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get idempotency manager statistics."""
        with self._lock:
            in_progress = sum(1 for r in self._records.values() if not r.completed)
            completed = sum(1 for r in self._records.values() if r.completed)
            failed = sum(1 for r in self._records.values() if r.completed and r.error)

            total_requests = self._stats["cache_hits"] + self._stats["cache_misses"]
            hit_rate = (
                self._stats["cache_hits"] / total_requests
                if total_requests > 0
                else 0.0
            )

            return {
                "enabled": self.config.enabled,
                "records_count": len(self._records),
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.cache_ttl,
                "in_progress": in_progress,
                "completed": completed,
                "failed": failed,
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "hit_rate": round(hit_rate, 3),
                "duplicates_prevented": self._stats["duplicates_prevented"],
                "expirations": self._stats["expirations"],
                "in_progress_blocks": self._stats["in_progress_blocks"],
            }

    def clear(self) -> int:
        """Clear all records. Returns count cleared."""
        with self._lock:
            count = len(self._records)
            self._records.clear()
            logger.info("idempotency_cleared", records_removed=count)
            return count


class OperationInProgressError(Exception):
    """Raised when attempting to start a duplicate operation."""

    pass


class IdempotencyRecordedError(Exception):
    """Raised when replaying a recorded error from a previous attempt."""

    pass


# Global instance (lazy initialization)
_manager: Optional[IdempotencyManager] = None
_manager_lock = threading.Lock()


def get_idempotency_manager(
    config: Optional[IdempotencyConfig] = None,
) -> IdempotencyManager:
    """Get or create the global idempotency manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = IdempotencyManager(config or IdempotencyConfig())
    return _manager
