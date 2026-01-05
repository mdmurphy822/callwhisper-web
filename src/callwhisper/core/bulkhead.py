"""
Bulkhead Pattern Implementation

Based on LibV2 orchestrator-architecture course:
- Resource isolation with separate thread pools
- Prevents cascade failures across subsystems
- Graceful degradation under load

The bulkhead pattern isolates different components so that
failure in one area doesn't bring down the entire system.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, TypeVar, Any, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import time

from .logging_config import get_core_logger

logger = get_core_logger()

T = TypeVar('T')


class PoolType(str, Enum):
    """Types of isolated thread pools."""
    AUDIO = "audio"           # Audio recording/processing
    TRANSCRIPTION = "transcription"  # Whisper transcription
    IO = "io"                 # File I/O operations
    GENERAL = "general"       # General purpose tasks


@dataclass
class PoolConfig:
    """Configuration for a thread pool."""
    max_workers: int
    thread_name_prefix: str
    queue_size: int = 100  # Max pending tasks before rejection


@dataclass
class PoolMetrics:
    """Metrics for a thread pool."""
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    rejected_tasks: int = 0
    total_execution_time_ms: float = 0.0
    peak_active_tasks: int = 0

    @property
    def avg_execution_time_ms(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 0.0
        return self.total_execution_time_ms / total


class BulkheadExecutor:
    """
    Bulkhead pattern: isolate failures with separate thread pools.

    Each pool type has its own isolated resources, so failures
    in transcription won't affect audio recording, etc.

    Features:
    - Separate thread pools for different operation types
    - Queue depth limiting to prevent memory exhaustion
    - Metrics collection per pool
    - Graceful shutdown with timeout
    """

    # Default pool configurations
    DEFAULT_CONFIGS: Dict[PoolType, PoolConfig] = {
        PoolType.AUDIO: PoolConfig(max_workers=2, thread_name_prefix="audio"),
        PoolType.TRANSCRIPTION: PoolConfig(max_workers=2, thread_name_prefix="transcribe"),
        PoolType.IO: PoolConfig(max_workers=4, thread_name_prefix="io"),
        PoolType.GENERAL: PoolConfig(max_workers=4, thread_name_prefix="general"),
    }

    def __init__(self, custom_configs: Optional[Dict[PoolType, PoolConfig]] = None):
        """
        Initialize bulkhead executor with isolated thread pools.

        Args:
            custom_configs: Optional custom configurations for pools
        """
        self._lock = threading.Lock()
        self._configs = {**self.DEFAULT_CONFIGS}
        if custom_configs:
            self._configs.update(custom_configs)

        self._pools: Dict[PoolType, ThreadPoolExecutor] = {}
        self._metrics: Dict[PoolType, PoolMetrics] = {}
        self._pending_tasks: Dict[PoolType, int] = {}
        self._shutdown = False

        # Initialize pools
        for pool_type, config in self._configs.items():
            self._pools[pool_type] = ThreadPoolExecutor(
                max_workers=config.max_workers,
                thread_name_prefix=config.thread_name_prefix
            )
            self._metrics[pool_type] = PoolMetrics()
            self._pending_tasks[pool_type] = 0

        logger.info(
            "bulkhead_executor_initialized",
            pools={pt.value: cfg.max_workers for pt, cfg in self._configs.items()}
        )

    def _check_queue_limit(self, pool_type: PoolType) -> bool:
        """Check if pool can accept more tasks."""
        with self._lock:
            config = self._configs[pool_type]
            pending = self._pending_tasks[pool_type]
            return pending < config.queue_size

    def _record_task_start(self, pool_type: PoolType) -> None:
        """Record that a task has started."""
        with self._lock:
            self._pending_tasks[pool_type] += 1
            metrics = self._metrics[pool_type]
            metrics.active_tasks += 1
            if metrics.active_tasks > metrics.peak_active_tasks:
                metrics.peak_active_tasks = metrics.active_tasks

    def _record_task_end(
        self,
        pool_type: PoolType,
        duration_ms: float,
        success: bool
    ) -> None:
        """Record that a task has completed."""
        with self._lock:
            self._pending_tasks[pool_type] -= 1
            metrics = self._metrics[pool_type]
            metrics.active_tasks -= 1
            metrics.total_execution_time_ms += duration_ms
            if success:
                metrics.completed_tasks += 1
            else:
                metrics.failed_tasks += 1

    def _record_rejection(self, pool_type: PoolType) -> None:
        """Record that a task was rejected due to queue limit."""
        with self._lock:
            self._metrics[pool_type].rejected_tasks += 1

    async def run_in_pool(
        self,
        pool_type: PoolType,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Run a function in the specified thread pool.

        Args:
            pool_type: Which pool to use
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function

        Raises:
            RuntimeError: If executor is shutdown or queue is full
        """
        if self._shutdown:
            raise RuntimeError("BulkheadExecutor is shutdown")

        if not self._check_queue_limit(pool_type):
            self._record_rejection(pool_type)
            logger.warning(
                "bulkhead_task_rejected",
                pool=pool_type.value,
                reason="queue_full"
            )
            raise RuntimeError(f"Pool {pool_type.value} queue is full")

        pool = self._pools[pool_type]
        self._record_task_start(pool_type)

        start_time = time.time()
        success = True

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                pool,
                lambda: func(*args, **kwargs)
            )
            return result
        except Exception as e:
            success = False
            logger.error(
                "bulkhead_task_failed",
                pool=pool_type.value,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._record_task_end(pool_type, duration_ms, success)

    async def run_audio_task(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Convenience method to run in audio pool."""
        return await self.run_in_pool(PoolType.AUDIO, func, *args, **kwargs)

    async def run_transcription_task(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Convenience method to run in transcription pool."""
        return await self.run_in_pool(PoolType.TRANSCRIPTION, func, *args, **kwargs)

    async def run_io_task(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Convenience method to run in I/O pool."""
        return await self.run_in_pool(PoolType.IO, func, *args, **kwargs)

    def get_pool_metrics(self, pool_type: PoolType) -> Dict[str, Any]:
        """Get metrics for a specific pool."""
        with self._lock:
            metrics = self._metrics[pool_type]
            config = self._configs[pool_type]
            pending = self._pending_tasks[pool_type]
            return {
                "max_workers": config.max_workers,
                "queue_size": config.queue_size,
                "pending_tasks": pending,
                "active_tasks": metrics.active_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "rejected_tasks": metrics.rejected_tasks,
                "avg_execution_time_ms": round(metrics.avg_execution_time_ms, 2),
                "peak_active_tasks": metrics.peak_active_tasks,
                "utilization": round(metrics.active_tasks / config.max_workers, 2)
                    if config.max_workers > 0 else 0.0
            }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all pools."""
        return {
            pool_type.value: self.get_pool_metrics(pool_type)
            for pool_type in PoolType
        }

    def is_healthy(self) -> bool:
        """Check if all pools are healthy (not overloaded)."""
        for pool_type in PoolType:
            metrics = self.get_pool_metrics(pool_type)
            # Consider unhealthy if rejection rate > 10% or utilization > 90%
            total_submitted = (
                metrics["completed_tasks"] +
                metrics["failed_tasks"] +
                metrics["rejected_tasks"]
            )
            if total_submitted > 0:
                rejection_rate = metrics["rejected_tasks"] / total_submitted
                if rejection_rate > 0.1:
                    return False
            if metrics["utilization"] > 0.9:
                return False
        return True

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Shutdown all thread pools.

        Args:
            wait: Whether to wait for pending tasks
            timeout: Maximum time to wait for shutdown
        """
        self._shutdown = True

        logger.info("bulkhead_executor_shutting_down", wait=wait, timeout=timeout)

        for pool_type, pool in self._pools.items():
            try:
                pool.shutdown(wait=wait)
                logger.debug("pool_shutdown", pool=pool_type.value)
            except Exception as e:
                logger.error(
                    "pool_shutdown_error",
                    pool=pool_type.value,
                    error=str(e)
                )

        logger.info("bulkhead_executor_shutdown_complete")


# Global executor instance (lazy initialization)
_executor: Optional[BulkheadExecutor] = None
_executor_lock = threading.Lock()


def get_executor() -> BulkheadExecutor:
    """Get the global BulkheadExecutor instance."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = BulkheadExecutor()
    return _executor


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the global executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait)
        _executor = None
