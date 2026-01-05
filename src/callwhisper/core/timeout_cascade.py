"""
Timeout Cascade Management Module

Based on LibV2 orchestrator-architecture patterns:
- Graceful Degradation (chunks 11104-11145)
- Hierarchical timeout budgets
- Fast failure for better UX

Key concepts:
- Global workflow deadline prevents indefinite hanging
- Each stage has its own timeout within the global budget
- Stages can't exceed their allocation OR remaining global time
- Enables predictable failure modes
"""

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

from .logging_config import get_core_logger
from .exceptions import ProcessTimeoutError

logger = get_core_logger()

T = TypeVar("T")


@dataclass
class TimeoutConfig:
    """
    Hierarchical timeout configuration.

    Workflow budget is the maximum total time for all stages combined.
    Individual stage timeouts are capped at both their own limit AND
    the remaining workflow time.

    Example:
        workflow_max=600s, transcribe=480s

        If normalize takes 120s, transcribe gets min(480, 600-120) = 480s
        If normalize takes 180s, transcribe gets min(480, 600-180) = 420s
    """

    workflow_max: float = 600.0  # 10 minutes total workflow

    stage_timeouts: Dict[str, float] = field(
        default_factory=lambda: {
            "upload": 120.0,  # 2 minutes for file upload
            "normalize": 60.0,  # 1 minute for FFmpeg normalization
            "transcribe": 480.0,  # 8 minutes for whisper.cpp
            "bundle": 60.0,  # 1 minute for output bundling
        }
    )

    # Minimum timeout for any stage (prevent too-short timeouts)
    min_stage_timeout: float = 5.0

    # Warning threshold (log warning when remaining time is low)
    warning_threshold_ratio: float = 0.2  # 20% remaining


@dataclass
class WorkflowContext:
    """
    Context for tracking workflow timeout state.

    Created at workflow start, passed through all stages.
    """

    __slots__ = (
        "workflow_id",
        "deadline",
        "started_at",
        "config",
        "completed_stages",
        "current_stage",
    )

    workflow_id: str
    deadline: float
    started_at: float
    config: TimeoutConfig
    completed_stages: Dict[str, float]  # stage -> duration
    current_stage: Optional[str]

    def __init__(self, workflow_id: str, config: TimeoutConfig = None):
        self.workflow_id = workflow_id
        self.config = config or TimeoutConfig()
        self.started_at = time.time()
        self.deadline = self.started_at + self.config.workflow_max
        self.completed_stages = {}
        self.current_stage = None


class TimeoutCascade:
    """
    Manage hierarchical timeouts across workflow stages.

    Ensures:
    1. No stage exceeds its individual timeout
    2. No stage exceeds remaining workflow time
    3. Predictable failure with useful error messages
    4. Logging of timeout consumption

    Example:
        cascade = TimeoutCascade()
        ctx = cascade.start_workflow("session_123")

        async with cascade.stage(ctx, "normalize"):
            await normalize_audio(audio_path)

        async with cascade.stage(ctx, "transcribe"):
            await transcribe_audio(audio_path)
    """

    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        self._active_workflows: Dict[str, WorkflowContext] = {}

    def start_workflow(
        self, workflow_id: str, config: TimeoutConfig = None
    ) -> WorkflowContext:
        """
        Start a new workflow with timeout tracking.

        Args:
            workflow_id: Unique identifier for the workflow
            config: Optional override for timeout config

        Returns:
            WorkflowContext to pass to stage() calls
        """
        ctx = WorkflowContext(workflow_id=workflow_id, config=config or self.config)
        self._active_workflows[workflow_id] = ctx

        logger.info(
            "timeout_workflow_start",
            workflow_id=workflow_id,
            deadline=ctx.deadline,
            workflow_max=ctx.config.workflow_max,
        )

        return ctx

    def end_workflow(self, ctx: WorkflowContext) -> Dict[str, Any]:
        """
        End a workflow and return timing summary.

        Returns:
            Dict with workflow timing statistics
        """
        total_duration = time.time() - ctx.started_at
        remaining = ctx.deadline - time.time()

        summary = {
            "workflow_id": ctx.workflow_id,
            "total_duration": round(total_duration, 2),
            "workflow_max": ctx.config.workflow_max,
            "time_remaining": round(max(0, remaining), 2),
            "utilization": round(total_duration / ctx.config.workflow_max, 3),
            "stages": ctx.completed_stages,
        }

        if ctx.workflow_id in self._active_workflows:
            del self._active_workflows[ctx.workflow_id]

        logger.info("timeout_workflow_end", **summary)
        return summary

    def get_remaining(self, ctx: WorkflowContext, stage: str) -> float:
        """
        Get effective timeout for a stage.

        Returns the minimum of:
        - Stage's configured timeout
        - Remaining workflow time

        Never returns less than min_stage_timeout.
        """
        remaining_workflow = ctx.deadline - time.time()
        stage_max = ctx.config.stage_timeouts.get(stage, 60.0)  # Default stage timeout

        effective = min(stage_max, remaining_workflow)

        # Enforce minimum
        if effective < ctx.config.min_stage_timeout:
            if remaining_workflow < ctx.config.min_stage_timeout:
                # Workflow is out of time
                return 0.0
            effective = ctx.config.min_stage_timeout

        return effective

    def check_deadline(self, ctx: WorkflowContext) -> None:
        """
        Check if workflow deadline has passed.

        Raises:
            ProcessTimeoutError if deadline exceeded
        """
        remaining = ctx.deadline - time.time()
        if remaining <= 0:
            elapsed = time.time() - ctx.started_at
            raise ProcessTimeoutError(
                f"Workflow {ctx.workflow_id} exceeded deadline. "
                f"Elapsed: {elapsed:.1f}s, Max: {ctx.config.workflow_max}s"
            )

        # Log warning if time is running low
        utilization = 1 - (remaining / ctx.config.workflow_max)
        if utilization > (1 - ctx.config.warning_threshold_ratio):
            logger.warning(
                "timeout_low_remaining",
                workflow_id=ctx.workflow_id,
                remaining=round(remaining, 1),
                utilization=round(utilization, 2),
            )

    @contextmanager
    def stage_sync(self, ctx: WorkflowContext, stage: str):
        """
        Synchronous context manager for a workflow stage.

        Tracks stage timing and enforces deadline.

        Raises:
            ProcessTimeoutError if stage or workflow times out
        """
        self.check_deadline(ctx)

        timeout = self.get_remaining(ctx, stage)
        if timeout <= 0:
            raise ProcessTimeoutError(
                f"No time remaining for stage '{stage}' "
                f"in workflow {ctx.workflow_id}"
            )

        ctx.current_stage = stage
        stage_start = time.time()

        logger.debug(
            "timeout_stage_start",
            workflow_id=ctx.workflow_id,
            stage=stage,
            timeout=round(timeout, 1),
        )

        try:
            yield timeout
        finally:
            duration = time.time() - stage_start
            ctx.completed_stages[stage] = duration
            ctx.current_stage = None

            logger.debug(
                "timeout_stage_end",
                workflow_id=ctx.workflow_id,
                stage=stage,
                duration=round(duration, 2),
                timeout_used=round(duration / timeout, 2) if timeout > 0 else 1.0,
            )

    async def stage(self, ctx: WorkflowContext, stage: str):
        """
        Async context manager for a workflow stage.

        Wraps async operations with timeout and tracking.
        """
        self.check_deadline(ctx)

        timeout = self.get_remaining(ctx, stage)
        if timeout <= 0:
            raise ProcessTimeoutError(
                f"No time remaining for stage '{stage}' "
                f"in workflow {ctx.workflow_id}"
            )

        ctx.current_stage = stage
        stage_start = time.time()

        logger.debug(
            "timeout_stage_start",
            workflow_id=ctx.workflow_id,
            stage=stage,
            timeout=round(timeout, 1),
        )

        class AsyncStageContext:
            def __init__(self, cascade, ctx, stage, timeout, stage_start):
                self.cascade = cascade
                self.ctx = ctx
                self.stage = stage
                self.timeout = timeout
                self.stage_start = stage_start

            async def __aenter__(self):
                return self.timeout

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.stage_start
                self.ctx.completed_stages[self.stage] = duration
                self.ctx.current_stage = None

                logger.debug(
                    "timeout_stage_end",
                    workflow_id=self.ctx.workflow_id,
                    stage=self.stage,
                    duration=round(duration, 2),
                    timeout_used=(
                        round(duration / self.timeout, 2) if self.timeout > 0 else 1.0
                    ),
                )

                return False  # Don't suppress exceptions

        return AsyncStageContext(self, ctx, stage, timeout, stage_start)

    async def run_with_timeout(self, ctx: WorkflowContext, stage: str, coro: Any) -> T:
        """
        Run a coroutine with stage timeout.

        Convenience method that handles the async context manager.

        Args:
            ctx: Workflow context
            stage: Stage name
            coro: Coroutine to execute

        Returns:
            Result of the coroutine

        Raises:
            ProcessTimeoutError if timeout exceeded
        """
        self.check_deadline(ctx)

        timeout = self.get_remaining(ctx, stage)
        if timeout <= 0:
            raise ProcessTimeoutError(f"No time remaining for stage '{stage}'")

        ctx.current_stage = stage
        stage_start = time.time()

        logger.debug(
            "timeout_stage_start",
            workflow_id=ctx.workflow_id,
            stage=stage,
            timeout=round(timeout, 1),
        )

        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise ProcessTimeoutError(
                f"Stage '{stage}' timed out after {timeout:.1f}s "
                f"in workflow {ctx.workflow_id}"
            )
        finally:
            duration = time.time() - stage_start
            ctx.completed_stages[stage] = duration
            ctx.current_stage = None

            logger.debug(
                "timeout_stage_end",
                workflow_id=ctx.workflow_id,
                stage=stage,
                duration=round(duration, 2),
            )

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active workflow."""
        ctx = self._active_workflows.get(workflow_id)
        if not ctx:
            return None

        now = time.time()
        remaining = ctx.deadline - now
        elapsed = now - ctx.started_at

        return {
            "workflow_id": workflow_id,
            "elapsed": round(elapsed, 2),
            "remaining": round(max(0, remaining), 2),
            "utilization": round(elapsed / ctx.config.workflow_max, 3),
            "current_stage": ctx.current_stage,
            "completed_stages": ctx.completed_stages,
            "is_expired": remaining <= 0,
        }

    def get_all_active(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active workflows."""
        return {wid: self.get_workflow_status(wid) for wid in self._active_workflows}


# Convenience function for simple use cases
def with_timeout(timeout: float):
    """
    Simple decorator for adding timeout to async functions.

    For more complex cases, use TimeoutCascade directly.

    Example:
        @with_timeout(30.0)
        async def process_audio(path):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                raise ProcessTimeoutError(f"{func.__name__} timed out after {timeout}s")

        return wrapper

    return decorator


# Global instance (lazy initialization)
_cascade: Optional[TimeoutCascade] = None


def get_timeout_cascade(config: TimeoutConfig = None) -> TimeoutCascade:
    """Get or create the global timeout cascade."""
    global _cascade
    if _cascade is None:
        _cascade = TimeoutCascade(config)
    return _cascade
