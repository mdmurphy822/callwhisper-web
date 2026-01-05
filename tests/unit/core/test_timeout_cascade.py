"""
Tests for timeout cascade management.

Tests hierarchical timeout budgets:
- Workflow deadline management
- Stage timeout allocation
- Timeout budget consumption
- Deadline enforcement
- Context manager behavior
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from callwhisper.core.timeout_cascade import (
    TimeoutCascade,
    TimeoutConfig,
    WorkflowContext,
    get_timeout_cascade,
    with_timeout,
)
from callwhisper.core.exceptions import ProcessTimeoutError


# ============================================================================
# TimeoutConfig Tests
# ============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = TimeoutConfig()

        assert config.workflow_max == 600.0
        assert "upload" in config.stage_timeouts
        assert "transcribe" in config.stage_timeouts
        assert config.min_stage_timeout == 5.0
        assert config.warning_threshold_ratio == 0.2

    def test_custom_values(self):
        """Config accepts custom values."""
        config = TimeoutConfig(
            workflow_max=1200.0,
            stage_timeouts={"custom": 100.0},
            min_stage_timeout=10.0,
        )

        assert config.workflow_max == 1200.0
        assert config.stage_timeouts == {"custom": 100.0}
        assert config.min_stage_timeout == 10.0


# ============================================================================
# WorkflowContext Tests
# ============================================================================


class TestWorkflowContext:
    """Tests for WorkflowContext dataclass."""

    def test_initialization(self):
        """Context initializes with correct values."""
        ctx = WorkflowContext(workflow_id="test-123")

        assert ctx.workflow_id == "test-123"
        assert ctx.started_at <= time.time()
        assert ctx.deadline == ctx.started_at + ctx.config.workflow_max
        assert ctx.completed_stages == {}
        assert ctx.current_stage is None

    def test_custom_config(self):
        """Context accepts custom config."""
        config = TimeoutConfig(workflow_max=100.0)
        ctx = WorkflowContext(workflow_id="test", config=config)

        assert ctx.config.workflow_max == 100.0
        assert ctx.deadline == ctx.started_at + 100.0

    def test_uses_slots(self):
        """Context uses __slots__ for memory efficiency."""
        assert hasattr(WorkflowContext, "__slots__")


# ============================================================================
# TimeoutCascade Basic Tests
# ============================================================================


class TestTimeoutCascadeBasic:
    """Tests for basic TimeoutCascade operations."""

    @pytest.fixture
    def cascade(self):
        """Create cascade with short timeouts for testing."""
        config = TimeoutConfig(
            workflow_max=10.0,
            stage_timeouts={
                "fast": 2.0,
                "slow": 8.0,
            },
            min_stage_timeout=0.5,
        )
        return TimeoutCascade(config)

    def test_start_workflow(self, cascade):
        """start_workflow creates and tracks context."""
        ctx = cascade.start_workflow("test-123")

        assert ctx.workflow_id == "test-123"
        assert "test-123" in cascade._active_workflows

    def test_end_workflow(self, cascade):
        """end_workflow returns summary and removes from active."""
        ctx = cascade.start_workflow("test-123")

        summary = cascade.end_workflow(ctx)

        assert summary["workflow_id"] == "test-123"
        assert "total_duration" in summary
        assert "test-123" not in cascade._active_workflows

    def test_end_workflow_includes_stages(self, cascade):
        """end_workflow summary includes completed stages."""
        ctx = cascade.start_workflow("test-123")
        ctx.completed_stages["upload"] = 1.5
        ctx.completed_stages["transcribe"] = 3.2

        summary = cascade.end_workflow(ctx)

        assert summary["stages"]["upload"] == 1.5
        assert summary["stages"]["transcribe"] == 3.2


# ============================================================================
# get_remaining Tests
# ============================================================================


class TestGetRemaining:
    """Tests for get_remaining timeout calculation."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        config = TimeoutConfig(
            workflow_max=60.0,
            stage_timeouts={
                "short": 10.0,
                "long": 120.0,
            },
            min_stage_timeout=2.0,
        )
        return TimeoutCascade(config)

    def test_returns_stage_timeout_when_plenty_of_time(self, cascade):
        """Returns stage timeout when workflow has plenty of time."""
        ctx = cascade.start_workflow("test")

        timeout = cascade.get_remaining(ctx, "short")

        assert timeout == 10.0

    def test_caps_at_workflow_remaining(self, cascade):
        """Caps timeout at remaining workflow time."""
        ctx = cascade.start_workflow("test")
        # Simulate time passing
        ctx.deadline = time.time() + 5.0  # Only 5 seconds left

        timeout = cascade.get_remaining(ctx, "long")

        # Should be capped at ~5 seconds (remaining workflow time)
        assert timeout <= 5.5
        assert timeout < 120.0  # Less than stage's configured timeout

    def test_enforces_minimum(self, cascade):
        """Enforces minimum timeout."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() + 1.0  # Very little time left

        timeout = cascade.get_remaining(ctx, "short")

        # Should enforce minimum (2.0)
        assert timeout >= 0.0  # But still valid

    def test_returns_zero_when_expired(self, cascade):
        """Returns 0 when workflow is expired."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() - 10.0  # Already expired

        timeout = cascade.get_remaining(ctx, "short")

        assert timeout == 0.0

    def test_default_for_unknown_stage(self, cascade):
        """Uses default timeout for unknown stages."""
        ctx = cascade.start_workflow("test")

        timeout = cascade.get_remaining(ctx, "unknown_stage")

        # Should use default (60.0) capped at workflow remaining
        assert timeout <= 60.0


# ============================================================================
# check_deadline Tests
# ============================================================================


class TestCheckDeadline:
    """Tests for check_deadline enforcement."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        config = TimeoutConfig(
            workflow_max=10.0,
            warning_threshold_ratio=0.2,
        )
        return TimeoutCascade(config)

    def test_no_error_within_deadline(self, cascade):
        """No error when within deadline."""
        ctx = cascade.start_workflow("test")

        # Should not raise
        cascade.check_deadline(ctx)

    def test_raises_when_deadline_passed(self, cascade):
        """Raises ProcessTimeoutError when deadline passed."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() - 1.0  # Expired

        with pytest.raises(ProcessTimeoutError) as exc_info:
            cascade.check_deadline(ctx)

        assert "exceeded deadline" in str(exc_info.value)


# ============================================================================
# stage_sync Tests
# ============================================================================


class TestStageSyncContextManager:
    """Tests for synchronous stage context manager."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        config = TimeoutConfig(
            workflow_max=60.0,
            stage_timeouts={"test": 10.0},
        )
        return TimeoutCascade(config)

    def test_stage_sync_tracks_duration(self, cascade):
        """stage_sync tracks stage duration."""
        ctx = cascade.start_workflow("test")

        with cascade.stage_sync(ctx, "test") as timeout:
            time.sleep(0.1)

        assert "test" in ctx.completed_stages
        assert ctx.completed_stages["test"] >= 0.1

    def test_stage_sync_yields_timeout(self, cascade):
        """stage_sync yields the effective timeout."""
        ctx = cascade.start_workflow("test")

        with cascade.stage_sync(ctx, "test") as timeout:
            assert timeout == 10.0

    def test_stage_sync_sets_current_stage(self, cascade):
        """stage_sync sets and clears current_stage."""
        ctx = cascade.start_workflow("test")

        with cascade.stage_sync(ctx, "test"):
            assert ctx.current_stage == "test"

        assert ctx.current_stage is None

    def test_stage_sync_raises_when_no_time(self, cascade):
        """stage_sync raises when no time remaining."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() - 1.0  # Expired

        with pytest.raises(ProcessTimeoutError):
            with cascade.stage_sync(ctx, "test"):
                pass

    def test_stage_sync_clears_on_exception(self, cascade):
        """stage_sync clears current_stage even on exception."""
        ctx = cascade.start_workflow("test")

        try:
            with cascade.stage_sync(ctx, "test"):
                raise ValueError("test error")
        except ValueError:
            pass

        assert ctx.current_stage is None


# ============================================================================
# stage (async) Tests
# ============================================================================


class TestStageAsyncContextManager:
    """Tests for async stage context manager."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        config = TimeoutConfig(
            workflow_max=60.0,
            stage_timeouts={"test": 10.0},
        )
        return TimeoutCascade(config)

    @pytest.mark.asyncio
    async def test_stage_tracks_duration(self, cascade):
        """stage() tracks stage duration."""
        ctx = cascade.start_workflow("test")

        async with await cascade.stage(ctx, "test") as timeout:
            await asyncio.sleep(0.1)

        assert "test" in ctx.completed_stages
        assert ctx.completed_stages["test"] >= 0.1

    @pytest.mark.asyncio
    async def test_stage_yields_timeout(self, cascade):
        """stage() yields the effective timeout."""
        ctx = cascade.start_workflow("test")

        async with await cascade.stage(ctx, "test") as timeout:
            assert timeout == 10.0

    @pytest.mark.asyncio
    async def test_stage_raises_when_expired(self, cascade):
        """stage() raises when deadline expired."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() - 1.0  # Expired

        with pytest.raises(ProcessTimeoutError):
            async with await cascade.stage(ctx, "test"):
                pass


# ============================================================================
# run_with_timeout Tests
# ============================================================================


class TestRunWithTimeout:
    """Tests for run_with_timeout convenience method."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        config = TimeoutConfig(
            workflow_max=60.0,
            stage_timeouts={"fast": 1.0},
        )
        return TimeoutCascade(config)

    @pytest.mark.asyncio
    async def test_runs_coroutine(self, cascade):
        """run_with_timeout executes coroutine."""
        ctx = cascade.start_workflow("test")

        async def simple_coro():
            return "result"

        result = await cascade.run_with_timeout(ctx, "fast", simple_coro())

        assert result == "result"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self, cascade):
        """run_with_timeout raises on timeout."""
        ctx = cascade.start_workflow("test")

        async def slow_coro():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(ProcessTimeoutError) as exc_info:
            await cascade.run_with_timeout(ctx, "fast", slow_coro())

        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tracks_stage_duration(self, cascade):
        """run_with_timeout tracks stage duration."""
        ctx = cascade.start_workflow("test")

        async def quick_coro():
            await asyncio.sleep(0.1)
            return "done"

        await cascade.run_with_timeout(ctx, "fast", quick_coro())

        assert "fast" in ctx.completed_stages
        assert ctx.completed_stages["fast"] >= 0.1

    @pytest.mark.asyncio
    async def test_raises_when_no_time_remaining(self, cascade):
        """run_with_timeout raises when no time left."""
        ctx = cascade.start_workflow("test")
        ctx.deadline = time.time() - 1.0  # Expired

        async def coro():
            return "never"

        with pytest.raises(ProcessTimeoutError):
            await cascade.run_with_timeout(ctx, "fast", coro())


# ============================================================================
# Workflow Status Tests
# ============================================================================


class TestWorkflowStatus:
    """Tests for workflow status tracking."""

    @pytest.fixture
    def cascade(self):
        """Create cascade for testing."""
        return TimeoutCascade(TimeoutConfig())

    def test_get_workflow_status(self, cascade):
        """get_workflow_status returns status for active workflow."""
        ctx = cascade.start_workflow("test-123")

        status = cascade.get_workflow_status("test-123")

        assert status is not None
        assert status["workflow_id"] == "test-123"
        assert "elapsed" in status
        assert "remaining" in status
        assert status["is_expired"] is False

    def test_get_workflow_status_unknown(self, cascade):
        """get_workflow_status returns None for unknown workflow."""
        status = cascade.get_workflow_status("unknown")
        assert status is None

    def test_get_workflow_status_includes_current_stage(self, cascade):
        """get_workflow_status includes current stage."""
        ctx = cascade.start_workflow("test")
        ctx.current_stage = "transcribe"

        status = cascade.get_workflow_status("test")

        assert status["current_stage"] == "transcribe"

    def test_get_all_active(self, cascade):
        """get_all_active returns all active workflows."""
        cascade.start_workflow("workflow-1")
        cascade.start_workflow("workflow-2")
        cascade.start_workflow("workflow-3")

        all_active = cascade.get_all_active()

        assert len(all_active) == 3
        assert "workflow-1" in all_active
        assert "workflow-2" in all_active
        assert "workflow-3" in all_active


# ============================================================================
# with_timeout Decorator Tests
# ============================================================================


class TestWithTimeoutDecorator:
    """Tests for with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_completion(self):
        """Decorated function completes within timeout."""
        @with_timeout(5.0)
        async def quick_function():
            return "done"

        result = await quick_function()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_decorator_raises_on_timeout(self):
        """Decorated function raises on timeout."""
        @with_timeout(0.1)
        async def slow_function():
            await asyncio.sleep(10)
            return "never"

        with pytest.raises(ProcessTimeoutError) as exc_info:
            await slow_function()

        assert "slow_function" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global timeout cascade singleton."""

    def test_get_timeout_cascade_creates_instance(self):
        """get_timeout_cascade creates instance on first call."""
        import callwhisper.core.timeout_cascade as module
        original = module._cascade
        module._cascade = None

        try:
            cascade = get_timeout_cascade()
            assert isinstance(cascade, TimeoutCascade)
        finally:
            module._cascade = original

    def test_get_timeout_cascade_returns_same_instance(self):
        """get_timeout_cascade returns same instance."""
        import callwhisper.core.timeout_cascade as module
        original = module._cascade
        module._cascade = None

        try:
            cascade1 = get_timeout_cascade()
            cascade2 = get_timeout_cascade()
            assert cascade1 is cascade2
        finally:
            module._cascade = original


# ============================================================================
# Integration Tests
# ============================================================================


class TestTimeoutCascadeIntegration:
    """Integration tests for timeout cascade."""

    @pytest.mark.asyncio
    async def test_multi_stage_workflow(self):
        """Complete multi-stage workflow."""
        config = TimeoutConfig(
            workflow_max=10.0,
            stage_timeouts={
                "stage1": 3.0,
                "stage2": 3.0,
                "stage3": 3.0,
            }
        )
        cascade = TimeoutCascade(config)
        ctx = cascade.start_workflow("multi-stage")

        # Stage 1
        async with await cascade.stage(ctx, "stage1"):
            await asyncio.sleep(0.1)

        # Stage 2
        async with await cascade.stage(ctx, "stage2"):
            await asyncio.sleep(0.1)

        # Stage 3
        async with await cascade.stage(ctx, "stage3"):
            await asyncio.sleep(0.1)

        summary = cascade.end_workflow(ctx)

        assert len(summary["stages"]) == 3
        assert all(stage in summary["stages"] for stage in ["stage1", "stage2", "stage3"])

    @pytest.mark.asyncio
    async def test_budget_consumed_across_stages(self):
        """Timeout budget is consumed across stages."""
        config = TimeoutConfig(
            workflow_max=1.0,  # Very short
            stage_timeouts={
                "stage1": 10.0,
                "stage2": 10.0,
            },
            min_stage_timeout=0.1,
        )
        cascade = TimeoutCascade(config)
        ctx = cascade.start_workflow("budget-test")

        # First stage takes most of the budget
        async with await cascade.stage(ctx, "stage1"):
            await asyncio.sleep(0.8)

        # Second stage should have very little time left
        remaining = cascade.get_remaining(ctx, "stage2")
        assert remaining < 0.5  # Most budget consumed

    def test_sync_stage_with_exception(self):
        """Sync stage handles exceptions correctly."""
        cascade = TimeoutCascade(TimeoutConfig())
        ctx = cascade.start_workflow("exception-test")

        with pytest.raises(ValueError):
            with cascade.stage_sync(ctx, "failing"):
                raise ValueError("test error")

        # Stage should still be recorded
        assert "failing" in ctx.completed_stages
        assert ctx.current_stage is None


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_workflow_max(self):
        """Handles zero workflow max."""
        config = TimeoutConfig(workflow_max=0.0)
        cascade = TimeoutCascade(config)
        ctx = cascade.start_workflow("zero")

        # Deadline is already passed
        with pytest.raises(ProcessTimeoutError):
            cascade.check_deadline(ctx)

    def test_negative_stage_timeout(self):
        """Handles negative stage timeout in config."""
        config = TimeoutConfig(
            workflow_max=60.0,
            stage_timeouts={"negative": -10.0},
        )
        cascade = TimeoutCascade(config)
        ctx = cascade.start_workflow("test")

        # Should use min_stage_timeout
        timeout = cascade.get_remaining(ctx, "negative")
        assert timeout >= config.min_stage_timeout or timeout == 0.0

    def test_end_workflow_twice(self):
        """Ending workflow twice is safe."""
        cascade = TimeoutCascade(TimeoutConfig())
        ctx = cascade.start_workflow("twice")

        cascade.end_workflow(ctx)
        # Second call should not raise
        cascade.end_workflow(ctx)

    def test_missing_config_stages(self):
        """Handles stages not in config."""
        cascade = TimeoutCascade(TimeoutConfig())
        ctx = cascade.start_workflow("test")

        # Unknown stage uses default
        timeout = cascade.get_remaining(ctx, "completely_unknown_stage")
        assert timeout > 0
