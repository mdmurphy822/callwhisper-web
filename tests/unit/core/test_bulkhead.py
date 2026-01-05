"""
Tests for bulkhead pattern implementation.

Tests thread pool isolation:
- Pool type configuration
- Task execution in pools
- Queue limiting and rejection
- Metrics tracking
- Health checks
- Graceful shutdown
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from callwhisper.core.bulkhead import (
    BulkheadExecutor,
    PoolConfig,
    PoolMetrics,
    PoolType,
    get_executor,
    shutdown_executor,
)


# ============================================================================
# PoolType Tests
# ============================================================================


class TestPoolType:
    """Tests for PoolType enum."""

    def test_pool_type_values(self):
        """Pool types have expected string values."""
        assert PoolType.AUDIO.value == "audio"
        assert PoolType.TRANSCRIPTION.value == "transcription"
        assert PoolType.IO.value == "io"
        assert PoolType.GENERAL.value == "general"

    def test_pool_type_from_string(self):
        """Pool types can be created from strings."""
        assert PoolType("audio") == PoolType.AUDIO
        assert PoolType("transcription") == PoolType.TRANSCRIPTION


# ============================================================================
# PoolConfig Tests
# ============================================================================


class TestPoolConfig:
    """Tests for PoolConfig dataclass."""

    def test_config_creation(self):
        """Config can be created with parameters."""
        config = PoolConfig(
            max_workers=8,
            thread_name_prefix="custom",
            queue_size=200
        )

        assert config.max_workers == 8
        assert config.thread_name_prefix == "custom"
        assert config.queue_size == 200

    def test_config_default_queue_size(self):
        """Config has default queue size of 100."""
        config = PoolConfig(
            max_workers=4,
            thread_name_prefix="test"
        )

        assert config.queue_size == 100


# ============================================================================
# PoolMetrics Tests
# ============================================================================


class TestPoolMetrics:
    """Tests for PoolMetrics dataclass."""

    def test_default_metrics(self):
        """Metrics have expected defaults."""
        metrics = PoolMetrics()

        assert metrics.active_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.rejected_tasks == 0
        assert metrics.total_execution_time_ms == 0.0
        assert metrics.peak_active_tasks == 0

    def test_avg_execution_time_no_tasks(self):
        """Average execution time is 0 with no tasks."""
        metrics = PoolMetrics()
        assert metrics.avg_execution_time_ms == 0.0

    def test_avg_execution_time_calculated(self):
        """Average execution time is correctly calculated."""
        metrics = PoolMetrics()
        metrics.completed_tasks = 3
        metrics.failed_tasks = 1
        metrics.total_execution_time_ms = 400.0

        # 400 / 4 = 100
        assert metrics.avg_execution_time_ms == 100.0


# ============================================================================
# BulkheadExecutor Basic Tests
# ============================================================================


class TestBulkheadExecutorBasic:
    """Tests for BulkheadExecutor basic operations."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=False)

    def test_default_pools_created(self, executor):
        """Default pools are created on initialization."""
        assert PoolType.AUDIO in executor._pools
        assert PoolType.TRANSCRIPTION in executor._pools
        assert PoolType.IO in executor._pools
        assert PoolType.GENERAL in executor._pools

    def test_custom_config_applied(self):
        """Custom configurations are applied."""
        custom_config = {
            PoolType.AUDIO: PoolConfig(max_workers=10, thread_name_prefix="custom-audio")
        }
        executor = BulkheadExecutor(custom_configs=custom_config)

        try:
            assert executor._configs[PoolType.AUDIO].max_workers == 10
        finally:
            executor.shutdown(wait=False)

    def test_metrics_initialized(self, executor):
        """Metrics are initialized for each pool."""
        for pool_type in PoolType:
            assert pool_type in executor._metrics
            assert isinstance(executor._metrics[pool_type], PoolMetrics)


# ============================================================================
# BulkheadExecutor Task Execution Tests
# ============================================================================


class TestBulkheadExecutorTasks:
    """Tests for task execution in pools."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_run_in_pool_success(self, executor):
        """Task runs successfully in pool."""
        def add(a, b):
            return a + b

        result = await executor.run_in_pool(PoolType.GENERAL, add, 2, 3)

        assert result == 5

    @pytest.mark.asyncio
    async def test_run_in_pool_with_kwargs(self, executor):
        """Task receives kwargs correctly."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = await executor.run_in_pool(
            PoolType.GENERAL, greet, "World", greeting="Hi"
        )

        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_run_in_pool_exception_propagates(self, executor):
        """Exceptions from task are propagated."""
        def failing_task():
            raise ValueError("Task failed")

        with pytest.raises(ValueError, match="Task failed"):
            await executor.run_in_pool(PoolType.GENERAL, failing_task)

    @pytest.mark.asyncio
    async def test_run_audio_task_convenience(self, executor):
        """Convenience method runs in audio pool."""
        def task():
            return "audio_result"

        result = await executor.run_audio_task(task)

        assert result == "audio_result"

    @pytest.mark.asyncio
    async def test_run_transcription_task_convenience(self, executor):
        """Convenience method runs in transcription pool."""
        def task():
            return "transcription_result"

        result = await executor.run_transcription_task(task)

        assert result == "transcription_result"

    @pytest.mark.asyncio
    async def test_run_io_task_convenience(self, executor):
        """Convenience method runs in I/O pool."""
        def task():
            return "io_result"

        result = await executor.run_io_task(task)

        assert result == "io_result"

    @pytest.mark.asyncio
    async def test_task_after_shutdown_raises(self, executor):
        """Running task after shutdown raises error."""
        executor.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="shutdown"):
            await executor.run_in_pool(PoolType.GENERAL, lambda: None)


# ============================================================================
# BulkheadExecutor Queue Limiting Tests
# ============================================================================


class TestBulkheadQueueLimiting:
    """Tests for queue depth limiting."""

    @pytest.mark.asyncio
    async def test_queue_limit_rejects_task(self):
        """Tasks are rejected when queue is full."""
        # Small queue size for testing
        custom_config = {
            PoolType.GENERAL: PoolConfig(
                max_workers=1,
                thread_name_prefix="test",
                queue_size=1
            )
        }
        executor = BulkheadExecutor(custom_configs=custom_config)

        try:
            # Fill the queue
            blocking_event = threading.Event()

            def blocking_task():
                blocking_event.wait(timeout=5)
                return "done"

            # Start first task (will be running)
            task1 = asyncio.create_task(
                executor.run_in_pool(PoolType.GENERAL, blocking_task)
            )
            await asyncio.sleep(0.1)  # Let it start

            # Start second task (will be pending)
            task2 = asyncio.create_task(
                executor.run_in_pool(PoolType.GENERAL, blocking_task)
            )
            await asyncio.sleep(0.1)

            # Third task should be rejected (queue full)
            with pytest.raises(RuntimeError, match="queue is full"):
                await executor.run_in_pool(PoolType.GENERAL, lambda: None)

            # Cleanup
            blocking_event.set()
            await task1
            # task2 may also be rejected due to timing, ignore if so
            try:
                await task2
            except RuntimeError:
                pass  # Expected if queue filled before task2 was added

        finally:
            executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_rejection_tracked_in_metrics(self):
        """Rejected tasks are tracked in metrics."""
        custom_config = {
            PoolType.GENERAL: PoolConfig(
                max_workers=1,
                thread_name_prefix="test",
                queue_size=0  # Reject immediately
            )
        }
        executor = BulkheadExecutor(custom_configs=custom_config)

        try:
            with pytest.raises(RuntimeError):
                await executor.run_in_pool(PoolType.GENERAL, lambda: None)

            metrics = executor.get_pool_metrics(PoolType.GENERAL)
            assert metrics["rejected_tasks"] == 1

        finally:
            executor.shutdown(wait=False)


# ============================================================================
# BulkheadExecutor Metrics Tests
# ============================================================================


class TestBulkheadMetrics:
    """Tests for metrics tracking."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_completed_tasks_tracked(self, executor):
        """Completed tasks are tracked in metrics."""
        await executor.run_in_pool(PoolType.GENERAL, lambda: None)
        await executor.run_in_pool(PoolType.GENERAL, lambda: None)

        metrics = executor.get_pool_metrics(PoolType.GENERAL)

        assert metrics["completed_tasks"] == 2
        assert metrics["failed_tasks"] == 0

    @pytest.mark.asyncio
    async def test_failed_tasks_tracked(self, executor):
        """Failed tasks are tracked in metrics."""
        try:
            await executor.run_in_pool(
                PoolType.GENERAL, lambda: (_ for _ in ()).throw(Exception("fail"))
            )
        except Exception:
            pass

        metrics = executor.get_pool_metrics(PoolType.GENERAL)

        assert metrics["failed_tasks"] == 1

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self, executor):
        """Execution time is tracked in metrics."""
        def slow_task():
            time.sleep(0.05)
            return "done"

        await executor.run_in_pool(PoolType.GENERAL, slow_task)

        metrics = executor.get_pool_metrics(PoolType.GENERAL)

        assert metrics["avg_execution_time_ms"] >= 40  # At least 40ms

    @pytest.mark.asyncio
    async def test_peak_active_tasks_tracked(self, executor):
        """Peak active tasks are tracked."""
        event = threading.Event()

        def blocking_task():
            event.wait(timeout=5)
            return "done"

        # Start multiple tasks concurrently
        tasks = [
            asyncio.create_task(
                executor.run_in_pool(PoolType.GENERAL, blocking_task)
            )
            for _ in range(3)
        ]

        await asyncio.sleep(0.1)  # Let tasks start

        metrics = executor.get_pool_metrics(PoolType.GENERAL)
        assert metrics["peak_active_tasks"] >= 1

        # Release tasks
        event.set()
        await asyncio.gather(*tasks)

    def test_get_all_metrics(self, executor):
        """get_all_metrics returns metrics for all pools."""
        all_metrics = executor.get_all_metrics()

        assert "audio" in all_metrics
        assert "transcription" in all_metrics
        assert "io" in all_metrics
        assert "general" in all_metrics

    def test_metrics_include_config(self, executor):
        """Metrics include configuration values."""
        metrics = executor.get_pool_metrics(PoolType.AUDIO)

        assert "max_workers" in metrics
        assert "queue_size" in metrics
        assert metrics["max_workers"] == 2  # Default for audio

    def test_utilization_calculated(self, executor):
        """Utilization is correctly calculated."""
        metrics = executor.get_pool_metrics(PoolType.GENERAL)

        assert "utilization" in metrics
        assert metrics["utilization"] == 0.0  # No active tasks


# ============================================================================
# BulkheadExecutor Health Check Tests
# ============================================================================


class TestBulkheadHealthCheck:
    """Tests for health check functionality."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=False)

    def test_healthy_with_no_load(self, executor):
        """Executor is healthy with no load."""
        assert executor.is_healthy() is True

    @pytest.mark.asyncio
    async def test_healthy_after_successful_tasks(self, executor):
        """Executor is healthy after successful tasks."""
        for _ in range(5):
            await executor.run_in_pool(PoolType.GENERAL, lambda: None)

        assert executor.is_healthy() is True

    def test_unhealthy_high_rejection_rate(self):
        """Executor is unhealthy with high rejection rate."""
        executor = BulkheadExecutor()

        try:
            # Manually set metrics to simulate high rejection
            executor._metrics[PoolType.GENERAL].completed_tasks = 8
            executor._metrics[PoolType.GENERAL].rejected_tasks = 2  # 20% rejection

            assert executor.is_healthy() is False

        finally:
            executor.shutdown(wait=False)


# ============================================================================
# BulkheadExecutor Shutdown Tests
# ============================================================================


class TestBulkheadShutdown:
    """Tests for shutdown functionality."""

    def test_shutdown_sets_flag(self):
        """Shutdown sets shutdown flag."""
        executor = BulkheadExecutor()
        executor.shutdown(wait=False)

        assert executor._shutdown is True

    def test_shutdown_with_wait(self):
        """Shutdown waits for pending tasks."""
        executor = BulkheadExecutor()

        # Should not hang or raise
        executor.shutdown(wait=True)

    def test_shutdown_without_wait(self):
        """Shutdown can skip waiting."""
        executor = BulkheadExecutor()

        # Should return immediately
        start = time.time()
        executor.shutdown(wait=False)
        duration = time.time() - start

        assert duration < 1.0  # Should be fast


# ============================================================================
# Global Executor Tests
# ============================================================================


class TestGlobalExecutor:
    """Tests for global executor singleton."""

    def test_get_executor_creates_instance(self):
        """get_executor creates executor on first call."""
        # Reset global state
        import callwhisper.core.bulkhead as module
        original = module._executor
        module._executor = None

        try:
            executor = get_executor()
            assert isinstance(executor, BulkheadExecutor)
        finally:
            if module._executor:
                module._executor.shutdown(wait=False)
            module._executor = original

    def test_get_executor_returns_same_instance(self):
        """get_executor returns same instance."""
        import callwhisper.core.bulkhead as module
        original = module._executor
        module._executor = None

        try:
            executor1 = get_executor()
            executor2 = get_executor()
            assert executor1 is executor2
        finally:
            if module._executor:
                module._executor.shutdown(wait=False)
            module._executor = original

    def test_shutdown_executor(self):
        """shutdown_executor shuts down and clears global."""
        import callwhisper.core.bulkhead as module
        original = module._executor

        try:
            module._executor = BulkheadExecutor()
            shutdown_executor(wait=False)

            assert module._executor is None
        finally:
            module._executor = original


# ============================================================================
# Pool Isolation Tests
# ============================================================================


class TestPoolIsolation:
    """Tests for pool isolation properties."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_pools_are_independent(self, executor):
        """Each pool type has independent workers."""
        results = {"audio": False, "transcription": False, "io": False}

        def set_result(pool_name):
            results[pool_name] = True
            return pool_name

        await asyncio.gather(
            executor.run_in_pool(PoolType.AUDIO, set_result, "audio"),
            executor.run_in_pool(PoolType.TRANSCRIPTION, set_result, "transcription"),
            executor.run_in_pool(PoolType.IO, set_result, "io"),
        )

        assert all(results.values())

    @pytest.mark.asyncio
    async def test_pool_metrics_isolated(self, executor):
        """Metrics are tracked separately per pool."""
        await executor.run_in_pool(PoolType.AUDIO, lambda: None)
        await executor.run_in_pool(PoolType.AUDIO, lambda: None)
        await executor.run_in_pool(PoolType.IO, lambda: None)

        audio_metrics = executor.get_pool_metrics(PoolType.AUDIO)
        io_metrics = executor.get_pool_metrics(PoolType.IO)
        transcription_metrics = executor.get_pool_metrics(PoolType.TRANSCRIPTION)

        assert audio_metrics["completed_tasks"] == 2
        assert io_metrics["completed_tasks"] == 1
        assert transcription_metrics["completed_tasks"] == 0


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    @pytest.fixture
    def executor(self):
        """Create executor for testing."""
        executor = BulkheadExecutor()
        yield executor
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_concurrent_submissions(self, executor):
        """Concurrent task submissions are thread-safe."""
        results = []

        async def submit_tasks():
            for i in range(10):
                result = await executor.run_in_pool(
                    PoolType.GENERAL, lambda x=i: x * 2
                )
                results.append(result)

        await asyncio.gather(
            submit_tasks(),
            submit_tasks(),
            submit_tasks(),
        )

        assert len(results) == 30

    @pytest.mark.asyncio
    async def test_concurrent_metric_reads(self, executor):
        """Concurrent metric reads are thread-safe."""
        errors = []

        async def read_metrics():
            try:
                for _ in range(100):
                    metrics = executor.get_all_metrics()
                    assert "general" in metrics
            except Exception as e:
                errors.append(e)

        await asyncio.gather(
            read_metrics(),
            read_metrics(),
            read_metrics(),
        )

        assert len(errors) == 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_none_result_handled(self):
        """None result from task is handled correctly."""
        executor = BulkheadExecutor()

        try:
            result = await executor.run_in_pool(
                PoolType.GENERAL, lambda: None
            )
            assert result is None
        finally:
            executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_empty_function(self):
        """Empty function executes correctly."""
        executor = BulkheadExecutor()

        try:
            def empty():
                pass

            result = await executor.run_in_pool(PoolType.GENERAL, empty)
            assert result is None
        finally:
            executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_function_with_side_effects(self):
        """Function with side effects works correctly."""
        executor = BulkheadExecutor()
        container = {"value": 0}

        try:
            def increment():
                container["value"] += 1
                return container["value"]

            result = await executor.run_in_pool(PoolType.GENERAL, increment)
            assert result == 1
            assert container["value"] == 1
        finally:
            executor.shutdown(wait=False)
