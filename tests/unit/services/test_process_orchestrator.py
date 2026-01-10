"""
Unit tests for Process Orchestrator module.

Tests:
- Exponential backoff with jitter
- Circuit breaker state transitions
- Fallback chain execution
- Load-aware routing
- Handler metrics tracking
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from callwhisper.services.process_orchestrator import (
    exponential_backoff_with_jitter,
    CircuitState,
    CircuitBreakerState,
    FallbackSpec,
    ExecutionAttempt,
    HandlerMetrics,
    LoadAwareRouter,
    ProcessOrchestrator,
)
from callwhisper.core.exceptions import (
    CircuitOpenError,
    AllHandlersFailedError,
    ProcessTimeoutError,
)


# ============================================================================
# Exponential Backoff Tests
# ============================================================================

class TestExponentialBackoff:
    """Tests for exponential_backoff_with_jitter function."""

    def test_first_attempt_returns_base_delay_range(self):
        """First attempt (0) returns delay in range [base, base * (1 + jitter)]."""
        delays = [exponential_backoff_with_jitter(0, base_delay=1.0, jitter_factor=0.5)
                  for _ in range(100)]

        # All delays should be between 1.0 and 1.5 (base + max jitter)
        for delay in delays:
            assert 1.0 <= delay <= 1.5

    def test_exponential_growth(self):
        """Delay grows exponentially with attempts."""
        delay_0 = exponential_backoff_with_jitter(0, base_delay=1.0, jitter_factor=0)
        delay_1 = exponential_backoff_with_jitter(1, base_delay=1.0, jitter_factor=0)
        delay_2 = exponential_backoff_with_jitter(2, base_delay=1.0, jitter_factor=0)
        delay_3 = exponential_backoff_with_jitter(3, base_delay=1.0, jitter_factor=0)

        assert delay_0 == 1.0   # 1 * 2^0 = 1
        assert delay_1 == 2.0   # 1 * 2^1 = 2
        assert delay_2 == 4.0   # 1 * 2^2 = 4
        assert delay_3 == 8.0   # 1 * 2^3 = 8

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        delay = exponential_backoff_with_jitter(
            10, base_delay=1.0, max_delay=10.0, jitter_factor=0
        )
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Jitter adds random variation to delays."""
        delays = [exponential_backoff_with_jitter(0, base_delay=1.0, jitter_factor=0.5)
                  for _ in range(100)]

        # With jitter, not all delays should be exactly 1.0
        unique_delays = set(delays)
        assert len(unique_delays) > 1

    def test_zero_jitter_is_deterministic(self):
        """Zero jitter produces deterministic delays."""
        delays = [exponential_backoff_with_jitter(2, base_delay=1.0, jitter_factor=0)
                  for _ in range(10)]

        assert all(d == 4.0 for d in delays)

    def test_custom_base_delay(self):
        """Custom base delay is respected."""
        delay = exponential_backoff_with_jitter(0, base_delay=2.0, jitter_factor=0)
        assert delay == 2.0

    def test_large_attempt_number(self):
        """Large attempt numbers don't cause overflow."""
        delay = exponential_backoff_with_jitter(
            100, base_delay=1.0, max_delay=60.0, jitter_factor=0
        )
        assert delay == 60.0


# ============================================================================
# Circuit Breaker State Tests
# ============================================================================

class TestCircuitBreakerState:
    """Tests for CircuitBreakerState dataclass."""

    def test_default_values(self):
        """CircuitBreakerState has correct defaults."""
        state = CircuitBreakerState()

        assert state.state == CircuitState.CLOSED
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.consecutive_failures == 0
        assert state.last_failure_time is None
        assert state.last_success_time is None

    def test_all_states_exist(self):
        """All circuit states are defined."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


# ============================================================================
# Handler Metrics Tests
# ============================================================================

class TestHandlerMetrics:
    """Tests for HandlerMetrics dataclass."""

    def test_default_values(self):
        """HandlerMetrics has correct defaults."""
        metrics = HandlerMetrics()

        assert metrics.queue_depth == 0
        assert metrics.error_count == 0
        assert metrics.success_count == 0
        assert len(metrics.durations_ms) == 0

    def test_total_count(self):
        """total_count returns sum of errors and successes."""
        metrics = HandlerMetrics(error_count=5, success_count=10)
        assert metrics.total_count == 15

    def test_error_rate_empty(self):
        """error_rate is 0 when no requests."""
        metrics = HandlerMetrics()
        assert metrics.error_rate == 0.0

    def test_error_rate_calculated(self):
        """error_rate is calculated correctly."""
        metrics = HandlerMetrics(error_count=3, success_count=7)
        assert metrics.error_rate == 0.3

    def test_p95_latency_empty(self):
        """p95_latency_ms is 0 when no durations."""
        metrics = HandlerMetrics()
        assert metrics.p95_latency_ms == 0.0

    def test_p95_latency_calculated(self):
        """p95_latency_ms is calculated correctly."""
        metrics = HandlerMetrics()
        # Add 100 durations from 1-100
        for i in range(1, 101):
            metrics.durations_ms.append(float(i))

        # p95 should be around 95
        assert 94 <= metrics.p95_latency_ms <= 96

    def test_durations_maxlen(self):
        """durations_ms respects maxlen of 100."""
        metrics = HandlerMetrics()
        for i in range(200):
            metrics.durations_ms.append(float(i))

        assert len(metrics.durations_ms) == 100
        # Should contain 100-199, not 0-99
        assert metrics.durations_ms[0] == 100.0


# ============================================================================
# Load Aware Router Tests
# ============================================================================

class TestLoadAwareRouter:
    """Tests for LoadAwareRouter class."""

    @pytest.fixture
    def router(self):
        """Create a LoadAwareRouter for testing."""
        return LoadAwareRouter(load_threshold=0.8)

    def test_record_start_increases_queue_depth(self, router):
        """record_start increases queue depth."""
        router.record_start("handler1")
        metrics = router.get_all_metrics()

        assert metrics["handler1"]["queue_depth"] == 1

    def test_record_completion_decreases_queue_depth(self, router):
        """record_completion decreases queue depth."""
        router.record_start("handler1")
        router.record_start("handler1")
        router.record_completion("handler1", 100.0, success=True)

        metrics = router.get_all_metrics()
        assert metrics["handler1"]["queue_depth"] == 1

    def test_record_completion_success(self, router):
        """record_completion tracks success metrics."""
        router.record_completion("handler1", 100.0, success=True)
        router.record_completion("handler1", 200.0, success=True)

        metrics = router.get_all_metrics()
        assert metrics["handler1"]["total_requests"] == 2
        assert metrics["handler1"]["error_rate"] == 0.0

    def test_record_completion_failure(self, router):
        """record_completion tracks failure metrics."""
        router.record_completion("handler1", 100.0, success=False)
        router.record_completion("handler1", 200.0, success=True)

        metrics = router.get_all_metrics()
        assert metrics["handler1"]["total_requests"] == 2
        assert metrics["handler1"]["error_rate"] == 0.5

    def test_calculate_load_score_empty(self, router):
        """calculate_load_score for new handler."""
        score = router.calculate_load_score("new_handler")
        assert score == 0.0

    def test_calculate_load_score_with_queue(self, router):
        """calculate_load_score includes queue depth."""
        for _ in range(5):
            router.record_start("busy_handler")

        score = router.calculate_load_score("busy_handler")
        assert score > 0

    def test_select_best_handler_single(self, router):
        """select_best_handler returns only handler when single option."""
        best = router.select_best_handler(["only_handler"])
        assert best == "only_handler"

    def test_select_best_handler_empty_raises(self, router):
        """select_best_handler raises on empty list."""
        with pytest.raises(ValueError, match="No handlers provided"):
            router.select_best_handler([])

    def test_select_best_handler_least_loaded(self, router):
        """select_best_handler picks least loaded."""
        # Make handler2 busy
        for _ in range(5):
            router.record_start("handler2")

        # handler1 should be preferred
        best = router.select_best_handler(["handler1", "handler2"])
        assert best == "handler1"

    def test_select_best_handler_all_overloaded(self, router):
        """select_best_handler picks least bad when all overloaded."""
        # Overload both handlers (threshold is 0.8)
        for _ in range(10):
            router.record_start("handler1")
            router.record_start("handler2")

        # Both overloaded, should pick one with lower score
        best = router.select_best_handler(["handler1", "handler2"])
        assert best in ["handler1", "handler2"]

    def test_queue_depth_never_negative(self, router):
        """Queue depth doesn't go negative on extra completions."""
        router.record_completion("handler1", 100.0, success=True)
        router.record_completion("handler1", 100.0, success=True)

        metrics = router.get_all_metrics()
        assert metrics["handler1"]["queue_depth"] == 0


# ============================================================================
# Process Orchestrator - Circuit Breaker Tests
# ============================================================================

class TestOrchestratorCircuitBreaker:
    """Tests for circuit breaker functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create ProcessOrchestrator with short timeouts for testing."""
        return ProcessOrchestrator(
            failure_threshold=3,
            success_threshold=2,
            cooldown_seconds=0.1,
            half_open_max_calls=2,
        )

    def test_circuit_starts_closed(self, orchestrator):
        """Circuit starts in closed state."""
        allowed, reason = orchestrator._check_circuit("handler1")
        assert allowed is True
        assert reason is None

    def test_circuit_opens_after_failures(self, orchestrator):
        """Circuit opens after reaching failure threshold."""
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        allowed, reason = orchestrator._check_circuit("handler1")
        assert allowed is False
        assert "circuit_open" in reason

    def test_circuit_transitions_to_half_open(self, orchestrator):
        """Circuit transitions to half-open after cooldown."""
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        # Wait for cooldown
        time.sleep(0.15)

        allowed, reason = orchestrator._check_circuit("handler1")
        assert allowed is True

        status = orchestrator.get_circuit_status()
        assert status["handler1"]["state"] == "half_open"

    def test_half_open_closes_on_success(self, orchestrator):
        """Circuit closes after successes in half-open state."""
        # Open circuit
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        # Wait for half-open
        time.sleep(0.15)
        orchestrator._check_circuit("handler1")

        # Record successes
        for _ in range(2):
            orchestrator._record_success("handler1")

        status = orchestrator.get_circuit_status()
        assert status["handler1"]["state"] == "closed"

    def test_half_open_reopens_on_failure(self, orchestrator):
        """Circuit reopens on failure in half-open state."""
        # Open circuit
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        # Wait for half-open
        time.sleep(0.15)
        orchestrator._check_circuit("handler1")

        # Record failure
        orchestrator._record_failure("handler1", "test error")

        status = orchestrator.get_circuit_status()
        assert status["handler1"]["state"] == "open"

    def test_half_open_limits_calls(self, orchestrator):
        """Half-open state limits number of test calls."""
        # Open circuit
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        # Wait for half-open
        time.sleep(0.15)

        # First two calls allowed
        allowed1, _ = orchestrator._check_circuit("handler1")
        allowed2, _ = orchestrator._check_circuit("handler1")

        # Third call rejected
        allowed3, reason = orchestrator._check_circuit("handler1")

        assert allowed1 is True
        assert allowed2 is True
        assert allowed3 is False
        assert "half_open_limit_reached" in reason

    def test_reset_circuit(self, orchestrator):
        """reset_circuit restores circuit to closed state."""
        # Open circuit
        for _ in range(3):
            orchestrator._record_failure("handler1", "test error")

        orchestrator.reset_circuit("handler1")

        status = orchestrator.get_circuit_status()
        assert status["handler1"]["state"] == "closed"
        assert status["handler1"]["failure_count"] == 0

    def test_success_resets_consecutive_failures(self, orchestrator):
        """Success resets consecutive failure count."""
        orchestrator._record_failure("handler1", "error")
        orchestrator._record_failure("handler1", "error")
        orchestrator._record_success("handler1")

        status = orchestrator.get_circuit_status()
        assert status["handler1"]["consecutive_failures"] == 0


# ============================================================================
# Process Orchestrator - Fallback Chain Tests
# ============================================================================

class TestOrchestratorFallbackChain:
    """Tests for fallback chain functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create ProcessOrchestrator with test fallback chain."""
        orch = ProcessOrchestrator(
            failure_threshold=2,
            cooldown_seconds=60,
        )
        orch.register_fallback_chain('test_task', [
            {'handler': 'primary', 'timeout': 1.0, 'retries': 1},
            {'handler': 'secondary', 'timeout': 1.0, 'retries': 1},
            {'handler': 'fallback', 'timeout': 1.0, 'retries': 1},
        ])
        return orch

    def test_register_fallback_chain(self, orchestrator):
        """register_fallback_chain stores chain correctly."""
        chains = orchestrator._fallback_chains

        assert 'test_task' in chains
        assert len(chains['test_task']) == 3
        assert chains['test_task'][0].handler == 'primary'
        assert chains['test_task'][1].handler == 'secondary'
        assert chains['test_task'][2].handler == 'fallback'

    @pytest.mark.asyncio
    async def test_execute_with_fallback_success(self, orchestrator):
        """execute_with_fallback succeeds with first handler."""
        async def execute_fn(handler, **kwargs):
            return f"result from {handler}"

        result, metadata = await orchestrator.execute_with_fallback(
            'test_task', execute_fn
        )

        assert result == "result from primary"
        assert metadata['handler'] == 'primary'
        assert metadata['fallback_level'] == 0

    @pytest.mark.asyncio
    async def test_execute_with_fallback_uses_secondary(self, orchestrator):
        """execute_with_fallback falls back on primary failure."""
        call_count = 0

        async def execute_fn(handler, **kwargs):
            nonlocal call_count
            call_count += 1
            if handler == 'primary':
                raise RuntimeError("Primary failed")
            return f"result from {handler}"

        result, metadata = await orchestrator.execute_with_fallback(
            'test_task', execute_fn
        )

        assert result == "result from secondary"
        assert metadata['handler'] == 'secondary'
        assert metadata['fallback_level'] == 1

    @pytest.mark.asyncio
    async def test_execute_with_fallback_all_fail(self, orchestrator):
        """execute_with_fallback raises when all handlers fail."""
        async def execute_fn(handler, **kwargs):
            raise RuntimeError(f"{handler} failed")

        with pytest.raises(AllHandlersFailedError) as exc:
            await orchestrator.execute_with_fallback('test_task', execute_fn)

        assert exc.value.task_type == 'test_task'

    @pytest.mark.asyncio
    async def test_execute_with_fallback_timeout(self, orchestrator):
        """execute_with_fallback handles timeout."""
        async def execute_fn(handler, **kwargs):
            await asyncio.sleep(2)  # Longer than 1.0s handler timeout, shorter than pytest timeout
            return "result"

        with pytest.raises(AllHandlersFailedError):
            await orchestrator.execute_with_fallback('test_task', execute_fn)

    @pytest.mark.asyncio
    async def test_execute_with_fallback_skips_open_circuit(self, orchestrator):
        """execute_with_fallback skips handlers with open circuit."""
        # Open circuit for primary
        for _ in range(2):
            orchestrator._record_failure('primary', 'error')

        async def execute_fn(handler, **kwargs):
            return f"result from {handler}"

        result, metadata = await orchestrator.execute_with_fallback(
            'test_task', execute_fn
        )

        # Should skip primary and use secondary
        assert result == "result from secondary"
        assert metadata['handler'] == 'secondary'

        # Check attempts show primary was skipped
        attempts = metadata['attempts']
        primary_attempt = next(a for a in attempts if a['handler'] == 'primary')
        assert primary_attempt['skipped_reason'] is not None

    @pytest.mark.asyncio
    async def test_execute_with_fallback_no_chain(self, orchestrator):
        """execute_with_fallback uses default for unregistered task."""
        async def execute_fn(handler, **kwargs):
            return f"result from {handler}"

        result, metadata = await orchestrator.execute_with_fallback(
            'unknown_task', execute_fn
        )

        assert metadata['handler'] == 'default'

    @pytest.mark.asyncio
    async def test_execute_tracks_duration(self, orchestrator):
        """execute_with_fallback tracks execution duration."""
        async def execute_fn(handler, **kwargs):
            await asyncio.sleep(0.1)
            return "result"

        result, metadata = await orchestrator.execute_with_fallback(
            'test_task', execute_fn
        )

        assert metadata['duration_ms'] >= 100


# ============================================================================
# Process Orchestrator - Best Handler Selection Tests
# ============================================================================

class TestOrchestratorBestHandler:
    """Tests for get_best_handler functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create ProcessOrchestrator with test chain."""
        orch = ProcessOrchestrator()
        orch.register_fallback_chain('test', [
            {'handler': 'h1'},
            {'handler': 'h2'},
            {'handler': 'h3'},
        ])
        return orch

    def test_get_best_handler_returns_available(self, orchestrator):
        """get_best_handler returns handler from chain."""
        best = orchestrator.get_best_handler('test')
        assert best in ['h1', 'h2', 'h3']

    def test_get_best_handler_unknown_task(self, orchestrator):
        """get_best_handler returns None for unknown task."""
        best = orchestrator.get_best_handler('nonexistent')
        assert best is None

    def test_get_best_handler_skips_broken_circuits(self, orchestrator):
        """get_best_handler skips handlers with open circuits."""
        # Open circuit for h1
        for _ in range(5):
            orchestrator._record_failure('h1', 'error')

        best = orchestrator.get_best_handler('test')
        assert best in ['h2', 'h3']

    def test_get_best_handler_all_circuits_open(self, orchestrator):
        """get_best_handler returns None when all circuits open."""
        for handler in ['h1', 'h2', 'h3']:
            for _ in range(5):
                orchestrator._record_failure(handler, 'error')

        best = orchestrator.get_best_handler('test')
        assert best is None


# ============================================================================
# Process Orchestrator - Load Metrics Tests
# ============================================================================

class TestOrchestratorLoadMetrics:
    """Tests for load metrics functionality."""

    @pytest.fixture
    def orchestrator(self):
        """Create ProcessOrchestrator."""
        return ProcessOrchestrator()

    def test_get_load_metrics_empty(self, orchestrator):
        """get_load_metrics returns empty dict initially."""
        metrics = orchestrator.get_load_metrics()
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_load_metrics_updated_on_execution(self, orchestrator):
        """Load metrics are updated after execution."""
        orchestrator.register_fallback_chain('test', [
            {'handler': 'h1', 'timeout': 1.0},
        ])

        async def execute_fn(handler, **kwargs):
            await asyncio.sleep(0.05)
            return "result"

        await orchestrator.execute_with_fallback('test', execute_fn)

        metrics = orchestrator.get_load_metrics()
        assert 'h1' in metrics
        assert metrics['h1']['total_requests'] == 1


# ============================================================================
# Concurrency Tests
# ============================================================================

class TestOrchestratorConcurrency:
    """Tests for thread safety."""

    @pytest.fixture
    def orchestrator(self):
        """Create ProcessOrchestrator."""
        return ProcessOrchestrator(failure_threshold=5)

    def test_concurrent_circuit_updates(self, orchestrator):
        """Circuit breaker handles concurrent updates."""
        errors = []

        def record_failures():
            try:
                for _ in range(100):
                    orchestrator._record_failure("handler1", "error")
            except Exception as e:
                errors.append(e)

        def record_successes():
            try:
                for _ in range(100):
                    orchestrator._record_success("handler1")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record_failures),
                executor.submit(record_successes),
                executor.submit(record_failures),
                executor.submit(record_successes),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_concurrent_executions(self, orchestrator):
        """Multiple concurrent executions are handled."""
        orchestrator.register_fallback_chain('test', [
            {'handler': 'h1', 'timeout': 1.0},
        ])

        call_count = 0

        async def execute_fn(handler, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return "result"

        # Run 10 concurrent executions
        tasks = [
            orchestrator.execute_with_fallback('test', execute_fn)
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert call_count == 10


# ============================================================================
# FallbackSpec and ExecutionAttempt Tests
# ============================================================================

class TestFallbackSpec:
    """Tests for FallbackSpec dataclass."""

    def test_default_values(self):
        """FallbackSpec has correct defaults."""
        spec = FallbackSpec(handler="test")

        assert spec.handler == "test"
        assert spec.timeout == 120.0
        assert spec.retries == 1
        assert spec.priority == 0


class TestExecutionAttempt:
    """Tests for ExecutionAttempt dataclass."""

    def test_success_attempt(self):
        """ExecutionAttempt for successful execution."""
        attempt = ExecutionAttempt(
            handler="h1",
            success=True,
            duration_ms=150.5,
        )

        assert attempt.handler == "h1"
        assert attempt.success is True
        assert attempt.duration_ms == 150.5
        assert attempt.error is None
        assert attempt.skipped_reason is None

    def test_failed_attempt(self):
        """ExecutionAttempt for failed execution."""
        attempt = ExecutionAttempt(
            handler="h1",
            success=False,
            duration_ms=50.0,
            error="Connection timeout",
        )

        assert attempt.success is False
        assert attempt.error == "Connection timeout"

    def test_skipped_attempt(self):
        """ExecutionAttempt for skipped handler."""
        attempt = ExecutionAttempt(
            handler="h1",
            success=False,
            duration_ms=0,
            skipped_reason="circuit_open",
        )

        assert attempt.skipped_reason == "circuit_open"
