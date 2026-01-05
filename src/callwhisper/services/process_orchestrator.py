"""
Process Orchestrator with Circuit Breaker

Based on LibV2 orchestrator-architecture course:
- "Fallback chains represent ordered sequences of specialists"
- "Circuit breaker pattern for resilience"
- "Graceful degradation when preferred options are unavailable"
- "Load-aware routing prevents capacity exhaustion"
- "Exponential backoff with jitter prevents thundering herd"
"""

import time
import asyncio
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional, Tuple
from enum import Enum
from collections import deque

from ..core.logging_config import get_service_logger
from ..core.exceptions import CircuitOpenError, AllHandlersFailedError, ProcessTimeoutError
from ..core.tracing import (
    RequestContext,
    get_request_context,
    set_request_context,
    get_request_id
)

logger = get_service_logger()


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.5
) -> float:
    """
    Calculate backoff delay with jitter to prevent thundering herd.

    Based on LibV2 orchestrator-architecture course:
    - Exponential backoff increases delay exponentially
    - Jitter adds randomness to prevent synchronized retries

    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Random jitter as fraction of delay (0-1)

    Returns:
        Delay in seconds
    """
    # Exponential delay: base * 2^attempt, capped at max
    delay = min(base_delay * (2 ** attempt), max_delay)

    # Add random jitter
    jitter = delay * jitter_factor * random.random()
    final_delay = delay + jitter

    logger.debug(
        "backoff_calculated",
        attempt=attempt,
        base_delay=base_delay,
        calculated_delay=round(delay, 2),
        jitter=round(jitter, 2),
        final_delay=round(final_delay, 2)
    )

    return final_delay


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerState:
    """State tracking for a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0


@dataclass
class FallbackSpec:
    """Specification for a fallback handler."""
    handler: str
    timeout: float = 120.0
    retries: int = 1
    priority: int = 0


@dataclass
class ExecutionAttempt:
    """Record of an execution attempt."""
    handler: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class HandlerMetrics:
    """Metrics for load-aware routing."""
    queue_depth: int = 0
    durations_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def total_count(self) -> int:
        return self.error_count + self.success_count

    @property
    def error_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.error_count / self.total_count

    @property
    def p95_latency_ms(self) -> float:
        if not self.durations_ms:
            return 0.0
        sorted_durations = sorted(self.durations_ms)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]


class LoadAwareRouter:
    """
    Route to least-loaded handler in fallback chain.

    Based on LibV2 orchestrator-architecture course:
    - Calculate load score from queue depth, latency, error rate
    - Select handler with lowest load score
    - Prevent routing to overloaded handlers
    """

    def __init__(self, load_threshold: float = 0.8):
        """
        Initialize load-aware router.

        Args:
            load_threshold: Maximum load score to consider handler available (0-1)
        """
        self.load_threshold = load_threshold
        self._metrics: Dict[str, HandlerMetrics] = {}
        self._lock = threading.Lock()

    def _get_metrics(self, handler: str) -> HandlerMetrics:
        """Get or create metrics for a handler."""
        with self._lock:
            if handler not in self._metrics:
                self._metrics[handler] = HandlerMetrics()
            return self._metrics[handler]

    def record_start(self, handler: str) -> None:
        """Record that a handler started processing (increases queue depth)."""
        with self._lock:
            m = self._get_metrics(handler)
            m.queue_depth += 1
            m.last_updated = time.time()

    def record_completion(self, handler: str, duration_ms: float, success: bool) -> None:
        """Record handler completion with metrics."""
        with self._lock:
            m = self._get_metrics(handler)
            m.queue_depth = max(0, m.queue_depth - 1)
            m.durations_ms.append(duration_ms)
            if success:
                m.success_count += 1
            else:
                m.error_count += 1
            m.last_updated = time.time()

    def calculate_load_score(self, handler: str) -> float:
        """
        Calculate composite load score for a handler.

        Score is weighted combination of:
        - Queue depth (40%)
        - P95 latency (40%)
        - Error rate (20%)

        Returns:
            Score from 0 (no load) to 1+ (overloaded)
        """
        m = self._get_metrics(handler)

        # Normalize each factor to 0-1 range
        queue_factor = min(1.0, m.queue_depth / 10)  # Assume 10 is max reasonable queue
        latency_factor = min(1.0, m.p95_latency_ms / 10000)  # 10s is max reasonable
        error_factor = m.error_rate  # Already 0-1

        # Weighted combination
        score = (
            queue_factor * 0.4 +
            latency_factor * 0.4 +
            error_factor * 0.2
        )

        return score

    def select_best_handler(self, handlers: List[str]) -> str:
        """
        Select the best handler based on current load.

        Args:
            handlers: List of available handlers

        Returns:
            Handler with lowest load score
        """
        if not handlers:
            raise ValueError("No handlers provided")

        if len(handlers) == 1:
            return handlers[0]

        # Calculate scores for all handlers
        scores = [(h, self.calculate_load_score(h)) for h in handlers]

        # Filter to handlers below threshold
        available = [(h, s) for h, s in scores if s < self.load_threshold]

        if not available:
            # All overloaded, select least loaded
            best = min(scores, key=lambda x: x[1])
            logger.warning(
                "all_handlers_overloaded",
                selected=best[0],
                score=round(best[1], 3)
            )
            return best[0]

        # Select handler with lowest score
        best = min(available, key=lambda x: x[1])
        logger.debug(
            "handler_selected",
            handler=best[0],
            score=round(best[1], 3),
            alternatives=len(handlers) - 1
        )
        return best[0]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all handlers."""
        with self._lock:
            return {
                handler: {
                    "queue_depth": m.queue_depth,
                    "p95_latency_ms": round(m.p95_latency_ms, 2),
                    "error_rate": round(m.error_rate, 4),
                    "total_requests": m.total_count,
                    "load_score": round(self.calculate_load_score(handler), 3)
                }
                for handler, m in self._metrics.items()
            }


class ProcessOrchestrator:
    """
    Orchestrates subprocess execution with resilience patterns.

    Features:
    - Circuit breaker per handler to prevent cascading failures
    - Fallback chains for graceful degradation
    - Retry logic with exponential backoff and jitter
    - Load-aware routing to least-loaded handlers
    - Execution tracking and metrics
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        half_open_max_calls: int = 3,
        load_threshold: float = 0.8,
        base_backoff_delay: float = 1.0
    ):
        """
        Initialize the orchestrator.

        Args:
            failure_threshold: Consecutive failures to open circuit
            success_threshold: Successes in half-open to close circuit
            cooldown_seconds: Time before attempting half-open
            half_open_max_calls: Max calls to allow in half-open state
            load_threshold: Max load score for handler availability
            base_backoff_delay: Base delay for exponential backoff
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls
        self.base_backoff_delay = base_backoff_delay

        self._circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self._fallback_chains: Dict[str, List[FallbackSpec]] = {}
        self._half_open_calls: Dict[str, int] = {}
        self._load_router = LoadAwareRouter(load_threshold=load_threshold)

    def register_fallback_chain(
        self,
        task_type: str,
        chain: List[Dict[str, Any]]
    ) -> None:
        """
        Register a fallback chain for a task type.

        Example:
            orchestrator.register_fallback_chain('transcription', [
                {'handler': 'whisper_large', 'timeout': 300},
                {'handler': 'whisper_medium', 'timeout': 180},
                {'handler': 'whisper_small', 'timeout': 60},
            ])
        """
        specs = [
            FallbackSpec(
                handler=item['handler'],
                timeout=item.get('timeout', 120.0),
                retries=item.get('retries', 1),
                priority=idx
            )
            for idx, item in enumerate(chain)
        ]
        self._fallback_chains[task_type] = specs

        logger.info(
            "fallback_chain_registered",
            task_type=task_type,
            handlers=[s.handler for s in specs]
        )

    def _get_circuit_breaker(self, handler: str) -> CircuitBreakerState:
        """Get or create circuit breaker for handler."""
        if handler not in self._circuit_breakers:
            self._circuit_breakers[handler] = CircuitBreakerState()
        return self._circuit_breakers[handler]

    def _check_circuit(self, handler: str) -> Tuple[bool, Optional[str]]:
        """
        Check if circuit allows request.

        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        cb = self._get_circuit_breaker(handler)

        if cb.state == CircuitState.CLOSED:
            return True, None

        if cb.state == CircuitState.OPEN:
            # Check if cooldown has passed
            if cb.last_failure_time:
                elapsed = time.time() - cb.last_failure_time
                if elapsed >= self.cooldown_seconds:
                    # Transition to half-open
                    cb.state = CircuitState.HALF_OPEN
                    self._half_open_calls[handler] = 0
                    logger.info(
                        "circuit_half_open",
                        handler=handler,
                        cooldown_elapsed=elapsed
                    )
                    return True, None

            cooldown_remaining = self.cooldown_seconds - (time.time() - (cb.last_failure_time or 0))
            return False, f"circuit_open (cooldown: {cooldown_remaining:.1f}s remaining)"

        if cb.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            calls = self._half_open_calls.get(handler, 0)
            if calls < self.half_open_max_calls:
                self._half_open_calls[handler] = calls + 1
                return True, None
            return False, "circuit_half_open_limit_reached"

        return True, None

    def _is_circuit_open(self, handler: str) -> bool:
        """
        Check if circuit breaker is open for a handler.

        Returns:
            True if circuit is open (blocking requests), False otherwise
        """
        allowed, _ = self._check_circuit(handler)
        return not allowed

    def _record_success(self, handler: str) -> None:
        """Record successful execution."""
        cb = self._get_circuit_breaker(handler)
        cb.success_count += 1
        cb.last_success_time = time.time()
        cb.consecutive_failures = 0

        if cb.state == CircuitState.HALF_OPEN:
            # Check if we can close the circuit
            if cb.success_count >= self.success_threshold:
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                logger.info(
                    "circuit_closed",
                    handler=handler,
                    success_count=cb.success_count
                )

    def _record_failure(self, handler: str, error: str) -> None:
        """Record failed execution."""
        cb = self._get_circuit_breaker(handler)
        cb.failure_count += 1
        cb.consecutive_failures += 1
        cb.last_failure_time = time.time()

        logger.warning(
            "handler_failure",
            handler=handler,
            consecutive_failures=cb.consecutive_failures,
            total_failures=cb.failure_count,
            error=error
        )

        # Check if circuit should open
        if cb.consecutive_failures >= self.failure_threshold:
            cb.state = CircuitState.OPEN
            logger.warning(
                "circuit_opened",
                handler=handler,
                consecutive_failures=cb.consecutive_failures
            )

        # If in half-open and failed, go back to open
        if cb.state == CircuitState.HALF_OPEN:
            cb.state = CircuitState.OPEN
            logger.info(
                "circuit_reopened",
                handler=handler,
                reason="failure_in_half_open"
            )

    async def execute_with_fallback(
        self,
        task_type: str,
        execute_fn: Callable[[str, Any], Any],
        context: Optional[RequestContext] = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute task with fallback chain and circuit breaker.

        Args:
            task_type: Type of task (must have registered fallback chain)
            execute_fn: Async function taking (handler_name, **kwargs)
            context: Optional RequestContext for deadline propagation
            **kwargs: Arguments passed to execute_fn

        Returns:
            Tuple of (result, execution_metadata)

        Raises:
            AllHandlersFailedError: If all handlers in chain fail
            ProcessTimeoutError: If request deadline exceeded
        """
        # Get context from parameter or context var
        ctx = context or get_request_context()

        # Check if we have time remaining (deadline propagation)
        if ctx and not ctx.has_time_remaining():
            raise ProcessTimeoutError(
                f"Request deadline exceeded before starting {task_type}",
                task_type=task_type,
                correlation_id=ctx.correlation_id
            )

        chain = self._fallback_chains.get(task_type)
        if not chain:
            # No fallback chain - use default
            chain = [FallbackSpec(handler='default', timeout=120.0)]

        attempts: List[ExecutionAttempt] = []
        request_id = get_request_id() if not ctx else ctx.correlation_id

        # Record stage start
        if ctx:
            ctx.add_stage(f"{task_type}_start")

        for level, spec in enumerate(chain):
            handler = spec.handler

            # Check circuit breaker
            allowed, reason = self._check_circuit(handler)
            if not allowed:
                attempts.append(ExecutionAttempt(
                    handler=handler,
                    success=False,
                    duration_ms=0,
                    skipped_reason=reason
                ))
                continue

            # Attempt execution with retries
            for retry in range(spec.retries):
                # Apply backoff delay for retries (not first attempt)
                if retry > 0:
                    backoff_delay = exponential_backoff_with_jitter(
                        attempt=retry - 1,
                        base_delay=self.base_backoff_delay
                    )
                    logger.debug(
                        "retry_backoff",
                        handler=handler,
                        retry=retry,
                        backoff_seconds=round(backoff_delay, 2)
                    )
                    await asyncio.sleep(backoff_delay)

                # Check context deadline before attempt
                if ctx and not ctx.has_time_remaining():
                    logger.warning(
                        "deadline_exceeded_during_fallback",
                        task_type=task_type,
                        handler=handler,
                        correlation_id=ctx.correlation_id
                    )
                    break  # Exit fallback loop

                start_time = time.time()
                self._load_router.record_start(handler)

                # Calculate effective timeout (min of spec timeout and remaining deadline)
                effective_timeout = spec.timeout
                if ctx and spec.timeout > 0:
                    remaining = ctx.remaining_time()
                    effective_timeout = min(spec.timeout, remaining) if remaining > 0 else spec.timeout

                try:
                    result = await asyncio.wait_for(
                        execute_fn(handler, **kwargs),
                        timeout=effective_timeout if effective_timeout > 0 else None
                    )

                    duration_ms = (time.time() - start_time) * 1000
                    self._record_success(handler)
                    self._load_router.record_completion(handler, duration_ms, success=True)

                    attempts.append(ExecutionAttempt(
                        handler=handler,
                        success=True,
                        duration_ms=duration_ms
                    ))

                    # Record stage completion
                    if ctx:
                        ctx.add_stage(f"{task_type}_complete")

                    logger.info(
                        "task_executed",
                        task_type=task_type,
                        handler=handler,
                        fallback_level=level,
                        retry=retry,
                        duration_ms=round(duration_ms, 2),
                        correlation_id=ctx.correlation_id if ctx else request_id
                    )

                    return result, {
                        'handler': handler,
                        'fallback_level': level,
                        'attempts': [a.__dict__ for a in attempts],
                        'duration_ms': duration_ms,
                        'correlation_id': ctx.correlation_id if ctx else request_id,
                        'processing_history': ctx.processing_history if ctx else []
                    }

                except asyncio.TimeoutError:
                    duration_ms = (time.time() - start_time) * 1000
                    error = f"timeout after {spec.timeout}s"
                    self._record_failure(handler, error)
                    self._load_router.record_completion(handler, duration_ms, success=False)
                    attempts.append(ExecutionAttempt(
                        handler=handler,
                        success=False,
                        duration_ms=duration_ms,
                        error=error
                    ))

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    error = f"{type(e).__name__}: {str(e)}"
                    self._record_failure(handler, error)
                    self._load_router.record_completion(handler, duration_ms, success=False)
                    attempts.append(ExecutionAttempt(
                        handler=handler,
                        success=False,
                        duration_ms=duration_ms,
                        error=error
                    ))

        # Record failure stage
        if ctx:
            ctx.add_stage(f"{task_type}_failed")

        # All handlers failed
        logger.error(
            "all_handlers_failed",
            task_type=task_type,
            attempts=[a.__dict__ for a in attempts],
            correlation_id=ctx.correlation_id if ctx else request_id
        )
        raise AllHandlersFailedError(
            task_type=task_type,
            attempts=[a.__dict__ for a in attempts],
            correlation_id=ctx.correlation_id if ctx else None
        )

    def get_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            handler: {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count,
                'consecutive_failures': cb.consecutive_failures,
                'last_failure': cb.last_failure_time,
                'last_success': cb.last_success_time,
            }
            for handler, cb in self._circuit_breakers.items()
        }

    def reset_circuit(self, handler: str) -> None:
        """Manually reset a circuit breaker."""
        if handler in self._circuit_breakers:
            self._circuit_breakers[handler] = CircuitBreakerState()
            logger.info("circuit_manually_reset", handler=handler)

    def get_load_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get load metrics for all handlers."""
        return self._load_router.get_all_metrics()

    def get_best_handler(self, task_type: str) -> Optional[str]:
        """
        Get the best handler for a task type based on load.

        Args:
            task_type: Type of task

        Returns:
            Best available handler name, or None if no chain registered
        """
        chain = self._fallback_chains.get(task_type)
        if not chain:
            return None

        # Get handlers that aren't circuit-broken
        available_handlers = []
        for spec in chain:
            allowed, _ = self._check_circuit(spec.handler)
            if allowed:
                available_handlers.append(spec.handler)

        if not available_handlers:
            return None

        return self._load_router.select_best_handler(available_handlers)


# Global orchestrator instance
process_orchestrator = ProcessOrchestrator()

# Register default fallback chains
process_orchestrator.register_fallback_chain('transcription', [
    {'handler': 'whisper_large', 'timeout': 600, 'retries': 1},
    {'handler': 'whisper_medium', 'timeout': 300, 'retries': 2},
    {'handler': 'whisper_small', 'timeout': 120, 'retries': 2},
])

process_orchestrator.register_fallback_chain('normalization', [
    {'handler': 'ffmpeg', 'timeout': 60, 'retries': 2},
])

process_orchestrator.register_fallback_chain('recording', [
    {'handler': 'ffmpeg_dshow', 'timeout': 0, 'retries': 1},  # No timeout for recording
])
