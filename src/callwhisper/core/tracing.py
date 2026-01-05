"""
Distributed Tracing Module

Based on LibV2 orchestrator-architecture course:
- Correlation IDs for request tracking
- Context propagation across async operations
- Integration with structured logging
"""

import uuid
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Dict
from functools import wraps

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RequestContext:
    """
    Context envelope for request propagation across processing stages.

    Based on LibV2 orchestrator-architecture patterns:
    - Correlation IDs for end-to-end tracing
    - Deadline propagation for timeout cascade management
    - Processing history for debugging and audit
    """

    correlation_id: str
    deadline: float  # Unix timestamp when request should complete
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def remaining_time(self) -> float:
        """Get remaining time until deadline in seconds."""
        return max(0, self.deadline - time.time())

    def has_time_remaining(self) -> bool:
        """Check if there's still time before deadline."""
        return self.remaining_time() > 0

    def add_stage(self, stage_name: str) -> None:
        """Record a processing stage completion."""
        self.processing_history.append(f"{stage_name}@{time.time():.3f}")

    def get_child_context(self, stage_timeout: float) -> "RequestContext":
        """
        Create a child context for a sub-operation with reduced deadline.

        Args:
            stage_timeout: Maximum time this stage should take

        Returns:
            New RequestContext with deadline = min(parent_remaining, stage_timeout)
        """
        child_deadline = min(self.deadline, time.time() + stage_timeout)
        return RequestContext(
            correlation_id=self.correlation_id,
            deadline=child_deadline,
            source_metadata=self.source_metadata.copy(),
            processing_history=self.processing_history.copy(),
            created_at=self.created_at,
        )

    @classmethod
    def create(
        cls, timeout_seconds: float = 600.0, metadata: Optional[Dict[str, Any]] = None
    ) -> "RequestContext":
        """
        Create a new RequestContext with specified timeout.

        Args:
            timeout_seconds: Total timeout for the request (default 10 minutes)
            metadata: Optional source metadata to include

        Returns:
            New RequestContext instance
        """
        return cls(
            correlation_id=generate_request_id(),
            deadline=time.time() + timeout_seconds,
            source_metadata=metadata or {},
        )


# Context variable for RequestContext - thread-safe and async-safe
request_context_var: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_context", default=None
)


def get_request_context() -> Optional[RequestContext]:
    """Get the current request context from context var."""
    return request_context_var.get()


def set_request_context(ctx: Optional[RequestContext]) -> None:
    """Set the current request context."""
    request_context_var.set(ctx)


# Context variable for request tracing - thread-safe and async-safe
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a unique session ID for recordings."""
    return f"ses_{uuid.uuid4().hex[:16]}"


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_var.get()


def get_session_id() -> str:
    """Get the current session ID from context."""
    return session_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set the current request ID in context."""
    request_id_var.set(request_id)


def set_session_id(session_id: str) -> None:
    """Set the current session ID in context."""
    session_id_var.set(session_id)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request tracing to all HTTP requests.

    - Generates or propagates X-Request-ID header
    - Adds request_id to response headers
    - Logs request start/end with timing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", generate_request_id())
        set_request_id(request_id)

        # Log request start
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        # Process request
        import time

        start_time = time.time()

        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Add tracing headers to response
            response.headers["X-Request-ID"] = request_id

            # Log request completion
            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise


def traced(operation_name: Optional[str] = None):
    """
    Decorator to add tracing to functions.

    Usage:
        @traced("transcribe_audio")
        async def transcribe_audio(folder, settings):
            ...
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            request_id = get_request_id()
            session_id = get_session_id()

            logger.debug(
                "operation_started",
                operation=op_name,
                request_id=request_id,
                session_id=session_id,
            )

            import time

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    "operation_completed",
                    operation=op_name,
                    request_id=request_id,
                    session_id=session_id,
                    duration_ms=round(duration_ms, 2),
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    "operation_failed",
                    operation=op_name,
                    request_id=request_id,
                    session_id=session_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(duration_ms, 2),
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            request_id = get_request_id()
            session_id = get_session_id()

            logger.debug(
                "operation_started",
                operation=op_name,
                request_id=request_id,
                session_id=session_id,
            )

            import time

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                logger.debug(
                    "operation_completed",
                    operation=op_name,
                    request_id=request_id,
                    session_id=session_id,
                    duration_ms=round(duration_ms, 2),
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    "operation_failed",
                    operation=op_name,
                    request_id=request_id,
                    session_id=session_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(duration_ms, 2),
                )
                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class TraceContext:
    """
    Context manager for creating a traced scope.

    Usage:
        with TraceContext("processing_audio", session_id="ses_abc123"):
            # Operations within this scope will use the trace context
            await process_audio()
    """

    def __init__(
        self,
        operation_name: str,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.operation_name = operation_name
        self.request_id = request_id or get_request_id() or generate_request_id()
        self.session_id = session_id or get_session_id()
        self._previous_request_id: str = ""
        self._previous_session_id: str = ""

    def __enter__(self):
        self._previous_request_id = get_request_id()
        self._previous_session_id = get_session_id()
        set_request_id(self.request_id)
        if self.session_id:
            set_session_id(self.session_id)

        logger.debug(
            "trace_scope_entered",
            operation=self.operation_name,
            request_id=self.request_id,
            session_id=self.session_id,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(
                "trace_scope_error",
                operation=self.operation_name,
                request_id=self.request_id,
                session_id=self.session_id,
                error=str(exc_val),
                error_type=exc_type.__name__ if exc_type else None,
            )
        else:
            logger.debug(
                "trace_scope_exited",
                operation=self.operation_name,
                request_id=self.request_id,
                session_id=self.session_id,
            )

        # Restore previous context
        set_request_id(self._previous_request_id)
        set_session_id(self._previous_session_id)
        return False  # Don't suppress exceptions
