"""
Rate Limiting Module

Based on LibV2 orchestrator-architecture patterns:
- Sliding window rate limiting
- Per-client request tracking
- Configurable limits with burst support
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from .logging_config import get_core_logger

logger = get_core_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    burst_size: int = 10
    enabled: bool = True
    # Paths to exclude from rate limiting
    excluded_paths: List[str] = field(
        default_factory=lambda: [
            "/api/health",
            "/api/health/ready",
            "/api/health/metrics",
        ]
    )


@dataclass
class ClientMetrics:
    """Metrics for a single client."""

    request_times: List[float] = field(default_factory=list)
    blocked_count: int = 0
    last_blocked: Optional[float] = None


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with per-client tracking.

    Features:
    - Thread-safe with Lock()
    - Configurable requests per minute
    - Burst allowance for short spikes
    - Automatic cleanup of old entries
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.Lock()
        self._clients: Dict[str, ClientMetrics] = defaultdict(ClientMetrics)
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_entries(self, now: float) -> None:
        """Remove old request times and inactive clients."""
        if now - self._last_cleanup < self._cleanup_interval:
            return

        window_start = now - 60

        # Clean up old request times
        clients_to_remove = []
        for client_id, metrics in self._clients.items():
            metrics.request_times = [
                t for t in metrics.request_times if t > window_start
            ]
            # Remove inactive clients (no requests in last 10 minutes)
            if not metrics.request_times and (
                metrics.last_blocked is None or now - metrics.last_blocked > 600
            ):
                clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            del self._clients[client_id]

        self._last_cleanup = now

        if clients_to_remove:
            logger.debug("rate_limiter_cleanup", removed_clients=len(clients_to_remove))

    def is_allowed(self, request: Request) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit.

        Returns:
            Tuple of (is_allowed, metadata)
            metadata contains: remaining, limit, reset_time
        """
        if not self.config.enabled:
            return True, {"remaining": -1, "limit": -1, "reset": 0}

        # Skip rate limiting for excluded paths
        if request.url.path in self.config.excluded_paths:
            return True, {"remaining": -1, "limit": -1, "reset": 0}

        now = time.time()
        window_start = now - 60
        client_id = self._get_client_id(request)

        with self._lock:
            self._cleanup_old_entries(now)

            metrics = self._clients[client_id]

            # Remove requests outside window
            metrics.request_times = [
                t for t in metrics.request_times if t > window_start
            ]

            current_count = len(metrics.request_times)
            remaining = self.config.requests_per_minute - current_count

            # Calculate reset time (when oldest request expires)
            if metrics.request_times:
                reset_time = int(metrics.request_times[0] + 60 - now)
            else:
                reset_time = 60

            metadata = {
                "remaining": max(0, remaining - 1),
                "limit": self.config.requests_per_minute,
                "reset": reset_time,
            }

            if current_count >= self.config.requests_per_minute:
                metrics.blocked_count += 1
                metrics.last_blocked = now

                logger.warning(
                    "rate_limit_exceeded",
                    client_id=client_id,
                    request_count=current_count,
                    limit=self.config.requests_per_minute,
                    blocked_count=metrics.blocked_count,
                )

                return False, metadata

            # Allow the request
            metrics.request_times.append(now)
            return True, metadata

    def get_client_stats(self, client_id: str) -> Optional[Dict]:
        """Get statistics for a specific client."""
        with self._lock:
            if client_id not in self._clients:
                return None

            metrics = self._clients[client_id]
            now = time.time()
            window_start = now - 60

            recent_requests = [t for t in metrics.request_times if t > window_start]

            return {
                "requests_in_window": len(recent_requests),
                "limit": self.config.requests_per_minute,
                "blocked_count": metrics.blocked_count,
                "last_blocked": metrics.last_blocked,
            }

    def get_all_stats(self) -> Dict:
        """Get statistics for all clients."""
        with self._lock:
            now = time.time()
            window_start = now - 60

            stats = {
                "total_clients": len(self._clients),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "burst_size": self.config.burst_size,
                    "enabled": self.config.enabled,
                },
                "clients": {},
            }

            for client_id, metrics in self._clients.items():
                recent = len([t for t in metrics.request_times if t > window_start])
                stats["clients"][client_id] = {
                    "requests_in_window": recent,
                    "blocked_count": metrics.blocked_count,
                }

            return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Adds rate limit headers to responses:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Remaining requests in window
    - X-RateLimit-Reset: Seconds until window resets
    """

    def __init__(self, app, rate_limiter: SlidingWindowRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        allowed, metadata = self.rate_limiter.is_allowed(request)

        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": metadata["reset"],
                },
            )
        else:
            response = await call_next(request)

        # Add rate limit headers
        if metadata["limit"] > 0:
            response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
            response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
            response.headers["X-RateLimit-Reset"] = str(metadata["reset"])

        return response


# Global rate limiter instance (lazy initialization)
_rate_limiter: Optional[SlidingWindowRateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(
    config: Optional[RateLimitConfig] = None,
) -> SlidingWindowRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = SlidingWindowRateLimiter(config or RateLimitConfig())
    return _rate_limiter
