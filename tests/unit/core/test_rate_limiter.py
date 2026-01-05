"""
Tests for rate limiter module.

Tests sliding window rate limiting:
- Request allowance/blocking
- Per-client tracking
- Excluded paths
- Cleanup of old entries
- Middleware integration
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from callwhisper.core.rate_limiter import (
    ClientMetrics,
    RateLimitConfig,
    RateLimitMiddleware,
    SlidingWindowRateLimiter,
    get_rate_limiter,
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_mock_request(
    client_ip: str = "192.168.1.1",
    path: str = "/api/test",
    forwarded_for: str = None
) -> MagicMock:
    """Create a mock Starlette request."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = client_ip
    request.url = MagicMock()
    request.url.path = path
    request.headers = {}
    if forwarded_for:
        request.headers["X-Forwarded-For"] = forwarded_for
    return request


# ============================================================================
# RateLimitConfig Tests
# ============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.burst_size == 10
        assert config.enabled is True
        assert "/api/health" in config.excluded_paths

    def test_custom_values(self):
        """Config accepts custom values."""
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_size=20,
            enabled=False,
            excluded_paths=["/custom/path"]
        )

        assert config.requests_per_minute == 100
        assert config.burst_size == 20
        assert config.enabled is False
        assert config.excluded_paths == ["/custom/path"]


# ============================================================================
# ClientMetrics Tests
# ============================================================================


class TestClientMetrics:
    """Tests for ClientMetrics dataclass."""

    def test_default_values(self):
        """Metrics have expected defaults."""
        metrics = ClientMetrics()

        assert metrics.request_times == []
        assert metrics.blocked_count == 0
        assert metrics.last_blocked is None


# ============================================================================
# SlidingWindowRateLimiter Basic Tests
# ============================================================================


class TestSlidingWindowRateLimiterBasic:
    """Tests for basic rate limiter operations."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter with low limits for testing."""
        config = RateLimitConfig(
            requests_per_minute=5,
            burst_size=2,
            enabled=True,
            excluded_paths=["/api/health"]
        )
        return SlidingWindowRateLimiter(config)

    def test_first_request_allowed(self, limiter):
        """First request is always allowed."""
        request = create_mock_request()

        allowed, metadata = limiter.is_allowed(request)

        assert allowed is True
        assert metadata["remaining"] >= 0

    def test_metadata_includes_limit(self, limiter):
        """Metadata includes rate limit configuration."""
        request = create_mock_request()

        allowed, metadata = limiter.is_allowed(request)

        assert metadata["limit"] == 5
        assert "remaining" in metadata
        assert "reset" in metadata

    def test_requests_within_limit_allowed(self, limiter):
        """Multiple requests within limit are allowed."""
        request = create_mock_request()

        for i in range(5):
            allowed, _ = limiter.is_allowed(request)
            assert allowed is True

    def test_requests_over_limit_blocked(self, limiter):
        """Requests over limit are blocked."""
        request = create_mock_request()

        # Use up the limit
        for _ in range(5):
            limiter.is_allowed(request)

        # Next request should be blocked
        allowed, metadata = limiter.is_allowed(request)

        assert allowed is False
        assert metadata["remaining"] == 0


# ============================================================================
# Excluded Paths Tests
# ============================================================================


class TestExcludedPaths:
    """Tests for excluded paths handling."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter with excluded paths."""
        config = RateLimitConfig(
            requests_per_minute=2,
            excluded_paths=["/api/health", "/api/health/ready"]
        )
        return SlidingWindowRateLimiter(config)

    def test_excluded_path_always_allowed(self, limiter):
        """Excluded paths bypass rate limiting."""
        request = create_mock_request(path="/api/health")

        # Make many requests - all should be allowed
        for _ in range(100):
            allowed, metadata = limiter.is_allowed(request)
            assert allowed is True
            assert metadata["limit"] == -1  # Indicates excluded

    def test_non_excluded_path_rate_limited(self, limiter):
        """Non-excluded paths are rate limited."""
        request = create_mock_request(path="/api/other")

        # First 2 allowed
        limiter.is_allowed(request)
        limiter.is_allowed(request)

        # Third blocked
        allowed, _ = limiter.is_allowed(request)
        assert allowed is False


# ============================================================================
# Client Identification Tests
# ============================================================================


class TestClientIdentification:
    """Tests for client identification."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        config = RateLimitConfig(requests_per_minute=3)
        return SlidingWindowRateLimiter(config)

    def test_uses_client_ip(self, limiter):
        """Uses client IP when no forwarding header."""
        request = create_mock_request(client_ip="10.0.0.1")

        limiter.is_allowed(request)

        assert "10.0.0.1" in limiter._clients

    def test_uses_forwarded_for_header(self, limiter):
        """Uses X-Forwarded-For when present."""
        request = create_mock_request(
            client_ip="10.0.0.1",
            forwarded_for="203.0.113.50, 70.41.3.18"
        )

        limiter.is_allowed(request)

        # Should use first IP in chain
        assert "203.0.113.50" in limiter._clients

    def test_separate_limits_per_client(self, limiter):
        """Each client has separate limits."""
        client1 = create_mock_request(client_ip="192.168.1.1")
        client2 = create_mock_request(client_ip="192.168.1.2")

        # Use up client1's limit
        for _ in range(3):
            limiter.is_allowed(client1)

        # Client1 blocked
        allowed, _ = limiter.is_allowed(client1)
        assert allowed is False

        # Client2 still allowed
        allowed, _ = limiter.is_allowed(client2)
        assert allowed is True


# ============================================================================
# Disabled Rate Limiter Tests
# ============================================================================


class TestDisabledRateLimiter:
    """Tests for disabled rate limiter."""

    def test_all_requests_allowed_when_disabled(self):
        """All requests are allowed when rate limiter is disabled."""
        config = RateLimitConfig(
            requests_per_minute=1,
            enabled=False
        )
        limiter = SlidingWindowRateLimiter(config)
        request = create_mock_request()

        # Should allow all requests
        for _ in range(100):
            allowed, metadata = limiter.is_allowed(request)
            assert allowed is True
            assert metadata["limit"] == -1


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics retrieval."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        config = RateLimitConfig(requests_per_minute=5)
        return SlidingWindowRateLimiter(config)

    def test_get_client_stats(self, limiter):
        """Get stats for specific client."""
        request = create_mock_request(client_ip="192.168.1.1")

        # Make some requests
        for _ in range(3):
            limiter.is_allowed(request)

        stats = limiter.get_client_stats("192.168.1.1")

        assert stats is not None
        assert stats["requests_in_window"] == 3
        assert stats["limit"] == 5
        assert stats["blocked_count"] == 0

    def test_get_client_stats_unknown_client(self, limiter):
        """Returns None for unknown client."""
        stats = limiter.get_client_stats("unknown-client")
        assert stats is None

    def test_get_client_stats_with_blocks(self, limiter):
        """Stats include blocked count."""
        request = create_mock_request(client_ip="192.168.1.1")

        # Exhaust limit
        for _ in range(5):
            limiter.is_allowed(request)

        # Get blocked
        limiter.is_allowed(request)
        limiter.is_allowed(request)

        stats = limiter.get_client_stats("192.168.1.1")

        assert stats["blocked_count"] == 2

    def test_get_all_stats(self, limiter):
        """Get stats for all clients."""
        request1 = create_mock_request(client_ip="192.168.1.1")
        request2 = create_mock_request(client_ip="192.168.1.2")

        limiter.is_allowed(request1)
        limiter.is_allowed(request2)
        limiter.is_allowed(request2)

        stats = limiter.get_all_stats()

        assert stats["total_clients"] == 2
        assert stats["config"]["requests_per_minute"] == 5
        assert "192.168.1.1" in stats["clients"]
        assert "192.168.1.2" in stats["clients"]
        assert stats["clients"]["192.168.1.2"]["requests_in_window"] == 2


# ============================================================================
# Cleanup Tests
# ============================================================================


class TestCleanup:
    """Tests for old entry cleanup."""

    def test_old_requests_expire(self):
        """Old requests are removed from the window."""
        config = RateLimitConfig(requests_per_minute=3)
        limiter = SlidingWindowRateLimiter(config)
        request = create_mock_request()

        # Manually add old request times
        with limiter._lock:
            old_time = time.time() - 120  # 2 minutes ago
            limiter._clients["192.168.1.1"].request_times = [
                old_time, old_time + 1, old_time + 2
            ]

        # New request should be allowed (old ones expired)
        allowed, _ = limiter.is_allowed(request)
        assert allowed is True

    def test_cleanup_removes_inactive_clients(self):
        """Inactive clients are removed during cleanup."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config)

        # Add a client with old data
        with limiter._lock:
            limiter._clients["old-client"].request_times = []
            limiter._clients["old-client"].last_blocked = time.time() - 1000

        # Trigger cleanup
        limiter._last_cleanup = time.time() - 400  # Force cleanup
        request = create_mock_request(client_ip="new-client")
        limiter.is_allowed(request)

        # Old client should be removed
        with limiter._lock:
            assert "old-client" not in limiter._clients


# ============================================================================
# Reset Time Tests
# ============================================================================


class TestResetTime:
    """Tests for reset time calculation."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        config = RateLimitConfig(requests_per_minute=5)
        return SlidingWindowRateLimiter(config)

    def test_reset_time_60_for_no_requests(self, limiter):
        """Reset time is 60 when no requests in window."""
        request = create_mock_request(client_ip="new-client")

        allowed, metadata = limiter.is_allowed(request)

        # Reset time should be close to 60 (first request just added)
        assert 55 <= metadata["reset"] <= 60

    def test_reset_time_decreases_with_window(self, limiter):
        """Reset time decreases as window progresses."""
        request = create_mock_request()

        # First request
        _, metadata1 = limiter.is_allowed(request)

        # Wait a bit
        time.sleep(0.1)

        # Second request
        _, metadata2 = limiter.is_allowed(request)

        # Reset time should be similar or slightly less
        assert metadata2["reset"] <= metadata1["reset"]


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_requests(self):
        """Concurrent requests are handled safely."""
        config = RateLimitConfig(requests_per_minute=100)
        limiter = SlidingWindowRateLimiter(config)
        request = create_mock_request()

        errors = []
        results = []

        def make_request():
            try:
                for _ in range(50):
                    allowed, _ = limiter.is_allowed(request)
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 200

    def test_concurrent_stats_access(self):
        """Concurrent stats access is thread-safe."""
        config = RateLimitConfig(requests_per_minute=100)
        limiter = SlidingWindowRateLimiter(config)

        errors = []

        def read_stats():
            try:
                for _ in range(100):
                    stats = limiter.get_all_stats()
                    assert "total_clients" in stats
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_stats) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# Middleware Tests
# ============================================================================


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    @pytest.fixture
    def limiter(self):
        """Create rate limiter for testing."""
        config = RateLimitConfig(requests_per_minute=3)
        return SlidingWindowRateLimiter(config)

    @pytest.mark.asyncio
    async def test_adds_rate_limit_headers(self, limiter):
        """Middleware adds rate limit headers."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, limiter)

        request = create_mock_request()
        response = MagicMock(spec=Response)
        response.headers = {}

        async def call_next(req):
            return response

        result = await middleware.dispatch(request, call_next)

        assert "X-RateLimit-Limit" in result.headers
        assert "X-RateLimit-Remaining" in result.headers
        assert "X-RateLimit-Reset" in result.headers

    @pytest.mark.asyncio
    async def test_returns_429_when_limited(self, limiter):
        """Middleware returns 429 when rate limited."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, limiter)

        request = create_mock_request()

        # Exhaust limit
        for _ in range(3):
            limiter.is_allowed(request)

        async def call_next(req):
            return MagicMock()

        result = await middleware.dispatch(request, call_next)

        assert result.status_code == 429

    @pytest.mark.asyncio
    async def test_passes_through_when_allowed(self, limiter):
        """Middleware passes through when allowed."""
        app = MagicMock()
        middleware = RateLimitMiddleware(app, limiter)

        request = create_mock_request()
        expected_response = MagicMock(spec=Response)
        expected_response.headers = {}

        async def call_next(req):
            return expected_response

        result = await middleware.dispatch(request, call_next)

        assert result == expected_response


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global rate limiter singleton."""

    def test_get_rate_limiter_creates_instance(self):
        """get_rate_limiter creates instance on first call."""
        import callwhisper.core.rate_limiter as module
        original = module._rate_limiter
        module._rate_limiter = None

        try:
            limiter = get_rate_limiter()
            assert isinstance(limiter, SlidingWindowRateLimiter)
        finally:
            module._rate_limiter = original

    def test_get_rate_limiter_returns_same_instance(self):
        """get_rate_limiter returns same instance."""
        import callwhisper.core.rate_limiter as module
        original = module._rate_limiter
        module._rate_limiter = None

        try:
            limiter1 = get_rate_limiter()
            limiter2 = get_rate_limiter()
            assert limiter1 is limiter2
        finally:
            module._rate_limiter = original

    def test_get_rate_limiter_with_custom_config(self):
        """get_rate_limiter accepts custom config on first call."""
        import callwhisper.core.rate_limiter as module
        original = module._rate_limiter
        module._rate_limiter = None

        try:
            config = RateLimitConfig(requests_per_minute=999)
            limiter = get_rate_limiter(config)
            assert limiter.config.requests_per_minute == 999
        finally:
            module._rate_limiter = original
