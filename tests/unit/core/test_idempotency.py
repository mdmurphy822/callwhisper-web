"""
Tests for idempotency manager.

Tests duplicate prevention and caching:
- Key generation strategies
- Cache hit/miss behavior
- Expiration handling
- Concurrent request blocking
- Statistics tracking
"""

import asyncio
import hashlib
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip tests with platform-specific path behavior
UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Path normalization")

from callwhisper.core.idempotency import (
    IdempotencyConfig,
    IdempotencyManager,
    IdempotencyRecord,
    IdempotencyRecordedError,
    OperationInProgressError,
    get_idempotency_manager,
)


# ============================================================================
# IdempotencyConfig Tests
# ============================================================================


class TestIdempotencyConfig:
    """Tests for IdempotencyConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = IdempotencyConfig()

        assert config.cache_ttl == 86400  # 24 hours
        assert config.max_entries == 1000
        assert config.enabled is True

    def test_custom_values(self):
        """Config accepts custom values."""
        config = IdempotencyConfig(
            cache_ttl=3600,
            max_entries=500,
            enabled=False,
        )

        assert config.cache_ttl == 3600
        assert config.max_entries == 500
        assert config.enabled is False


# ============================================================================
# IdempotencyRecord Tests
# ============================================================================


class TestIdempotencyRecord:
    """Tests for IdempotencyRecord dataclass."""

    def test_default_initialization(self):
        """Record initializes with defaults."""
        record = IdempotencyRecord()

        assert record.result is None
        assert record.created_at is not None
        assert record.completed is False
        assert record.error is None
        assert record.metadata == {}

    def test_custom_initialization(self):
        """Record accepts custom values."""
        created = time.time() - 100
        record = IdempotencyRecord(
            result="test_result",
            created_at=created,
            completed=True,
            error="test_error",
            metadata={"key": "value"},
        )

        assert record.result == "test_result"
        assert record.created_at == created
        assert record.completed is True
        assert record.error == "test_error"
        assert record.metadata == {"key": "value"}

    def test_uses_slots(self):
        """Record uses __slots__ for memory efficiency."""
        assert hasattr(IdempotencyRecord, "__slots__")
        expected_slots = ('result', 'created_at', 'completed', 'error', 'metadata')
        assert IdempotencyRecord.__slots__ == expected_slots


# ============================================================================
# Key Generation Tests
# ============================================================================


class TestKeyGeneration:
    """Tests for idempotency key generation."""

    def test_generate_audio_key(self):
        """Audio key includes file content hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test audio content")
            temp_path = Path(f.name)

        try:
            key = IdempotencyManager.generate_audio_key(temp_path)

            assert key.startswith("audio_")
            assert len(key) > 10  # Has hash suffix
        finally:
            temp_path.unlink()

    def test_generate_audio_key_different_content(self):
        """Different files produce different keys."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content one")
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content two")
            path2 = Path(f2.name)

        try:
            key1 = IdempotencyManager.generate_audio_key(path1)
            key2 = IdempotencyManager.generate_audio_key(path2)

            assert key1 != key2
        finally:
            path1.unlink()
            path2.unlink()

    @UNIX_ONLY
    def test_generate_audio_key_same_content(self):
        """Same content at different times produces different keys (mtime included)."""
        import time
        content = b"identical content"

        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(content)
            path1 = Path(f1.name)

        # Ensure different mtime by sleeping briefly
        time.sleep(0.02)

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(content)
            path2 = Path(f2.name)

        try:
            key1 = IdempotencyManager.generate_audio_key(path1)
            key2 = IdempotencyManager.generate_audio_key(path2)

            # Keys should be different because mtime is included
            # (files created at different times)
            assert key1 != key2  # mtime differs
        finally:
            path1.unlink()
            path2.unlink()

    def test_generate_session_key(self):
        """Session key combines session_id and operation."""
        key = IdempotencyManager.generate_session_key("session123", "transcribe")

        assert key.startswith("session_")
        assert len(key) == len("session_") + 32

    def test_generate_session_key_deterministic(self):
        """Same inputs produce same key."""
        key1 = IdempotencyManager.generate_session_key("session123", "transcribe")
        key2 = IdempotencyManager.generate_session_key("session123", "transcribe")

        assert key1 == key2

    def test_generate_session_key_different_inputs(self):
        """Different inputs produce different keys."""
        key1 = IdempotencyManager.generate_session_key("session1", "op1")
        key2 = IdempotencyManager.generate_session_key("session2", "op1")
        key3 = IdempotencyManager.generate_session_key("session1", "op2")

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_generate_request_key_valid(self):
        """Request key from valid client token."""
        key = IdempotencyManager.generate_request_key("client-token-123")

        assert key == "request_client-token-123"

    def test_generate_request_key_empty_raises(self):
        """Empty token raises ValueError."""
        with pytest.raises(ValueError, match="Invalid idempotency token"):
            IdempotencyManager.generate_request_key("")

    def test_generate_request_key_too_long_raises(self):
        """Token longer than 128 chars raises ValueError."""
        long_token = "x" * 129
        with pytest.raises(ValueError, match="Invalid idempotency token"):
            IdempotencyManager.generate_request_key(long_token)


# ============================================================================
# Check Operation Tests
# ============================================================================


class TestCheckOperation:
    """Tests for checking existing operations."""

    @pytest.fixture
    def manager(self):
        """Create manager with short TTL for testing."""
        config = IdempotencyConfig(cache_ttl=60, max_entries=100)
        return IdempotencyManager(config)

    def test_check_not_found(self, manager):
        """Check returns None for unknown key."""
        result = manager.check("unknown_key")
        assert result is None

    def test_check_completed_success(self, manager):
        """Check returns cached result for completed operation."""
        manager._records["test_key"] = IdempotencyRecord(
            result="cached_result",
            completed=True,
        )

        result = manager.check("test_key")

        assert result == (True, "cached_result")
        assert manager._stats["cache_hits"] == 1
        assert manager._stats["duplicates_prevented"] == 1

    def test_check_completed_with_error(self, manager):
        """Check returns cached error for failed operation."""
        manager._records["test_key"] = IdempotencyRecord(
            result=None,
            completed=True,
            error="Previous error",
        )

        result = manager.check("test_key")

        assert result == (False, "Previous error")

    def test_check_expired_removes_record(self, manager):
        """Expired records are removed on check."""
        old_time = time.time() - 1000  # Well past TTL
        manager._records["expired_key"] = IdempotencyRecord(
            result="old_result",
            created_at=old_time,
            completed=True,
        )

        result = manager.check("expired_key")

        assert result is None
        assert "expired_key" not in manager._records
        assert manager._stats["expirations"] == 1

    def test_check_in_progress_raises(self, manager):
        """In-progress operation raises OperationInProgressError."""
        manager._records["in_progress_key"] = IdempotencyRecord(
            completed=False,
        )

        with pytest.raises(OperationInProgressError):
            manager.check("in_progress_key")

        assert manager._stats["in_progress_blocks"] == 1

    def test_check_disabled(self):
        """Check returns None when disabled."""
        config = IdempotencyConfig(enabled=False)
        manager = IdempotencyManager(config)
        manager._records["test_key"] = IdempotencyRecord(
            result="cached",
            completed=True,
        )

        result = manager.check("test_key")

        assert result is None  # Returns None, doesn't check cache


# ============================================================================
# Start Operation Tests
# ============================================================================


class TestStartOperation:
    """Tests for starting operations."""

    @pytest.fixture
    def manager(self):
        """Create manager with small capacity for testing."""
        config = IdempotencyConfig(max_entries=3)
        return IdempotencyManager(config)

    def test_start_creates_record(self, manager):
        """Start creates in-progress record."""
        manager.start("new_key", metadata={"job_id": "123"})

        assert "new_key" in manager._records
        record = manager._records["new_key"]
        assert record.completed is False
        assert record.metadata == {"job_id": "123"}

    def test_start_evicts_oldest_at_capacity(self, manager):
        """Start evicts oldest completed record at capacity."""
        # Fill to capacity with completed records
        for i in range(3):
            manager._records[f"key_{i}"] = IdempotencyRecord(
                result=f"result_{i}",
                created_at=time.time() - (100 - i * 10),  # Oldest first
                completed=True,
            )

        # Start new operation should evict oldest
        manager.start("new_key")

        assert len(manager._records) == 3
        assert "key_0" not in manager._records  # Oldest evicted
        assert "new_key" in manager._records

    def test_start_skips_incomplete_during_eviction(self, manager):
        """Eviction skips incomplete records."""
        # Add incomplete record (oldest)
        manager._records["incomplete"] = IdempotencyRecord(
            created_at=time.time() - 1000,
            completed=False,
        )
        # Add completed records
        manager._records["complete_1"] = IdempotencyRecord(
            result="r1",
            created_at=time.time() - 500,
            completed=True,
        )
        manager._records["complete_2"] = IdempotencyRecord(
            result="r2",
            created_at=time.time() - 100,
            completed=True,
        )

        manager.start("new_key")

        # Incomplete should remain, oldest completed evicted
        assert "incomplete" in manager._records
        assert "complete_1" not in manager._records  # Oldest completed
        assert "complete_2" in manager._records

    def test_start_disabled(self):
        """Start does nothing when disabled."""
        config = IdempotencyConfig(enabled=False)
        manager = IdempotencyManager(config)

        manager.start("test_key")

        assert len(manager._records) == 0


# ============================================================================
# Complete Operation Tests
# ============================================================================


class TestCompleteOperation:
    """Tests for completing operations."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_complete_updates_existing_record(self, manager):
        """Complete updates an existing started record."""
        manager.start("test_key")

        manager.complete("test_key", "final_result", metadata={"duration": 5})

        record = manager._records["test_key"]
        assert record.result == "final_result"
        assert record.completed is True
        assert record.metadata["duration"] == 5

    def test_complete_creates_record_if_missing(self, manager):
        """Complete creates record if not started with start()."""
        manager.complete("direct_key", "result")

        assert "direct_key" in manager._records
        record = manager._records["direct_key"]
        assert record.result == "result"
        assert record.completed is True

    def test_complete_disabled(self):
        """Complete does nothing when disabled."""
        config = IdempotencyConfig(enabled=False)
        manager = IdempotencyManager(config)

        manager.complete("test_key", "result")

        assert len(manager._records) == 0


# ============================================================================
# Fail Operation Tests
# ============================================================================


class TestFailOperation:
    """Tests for failing operations."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_fail_updates_existing_record(self, manager):
        """Fail updates an existing started record."""
        manager.start("test_key")

        manager.fail("test_key", "Error occurred", metadata={"attempt": 1})

        record = manager._records["test_key"]
        assert record.error == "Error occurred"
        assert record.completed is True
        assert record.metadata["attempt"] == 1

    def test_fail_creates_record_if_missing(self, manager):
        """Fail creates record if not started with start()."""
        manager.fail("direct_key", "Direct error")

        assert "direct_key" in manager._records
        record = manager._records["direct_key"]
        assert record.error == "Direct error"
        assert record.completed is True

    def test_fail_disabled(self):
        """Fail does nothing when disabled."""
        config = IdempotencyConfig(enabled=False)
        manager = IdempotencyManager(config)

        manager.fail("test_key", "error")

        assert len(manager._records) == 0


# ============================================================================
# Cancel Operation Tests
# ============================================================================


class TestCancelOperation:
    """Tests for canceling operations."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_cancel_removes_record(self, manager):
        """Cancel removes in-progress record."""
        manager.start("test_key")

        result = manager.cancel("test_key")

        assert result is True
        assert "test_key" not in manager._records

    def test_cancel_returns_false_if_not_found(self, manager):
        """Cancel returns False if key not found."""
        result = manager.cancel("unknown_key")

        assert result is False


# ============================================================================
# Get Or Execute Tests
# ============================================================================


class TestGetOrExecute:
    """Tests for get_or_execute high-level API."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_cache_miss_executes_function(self, manager):
        """Cache miss executes function and caches result."""
        func = MagicMock(return_value="computed_result")

        result = manager.get_or_execute("key1", func, "arg1", kwarg1="val1")

        assert result == "computed_result"
        func.assert_called_once_with("arg1", kwarg1="val1")
        assert manager._stats["cache_misses"] == 1

    def test_cache_hit_returns_cached(self, manager):
        """Cache hit returns cached result without calling function."""
        # Pre-populate cache
        manager._records["key1"] = IdempotencyRecord(
            result="cached_result",
            completed=True,
        )
        func = MagicMock()

        result = manager.get_or_execute("key1", func)

        assert result == "cached_result"
        func.assert_not_called()
        assert manager._stats["cache_hits"] == 1

    def test_function_error_records_failure(self, manager):
        """Function error is recorded and re-raised."""
        func = MagicMock(side_effect=ValueError("test error"))

        with pytest.raises(ValueError, match="test error"):
            manager.get_or_execute("key1", func)

        record = manager._records["key1"]
        assert record.completed is True
        assert record.error == "test error"

    def test_cached_error_raises_recorded_error(self, manager):
        """Cached error raises IdempotencyRecordedError."""
        manager._records["key1"] = IdempotencyRecord(
            result=None,
            completed=True,
            error="Previous failure",
        )
        func = MagicMock()

        with pytest.raises(IdempotencyRecordedError, match="Previous failure"):
            manager.get_or_execute("key1", func)

        func.assert_not_called()

    def test_in_progress_raises_error(self, manager):
        """In-progress operation raises OperationInProgressError."""
        manager._records["key1"] = IdempotencyRecord(completed=False)
        func = MagicMock()

        with pytest.raises(OperationInProgressError):
            manager.get_or_execute("key1", func)

        func.assert_not_called()


# ============================================================================
# Get Or Execute Async Tests
# ============================================================================


class TestGetOrExecuteAsync:
    """Tests for async get_or_execute."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    @pytest.mark.asyncio
    async def test_cache_miss_executes_async_function(self, manager):
        """Cache miss executes async function and caches result."""
        async def async_func(value):
            await asyncio.sleep(0.001)
            return f"computed_{value}"

        result = await manager.get_or_execute_async("key1", async_func, "test")

        assert result == "computed_test"
        assert manager._stats["cache_misses"] == 1

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_async(self, manager):
        """Cache hit returns cached result without calling async function."""
        manager._records["key1"] = IdempotencyRecord(
            result="cached_result",
            completed=True,
        )

        async def async_func():
            return "should_not_be_called"

        result = await manager.get_or_execute_async("key1", async_func)

        assert result == "cached_result"
        assert manager._stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_async_function_error_records_failure(self, manager):
        """Async function error is recorded and re-raised."""
        async def async_func():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await manager.get_or_execute_async("key1", async_func)

        record = manager._records["key1"]
        assert record.completed is True
        assert record.error == "async error"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_get_stats_empty(self, manager):
        """Stats are correct for empty manager."""
        stats = manager.get_stats()

        assert stats["enabled"] is True
        assert stats["records_count"] == 0
        assert stats["in_progress"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["hit_rate"] == 0.0

    def test_get_stats_with_records(self, manager):
        """Stats reflect record states."""
        # Add various records
        manager._records["in_progress"] = IdempotencyRecord(completed=False)
        manager._records["completed"] = IdempotencyRecord(
            result="r", completed=True
        )
        manager._records["failed"] = IdempotencyRecord(
            completed=True, error="err"
        )
        manager._stats["cache_hits"] = 3
        manager._stats["cache_misses"] = 7

        stats = manager.get_stats()

        assert stats["records_count"] == 3
        assert stats["in_progress"] == 1
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert stats["hit_rate"] == 0.3  # 3/10

    def test_clear_removes_all_records(self, manager):
        """Clear removes all records."""
        manager._records["key1"] = IdempotencyRecord()
        manager._records["key2"] = IdempotencyRecord()
        manager._records["key3"] = IdempotencyRecord()

        count = manager.clear()

        assert count == 3
        assert len(manager._records) == 0


# ============================================================================
# Expiration Tests
# ============================================================================


class TestExpiration:
    """Tests for record expiration."""

    def test_is_expired_false(self):
        """Recent record is not expired."""
        config = IdempotencyConfig(cache_ttl=60)
        manager = IdempotencyManager(config)
        record = IdempotencyRecord(created_at=time.time())

        assert manager._is_expired(record) is False

    def test_is_expired_true(self):
        """Old record is expired."""
        config = IdempotencyConfig(cache_ttl=60)
        manager = IdempotencyManager(config)
        record = IdempotencyRecord(created_at=time.time() - 100)

        assert manager._is_expired(record) is True

    def test_cleanup_expired_removes_old_records(self):
        """Cleanup removes expired records."""
        config = IdempotencyConfig(cache_ttl=60)
        manager = IdempotencyManager(config)

        # Add expired and fresh records
        manager._records["expired1"] = IdempotencyRecord(
            created_at=time.time() - 100
        )
        manager._records["expired2"] = IdempotencyRecord(
            created_at=time.time() - 200
        )
        manager._records["fresh"] = IdempotencyRecord(
            created_at=time.time()
        )

        count = manager._cleanup_expired()

        assert count == 2
        assert "fresh" in manager._records
        assert "expired1" not in manager._records
        assert "expired2" not in manager._records


# ============================================================================
# Eviction Tests
# ============================================================================


class TestEviction:
    """Tests for LRU eviction."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_evict_oldest_removes_oldest_completed(self, manager):
        """Eviction removes oldest completed record."""
        manager._records["oldest"] = IdempotencyRecord(
            result="r", created_at=time.time() - 100, completed=True
        )
        manager._records["newer"] = IdempotencyRecord(
            result="r", created_at=time.time() - 50, completed=True
        )
        manager._records["newest"] = IdempotencyRecord(
            result="r", created_at=time.time(), completed=True
        )

        manager._evict_oldest()

        assert "oldest" not in manager._records
        assert "newer" in manager._records
        assert "newest" in manager._records

    def test_evict_oldest_skips_incomplete(self, manager):
        """Eviction skips incomplete records."""
        manager._records["oldest_incomplete"] = IdempotencyRecord(
            created_at=time.time() - 100, completed=False
        )
        manager._records["oldest_complete"] = IdempotencyRecord(
            result="r", created_at=time.time() - 50, completed=True
        )

        manager._evict_oldest()

        assert "oldest_incomplete" in manager._records
        assert "oldest_complete" not in manager._records

    def test_evict_oldest_empty_records(self, manager):
        """Eviction handles empty records gracefully."""
        # Should not raise
        manager._evict_oldest()

    def test_evict_oldest_only_incomplete(self, manager):
        """Eviction does nothing if all records incomplete."""
        manager._records["key1"] = IdempotencyRecord(completed=False)
        manager._records["key2"] = IdempotencyRecord(completed=False)

        manager._evict_oldest()

        assert len(manager._records) == 2


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_starts(self):
        """Concurrent start operations are thread-safe."""
        manager = IdempotencyManager(IdempotencyConfig(max_entries=1000))
        errors = []

        def start_operation(key):
            try:
                manager.start(key)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=start_operation, args=(f"key_{i}",))
            for i in range(100)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.get_stats()["records_count"] == 100

    def test_concurrent_get_or_execute(self):
        """Concurrent get_or_execute with same key blocks duplicates."""
        manager = IdempotencyManager(IdempotencyConfig())
        call_count = 0
        call_lock = threading.Lock()
        blocking_errors = []

        def slow_func():
            nonlocal call_count
            with call_lock:
                call_count += 1
            time.sleep(0.05)
            return "result"

        def execute(key):
            try:
                manager.get_or_execute(key, slow_func)
            except OperationInProgressError:
                blocking_errors.append(True)

        # Launch multiple threads with same key
        threads = [
            threading.Thread(target=execute, args=("same_key",))
            for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one should succeed, others should be blocked
        assert call_count == 1
        assert len(blocking_errors) == 4  # 4 blocked

    def test_concurrent_check_and_complete(self):
        """Concurrent check and complete operations are thread-safe."""
        manager = IdempotencyManager(IdempotencyConfig())
        manager.start("test_key")

        def complete_operation():
            manager.complete("test_key", "result")

        def check_operation():
            try:
                manager.check("test_key")
            except OperationInProgressError:
                pass  # Expected during race

        threads = []
        for i in range(50):
            if i % 2 == 0:
                threads.append(threading.Thread(target=complete_operation))
            else:
                threads.append(threading.Thread(target=check_operation))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should end with a completed record
        record = manager._records.get("test_key")
        assert record is not None
        assert record.completed is True


# ============================================================================
# Global Singleton Tests
# ============================================================================


class TestGlobalSingleton:
    """Tests for global idempotency manager singleton."""

    def test_get_idempotency_manager_creates_instance(self):
        """get_idempotency_manager creates manager on first call."""
        # Reset global state for test
        import callwhisper.core.idempotency as module
        original = module._manager
        module._manager = None

        try:
            manager = get_idempotency_manager()
            assert isinstance(manager, IdempotencyManager)
        finally:
            module._manager = original

    def test_get_idempotency_manager_returns_same_instance(self):
        """get_idempotency_manager returns same instance."""
        manager1 = get_idempotency_manager()
        manager2 = get_idempotency_manager()

        assert manager1 is manager2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIdempotencyIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return IdempotencyManager(IdempotencyConfig())

    def test_full_success_workflow(self, manager):
        """Complete successful operation workflow."""
        # First request
        result1 = manager.get_or_execute(
            "op_key",
            lambda: "computed_value"
        )

        # Second request (should return cached)
        result2 = manager.get_or_execute(
            "op_key",
            lambda: "should_not_compute"
        )

        assert result1 == "computed_value"
        assert result2 == "computed_value"
        assert manager._stats["cache_hits"] == 1
        assert manager._stats["cache_misses"] == 1

    def test_full_failure_workflow(self, manager):
        """Complete failed operation workflow."""
        # First request fails
        with pytest.raises(ValueError):
            manager.get_or_execute(
                "fail_key",
                lambda: (_ for _ in ()).throw(ValueError("first error"))
            )

        # Second request should return cached error
        with pytest.raises(IdempotencyRecordedError, match="first error"):
            manager.get_or_execute(
                "fail_key",
                lambda: "should_not_run"
            )

    def test_cancel_allows_retry(self, manager):
        """Canceling allows retrying operation."""
        # Start operation
        manager.start("retry_key")

        # Cancel it
        manager.cancel("retry_key")

        # Should be able to retry
        result = manager.get_or_execute(
            "retry_key",
            lambda: "retry_result"
        )

        assert result == "retry_result"

    @pytest.mark.asyncio
    async def test_async_integration(self, manager):
        """Async workflow integration."""
        async def async_operation(value):
            await asyncio.sleep(0.01)
            return f"async_{value}"

        # First async call
        result1 = await manager.get_or_execute_async(
            "async_key", async_operation, "test"
        )

        # Second async call (cached)
        result2 = await manager.get_or_execute_async(
            "async_key", async_operation, "different"
        )

        assert result1 == "async_test"
        assert result2 == "async_test"  # Cached result
