"""
Tests for Transcription Cache Module.

Tests cache operations:
- Cache entry creation and access
- TTL expiration
- LRU eviction
- Statistics tracking
"""

import pytest
import time
from pathlib import Path
from unittest.mock import patch

from callwhisper.core.cache import (
    CacheConfig,
    CacheEntry,
    TranscriptionCache,
    get_cache,
)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """CacheConfig has sensible defaults."""
        config = CacheConfig()

        assert config.max_entries == 100
        assert config.ttl_seconds == 3600
        assert config.enabled is True

    def test_custom_values(self):
        """CacheConfig accepts custom values."""
        config = CacheConfig(max_entries=50, ttl_seconds=300, enabled=False)

        assert config.max_entries == 50
        assert config.ttl_seconds == 300
        assert config.enabled is False


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_creation(self):
        """CacheEntry stores all fields."""
        now = time.time()
        entry = CacheEntry(
            transcript="Hello world",
            created_at=now,
            access_count=5,
            last_accessed=now,
            metadata={"model": "medium"},
        )

        assert entry.transcript == "Hello world"
        assert entry.created_at == now
        assert entry.access_count == 5
        assert entry.metadata == {"model": "medium"}

    def test_defaults(self):
        """CacheEntry has sensible defaults."""
        now = time.time()
        entry = CacheEntry(transcript="Test", created_at=now)

        assert entry.access_count == 0
        assert entry.last_accessed >= now
        assert entry.metadata == {}


class TestTranscriptionCache:
    """Tests for TranscriptionCache class."""

    @pytest.fixture
    def cache(self):
        """Create a cache with short TTL for testing."""
        config = CacheConfig(max_entries=5, ttl_seconds=2)
        return TranscriptionCache(config)

    def test_set_and_get(self, cache):
        """Basic set and get operations work."""
        cache.set("hash1", "Transcript 1")
        cache.set("hash2", "Transcript 2")

        assert cache.get("hash1") == "Transcript 1"
        assert cache.get("hash2") == "Transcript 2"

    def test_get_nonexistent(self, cache):
        """get() returns None for nonexistent keys."""
        result = cache.get("nonexistent")
        assert result is None

    def test_get_updates_access_tracking(self, cache):
        """get() updates access count and last_accessed."""
        cache.set("hash1", "Transcript")

        cache.get("hash1")
        cache.get("hash1")

        info = cache.get_entry_info("hash1")
        assert info["access_count"] == 2

    def test_ttl_expiration(self, cache):
        """Entries expire after TTL."""
        cache.set("expire-test", "Will expire")

        # Should be available immediately
        assert cache.get("expire-test") == "Will expire"

        # Wait for TTL to expire
        time.sleep(2.5)

        # Should be expired now
        assert cache.get("expire-test") is None

    def test_lru_eviction(self, cache):
        """Oldest entry is evicted when at capacity."""
        # Fill cache to capacity (max_entries=5)
        for i in range(5):
            cache.set(f"hash{i}", f"Transcript {i}")
            time.sleep(0.01)  # Ensure different timestamps

        # Access some entries to update their last_accessed
        cache.get("hash2")
        cache.get("hash4")

        # Add one more entry - should evict the oldest not-accessed entry
        cache.set("hash5", "Transcript 5")

        stats = cache.get_stats()
        assert stats["size"] == 5
        assert stats["evictions"] >= 1

    def test_invalidate(self, cache):
        """invalidate() removes specific entry."""
        cache.set("to-remove", "Will be removed")
        assert cache.get("to-remove") == "Will be removed"

        result = cache.invalidate("to-remove")

        assert result is True
        assert cache.get("to-remove") is None

    def test_invalidate_nonexistent(self, cache):
        """invalidate() returns False for nonexistent key."""
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self, cache):
        """clear() removes all entries."""
        cache.set("hash1", "Transcript 1")
        cache.set("hash2", "Transcript 2")
        cache.set("hash3", "Transcript 3")

        count = cache.clear()

        assert count == 3
        assert cache.get("hash1") is None
        assert cache.get_stats()["size"] == 0

    def test_disabled_cache(self):
        """Disabled cache does not store entries."""
        config = CacheConfig(enabled=False)
        cache = TranscriptionCache(config)

        cache.set("hash1", "Transcript")

        assert cache.get("hash1") is None

    def test_stats_tracking(self, cache):
        """Statistics are tracked correctly."""
        cache.set("hash1", "Transcript 1")
        cache.set("hash2", "Transcript 2")

        # Generate hits and misses
        cache.get("hash1")  # hit
        cache.get("hash1")  # hit
        cache.get("hash2")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()

        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["size"] == 2
        assert stats["hit_rate"] == 0.75

    def test_stats_with_no_requests(self):
        """Stats handle zero requests gracefully."""
        config = CacheConfig()
        cache = TranscriptionCache(config)

        stats = cache.get_stats()

        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_get_entry_info(self, cache):
        """get_entry_info() returns entry details."""
        cache.set("hash1", "Transcript 1", metadata={"duration": 30.5})

        info = cache.get_entry_info("hash1")

        assert info is not None
        assert info["audio_hash"] == "hash1"
        assert info["transcript_length"] == len("Transcript 1")
        assert info["access_count"] == 0
        assert info["metadata"] == {"duration": 30.5}
        assert info["ttl_remaining"] > 0

    def test_get_entry_info_nonexistent(self, cache):
        """get_entry_info() returns None for nonexistent entry."""
        info = cache.get_entry_info("nonexistent")
        assert info is None

    def test_metadata_storage(self, cache):
        """Metadata is stored with entry."""
        metadata = {"model": "large", "duration": 45.0, "language": "en"}
        cache.set("hash1", "Transcript", metadata=metadata)

        info = cache.get_entry_info("hash1")
        assert info["metadata"] == metadata


class TestComputeAudioHash:
    """Tests for compute_audio_hash static method."""

    def test_hash_small_file(self, tmp_path):
        """Hashes small file correctly."""
        test_file = tmp_path / "small.wav"
        test_file.write_bytes(b"x" * 500)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)

        assert len(hash1) == 16
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_hash_large_file(self, tmp_path):
        """Hashes large file using first/last bytes."""
        test_file = tmp_path / "large.wav"
        test_file.write_bytes(b"x" * 10000)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)

        assert len(hash1) == 16

    def test_different_files_different_hashes(self, tmp_path):
        """Different files produce different hashes."""
        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"
        file1.write_bytes(b"content A" * 100)
        file2.write_bytes(b"content B" * 100)

        hash1 = TranscriptionCache.compute_audio_hash(file1)
        hash2 = TranscriptionCache.compute_audio_hash(file2)

        assert hash1 != hash2

    def test_same_file_same_hash(self, tmp_path):
        """Same file produces consistent hash."""
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"consistent content")

        hash1 = TranscriptionCache.compute_audio_hash(test_file)
        hash2 = TranscriptionCache.compute_audio_hash(test_file)

        assert hash1 == hash2


class TestGetCache:
    """Tests for get_cache function."""

    def test_returns_singleton(self):
        """get_cache returns same instance."""
        import callwhisper.core.cache as cache_module
        cache_module._cache = None

        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

        # Cleanup
        cache_module._cache = None

    def test_uses_provided_config(self):
        """get_cache uses provided config for initialization."""
        import callwhisper.core.cache as cache_module
        cache_module._cache = None

        config = CacheConfig(max_entries=50, ttl_seconds=300)
        cache = get_cache(config)

        assert cache.config.max_entries == 50
        assert cache.config.ttl_seconds == 300

        # Cleanup
        cache_module._cache = None


# ============================================================================
# Batch 7: Advanced Edge Case and Concurrency Tests
# ============================================================================

import threading
from concurrent.futures import ThreadPoolExecutor


class TestCacheHashBoundaries:
    """Tests for file hash boundary conditions."""

    def test_file_exactly_2048_bytes(self, tmp_path):
        """Hash boundary at exactly 2048 bytes."""
        test_file = tmp_path / "boundary.wav"
        test_file.write_bytes(b"x" * 2048)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)

        assert len(hash1) == 16
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_file_2047_bytes(self, tmp_path):
        """Hash just below boundary."""
        test_file = tmp_path / "small.wav"
        test_file.write_bytes(b"x" * 2047)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)
        assert len(hash1) == 16

    def test_file_2049_bytes(self, tmp_path):
        """Hash just above boundary."""
        test_file = tmp_path / "large.wav"
        test_file.write_bytes(b"x" * 2049)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)
        assert len(hash1) == 16

    def test_file_1024_bytes(self, tmp_path):
        """Hash at first chunk boundary."""
        test_file = tmp_path / "chunk.wav"
        test_file.write_bytes(b"x" * 1024)

        hash1 = TranscriptionCache.compute_audio_hash(test_file)
        assert len(hash1) == 16

    def test_empty_file(self, tmp_path):
        """Hash of empty file."""
        test_file = tmp_path / "empty.wav"
        test_file.write_bytes(b"")

        hash1 = TranscriptionCache.compute_audio_hash(test_file)

        # Should still produce a valid hash (based on file size = 0)
        assert len(hash1) == 16

    def test_one_byte_file(self, tmp_path):
        """Hash of single byte file."""
        test_file = tmp_path / "one.wav"
        test_file.write_bytes(b"x")

        hash1 = TranscriptionCache.compute_audio_hash(test_file)
        assert len(hash1) == 16

    def test_hash_includes_size(self, tmp_path):
        """Files with same content but different sizes have different hashes."""
        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"

        # Both files have 'x' as content in first 1KB, but different sizes
        file1.write_bytes(b"x" * 1000)
        file2.write_bytes(b"x" * 2000)

        hash1 = TranscriptionCache.compute_audio_hash(file1)
        hash2 = TranscriptionCache.compute_audio_hash(file2)

        assert hash1 != hash2


class TestCacheEvictionEdgeCases:
    """Tests for LRU eviction edge cases."""

    def test_eviction_equal_access_times(self):
        """Eviction when entries have equal access times."""
        config = CacheConfig(max_entries=3, ttl_seconds=3600)
        cache = TranscriptionCache(config)

        # Add entries as fast as possible (potentially same timestamp)
        cache.set("hash1", "Transcript 1")
        cache.set("hash2", "Transcript 2")
        cache.set("hash3", "Transcript 3")

        # Add one more to trigger eviction
        cache.set("hash4", "Transcript 4")

        # One entry should be evicted
        stats = cache.get_stats()
        assert stats["size"] == 3
        assert stats["evictions"] == 1

    def test_multiple_evictions(self):
        """Multiple evictions in sequence."""
        config = CacheConfig(max_entries=2, ttl_seconds=3600)
        cache = TranscriptionCache(config)

        # Add entries to trigger multiple evictions
        for i in range(10):
            cache.set(f"hash{i}", f"Transcript {i}")
            time.sleep(0.01)  # Ensure different timestamps

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["evictions"] == 8

    def test_eviction_preserves_most_recent(self):
        """LRU eviction preserves most recently accessed."""
        config = CacheConfig(max_entries=3, ttl_seconds=3600)
        cache = TranscriptionCache(config)

        cache.set("old", "Old transcript")
        time.sleep(0.01)
        cache.set("middle", "Middle transcript")
        time.sleep(0.01)
        cache.set("new", "New transcript")

        # Access "old" to make it recently used
        cache.get("old")

        # Add new entry - should evict "middle" (oldest not accessed)
        cache.set("newest", "Newest transcript")

        assert cache.get("old") is not None
        assert cache.get("new") is not None
        assert cache.get("newest") is not None


class TestCacheTTLEdgeCases:
    """Tests for TTL edge cases."""

    def test_ttl_zero(self):
        """TTL of 0 seconds means immediate expiration."""
        config = CacheConfig(max_entries=10, ttl_seconds=0)
        cache = TranscriptionCache(config)

        cache.set("instant", "Expires immediately")

        # Even immediate get should fail with TTL=0
        result = cache.get("instant")
        assert result is None

        stats = cache.get_stats()
        assert stats["expirations"] == 1

    def test_ttl_very_short(self):
        """Very short TTL (< 1 second)."""
        config = CacheConfig(max_entries=10, ttl_seconds=0.1)
        cache = TranscriptionCache(config)

        cache.set("short", "Short lived")

        # Should work immediately
        assert cache.get("short") == "Short lived"

        # Wait for expiration
        time.sleep(0.2)

        assert cache.get("short") is None

    def test_ttl_very_long(self):
        """Very long TTL doesn't cause issues."""
        config = CacheConfig(max_entries=10, ttl_seconds=86400 * 365)  # 1 year
        cache = TranscriptionCache(config)

        cache.set("long", "Long lived")
        info = cache.get_entry_info("long")

        assert info["ttl_remaining"] > 0

    def test_cleanup_expired_removes_multiple(self):
        """cleanup_expired removes all expired entries."""
        config = CacheConfig(max_entries=10, ttl_seconds=0.1)
        cache = TranscriptionCache(config)

        for i in range(5):
            cache.set(f"expire{i}", f"Transcript {i}")

        time.sleep(0.2)

        removed = cache._cleanup_expired()
        assert removed == 5
        assert cache.get_stats()["size"] == 0


class TestCacheConcurrency:
    """Tests for thread safety of cache operations."""

    def test_concurrent_set_get(self):
        """Concurrent set and get operations are thread-safe."""
        config = CacheConfig(max_entries=100, ttl_seconds=3600)
        cache = TranscriptionCache(config)
        errors = []

        def setter():
            try:
                for i in range(100):
                    cache.set(f"hash{i}", f"Transcript {i}")
            except Exception as e:
                errors.append(e)

        def getter():
            try:
                for i in range(100):
                    cache.get(f"hash{i}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(setter),
                executor.submit(setter),
                executor.submit(getter),
                executor.submit(getter),
                executor.submit(getter),
                executor.submit(getter),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0

    def test_concurrent_eviction(self):
        """Concurrent operations causing eviction."""
        config = CacheConfig(max_entries=10, ttl_seconds=3600)
        cache = TranscriptionCache(config)
        errors = []

        def heavy_setter(thread_id):
            try:
                for i in range(50):
                    cache.set(f"t{thread_id}_h{i}", f"Transcript {i}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(heavy_setter, i) for i in range(5)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        stats = cache.get_stats()
        assert stats["size"] <= 10

    def test_concurrent_clear(self):
        """Concurrent operations during clear."""
        config = CacheConfig(max_entries=100, ttl_seconds=3600)
        cache = TranscriptionCache(config)
        errors = []

        def setter():
            try:
                for i in range(100):
                    cache.set(f"hash{i}", f"Transcript {i}")
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                time.sleep(0.01)
                cache.clear()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(setter),
                executor.submit(setter),
                executor.submit(clearer),
                executor.submit(setter),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0


class TestCacheTranscriptEdgeCases:
    """Tests for edge cases in transcript content."""

    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        config = CacheConfig(max_entries=100, ttl_seconds=3600)
        return TranscriptionCache(config)

    def test_empty_transcript(self, cache):
        """Cache handles empty transcript string."""
        cache.set("empty", "")

        result = cache.get("empty")
        assert result == ""

    def test_very_large_transcript(self, cache):
        """Cache handles very large transcripts."""
        large_transcript = "x" * 1_000_000  # 1MB
        cache.set("large", large_transcript)

        result = cache.get("large")
        assert result == large_transcript
        assert len(result) == 1_000_000

    def test_unicode_transcript(self, cache):
        """Cache handles unicode transcripts."""
        unicode_text = "日本語テスト 🎤 émojis and special chars"
        cache.set("unicode", unicode_text)

        result = cache.get("unicode")
        assert result == unicode_text

    def test_newlines_in_transcript(self, cache):
        """Cache handles transcripts with newlines."""
        multiline = "Line 1\nLine 2\nLine 3\n\n\nLine 6"
        cache.set("multiline", multiline)

        result = cache.get("multiline")
        assert result == multiline

    def test_special_characters_in_hash(self, cache):
        """Cache handles unusual hash strings."""
        # These shouldn't typically occur but test robustness
        cache.set("special-hash-with-dashes", "Transcript")
        cache.set("hash_with_underscores", "Transcript")
        cache.set("", "Empty key transcript")

        assert cache.get("special-hash-with-dashes") == "Transcript"
        assert cache.get("hash_with_underscores") == "Transcript"
        assert cache.get("") == "Empty key transcript"


class TestCacheMetadataEdgeCases:
    """Tests for metadata edge cases."""

    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        config = CacheConfig(max_entries=100, ttl_seconds=3600)
        return TranscriptionCache(config)

    def test_nested_metadata(self, cache):
        """Cache handles deeply nested metadata."""
        nested = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
        cache.set("nested", "Transcript", metadata=nested)

        info = cache.get_entry_info("nested")
        assert info["metadata"]["level1"]["level2"]["level3"]["value"] == "deep"

    def test_large_metadata(self, cache):
        """Cache handles large metadata."""
        large_meta = {f"key{i}": f"value{i}" for i in range(1000)}
        cache.set("large-meta", "Transcript", metadata=large_meta)

        info = cache.get_entry_info("large-meta")
        assert len(info["metadata"]) == 1000

    def test_none_metadata(self, cache):
        """Cache handles None metadata."""
        cache.set("none-meta", "Transcript", metadata=None)

        info = cache.get_entry_info("none-meta")
        assert info["metadata"] == {}


class TestCacheStatsEdgeCases:
    """Tests for statistics edge cases."""

    def test_stats_after_heavy_use(self):
        """Stats are accurate after heavy use."""
        config = CacheConfig(max_entries=10, ttl_seconds=3600)
        cache = TranscriptionCache(config)

        # Generate many operations
        for i in range(100):
            cache.set(f"hash{i % 20}", f"Transcript {i}")

        for i in range(50):
            cache.get(f"hash{i % 25}")

        stats = cache.get_stats()

        # Verify stats are reasonable
        assert stats["hits"] + stats["misses"] == 50
        assert stats["evictions"] >= 0
        assert stats["size"] <= 10

    def test_hit_rate_precision(self):
        """Hit rate has correct precision."""
        config = CacheConfig(max_entries=100, ttl_seconds=3600)
        cache = TranscriptionCache(config)

        cache.set("hash1", "Transcript")

        # 7 hits, 3 misses = 70% hit rate
        for _ in range(7):
            cache.get("hash1")
        for _ in range(3):
            cache.get("missing")

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.7
