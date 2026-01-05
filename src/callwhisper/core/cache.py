"""
Transcription Cache Module

Based on LibV2 Python course patterns:
- LRU-style cache with TTL
- Memory-efficient dictionary views
- Thread-safe operations
"""

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from .logging_config import get_core_logger

logger = get_core_logger()


@dataclass
class CacheConfig:
    """Configuration for transcription cache."""

    max_entries: int = 100
    ttl_seconds: int = 3600  # 1 hour
    enabled: bool = True


class CacheEntry:
    """
    A single cache entry with metadata.

    Uses __slots__ for memory efficiency (~40% savings per entry).
    Based on LibV2 introduction-to-python patterns.
    """

    __slots__ = (
        "transcript",
        "created_at",
        "access_count",
        "last_accessed",
        "metadata",
    )

    def __init__(
        self,
        transcript: str,
        created_at: float,
        access_count: int = 0,
        last_accessed: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.transcript = transcript
        self.created_at = created_at
        self.access_count = access_count
        self.last_accessed = last_accessed if last_accessed is not None else time.time()
        self.metadata = metadata if metadata is not None else {}


class TranscriptionCache:
    """
    LRU-style cache for transcription results.

    Features:
    - TTL-based expiration
    - LRU eviction when at capacity
    - Thread-safe with Lock()
    - Memory-efficient using dictionary views
    - Cache statistics tracking
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._lock = threading.Lock()
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    @staticmethod
    def compute_audio_hash(audio_path: Path) -> str:
        """
        Compute a hash for an audio file.

        Uses file size + first/last 1KB for fast hashing of large files.
        """
        file_size = audio_path.stat().st_size
        hasher = hashlib.sha256()

        # Add file size to hash
        hasher.update(str(file_size).encode())

        with open(audio_path, "rb") as f:
            # Read first 1KB
            hasher.update(f.read(1024))

            # Read last 1KB if file is large enough
            if file_size > 2048:
                f.seek(-1024, 2)
                hasher.update(f.read(1024))

        return hasher.hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - entry.created_at > self.config.ttl_seconds

    def _evict_oldest(self) -> None:
        """Evict the least recently accessed entry."""
        if not self._cache:
            return

        # Use dict.items() view for memory-efficient iteration
        oldest_key = min(self._cache.items(), key=lambda x: x[1].last_accessed)[0]

        del self._cache[oldest_key]
        self._stats["evictions"] += 1

        logger.debug("cache_eviction", evicted_key=oldest_key[:8])

    def _cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if now - entry.created_at > self.config.ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]
            self._stats["expirations"] += 1

        return len(expired_keys)

    def get(self, audio_hash: str) -> Optional[str]:
        """
        Get cached transcript for audio hash.

        Returns:
            Transcript string if found and not expired, None otherwise.
        """
        if not self.config.enabled:
            return None

        with self._lock:
            if audio_hash not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[audio_hash]

            # Check expiration
            if self._is_expired(entry):
                del self._cache[audio_hash]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                logger.debug("cache_expired", audio_hash=audio_hash[:8])
                return None

            # Update access tracking
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1

            logger.debug(
                "cache_hit", audio_hash=audio_hash[:8], access_count=entry.access_count
            )

            return entry.transcript

    def set(
        self,
        audio_hash: str,
        transcript: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache a transcript for an audio hash.

        Args:
            audio_hash: Hash of the audio file
            transcript: Transcription result
            metadata: Optional metadata (duration, model, etc.)
        """
        if not self.config.enabled:
            return

        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.config.max_entries:
                self._evict_oldest()

            now = time.time()
            self._cache[audio_hash] = CacheEntry(
                transcript=transcript,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
            )

            logger.debug(
                "cache_set",
                audio_hash=audio_hash[:8],
                transcript_length=len(transcript),
                cache_size=len(self._cache),
            )

    def invalidate(self, audio_hash: str) -> bool:
        """
        Remove a specific entry from cache.

        Returns:
            True if entry was found and removed, False otherwise.
        """
        with self._lock:
            if audio_hash in self._cache:
                del self._cache[audio_hash]
                logger.debug("cache_invalidate", audio_hash=audio_hash[:8])
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info("cache_cleared", entries_removed=count)
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            )

            return {
                "enabled": self.config.enabled,
                "size": len(self._cache),
                "max_size": self.config.max_entries,
                "ttl_seconds": self.config.ttl_seconds,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 3),
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"],
            }

    def get_entry_info(self, audio_hash: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific cache entry."""
        with self._lock:
            if audio_hash not in self._cache:
                return None

            entry = self._cache[audio_hash]
            now = time.time()

            return {
                "audio_hash": audio_hash,
                "transcript_length": len(entry.transcript),
                "created_at": entry.created_at,
                "age_seconds": now - entry.created_at,
                "ttl_remaining": max(
                    0, self.config.ttl_seconds - (now - entry.created_at)
                ),
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
                "metadata": entry.metadata,
            }


# Global cache instance (lazy initialization)
_cache: Optional[TranscriptionCache] = None
_cache_lock = threading.Lock()


def get_cache(config: Optional[CacheConfig] = None) -> TranscriptionCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = TranscriptionCache(config or CacheConfig())
    return _cache
