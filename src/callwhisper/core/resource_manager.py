"""
Resource Manager Module

Based on LibV2 introduction-to-python patterns:
- Context managers for resource cleanup (chunk 02236)
- File descriptor tracking to prevent OS limits
- Automatic resource eviction

Key concepts:
- Track open files to prevent hitting OS limits (~1024 per process)
- LRU eviction of least recently used resources
- Context managers ensure cleanup even on exceptions
- Monitoring for resource leaks
"""

import os
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, Optional, TextIO, Union

from .logging_config import get_core_logger

logger = get_core_logger()


@dataclass
class ResourceConfig:
    """Configuration for resource manager."""

    max_open_files: int = 100
    max_temp_files: int = 50
    cleanup_interval: float = 60.0  # seconds
    warn_threshold: float = 0.8  # warn at 80% capacity


@dataclass
class ResourceEntry:
    """
    Tracked resource entry.

    Uses __slots__ for memory efficiency.
    """

    __slots__ = ("path", "handle", "opened_at", "last_accessed", "access_count", "mode")

    path: str
    handle: Any
    opened_at: float
    last_accessed: float
    access_count: int
    mode: str

    def __init__(self, path: str, handle: Any, mode: str = "rb"):
        self.path = path
        self.handle = handle
        self.mode = mode
        self.opened_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0


class ResourceManager:
    """
    Track and limit open file descriptors.

    Prevents hitting OS limits (~1024 per process on most systems).
    Uses LRU eviction when approaching limits.

    Example:
        manager = ResourceManager()

        with manager.open_file(audio_path) as f:
            data = f.read()

        # Or for audio files specifically
        with manager.open_audio(audio_path) as f:
            chunk = f.read(1024)
    """

    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self._lock = threading.Lock()
        self._open_files: Dict[str, ResourceEntry] = {}
        self._temp_files: Dict[str, Path] = {}
        self._stats = {
            "opens": 0,
            "closes": 0,
            "evictions": 0,
            "errors": 0,
        }

        # Start cleanup thread if interval is set
        if self.config.cleanup_interval > 0:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""

        def cleanup_loop():
            while True:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_stale()

        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()

    @contextmanager
    def open_file(
        self, path: Path, mode: str = "rb"
    ) -> Generator[Union[BinaryIO, TextIO], None, None]:
        """
        Context manager for opening files with tracking.

        Ensures file is closed even on exceptions.
        Evicts LRU files if at capacity.

        Args:
            path: Path to file
            mode: File open mode

        Yields:
            File handle
        """
        path_str = str(path)

        with self._lock:
            # Check capacity and evict if needed
            self._ensure_capacity()

            # Check if already open
            if path_str in self._open_files:
                entry = self._open_files[path_str]
                entry.last_accessed = time.time()
                entry.access_count += 1
                logger.debug("resource_reuse", path=path_str)
                yield entry.handle
                return

        # Open new file
        try:
            handle = open(path, mode)
            self._stats["opens"] += 1

            with self._lock:
                self._open_files[path_str] = ResourceEntry(
                    path=path_str, handle=handle, mode=mode
                )

            logger.debug(
                "resource_open",
                path=path_str,
                mode=mode,
                open_count=len(self._open_files),
            )

            try:
                yield handle
            finally:
                # Close and remove from tracking
                self._close_file(path_str)

        except Exception as e:
            self._stats["errors"] += 1
            logger.error("resource_open_error", path=path_str, error=str(e))
            raise

    @contextmanager
    def open_audio(self, path: Path) -> Generator[BinaryIO, None, None]:
        """
        Context manager for opening audio files.

        Convenience wrapper with binary read mode.

        Args:
            path: Path to audio file

        Yields:
            Binary file handle
        """
        with self.open_file(path, "rb") as f:
            yield f

    def _close_file(self, path_str: str):
        """Close a tracked file."""
        with self._lock:
            if path_str not in self._open_files:
                return

            entry = self._open_files[path_str]
            try:
                if entry.handle and not entry.handle.closed:
                    entry.handle.close()
                self._stats["closes"] += 1
            except Exception as e:
                logger.warning("resource_close_error", path=path_str, error=str(e))
            finally:
                del self._open_files[path_str]

            logger.debug(
                "resource_close", path=path_str, open_count=len(self._open_files)
            )

    def _ensure_capacity(self):
        """Ensure we have capacity for new files."""
        while len(self._open_files) >= self.config.max_open_files:
            self._evict_lru()

        # Warn if approaching limit
        utilization = len(self._open_files) / self.config.max_open_files
        if utilization >= self.config.warn_threshold:
            logger.warning(
                "resource_high_utilization",
                open_count=len(self._open_files),
                max_count=self.config.max_open_files,
                utilization=round(utilization, 2),
            )

    def _evict_lru(self):
        """Evict least recently used file."""
        if not self._open_files:
            return

        # Find LRU entry
        lru_path = min(
            self._open_files.keys(), key=lambda p: self._open_files[p].last_accessed
        )

        logger.debug("resource_evict", path=lru_path)
        self._close_file(lru_path)
        self._stats["evictions"] += 1

    def _cleanup_stale(self):
        """Clean up files that haven't been accessed recently."""
        stale_threshold = time.time() - (self.config.cleanup_interval * 2)

        with self._lock:
            stale_paths = [
                path
                for path, entry in self._open_files.items()
                if entry.last_accessed < stale_threshold
            ]

        for path in stale_paths:
            logger.debug("resource_stale_cleanup", path=path)
            self._close_file(path)

    @contextmanager
    def temp_file(
        self,
        suffix: str = ".tmp",
        prefix: str = "callwhisper_",
        delete_on_exit: bool = True,
    ) -> Generator[Path, None, None]:
        """
        Context manager for temporary files.

        Tracks temp files and ensures cleanup.

        Args:
            suffix: File extension
            prefix: File name prefix
            delete_on_exit: Delete file when context exits

        Yields:
            Path to temp file
        """
        import tempfile

        # Check temp file capacity
        with self._lock:
            if len(self._temp_files) >= self.config.max_temp_files:
                self._cleanup_temp_files()

        # Create temp file
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        path = Path(path)

        with self._lock:
            self._temp_files[str(path)] = path

        logger.debug("temp_file_created", path=str(path))

        try:
            yield path
        finally:
            if delete_on_exit:
                self._delete_temp_file(path)

    def _delete_temp_file(self, path: Path):
        """Delete a temporary file."""
        path_str = str(path)
        try:
            if path.exists():
                path.unlink()
            with self._lock:
                if path_str in self._temp_files:
                    del self._temp_files[path_str]
            logger.debug("temp_file_deleted", path=path_str)
        except Exception as e:
            logger.warning("temp_file_delete_error", path=path_str, error=str(e))

    def _cleanup_temp_files(self):
        """Clean up oldest temp files to make room."""
        with self._lock:
            if not self._temp_files:
                return

            # Delete oldest temp files
            paths = list(self._temp_files.values())

        # Sort by modification time and delete oldest
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        for path in paths[: len(paths) // 2]:  # Delete oldest half
            self._delete_temp_file(path)

    def close_all(self):
        """Close all tracked files and temp files."""
        with self._lock:
            paths = list(self._open_files.keys())

        for path in paths:
            self._close_file(path)

        with self._lock:
            temp_paths = list(self._temp_files.values())

        for path in temp_paths:
            self._delete_temp_file(path)

        logger.info(
            "resource_close_all", closed=len(paths), temp_deleted=len(temp_paths)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        with self._lock:
            return {
                "open_files": len(self._open_files),
                "max_open_files": self.config.max_open_files,
                "utilization": round(
                    len(self._open_files) / self.config.max_open_files, 2
                ),
                "temp_files": len(self._temp_files),
                "max_temp_files": self.config.max_temp_files,
                "opens": self._stats["opens"],
                "closes": self._stats["closes"],
                "evictions": self._stats["evictions"],
                "errors": self._stats["errors"],
            }

    def get_open_files(self) -> Dict[str, Dict[str, Any]]:
        """Get info about all open files."""
        with self._lock:
            now = time.time()
            return {
                path: {
                    "mode": entry.mode,
                    "opened_at": entry.opened_at,
                    "age_seconds": round(now - entry.opened_at, 1),
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                }
                for path, entry in self._open_files.items()
            }


class AudioResourceManager(ResourceManager):
    """
    Specialized resource manager for audio files.

    Adds audio-specific functionality like format detection
    and streaming support.
    """

    def __init__(self, config: ResourceConfig = None):
        super().__init__(config)

    @contextmanager
    def open_audio_stream(
        self, path: Path, chunk_size: int = 65536
    ) -> Generator[Generator[bytes, None, None], None, None]:
        """
        Open audio file as a streaming generator.

        Memory-efficient reading of large audio files.

        Args:
            path: Path to audio file
            chunk_size: Bytes per chunk

        Yields:
            Generator that yields chunks
        """

        def chunk_generator(handle):
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        with self.open_audio(path) as handle:
            yield chunk_generator(handle)


# Global instance (lazy initialization)
_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()


def get_resource_manager(config: ResourceConfig = None) -> ResourceManager:
    """Get or create the global resource manager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ResourceManager(config)
    return _manager


def get_audio_resource_manager(config: ResourceConfig = None) -> AudioResourceManager:
    """Get or create an audio-specialized resource manager."""
    return AudioResourceManager(config)
