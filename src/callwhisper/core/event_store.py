"""
Event Store Module for Audit Trail

Based on LibV2 orchestrator-architecture patterns:
- State Persistence (chunks 205-206)
- Event sourcing for complete audit trail
- Append-only log for debugging and compliance

Key concepts:
- Every state change is recorded as an event
- Events are immutable once written
- State can be reconstructed by replaying events
- JSONL format for easy parsing and streaming
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import hashlib

from .logging_config import get_core_logger

logger = get_core_logger()


# Event types
class EventType:
    """Standard event types for transcription workflow."""
    SESSION_STARTED = "SESSION_STARTED"
    RECORDING_STARTED = "RECORDING_STARTED"
    RECORDING_STOPPED = "RECORDING_STOPPED"
    NORMALIZATION_STARTED = "NORMALIZATION_STARTED"
    NORMALIZATION_COMPLETED = "NORMALIZATION_COMPLETED"
    TRANSCRIPTION_STARTED = "TRANSCRIPTION_STARTED"
    TRANSCRIPTION_COMPLETED = "TRANSCRIPTION_COMPLETED"
    BUNDLE_STARTED = "BUNDLE_STARTED"
    BUNDLE_COMPLETED = "BUNDLE_COMPLETED"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    ERROR = "ERROR"
    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"
    STATE_CHANGE = "STATE_CHANGE"
    CONFIG_CHANGE = "CONFIG_CHANGE"


@dataclass
class TranscriptionEvent:
    """
    An immutable event in the transcription lifecycle.

    Once created, events should not be modified.
    """
    event_type: str
    timestamp: datetime
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default="")
    previous_hash: str = field(default="")

    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID
            content = f"{self.timestamp.isoformat()}-{self.session_id}-{self.event_type}"
            self.event_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data.get("event_id", ""),
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data["session_id"],
            data=data.get("data", {}),
            previous_hash=data.get("previous_hash", ""),
        )

    def compute_hash(self) -> str:
        """Compute hash of this event for chain integrity."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]


@dataclass
class EventStoreConfig:
    """Configuration for event store."""
    log_dir: Path = field(default_factory=lambda: Path("data/events"))
    max_file_size_mb: int = 10  # Rotate after this size
    max_files: int = 100  # Keep this many rotated files
    sync_interval: float = 1.0  # Flush to disk interval
    enable_hash_chain: bool = True  # Verify event chain integrity


class EventStore:
    """
    Append-only event log for transcription lifecycle.

    Features:
    - Events stored as JSONL (one JSON object per line)
    - Hash chain for integrity verification
    - Async and sync append methods
    - Replay for state reconstruction
    - Temporal queries for analytics

    Example:
        store = EventStore(config)

        # Append event
        await store.append(TranscriptionEvent(
            event_type=EventType.RECORDING_STARTED,
            timestamp=datetime.now(),
            session_id="session_123",
            data={"device": "CABLE Output"}
        ))

        # Replay session
        events = await store.replay("session_123")

        # Query by time range
        events = await store.query_by_time(start_dt, end_dt)
    """

    def __init__(self, config: EventStoreConfig = None):
        self.config = config or EventStoreConfig()
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._last_hash = ""
        self._buffer: List[TranscriptionEvent] = []
        self._listeners: List[Callable[[TranscriptionEvent], None]] = []

        # Ensure log directory exists
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Load last hash for chain continuity
        self._load_last_hash()

        logger.info(
            "event_store_init",
            log_dir=str(self.config.log_dir),
            hash_chain=self.config.enable_hash_chain
        )

    def _get_current_log_path(self) -> Path:
        """Get path to current log file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.config.log_dir / f"events_{date_str}.jsonl"

    def _load_last_hash(self):
        """Load last event hash for chain continuity."""
        if not self.config.enable_hash_chain:
            return

        try:
            log_files = sorted(self.config.log_dir.glob("events_*.jsonl"))
            if not log_files:
                return

            # Read last line of most recent file
            with open(log_files[-1], "r") as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line

                if last_line:
                    event = TranscriptionEvent.from_dict(json.loads(last_line))
                    self._last_hash = event.compute_hash()

        except Exception as e:
            logger.warning("event_store_load_hash_error", error=str(e))

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        log_path = self._get_current_log_path()
        if not log_path.exists():
            return

        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb < self.config.max_file_size_mb:
            return

        # Rotate: rename with timestamp
        timestamp = datetime.now().strftime("%H%M%S")
        rotated_path = log_path.with_suffix(f".{timestamp}.jsonl")
        log_path.rename(rotated_path)

        logger.info("event_store_rotate", old_path=str(log_path), new_path=str(rotated_path))

        # Cleanup old files
        self._cleanup_old_files()

    def _cleanup_old_files(self):
        """Remove oldest files if we exceed max_files."""
        log_files = sorted(self.config.log_dir.glob("events_*.jsonl"))
        while len(log_files) > self.config.max_files:
            oldest = log_files.pop(0)
            oldest.unlink()
            logger.debug("event_store_cleanup", deleted=str(oldest))

    async def append(self, event: TranscriptionEvent) -> None:
        """
        Append event to log (async).

        Thread-safe and adds hash chain if enabled.
        """
        async with self._async_lock:
            self._append_internal(event)

    def append_sync(self, event: TranscriptionEvent) -> None:
        """
        Append event to log (synchronous).

        Thread-safe and adds hash chain if enabled.
        """
        with self._lock:
            self._append_internal(event)

    def _append_internal(self, event: TranscriptionEvent):
        """Internal append implementation."""
        # Add hash chain
        if self.config.enable_hash_chain:
            event.previous_hash = self._last_hash

        # Write to file
        self._rotate_if_needed()
        log_path = self._get_current_log_path()

        with open(log_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        # Update last hash
        if self.config.enable_hash_chain:
            self._last_hash = event.compute_hash()

        logger.debug(
            "event_appended",
            event_type=event.event_type,
            session_id=event.session_id,
            event_id=event.event_id
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("event_listener_error", error=str(e))

    async def replay(
        self,
        session_id: str,
        start_from: Optional[str] = None
    ) -> List[TranscriptionEvent]:
        """
        Replay all events for a session.

        Reconstructs state by returning all events in order.

        Args:
            session_id: Session to replay
            start_from: Optional event_id to start from

        Returns:
            List of events in chronological order
        """
        events = []
        started = start_from is None

        for log_path in sorted(self.config.log_dir.glob("events_*.jsonl")):
            try:
                with open(log_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        event = TranscriptionEvent.from_dict(json.loads(line))

                        if event.session_id != session_id:
                            continue

                        if not started:
                            if event.event_id == start_from:
                                started = True
                            continue

                        events.append(event)

            except Exception as e:
                logger.error("event_replay_error", path=str(log_path), error=str(e))

        logger.info(
            "event_replay",
            session_id=session_id,
            event_count=len(events)
        )

        return events

    async def query_by_time(
        self,
        start: datetime,
        end: datetime,
        event_types: Optional[List[str]] = None
    ) -> List[TranscriptionEvent]:
        """
        Query events in a time range.

        Useful for analytics and debugging.

        Args:
            start: Start datetime
            end: End datetime
            event_types: Optional filter by event type

        Returns:
            List of matching events
        """
        events = []

        for log_path in sorted(self.config.log_dir.glob("events_*.jsonl")):
            try:
                with open(log_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        event = TranscriptionEvent.from_dict(json.loads(line))

                        if event.timestamp < start or event.timestamp > end:
                            continue

                        if event_types and event.event_type not in event_types:
                            continue

                        events.append(event)

            except Exception as e:
                logger.error("event_query_error", path=str(log_path), error=str(e))

        return events

    async def query_by_type(
        self,
        event_type: str,
        limit: int = 100
    ) -> List[TranscriptionEvent]:
        """
        Query most recent events of a specific type.

        Args:
            event_type: Event type to query
            limit: Maximum events to return

        Returns:
            List of matching events (most recent first)
        """
        events = []

        # Read files in reverse order (newest first)
        for log_path in sorted(self.config.log_dir.glob("events_*.jsonl"), reverse=True):
            if len(events) >= limit:
                break

            try:
                # Read file and collect matching events
                file_events = []
                with open(log_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        event = TranscriptionEvent.from_dict(json.loads(line))
                        if event.event_type == event_type:
                            file_events.append(event)

                # Add in reverse order (newest first)
                events.extend(reversed(file_events))

            except Exception as e:
                logger.error("event_query_error", path=str(log_path), error=str(e))

        return events[:limit]

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of a session's events.

        Useful for debugging and status display.
        """
        events = await self.replay(session_id)

        if not events:
            return {"session_id": session_id, "found": False}

        event_counts = {}
        for event in events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        first_event = events[0]
        last_event = events[-1]

        return {
            "session_id": session_id,
            "found": True,
            "event_count": len(events),
            "event_types": event_counts,
            "started_at": first_event.timestamp.isoformat(),
            "last_event_at": last_event.timestamp.isoformat(),
            "duration_seconds": (last_event.timestamp - first_event.timestamp).total_seconds(),
            "last_event_type": last_event.event_type,
            "has_error": any(e.event_type == EventType.ERROR for e in events),
            "is_completed": last_event.event_type in [
                EventType.SESSION_COMPLETED,
                EventType.BUNDLE_COMPLETED
            ],
        }

    async def verify_chain(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify hash chain integrity.

        Returns:
            Dict with verification results
        """
        if not self.config.enable_hash_chain:
            return {"verified": False, "reason": "hash_chain_disabled"}

        events = []
        if session_id:
            events = await self.replay(session_id)
        else:
            # Verify all events
            for log_path in sorted(self.config.log_dir.glob("events_*.jsonl")):
                with open(log_path, "r") as f:
                    for line in f:
                        if line.strip():
                            events.append(TranscriptionEvent.from_dict(json.loads(line)))

        if not events:
            return {"verified": True, "event_count": 0}

        # Verify chain
        previous_hash = ""
        broken_at = None

        for i, event in enumerate(events):
            if event.previous_hash != previous_hash:
                broken_at = i
                break
            previous_hash = event.compute_hash()

        if broken_at is not None:
            return {
                "verified": False,
                "event_count": len(events),
                "broken_at_index": broken_at,
                "broken_event_id": events[broken_at].event_id,
            }

        return {
            "verified": True,
            "event_count": len(events),
        }

    def subscribe(self, callback: Callable[[TranscriptionEvent], None]) -> Callable:
        """
        Subscribe to new events.

        Args:
            callback: Function to call with each new event

        Returns:
            Unsubscribe function
        """
        self._listeners.append(callback)
        return lambda: self._listeners.remove(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get event store statistics."""
        log_files = list(self.config.log_dir.glob("events_*.jsonl"))
        total_size = sum(f.stat().st_size for f in log_files)

        return {
            "log_dir": str(self.config.log_dir),
            "file_count": len(log_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "hash_chain_enabled": self.config.enable_hash_chain,
            "listener_count": len(self._listeners),
        }


# Convenience functions

def create_event(
    event_type: str,
    session_id: str,
    **data
) -> TranscriptionEvent:
    """Create a new event with current timestamp."""
    return TranscriptionEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        session_id=session_id,
        data=data
    )


# Global instance (lazy initialization)
_store: Optional[EventStore] = None
_store_lock = threading.Lock()


def get_event_store(config: EventStoreConfig = None) -> EventStore:
    """Get or create the global event store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = EventStore(config)
    return _store
