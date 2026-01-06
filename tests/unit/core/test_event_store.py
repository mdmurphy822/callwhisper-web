"""
Tests for event store module.

Tests event sourcing and audit trail:
- Event creation and serialization
- Append and replay operations
- Hash chain verification
- Temporal and type queries
- Event subscription
"""

import asyncio
import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip tests with platform-specific issues
UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Platform-specific behavior")

from callwhisper.core.event_store import (
    EventStore,
    EventStoreConfig,
    EventType,
    TranscriptionEvent,
    create_event,
    get_event_store,
)


# ============================================================================
# EventType Tests
# ============================================================================


class TestEventType:
    """Tests for EventType constants."""

    def test_standard_event_types_exist(self):
        """Standard event types are defined."""
        assert EventType.SESSION_STARTED == "SESSION_STARTED"
        assert EventType.RECORDING_STARTED == "RECORDING_STARTED"
        assert EventType.TRANSCRIPTION_COMPLETED == "TRANSCRIPTION_COMPLETED"
        assert EventType.ERROR == "ERROR"
        assert EventType.STATE_CHANGE == "STATE_CHANGE"


# ============================================================================
# TranscriptionEvent Tests
# ============================================================================


class TestTranscriptionEvent:
    """Tests for TranscriptionEvent dataclass."""

    def test_event_creation(self):
        """Event can be created with required fields."""
        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-123",
        )

        assert event.event_type == EventType.SESSION_STARTED
        assert event.session_id == "session-123"
        assert event.event_id != ""  # Auto-generated

    def test_event_with_data(self):
        """Event can include additional data."""
        event = TranscriptionEvent(
            event_type=EventType.RECORDING_STARTED,
            timestamp=datetime.now(),
            session_id="session-123",
            data={"device": "CABLE Output", "sample_rate": 48000},
        )

        assert event.data["device"] == "CABLE Output"
        assert event.data["sample_rate"] == 48000

    def test_event_id_auto_generated(self):
        """Event ID is automatically generated."""
        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-123",
        )

        assert len(event.event_id) == 16
        assert event.event_id.isalnum()

    def test_to_dict_serialization(self):
        """Event can be serialized to dict."""
        timestamp = datetime.now()
        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=timestamp,
            session_id="session-123",
            data={"key": "value"},
        )

        d = event.to_dict()

        assert d["event_type"] == EventType.SESSION_STARTED
        assert d["session_id"] == "session-123"
        assert d["timestamp"] == timestamp.isoformat()
        assert d["data"] == {"key": "value"}

    def test_from_dict_deserialization(self):
        """Event can be deserialized from dict."""
        data = {
            "event_id": "abc123",
            "event_type": EventType.RECORDING_STOPPED,
            "timestamp": "2024-01-15T10:30:00",
            "session_id": "session-456",
            "data": {"duration": 120},
            "previous_hash": "xyz789",
        }

        event = TranscriptionEvent.from_dict(data)

        assert event.event_id == "abc123"
        assert event.event_type == EventType.RECORDING_STOPPED
        assert event.session_id == "session-456"
        assert event.data["duration"] == 120
        assert event.previous_hash == "xyz789"

    def test_compute_hash(self):
        """Event hash is deterministic."""
        event = TranscriptionEvent(
            event_id="fixed-id",
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            session_id="session-123",
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 32

    def test_different_events_different_hashes(self):
        """Different events have different hashes."""
        event1 = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-1",
        )
        event2 = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-2",
        )

        assert event1.compute_hash() != event2.compute_hash()


# ============================================================================
# EventStoreConfig Tests
# ============================================================================


class TestEventStoreConfig:
    """Tests for EventStoreConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = EventStoreConfig()

        assert config.max_file_size_mb == 10
        assert config.max_files == 100
        assert config.sync_interval == 1.0
        assert config.enable_hash_chain is True

    def test_custom_values(self):
        """Config accepts custom values."""
        config = EventStoreConfig(
            log_dir=Path("/custom/path"),
            max_file_size_mb=50,
            enable_hash_chain=False,
        )

        assert config.log_dir == Path("/custom/path")
        assert config.max_file_size_mb == 50
        assert config.enable_hash_chain is False


# ============================================================================
# EventStore Basic Tests
# ============================================================================


class TestEventStoreBasic:
    """Tests for basic EventStore operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for event logs."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store with temp directory."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_append_event(self, store, temp_dir):
        """Events can be appended."""
        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="test-session",
        )

        await store.append(event)

        # Verify file was created
        log_files = list(temp_dir.glob("events_*.jsonl"))
        assert len(log_files) == 1

    def test_append_sync(self, store, temp_dir):
        """Events can be appended synchronously."""
        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="test-session",
        )

        store.append_sync(event)

        log_files = list(temp_dir.glob("events_*.jsonl"))
        assert len(log_files) == 1

    @pytest.mark.asyncio
    async def test_append_multiple_events(self, store, temp_dir):
        """Multiple events can be appended."""
        for i in range(5):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
                data={"index": i},
            )
            await store.append(event)

        # Read and count events
        log_path = list(temp_dir.glob("events_*.jsonl"))[0]
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
            assert len(lines) == 5


# ============================================================================
# Replay Tests
# ============================================================================


class TestReplay:
    """Tests for event replay."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_replay_session(self, store):
        """Replay returns all events for a session."""
        # Add events for the session
        for event_type in [
            EventType.SESSION_STARTED,
            EventType.RECORDING_STARTED,
            EventType.RECORDING_STOPPED,
        ]:
            event = TranscriptionEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id="target-session",
            )
            await store.append(event)

        # Add event for different session
        other_event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="other-session",
        )
        await store.append(other_event)

        # Replay target session
        events = await store.replay("target-session")

        assert len(events) == 3
        assert all(e.session_id == "target-session" for e in events)

    @pytest.mark.asyncio
    async def test_replay_empty_session(self, store):
        """Replay returns empty list for unknown session."""
        events = await store.replay("nonexistent-session")
        assert events == []

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_replay_start_from(self, store):
        """Replay can start from specific event."""
        events_created = []
        for i in range(5):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
                data={"index": i},
            )
            await store.append(event)
            events_created.append(event)

        # Start from third event
        events = await store.replay(
            "test-session",
            start_from=events_created[2].event_id
        )

        # Should get events after the start point
        assert len(events) == 2


# ============================================================================
# Query Tests
# ============================================================================


class TestQueries:
    """Tests for event queries."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_query_by_time(self, store):
        """Query events by time range."""
        now = datetime.now()

        # Create events at different times
        for i in range(5):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=now - timedelta(hours=i),
                session_id="test-session",
            )
            await store.append(event)

        # Query last 2 hours
        events = await store.query_by_time(
            start=now - timedelta(hours=2),
            end=now + timedelta(hours=1),
        )

        # Should get recent events
        assert len(events) >= 2

    @pytest.mark.asyncio
    async def test_query_by_type(self, store):
        """Query events by type."""
        # Create various event types
        for event_type in [
            EventType.SESSION_STARTED,
            EventType.ERROR,
            EventType.ERROR,
            EventType.SESSION_COMPLETED,
        ]:
            event = TranscriptionEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id="test-session",
            )
            await store.append(event)

        errors = await store.query_by_type(EventType.ERROR, limit=10)

        assert len(errors) == 2
        assert all(e.event_type == EventType.ERROR for e in errors)

    @pytest.mark.asyncio
    async def test_query_by_type_with_limit(self, store):
        """Query respects limit parameter."""
        for i in range(10):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
            )
            await store.append(event)

        events = await store.query_by_type(EventType.STATE_CHANGE, limit=3)

        assert len(events) == 3


# ============================================================================
# Hash Chain Tests
# ============================================================================


class TestHashChain:
    """Tests for hash chain integrity."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store with hash chain enabled."""
        config = EventStoreConfig(
            log_dir=temp_dir,
            enable_hash_chain=True
        )
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_hash_chain_created(self, store, temp_dir):
        """Events are linked via hash chain."""
        for i in range(3):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
            )
            await store.append(event)

        # Read events and check chain
        log_path = list(temp_dir.glob("events_*.jsonl"))[0]
        events = []
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    events.append(TranscriptionEvent.from_dict(json.loads(line)))

        # Second event should reference first event's hash
        assert events[0].previous_hash == ""  # First has no previous
        assert events[1].previous_hash != ""  # Second has previous
        assert events[2].previous_hash != ""  # Third has previous

    @pytest.mark.asyncio
    async def test_verify_chain_valid(self, store):
        """Valid chain passes verification."""
        for i in range(5):
            event = TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
            )
            await store.append(event)

        result = await store.verify_chain("test-session")

        assert result["verified"] is True
        assert result["event_count"] == 5


# ============================================================================
# Session Summary Tests
# ============================================================================


class TestSessionSummary:
    """Tests for session summary."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_get_session_summary(self, store):
        """Get summary of session events."""
        # Create a complete session
        events = [
            EventType.SESSION_STARTED,
            EventType.RECORDING_STARTED,
            EventType.RECORDING_STOPPED,
            EventType.TRANSCRIPTION_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]

        for event_type in events:
            event = TranscriptionEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id="test-session",
            )
            await store.append(event)
            await asyncio.sleep(0.01)  # Ensure different timestamps

        summary = await store.get_session_summary("test-session")

        assert summary["found"] is True
        assert summary["event_count"] == 5
        assert summary["is_completed"] is True
        assert summary["has_error"] is False

    @pytest.mark.asyncio
    async def test_get_session_summary_not_found(self, store):
        """Summary for unknown session."""
        summary = await store.get_session_summary("nonexistent")

        assert summary["found"] is False

    @pytest.mark.asyncio
    async def test_get_session_summary_with_error(self, store):
        """Summary detects error events."""
        events = [
            EventType.SESSION_STARTED,
            EventType.ERROR,
        ]

        for event_type in events:
            event = TranscriptionEvent(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id="error-session",
            )
            await store.append(event)

        summary = await store.get_session_summary("error-session")

        assert summary["has_error"] is True
        assert summary["is_completed"] is False


# ============================================================================
# Subscription Tests
# ============================================================================


class TestSubscription:
    """Tests for event subscription."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_subscribe_receives_events(self, store):
        """Subscriber receives appended events."""
        received = []

        def callback(event):
            received.append(event)

        store.subscribe(callback)

        event = TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="test-session",
        )
        await store.append(event)

        assert len(received) == 1
        assert received[0].session_id == "test-session"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, store):
        """Unsubscribe stops receiving events."""
        received = []

        def callback(event):
            received.append(event)

        unsubscribe = store.subscribe(callback)

        # First event
        await store.append(TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-1",
        ))

        # Unsubscribe
        unsubscribe()

        # Second event (should not be received)
        await store.append(TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="session-2",
        ))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_subscriber_exception_caught(self, store):
        """Subscriber exceptions don't break append."""
        def bad_callback(event):
            raise ValueError("test error")

        good_received = []

        def good_callback(event):
            good_received.append(event)

        store.subscribe(bad_callback)
        store.subscribe(good_callback)

        # Should not raise
        await store.append(TranscriptionEvent(
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(),
            session_id="test-session",
        ))

        # Good callback should still receive
        assert len(good_received) == 1


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for event store statistics."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def store(self, temp_dir):
        """Create event store."""
        config = EventStoreConfig(log_dir=temp_dir)
        return EventStore(config)

    @pytest.mark.asyncio
    async def test_get_stats(self, store, temp_dir):
        """Get store statistics."""
        # Add some events
        for i in range(5):
            await store.append(TranscriptionEvent(
                event_type=EventType.STATE_CHANGE,
                timestamp=datetime.now(),
                session_id="test-session",
            ))

        stats = store.get_stats()

        assert stats["log_dir"] == str(temp_dir)
        assert stats["file_count"] >= 1
        assert stats["total_size_bytes"] > 0


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_event(self):
        """create_event helper creates event."""
        event = create_event(
            EventType.RECORDING_STARTED,
            "session-123",
            device="Microphone",
            sample_rate=44100
        )

        assert event.event_type == EventType.RECORDING_STARTED
        assert event.session_id == "session-123"
        assert event.data["device"] == "Microphone"
        assert event.data["sample_rate"] == 44100
        assert event.timestamp is not None


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global event store singleton."""

    def test_get_event_store_creates_instance(self):
        """get_event_store creates instance on first call."""
        import callwhisper.core.event_store as module
        original = module._store
        module._store = None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = EventStoreConfig(log_dir=Path(temp_dir))
                store = get_event_store(config)
                assert isinstance(store, EventStore)
        finally:
            module._store = original

    def test_get_event_store_returns_same_instance(self):
        """get_event_store returns same instance."""
        import callwhisper.core.event_store as module
        original = module._store
        module._store = None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = EventStoreConfig(log_dir=Path(temp_dir))
                store1 = get_event_store(config)
                store2 = get_event_store()
                assert store1 is store2
        finally:
            module._store = original
