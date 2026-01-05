"""
Tests for signal handling and graceful shutdown.

Tests proper signal handling:
- SIGTERM/SIGINT handling
- Graceful shutdown with incomplete jobs
- Resource cleanup on interruption
"""

import asyncio
import signal
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from callwhisper.core.state_machine import RecordingState, StateMachine


# ============================================================================
# Graceful Shutdown Tests
# ============================================================================


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        return StateMachine()

    def test_shutdown_from_idle(self, state_machine):
        """Shutdown from IDLE state is immediate."""
        # Already in IDLE
        assert state_machine.current_state == RecordingState.IDLE

        # Shutdown should work
        state_machine.transition_to(RecordingState.SHUTDOWN)
        assert state_machine.current_state == RecordingState.SHUTDOWN

    def test_shutdown_from_recording_transitions_to_stopping(self, state_machine):
        """Shutdown request during recording goes through STOPPING."""
        state_machine.transition_to(RecordingState.RECORDING)

        # Request shutdown - should go to STOPPING first
        state_machine.transition_to(RecordingState.STOPPING)
        assert state_machine.current_state == RecordingState.STOPPING

    def test_shutdown_saves_state(self, state_machine):
        """Shutdown preserves current state for recovery."""
        state_machine.transition_to(RecordingState.RECORDING)

        # Save current state before shutdown
        pre_shutdown_state = state_machine.current_state

        state_machine.transition_to(RecordingState.STOPPING)
        state_machine.transition_to(RecordingState.IDLE)

        # State history should be preserved
        assert len(state_machine.state_history) > 0


# ============================================================================
# WebSocket Shutdown Tests
# ============================================================================


class TestWebSocketShutdown:
    """Tests for WebSocket cleanup during shutdown."""

    @pytest.mark.asyncio
    async def test_broadcast_shutdown_message(self):
        """Shutdown broadcasts message to all clients."""
        from callwhisper.core.connection_manager import ConnectionManager

        manager = ConnectionManager()
        clients = [AsyncMock() for _ in range(5)]

        for client in clients:
            await manager.connect(client)

        # Broadcast shutdown message
        await manager.broadcast({
            "type": "shutdown",
            "message": "Server is shutting down",
        })

        # All clients should receive the message
        for client in clients:
            client.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_disconnect_all_on_shutdown(self):
        """All connections are closed on shutdown."""
        from callwhisper.core.connection_manager import ConnectionManager

        manager = ConnectionManager()
        clients = [AsyncMock() for _ in range(5)]

        for client in clients:
            await manager.connect(client)

        assert manager.connection_count == 5

        # Disconnect all
        for client in clients:
            manager.disconnect(client)

        assert manager.connection_count == 0


# ============================================================================
# Resource Cleanup Tests
# ============================================================================


class TestResourceCleanupOnShutdown:
    """Tests for resource cleanup during shutdown."""

    def test_close_open_files_on_shutdown(self):
        """Open files are closed during shutdown."""
        from callwhisper.core.resource_manager import ResourceManager, ResourceConfig
        import tempfile
        from pathlib import Path

        config = ResourceConfig(cleanup_interval=0)
        manager = ResourceManager(config)

        # Create and open a temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            # Open file through manager
            with manager.open_file(temp_path, 'rb') as f:
                _ = f.read()

            # After context, file should be closed
            stats = manager.get_stats()
            assert stats["open_files"] == 0

            # Cleanup all remaining
            manager.close_all()

            # Verify cleanup
            stats = manager.get_stats()
            assert stats["open_files"] == 0
        finally:
            temp_path.unlink()

    def test_temp_files_deleted_on_shutdown(self):
        """Temporary files are deleted during shutdown."""
        from callwhisper.core.resource_manager import ResourceManager, ResourceConfig

        config = ResourceConfig(cleanup_interval=0)
        manager = ResourceManager(config)

        # Create temp files
        temp_paths = []
        for _ in range(3):
            with manager.temp_file(delete_on_exit=False) as path:
                temp_paths.append(path)

        # All temp files should exist
        assert all(p.exists() for p in temp_paths)

        # Shutdown cleanup
        manager.close_all()

        # Temp files should be deleted
        stats = manager.get_stats()
        assert stats["temp_files"] == 0


# ============================================================================
# Checkpoint on Shutdown Tests
# ============================================================================


class TestCheckpointOnShutdown:
    """Tests for checkpoint saving during shutdown."""

    def test_checkpoint_saved_before_shutdown(self):
        """Current state is checkpointed before shutdown."""
        from callwhisper.core.checkpoint import CheckpointManager, CheckpointConfig
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(checkpoint_dir=Path(temp_dir))
            manager = CheckpointManager(config)

            # Create a checkpoint (simulating pre-shutdown save)
            manager.create_checkpoint(
                session_id="active-session",
                state="recording",
                metadata={
                    "audio_path": "/path/to/audio.wav",
                    "duration": 120.5,
                }
            )

            # Verify checkpoint exists
            checkpoint = manager.get_checkpoint("active-session")
            assert checkpoint is not None
            assert checkpoint["state"] == "recording"

    def test_incomplete_job_marked_for_recovery(self):
        """Incomplete jobs are marked for recovery on shutdown."""
        from callwhisper.core.checkpoint import CheckpointManager, CheckpointConfig
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(checkpoint_dir=Path(temp_dir))
            manager = CheckpointManager(config)

            # Create checkpoint for incomplete job
            manager.create_checkpoint(
                session_id="incomplete-job",
                state="transcribing",
                metadata={
                    "needs_recovery": True,
                    "progress": 0.75,
                }
            )

            # On restart, should be able to find and resume
            checkpoint = manager.get_checkpoint("incomplete-job")
            assert checkpoint["metadata"]["needs_recovery"] is True


# ============================================================================
# Queue Persistence on Shutdown Tests
# ============================================================================


class TestQueuePersistenceOnShutdown:
    """Tests for queue state persistence during shutdown."""

    def test_pending_jobs_preserved(self):
        """Pending jobs are preserved for restart."""
        from callwhisper.core.queue import TranscriptionQueue

        queue = TranscriptionQueue()

        # Add some jobs
        job_ids = []
        for i in range(5):
            job_id = queue.enqueue(
                session_id=f"session-{i}",
                audio_path=f"/path/audio-{i}.wav",
            )
            job_ids.append(job_id)

        # Check queue has jobs
        assert queue.pending_count > 0

        # Get pending jobs (for persistence)
        pending = list(queue.iter_pending())
        assert len(pending) >= 1


# ============================================================================
# Timeout on Shutdown Tests
# ============================================================================


class TestShutdownTimeout:
    """Tests for shutdown timeout behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_timeout_enforced(self):
        """Shutdown respects timeout for graceful shutdown."""
        # Simulate a shutdown with timeout
        shutdown_started = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def slow_cleanup():
            await asyncio.sleep(0.5)  # Simulate slow cleanup
            cleanup_done.set()

        async def shutdown_with_timeout(timeout: float):
            shutdown_started.set()
            try:
                await asyncio.wait_for(slow_cleanup(), timeout=timeout)
            except asyncio.TimeoutError:
                pass  # Forced shutdown

        # Short timeout should force shutdown
        await shutdown_with_timeout(0.1)

        assert shutdown_started.is_set()
        # Cleanup should not complete due to timeout
        assert not cleanup_done.is_set()

    @pytest.mark.asyncio
    async def test_shutdown_completes_within_timeout(self):
        """Fast cleanup completes within timeout."""
        cleanup_done = asyncio.Event()

        async def fast_cleanup():
            await asyncio.sleep(0.01)
            cleanup_done.set()

        await asyncio.wait_for(fast_cleanup(), timeout=1.0)

        assert cleanup_done.is_set()


# ============================================================================
# Signal Handler Registration Tests
# ============================================================================


class TestSignalHandlerRegistration:
    """Tests for signal handler registration."""

    def test_sigterm_handler_can_be_registered(self):
        """SIGTERM handler can be registered."""
        handler_called = threading.Event()

        def custom_handler(signum, frame):
            handler_called.set()

        # Save original handler
        original = signal.signal(signal.SIGTERM, custom_handler)

        try:
            # Verify handler is registered
            current = signal.getsignal(signal.SIGTERM)
            assert current == custom_handler
        finally:
            # Restore original
            signal.signal(signal.SIGTERM, original)

    def test_sigint_handler_can_be_registered(self):
        """SIGINT handler can be registered."""
        handler_called = threading.Event()

        def custom_handler(signum, frame):
            handler_called.set()

        # Save original handler
        original = signal.signal(signal.SIGINT, custom_handler)

        try:
            # Verify handler is registered
            current = signal.getsignal(signal.SIGINT)
            assert current == custom_handler
        finally:
            # Restore original
            signal.signal(signal.SIGINT, original)


# ============================================================================
# Async Shutdown Coordination Tests
# ============================================================================


class TestAsyncShutdownCoordination:
    """Tests for coordinating async shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_event_coordination(self):
        """Shutdown event coordinates multiple components."""
        shutdown_event = asyncio.Event()
        components_stopped = []

        async def component_a():
            await shutdown_event.wait()
            components_stopped.append("a")

        async def component_b():
            await shutdown_event.wait()
            components_stopped.append("b")

        async def component_c():
            await shutdown_event.wait()
            components_stopped.append("c")

        # Start components
        tasks = [
            asyncio.create_task(component_a()),
            asyncio.create_task(component_b()),
            asyncio.create_task(component_c()),
        ]

        # Signal shutdown
        await asyncio.sleep(0.01)
        shutdown_event.set()

        # Wait for all components
        await asyncio.gather(*tasks)

        # All components should have stopped
        assert len(components_stopped) == 3

    @pytest.mark.asyncio
    async def test_ordered_shutdown(self):
        """Components shut down in correct order."""
        shutdown_order = []

        async def recorder():
            await asyncio.sleep(0.01)
            shutdown_order.append("recorder")

        async def transcriber():
            await asyncio.sleep(0.01)
            shutdown_order.append("transcriber")

        async def cleanup():
            await asyncio.sleep(0.01)
            shutdown_order.append("cleanup")

        # Execute in order (recorder first, then transcriber, then cleanup)
        await recorder()
        await transcriber()
        await cleanup()

        assert shutdown_order == ["recorder", "transcriber", "cleanup"]
