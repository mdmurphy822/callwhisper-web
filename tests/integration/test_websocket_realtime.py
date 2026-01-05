"""
Integration tests for WebSocket real-time updates.

Tests the WebSocket system including:
- State change broadcasts
- Timer updates during recording
- Transcription progress updates
- Multiple concurrent clients
- Connection/disconnection handling
- Ping/pong keepalive
- State request responses
- Reconnection scenarios
"""

import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from callwhisper.core.connection_manager import ConnectionManager, manager
from callwhisper.api.websocket import (
    websocket_endpoint,
    broadcast_state_change,
    setup_state_callbacks,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def connection_manager():
    """Create a fresh connection manager for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def multiple_mock_websockets():
    """Create multiple mock WebSockets."""
    websockets = []
    for i in range(5):
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.close = AsyncMock()
        ws.id = i  # For identification
        websockets.append(ws)
    return websockets


# ============================================================================
# ConnectionManager Tests
# ============================================================================

class TestConnectionManagerBasics:
    """Basic tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, connection_manager, mock_websocket):
        """Connect accepts the WebSocket."""
        await connection_manager.connect(mock_websocket)

        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_adds_to_list(self, connection_manager, mock_websocket):
        """Connect adds WebSocket to active connections."""
        await connection_manager.connect(mock_websocket)

        assert mock_websocket in connection_manager.active_connections
        assert connection_manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_list(self, connection_manager, mock_websocket):
        """Disconnect removes WebSocket from active connections."""
        await connection_manager.connect(mock_websocket)
        connection_manager.disconnect(mock_websocket)

        assert mock_websocket not in connection_manager.active_connections
        assert connection_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_safe(self, connection_manager, mock_websocket):
        """Disconnecting nonexistent WebSocket is safe."""
        # Should not raise
        connection_manager.disconnect(mock_websocket)
        assert connection_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_connection_count(self, connection_manager, multiple_mock_websockets):
        """Connection count tracks active connections."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        assert connection_manager.connection_count == len(multiple_mock_websockets)

        # Disconnect some
        for ws in multiple_mock_websockets[:2]:
            connection_manager.disconnect(ws)

        assert connection_manager.connection_count == len(multiple_mock_websockets) - 2


class TestConnectionManagerMessaging:
    """Tests for ConnectionManager messaging."""

    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager, mock_websocket):
        """Send message to specific client."""
        await connection_manager.connect(mock_websocket)

        message = {"type": "test", "data": "hello"}
        await connection_manager.send_personal_message(message, mock_websocket)

        mock_websocket.send_json.assert_called_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_all_clients(
        self, connection_manager, multiple_mock_websockets
    ):
        """Broadcast sends to all connected clients."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        message = {"type": "broadcast", "data": "hello all"}
        await connection_manager.broadcast(message)

        for ws in multiple_mock_websockets:
            ws.send_json.assert_called_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_list(self, connection_manager):
        """Broadcast to empty list is safe."""
        message = {"type": "broadcast", "data": "hello"}
        # Should not raise
        await connection_manager.broadcast(message)

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(
        self, connection_manager, multiple_mock_websockets
    ):
        """Broadcast removes clients that fail to receive."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Make first two fail
        multiple_mock_websockets[0].send_json.side_effect = Exception("Connection lost")
        multiple_mock_websockets[1].send_json.side_effect = Exception("Disconnected")

        message = {"type": "test"}
        await connection_manager.broadcast(message)

        # Failed connections should be removed
        assert connection_manager.connection_count == 3
        assert multiple_mock_websockets[0] not in connection_manager.active_connections
        assert multiple_mock_websockets[1] not in connection_manager.active_connections


class TestConnectionManagerConcurrency:
    """Concurrency tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connection_manager):
        """Multiple concurrent connections are handled."""
        websockets = []
        for i in range(20):
            ws = AsyncMock(spec=WebSocket)
            ws.accept = AsyncMock()
            websockets.append(ws)

        # Connect all concurrently
        await asyncio.gather(*[
            connection_manager.connect(ws) for ws in websockets
        ])

        assert connection_manager.connection_count == 20

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(
        self, connection_manager, multiple_mock_websockets
    ):
        """Multiple concurrent broadcasts are handled."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Send multiple broadcasts concurrently
        messages = [{"type": "test", "n": i} for i in range(10)]
        await asyncio.gather(*[
            connection_manager.broadcast(msg) for msg in messages
        ])

        # Each client should receive all messages
        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 10

    @pytest.mark.asyncio
    async def test_connect_during_broadcast(self, connection_manager):
        """New connection during broadcast is safe."""
        existing = AsyncMock(spec=WebSocket)
        existing.accept = AsyncMock()
        existing.send_json = AsyncMock()
        await connection_manager.connect(existing)

        # Start broadcast
        broadcast_started = asyncio.Event()
        broadcast_done = asyncio.Event()

        async def slow_send(msg):
            broadcast_started.set()
            await asyncio.sleep(0.1)

        existing.send_json = slow_send

        async def do_broadcast():
            await connection_manager.broadcast({"type": "test"})
            broadcast_done.set()

        async def connect_new():
            await broadcast_started.wait()
            new_ws = AsyncMock(spec=WebSocket)
            new_ws.accept = AsyncMock()
            # This should not crash even during broadcast
            # Note: This may or may not receive the current broadcast depending on timing
            await connection_manager.connect(new_ws)

        await asyncio.gather(do_broadcast(), connect_new())


# ============================================================================
# WebSocket Endpoint Tests
# ============================================================================

class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint handler."""

    @pytest.mark.asyncio
    async def test_endpoint_accepts_connection(self, mock_websocket):
        """Endpoint accepts WebSocket connection."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.disconnect = MagicMock()

            await websocket_endpoint(mock_websocket)

            mock_manager.connect.assert_called_once_with(mock_websocket)

    @pytest.mark.asyncio
    async def test_endpoint_sends_initial_state(self, mock_websocket):
        """Endpoint sends current state on connect."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.disconnect = MagicMock()

            with patch('callwhisper.api.websocket.app_state') as mock_state:
                mock_state.get_state_info.return_value = {"state": "idle"}

                await websocket_endpoint(mock_websocket)

                mock_manager.send_personal_message.assert_called()
                call_args = mock_manager.send_personal_message.call_args[0]
                assert call_args[0]["type"] == "connected"

    @pytest.mark.asyncio
    async def test_endpoint_handles_ping(self, mock_websocket):
        """Endpoint responds to ping with pong."""
        # Return ping then disconnect
        mock_websocket.receive_text.side_effect = ["ping", WebSocketDisconnect()]

        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.disconnect = MagicMock()

            with patch('callwhisper.api.websocket.app_state') as mock_state:
                mock_state.get_state_info.return_value = {}

                await websocket_endpoint(mock_websocket)

                # Check pong was sent
                calls = mock_manager.send_personal_message.call_args_list
                pong_calls = [c for c in calls if c[0][0].get("type") == "pong"]
                assert len(pong_calls) == 1

    @pytest.mark.asyncio
    async def test_endpoint_handles_state_request(self, mock_websocket):
        """Endpoint responds to state request."""
        mock_websocket.receive_text.side_effect = ["state", WebSocketDisconnect()]

        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.disconnect = MagicMock()

            with patch('callwhisper.api.websocket.app_state') as mock_state:
                mock_state.get_state_info.return_value = {"state": "recording"}

                await websocket_endpoint(mock_websocket)

                # Check state response was sent
                calls = mock_manager.send_personal_message.call_args_list
                state_calls = [c for c in calls if c[0][0].get("type") == "state"]
                assert len(state_calls) == 1

    @pytest.mark.asyncio
    async def test_endpoint_disconnects_on_error(self, mock_websocket):
        """Endpoint cleans up on WebSocket disconnect."""
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.disconnect = MagicMock()

            with patch('callwhisper.api.websocket.app_state') as mock_state:
                mock_state.get_state_info.return_value = {}

                await websocket_endpoint(mock_websocket)

                mock_manager.disconnect.assert_called_once_with(mock_websocket)


class TestBroadcastStateChange:
    """Tests for state change broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_state_change_calls_manager(self):
        """State change broadcasts through manager."""
        with patch('callwhisper.api.websocket.manager') as mock_manager:
            mock_manager.broadcast = AsyncMock()

            data = {"state": "recording", "progress": 50}
            await broadcast_state_change(data)

            mock_manager.broadcast.assert_called_once_with(data)


# ============================================================================
# State Update Broadcasting Tests
# ============================================================================

class TestStateUpdateBroadcasting:
    """Tests for real-time state update broadcasting."""

    @pytest.mark.asyncio
    async def test_recording_start_broadcast(
        self, connection_manager, multiple_mock_websockets
    ):
        """Recording start is broadcast to all clients."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        state_update = {
            "type": "state_change",
            "state": "recording",
            "session": {
                "id": "20241229_120000_TEST",
                "device": "Test Device",
                "ticket_id": "TEST-001"
            }
        }

        await connection_manager.broadcast(state_update)

        for ws in multiple_mock_websockets:
            ws.send_json.assert_called_with(state_update)

    @pytest.mark.asyncio
    async def test_progress_update_broadcast(
        self, connection_manager, multiple_mock_websockets
    ):
        """Progress updates are broadcast."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Simulate multiple progress updates
        for progress in range(0, 101, 10):
            update = {
                "type": "progress",
                "progress": progress,
                "message": f"Processing... {progress}%"
            }
            await connection_manager.broadcast(update)

        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 11  # 0, 10, 20, ..., 100

    @pytest.mark.asyncio
    async def test_timer_update_broadcast(
        self, connection_manager, multiple_mock_websockets
    ):
        """Timer updates during recording are broadcast."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Simulate timer updates
        for seconds in range(1, 6):
            update = {
                "type": "timer",
                "elapsed_seconds": seconds,
                "formatted": f"00:00:{seconds:02d}"
            }
            await connection_manager.broadcast(update)

        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 5

    @pytest.mark.asyncio
    async def test_transcription_progress_broadcast(
        self, connection_manager, multiple_mock_websockets
    ):
        """Transcription progress is broadcast."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        stages = [
            (0, "Starting transcription..."),
            (20, "Normalizing audio..."),
            (50, "Transcribing..."),
            (80, "Processing segments..."),
            (100, "Complete"),
        ]

        for progress, message in stages:
            update = {
                "type": "transcription_progress",
                "progress": progress,
                "message": message
            }
            await connection_manager.broadcast(update)

        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == len(stages)

    @pytest.mark.asyncio
    async def test_error_broadcast(
        self, connection_manager, multiple_mock_websockets
    ):
        """Error states are broadcast."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        error_update = {
            "type": "error",
            "state": "error",
            "error_message": "Recording failed: device disconnected"
        }

        await connection_manager.broadcast(error_update)

        for ws in multiple_mock_websockets:
            ws.send_json.assert_called_with(error_update)


# ============================================================================
# Multiple Client Tests
# ============================================================================

class TestMultipleClients:
    """Tests for multiple concurrent WebSocket clients."""

    @pytest.mark.asyncio
    async def test_each_client_receives_updates(
        self, connection_manager, multiple_mock_websockets
    ):
        """Each connected client receives all updates."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Send several updates
        updates = [
            {"type": "state", "state": "idle"},
            {"type": "state", "state": "recording"},
            {"type": "progress", "progress": 50},
            {"type": "state", "state": "complete"},
        ]

        for update in updates:
            await connection_manager.broadcast(update)

        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == len(updates)

    @pytest.mark.asyncio
    async def test_late_joiner_receives_current_state(
        self, connection_manager, mock_websocket
    ):
        """Client connecting mid-session receives current state."""
        # Simulate a client joining during recording
        current_state = {
            "type": "connected",
            "state": "recording",
            "session": {
                "elapsed_seconds": 120,
                "device": "Test Device"
            }
        }

        await connection_manager.connect(mock_websocket)
        await connection_manager.send_personal_message(current_state, mock_websocket)

        mock_websocket.send_json.assert_called_with(current_state)

    @pytest.mark.asyncio
    async def test_client_disconnect_doesnt_affect_others(
        self, connection_manager, multiple_mock_websockets
    ):
        """One client disconnecting doesn't affect others."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Disconnect first client
        connection_manager.disconnect(multiple_mock_websockets[0])

        # Broadcast should still work for others
        update = {"type": "test"}
        await connection_manager.broadcast(update)

        # First client shouldn't receive (already disconnected)
        # Others should receive
        for ws in multiple_mock_websockets[1:]:
            ws.send_json.assert_called_with(update)


# ============================================================================
# Reconnection Tests
# ============================================================================

class TestReconnection:
    """Tests for client reconnection scenarios."""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(
        self, connection_manager, mock_websocket
    ):
        """Client can reconnect after disconnect."""
        await connection_manager.connect(mock_websocket)
        connection_manager.disconnect(mock_websocket)

        # Should be able to reconnect
        mock_websocket.accept.reset_mock()
        await connection_manager.connect(mock_websocket)

        assert mock_websocket in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_reconnect_receives_state(
        self, connection_manager, mock_websocket
    ):
        """Reconnected client receives current state."""
        await connection_manager.connect(mock_websocket)
        connection_manager.disconnect(mock_websocket)

        # Reconnect
        await connection_manager.connect(mock_websocket)

        # Should be able to receive messages
        update = {"type": "state", "state": "idle"}
        await connection_manager.send_personal_message(update, mock_websocket)

        # Should have been called during both connections plus the state message
        assert mock_websocket.send_json.call_count >= 1


# ============================================================================
# Edge Cases
# ============================================================================

class TestWebSocketEdgeCases:
    """Edge case tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_very_large_message(self, connection_manager, mock_websocket):
        """Handle very large broadcast messages."""
        await connection_manager.connect(mock_websocket)

        # Large message with lots of data
        large_data = {
            "type": "transcript",
            "text": "word " * 10000,  # ~50KB of text
            "segments": [{"start": i, "end": i+1, "text": f"seg{i}"} for i in range(1000)]
        }

        await connection_manager.broadcast(large_data)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_rapid_state_changes(
        self, connection_manager, multiple_mock_websockets
    ):
        """Handle rapid succession of state changes."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Rapid state changes
        states = ["idle", "recording", "processing", "complete"] * 25  # 100 changes

        for state in states:
            await connection_manager.broadcast({"type": "state", "state": state})

        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 100

    @pytest.mark.asyncio
    async def test_special_characters_in_message(
        self, connection_manager, mock_websocket
    ):
        """Handle special characters in messages."""
        await connection_manager.connect(mock_websocket)

        message = {
            "type": "transcript",
            "text": "Unicode: \u4e2d\u6587 \ud83c\udf89 <script>alert('xss')</script>",
            "special": "Line\nBreak\tTab"
        }

        await connection_manager.broadcast(message)

        mock_websocket.send_json.assert_called_with(message)

    @pytest.mark.asyncio
    async def test_empty_message(self, connection_manager, mock_websocket):
        """Handle empty message broadcast."""
        await connection_manager.connect(mock_websocket)

        await connection_manager.broadcast({})

        mock_websocket.send_json.assert_called_with({})


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestWebSocketErrorHandling:
    """Error handling tests for WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_send_to_closed_connection(
        self, connection_manager, mock_websocket
    ):
        """Handle sending to closed connection."""
        await connection_manager.connect(mock_websocket)

        # Simulate closed connection
        mock_websocket.send_json.side_effect = Exception("Connection closed")

        # Should not raise
        await connection_manager.broadcast({"type": "test"})

        # Should have removed the failed connection
        assert connection_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_partial_broadcast_failure(
        self, connection_manager, multiple_mock_websockets
    ):
        """Some clients fail during broadcast, others succeed."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Make middle clients fail
        multiple_mock_websockets[2].send_json.side_effect = Exception("Failed")

        await connection_manager.broadcast({"type": "test"})

        # Failed connections removed, others remain
        assert connection_manager.connection_count == 4
        assert multiple_mock_websockets[2] not in connection_manager.active_connections

    @pytest.mark.asyncio
    async def test_all_connections_fail(
        self, connection_manager, multiple_mock_websockets
    ):
        """All connections fail during broadcast."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)
            ws.send_json.side_effect = Exception("All fail")

        await connection_manager.broadcast({"type": "test"})

        assert connection_manager.connection_count == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestWebSocketIntegration:
    """Integration tests for WebSocket system."""

    @pytest.mark.asyncio
    async def test_full_recording_session_updates(
        self, connection_manager, multiple_mock_websockets
    ):
        """Simulate a full recording session with updates."""
        # Connect clients
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # 1. Recording starts
        await connection_manager.broadcast({
            "type": "state_change",
            "state": "recording",
            "session": {"id": "test123", "device": "Test"}
        })

        # 2. Timer updates during recording
        for seconds in [1, 2, 3, 4, 5]:
            await connection_manager.broadcast({
                "type": "timer",
                "elapsed_seconds": seconds
            })

        # 3. Recording stops
        await connection_manager.broadcast({
            "type": "state_change",
            "state": "processing"
        })

        # 4. Transcription progress
        for progress in [0, 25, 50, 75, 100]:
            await connection_manager.broadcast({
                "type": "progress",
                "progress": progress
            })

        # 5. Complete
        await connection_manager.broadcast({
            "type": "state_change",
            "state": "idle",
            "recording_complete": True
        })

        # Verify all clients received all updates
        # 1 + 5 + 1 + 5 + 1 = 13 updates
        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 13

    @pytest.mark.asyncio
    async def test_queue_progress_updates(
        self, connection_manager, multiple_mock_websockets
    ):
        """Simulate queue processing with updates."""
        for ws in multiple_mock_websockets:
            await connection_manager.connect(ws)

        # Simulate queue with 3 jobs
        for job_num in range(1, 4):
            # Job starts
            await connection_manager.broadcast({
                "type": "queue_update",
                "current_job": job_num,
                "total_jobs": 3,
                "status": "processing"
            })

            # Job progress
            for progress in [0, 50, 100]:
                await connection_manager.broadcast({
                    "type": "job_progress",
                    "job": job_num,
                    "progress": progress
                })

            # Job complete
            await connection_manager.broadcast({
                "type": "queue_update",
                "current_job": job_num,
                "total_jobs": 3,
                "status": "complete"
            })

        # Queue complete
        await connection_manager.broadcast({
            "type": "queue_update",
            "status": "queue_complete"
        })

        # Each client should receive all updates
        # 3 jobs * (1 start + 3 progress + 1 complete) = 15 + 1 final = 16
        for ws in multiple_mock_websockets:
            assert ws.send_json.call_count == 16
