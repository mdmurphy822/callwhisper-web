"""
Tests for WebSocket connection manager.

Tests connection lifecycle:
- Connect and disconnect
- Personal message sending
- Broadcast to all clients
- Connection count tracking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from callwhisper.core.connection_manager import ConnectionManager


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(self, manager, mock_websocket):
        """connect() calls accept on websocket."""
        await manager.connect(mock_websocket)

        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_adds_to_connections(self, manager, mock_websocket):
        """connect() adds websocket to active connections."""
        await manager.connect(mock_websocket)

        assert mock_websocket in manager.active_connections
        assert manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_connect_multiple(self, manager):
        """connect() can handle multiple connections."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)
        await manager.connect(ws3)

        assert manager.connection_count == 3

    def test_disconnect_removes_connection(self, manager, mock_websocket):
        """disconnect() removes websocket from active connections."""
        manager.active_connections.append(mock_websocket)
        assert manager.connection_count == 1

        manager.disconnect(mock_websocket)

        assert mock_websocket not in manager.active_connections
        assert manager.connection_count == 0

    def test_disconnect_nonexistent(self, manager, mock_websocket):
        """disconnect() handles non-existent websocket gracefully."""
        # Should not raise
        manager.disconnect(mock_websocket)
        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_send_personal_message(self, manager, mock_websocket):
        """send_personal_message() sends JSON to specific client."""
        message = {"type": "status", "data": "test"}

        await manager.send_personal_message(message, mock_websocket)

        mock_websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self, manager):
        """broadcast() sends message to all connected clients."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()
        manager.active_connections = [ws1, ws2, ws3]

        message = {"type": "broadcast", "data": "hello"}
        await manager.broadcast(message)

        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)
        ws3.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self, manager):
        """broadcast() cleans up connections that fail to receive."""
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("Connection closed")

        manager.active_connections = [ws_good, ws_bad]

        await manager.broadcast({"type": "test"})

        # Good connection should remain
        assert ws_good in manager.active_connections
        # Bad connection should be removed
        assert ws_bad not in manager.active_connections
        assert manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_empty_connections(self, manager):
        """broadcast() handles empty connection list."""
        message = {"type": "test"}

        # Should not raise
        await manager.broadcast(message)

    def test_connection_count_property(self, manager):
        """connection_count returns correct count."""
        assert manager.connection_count == 0

        manager.active_connections = [MagicMock(), MagicMock()]
        assert manager.connection_count == 2

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, manager):
        """Test full connect/message/disconnect lifecycle."""
        ws = AsyncMock()

        # Connect
        await manager.connect(ws)
        assert manager.connection_count == 1

        # Send message
        await manager.send_personal_message({"status": "ok"}, ws)
        ws.send_json.assert_called()

        # Broadcast
        await manager.broadcast({"event": "update"})

        # Disconnect
        manager.disconnect(ws)
        assert manager.connection_count == 0


# ============================================================================
# Batch 7: Advanced Broadcast and Concurrency Tests
# ============================================================================

import asyncio


class TestBroadcastFailures:
    """Tests for partial broadcast failures."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_partial_broadcast_failure(self, manager):
        """Some connections fail during broadcast, others succeed."""
        ws_good1 = AsyncMock()
        ws_good2 = AsyncMock()
        ws_bad1 = AsyncMock()
        ws_bad2 = AsyncMock()

        ws_bad1.send_json.side_effect = Exception("Connection lost")
        ws_bad2.send_json.side_effect = Exception("Timeout")

        manager.active_connections = [ws_good1, ws_bad1, ws_good2, ws_bad2]

        await manager.broadcast({"type": "test"})

        # Good connections should remain
        assert ws_good1 in manager.active_connections
        assert ws_good2 in manager.active_connections

        # Bad connections should be removed
        assert ws_bad1 not in manager.active_connections
        assert ws_bad2 not in manager.active_connections

        assert manager.connection_count == 2

    @pytest.mark.asyncio
    async def test_all_connections_fail(self, manager):
        """All connections fail during broadcast."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        ws1.send_json.side_effect = Exception("Error 1")
        ws2.send_json.side_effect = Exception("Error 2")
        ws3.send_json.side_effect = Exception("Error 3")

        manager.active_connections = [ws1, ws2, ws3]

        await manager.broadcast({"type": "test"})

        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_single_connection_fails(self, manager):
        """Single failing connection in broadcast."""
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("Connection closed")

        manager.active_connections = [ws_bad]

        await manager.broadcast({"type": "test"})

        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_exception_types(self, manager):
        """Different exception types during broadcast."""
        ws_conn_error = AsyncMock()
        ws_timeout = AsyncMock()
        ws_good = AsyncMock()

        ws_conn_error.send_json.side_effect = ConnectionError("Disconnected")
        ws_timeout.send_json.side_effect = TimeoutError("Timeout")

        manager.active_connections = [ws_conn_error, ws_timeout, ws_good]

        await manager.broadcast({"type": "test"})

        assert ws_good in manager.active_connections
        assert manager.connection_count == 1


class TestConnectionEdgeCases:
    """Tests for connection edge cases."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_disconnect_during_broadcast(self, manager):
        """Connection disconnects while broadcast is in progress."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        async def delayed_success(msg):
            await asyncio.sleep(0.01)
            return None

        ws1.send_json = delayed_success
        ws2.send_json = AsyncMock()

        manager.active_connections = [ws1, ws2]

        # Start broadcast
        broadcast_task = asyncio.create_task(manager.broadcast({"type": "test"}))

        # Immediately disconnect ws1
        manager.disconnect(ws1)

        await broadcast_task

        # ws2 should still be connected
        assert ws2 in manager.active_connections

    @pytest.mark.asyncio
    async def test_connect_during_broadcast(self, manager):
        """New connection added while broadcast is in progress."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        async def delayed_send(msg):
            await asyncio.sleep(0.02)
            return None

        ws1.send_json = delayed_send
        manager.active_connections = [ws1]

        # Start broadcast
        broadcast_task = asyncio.create_task(manager.broadcast({"type": "initial"}))

        # Connect new client during broadcast
        await asyncio.sleep(0.01)
        await manager.connect(ws2)

        await broadcast_task

        # Both connections should exist
        assert manager.connection_count == 2

    @pytest.mark.asyncio
    async def test_very_large_message(self, manager):
        """Broadcast with very large message."""
        ws = AsyncMock()
        await manager.connect(ws)

        large_message = {
            "type": "large",
            "data": "x" * 1_000_000,
            "nested": {"key": "value" * 10000}
        }

        await manager.broadcast(large_message)

        ws.send_json.assert_called_once_with(large_message)

    @pytest.mark.asyncio
    async def test_unicode_message(self, manager):
        """Broadcast with unicode content."""
        ws = AsyncMock()
        await manager.connect(ws)

        unicode_message = {
            "type": "unicode",
            "content": "日本語テスト 🎤 émojis",
            "special": "café résumé"
        }

        await manager.broadcast(unicode_message)

        ws.send_json.assert_called_once_with(unicode_message)

    @pytest.mark.asyncio
    async def test_nested_message(self, manager):
        """Broadcast with deeply nested message."""
        ws = AsyncMock()
        await manager.connect(ws)

        nested_message = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep"
                        }
                    }
                }
            }
        }

        await manager.broadcast(nested_message)

        ws.send_json.assert_called_once_with(nested_message)

    @pytest.mark.asyncio
    async def test_empty_message(self, manager):
        """Broadcast with empty message."""
        ws = AsyncMock()
        await manager.connect(ws)

        await manager.broadcast({})

        ws.send_json.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_null_values_in_message(self, manager):
        """Broadcast with null values."""
        ws = AsyncMock()
        await manager.connect(ws)

        message = {
            "type": "status",
            "data": None,
            "optional": None
        }

        await manager.broadcast(message)

        ws.send_json.assert_called_once_with(message)


class TestPersonalMessageEdgeCases:
    """Tests for personal message edge cases."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_personal_message_to_disconnected(self, manager):
        """Send personal message to disconnected client raises."""
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("Connection closed")

        # Should raise since send_json fails
        with pytest.raises(Exception):
            await manager.send_personal_message({"type": "test"}, ws)

    @pytest.mark.asyncio
    async def test_personal_message_very_large(self, manager):
        """Send very large personal message."""
        ws = AsyncMock()
        await manager.connect(ws)

        large_message = {"data": "x" * 1_000_000}
        await manager.send_personal_message(large_message, ws)

        ws.send_json.assert_called_once_with(large_message)


class TestConnectionManagerConcurrency:
    """Tests for concurrent connection operations."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_concurrent_connects(self, manager):
        """Multiple concurrent connection attempts."""
        websockets = [AsyncMock() for _ in range(10)]

        # Connect all concurrently
        await asyncio.gather(*[manager.connect(ws) for ws in websockets])

        assert manager.connection_count == 10
        for ws in websockets:
            assert ws in manager.active_connections

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, manager):
        """Multiple concurrent broadcasts."""
        ws = AsyncMock()
        call_count = 0

        async def track_calls(msg):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)

        ws.send_json = track_calls
        await manager.connect(ws)

        # Send multiple broadcasts concurrently
        await asyncio.gather(*[
            manager.broadcast({"id": i}) for i in range(10)
        ])

        assert call_count == 10

    @pytest.mark.asyncio
    async def test_connect_disconnect_race(self, manager):
        """Rapid connect/disconnect cycles."""
        for _ in range(100):
            ws = AsyncMock()
            await manager.connect(ws)
            manager.disconnect(ws)

        assert manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_during_mass_disconnect(self, manager):
        """Broadcast while many clients disconnect."""
        websockets = []
        for _ in range(20):
            ws = AsyncMock()
            await manager.connect(ws)
            websockets.append(ws)

        # Start broadcast and disconnect half during it
        broadcast_task = asyncio.create_task(manager.broadcast({"type": "test"}))

        for ws in websockets[:10]:
            manager.disconnect(ws)

        await broadcast_task

        # Remaining connections should be <= 10
        assert manager.connection_count <= 10


class TestMultipleConnectionsFromSameClient:
    """Tests for scenarios with same client reconnecting."""

    @pytest.fixture
    def manager(self):
        """Create fresh ConnectionManager for each test."""
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_duplicate_connection(self, manager):
        """Same websocket object connected twice."""
        ws = AsyncMock()

        await manager.connect(ws)
        await manager.connect(ws)

        # Should have 2 entries (no dedup in current implementation)
        assert manager.connection_count == 2

    @pytest.mark.asyncio
    async def test_disconnect_removes_one_instance(self, manager):
        """Disconnect removes only one instance of duplicate."""
        ws = AsyncMock()

        await manager.connect(ws)
        await manager.connect(ws)

        manager.disconnect(ws)

        # Should still have 1 entry
        assert manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_to_duplicates(self, manager):
        """Broadcast sends to same client multiple times if connected multiple times."""
        ws = AsyncMock()

        await manager.connect(ws)
        await manager.connect(ws)

        await manager.broadcast({"type": "test"})

        # Should be called twice
        assert ws.send_json.call_count == 2
