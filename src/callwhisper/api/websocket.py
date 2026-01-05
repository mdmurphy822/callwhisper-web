"""WebSocket handler for real-time updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.connection_manager import manager
from ..core.state import app_state
from ..core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time state updates."""
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else 0

    logger.info(
        "websocket_connected",
        client_host=client_host,
        client_port=client_port,
        active_connections=len(manager.active_connections) + 1,
    )

    await manager.connect(websocket)

    # Send current state on connect
    await manager.send_personal_message(
        {
            "type": "connected",
            **app_state.get_state_info(),
        },
        websocket,
    )

    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()

            # Client can send ping/pong or request state
            if data == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
            elif data == "state":
                await manager.send_personal_message(
                    {"type": "state", **app_state.get_state_info()}, websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(
            "websocket_disconnected",
            client_host=client_host,
            client_port=client_port,
            active_connections=len(manager.active_connections),
            reason="client_disconnect",
        )
    except Exception as e:
        manager.disconnect(websocket)
        logger.warning(
            "websocket_error",
            client_host=client_host,
            client_port=client_port,
            error=str(e),
            error_type=type(e).__name__,
            active_connections=len(manager.active_connections),
        )


# Register the broadcast callback with app state
async def broadcast_state_change(data: dict):
    """Broadcast state changes to all connected clients."""
    await manager.broadcast(data)


# This will be called during app startup
def setup_state_callbacks():
    """Set up state change callbacks for WebSocket broadcasting."""
    app_state.add_state_callback(broadcast_state_change)


# Auto-setup when module loads
setup_state_callbacks()
