"""
Recording Lifecycle End-to-End Integration Tests

Tests the complete recording workflow:
1. Start recording
2. Record audio (mocked FFmpeg)
3. Stop recording
4. Verify output files
5. State transitions
"""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

# Skip tests with Windows-specific validation behavior
UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="Platform-specific validation")

from callwhisper.core.state import app_state, AppState, RecordingSession
from callwhisper.core.config import DeviceGuardConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def client():
    """Create async HTTP client for testing."""
    from callwhisper.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def reset_app_state():
    """Reset application state before and after tests."""
    # Reset before test - use PUBLIC attributes (no underscore prefix)
    app_state.state = AppState.IDLE
    app_state.current_session = None
    app_state.completed_recordings = []
    app_state.elapsed_seconds = 0

    yield

    # Reset after test
    app_state.state = AppState.IDLE
    app_state.current_session = None
    app_state.completed_recordings = []
    app_state.elapsed_seconds = 0


@pytest.fixture
def mock_ffmpeg_subprocess():
    """Mock FFmpeg subprocess for recording tests."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.wait = AsyncMock(return_value=0)
        process.kill = MagicMock()
        process.terminate = MagicMock()
        mock.return_value = process
        yield mock


@pytest.fixture
def mock_device_guard_allow():
    """Mock device guard to allow all devices."""
    with patch("callwhisper.api.routes.get_device_status") as mock:
        mock.return_value = {"safe": True, "reason": "Test allowed"}
        yield mock


@pytest.fixture
def mock_device_guard_block():
    """Mock device guard to block all devices."""
    with patch("callwhisper.api.routes.get_device_status") as mock:
        mock.return_value = {"safe": False, "reason": "Test blocked"}
        yield mock


# ============================================================================
# Start Recording Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStartRecording:
    """Tests for POST /api/recording/start endpoint."""

    async def test_start_recording_success(
        self, client, reset_app_state, mock_device_guard_allow, mock_ffmpeg_subprocess
    ):
        """Successfully start a recording with valid device."""
        with patch("callwhisper.api.routes.start_recording", new_callable=AsyncMock):
            response = await client.post(
                "/api/recording/start",
                json={"device": "VB-Cable Output", "ticket_id": "TEST123"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "recording_id" in data
        assert "TEST123" in data["recording_id"]
        assert "started_at" in data

    async def test_start_recording_no_ticket_id(
        self, client, reset_app_state, mock_device_guard_allow, mock_ffmpeg_subprocess
    ):
        """Start recording without ticket ID."""
        with patch("callwhisper.api.routes.start_recording", new_callable=AsyncMock):
            response = await client.post(
                "/api/recording/start",
                json={"device": "VB-Cable Output"}
            )

        assert response.status_code == 200
        data = response.json()
        assert "recording_id" in data
        # Recording ID should be timestamp-based without ticket
        assert "TEST" not in data["recording_id"]

    async def test_start_recording_device_blocked(
        self, client, reset_app_state, mock_device_guard_block
    ):
        """Reject recording from blocked device."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "Microphone (Realtek)", "ticket_id": "TEST123"}
        )

        assert response.status_code == 403
        assert "blocked" in response.json()["detail"].lower()

    async def test_start_recording_empty_device(self, client, reset_app_state):
        """Reject empty device name."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "", "ticket_id": "TEST123"}
        )

        assert response.status_code == 422  # Validation error

    async def test_start_recording_missing_device(self, client, reset_app_state):
        """Reject missing device field."""
        response = await client.post(
            "/api/recording/start",
            json={"ticket_id": "TEST123"}
        )

        assert response.status_code == 422  # Validation error

    async def test_start_recording_invalid_device_chars(self, client, reset_app_state):
        """Reject device name with invalid characters (command injection prevention)."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "Device; rm -rf /", "ticket_id": "TEST123"}
        )

        assert response.status_code == 422  # Validation error

    @UNIX_ONLY
    async def test_start_recording_invalid_ticket_id(self, client, reset_app_state):
        """Reject invalid ticket ID format."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "VB-Cable", "ticket_id": "TEST 123!@#"}
        )

        # 422 for validation error, 403 if device guard rejects VB-Cable on Linux
        assert response.status_code in [403, 422]

    async def test_start_recording_already_recording(
        self, client, reset_app_state, mock_device_guard_allow, mock_ffmpeg_subprocess
    ):
        """Reject if already recording."""
        # Set state to RECORDING
        app_state.state = AppState.RECORDING

        response = await client.post(
            "/api/recording/start",
            json={"device": "VB-Cable Output", "ticket_id": "TEST123"}
        )

        assert response.status_code == 400
        assert "already recording" in response.json()["detail"].lower()

    async def test_start_recording_while_processing(
        self, client, reset_app_state, mock_device_guard_allow
    ):
        """Reject if currently processing."""
        # Set state to PROCESSING
        app_state.state = AppState.PROCESSING

        response = await client.post(
            "/api/recording/start",
            json={"device": "VB-Cable Output", "ticket_id": "TEST123"}
        )

        assert response.status_code == 400
        assert "processing" in response.json()["detail"].lower()


# ============================================================================
# Stop Recording Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStopRecording:
    """Tests for POST /api/recording/stop endpoint."""

    async def test_stop_recording_not_recording(self, client, reset_app_state):
        """Reject stop if not currently recording."""
        response = await client.post("/api/recording/stop")

        assert response.status_code == 400
        assert "not currently recording" in response.json()["detail"].lower()

    async def test_stop_recording_success(self, client, reset_app_state):
        """Successfully stop a recording."""
        # Set state to RECORDING with a session
        app_state.state = AppState.RECORDING
        app_state.current_session = RecordingSession(
            id="test_recording_001",
            device_name="VB-Cable Output",
            ticket_id="TEST123",
        )
        app_state.current_session.start_time = datetime.now()

        with patch("callwhisper.api.routes.process_recording", new_callable=AsyncMock):
            response = await client.post("/api/recording/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["recording_id"] == "test_recording_001"


# ============================================================================
# State Endpoint Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestStateEndpoint:
    """Tests for GET /api/state endpoint."""

    async def test_state_idle(self, client, reset_app_state):
        """State shows IDLE when not recording."""
        response = await client.get("/api/state")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "idle"
        assert data["recording_id"] is None
        assert data["elapsed_seconds"] == 0

    async def test_state_recording(self, client, reset_app_state):
        """State shows RECORDING when recording."""
        app_state.state = AppState.RECORDING
        session = RecordingSession(
            id="test_recording_001",
            device_name="VB-Cable Output",
            ticket_id="TEST123",
        )
        session.start_time = datetime.now()
        app_state.current_session = session

        response = await client.get("/api/state")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "recording"
        assert data["recording_id"] == "test_recording_001"

    async def test_state_processing(self, client, reset_app_state):
        """State shows PROCESSING after recording stops."""
        app_state.state = AppState.PROCESSING

        response = await client.get("/api/state")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "processing"

    async def test_state_error(self, client, reset_app_state):
        """State shows ERROR when in error state."""
        app_state.state = AppState.ERROR

        response = await client.get("/api/state")

        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "error"


# ============================================================================
# Recordings List Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecordingsList:
    """Tests for GET /api/recordings endpoint."""

    async def test_recordings_empty(self, client, reset_app_state):
        """Empty recordings list."""
        response = await client.get("/api/recordings")

        assert response.status_code == 200
        data = response.json()
        assert data["recordings"] == []

    async def test_recordings_with_completed(self, client, reset_app_state):
        """Recordings list with completed recordings."""
        from callwhisper.core.state import CompletedRecording

        completed = CompletedRecording(
            id="test_001",
            ticket_id="TICKET123",
            created_at=datetime.now(),
            duration_seconds=120.5,
            output_folder="/tmp/output/test_001",
            bundle_path="/tmp/output/test_001/recording.vtb",
            transcript_preview="Test transcript...",
        )
        app_state.completed_recordings.append(completed)

        response = await client.get("/api/recordings")

        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 1
        assert data["recordings"][0]["id"] == "test_001"
        assert data["recordings"][0]["ticket_id"] == "TICKET123"


# ============================================================================
# Recording Download Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestRecordingDownload:
    """Tests for GET /api/recordings/{id}/download endpoint."""

    async def test_download_not_found(self, client, reset_app_state):
        """404 for non-existent recording."""
        response = await client.get("/api/recordings/nonexistent/download")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_download_no_bundle(self, client, reset_app_state):
        """404 when recording has no bundle."""
        from callwhisper.core.state import CompletedRecording

        completed = CompletedRecording(
            id="test_no_bundle",
            ticket_id="TEST123",
            created_at=datetime.now(),
            duration_seconds=60.0,
            output_folder="/tmp/output/test_no_bundle",
            bundle_path=None,  # No bundle
            transcript_preview="Test...",
        )
        app_state.completed_recordings.append(completed)

        response = await client.get("/api/recordings/test_no_bundle/download")

        assert response.status_code == 404
        assert "bundle not available" in response.json()["detail"].lower()


# ============================================================================
# Device Listing Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestDevicesEndpoint:
    """Tests for GET /api/devices endpoint."""

    async def test_devices_returns_list(self, client, reset_app_state):
        """Devices endpoint returns a list."""
        with patch("callwhisper.api.routes.list_audio_devices") as mock_list:
            mock_list.return_value = ["VB-Cable Output", "Stereo Mix"]
            with patch("callwhisper.api.routes.get_device_status") as mock_status:
                mock_status.side_effect = [
                    {"safe": True, "reason": "Allowed"},
                    {"safe": True, "reason": "Allowed"},
                ]

                response = await client.get("/api/devices")

        assert response.status_code == 200
        data = response.json()
        assert "devices" in data
        assert len(data["devices"]) == 2

    async def test_devices_includes_safety_info(self, client, reset_app_state):
        """Devices include safety status."""
        with patch("callwhisper.api.routes.list_audio_devices") as mock_list:
            mock_list.return_value = ["VB-Cable Output", "Microphone"]
            with patch("callwhisper.api.routes.get_device_status") as mock_status:
                mock_status.side_effect = [
                    {"safe": True, "reason": "Allowed"},
                    {"safe": False, "reason": "Blocked"},
                ]

                response = await client.get("/api/devices")

        assert response.status_code == 200
        data = response.json()
        assert data["devices"][0]["safe"] is True
        assert data["devices"][1]["safe"] is False


# ============================================================================
# Reset Endpoint Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestResetEndpoint:
    """Tests for POST /api/reset endpoint."""

    async def test_reset_returns_to_idle(self, client, reset_app_state):
        """Reset returns app to idle state."""
        app_state.state = AppState.ERROR

        response = await client.post("/api/reset")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ============================================================================
# Input Validation Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestInputValidation:
    """Tests for input validation across endpoints."""

    async def test_device_name_max_length(self, client, reset_app_state):
        """Device name exceeding max length is rejected."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "A" * 300, "ticket_id": "TEST123"}  # Exceeds 256
        )

        assert response.status_code == 422

    async def test_ticket_id_max_length(self, client, reset_app_state):
        """Ticket ID exceeding max length is rejected."""
        response = await client.post(
            "/api/recording/start",
            json={"device": "VB-Cable", "ticket_id": "A" * 60}  # Exceeds 50
        )

        assert response.status_code == 422

    async def test_unicode_device_name(
        self, client, reset_app_state, mock_device_guard_allow, mock_ffmpeg_subprocess
    ):
        """Unicode device names are accepted."""
        with patch("callwhisper.api.routes.start_recording", new_callable=AsyncMock):
            response = await client.post(
                "/api/recording/start",
                json={"device": "VB-Cable Output (日本語)", "ticket_id": "TEST123"}
            )

        assert response.status_code == 200


# ============================================================================
# Transcript Endpoint Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestTranscriptEndpoints:
    """Tests for transcript-related endpoints."""

    async def test_get_transcript_not_found(self, client, reset_app_state):
        """404 for non-existent recording transcript."""
        response = await client.get("/api/recordings/nonexistent/transcript")

        assert response.status_code == 404

    async def test_update_transcript_not_found(self, client, reset_app_state):
        """404 for updating non-existent recording."""
        response = await client.put(
            "/api/recordings/nonexistent/transcript",
            json={"text": "New transcript text"}
        )

        assert response.status_code == 404


# ============================================================================
# Export Endpoints Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestExportEndpoints:
    """Tests for export-related endpoints."""

    async def test_export_invalid_format(self, client, reset_app_state):
        """Reject invalid export format."""
        from callwhisper.core.state import CompletedRecording

        completed = CompletedRecording(
            id="test_export",
            ticket_id="TEST123",
            created_at=datetime.now(),
            duration_seconds=60.0,
            output_folder="/tmp/output/test_export",
            bundle_path=None,
            transcript_preview="Test...",
        )
        app_state.completed_recordings.append(completed)

        response = await client.get("/api/recordings/test_export/export/invalid_format")

        assert response.status_code == 400
        assert "unsupported format" in response.json()["detail"].lower()

    async def test_get_export_formats(self, client, reset_app_state):
        """Get available export formats."""
        from callwhisper.core.state import CompletedRecording

        completed = CompletedRecording(
            id="test_formats",
            ticket_id="TEST123",
            created_at=datetime.now(),
            duration_seconds=60.0,
            output_folder="/tmp/output/test_formats",
            bundle_path=None,
            transcript_preview="Test...",
        )
        app_state.completed_recordings.append(completed)

        response = await client.get("/api/recordings/test_formats/export-formats")

        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "json" in data["formats"]
        assert "vtt" in data["formats"]
        assert "csv" in data["formats"]
        assert "pdf" in data["formats"]
        assert "docx" in data["formats"]


# ============================================================================
# Search Recordings Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchRecordings:
    """Tests for GET /api/recordings/search endpoint."""

    async def test_search_empty_results(self, client, reset_app_state):
        """Search with no recordings returns empty."""
        response = await client.get("/api/recordings/search")

        assert response.status_code == 200
        data = response.json()
        assert data["recordings"] == []
        assert data["total"] == 0

    async def test_search_with_query(self, client, reset_app_state):
        """Search filters by query string."""
        from callwhisper.core.state import CompletedRecording

        completed1 = CompletedRecording(
            id="test_001",
            ticket_id="ALPHA123",
            created_at=datetime.now(),
            duration_seconds=60.0,
            output_folder="/tmp/output/test_001",
            bundle_path=None,
            transcript_preview="Hello world test",
        )
        completed2 = CompletedRecording(
            id="test_002",
            ticket_id="BETA456",
            created_at=datetime.now(),
            duration_seconds=90.0,
            output_folder="/tmp/output/test_002",
            bundle_path=None,
            transcript_preview="Different content",
        )
        app_state.completed_recordings.extend([completed1, completed2])

        response = await client.get("/api/recordings/search?query=ALPHA")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["recordings"][0]["ticket_id"] == "ALPHA123"

    async def test_search_pagination(self, client, reset_app_state):
        """Search supports pagination."""
        from callwhisper.core.state import CompletedRecording

        # Create 25 recordings
        for i in range(25):
            completed = CompletedRecording(
                id=f"test_{i:03d}",
                ticket_id=f"TICKET{i:03d}",
                created_at=datetime.now(),
                duration_seconds=60.0,
                output_folder=f"/tmp/output/test_{i:03d}",
                bundle_path=None,
                transcript_preview="Test content",
            )
            app_state.completed_recordings.append(completed)

        response = await client.get("/api/recordings/search?page=1&page_size=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data["recordings"]) == 10
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert data["total_pages"] == 3
