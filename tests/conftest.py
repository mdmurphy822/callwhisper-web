"""
Shared pytest fixtures for CallWhisper tests.

Based on LibV2 Python programming course patterns:
- Fixtures for dependency injection
- Async fixtures for async tests
- Mock fixtures for external dependencies
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest
from httpx import AsyncClient, ASGITransport

# Add src to path for imports
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Disable rate limiting for all tests by patching get_settings
# This must happen BEFORE the app module is imported
# =============================================================================
def _create_test_settings():
    """Create settings with rate limiting disabled for testing."""
    from callwhisper.core.config import Settings, SecurityConfig

    # Create settings with rate limiting disabled and debug endpoints enabled
    security = SecurityConfig(
        rate_limit_enabled=False,
        debug_endpoints_enabled=True,  # Enable debug endpoints for testing
    )
    return Settings(security=security)


# Patch get_settings before any test imports the app
_original_get_settings = None


def _patched_get_settings():
    """Return settings with rate limiting disabled."""
    return _create_test_settings()


# Apply the patch at module load time
from callwhisper.core import config as config_module
_original_get_settings = config_module.get_settings
config_module.get_settings = _patched_get_settings
# Also clear the lru_cache to ensure fresh settings
config_module.get_settings.cache_clear = lambda: None  # No-op since we replaced it


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir: Path) -> Path:
    """Create a temporary checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_output_dir(temp_dir: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.server.host = "127.0.0.1"
    settings.server.port = 8000
    settings.server.open_browser = False
    settings.output.directory = "output"
    settings.output.create_bundle = True
    settings.device_guard.enabled = True
    settings.device_guard.allowlist = ["Stereo Mix", "VB-Audio"]
    settings.device_guard.blocklist = ["Microphone"]
    settings.transcription.model = "small"
    settings.transcription.timeout = 300
    settings.recording.timeout = 7200
    settings.recording.sample_rate = 44100
    return settings


@pytest.fixture
def sample_session():
    """Create a sample recording session for testing."""
    from callwhisper.core.state import RecordingSession
    return RecordingSession(
        id="20241229_120000_TEST123",
        device_name="Stereo Mix (Realtek)",
        ticket_id="TEST123",
    )


@pytest.fixture
def sample_completed_recording():
    """Create a sample completed recording for testing."""
    from callwhisper.core.state import CompletedRecording
    return CompletedRecording(
        id="20241229_120000_TEST123",
        ticket_id="TEST123",
        created_at=datetime.now(),
        duration_seconds=120.5,
        output_folder="/tmp/output/20241229_120000_TEST123",
        bundle_path="/tmp/output/20241229_120000_TEST123/recording.vtb",
        transcript_preview="This is a test transcript...",
    )


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing without running actual processes."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b"output", b""))
        process.wait = AsyncMock(return_value=0)
        process.kill = MagicMock()
        mock.return_value = process
        yield mock


@pytest.fixture
def mock_subprocess_failure():
    """Mock subprocess that fails."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.returncode = 1
        process.communicate = AsyncMock(return_value=(b"", b"error"))
        process.wait = AsyncMock(return_value=1)
        mock.return_value = process
        yield mock


@pytest.fixture
def mock_subprocess_timeout():
    """Mock subprocess that times out."""
    with patch("asyncio.wait_for") as mock:
        mock.side_effect = asyncio.TimeoutError()
        yield mock


@pytest.fixture
async def app():
    """Create FastAPI application for testing."""
    from callwhisper.main import app as fastapi_app
    yield fastapi_app


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def reset_app_state():
    """Reset application state before and after tests."""
    from callwhisper.core.state import app_state, AppState

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
def reset_metrics():
    """Reset metrics collector before tests."""
    from callwhisper.core.metrics import metrics
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:
    """Create a sample WAV file for testing."""
    import wave
    import struct

    audio_path = temp_dir / "test_audio.wav"

    # Create a simple 1-second mono WAV file
    sample_rate = 44100
    duration = 1  # seconds
    frequency = 440  # Hz (A note)

    with wave.open(str(audio_path), 'w') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        for i in range(sample_rate * duration):
            # Generate sine wave
            import math
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

    return audio_path


# Fault injection fixtures

@pytest.fixture
def inject_ffmpeg_failure():
    """Inject FFmpeg failure for fault injection testing."""
    from callwhisper.core.exceptions import ProcessExecutionError
    with patch("callwhisper.services.recorder.run_ffmpeg") as mock:
        mock.side_effect = ProcessExecutionError("FFmpeg crashed", "ffmpeg", 1)
        yield mock


@pytest.fixture
def inject_whisper_failure():
    """Inject Whisper failure for fault injection testing."""
    from callwhisper.core.exceptions import TranscriptionError
    with patch("callwhisper.services.transcriber.run_whisper") as mock:
        mock.side_effect = TranscriptionError("Whisper failed", details={"model": "small"})
        yield mock


@pytest.fixture
def inject_disk_full():
    """Inject disk full error for fault injection testing."""
    with patch("shutil.disk_usage") as mock:
        mock.return_value = MagicMock(free=0, total=100*1024*1024*1024)
        yield mock


# ============================================================================
# Service Test Fixtures (Phase 6)
# ============================================================================

@pytest.fixture
def sample_transcript_files(temp_dir: Path) -> Path:
    """Create sample transcript files for export testing."""
    (temp_dir / "transcript.txt").write_text(
        "Hello world. This is a test transcript for export testing."
    )
    (temp_dir / "transcript.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,500\nHello world.\n\n"
        "2\n00:00:02,500 --> 00:00:05,000\nThis is a test transcript."
    )
    return temp_dir


@pytest.fixture
def mock_recording_session():
    """Create mock recording session for bundler tests."""
    from callwhisper.core.state import RecordingSession
    from datetime import timezone

    session = MagicMock(spec=RecordingSession)
    session.id = "20241229_120000_TEST123"
    session.ticket_id = "TEST123"
    session.device_name = "Test Device"
    session.start_time = datetime(2024, 12, 29, 12, 0, 0, tzinfo=timezone.utc)
    session.end_time = datetime(2024, 12, 29, 12, 1, 0, tzinfo=timezone.utc)
    return session


@pytest.fixture
def sample_wav_file(temp_dir: Path) -> Path:
    """Create a sample WAV file with actual audio data."""
    import wave
    import struct
    import math

    audio_path = temp_dir / "test_audio.wav"
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz

    with wave.open(str(audio_path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(sample_rate * duration):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

    return audio_path


@pytest.fixture
def sample_vtb_bundle(temp_dir: Path) -> Path:
    """Create a sample VTB bundle for testing."""
    import zipfile
    import json
    from callwhisper.services.bundler import VTB_MIMETYPE

    vtb_path = temp_dir / "test_bundle.vtb"

    manifest = {
        "version": "1.0.0",
        "format": "vtb",
        "recording_id": "test_recording",
        "ticket_id": "TEST-001",
        "duration_seconds": 60.0,
        "transcript_word_count": 10,
    }

    with zipfile.ZipFile(vtb_path, 'w') as zf:
        zf.writestr("mimetype", VTB_MIMETYPE, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.json", json.dumps(manifest))
        zf.writestr("META-INF/hashes.json", json.dumps({"files": {}}))
        zf.writestr("transcript/transcript.txt", "Test transcript content")

    return vtb_path


@pytest.fixture
def reset_job_queue():
    """Reset job queue singleton before and after tests."""
    from callwhisper.services.job_queue import reset_job_queue as _reset

    _reset()
    yield
    _reset()
