"""
Integration tests for process failure handling.

Tests how the system handles various failure scenarios:
- FFmpeg process failures (killed, crashed, non-zero exit)
- Whisper process failures (timeout, crash, missing model)
- Process never starts (missing executable)
- Signal handling (SIGTERM, SIGKILL)
- Output file corruption
- Cleanup after failures
"""

import asyncio
import pytest
import subprocess
import signal
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import wave
import struct
import math

# Skip tests that have platform-specific subprocess behavior
UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="UNIX subprocess behavior")

from callwhisper.services.recorder import (
    start_recording,
    stop_recording,
    finalize_recording,
    is_recording,
    get_recording_stats,
)
from callwhisper.services.transcriber import (
    transcribe_audio,
    transcribe_chunk,
    get_audio_duration_seconds,
)
from callwhisper.services.normalizer import normalize_audio
from callwhisper.core.exceptions import (
    RecordingError,
    TranscriptionError,
    ProcessTimeoutError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_session():
    """Create a mock recording session."""
    session = MagicMock()
    session.id = "20241229_120000_TEST123"
    session.device_name = "Stereo Mix (Realtek)"
    session.output_folder = None
    session.ticket_id = "TEST-001"
    return session


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.audio.sample_rate = 44100
    settings.audio.channels = 1
    settings.transcription.model = "ggml-base.bin"
    settings.transcription.language = "en"
    settings.transcription.beam_size = 5
    settings.transcription.best_of = 5
    return settings


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file."""
    audio_path = temp_dir / "audio_raw.wav"
    sample_rate = 16000
    duration = 2

    with wave.open(str(audio_path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(sample_rate * duration):
            value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

    return audio_path


@pytest.fixture
def reset_recorder_state():
    """Reset global recorder state before/after tests."""
    import callwhisper.services.recorder as recorder
    # Reset global state
    with recorder._state_lock:
        recorder._ffmpeg_process = None
        recorder._current_output_folder = None
    yield
    with recorder._state_lock:
        recorder._ffmpeg_process = None
        recorder._current_output_folder = None


# ============================================================================
# FFmpeg Recording Failure Tests
# ============================================================================

class TestFFmpegStartupFailures:
    """Tests for FFmpeg startup failures."""

    @pytest.mark.asyncio
    async def test_ffmpeg_not_found(
        self, mock_session, mock_settings, reset_recorder_state
    ):
        """FFmpeg executable not found."""
        with patch(
            "callwhisper.services.recorder.get_ffmpeg_path",
            return_value=Path("/nonexistent/ffmpeg")
        ):
            with patch(
                "subprocess.Popen",
                side_effect=FileNotFoundError("ffmpeg not found")
            ):
                with pytest.raises(RecordingError, match="Failed to start"):
                    await start_recording(mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_ffmpeg_permission_denied(
        self, mock_session, mock_settings, reset_recorder_state
    ):
        """FFmpeg executable has no execute permission."""
        with patch(
            "subprocess.Popen",
            side_effect=PermissionError("Permission denied")
        ):
            with pytest.raises(RecordingError, match="Failed to start"):
                await start_recording(mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_ffmpeg_immediate_exit(
        self, mock_session, mock_settings, temp_dir, reset_recorder_state
    ):
        """FFmpeg exits immediately after start (bad arguments, etc.)."""
        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = 1  # Already exited
                process.stdin = MagicMock()
                mock_popen.return_value = process

                with pytest.raises(RecordingError, match="FFmpeg failed to start"):
                    await start_recording(mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_ffmpeg_invalid_device(
        self, mock_session, mock_settings, temp_dir, reset_recorder_state
    ):
        """FFmpeg fails because device is invalid."""
        mock_session.device_name = "Nonexistent Device"

        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = 1  # Exit with error
                process.stdin = MagicMock()
                mock_popen.return_value = process

                # Create log file with error message
                output_folder = temp_dir / mock_session.id
                output_folder.mkdir(parents=True, exist_ok=True)
                (output_folder / "ffmpeg.log").write_text(
                    "Device not found: Nonexistent Device"
                )

                with pytest.raises(RecordingError):
                    await start_recording(mock_session, mock_settings)


class TestFFmpegRuntimeFailures:
    """Tests for FFmpeg failures during recording."""

    @pytest.mark.asyncio
    async def test_ffmpeg_crashes_during_recording(self, temp_dir):
        """FFmpeg process crashes during recording."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = -11  # SIGSEGV
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
            mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
            mock_process.terminate = MagicMock()
            mock_process.kill = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                result = await stop_recording()

                # Should clean up despite crash
                mock_process.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_ffmpeg_stops_responding(self, temp_dir):
        """FFmpeg stops responding and needs force kill."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = None  # Still running
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
            # Keeps timing out
            mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
            mock_process.terminate = MagicMock()
            mock_process.kill = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                await stop_recording()

                # Should escalate to kill
                mock_process.kill.assert_called()

    @pytest.mark.asyncio
    async def test_ffmpeg_stdin_write_fails(self, temp_dir):
        """Writing 'q' to FFmpeg stdin fails."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.stdin.write.side_effect = BrokenPipeError()
            mock_process.stdin.flush = MagicMock()
            mock_process.wait.return_value = 0
            mock_process.terminate = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                # Should handle gracefully
                result = await stop_recording()
                # Process should still be cleaned up


class TestOutputFileCorruption:
    """Tests for output file corruption scenarios."""

    @pytest.mark.asyncio
    async def test_output_file_zero_bytes(self, temp_dir):
        """Output file is empty (0 bytes)."""
        # Create empty audio file
        audio_path = temp_dir / "audio_raw.wav"
        audio_path.write_bytes(b"")

        # Normalization should handle this
        with patch("asyncio.create_subprocess_exec") as mock_proc:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(
                return_value=(b"", b"Empty or invalid audio file")
            )
            mock_proc.return_value = process

            with pytest.raises(RuntimeError, match="FFmpeg normalization failed"):
                await normalize_audio(audio_path)

    @pytest.mark.asyncio
    async def test_output_file_truncated(self, temp_dir):
        """Output file is truncated/incomplete."""
        audio_path = temp_dir / "audio_raw.wav"
        # Write incomplete WAV header
        audio_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        with patch("asyncio.create_subprocess_exec") as mock_proc:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(
                return_value=(b"", b"Invalid audio header")
            )
            mock_proc.return_value = process

            with pytest.raises(RuntimeError, match="FFmpeg normalization failed"):
                await normalize_audio(audio_path)

    @pytest.mark.asyncio
    async def test_output_folder_deleted(self, temp_dir, mock_session, mock_settings):
        """Output folder deleted during recording."""
        import callwhisper.services.recorder as recorder

        output_folder = temp_dir / mock_session.id
        output_folder.mkdir(parents=True)

        with patch.object(recorder, '_current_output_folder', output_folder):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.wait.return_value = 0

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                # Delete the folder
                import shutil
                shutil.rmtree(output_folder)

                # Raises FileNotFoundError when trying to write log
                with pytest.raises(FileNotFoundError):
                    await stop_recording()


# ============================================================================
# Whisper Transcription Failure Tests
# ============================================================================

class TestWhisperStartupFailures:
    """Tests for Whisper startup failures."""

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_whisper_not_found(self, temp_dir, mock_settings):
        """Whisper executable not found."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            # Create normalized file
            (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch(
                    "asyncio.create_subprocess_exec",
                    side_effect=FileNotFoundError("whisper not found")
                ):
                    with pytest.raises(TranscriptionError, match="not found"):
                        await transcribe_audio(temp_dir, mock_settings)

    @pytest.mark.asyncio
    async def test_whisper_model_not_found(self, temp_dir, mock_settings):
        """Whisper model file not found."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)

            with patch(
                "callwhisper.services.transcriber.get_whisper_path",
                return_value=Path("/path/to/whisper")
            ):
                with patch(
                    "callwhisper.services.transcriber.get_models_dir",
                    return_value=temp_dir / "models"
                ):
                    # Create empty models directory
                    (temp_dir / "models").mkdir()

                    with pytest.raises(FileNotFoundError, match="model"):
                        await transcribe_audio(temp_dir, mock_settings)


class TestWhisperRuntimeFailures:
    """Tests for Whisper failures during transcription."""

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_whisper_timeout(self, temp_dir, mock_settings):
        """Whisper transcription times out."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(
                        side_effect=asyncio.TimeoutError()
                    )
                    process.kill = MagicMock()
                    process.wait = AsyncMock()
                    mock_proc.return_value = process

                    with pytest.raises(ProcessTimeoutError):
                        await transcribe_audio(temp_dir, mock_settings)

                    # Process should be killed on timeout
                    process.kill.assert_called()

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_whisper_crashes(self, temp_dir, mock_settings):
        """Whisper process crashes during transcription."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = -11  # SIGSEGV
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(
                        side_effect=[b"Segmentation fault", b""]
                    )
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=-11)
                    mock_proc.return_value = process

                    with pytest.raises(TranscriptionError, match="failed"):
                        await transcribe_audio(temp_dir, mock_settings)

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_whisper_out_of_memory(self, temp_dir, mock_settings):
        """Whisper runs out of memory."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 1
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(
                        side_effect=[b"CUDA out of memory", b""]
                    )
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=1)
                    mock_proc.return_value = process

                    with pytest.raises(TranscriptionError, match="failed"):
                        await transcribe_audio(temp_dir, mock_settings)


class TestChunkTranscriptionFailures:
    """Tests for chunk transcription failures."""

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_chunk_transcription_timeout(self, temp_dir, mock_settings):
        """Single chunk transcription times out."""
        chunk_path = temp_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b'\x00' * 1024)

        with patch(
            "callwhisper.services.transcriber.get_audio_duration_seconds",
            return_value=5.0
        ):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.kill = MagicMock()
                process.wait = AsyncMock()
                mock_proc.return_value = process

                with patch(
                    "asyncio.wait_for",
                    side_effect=asyncio.TimeoutError()
                ):
                    with pytest.raises(ProcessTimeoutError, match="timed out"):
                        await transcribe_chunk(chunk_path, mock_settings)

                    process.kill.assert_called()

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_chunk_no_output_file(self, temp_dir, mock_settings):
        """Chunk transcription produces no output file."""
        chunk_path = temp_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b'\x00' * 1024)

        with patch(
            "callwhisper.services.transcriber.get_audio_duration_seconds",
            return_value=5.0
        ):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.returncode = 0
                process.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.return_value = process

                # No output file created
                result = await transcribe_chunk(chunk_path, mock_settings)

                # Should return empty string
                assert result == ""


# ============================================================================
# Signal Handling Tests
# ============================================================================

class TestSignalHandling:
    """Tests for process signal handling."""

    @pytest.mark.asyncio
    async def test_ffmpeg_receives_sigterm(self, temp_dir):
        """FFmpeg process receives SIGTERM."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
            # First wait times out, then succeeds after terminate
            mock_process.wait.side_effect = [
                subprocess.TimeoutExpired("cmd", 5),
                0
            ]
            mock_process.terminate = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                await stop_recording()

                mock_process.terminate.assert_called()

    @pytest.mark.asyncio
    async def test_ffmpeg_receives_sigkill(self, temp_dir):
        """FFmpeg process requires SIGKILL."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.stdin.write = MagicMock()
            mock_process.stdin.flush = MagicMock()
            # All waits time out
            mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
            mock_process.terminate = MagicMock()
            mock_process.kill = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                await stop_recording()

                mock_process.terminate.assert_called()
                mock_process.kill.assert_called()


# ============================================================================
# Cleanup After Failures Tests
# ============================================================================

class TestCleanupAfterFailures:
    """Tests for cleanup after failures."""

    @pytest.mark.asyncio
    async def test_cleanup_on_start_failure(
        self, mock_session, mock_settings, temp_dir, reset_recorder_state
    ):
        """Resources cleaned up when start fails."""
        import callwhisper.services.recorder as recorder

        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = 1  # Already exited
                process.stdin = MagicMock()
                mock_popen.return_value = process

                with pytest.raises(RecordingError):
                    await start_recording(mock_session, mock_settings)

                # Global state should be cleaned up
                with recorder._state_lock:
                    assert recorder._ffmpeg_process is None

    @pytest.mark.asyncio
    async def test_cleanup_on_stop_failure(self, temp_dir):
        """Resources cleaned up even when stop fails."""
        import callwhisper.services.recorder as recorder

        with patch.object(recorder, '_current_output_folder', temp_dir):
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_process.stdin = MagicMock()
            mock_process.stdin.write.side_effect = Exception("Write failed")
            mock_process.wait.side_effect = Exception("Wait failed")
            mock_process.terminate = MagicMock()

            with patch.object(recorder, '_ffmpeg_process', mock_process):
                await stop_recording()

                # Global state should still be cleaned up
                with recorder._state_lock:
                    assert recorder._ffmpeg_process is None

    @pytest.mark.asyncio
    async def test_is_recording_after_failure(
        self, mock_session, mock_settings, temp_dir, reset_recorder_state
    ):
        """is_recording returns False after failure."""
        import callwhisper.services.recorder as recorder

        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = 1
                process.stdin = MagicMock()
                mock_popen.return_value = process

                try:
                    await start_recording(mock_session, mock_settings)
                except RecordingError:
                    pass

                assert not is_recording()


# ============================================================================
# Process Orchestration Failure Tests
# ============================================================================

class TestProcessOrchestrationFailures:
    """Tests for failures in process orchestration."""

    @pytest.mark.asyncio
    async def test_normalization_failure_before_transcription(
        self, temp_dir, mock_settings
    ):
        """Normalization failure prevents transcription."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            side_effect=RuntimeError("Normalization failed")
        ):
            with pytest.raises(RuntimeError, match="Normalization failed"):
                await transcribe_audio(temp_dir, mock_settings)

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_duration_detection_failure(self, temp_dir, mock_settings):
        """Duration detection failure falls back to default timeout."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)
        (temp_dir / "audio_16k.txt").write_text("Test", encoding="utf-8")

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            return_value=temp_dir / "audio_16k.wav"
        ):
            # Duration detection fails
            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                side_effect=RuntimeError("ffprobe failed")
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 0
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(return_value=b"")
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=0)
                    mock_proc.return_value = process

                    # Should proceed with default timeout
                    result = await transcribe_audio(temp_dir, mock_settings)
                    assert result.exists()


# ============================================================================
# Helper Functions
# ============================================================================

def _create_test_audio(path: Path, duration: int = 1, sample_rate: int = 16000):
    """Create a test audio file."""
    with wave.open(str(path), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(sample_rate * duration):
            value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))


# ============================================================================
# Integration Tests
# ============================================================================

class TestFailureRecoveryIntegration:
    """Integration tests for failure recovery."""

    @pytest.mark.asyncio
    async def test_recording_failure_allows_new_recording(
        self, mock_session, mock_settings, temp_dir, reset_recorder_state
    ):
        """After failure, a new recording can be started."""
        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            # First recording fails
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = 1
                process.stdin = MagicMock()
                mock_popen.return_value = process

                with pytest.raises(RecordingError):
                    await start_recording(mock_session, mock_settings)

            # Second recording should be able to start
            mock_session.id = "20241229_120001_TEST456"
            with patch("subprocess.Popen") as mock_popen:
                process = MagicMock()
                process.poll.return_value = None  # Running
                process.stdin = MagicMock()
                process.pid = 12345
                mock_popen.return_value = process

                output = await start_recording(mock_session, mock_settings)
                assert output.exists()

    @pytest.mark.asyncio
    async def test_transcription_failure_preserves_audio(
        self, temp_dir, mock_settings
    ):
        """Transcription failure preserves audio file."""
        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        original_size = audio_path.stat().st_size

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            side_effect=RuntimeError("Normalization failed")
        ):
            with pytest.raises(RuntimeError):
                await transcribe_audio(temp_dir, mock_settings)

        # Audio file should still exist and be unchanged
        assert audio_path.exists()
        assert audio_path.stat().st_size == original_size
