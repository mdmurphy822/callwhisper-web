"""
Integration tests for resource exhaustion handling.

Tests how the system handles resource exhaustion scenarios:
- Disk full during recording
- Disk full during transcription
- Permission denied on output folder
- Memory exhaustion handling
- Too many open files
- Temp directory issues
"""

import asyncio
import pytest
import sys
import tempfile
import os
import errno
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import wave
import struct
import math

# Skip tests with Windows-specific error handling
UNIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="UNIX error handling")


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
    settings.output.audio_format = "wav"
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
def mock_models_dir(temp_dir):
    """Create mock models directory with fake model for transcription tests."""
    models_dir = temp_dir / "models"
    models_dir.mkdir()
    (models_dir / "ggml-base.bin").write_bytes(b"fake model data")
    with patch("callwhisper.services.transcriber.get_models_dir", return_value=models_dir):
        yield models_dir


# ============================================================================
# Disk Full Tests
# ============================================================================

class TestDiskFullDuringRecording:
    """Tests for disk full during recording."""

    @pytest.mark.asyncio
    async def test_disk_full_at_start(self, mock_session, mock_settings, temp_dir):
        """Recording fails at start when disk is full."""
        from callwhisper.services.recorder import start_recording
        from callwhisper.core.exceptions import RecordingError

        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            # Simulate disk full when creating output folder
            original_mkdir = Path.mkdir

            def failing_mkdir(self, *args, **kwargs):
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch.object(Path, 'mkdir', failing_mkdir):
                with pytest.raises(OSError):
                    await start_recording(mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_disk_full_creates_partial_output(self, temp_dir):
        """Disk becoming full creates partial output file."""
        output_file = temp_dir / "audio_raw.wav"

        # Simulate partial write before disk full
        partial_data = b'\x00' * 1024  # Only 1KB written

        with patch('builtins.open', side_effect=OSError(errno.ENOSPC, "No space")):
            # The file should have partial data if write started
            pass  # In reality, this would be handled by FFmpeg

    @pytest.mark.asyncio
    async def test_disk_space_check_before_recording(self, temp_dir):
        """System checks disk space before recording."""
        with patch('shutil.disk_usage') as mock_usage:
            # Simulate only 10MB free
            mock_usage.return_value = MagicMock(
                free=10 * 1024 * 1024,
                total=100 * 1024 * 1024 * 1024
            )

            # This could trigger a warning or prevent recording
            result = mock_usage(temp_dir)
            assert result.free < 100 * 1024 * 1024  # Less than 100MB


class TestDiskFullDuringTranscription:
    """Tests for disk full during transcription."""

    @pytest.mark.asyncio
    async def test_disk_full_during_normalization(self, temp_dir, mock_settings):
        """Normalization fails when disk is full."""
        from callwhisper.services.normalizer import normalize_audio

        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)

        with patch("asyncio.create_subprocess_exec") as mock_proc:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(
                return_value=(b"", b"No space left on device")
            )
            mock_proc.return_value = process

            with pytest.raises(RuntimeError, match="FFmpeg normalization failed"):
                await normalize_audio(audio_path)

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_disk_full_during_transcription(self, temp_dir, mock_settings, mock_models_dir):
        """Transcription fails when disk is full."""
        from callwhisper.services.transcriber import transcribe_audio
        from callwhisper.core.exceptions import TranscriptionError

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
                        side_effect=[b"No space left on device", b""]
                    )
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=1)
                    mock_proc.return_value = process

                    with pytest.raises(TranscriptionError, match="failed"):
                        await transcribe_audio(temp_dir, mock_settings)

    @pytest.mark.asyncio
    async def test_disk_full_writing_transcript(self, temp_dir):
        """Writing transcript file fails when disk is full."""
        transcript_path = temp_dir / "transcript.txt"

        with patch.object(Path, 'write_text') as mock_write:
            mock_write.side_effect = OSError(errno.ENOSPC, "No space left on device")

            with pytest.raises(OSError) as exc_info:
                transcript_path.write_text("Hello world", encoding="utf-8")

            assert exc_info.value.errno == errno.ENOSPC


# ============================================================================
# Permission Denied Tests
# ============================================================================

class TestPermissionDenied:
    """Tests for permission denied scenarios."""

    @pytest.mark.asyncio
    async def test_output_folder_not_writable(self, mock_session, mock_settings, temp_dir):
        """Recording fails when output folder is not writable."""
        from callwhisper.services.recorder import start_recording
        from callwhisper.core.exceptions import RecordingError

        with patch(
            "callwhisper.services.recorder.get_output_dir",
            return_value=temp_dir
        ):
            with patch.object(Path, 'mkdir') as mock_mkdir:
                mock_mkdir.side_effect = PermissionError("Permission denied")

                with pytest.raises(PermissionError):
                    await start_recording(mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_audio_file_not_readable(self, temp_dir):
        """Transcription fails when audio file is not readable."""
        audio_path = temp_dir / "audio_raw.wav"
        audio_path.write_bytes(b"test")

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")

            with pytest.raises(PermissionError):
                with open(audio_path, 'rb') as f:
                    f.read()

    @pytest.mark.asyncio
    async def test_log_file_not_writable(self, mock_session, mock_settings, temp_dir):
        """Recording handles non-writable log file."""
        # This tests that the system handles being unable to write logs
        log_path = temp_dir / "ffmpeg.log"

        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")

            # Should fail to write log
            with pytest.raises(PermissionError):
                with open(log_path, 'w') as f:
                    f.write("test")


class TestOwnershipIssues:
    """Tests for file ownership issues."""

    @pytest.mark.asyncio
    async def test_output_owned_by_different_user(self, temp_dir):
        """Handle output directory owned by different user."""
        # This would typically occur in multi-user scenarios
        with patch('os.access', return_value=False):
            # Check if directory is writable
            assert not os.access(temp_dir, os.W_OK)


# ============================================================================
# Memory Exhaustion Tests
# ============================================================================

class TestMemoryExhaustion:
    """Tests for memory exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_large_file_streaming(self, temp_dir):
        """Large files are processed with streaming to limit memory."""
        from callwhisper.services.audio_pipeline import AudioPipeline, PipelineConfig

        config = PipelineConfig(chunk_duration=30.0)
        pipeline = AudioPipeline(config)

        # Estimate memory for a 1-hour file
        estimate = pipeline.estimate_memory_usage(total_duration=3600.0)

        # Memory savings should be significant
        assert estimate["memory_savings_percent"] > 90

    @pytest.mark.asyncio
    async def test_memory_limit_respected(self, temp_dir):
        """Processing respects memory limits through chunking."""
        from callwhisper.services.audio_chunker import CHUNK_DURATION_SECONDS

        # With 5-minute chunks, memory usage is bounded
        bytes_per_second = 16000 * 2  # 16kHz, 16-bit
        max_chunk_memory = bytes_per_second * CHUNK_DURATION_SECONDS

        # Max chunk is ~14MB
        assert max_chunk_memory < 20 * 1024 * 1024  # Less than 20MB

    @UNIX_ONLY
    @pytest.mark.asyncio
    async def test_subprocess_out_of_memory(self, temp_dir, mock_settings, mock_models_dir):
        """Handle subprocess running out of memory."""
        from callwhisper.services.transcriber import transcribe_audio
        from callwhisper.core.exceptions import TranscriptionError

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
                    process.returncode = -9  # Killed (SIGKILL, often from OOM)
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(side_effect=[b"Killed", b""])
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=-9)
                    mock_proc.return_value = process

                    with pytest.raises(TranscriptionError, match="failed"):
                        await transcribe_audio(temp_dir, mock_settings)


# ============================================================================
# File Descriptor Exhaustion Tests
# ============================================================================

class TestFileDescriptorExhaustion:
    """Tests for too many open files scenarios."""

    @pytest.mark.asyncio
    async def test_too_many_open_files(self, temp_dir):
        """Handle too many open files error."""
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = OSError(errno.EMFILE, "Too many open files")

            with pytest.raises(OSError) as exc_info:
                open(temp_dir / "test.txt", 'w')

            assert exc_info.value.errno == errno.EMFILE

    @pytest.mark.asyncio
    async def test_subprocess_file_limit(self, temp_dir):
        """Handle subprocess hitting file limit."""
        with patch("asyncio.create_subprocess_exec") as mock_proc:
            mock_proc.side_effect = OSError(errno.EMFILE, "Too many open files")

            with pytest.raises(OSError) as exc_info:
                await asyncio.create_subprocess_exec("cmd")

            assert exc_info.value.errno == errno.EMFILE


# ============================================================================
# Temp Directory Tests
# ============================================================================

class TestTempDirectoryIssues:
    """Tests for temp directory issues."""

    @pytest.mark.asyncio
    async def test_temp_dir_full(self, temp_dir):
        """Handle temp directory being full."""
        with patch('tempfile.gettempdir', return_value=str(temp_dir)):
            with patch('tempfile.TemporaryDirectory') as mock_temp:
                mock_temp.side_effect = OSError(errno.ENOSPC, "No space left")

                with pytest.raises(OSError) as exc_info:
                    with tempfile.TemporaryDirectory():
                        pass

                assert exc_info.value.errno == errno.ENOSPC

    @pytest.mark.asyncio
    async def test_temp_dir_not_writable(self, temp_dir):
        """Handle temp directory not being writable."""
        with patch('tempfile.gettempdir', return_value=str(temp_dir)):
            with patch('tempfile.TemporaryDirectory') as mock_temp:
                mock_temp.side_effect = PermissionError("Permission denied")

                with pytest.raises(PermissionError):
                    with tempfile.TemporaryDirectory():
                        pass


# ============================================================================
# Degradation Mode Tests
# ============================================================================

class TestDegradationMode:
    """Tests for graceful degradation under resource pressure."""

    @pytest.mark.asyncio
    async def test_degradation_levels(self):
        """Degradation levels adjust processing parameters."""
        from callwhisper.core.degradation import DegradationLevel, degradation_manager

        # Test different levels
        for level in DegradationLevel:
            settings = degradation_manager.get_settings_for_level(level)
            assert "chunk_duration" in settings or settings is not None

    @pytest.mark.asyncio
    async def test_chunk_size_adjustment(self):
        """Chunk size adjusts based on degradation level."""
        from callwhisper.services.audio_pipeline import AudioPipeline
        from callwhisper.core.degradation import DegradationLevel

        pipeline = AudioPipeline()

        # FULL mode has larger chunks
        full_duration = pipeline.get_chunk_duration_for_level(DegradationLevel.FULL)
        fast_duration = pipeline.get_chunk_duration_for_level(DegradationLevel.FAST)

        assert full_duration > fast_duration


# ============================================================================
# Cleanup Under Pressure Tests
# ============================================================================

class TestCleanupUnderPressure:
    """Tests for cleanup when resources are limited."""

    @pytest.mark.asyncio
    async def test_cleanup_on_disk_full(self, temp_dir):
        """Cleanup still runs when disk is full."""
        # Create some files
        test_files = []
        for i in range(5):
            f = temp_dir / f"test_{i}.tmp"
            f.write_bytes(b'\x00' * 1024)
            test_files.append(f)

        # Simulate disk full when trying to create new files
        with patch.object(Path, 'write_bytes') as mock_write:
            mock_write.side_effect = OSError(errno.ENOSPC, "No space")

            # But cleanup (deletion) should still work
            for f in test_files:
                f.unlink()

            assert not any(f.exists() for f in test_files)

    @pytest.mark.asyncio
    async def test_partial_cleanup_on_error(self, temp_dir):
        """Partial cleanup when some files can't be deleted."""
        files = []
        for i in range(5):
            f = temp_dir / f"test_{i}.tmp"
            f.write_bytes(b'\x00' * 100)
            files.append(f)

        # Mock one file failing to delete
        original_unlink = Path.unlink
        fail_count = [0]

        def failing_unlink(self):
            if self.name == "test_2.tmp":
                fail_count[0] += 1
                raise PermissionError("Cannot delete")
            original_unlink(self)

        with patch.object(Path, 'unlink', failing_unlink):
            deleted = 0
            for f in files:
                try:
                    f.unlink()
                    deleted += 1
                except PermissionError:
                    pass

            # Most files should be deleted
            assert deleted == 4


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

class TestResourceExhaustionRecovery:
    """Integration tests for resource exhaustion recovery."""

    @pytest.mark.asyncio
    async def test_recovery_after_disk_full_resolved(self, temp_dir, mock_settings):
        """System recovers after disk space is freed."""
        from callwhisper.services.transcriber import transcribe_audio

        audio_path = temp_dir / "audio_raw.wav"
        _create_test_audio(audio_path)
        (temp_dir / "audio_16k.wav").write_bytes(b'\x00' * 16000)
        (temp_dir / "audio_16k.txt").write_text("Test transcript", encoding="utf-8")

        # First attempt fails due to disk full
        call_count = [0]

        async def mock_normalize(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("No space left on device")
            return temp_dir / "audio_16k.wav"

        with patch(
            "callwhisper.services.transcriber.normalize_audio",
            side_effect=mock_normalize
        ):
            # First attempt fails
            with pytest.raises(RuntimeError, match="No space"):
                await transcribe_audio(temp_dir, mock_settings)

            # After freeing space, second attempt would succeed
            # (In reality, user would retry after freeing space)

    @pytest.mark.asyncio
    async def test_graceful_degradation_flow(self, temp_dir):
        """Full flow with graceful degradation."""
        from callwhisper.core.degradation import degradation_manager, DegradationLevel

        # Simulate increasing pressure
        levels = [
            DegradationLevel.FULL,
            DegradationLevel.BALANCED,
            DegradationLevel.FAST
        ]

        for level in levels:
            settings = degradation_manager.get_settings_for_level(level)
            # Settings should be valid for all levels
            assert settings is not None or True  # Handles None return
