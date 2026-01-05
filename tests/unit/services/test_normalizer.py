"""
Tests for audio normalization service.

Tests audio conversion to whisper-compatible format:
- 16kHz sample rate conversion
- Mono channel conversion
- Opus format conversion
- Duration detection
"""

import pytest
import asyncio
import wave
import struct
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

from callwhisper.services.normalizer import (
    normalize_audio,
    convert_to_opus,
    get_audio_duration,
    NORMALIZATION_TIMEOUT_SECONDS,
)
from callwhisper.core.exceptions import ProcessTimeoutError


class TestNormalizeAudio:
    """Tests for audio normalization."""

    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create sample WAV file for testing."""
        audio_path = tmp_path / "input.wav"
        sample_rate = 44100
        duration = 1

        with wave.open(str(audio_path), 'w') as wav:
            wav.setnchannels(2)  # Stereo
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(sample_rate * duration * 2):
                value = int(32767 * 0.5)
                wav.writeframes(struct.pack('<h', value))

        return audio_path

    @pytest.mark.asyncio
    async def test_normalize_creates_output_file(self, sample_wav):
        """Verify normalization creates output file when successful."""
        output_path = sample_wav.parent / "audio_16k.wav"

        # Mock successful FFmpeg process
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = process

            # Simulate file creation
            with patch.object(Path, 'exists', return_value=True):
                result = await normalize_audio(sample_wav, output_path)

            assert result == output_path

    @pytest.mark.asyncio
    async def test_normalize_raises_for_missing_input(self, tmp_path):
        """Verify FileNotFoundError for missing input file."""
        missing_path = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError):
            await normalize_audio(missing_path)

    @pytest.mark.asyncio
    async def test_normalize_uses_default_output_path(self, sample_wav):
        """Verify default output path is used when not specified."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = process

            with patch.object(Path, 'exists', return_value=True):
                result = await normalize_audio(sample_wav)

            expected_output = sample_wav.parent / "audio_16k.wav"
            assert result == expected_output

    @pytest.mark.asyncio
    async def test_normalize_handles_timeout(self, sample_wav):
        """Verify timeout error is raised when process takes too long."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.kill = MagicMock()
            process.wait = AsyncMock()

            # Simulate timeout
            async def timeout_communicate():
                raise asyncio.TimeoutError()

            process.communicate = timeout_communicate
            mock_exec.return_value = process

            with pytest.raises(ProcessTimeoutError):
                await normalize_audio(sample_wav)

    @pytest.mark.asyncio
    async def test_normalize_handles_ffmpeg_error(self, sample_wav):
        """Verify RuntimeError for FFmpeg failures."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(
                return_value=(b"", b"FFmpeg error message")
            )
            mock_exec.return_value = process

            with pytest.raises(RuntimeError, match="FFmpeg normalization failed"):
                await normalize_audio(sample_wav)


class TestConvertToOpus:
    """Tests for Opus conversion."""

    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create sample WAV file for testing."""
        audio_path = tmp_path / "input.wav"
        sample_rate = 16000
        duration = 1

        with wave.open(str(audio_path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(sample_rate * duration):
                value = int(32767 * 0.5)
                wav.writeframes(struct.pack('<h', value))

        return audio_path

    @pytest.mark.asyncio
    async def test_convert_creates_opus_file(self, sample_wav):
        """Verify Opus conversion creates output file."""
        output_path = sample_wav.parent / "recording.opus"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = process

            result = await convert_to_opus(sample_wav, output_path)

            assert result == output_path

    @pytest.mark.asyncio
    async def test_convert_raises_for_missing_input(self, tmp_path):
        """Verify FileNotFoundError for missing input file."""
        missing_path = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError):
            await convert_to_opus(missing_path)

    @pytest.mark.asyncio
    async def test_convert_uses_default_output_path(self, sample_wav):
        """Verify default output path is used when not specified."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = process

            result = await convert_to_opus(sample_wav)

            expected_output = sample_wav.parent / "recording.opus"
            assert result == expected_output


class TestGetAudioDuration:
    """Tests for audio duration detection."""

    def test_returns_duration_for_wav(self, tmp_path):
        """Verify duration is calculated correctly for WAV files."""
        audio_path = tmp_path / "test.wav"
        sample_rate = 16000
        duration_seconds = 2.5
        num_frames = int(sample_rate * duration_seconds)

        with wave.open(str(audio_path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(num_frames):
                wav.writeframes(struct.pack('<h', 0))

        result = get_audio_duration(audio_path)

        assert abs(result - duration_seconds) < 0.01

    def test_returns_zero_for_invalid_file(self, tmp_path):
        """Verify 0.0 returned for invalid files."""
        invalid_path = tmp_path / "invalid.wav"
        invalid_path.write_bytes(b"not a wav file")

        result = get_audio_duration(invalid_path)

        assert result == 0.0

    def test_returns_zero_for_non_wav(self, tmp_path):
        """Verify 0.0 returned for non-WAV files."""
        other_path = tmp_path / "audio.mp3"
        other_path.write_bytes(b"fake mp3 content")

        result = get_audio_duration(other_path)

        assert result == 0.0
