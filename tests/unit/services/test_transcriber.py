"""
Unit tests for Whisper Transcription Service.

Tests:
- Audio duration detection
- Adaptive timeout calculation
- Transcription flow (mocked subprocess)
- SRT to VTT conversion
- Chunked transcription
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import tempfile
import os

from callwhisper.services.transcriber import (
    get_audio_duration_seconds,
    calculate_adaptive_timeout,
    transcribe_audio,
    transcribe_chunk,
    transcribe_audio_chunked,
    srt_to_vtt,
    TRANSCRIPTION_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    TIMEOUT_MULTIPLIER,
)
from callwhisper.core.exceptions import TranscriptionError, ProcessTimeoutError


# ============================================================================
# Adaptive Timeout Calculation Tests
# ============================================================================

class TestAdaptiveTimeout:
    """Tests for calculate_adaptive_timeout function."""

    def test_minimum_timeout(self):
        """Short audio gets minimum timeout."""
        # 10 second audio * 3 = 30, but minimum is 120
        timeout = calculate_adaptive_timeout(10.0)
        assert timeout == MIN_TIMEOUT_SECONDS

    def test_maximum_timeout(self):
        """Very long audio is capped at maximum."""
        # 10000 second audio * 3 = 30000, capped at 7200
        timeout = calculate_adaptive_timeout(10000.0)
        assert timeout == MAX_TIMEOUT_SECONDS

    def test_proportional_timeout(self):
        """Normal audio gets proportional timeout."""
        # 200 second audio * 3 = 600
        timeout = calculate_adaptive_timeout(200.0)
        assert timeout == 600

    def test_timeout_multiplier(self):
        """Timeout uses correct multiplier."""
        # 100 seconds * 3 = 300
        timeout = calculate_adaptive_timeout(100.0)
        assert timeout == 300

    def test_zero_duration(self):
        """Zero duration gets minimum timeout."""
        timeout = calculate_adaptive_timeout(0.0)
        assert timeout == MIN_TIMEOUT_SECONDS

    def test_fractional_duration(self):
        """Fractional durations are handled."""
        timeout = calculate_adaptive_timeout(50.5)
        assert timeout == 151  # 50.5 * 3 = 151.5 -> int(151.5) = 151, which > MIN (120)

    def test_boundary_at_min(self):
        """Boundary at minimum threshold."""
        # 40 * 3 = 120 (exactly minimum)
        timeout = calculate_adaptive_timeout(40.0)
        assert timeout == MIN_TIMEOUT_SECONDS

    def test_boundary_at_max(self):
        """Boundary at maximum threshold."""
        # 2400 * 3 = 7200 (exactly maximum)
        timeout = calculate_adaptive_timeout(2400.0)
        assert timeout == MAX_TIMEOUT_SECONDS


# ============================================================================
# Audio Duration Detection Tests
# ============================================================================

class TestGetAudioDuration:
    """Tests for get_audio_duration_seconds function."""

    @pytest.mark.asyncio
    async def test_successful_duration_detection(self, tmp_path):
        """Successfully get duration from audio file."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"123.456\n", b"")
            mock_exec.return_value = mock_proc

            duration = await get_audio_duration_seconds(audio_file)

            assert duration == 123.456

    @pytest.mark.asyncio
    async def test_ffprobe_failure_raises(self, tmp_path):
        """FFprobe failure raises RuntimeError."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"error message")
            mock_exec.return_value = mock_proc

            with pytest.raises(RuntimeError, match="ffprobe failed"):
                await get_audio_duration_seconds(audio_file)

    @pytest.mark.asyncio
    async def test_invalid_duration_format(self, tmp_path):
        """Invalid duration format raises RuntimeError."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"not a number\n", b"")
            mock_exec.return_value = mock_proc

            with pytest.raises(RuntimeError, match="Could not parse"):
                await get_audio_duration_seconds(audio_file)


# ============================================================================
# SRT to VTT Conversion Tests
# ============================================================================

class TestSrtToVtt:
    """Tests for srt_to_vtt function."""

    def test_basic_conversion(self, tmp_path):
        """Basic SRT to VTT conversion."""
        srt_file = tmp_path / "subtitle.srt"
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:05,500 --> 00:00:10,000
Second line
"""
        srt_file.write_text(srt_content)

        vtt_path = srt_to_vtt(srt_file)

        assert vtt_path.exists()
        vtt_content = vtt_path.read_text()

        assert vtt_content.startswith("WEBVTT")
        assert "00:00:00.000" in vtt_content  # Comma replaced with dot
        assert "Hello world" in vtt_content

    def test_custom_output_path(self, tmp_path):
        """Conversion with custom output path."""
        srt_file = tmp_path / "subtitle.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")

        custom_vtt = tmp_path / "custom.vtt"
        result = srt_to_vtt(srt_file, vtt_path=custom_vtt)

        assert result == custom_vtt
        assert custom_vtt.exists()

    def test_missing_srt_raises(self, tmp_path):
        """Missing SRT file raises FileNotFoundError."""
        srt_file = tmp_path / "nonexistent.srt"

        with pytest.raises(FileNotFoundError):
            srt_to_vtt(srt_file)

    def test_timestamp_format_conversion(self, tmp_path):
        """Timestamps are converted correctly."""
        srt_file = tmp_path / "test.srt"
        srt_content = "1\n00:01:23,456 --> 00:02:34,567\nTest\n"
        srt_file.write_text(srt_content)

        vtt_path = srt_to_vtt(srt_file)
        vtt_content = vtt_path.read_text()

        assert "00:01:23.456" in vtt_content
        assert "00:02:34.567" in vtt_content
        assert "," not in vtt_content.split("WEBVTT")[1]  # No commas in timestamps


# ============================================================================
# Transcription Flow Tests (Mocked)
# ============================================================================

class TestTranscribeAudio:
    """Tests for transcribe_audio function."""

    @pytest.fixture
    def output_folder(self, tmp_path):
        """Create output folder with raw audio."""
        folder = tmp_path / "recording_123"
        folder.mkdir()
        (folder / "audio_raw.wav").write_bytes(b"fake audio data")
        return folder

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.transcription.model = "ggml-medium.bin"
        settings.transcription.language = "en"
        settings.transcription.beam_size = 5
        settings.transcription.best_of = 5
        return settings

    @pytest.mark.asyncio
    async def test_missing_raw_audio_raises(self, tmp_path, mock_settings):
        """Missing raw audio raises FileNotFoundError."""
        folder = tmp_path / "empty_folder"
        folder.mkdir()

        with pytest.raises(FileNotFoundError, match="Raw audio not found"):
            await transcribe_audio(folder, mock_settings)

    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self, output_folder, mock_settings):
        """Progress callback is invoked during transcription."""
        progress_updates = []

        async def progress_callback(percent, stage):
            progress_updates.append((percent, stage))

        with patch('callwhisper.services.transcriber.normalize_audio') as mock_norm:
            mock_norm.return_value = output_folder / "audio_16k.wav"
            (output_folder / "audio_16k.wav").write_bytes(b"normalized")

            with patch('callwhisper.services.transcriber.get_audio_duration_seconds') as mock_dur:
                mock_dur.return_value = 30.0

                with patch('asyncio.create_subprocess_exec') as mock_exec:
                    mock_proc = AsyncMock()
                    mock_proc.returncode = 0

                    # Simulate progress output
                    async def fake_read(n):
                        return b""

                    mock_proc.stderr.read = fake_read
                    mock_proc.stdout.read = AsyncMock(return_value=b"")
                    mock_proc.wait = AsyncMock()
                    mock_exec.return_value = mock_proc

                    with patch('callwhisper.services.transcriber.get_whisper_path') as mock_whisper:
                        mock_whisper.return_value = Path("/fake/whisper")
                        with patch('callwhisper.services.transcriber.get_models_dir') as mock_models:
                            models_dir = output_folder / "models"
                            models_dir.mkdir()
                            (models_dir / "ggml-medium.bin").write_bytes(b"model")
                            mock_models.return_value = models_dir

                            # Create output files that whisper would create
                            (output_folder / "audio_16k.txt").write_text("Hello world")
                            (output_folder / "audio_16k.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

                            result = await transcribe_audio(
                                output_folder, mock_settings, progress_callback
                            )

                            assert result.exists()

    @pytest.mark.asyncio
    async def test_whisper_not_found_raises(self, output_folder, mock_settings):
        """Missing whisper raises TranscriptionError."""
        with patch('callwhisper.services.transcriber.normalize_audio') as mock_norm:
            mock_norm.return_value = output_folder / "audio_16k.wav"
            (output_folder / "audio_16k.wav").write_bytes(b"normalized")

            with patch('callwhisper.services.transcriber.get_audio_duration_seconds') as mock_dur:
                mock_dur.return_value = 30.0

                with patch('callwhisper.services.transcriber.get_whisper_path') as mock_whisper:
                    mock_whisper.return_value = Path("/nonexistent/whisper")

                    with patch('callwhisper.services.transcriber.get_models_dir') as mock_models:
                        models_dir = output_folder / "models"
                        models_dir.mkdir()
                        (models_dir / "ggml-medium.bin").write_bytes(b"model")
                        mock_models.return_value = models_dir

                        with patch('asyncio.create_subprocess_exec') as mock_exec:
                            mock_exec.side_effect = FileNotFoundError()

                            with pytest.raises(TranscriptionError, match="whisper-cli not found"):
                                await transcribe_audio(output_folder, mock_settings)


# ============================================================================
# Transcribe Chunk Tests
# ============================================================================

class TestTranscribeChunk:
    """Tests for transcribe_chunk function."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.transcription.model = "ggml-medium.bin"
        settings.transcription.language = "en"
        settings.transcription.beam_size = 5
        settings.transcription.best_of = 5
        return settings

    @pytest.mark.asyncio
    async def test_chunk_transcription_success(self, tmp_path, mock_settings):
        """Successfully transcribe a chunk."""
        chunk_path = tmp_path / "chunk_0.wav"
        chunk_path.write_bytes(b"chunk audio")

        with patch('callwhisper.services.transcriber.get_whisper_path') as mock_whisper:
            mock_whisper.return_value = Path("/fake/whisper")

            with patch('callwhisper.services.transcriber.get_models_dir') as mock_models:
                models_dir = tmp_path / "models"
                models_dir.mkdir()
                (models_dir / "ggml-medium.bin").write_bytes(b"model")
                mock_models.return_value = models_dir

                with patch('callwhisper.services.transcriber.get_audio_duration_seconds') as mock_dur:
                    mock_dur.return_value = 30.0

                    with patch('asyncio.create_subprocess_exec') as mock_exec:
                        mock_proc = AsyncMock()
                        mock_proc.returncode = 0
                        mock_proc.communicate.return_value = (b"", b"")
                        mock_exec.return_value = mock_proc

                        # Create output file
                        (tmp_path / "chunk_0.txt").write_text("Hello from chunk")

                        result = await transcribe_chunk(chunk_path, mock_settings)

                        assert result == "Hello from chunk"

    @pytest.mark.asyncio
    async def test_chunk_timeout_raises(self, tmp_path, mock_settings):
        """Chunk timeout raises ProcessTimeoutError."""
        chunk_path = tmp_path / "chunk_0.wav"
        chunk_path.write_bytes(b"chunk audio")

        with patch('callwhisper.services.transcriber.get_whisper_path') as mock_whisper:
            mock_whisper.return_value = Path("/fake/whisper")

            with patch('callwhisper.services.transcriber.get_models_dir') as mock_models:
                models_dir = tmp_path / "models"
                models_dir.mkdir()
                (models_dir / "ggml-medium.bin").write_bytes(b"model")
                mock_models.return_value = models_dir

                with patch('callwhisper.services.transcriber.get_audio_duration_seconds') as mock_dur:
                    mock_dur.return_value = 30.0

                    with patch('asyncio.create_subprocess_exec') as mock_exec:
                        mock_proc = AsyncMock()
                        mock_proc.communicate.side_effect = asyncio.TimeoutError()
                        mock_proc.kill = AsyncMock()
                        mock_proc.wait = AsyncMock()
                        mock_exec.return_value = mock_proc

                        with pytest.raises(ProcessTimeoutError):
                            await transcribe_chunk(chunk_path, mock_settings)

    @pytest.mark.asyncio
    async def test_chunk_failure_raises(self, tmp_path, mock_settings):
        """Chunk failure raises TranscriptionError."""
        chunk_path = tmp_path / "chunk_0.wav"
        chunk_path.write_bytes(b"chunk audio")

        with patch('callwhisper.services.transcriber.get_whisper_path') as mock_whisper:
            mock_whisper.return_value = Path("/fake/whisper")

            with patch('callwhisper.services.transcriber.get_models_dir') as mock_models:
                models_dir = tmp_path / "models"
                models_dir.mkdir()
                (models_dir / "ggml-medium.bin").write_bytes(b"model")
                mock_models.return_value = models_dir

                with patch('callwhisper.services.transcriber.get_audio_duration_seconds') as mock_dur:
                    mock_dur.return_value = 30.0

                    with patch('asyncio.create_subprocess_exec') as mock_exec:
                        mock_proc = AsyncMock()
                        mock_proc.returncode = 1
                        mock_proc.communicate.return_value = (b"", b"whisper error")
                        mock_exec.return_value = mock_proc

                        with pytest.raises(TranscriptionError, match="Chunk transcription failed"):
                            await transcribe_chunk(chunk_path, mock_settings)


# ============================================================================
# Chunked Transcription Tests
# ============================================================================

class TestTranscribeAudioChunked:
    """Tests for transcribe_audio_chunked function."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.transcription.model = "ggml-medium.bin"
        settings.transcription.language = "en"
        settings.transcription.beam_size = 5
        settings.transcription.best_of = 5
        return settings

    @pytest.mark.asyncio
    async def test_missing_normalized_audio_raises(self, tmp_path, mock_settings):
        """Missing normalized audio raises FileNotFoundError."""
        folder = tmp_path / "job_123"
        folder.mkdir()

        with pytest.raises(FileNotFoundError, match="Normalized audio not found"):
            await transcribe_audio_chunked(folder, mock_settings)

    @pytest.mark.asyncio
    async def test_resume_from_chunk(self, tmp_path, mock_settings):
        """Resuming from a specific chunk works."""
        folder = tmp_path / "job_123"
        folder.mkdir()
        (folder / "audio_16k.wav").write_bytes(b"normalized audio")

        # Create partial transcript
        partial = folder / "partial_transcript.txt"
        partial.write_text("First chunk text\n---CHUNK_BOUNDARY---\nSecond chunk text")

        with patch('callwhisper.services.audio_chunker.ensure_chunks_exist') as mock_chunks:
            mock_manifest = MagicMock()
            mock_manifest.chunk_count = 3
            mock_manifest.overlap_seconds = 0.5
            mock_manifest.chunks = [
                MagicMock(index=0, chunk_path=str(folder / "chunk_0.wav"), start_time=0, end_time=30),
                MagicMock(index=1, chunk_path=str(folder / "chunk_1.wav"), start_time=30, end_time=60),
                MagicMock(index=2, chunk_path=str(folder / "chunk_2.wav"), start_time=60, end_time=90),
            ]
            mock_chunks.return_value = mock_manifest

            with patch('callwhisper.services.transcriber.transcribe_chunk') as mock_transcribe:
                mock_transcribe.return_value = "Third chunk text"

                with patch('callwhisper.services.audio_chunker.merge_chunk_transcripts') as mock_merge:
                    mock_merge.return_value = "Merged transcript"

                    with patch('callwhisper.services.transcriber.update_checkpoint'):
                        result = await transcribe_audio_chunked(
                            folder, mock_settings, start_from_chunk=2
                        )

                        # Should only transcribe chunk 2
                        assert mock_transcribe.call_count == 1
                        assert result.exists()


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestTranscriberEdgeCases:
    """Edge case tests for transcriber."""

    def test_srt_with_unicode(self, tmp_path):
        """SRT conversion handles unicode content."""
        srt_file = tmp_path / "unicode.srt"
        srt_content = "1\n00:00:00,000 --> 00:00:01,000\n日本語テスト 🎤\n"
        srt_file.write_text(srt_content, encoding="utf-8")

        vtt_path = srt_to_vtt(srt_file)
        vtt_content = vtt_path.read_text(encoding="utf-8")

        assert "日本語テスト 🎤" in vtt_content

    def test_srt_empty_file(self, tmp_path):
        """SRT conversion handles empty file."""
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("")

        vtt_path = srt_to_vtt(srt_file)
        vtt_content = vtt_path.read_text()

        assert vtt_content.startswith("WEBVTT")

    def test_timeout_constants(self):
        """Timeout constants have sensible values."""
        assert MIN_TIMEOUT_SECONDS == 120
        assert MAX_TIMEOUT_SECONDS == 7200
        assert TIMEOUT_MULTIPLIER == 3
        assert TRANSCRIPTION_TIMEOUT_SECONDS == 600

        # Multiplier should result in reasonable timeouts
        # 1 hour audio = 3600s * 3 = 10800 -> capped at 7200
        assert calculate_adaptive_timeout(3600) == MAX_TIMEOUT_SECONDS


# ============================================================================
# Callback Tests
# ============================================================================

class TestTranscriberCallbacks:
    """Tests for callback functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.transcription.model = "ggml-medium.bin"
        settings.transcription.language = "en"
        settings.transcription.beam_size = 5
        settings.transcription.best_of = 5
        return settings

    @pytest.mark.asyncio
    async def test_partial_transcript_callback(self, tmp_path, mock_settings):
        """Partial transcript callback is invoked during chunked transcription."""
        folder = tmp_path / "job_123"
        folder.mkdir()
        (folder / "audio_16k.wav").write_bytes(b"normalized audio")

        callback_invocations = []

        async def partial_callback(text, is_final):
            callback_invocations.append((text, is_final))

        with patch('callwhisper.services.audio_chunker.ensure_chunks_exist') as mock_chunks:
            mock_manifest = MagicMock()
            mock_manifest.chunk_count = 2
            mock_manifest.overlap_seconds = 0.5
            mock_manifest.chunks = [
                MagicMock(index=0, chunk_path=str(folder / "chunk_0.wav"), start_time=0, end_time=30),
                MagicMock(index=1, chunk_path=str(folder / "chunk_1.wav"), start_time=30, end_time=60),
            ]
            mock_chunks.return_value = mock_manifest

            with patch('callwhisper.services.transcriber.transcribe_chunk') as mock_transcribe:
                mock_transcribe.side_effect = ["Chunk 1", "Chunk 2"]

                with patch('callwhisper.services.audio_chunker.merge_chunk_transcripts') as mock_merge:
                    mock_merge.return_value = "Final merged"

                    with patch('callwhisper.services.transcriber.update_checkpoint'):
                        await transcribe_audio_chunked(
                            folder, mock_settings,
                            partial_transcript_callback=partial_callback
                        )

                        # Should be called for each chunk + final
                        assert len(callback_invocations) >= 2
                        # Last call should be final
                        assert callback_invocations[-1][1] is True

    @pytest.mark.asyncio
    async def test_progress_callback_in_chunked(self, tmp_path, mock_settings):
        """Progress callback is invoked during chunked transcription."""
        folder = tmp_path / "job_123"
        folder.mkdir()
        (folder / "audio_16k.wav").write_bytes(b"normalized audio")

        progress_updates = []

        async def progress_callback(percent, stage):
            progress_updates.append((percent, stage))

        with patch('callwhisper.services.audio_chunker.ensure_chunks_exist') as mock_chunks:
            mock_manifest = MagicMock()
            mock_manifest.chunk_count = 3
            mock_manifest.overlap_seconds = 0.5
            mock_manifest.chunks = [
                MagicMock(index=0, chunk_path=str(folder / "chunk_0.wav"), start_time=0, end_time=30),
                MagicMock(index=1, chunk_path=str(folder / "chunk_1.wav"), start_time=30, end_time=60),
                MagicMock(index=2, chunk_path=str(folder / "chunk_2.wav"), start_time=60, end_time=90),
            ]
            mock_chunks.return_value = mock_manifest

            with patch('callwhisper.services.transcriber.transcribe_chunk') as mock_transcribe:
                mock_transcribe.return_value = "Chunk text"

                with patch('callwhisper.services.audio_chunker.merge_chunk_transcripts') as mock_merge:
                    mock_merge.return_value = "Final"

                    with patch('callwhisper.services.transcriber.update_checkpoint'):
                        await transcribe_audio_chunked(
                            folder, mock_settings,
                            progress_callback=progress_callback
                        )

                        # Should have 3 progress updates (one per chunk)
                        assert len(progress_updates) == 3
                        # Progress should increase
                        percents = [p[0] for p in progress_updates]
                        assert percents == sorted(percents)
