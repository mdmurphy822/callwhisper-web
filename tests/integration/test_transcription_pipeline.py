"""
Integration tests for transcription pipeline.

Tests the full transcription flow including:
- Audio normalization (WAV to 16kHz mono)
- Whisper transcription subprocess
- Progress callback updates
- Chunked transcription with checkpoints
- SRT to VTT conversion
- Transcript merging from chunks
- Resume from checkpoint
- Partial transcript callbacks
"""

import asyncio
import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import tempfile
import wave
import struct
import math

# Import transcription modules
from callwhisper.services.transcriber import (
    transcribe_audio,
    transcribe_chunk,
    transcribe_audio_chunked,
    get_audio_duration_seconds,
    calculate_adaptive_timeout,
    srt_to_vtt,
    MIN_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    TIMEOUT_MULTIPLIER,
)
from callwhisper.services.normalizer import (
    normalize_audio,
    convert_to_opus,
    get_audio_duration as get_wav_duration,
)
from callwhisper.services.audio_chunker import (
    split_audio_to_chunks,
    merge_chunk_transcripts,
    ensure_chunks_exist,
    ChunkManifest,
    AudioChunk,
    CHUNK_DURATION_SECONDS,
    CHUNK_OVERLAP_SECONDS,
)
from callwhisper.services.srt_merger import (
    parse_srt,
    write_srt,
    merge_srt_segments,
    SrtSegment,
)
from callwhisper.services.audio_pipeline import (
    AudioPipeline,
    AudioChunk as PipelineChunk,
    AudioMetadata,
    PipelineConfig,
    stream_audio_chunks,
    get_audio_duration,
)
from callwhisper.core.exceptions import TranscriptionError, ProcessTimeoutError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_wav_16k(temp_output_dir):
    """Create a sample 16kHz mono WAV file."""
    audio_path = temp_output_dir / "audio_16k.wav"
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
def sample_raw_audio(temp_output_dir):
    """Create a sample raw audio file for transcription."""
    audio_path = temp_output_dir / "audio_raw.wav"
    sample_rate = 44100  # Higher rate to test normalization
    duration = 2  # seconds
    frequency = 440  # Hz

    with wave.open(str(audio_path), 'w') as wav:
        wav.setnchannels(2)  # Stereo to test mono conversion
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(sample_rate * duration):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            # Write stereo (same value to both channels)
            wav.writeframes(struct.pack('<hh', value, value))

    return audio_path


@pytest.fixture
def sample_srt_file(temp_output_dir):
    """Create a sample SRT file for testing."""
    srt_path = temp_output_dir / "transcript.srt"
    srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
This is a test.

3
00:00:05,000 --> 00:00:07,500
Testing transcription.
"""
    srt_path.write_text(srt_content, encoding='utf-8')
    return srt_path


@pytest.fixture
def mock_settings():
    """Create mock settings for transcription."""
    settings = MagicMock()
    settings.transcription.model = "ggml-base.bin"
    settings.transcription.language = "en"
    settings.transcription.beam_size = 5
    settings.transcription.best_of = 5
    return settings


@pytest.fixture
def mock_whisper_success():
    """Mock successful whisper subprocess."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.returncode = 0
        # Simulate progress output
        process.stderr = AsyncMock()
        process.stderr.read = AsyncMock(side_effect=[
            b"progress = 25%\r",
            b"progress = 50%\r",
            b"progress = 75%\r",
            b"progress = 100%\r",
            b"",  # EOF
        ])
        process.stdout = AsyncMock()
        process.stdout.read = AsyncMock(return_value=b"Transcription output")
        process.wait = AsyncMock(return_value=0)
        process.kill = MagicMock()
        mock.return_value = process
        yield mock, process


@pytest.fixture
def mock_ffmpeg_success():
    """Mock successful ffmpeg subprocess."""
    with patch("asyncio.create_subprocess_exec") as mock:
        process = AsyncMock()
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b"", b""))
        process.wait = AsyncMock(return_value=0)
        mock.return_value = process
        yield mock


# ============================================================================
# Audio Normalization Tests
# ============================================================================

class TestAudioNormalization:
    """Tests for audio normalization functionality."""

    @pytest.mark.asyncio
    async def test_normalize_audio_creates_16k_mono(
        self, sample_raw_audio, temp_output_dir, mock_ffmpeg_success
    ):
        """Normalization creates 16kHz mono output."""
        output_path = temp_output_dir / "audio_16k.wav"

        # Create expected output file
        with wave.open(str(output_path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000 * 2)

        result = await normalize_audio(sample_raw_audio, output_path)

        assert result == output_path
        mock_ffmpeg_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_normalize_audio_default_output_path(
        self, sample_raw_audio, mock_ffmpeg_success
    ):
        """Normalization uses default output path."""
        expected_output = sample_raw_audio.parent / "audio_16k.wav"

        # Create the expected output file
        with wave.open(str(expected_output), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000 * 2)

        result = await normalize_audio(sample_raw_audio)

        assert result == expected_output

    @pytest.mark.asyncio
    async def test_normalize_audio_file_not_found(self, temp_output_dir):
        """Normalization raises for missing input."""
        nonexistent = temp_output_dir / "nonexistent.wav"

        with pytest.raises(FileNotFoundError):
            await normalize_audio(nonexistent)

    @pytest.mark.asyncio
    async def test_normalize_audio_ffmpeg_failure(self, sample_raw_audio):
        """Normalization raises on ffmpeg failure."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(return_value=(b"", b"FFmpeg error"))
            mock.return_value = process

            with pytest.raises(RuntimeError, match="FFmpeg normalization failed"):
                await normalize_audio(sample_raw_audio)

    @pytest.mark.asyncio
    async def test_normalize_audio_timeout(self, sample_raw_audio):
        """Normalization handles timeout correctly."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.kill = MagicMock()
            process.wait = AsyncMock()
            mock.return_value = process

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(ProcessTimeoutError, match="timed out"):
                    await normalize_audio(sample_raw_audio)

            process.kill.assert_called_once()


class TestOpusConversion:
    """Tests for Opus audio conversion."""

    @pytest.mark.asyncio
    async def test_convert_to_opus_success(
        self, sample_raw_audio, temp_output_dir, mock_ffmpeg_success
    ):
        """Opus conversion creates output file."""
        output_path = temp_output_dir / "recording.opus"
        output_path.write_bytes(b"opus data")

        result = await convert_to_opus(sample_raw_audio, output_path)

        assert result == output_path
        mock_ffmpeg_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_convert_to_opus_custom_bitrate(
        self, sample_raw_audio, temp_output_dir, mock_ffmpeg_success
    ):
        """Opus conversion uses custom bitrate."""
        output_path = temp_output_dir / "recording.opus"
        output_path.write_bytes(b"opus data")

        await convert_to_opus(sample_raw_audio, output_path, bitrate="128k")

        # Verify ffmpeg was called with correct bitrate
        call_args = mock_ffmpeg_success.call_args
        assert "-b:a" in call_args[0]
        assert "128k" in call_args[0]

    @pytest.mark.asyncio
    async def test_convert_to_opus_timeout(self, sample_raw_audio):
        """Opus conversion handles timeout."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.kill = MagicMock()
            process.wait = AsyncMock()
            mock.return_value = process

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(ProcessTimeoutError, match="timed out"):
                    await convert_to_opus(sample_raw_audio)


# ============================================================================
# Audio Duration Tests
# ============================================================================

class TestAudioDuration:
    """Tests for audio duration detection."""

    @pytest.mark.asyncio
    async def test_get_audio_duration_seconds(self, sample_wav_16k):
        """Duration detection returns correct value."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"2.0", b""))
            mock.return_value = process

            duration = await get_audio_duration_seconds(sample_wav_16k)

            assert duration == 2.0

    @pytest.mark.asyncio
    async def test_get_audio_duration_ffprobe_failure(self, sample_wav_16k):
        """Duration detection raises on ffprobe failure."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 1
            process.communicate = AsyncMock(return_value=(b"", b"ffprobe error"))
            mock.return_value = process

            with pytest.raises(RuntimeError, match="ffprobe failed"):
                await get_audio_duration_seconds(sample_wav_16k)

    @pytest.mark.asyncio
    async def test_get_audio_duration_invalid_output(self, sample_wav_16k):
        """Duration detection raises on invalid output."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"not_a_number", b""))
            mock.return_value = process

            with pytest.raises(RuntimeError, match="Could not parse"):
                await get_audio_duration_seconds(sample_wav_16k)

    def test_wav_duration_direct(self, sample_wav_16k):
        """Direct WAV duration reading works."""
        duration = get_wav_duration(sample_wav_16k)
        assert duration == pytest.approx(2.0, abs=0.01)


class TestAdaptiveTimeout:
    """Tests for adaptive timeout calculation."""

    def test_short_audio_gets_minimum_timeout(self):
        """Short audio files get minimum timeout."""
        # 10 seconds * 3 = 30 seconds, but minimum is 120
        timeout = calculate_adaptive_timeout(10.0)
        assert timeout == MIN_TIMEOUT_SECONDS

    def test_medium_audio_gets_proportional_timeout(self):
        """Medium audio files get proportional timeout."""
        # 60 seconds * 3 = 180 seconds
        timeout = calculate_adaptive_timeout(60.0)
        assert timeout == 180

    def test_long_audio_gets_maximum_timeout(self):
        """Long audio files are capped at maximum timeout."""
        # 1 hour * 3 = 3 hours, but max is 2 hours
        timeout = calculate_adaptive_timeout(3600.0)
        assert timeout == MAX_TIMEOUT_SECONDS

    def test_zero_duration_gets_minimum(self):
        """Zero duration gets minimum timeout."""
        timeout = calculate_adaptive_timeout(0.0)
        assert timeout == MIN_TIMEOUT_SECONDS

    def test_negative_duration_gets_minimum(self):
        """Negative duration handled gracefully."""
        timeout = calculate_adaptive_timeout(-100.0)
        assert timeout == MIN_TIMEOUT_SECONDS


# ============================================================================
# SRT Processing Tests
# ============================================================================

class TestSrtParsing:
    """Tests for SRT file parsing."""

    def test_parse_srt_basic(self, sample_srt_file):
        """Parsing extracts all segments."""
        segments = parse_srt(sample_srt_file)

        assert len(segments) == 3
        assert segments[0].index == 1
        assert segments[0].text == "Hello world."
        assert segments[0].start_time == "00:00:00,000"
        assert segments[0].end_time == "00:00:02,500"

    def test_parse_srt_multiline_text(self, temp_output_dir):
        """Parsing handles multiline text."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First line
Second line
Third line
"""
        srt_path = temp_output_dir / "multiline.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = parse_srt(srt_path)

        assert len(segments) == 1
        assert "First line\nSecond line\nThird line" in segments[0].text

    def test_parse_srt_empty_file(self, temp_output_dir):
        """Parsing handles empty file."""
        srt_path = temp_output_dir / "empty.srt"
        srt_path.write_text("", encoding='utf-8')

        segments = parse_srt(srt_path)

        assert segments == []

    def test_srt_segment_duration(self):
        """Segment duration calculation works."""
        segment = SrtSegment(
            index=1,
            start_time="00:01:30,500",
            end_time="00:01:35,000",
            text="Test"
        )

        assert segment.duration() == pytest.approx(4.5, abs=0.01)

    def test_srt_segment_time_conversion(self):
        """Time conversion handles hours correctly."""
        segment = SrtSegment(
            index=1,
            start_time="01:30:45,123",
            end_time="01:30:50,000",
            text="Test"
        )

        # 1h 30m 45.123s = 5445.123 seconds
        assert segment.start_seconds() == pytest.approx(5445.123, abs=0.001)


class TestSrtMerging:
    """Tests for SRT segment merging."""

    def test_merge_segments_by_sentence(self, temp_output_dir):
        """Segments merge until sentence ending."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
This is

2
00:00:01,000 --> 00:00:02,000
a sentence.

3
00:00:02,000 --> 00:00:03,000
Another one.
"""
        srt_path = temp_output_dir / "merge.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        merge_srt_segments(srt_path)

        segments = parse_srt(srt_path)
        # First two should merge, third stays separate
        assert len(segments) == 2

    def test_merge_segments_max_duration(self, temp_output_dir):
        """Segments split at max duration."""
        # Create segments that would exceed max duration
        srt_content = """1
00:00:00,000 --> 00:00:20,000
First part

2
00:00:20,000 --> 00:00:40,000
Second part
"""
        srt_path = temp_output_dir / "long.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        # With max_duration of 25 seconds, should stay separate
        merge_srt_segments(srt_path, max_duration_sec=25.0)

        segments = parse_srt(srt_path)
        assert len(segments) == 2

    def test_merge_empty_srt(self, temp_output_dir):
        """Merging handles empty SRT file."""
        srt_path = temp_output_dir / "empty.srt"
        srt_path.write_text("", encoding='utf-8')

        result = merge_srt_segments(srt_path)

        assert result == srt_path


class TestSrtToVtt:
    """Tests for SRT to VTT conversion."""

    def test_srt_to_vtt_basic(self, sample_srt_file, temp_output_dir):
        """Conversion creates valid VTT file."""
        vtt_path = temp_output_dir / "transcript.vtt"

        result = srt_to_vtt(sample_srt_file, vtt_path)

        assert result == vtt_path
        content = vtt_path.read_text(encoding='utf-8')
        assert content.startswith("WEBVTT")
        # VTT uses dots instead of commas
        assert "00:00:00.000" in content
        assert "00:00:02.500" in content

    def test_srt_to_vtt_default_path(self, sample_srt_file):
        """Conversion uses default output path."""
        result = srt_to_vtt(sample_srt_file)

        expected = sample_srt_file.with_suffix('.vtt')
        assert result == expected
        assert expected.exists()

    def test_srt_to_vtt_file_not_found(self, temp_output_dir):
        """Conversion raises for missing input."""
        nonexistent = temp_output_dir / "nonexistent.srt"

        with pytest.raises(FileNotFoundError):
            srt_to_vtt(nonexistent)


# ============================================================================
# Audio Chunking Tests
# ============================================================================

class TestAudioChunking:
    """Tests for audio chunking functionality."""

    @pytest.mark.asyncio
    async def test_split_audio_to_chunks(self, sample_wav_16k, temp_output_dir):
        """Audio splitting creates chunks."""
        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(b"2.0", b""))
            mock.return_value = process

            # Create mock chunk files
            chunks_dir = temp_output_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            (chunks_dir / "chunk_0000.wav").write_bytes(b"chunk data")

            # Duration is 2 seconds, chunk duration is 300 seconds
            # So we get just 1 chunk
            manifest = await split_audio_to_chunks(
                sample_wav_16k,
                chunks_dir,
                chunk_duration=300
            )

            assert manifest.chunk_count == 1
            assert len(manifest.chunks) == 1
            assert manifest.chunks[0].index == 0

    def test_chunk_manifest_serialization(self, temp_output_dir):
        """Manifest serializes and deserializes correctly."""
        chunks = [
            AudioChunk(
                index=0,
                chunk_path="/path/to/chunk_0.wav",
                start_time=0.0,
                end_time=300.0,
                duration=300.0
            ),
            AudioChunk(
                index=1,
                chunk_path="/path/to/chunk_1.wav",
                start_time=295.0,
                end_time=500.0,
                duration=205.0
            ),
        ]

        manifest = ChunkManifest(
            audio_path="/path/to/audio.wav",
            total_duration=500.0,
            chunk_count=2,
            chunk_duration=300,
            overlap_seconds=5,
            chunks=chunks
        )

        # Save and load
        manifest_path = temp_output_dir / "manifest.json"
        manifest.save(manifest_path)
        loaded = ChunkManifest.load(manifest_path)

        assert loaded.chunk_count == 2
        assert loaded.overlap_seconds == 5
        assert loaded.chunks[0].start_time == 0.0
        assert loaded.chunks[1].end_time == 500.0

    @pytest.mark.asyncio
    async def test_ensure_chunks_exist_creates_new(
        self, sample_wav_16k, temp_output_dir
    ):
        """ensure_chunks_exist creates chunks if needed."""
        with patch(
            "callwhisper.services.audio_chunker.split_audio_to_chunks"
        ) as mock_split:
            mock_manifest = ChunkManifest(
                audio_path=str(sample_wav_16k),
                total_duration=2.0,
                chunk_count=1,
                chunk_duration=CHUNK_DURATION_SECONDS,
                overlap_seconds=CHUNK_OVERLAP_SECONDS,
                chunks=[
                    AudioChunk(
                        index=0,
                        chunk_path=str(temp_output_dir / "chunks" / "chunk_0000.wav"),
                        start_time=0.0,
                        end_time=2.0,
                        duration=2.0
                    )
                ]
            )
            mock_split.return_value = mock_manifest

            manifest = await ensure_chunks_exist(sample_wav_16k, temp_output_dir)

            mock_split.assert_called_once()
            assert manifest.chunk_count == 1

    @pytest.mark.asyncio
    async def test_ensure_chunks_exist_reuses_existing(self, temp_output_dir):
        """ensure_chunks_exist reuses existing chunks."""
        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create existing chunk file
        chunk_path = chunks_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b"chunk data")

        # Create manifest
        manifest = ChunkManifest(
            audio_path="/path/to/audio.wav",
            total_duration=2.0,
            chunk_count=1,
            chunk_duration=CHUNK_DURATION_SECONDS,
            overlap_seconds=CHUNK_OVERLAP_SECONDS,
            chunks=[
                AudioChunk(
                    index=0,
                    chunk_path=str(chunk_path),
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0
                )
            ]
        )
        manifest.save(chunks_dir / "manifest.json")

        # Should reuse existing
        audio_path = temp_output_dir / "audio_16k.wav"
        audio_path.write_bytes(b"audio")

        with patch(
            "callwhisper.services.audio_chunker.split_audio_to_chunks"
        ) as mock_split:
            loaded = await ensure_chunks_exist(audio_path, temp_output_dir)

            mock_split.assert_not_called()
            assert loaded.chunk_count == 1


class TestChunkTranscriptMerging:
    """Tests for merging chunk transcripts."""

    def test_merge_single_transcript(self):
        """Single transcript returned unchanged."""
        transcripts = ["Hello world."]
        result = merge_chunk_transcripts(transcripts)
        assert result == "Hello world."

    def test_merge_multiple_transcripts(self):
        """Multiple transcripts merged correctly."""
        transcripts = [
            "Hello world.",
            "This is a test.",
            "Testing complete."
        ]
        result = merge_chunk_transcripts(transcripts)
        assert "Hello world." in result
        assert "This is a test." in result
        assert "Testing complete." in result

    def test_merge_with_overlap_deduplication(self):
        """Overlapping words are deduplicated."""
        transcripts = [
            "Hello world testing",
            "testing one two three"  # "testing" overlaps
        ]
        result = merge_chunk_transcripts(transcripts)
        # Should not have duplicate "testing"
        assert result.count("testing") == 1

    def test_merge_empty_transcripts(self):
        """Empty transcripts handled correctly."""
        transcripts = []
        result = merge_chunk_transcripts(transcripts)
        assert result == ""

    def test_merge_with_empty_chunks(self):
        """Empty chunks in middle handled correctly."""
        transcripts = [
            "Hello world.",
            "",
            "Goodbye world."
        ]
        result = merge_chunk_transcripts(transcripts)
        assert "Hello world." in result
        assert "Goodbye world." in result


# ============================================================================
# Transcription Pipeline Tests
# ============================================================================

class TestTranscribeAudio:
    """Tests for main transcription function."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_basic_flow(
        self, temp_output_dir, mock_settings
    ):
        """Basic transcription flow completes successfully."""
        # Create raw audio file
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b'\x00' * 44100 * 2)

        # Create normalized audio (mock will create it)
        normalized = temp_output_dir / "audio_16k.wav"

        # Create mock whisper output
        whisper_txt = temp_output_dir / "audio_16k.txt"
        whisper_srt = temp_output_dir / "audio_16k.srt"

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            # Create the file mock_norm would create
            normalized.write_bytes(b'\x00' * 16000 * 2)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 0
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(side_effect=[b"", b""])
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=0)
                    mock_proc.return_value = process

                    # Create whisper outputs
                    whisper_txt.write_text("Test transcript", encoding='utf-8')
                    whisper_srt.write_text(
                        "1\n00:00:00,000 --> 00:00:01,000\nTest\n",
                        encoding='utf-8'
                    )

                    result = await transcribe_audio(temp_output_dir, mock_settings)

                    assert result.name == "transcript.txt"
                    assert result.exists()

    @pytest.mark.asyncio
    async def test_transcribe_audio_missing_raw(
        self, temp_output_dir, mock_settings
    ):
        """Transcription raises for missing raw audio."""
        with pytest.raises(FileNotFoundError, match="Raw audio not found"):
            await transcribe_audio(temp_output_dir, mock_settings)

    @pytest.mark.asyncio
    async def test_transcribe_audio_progress_callback(
        self, temp_output_dir, mock_settings
    ):
        """Progress callback receives updates."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000 * 2)

        normalized = temp_output_dir / "audio_16k.wav"
        whisper_txt = temp_output_dir / "audio_16k.txt"

        progress_updates = []

        async def progress_callback(percent, message):
            progress_updates.append((percent, message))

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000 * 2)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 0
                    process.stderr = AsyncMock()
                    # Simulate progress updates
                    process.stderr.read = AsyncMock(side_effect=[
                        b"progress = 50%\r",
                        b"progress = 100%\r",
                        b""
                    ])
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=0)
                    mock_proc.return_value = process

                    whisper_txt.write_text("Test", encoding='utf-8')

                    await transcribe_audio(
                        temp_output_dir,
                        mock_settings,
                        progress_callback=progress_callback
                    )

                    assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_speech_detected(
        self, temp_output_dir, mock_settings
    ):
        """Handles no speech detected gracefully."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000)

        normalized = temp_output_dir / "audio_16k.wav"

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
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

                    # Don't create whisper output - simulates no speech
                    result = await transcribe_audio(temp_output_dir, mock_settings)

                    content = result.read_text(encoding='utf-8')
                    assert "[No speech detected]" in content


class TestTranscribeChunk:
    """Tests for single chunk transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_chunk_success(
        self, sample_wav_16k, temp_output_dir, mock_settings
    ):
        """Single chunk transcription succeeds."""
        chunk_path = temp_output_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b"chunk audio data")

        with patch(
            "callwhisper.services.transcriber.get_audio_duration_seconds",
            return_value=5.0
        ):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.returncode = 0
                process.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.return_value = process

                # Create output file
                txt_path = temp_output_dir / "chunk_0000.txt"
                txt_path.write_text("Chunk transcript", encoding='utf-8')

                result = await transcribe_chunk(
                    chunk_path,
                    mock_settings,
                    output_dir=temp_output_dir
                )

                assert result == "Chunk transcript"

    @pytest.mark.asyncio
    async def test_transcribe_chunk_timeout(
        self, temp_output_dir, mock_settings
    ):
        """Chunk transcription handles timeout."""
        chunk_path = temp_output_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b"chunk audio data")

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

                    process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_chunk_failure(
        self, temp_output_dir, mock_settings
    ):
        """Chunk transcription raises on failure."""
        chunk_path = temp_output_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b"chunk audio data")

        with patch(
            "callwhisper.services.transcriber.get_audio_duration_seconds",
            return_value=5.0
        ):
            with patch("asyncio.create_subprocess_exec") as mock_proc:
                process = AsyncMock()
                process.returncode = 1
                process.communicate = AsyncMock(
                    return_value=(b"", b"whisper error")
                )
                mock_proc.return_value = process

                with pytest.raises(TranscriptionError, match="failed"):
                    await transcribe_chunk(chunk_path, mock_settings)


class TestTranscribeAudioChunked:
    """Tests for chunked transcription with checkpoints."""

    @pytest.mark.asyncio
    async def test_chunked_transcription_basic(
        self, temp_output_dir, mock_settings
    ):
        """Chunked transcription processes all chunks."""
        normalized = temp_output_dir / "audio_16k.wav"
        normalized.write_bytes(b'\x00' * 16000 * 10)  # 10 seconds

        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create mock manifest
        chunk_paths = []
        for i in range(2):
            chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
            chunk_path.write_bytes(b"chunk data")
            chunk_paths.append(chunk_path)

        manifest = ChunkManifest(
            audio_path=str(normalized),
            total_duration=10.0,
            chunk_count=2,
            chunk_duration=5,
            overlap_seconds=1,
            chunks=[
                AudioChunk(
                    index=i,
                    chunk_path=str(chunk_paths[i]),
                    start_time=i * 4.0,
                    end_time=(i + 1) * 5.0,
                    duration=5.0
                )
                for i in range(2)
            ]
        )

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist",
            return_value=manifest
        ):
            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                side_effect=["First chunk.", "Second chunk."]
            ):
                with patch(
                    "callwhisper.services.transcriber.update_checkpoint"
                ):
                    with patch(
                        "callwhisper.services.transcriber.merge_chunk_transcripts",
                        return_value="First chunk. Second chunk."
                    ):
                        result = await transcribe_audio_chunked(
                            temp_output_dir,
                            mock_settings
                        )

                        assert result.name == "transcript.txt"
                        content = result.read_text(encoding='utf-8')
                        assert "First chunk" in content
                        assert "Second chunk" in content

    @pytest.mark.asyncio
    async def test_chunked_transcription_resume(
        self, temp_output_dir, mock_settings
    ):
        """Chunked transcription resumes from checkpoint."""
        normalized = temp_output_dir / "audio_16k.wav"
        normalized.write_bytes(b'\x00' * 16000 * 10)

        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create partial transcript (already completed first chunk)
        partial = temp_output_dir / "partial_transcript.txt"
        partial.write_text("First chunk already done.", encoding='utf-8')

        chunk_paths = []
        for i in range(2):
            chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
            chunk_path.write_bytes(b"chunk data")
            chunk_paths.append(chunk_path)

        manifest = ChunkManifest(
            audio_path=str(normalized),
            total_duration=10.0,
            chunk_count=2,
            chunk_duration=5,
            overlap_seconds=1,
            chunks=[
                AudioChunk(
                    index=i,
                    chunk_path=str(chunk_paths[i]),
                    start_time=i * 4.0,
                    end_time=(i + 1) * 5.0,
                    duration=5.0
                )
                for i in range(2)
            ]
        )

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist",
            return_value=manifest
        ):
            # Only transcribe chunk 1 (skip chunk 0)
            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                return_value="Second chunk."
            ) as mock_transcribe:
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with patch(
                        "callwhisper.services.transcriber.merge_chunk_transcripts",
                        return_value="First chunk already done. Second chunk."
                    ):
                        result = await transcribe_audio_chunked(
                            temp_output_dir,
                            mock_settings,
                            start_from_chunk=1  # Resume from chunk 1
                        )

                        # Should only call transcribe_chunk once (for chunk 1)
                        assert mock_transcribe.call_count == 1

    @pytest.mark.asyncio
    async def test_chunked_transcription_partial_callback(
        self, temp_output_dir, mock_settings
    ):
        """Partial transcript callback receives updates."""
        normalized = temp_output_dir / "audio_16k.wav"
        normalized.write_bytes(b'\x00' * 16000 * 5)

        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunk_path = chunks_dir / "chunk_0000.wav"
        chunk_path.write_bytes(b"chunk data")

        manifest = ChunkManifest(
            audio_path=str(normalized),
            total_duration=5.0,
            chunk_count=1,
            chunk_duration=5,
            overlap_seconds=1,
            chunks=[
                AudioChunk(
                    index=0,
                    chunk_path=str(chunk_path),
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0
                )
            ]
        )

        partial_updates = []

        async def partial_callback(text, is_final):
            partial_updates.append((text, is_final))

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist",
            return_value=manifest
        ):
            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                return_value="Test transcript"
            ):
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with patch(
                        "callwhisper.services.transcriber.merge_chunk_transcripts",
                        return_value="Test transcript"
                    ):
                        await transcribe_audio_chunked(
                            temp_output_dir,
                            mock_settings,
                            partial_transcript_callback=partial_callback
                        )

                        # Should have at least one partial and one final
                        assert len(partial_updates) >= 1
                        # Last should be final
                        assert partial_updates[-1][1] is True


# ============================================================================
# Audio Pipeline Tests
# ============================================================================

class TestAudioPipeline:
    """Tests for the generator-based audio pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_get_metadata(self, sample_wav_16k):
        """Pipeline gets audio metadata."""
        pipeline = AudioPipeline()

        with patch("asyncio.create_subprocess_exec") as mock:
            process = AsyncMock()
            process.returncode = 0
            process.communicate = AsyncMock(return_value=(
                json.dumps({
                    "format": {
                        "duration": "2.0",
                        "bit_rate": "256000",
                        "size": "64000",
                        "format_name": "wav"
                    },
                    "streams": [{
                        "codec_type": "audio",
                        "sample_rate": "16000",
                        "channels": 1,
                        "codec_name": "pcm_s16le"
                    }]
                }).encode(),
                b""
            ))
            mock.return_value = process

            metadata = await pipeline.get_metadata(sample_wav_16k)

            assert metadata.duration == 2.0
            assert metadata.sample_rate == 16000
            assert metadata.channels == 1

    @pytest.mark.asyncio
    async def test_pipeline_stream_chunks(self, sample_wav_16k):
        """Pipeline streams audio in chunks."""
        config = PipelineConfig(chunk_duration=1.0)
        pipeline = AudioPipeline(config)

        with patch.object(
            pipeline, "get_metadata",
            return_value=AudioMetadata(
                duration=2.0,
                sample_rate=16000,
                channels=1,
                codec="pcm_s16le",
                bit_rate=256000,
                file_size=64000,
                format_name="wav"
            )
        ):
            with patch.object(
                pipeline, "_extract_chunk",
                return_value=b"chunk_data"
            ):
                chunks = []
                async for chunk in pipeline.stream_chunks(
                    sample_wav_16k,
                    chunk_duration=1.0,
                    use_degradation_aware_chunking=False
                ):
                    chunks.append(chunk)

                assert len(chunks) == 2
                assert chunks[0].index == 0
                assert chunks[1].index == 1

    def test_pipeline_memory_estimate(self):
        """Pipeline provides memory usage estimate."""
        pipeline = AudioPipeline()

        estimate = pipeline.estimate_memory_usage(
            total_duration=600.0,  # 10 minutes
            chunk_duration=30.0
        )

        assert estimate["total_duration"] == 600.0
        assert estimate["chunk_duration"] == 30.0
        assert estimate["num_chunks"] == 21  # 600/30 + 1
        assert "memory_savings_percent" in estimate

    def test_pipeline_degradation_settings(self):
        """Pipeline adjusts for degradation level."""
        pipeline = AudioPipeline()

        with patch.object(
            pipeline._degradation_manager,
            "get_current_level",
            return_value=MagicMock(value="FAST")
        ):
            from callwhisper.core.degradation import DegradationLevel
            duration = pipeline.get_chunk_duration_for_level(DegradationLevel.FAST)
            assert duration == 10.0  # FAST level uses 10s chunks

    @pytest.mark.asyncio
    async def test_convenience_stream_audio_chunks(self, sample_wav_16k):
        """Convenience function streams chunks."""
        with patch(
            "callwhisper.services.audio_pipeline.AudioPipeline"
        ) as MockPipeline:
            mock_instance = MagicMock()
            MockPipeline.return_value = mock_instance

            async def mock_stream(*args, **kwargs):
                yield PipelineChunk(0, 0.0, 1.0, b"data", {})

            mock_instance.stream_chunks = mock_stream

            chunks = []
            async for chunk in stream_audio_chunks(sample_wav_16k):
                chunks.append(chunk)

            assert len(chunks) == 1


# ============================================================================
# Transcription Error Handling Tests
# ============================================================================

class TestTranscriptionErrorHandling:
    """Tests for error handling in transcription."""

    @pytest.mark.asyncio
    async def test_whisper_not_found(self, temp_output_dir, mock_settings):
        """Handles missing whisper executable."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000)

        normalized = temp_output_dir / "audio_16k.wav"

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch(
                    "callwhisper.services.transcriber.get_whisper_path",
                    return_value=Path("/nonexistent/whisper")
                ):
                    with patch(
                        "asyncio.create_subprocess_exec",
                        side_effect=FileNotFoundError()
                    ):
                        with pytest.raises(TranscriptionError, match="not found"):
                            await transcribe_audio(temp_output_dir, mock_settings)

    @pytest.mark.asyncio
    async def test_transcription_timeout_kills_process(
        self, temp_output_dir, mock_settings
    ):
        """Timeout kills the whisper process."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000)

        normalized = temp_output_dir / "audio_16k.wav"

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000)

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
                        await transcribe_audio(temp_output_dir, mock_settings)

                    process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_whisper_nonzero_exit(self, temp_output_dir, mock_settings):
        """Handles whisper non-zero exit code."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b'\x00' * 16000)

        normalized = temp_output_dir / "audio_16k.wav"

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=1.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 1
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(
                        side_effect=[b"Error: model not found", b""]
                    )
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=1)
                    mock_proc.return_value = process

                    with pytest.raises(TranscriptionError, match="failed"):
                        await transcribe_audio(temp_output_dir, mock_settings)


# ============================================================================
# Integration Tests
# ============================================================================

class TestTranscriptionIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_progress(self, temp_output_dir, mock_settings):
        """Full pipeline with progress tracking."""
        raw_audio = temp_output_dir / "audio_raw.wav"
        with wave.open(str(raw_audio), 'w') as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b'\x00' * 44100 * 4)

        normalized = temp_output_dir / "audio_16k.wav"
        whisper_txt = temp_output_dir / "audio_16k.txt"
        whisper_srt = temp_output_dir / "audio_16k.srt"

        progress_calls = []
        partial_calls = []

        async def progress_cb(pct, msg):
            progress_calls.append((pct, msg))

        async def partial_cb(text, is_final):
            partial_calls.append((text, is_final))

        with patch("callwhisper.services.transcriber.normalize_audio") as mock_norm:
            mock_norm.return_value = normalized
            normalized.write_bytes(b'\x00' * 16000 * 2)

            with patch(
                "callwhisper.services.transcriber.get_audio_duration_seconds",
                return_value=2.0
            ):
                with patch("asyncio.create_subprocess_exec") as mock_proc:
                    process = AsyncMock()
                    process.returncode = 0
                    process.stderr = AsyncMock()
                    process.stderr.read = AsyncMock(side_effect=[
                        b"progress = 25%\r",
                        b"progress = 50%\r",
                        b"progress = 75%\r",
                        b"progress = 100%\r",
                        b""
                    ])
                    process.stdout = AsyncMock()
                    process.stdout.read = AsyncMock(return_value=b"")
                    process.wait = AsyncMock(return_value=0)
                    mock_proc.return_value = process

                    whisper_txt.write_text("Hello world.", encoding='utf-8')
                    whisper_srt.write_text(
                        "1\n00:00:00,000 --> 00:00:02,000\nHello world.\n",
                        encoding='utf-8'
                    )

                    result = await transcribe_audio(
                        temp_output_dir,
                        mock_settings,
                        progress_callback=progress_cb,
                        partial_transcript_callback=partial_cb
                    )

                    # Verify result
                    assert result.exists()
                    content = result.read_text(encoding='utf-8')
                    assert "Hello world" in content

                    # Verify progress was tracked
                    assert len(progress_calls) > 0

                    # Verify partial callback called with final
                    assert len(partial_calls) == 1
                    assert partial_calls[0][1] is True  # is_final

    @pytest.mark.asyncio
    async def test_chunked_pipeline_failure_recovery(
        self, temp_output_dir, mock_settings
    ):
        """Chunked pipeline saves progress on failure."""
        normalized = temp_output_dir / "audio_16k.wav"
        normalized.write_bytes(b'\x00' * 16000 * 10)

        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        chunk_paths = []
        for i in range(3):
            chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
            chunk_path.write_bytes(b"chunk data")
            chunk_paths.append(chunk_path)

        manifest = ChunkManifest(
            audio_path=str(normalized),
            total_duration=15.0,
            chunk_count=3,
            chunk_duration=5,
            overlap_seconds=0,
            chunks=[
                AudioChunk(
                    index=i,
                    chunk_path=str(chunk_paths[i]),
                    start_time=i * 5.0,
                    end_time=(i + 1) * 5.0,
                    duration=5.0
                )
                for i in range(3)
            ]
        )

        call_count = 0

        async def mock_transcribe_chunk(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "First chunk."
            elif call_count == 2:
                raise TranscriptionError("Simulated failure")
            return "Third chunk."

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist",
            return_value=manifest
        ):
            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                side_effect=mock_transcribe_chunk
            ):
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with pytest.raises(TranscriptionError):
                        await transcribe_audio_chunked(
                            temp_output_dir,
                            mock_settings
                        )

                    # Partial transcript should be saved
                    partial = temp_output_dir / "partial_transcript.txt"
                    assert partial.exists()
                    content = partial.read_text(encoding='utf-8')
                    assert "First chunk." in content
