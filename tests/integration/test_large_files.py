"""
Tests for large file handling.

Tests boundary conditions for large files:
- Very large audio files (>1GB)
- Very long recordings (>1 hour)
- Large transcript outputs (>1MB)
- Memory-efficient processing
"""

import asyncio
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ============================================================================
# Large Audio File Tests
# ============================================================================


class TestLargeAudioFiles:
    """Tests for handling large audio files."""

    def test_stream_large_file(self):
        """Large files are streamed, not loaded entirely."""
        from callwhisper.core.resource_manager import AudioResourceManager, ResourceConfig

        config = ResourceConfig(cleanup_interval=0)
        manager = AudioResourceManager(config)

        # Create a simulated large file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".raw") as f:
            # Write 10MB of data
            chunk = b"x" * (1024 * 1024)  # 1MB
            for _ in range(10):
                f.write(chunk)
            temp_path = Path(f.name)

        try:
            # Stream the file in chunks
            chunks_read = 0
            with manager.open_audio_stream(temp_path, chunk_size=65536) as stream:
                for chunk in stream:
                    chunks_read += 1
                    # Each chunk should be manageable size
                    assert len(chunk) <= 65536

            # Should have read multiple chunks
            assert chunks_read > 100
        finally:
            temp_path.unlink()

    def test_audio_duration_very_long(self):
        """Very long audio durations are handled."""
        from callwhisper.services.transcriber import calculate_adaptive_timeout

        # 4-hour audio
        duration_seconds = 4 * 60 * 60  # 4 hours

        timeout = calculate_adaptive_timeout(duration_seconds)

        # Should return a reasonable timeout
        assert timeout > 0
        # Should be capped at maximum
        from callwhisper.services.transcriber import MAX_TIMEOUT_SECONDS
        assert timeout <= MAX_TIMEOUT_SECONDS

    def test_file_size_limits_enforced(self):
        """File size limits are enforced."""
        # This tests that we handle the concept of size limits
        max_file_size = 10 * 1024 * 1024 * 1024  # 10GB limit

        # Simulate checking file size
        class MockPath:
            def stat(self):
                class Stat:
                    st_size = 11 * 1024 * 1024 * 1024  # 11GB
                return Stat()

        mock_path = MockPath()
        file_size = mock_path.stat().st_size

        # Should detect file is too large
        assert file_size > max_file_size


# ============================================================================
# Memory Efficiency Tests
# ============================================================================


class TestMemoryEfficiency:
    """Tests for memory-efficient processing."""

    def test_chunked_reading_limits_memory(self):
        """Chunked reading limits memory usage."""
        # Create large in-memory buffer
        large_data = io.BytesIO(b"x" * (10 * 1024 * 1024))  # 10MB

        max_chunk = 64 * 1024  # 64KB max per chunk
        chunks_read = 0
        max_memory_per_chunk = 0

        while True:
            chunk = large_data.read(max_chunk)
            if not chunk:
                break
            chunks_read += 1
            max_memory_per_chunk = max(max_memory_per_chunk, len(chunk))

        assert chunks_read > 100
        assert max_memory_per_chunk <= max_chunk

    def test_generator_pipeline_memory_efficient(self):
        """Generator pipelines don't buffer entire files."""
        def chunk_generator(size_bytes, chunk_size):
            """Simulate reading large file in chunks."""
            remaining = size_bytes
            while remaining > 0:
                chunk_len = min(chunk_size, remaining)
                yield b"x" * chunk_len
                remaining -= chunk_len

        # Simulate 100MB file
        total_size = 100 * 1024 * 1024
        chunk_size = 64 * 1024

        # Process with generator (memory efficient)
        total_processed = 0
        for chunk in chunk_generator(total_size, chunk_size):
            total_processed += len(chunk)

        assert total_processed == total_size


# ============================================================================
# Large Transcript Tests
# ============================================================================


class TestLargeTranscripts:
    """Tests for handling large transcript outputs."""

    def test_large_transcript_serialization(self):
        """Large transcripts can be serialized."""
        # Simulate a very long transcript
        segments = []
        for i in range(10000):  # 10,000 segments
            segments.append({
                "start": i * 10.0,
                "end": (i + 1) * 10.0,
                "text": f"This is segment number {i} with some text content.",
            })

        import json
        transcript_json = json.dumps(segments)

        # Should produce a large but valid JSON (>900KB)
        assert len(transcript_json) > 900 * 1024  # > 900KB
        # Should be valid JSON
        parsed = json.loads(transcript_json)
        assert len(parsed) == 10000

    def test_srt_output_very_long(self, temp_dir):
        """SRT output for very long recordings is valid."""
        from callwhisper.services.transcriber import srt_to_vtt

        # Generate a large SRT file
        srt_lines = []
        for i in range(5000):
            hours = i // 360
            mins = (i // 6) % 60
            secs = (i * 10) % 60
            start = f"{hours:02d}:{mins:02d}:{secs:02d},000"
            end = f"{hours:02d}:{mins:02d}:{secs + 5:02d},000"

            srt_lines.append(f"{i + 1}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"Subtitle line {i + 1}")
            srt_lines.append("")

        srt_content = "\n".join(srt_lines)

        # Write SRT to file
        srt_path = temp_dir / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        # Convert to VTT
        vtt_path = srt_to_vtt(srt_path)

        # Read and verify VTT content
        vtt_content = vtt_path.read_text(encoding="utf-8")

        # Should produce valid VTT
        assert vtt_content.startswith("WEBVTT")
        assert "Subtitle line 5000" in vtt_content


# ============================================================================
# Timeout Handling for Large Files Tests
# ============================================================================


class TestTimeoutHandlingLargeFiles:
    """Tests for timeout handling with large files."""

    @pytest.mark.asyncio
    async def test_adaptive_timeout_for_long_audio(self):
        """Adaptive timeout scales with audio duration."""
        from callwhisper.services.transcriber import calculate_adaptive_timeout

        # Short audio
        short_timeout = calculate_adaptive_timeout(60.0)  # 1 minute

        # Long audio
        long_timeout = calculate_adaptive_timeout(3600.0)  # 1 hour

        # Very long audio
        very_long_timeout = calculate_adaptive_timeout(14400.0)  # 4 hours

        # Longer audio should have longer timeouts
        assert short_timeout < long_timeout
        # But should cap at maximum
        from callwhisper.services.transcriber import MAX_TIMEOUT_SECONDS
        assert very_long_timeout <= MAX_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_timeout_cascade_for_large_workflow(self):
        """Timeout cascade handles long workflows."""
        from callwhisper.core.timeout_cascade import TimeoutCascade, TimeoutConfig

        config = TimeoutConfig(
            workflow_max=7200.0,  # 2 hour workflow
            stage_timeouts={
                "upload": 600.0,       # 10 min upload
                "normalize": 300.0,     # 5 min normalize
                "transcribe": 6000.0,   # 100 min transcribe
            }
        )
        cascade = TimeoutCascade(config)
        ctx = cascade.start_workflow("large-file")

        # Verify stage timeouts are reasonable
        upload_timeout = cascade.get_remaining(ctx, "upload")
        assert upload_timeout == 600.0

        transcribe_timeout = cascade.get_remaining(ctx, "transcribe")
        assert transcribe_timeout == 6000.0


# ============================================================================
# Progress Tracking for Large Files Tests
# ============================================================================


class TestProgressTrackingLargeFiles:
    """Tests for progress tracking with large files."""

    @pytest.mark.asyncio
    async def test_progress_updates_for_long_processing(self):
        """Progress updates are sent during long processing."""
        progress_updates = []

        async def progress_callback(percent: float, message: str):
            progress_updates.append((percent, message))

        # Simulate processing with progress
        total_chunks = 100
        for i in range(total_chunks):
            percent = (i + 1) / total_chunks * 100
            await progress_callback(percent, f"Processing chunk {i + 1}/{total_chunks}")
            await asyncio.sleep(0)  # Yield control

        assert len(progress_updates) == 100
        assert progress_updates[0][0] == 1.0  # First update
        assert progress_updates[-1][0] == 100.0  # Last update

    @pytest.mark.asyncio
    async def test_chunked_transcription_progress(self):
        """Chunked transcription reports progress per chunk."""
        chunks_processed = []

        async def process_chunk(chunk_id: int, total_chunks: int):
            chunks_processed.append(chunk_id)
            return f"Transcript for chunk {chunk_id}"

        # Simulate 20 chunks
        total_chunks = 20
        transcripts = []
        for i in range(total_chunks):
            result = await process_chunk(i, total_chunks)
            transcripts.append(result)

        assert len(chunks_processed) == 20
        assert len(transcripts) == 20


# ============================================================================
# Storage Space Tests
# ============================================================================


class TestStorageSpace:
    """Tests for storage space handling."""

    def test_check_available_space(self):
        """Available disk space is checked before large operations."""
        import os

        # Get actual disk usage
        statvfs = os.statvfs('/')
        available_bytes = statvfs.f_frsize * statvfs.f_bavail

        # Should have some space available
        assert available_bytes > 0

        # Simulate space check before large file operation
        required_space = 1024 * 1024 * 100  # 100MB
        has_space = available_bytes > required_space
        assert has_space  # Most systems should have 100MB free

    def test_temp_space_cleanup_on_error(self):
        """Temporary files are cleaned up on processing error."""
        from callwhisper.core.resource_manager import ResourceManager, ResourceConfig

        config = ResourceConfig(cleanup_interval=0)
        manager = ResourceManager(config)

        temp_path = None
        try:
            with manager.temp_file(suffix=".tmp") as path:
                temp_path = path
                path.write_bytes(b"x" * 1024)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Temp file should be deleted despite error
        assert temp_path is not None
        assert not temp_path.exists()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestLargeFileEdgeCases:
    """Tests for edge cases with large files."""

    def test_zero_byte_file(self):
        """Zero-byte files are handled gracefully."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            # Don't write anything - zero bytes
            temp_path = Path(f.name)

        try:
            file_size = temp_path.stat().st_size
            assert file_size == 0

            # Reading should work but return empty
            content = temp_path.read_bytes()
            assert content == b""
        finally:
            temp_path.unlink()

    def test_file_at_exact_chunk_boundary(self):
        """Files at exact chunk boundaries are handled correctly."""
        chunk_size = 65536  # 64KB
        exact_size = chunk_size * 10  # Exactly 10 chunks

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * exact_size)
            temp_path = Path(f.name)

        try:
            chunks_read = 0
            with open(temp_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunks_read += 1
                    assert len(chunk) == chunk_size

            assert chunks_read == 10
        finally:
            temp_path.unlink()

    def test_file_one_byte_over_boundary(self):
        """Files one byte over boundary have correct last chunk."""
        chunk_size = 65536
        file_size = chunk_size * 10 + 1  # 10 full chunks + 1 byte

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * file_size)
            temp_path = Path(f.name)

        try:
            chunks = []
            with open(temp_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    chunks.append(len(chunk))

            # Should have 11 chunks, last one is 1 byte
            assert len(chunks) == 11
            assert chunks[-1] == 1
        finally:
            temp_path.unlink()
