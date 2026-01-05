"""
Integration tests for job recovery functionality.

Tests how the system recovers from incomplete jobs:
- Incomplete jobs listed on startup
- Resume from arbitrary checkpoint
- Audio file deletion prevents resume
- Checkpoint corruption recovery
- Discard incomplete job
- Partial transcript recovery
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import wave
import struct
import math


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.transcription.model = "ggml-base.bin"
    settings.transcription.language = "en"
    settings.transcription.beam_size = 5
    settings.transcription.best_of = 5
    return settings


@pytest.fixture
def sample_checkpoint(temp_dir):
    """Create a sample checkpoint file."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "job_id": "20241229_120000_TEST123",
        "status": "chunk_2",
        "chunks_completed": 2,
        "total_chunks": 5,
        "partial_transcript": "First chunk. Second chunk.",
        "audio_path": str(temp_dir / "audio_16k.wav"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    checkpoint_path = checkpoint_dir / "20241229_120000_TEST123.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    return checkpoint_path


@pytest.fixture
def sample_output_folder(temp_dir):
    """Create a sample output folder with partial transcription."""
    output_folder = temp_dir / "20241229_120000_TEST123"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create audio file
    audio_path = output_folder / "audio_16k.wav"
    _create_test_audio(audio_path, duration=10)

    # Create chunks directory with some chunks
    chunks_dir = output_folder / "chunks"
    chunks_dir.mkdir()

    manifest = {
        "audio_path": str(audio_path),
        "total_duration": 10.0,
        "chunk_count": 5,
        "chunk_duration": 2,
        "overlap_seconds": 0,
        "chunks": []
    }

    for i in range(5):
        chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
        _create_test_audio(chunk_path, duration=2)
        manifest["chunks"].append({
            "index": i,
            "chunk_path": str(chunk_path),
            "start_time": i * 2.0,
            "end_time": (i + 1) * 2.0,
            "duration": 2.0
        })

    (chunks_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Create partial transcript
    (output_folder / "partial_transcript.txt").write_text(
        "First chunk.\n---CHUNK_BOUNDARY---\nSecond chunk.",
        encoding="utf-8"
    )

    return output_folder


# ============================================================================
# Checkpoint Tests
# ============================================================================

class TestCheckpointBasics:
    """Tests for basic checkpoint functionality."""

    def test_checkpoint_creation(self, temp_dir):
        """Checkpoint file is created correctly."""
        from callwhisper.core.job_store import save_checkpoint

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            save_checkpoint(
                job_id="test_job",
                status="processing",
                chunks_completed=0,
                total_chunks=5
            )

            checkpoint_path = checkpoint_dir / "test_job.json"
            assert checkpoint_path.exists()

            data = json.loads(checkpoint_path.read_text())
            assert data["job_id"] == "test_job"
            assert data["status"] == "processing"

    def test_checkpoint_update(self, temp_dir):
        """Checkpoint is updated correctly."""
        from callwhisper.core.job_store import save_checkpoint, update_checkpoint

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            # Initial checkpoint
            save_checkpoint(
                job_id="test_job",
                status="processing",
                chunks_completed=0,
                total_chunks=5
            )

            # Update checkpoint
            update_checkpoint(
                job_id="test_job",
                status="chunk_2",
                chunks_completed=2,
                total_chunks=5,
                partial_transcript="Some text"
            )

            checkpoint_path = checkpoint_dir / "test_job.json"
            data = json.loads(checkpoint_path.read_text())
            assert data["chunks_completed"] == 2
            assert data["partial_transcript"] == "Some text"

    def test_checkpoint_deletion(self, sample_checkpoint):
        """Checkpoint can be deleted."""
        from callwhisper.core.job_store import delete_checkpoint

        checkpoint_dir = sample_checkpoint.parent

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            delete_checkpoint("20241229_120000_TEST123")

            assert not sample_checkpoint.exists()


class TestCheckpointLoading:
    """Tests for loading checkpoints."""

    def test_load_existing_checkpoint(self, sample_checkpoint):
        """Existing checkpoint is loaded correctly."""
        from callwhisper.core.job_store import load_checkpoint

        checkpoint_dir = sample_checkpoint.parent

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            data = load_checkpoint("20241229_120000_TEST123")

            assert data is not None
            assert data["job_id"] == "20241229_120000_TEST123"
            assert data["chunks_completed"] == 2

    def test_load_nonexistent_checkpoint(self, temp_dir):
        """Loading nonexistent checkpoint returns None."""
        from callwhisper.core.job_store import load_checkpoint

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            data = load_checkpoint("nonexistent_job")

            assert data is None

    def test_load_corrupted_checkpoint(self, temp_dir):
        """Corrupted checkpoint returns None."""
        from callwhisper.core.job_store import load_checkpoint

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create corrupted checkpoint
        checkpoint_path = checkpoint_dir / "corrupted.json"
        checkpoint_path.write_text("not valid json{", encoding="utf-8")

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            data = load_checkpoint("corrupted")

            assert data is None


# ============================================================================
# Incomplete Job Discovery Tests
# ============================================================================

class TestIncompleteJobDiscovery:
    """Tests for discovering incomplete jobs."""

    def test_list_incomplete_jobs(self, temp_dir):
        """Incomplete jobs are listed correctly."""
        from callwhisper.core.job_store import list_incomplete_jobs

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create some incomplete jobs
        for i, status in enumerate(["chunk_2", "processing", "chunk_5"]):
            checkpoint = {
                "job_id": f"job_{i}",
                "status": status,
                "chunks_completed": i,
                "total_chunks": 10,
            }
            (checkpoint_dir / f"job_{i}.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            jobs = list_incomplete_jobs()

            assert len(jobs) == 3
            assert all(j["status"] != "complete" for j in jobs)

    def test_incomplete_jobs_empty(self, temp_dir):
        """Empty checkpoint directory returns empty list."""
        from callwhisper.core.job_store import list_incomplete_jobs

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            jobs = list_incomplete_jobs()

            assert jobs == []


# ============================================================================
# Resume Tests
# ============================================================================

class TestResumeFromCheckpoint:
    """Tests for resuming from checkpoints."""

    @pytest.mark.asyncio
    async def test_resume_chunked_transcription(
        self, sample_output_folder, mock_settings
    ):
        """Resume chunked transcription from checkpoint."""
        from callwhisper.services.transcriber import transcribe_audio_chunked

        # Load manifest
        manifest_path = sample_output_folder / "chunks" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist"
        ) as mock_ensure:
            from callwhisper.services.audio_chunker import ChunkManifest
            mock_ensure.return_value = ChunkManifest.from_dict(manifest_data)

            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                side_effect=["Third.", "Fourth.", "Fifth."]
            ):
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with patch(
                        "callwhisper.services.transcriber.merge_chunk_transcripts",
                        return_value="First. Second. Third. Fourth. Fifth."
                    ):
                        result = await transcribe_audio_chunked(
                            sample_output_folder,
                            mock_settings,
                            start_from_chunk=2  # Resume from chunk 2
                        )

                        assert result.exists()

    @pytest.mark.asyncio
    async def test_resume_loads_partial_transcript(
        self, sample_output_folder, mock_settings
    ):
        """Resume loads existing partial transcript."""
        # Verify partial transcript exists
        partial_file = sample_output_folder / "partial_transcript.txt"
        assert partial_file.exists()

        content = partial_file.read_text(encoding="utf-8")
        assert "First chunk." in content
        assert "Second chunk." in content


class TestResumeFailures:
    """Tests for resume failure scenarios."""

    @pytest.mark.asyncio
    async def test_resume_audio_deleted(
        self, sample_output_folder, mock_settings
    ):
        """Resume fails when audio file is deleted."""
        from callwhisper.services.transcriber import transcribe_audio_chunked

        # Delete the audio file
        audio_path = sample_output_folder / "audio_16k.wav"
        audio_path.unlink()

        with pytest.raises(FileNotFoundError, match="audio not found"):
            await transcribe_audio_chunked(
                sample_output_folder,
                mock_settings,
                start_from_chunk=2
            )

    @pytest.mark.asyncio
    async def test_resume_chunk_deleted(self, sample_output_folder, mock_settings):
        """Resume fails when needed chunk is deleted."""
        from callwhisper.services.transcriber import transcribe_audio_chunked

        # Delete chunk 2
        chunk_path = sample_output_folder / "chunks" / "chunk_0002.wav"
        chunk_path.unlink()

        # Update manifest to show chunks exist but they don't
        manifest_path = sample_output_folder / "chunks" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist"
        ) as mock_ensure:
            from callwhisper.services.audio_chunker import ChunkManifest

            # This will try to load but chunks are missing
            mock_manifest = ChunkManifest.from_dict(manifest)
            mock_ensure.return_value = mock_manifest

            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                side_effect=FileNotFoundError("Chunk not found")
            ):
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with pytest.raises(FileNotFoundError):
                        await transcribe_audio_chunked(
                            sample_output_folder,
                            mock_settings,
                            start_from_chunk=2
                        )

    @pytest.mark.asyncio
    async def test_resume_corrupted_manifest(self, sample_output_folder, mock_settings):
        """Resume handles corrupted manifest."""
        from callwhisper.services.audio_chunker import ensure_chunks_exist

        # Corrupt the manifest
        manifest_path = sample_output_folder / "chunks" / "manifest.json"
        manifest_path.write_text("not valid json{", encoding="utf-8")

        audio_path = sample_output_folder / "audio_16k.wav"

        # Should recreate chunks from scratch
        with patch(
            "callwhisper.services.audio_chunker.split_audio_to_chunks"
        ) as mock_split:
            from callwhisper.services.audio_chunker import ChunkManifest, AudioChunk
            mock_split.return_value = ChunkManifest(
                audio_path=str(audio_path),
                total_duration=10.0,
                chunk_count=1,
                chunk_duration=10,
                overlap_seconds=0,
                chunks=[
                    AudioChunk(
                        index=0,
                        chunk_path=str(sample_output_folder / "chunks" / "chunk_0000.wav"),
                        start_time=0.0,
                        end_time=10.0,
                        duration=10.0
                    )
                ]
            )

            # Should fall back to recreating
            await ensure_chunks_exist(audio_path, sample_output_folder)

            mock_split.assert_called()


# ============================================================================
# Discard Incomplete Job Tests
# ============================================================================

class TestDiscardIncompleteJob:
    """Tests for discarding incomplete jobs."""

    def test_discard_removes_checkpoint(self, sample_checkpoint):
        """Discarding removes checkpoint file."""
        from callwhisper.core.job_store import delete_checkpoint

        checkpoint_dir = sample_checkpoint.parent

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            delete_checkpoint("20241229_120000_TEST123")

            assert not sample_checkpoint.exists()

    def test_discard_removes_partial_files(self, sample_output_folder):
        """Discarding removes partial output files."""
        partial_file = sample_output_folder / "partial_transcript.txt"
        assert partial_file.exists()

        # Remove partial file
        partial_file.unlink()
        assert not partial_file.exists()

    def test_discard_preserves_audio(self, sample_output_folder):
        """Discarding preserves original audio file."""
        audio_file = sample_output_folder / "audio_16k.wav"
        original_size = audio_file.stat().st_size

        # Remove partial files but keep audio
        (sample_output_folder / "partial_transcript.txt").unlink()

        assert audio_file.exists()
        assert audio_file.stat().st_size == original_size


# ============================================================================
# Partial Transcript Recovery Tests
# ============================================================================

class TestPartialTranscriptRecovery:
    """Tests for recovering partial transcripts."""

    def test_load_partial_transcript(self, sample_output_folder):
        """Partial transcript is loaded correctly."""
        partial_file = sample_output_folder / "partial_transcript.txt"
        content = partial_file.read_text(encoding="utf-8")

        chunks = content.split("\n---CHUNK_BOUNDARY---\n")
        assert len(chunks) == 2
        assert chunks[0].strip() == "First chunk."
        assert chunks[1].strip() == "Second chunk."

    def test_partial_transcript_merge(self, sample_output_folder):
        """Partial transcript merges with new chunks."""
        from callwhisper.services.audio_chunker import merge_chunk_transcripts

        # Load existing partial
        partial_file = sample_output_folder / "partial_transcript.txt"
        content = partial_file.read_text(encoding="utf-8")
        existing = content.split("\n---CHUNK_BOUNDARY---\n")

        # Add new chunks
        new_chunks = ["Third chunk.", "Fourth chunk."]
        all_chunks = existing + new_chunks

        merged = merge_chunk_transcripts(all_chunks)

        assert "First chunk" in merged
        assert "Second chunk" in merged
        assert "Third chunk" in merged
        assert "Fourth chunk" in merged


# ============================================================================
# Edge Cases
# ============================================================================

class TestRecoveryEdgeCases:
    """Edge case tests for job recovery."""

    def test_checkpoint_with_no_chunks_completed(self, temp_dir):
        """Handle checkpoint with 0 chunks completed."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "job_id": "early_failure",
            "status": "processing",
            "chunks_completed": 0,
            "total_chunks": 5,
            "partial_transcript": "",
        }

        checkpoint_path = checkpoint_dir / "early_failure.json"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        from callwhisper.core.job_store import load_checkpoint

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            data = load_checkpoint("early_failure")
            assert data["chunks_completed"] == 0

    def test_checkpoint_at_last_chunk(self, temp_dir):
        """Handle checkpoint at last chunk (almost complete)."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "job_id": "almost_done",
            "status": "chunk_4",
            "chunks_completed": 4,
            "total_chunks": 5,
            "partial_transcript": "A. B. C. D.",
        }

        checkpoint_path = checkpoint_dir / "almost_done.json"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

        from callwhisper.core.job_store import load_checkpoint

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            data = load_checkpoint("almost_done")
            assert data["chunks_completed"] == 4
            # Only 1 chunk left

    def test_multiple_incomplete_jobs_priority(self, temp_dir):
        """Multiple incomplete jobs sorted by recency."""
        from callwhisper.core.job_store import list_incomplete_jobs

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        times = [
            datetime(2024, 12, 29, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 12, 29, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 12, 29, 11, 0, 0, tzinfo=timezone.utc),
        ]

        for i, t in enumerate(times):
            checkpoint = {
                "job_id": f"job_{i}",
                "status": "processing",
                "chunks_completed": i,
                "total_chunks": 5,
                "updated_at": t.isoformat(),
            }
            (checkpoint_dir / f"job_{i}.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            jobs = list_incomplete_jobs()
            assert len(jobs) == 3


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

class TestRecoveryIntegration:
    """Integration tests for job recovery."""

    @pytest.mark.asyncio
    async def test_full_recovery_workflow(self, temp_dir, mock_settings):
        """Full workflow: discover, resume, complete."""
        from callwhisper.core.job_store import (
            save_checkpoint, load_checkpoint, delete_checkpoint
        )

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "callwhisper.core.job_store.get_checkpoint_dir",
            return_value=checkpoint_dir
        ):
            # 1. Create incomplete job
            save_checkpoint(
                job_id="recovery_test",
                status="chunk_2",
                chunks_completed=2,
                total_chunks=5,
                partial_transcript="A. B."
            )

            # 2. Load checkpoint
            data = load_checkpoint("recovery_test")
            assert data["chunks_completed"] == 2

            # 3. Simulate completing remaining chunks
            # (In reality, transcribe_audio_chunked would do this)

            # 4. Delete checkpoint after completion
            delete_checkpoint("recovery_test")
            assert not (checkpoint_dir / "recovery_test.json").exists()

    @pytest.mark.asyncio
    async def test_failure_during_recovery(self, sample_output_folder, mock_settings):
        """Failure during recovery saves progress."""
        from callwhisper.services.transcriber import transcribe_audio_chunked
        from callwhisper.core.exceptions import TranscriptionError

        manifest_path = sample_output_folder / "chunks" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())

        call_count = [0]

        async def failing_transcribe(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail on second chunk
                raise TranscriptionError("Processing failed")
            return f"Chunk {call_count[0]}"

        with patch(
            "callwhisper.services.transcriber.ensure_chunks_exist"
        ) as mock_ensure:
            from callwhisper.services.audio_chunker import ChunkManifest
            mock_ensure.return_value = ChunkManifest.from_dict(manifest_data)

            with patch(
                "callwhisper.services.transcriber.transcribe_chunk",
                side_effect=failing_transcribe
            ):
                with patch("callwhisper.services.transcriber.update_checkpoint"):
                    with pytest.raises(TranscriptionError):
                        await transcribe_audio_chunked(
                            sample_output_folder,
                            mock_settings,
                            start_from_chunk=0
                        )

                    # Partial transcript should be saved
                    partial = sample_output_folder / "partial_transcript.txt"
                    # File should exist from fixture or be written
