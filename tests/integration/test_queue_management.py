"""
Integration tests for job queue management.

Tests the job queue system including:
- FIFO queue processing
- Priority-based ordering
- Job lifecycle (queued -> processing -> complete/failed)
- Job cancellation
- Worker start/stop
- Queue status endpoint
- Folder scanning and import
- WebSocket queue notifications
- Concurrent job operations
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from callwhisper.services.job_queue import (
    JobQueue,
    QueuedJob,
    get_job_queue,
    reset_job_queue,
)
from callwhisper.services.folder_scanner import (
    scan_folder,
    scan_folder_paths,
    get_folder_stats,
    is_audio_file,
    ScannedFile,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
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
def job_queue():
    """Create a fresh job queue for testing."""
    reset_job_queue()
    queue = JobQueue()
    yield queue


@pytest.fixture
def sample_audio_files(temp_dir):
    """Create sample audio files for testing."""
    files = []
    for ext in ['.wav', '.mp3', '.m4a', '.ogg']:
        path = temp_dir / f"audio_{ext[1:]}{ext}"
        path.write_bytes(b'\x00' * 1024)  # 1KB dummy file
        files.append(path)
    return files


@pytest.fixture
def sample_folder_with_audio(temp_dir):
    """Create a folder structure with audio files."""
    # Create main folder files
    (temp_dir / "song1.mp3").write_bytes(b'\x00' * 2048)
    (temp_dir / "song2.wav").write_bytes(b'\x00' * 4096)
    (temp_dir / "song3.m4a").write_bytes(b'\x00' * 1024)

    # Create a non-audio file
    (temp_dir / "readme.txt").write_text("Not an audio file")

    # Create subfolder with files
    subfolder = temp_dir / "subfolder"
    subfolder.mkdir()
    (subfolder / "nested.wav").write_bytes(b'\x00' * 1024)
    (subfolder / "nested.mp3").write_bytes(b'\x00' * 2048)

    return temp_dir


# ============================================================================
# QueuedJob Tests
# ============================================================================

class TestQueuedJob:
    """Tests for QueuedJob dataclass."""

    def test_job_creation_defaults(self, temp_dir):
        """Job has correct default values."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b'\x00' * 100)

        job = QueuedJob(
            job_id="test123",
            audio_path=audio_path,
            original_filename="test.wav"
        )

        assert job.job_id == "test123"
        assert job.status == "queued"
        assert job.priority == 0
        assert job.progress == 0
        assert job.error_message is None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.created_at is not None

    def test_job_to_dict(self, temp_dir):
        """Job serialization to dict works."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b'\x00' * 100)

        job = QueuedJob(
            job_id="test123",
            audio_path=audio_path,
            original_filename="test.wav",
            ticket_id="TICKET-001",
            priority=5
        )

        data = job.to_dict()

        assert data["job_id"] == "test123"
        assert data["original_filename"] == "test.wav"
        assert data["ticket_id"] == "TICKET-001"
        assert data["priority"] == 5
        assert data["status"] == "queued"

    def test_job_with_ticket_id(self, temp_dir):
        """Job stores ticket ID correctly."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b'\x00' * 100)

        job = QueuedJob(
            job_id="abc",
            audio_path=audio_path,
            original_filename="test.wav",
            ticket_id="CALL-12345"
        )

        assert job.ticket_id == "CALL-12345"


# ============================================================================
# JobQueue Tests
# ============================================================================

class TestJobQueueBasicOperations:
    """Tests for basic job queue operations."""

    @pytest.mark.asyncio
    async def test_add_job_returns_id(self, job_queue, temp_dir):
        """Adding a job returns a unique ID."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav"
        )

        assert job_id is not None
        assert len(job_id) == 8  # UUID first 8 chars

    @pytest.mark.asyncio
    async def test_add_multiple_jobs(self, job_queue, temp_dir):
        """Multiple jobs can be added."""
        job_ids = []
        for i in range(5):
            audio_path = temp_dir / f"test_{i}.wav"
            audio_path.write_bytes(b'\x00' * 100)
            job_id = await job_queue.add_job(
                audio_path=audio_path,
                original_filename=f"test_{i}.wav"
            )
            job_ids.append(job_id)

        assert len(set(job_ids)) == 5  # All unique
        status = job_queue.get_status()
        assert status["counts"]["queued"] == 5

    @pytest.mark.asyncio
    async def test_get_next_job_fifo(self, job_queue, temp_dir):
        """Jobs retrieved in FIFO order (same priority)."""
        for i in range(3):
            audio_path = temp_dir / f"test_{i}.wav"
            audio_path.write_bytes(b'\x00' * 100)
            await job_queue.add_job(
                audio_path=audio_path,
                original_filename=f"test_{i}.wav"
            )
            await asyncio.sleep(0.01)  # Ensure different timestamps

        next_job = await job_queue.get_next_job()
        assert next_job.original_filename == "test_0.wav"

    @pytest.mark.asyncio
    async def test_get_next_job_empty_queue(self, job_queue):
        """Getting next job from empty queue returns None."""
        next_job = await job_queue.get_next_job()
        assert next_job is None

    @pytest.mark.asyncio
    async def test_get_job_by_id(self, job_queue, temp_dir):
        """Retrieve job by ID."""
        audio_path = temp_dir / "test.wav"
        audio_path.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav"
        )

        job = job_queue.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, job_queue):
        """Getting nonexistent job returns None."""
        job = job_queue.get_job("nonexistent")
        assert job is None


class TestJobQueuePriority:
    """Tests for priority-based queue ordering."""

    @pytest.mark.asyncio
    async def test_higher_priority_processed_first(self, job_queue, temp_dir):
        """Higher priority jobs are processed first."""
        # Add low priority job first
        audio1 = temp_dir / "low.wav"
        audio1.write_bytes(b'\x00' * 100)
        await job_queue.add_job(
            audio_path=audio1,
            original_filename="low.wav",
            priority=1
        )

        # Add high priority job second
        audio2 = temp_dir / "high.wav"
        audio2.write_bytes(b'\x00' * 100)
        await job_queue.add_job(
            audio_path=audio2,
            original_filename="high.wav",
            priority=10
        )

        next_job = await job_queue.get_next_job()
        assert next_job.original_filename == "high.wav"

    @pytest.mark.asyncio
    async def test_same_priority_fifo(self, job_queue, temp_dir):
        """Same priority jobs follow FIFO."""
        for name in ["first", "second", "third"]:
            audio = temp_dir / f"{name}.wav"
            audio.write_bytes(b'\x00' * 100)
            await job_queue.add_job(
                audio_path=audio,
                original_filename=f"{name}.wav",
                priority=5
            )
            await asyncio.sleep(0.01)  # Ensure order

        next_job = await job_queue.get_next_job()
        assert next_job.original_filename == "first.wav"


class TestJobQueueLifecycle:
    """Tests for job lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_start_processing(self, job_queue, temp_dir):
        """Job transitions to processing state."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        job = await job_queue.start_processing(job_id)

        assert job is not None
        assert job.status == "processing"
        assert job.started_at is not None
        assert job_queue._processing == job
        assert job_queue.get_status()["counts"]["queued"] == 0

    @pytest.mark.asyncio
    async def test_start_processing_removes_from_queue(self, job_queue, temp_dir):
        """Starting processing removes job from queue."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_processing(job_id)

        status = job_queue.get_status()
        assert status["counts"]["queued"] == 0
        assert status["processing"] is not None

    @pytest.mark.asyncio
    async def test_complete_job_success(self, job_queue, temp_dir):
        """Job completion marks as complete."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_processing(job_id)
        await job_queue.complete_job(job_id, success=True)

        status = job_queue.get_status()
        assert status["processing"] is None
        assert status["counts"]["completed"] == 1

        job = job_queue.get_job(job_id)
        assert job.status == "complete"
        assert job.progress == 100
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_job_failure(self, job_queue, temp_dir):
        """Job failure marks as failed with error."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_processing(job_id)
        await job_queue.complete_job(job_id, success=False, error="Transcription failed")

        status = job_queue.get_status()
        assert status["processing"] is None
        assert status["counts"]["failed"] == 1

        job = job_queue.get_job(job_id)
        assert job.status == "failed"
        assert job.error_message == "Transcription failed"

    @pytest.mark.asyncio
    async def test_update_progress(self, job_queue, temp_dir):
        """Progress updates work during processing."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_processing(job_id)
        await job_queue.update_progress(job_id, 50)

        job = job_queue.get_job(job_id)
        assert job.progress == 50

    @pytest.mark.asyncio
    async def test_progress_clamped_to_range(self, job_queue, temp_dir):
        """Progress is clamped to 0-100 range."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_processing(job_id)

        await job_queue.update_progress(job_id, 150)
        job = job_queue.get_job(job_id)
        assert job.progress == 100

        await job_queue.update_progress(job_id, -50)
        job = job_queue.get_job(job_id)
        assert job.progress == 0


class TestJobQueueCancellation:
    """Tests for job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, job_queue, temp_dir):
        """Queued job can be cancelled."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        result = await job_queue.cancel_job(job_id)

        assert result is True
        assert job_queue.get_status()["counts"]["queued"] == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, job_queue):
        """Cancelling nonexistent job returns False."""
        result = await job_queue.cancel_job("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_cleans_up_file(self, job_queue, temp_dir):
        """Cancellation removes the temp audio file."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.cancel_job(job_id)

        assert not audio.exists()

    @pytest.mark.asyncio
    async def test_clear_completed_jobs(self, job_queue, temp_dir):
        """Clearing removes completed and failed jobs."""
        # Create and complete a job
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )
        await job_queue.start_processing(job_id)
        await job_queue.complete_job(job_id, success=True)

        status = job_queue.get_status()
        assert status["counts"]["completed"] == 1

        await job_queue.clear_completed()

        status = job_queue.get_status()
        assert status["counts"]["completed"] == 0
        assert status["counts"]["failed"] == 0


class TestJobQueueStatus:
    """Tests for queue status reporting."""

    @pytest.mark.asyncio
    async def test_get_status_structure(self, job_queue):
        """Status has correct structure."""
        status = job_queue.get_status()

        assert "queued" in status
        assert "processing" in status
        assert "completed" in status
        assert "failed" in status
        assert "counts" in status
        assert isinstance(status["queued"], list)
        assert isinstance(status["counts"], dict)

    @pytest.mark.asyncio
    async def test_status_counts_accurate(self, job_queue, temp_dir):
        """Status counts are accurate."""
        for i in range(3):
            audio = temp_dir / f"test_{i}.wav"
            audio.write_bytes(b'\x00' * 100)
            await job_queue.add_job(
                audio_path=audio,
                original_filename=f"test_{i}.wav"
            )

        status = job_queue.get_status()
        assert status["counts"]["queued"] == 3
        assert status["counts"]["processing"] == 0

    @pytest.mark.asyncio
    async def test_status_limits_history(self, job_queue, temp_dir):
        """Status limits completed/failed to last 10."""
        # Complete 15 jobs
        for i in range(15):
            audio = temp_dir / f"test_{i}.wav"
            audio.write_bytes(b'\x00' * 100)
            job_id = await job_queue.add_job(
                audio_path=audio,
                original_filename=f"test_{i}.wav"
            )
            await job_queue.start_processing(job_id)
            await job_queue.complete_job(job_id, success=True)

        status = job_queue.get_status()
        assert len(status["completed"]) == 10  # Limited to 10
        assert status["counts"]["completed"] == 15  # Total count accurate


class TestJobQueueWorker:
    """Tests for background worker."""

    @pytest.mark.asyncio
    async def test_start_worker(self, job_queue):
        """Worker starts successfully."""
        async def dummy_processor(job):
            pass

        await job_queue.start_worker(dummy_processor)

        assert job_queue.is_worker_running()

        await job_queue.stop_worker()

    @pytest.mark.asyncio
    async def test_stop_worker(self, job_queue):
        """Worker stops cleanly."""
        async def dummy_processor(job):
            pass

        await job_queue.start_worker(dummy_processor)
        await job_queue.stop_worker()

        assert not job_queue.is_worker_running()

    @pytest.mark.asyncio
    async def test_worker_processes_jobs(self, job_queue, temp_dir):
        """Worker processes queued jobs."""
        processed_jobs = []

        async def processor(job):
            processed_jobs.append(job.job_id)

        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_worker(processor)
        await asyncio.sleep(0.2)  # Let worker run
        await job_queue.stop_worker()

        assert job_id in processed_jobs

    @pytest.mark.asyncio
    async def test_worker_handles_failures(self, job_queue, temp_dir):
        """Worker handles processing failures."""
        async def failing_processor(job):
            raise ValueError("Processing failed")

        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        await job_queue.start_worker(failing_processor)
        await asyncio.sleep(0.2)
        await job_queue.stop_worker()

        job = job_queue.get_job(job_id)
        assert job.status == "failed"
        assert "Processing failed" in job.error_message


class TestJobQueueStatusCallback:
    """Tests for WebSocket status notifications."""

    @pytest.mark.asyncio
    async def test_callback_on_add_job(self, job_queue, temp_dir):
        """Callback triggered when job added."""
        callback_calls = []

        async def callback(status):
            callback_calls.append(status)

        job_queue.set_status_callback(callback)

        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )

        assert len(callback_calls) == 1
        assert callback_calls[0]["counts"]["queued"] == 1

    @pytest.mark.asyncio
    async def test_callback_on_complete(self, job_queue, temp_dir):
        """Callback triggered on job completion."""
        callback_calls = []

        async def callback(status):
            callback_calls.append(status)

        job_queue.set_status_callback(callback)

        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )
        await job_queue.start_processing(job_id)
        await job_queue.complete_job(job_id, success=True)

        # Should have: add, start_processing, complete
        assert len(callback_calls) >= 3

    @pytest.mark.asyncio
    async def test_callback_error_handled(self, job_queue, temp_dir):
        """Callback errors don't crash queue."""
        async def failing_callback(status):
            raise ValueError("Callback error")

        job_queue.set_status_callback(failing_callback)

        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        # Should not raise
        await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )


class TestJobQueueConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_add_jobs(self, job_queue, temp_dir):
        """Multiple concurrent add_job calls are safe."""
        async def add_job(i):
            audio = temp_dir / f"test_{i}.wav"
            audio.write_bytes(b'\x00' * 100)
            return await job_queue.add_job(
                audio_path=audio,
                original_filename=f"test_{i}.wav"
            )

        tasks = [add_job(i) for i in range(10)]
        job_ids = await asyncio.gather(*tasks)

        assert len(set(job_ids)) == 10
        assert job_queue.get_status()["counts"]["queued"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_progress_updates(self, job_queue, temp_dir):
        """Concurrent progress updates are safe."""
        audio = temp_dir / "test.wav"
        audio.write_bytes(b'\x00' * 100)

        job_id = await job_queue.add_job(
            audio_path=audio,
            original_filename="test.wav"
        )
        await job_queue.start_processing(job_id)

        async def update(progress):
            await job_queue.update_progress(job_id, progress)

        tasks = [update(i) for i in range(0, 101, 10)]
        await asyncio.gather(*tasks)

        # Should complete without errors


# ============================================================================
# Folder Scanner Tests
# ============================================================================

class TestFolderScannerBasics:
    """Basic tests for folder scanning."""

    def test_is_audio_file_supported(self):
        """Supported extensions are recognized."""
        for ext in SUPPORTED_EXTENSIONS:
            path = Path(f"test{ext}")
            assert is_audio_file(path)

    def test_is_audio_file_unsupported(self):
        """Unsupported extensions are rejected."""
        unsupported = ['.txt', '.pdf', '.exe', '.py', '.doc']
        for ext in unsupported:
            path = Path(f"test{ext}")
            assert not is_audio_file(path)

    def test_is_audio_file_case_insensitive(self):
        """Extension check is case-insensitive."""
        assert is_audio_file(Path("test.WAV"))
        assert is_audio_file(Path("test.Mp3"))
        assert is_audio_file(Path("test.M4A"))


class TestScanFolder:
    """Tests for scan_folder function."""

    def test_scan_finds_audio_files(self, sample_folder_with_audio):
        """Scan finds all audio files in folder."""
        files = scan_folder(sample_folder_with_audio)

        filenames = [f.filename for f in files]
        assert "song1.mp3" in filenames
        assert "song2.wav" in filenames
        assert "song3.m4a" in filenames
        assert "readme.txt" not in filenames

    def test_scan_nonrecursive_default(self, sample_folder_with_audio):
        """Default scan is not recursive."""
        files = scan_folder(sample_folder_with_audio)

        filenames = [f.filename for f in files]
        assert "nested.wav" not in filenames
        assert "nested.mp3" not in filenames

    def test_scan_recursive(self, sample_folder_with_audio):
        """Recursive scan finds files in subdirectories."""
        files = scan_folder(sample_folder_with_audio, recursive=True)

        filenames = [f.filename for f in files]
        assert "nested.wav" in filenames
        assert "nested.mp3" in filenames

    def test_scan_sorted_by_name(self, sample_folder_with_audio):
        """Results are sorted by filename."""
        files = scan_folder(sample_folder_with_audio)

        filenames = [f.filename for f in files]
        assert filenames == sorted(filenames, key=str.lower)

    def test_scan_nonexistent_folder(self, temp_dir):
        """Scanning nonexistent folder raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            scan_folder(temp_dir / "nonexistent")

    def test_scan_file_not_directory(self, temp_dir):
        """Scanning a file raises error."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Not a directory"):
            scan_folder(file_path)

    def test_scan_empty_folder(self, temp_dir):
        """Scanning empty folder returns empty list."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        files = scan_folder(empty_dir)
        assert files == []

    def test_scan_skips_large_files(self, temp_dir):
        """Files exceeding max size are skipped."""
        small_file = temp_dir / "small.wav"
        small_file.write_bytes(b'\x00' * 1024)

        # Don't actually create huge file, just test the logic
        files = scan_folder(temp_dir, max_size_bytes=512)

        filenames = [f.filename for f in files]
        assert "small.wav" not in filenames  # Too large

    def test_scan_skips_empty_files(self, temp_dir):
        """Zero-byte files are skipped."""
        empty_file = temp_dir / "empty.wav"
        empty_file.write_bytes(b'')

        nonempty_file = temp_dir / "nonempty.wav"
        nonempty_file.write_bytes(b'\x00' * 100)

        files = scan_folder(temp_dir)

        filenames = [f.filename for f in files]
        assert "empty.wav" not in filenames
        assert "nonempty.wav" in filenames


class TestScannedFile:
    """Tests for ScannedFile dataclass."""

    def test_size_mb_property(self):
        """Size in MB is calculated correctly."""
        scanned = ScannedFile(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=5 * 1024 * 1024,  # 5 MB
            modified_at=datetime.now().timestamp(),
            extension=".wav"
        )

        assert scanned.size_mb == pytest.approx(5.0)

    def test_size_mb_small_file(self):
        """Small files report fractional MB."""
        scanned = ScannedFile(
            path=Path("/test.wav"),
            filename="test.wav",
            size_bytes=512 * 1024,  # 0.5 MB
            modified_at=datetime.now().timestamp(),
            extension=".wav"
        )

        assert scanned.size_mb == pytest.approx(0.5)


class TestScanFolderPaths:
    """Tests for scan_folder_paths convenience function."""

    def test_returns_path_objects(self, sample_folder_with_audio):
        """Returns list of Path objects."""
        paths = scan_folder_paths(sample_folder_with_audio)

        assert all(isinstance(p, Path) for p in paths)

    def test_recursive_option_works(self, sample_folder_with_audio):
        """Recursive option is passed through."""
        paths_flat = scan_folder_paths(sample_folder_with_audio, recursive=False)
        paths_recursive = scan_folder_paths(sample_folder_with_audio, recursive=True)

        assert len(paths_recursive) > len(paths_flat)


class TestGetFolderStats:
    """Tests for get_folder_stats function."""

    def test_stats_total_files(self, sample_folder_with_audio):
        """Stats include total file count."""
        stats = get_folder_stats(sample_folder_with_audio)

        assert stats["total_files"] == 3  # 3 files in main folder

    def test_stats_total_size(self, sample_folder_with_audio):
        """Stats include total size."""
        stats = get_folder_stats(sample_folder_with_audio)

        assert stats["total_size_mb"] > 0

    def test_stats_by_extension(self, sample_folder_with_audio):
        """Stats break down by extension."""
        stats = get_folder_stats(sample_folder_with_audio)

        assert ".mp3" in stats["extensions"]
        assert ".wav" in stats["extensions"]
        assert ".m4a" in stats["extensions"]

    def test_stats_oldest_newest(self, sample_folder_with_audio):
        """Stats include oldest and newest files."""
        stats = get_folder_stats(sample_folder_with_audio)

        assert stats["oldest_file"] is not None
        assert stats["newest_file"] is not None
        assert "filename" in stats["oldest_file"]
        assert "modified_at" in stats["oldest_file"]

    def test_stats_empty_folder(self, temp_dir):
        """Empty folder stats are correct."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        stats = get_folder_stats(empty_dir)

        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["oldest_file"] is None
        assert stats["newest_file"] is None

    def test_stats_recursive(self, sample_folder_with_audio):
        """Recursive stats include subdirectories."""
        stats_flat = get_folder_stats(sample_folder_with_audio, recursive=False)
        stats_recursive = get_folder_stats(sample_folder_with_audio, recursive=True)

        assert stats_recursive["total_files"] > stats_flat["total_files"]


# ============================================================================
# Global Queue Singleton Tests
# ============================================================================

class TestGlobalQueueSingleton:
    """Tests for global job queue singleton."""

    def test_get_job_queue_creates_singleton(self):
        """get_job_queue creates and returns singleton."""
        reset_job_queue()

        queue1 = get_job_queue()
        queue2 = get_job_queue()

        assert queue1 is queue2

    def test_reset_job_queue(self):
        """reset_job_queue clears singleton."""
        queue1 = get_job_queue()
        reset_job_queue()
        queue2 = get_job_queue()

        assert queue1 is not queue2


# ============================================================================
# Integration Tests
# ============================================================================

class TestQueueFolderScannerIntegration:
    """Integration tests for queue + folder scanner."""

    @pytest.mark.asyncio
    async def test_scan_and_queue_all_files(self, job_queue, sample_folder_with_audio):
        """Scan folder and queue all files."""
        files = scan_folder(sample_folder_with_audio)

        job_ids = []
        for scanned_file in files:
            job_id = await job_queue.add_job(
                audio_path=scanned_file.path,
                original_filename=scanned_file.filename
            )
            job_ids.append(job_id)

        assert len(job_ids) == len(files)
        assert job_queue.get_status()["counts"]["queued"] == len(files)

    @pytest.mark.asyncio
    async def test_full_batch_workflow(self, job_queue, temp_dir):
        """Complete workflow: scan, queue, process, complete."""
        # Create test files
        for i in range(3):
            (temp_dir / f"audio_{i}.wav").write_bytes(b'\x00' * 1024)

        # Scan folder
        files = scan_folder(temp_dir)
        assert len(files) == 3

        # Queue all files
        for scanned_file in files:
            await job_queue.add_job(
                audio_path=scanned_file.path,
                original_filename=scanned_file.filename
            )

        # Process queue with mock processor
        processed = []

        async def processor(job):
            processed.append(job.original_filename)
            await asyncio.sleep(0.01)

        await job_queue.start_worker(processor)

        # Wait for processing
        for _ in range(50):  # Max 5 seconds
            await asyncio.sleep(0.1)
            if job_queue.get_status()["counts"]["queued"] == 0:
                break

        await job_queue.stop_worker()

        # Verify all processed
        assert len(processed) == 3
        assert job_queue.get_status()["counts"]["completed"] == 3
