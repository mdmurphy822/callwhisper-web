"""
Tests for job queue service.

Tests batch transcription queue functionality:
- Job creation and management
- Priority sorting
- Worker processing loop
- Status tracking
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from callwhisper.services.job_queue import (
    JobQueue,
    QueuedJob,
    get_job_queue,
    reset_job_queue,
)


class TestQueuedJob:
    """Tests for QueuedJob dataclass."""

    def test_creates_job_with_defaults(self, tmp_path):
        """Verify job is created with correct defaults."""
        job = QueuedJob(
            job_id="test123",
            audio_path=tmp_path / "test.wav",
            original_filename="test.wav",
        )

        assert job.job_id == "test123"
        assert job.status == "queued"
        assert job.priority == 0
        assert job.error_message is None
        assert job.progress == 0
        assert job.started_at is None
        assert job.completed_at is None

    def test_to_dict_serialization(self, tmp_path):
        """Verify job serializes to dictionary correctly."""
        job = QueuedJob(
            job_id="test123",
            audio_path=tmp_path / "test.wav",
            original_filename="test.wav",
            ticket_id="TICKET-001",
            priority=5,
        )

        result = job.to_dict()

        assert result["job_id"] == "test123"
        assert result["original_filename"] == "test.wav"
        assert result["ticket_id"] == "TICKET-001"
        assert result["priority"] == 5
        assert result["status"] == "queued"


class TestJobQueueAddJob:
    """Tests for adding jobs to queue."""

    @pytest.fixture
    def queue(self):
        """Create fresh job queue."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_add_job_returns_job_id(self, queue, tmp_path):
        """Verify add_job returns a job ID."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        assert job_id is not None
        assert len(job_id) == 8  # UUID[:8]

    @pytest.mark.asyncio
    async def test_add_job_creates_queued_job(self, queue, tmp_path):
        """Verify job is added to queue."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
            ticket_id="TICKET-001",
        )

        status = queue.get_status()
        assert status["counts"]["queued"] == 1
        assert status["queued"][0]["job_id"] == job_id
        assert status["queued"][0]["ticket_id"] == "TICKET-001"

    @pytest.mark.asyncio
    async def test_add_job_with_priority_sorting(self, queue, tmp_path):
        """Verify jobs are sorted by priority."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        # Add low priority first
        low_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="low.wav",
            priority=1,
        )

        # Add high priority second
        high_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="high.wav",
            priority=10,
        )

        status = queue.get_status()
        # High priority should be first
        assert status["queued"][0]["job_id"] == high_id
        assert status["queued"][1]["job_id"] == low_id


class TestJobQueueProcessing:
    """Tests for job processing methods."""

    @pytest.fixture
    def queue(self):
        """Create fresh job queue."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_get_next_job(self, queue, tmp_path):
        """Verify get_next_job returns first queued job."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        next_job = await queue.get_next_job()

        assert next_job is not None
        assert next_job.job_id == job_id

    @pytest.mark.asyncio
    async def test_get_next_job_returns_none_when_empty(self, queue):
        """Verify get_next_job returns None for empty queue."""
        next_job = await queue.get_next_job()
        assert next_job is None

    @pytest.mark.asyncio
    async def test_start_processing(self, queue, tmp_path):
        """Verify start_processing moves job from queue."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        job = await queue.start_processing(job_id)

        assert job is not None
        assert job.status == "processing"
        assert job.started_at is not None

        status = queue.get_status()
        assert status["counts"]["queued"] == 0
        assert status["counts"]["processing"] == 1

    @pytest.mark.asyncio
    async def test_complete_job_success(self, queue, tmp_path):
        """Verify completing job marks it as complete."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )
        await queue.start_processing(job_id)

        await queue.complete_job(job_id, success=True)

        status = queue.get_status()
        assert status["counts"]["processing"] == 0
        assert status["counts"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_complete_job_failure(self, queue, tmp_path):
        """Verify failing job marks it with error."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )
        await queue.start_processing(job_id)

        await queue.complete_job(job_id, success=False, error="Test error")

        status = queue.get_status()
        assert status["counts"]["processing"] == 0
        assert status["counts"]["failed"] == 1
        assert status["failed"][0]["error_message"] == "Test error"


class TestJobQueueCancel:
    """Tests for job cancellation."""

    @pytest.fixture
    def queue(self):
        """Create fresh job queue."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, queue, tmp_path):
        """Verify cancelling removes job from queue."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        result = await queue.cancel_job(job_id)

        assert result is True
        status = queue.get_status()
        assert status["counts"]["queued"] == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, queue):
        """Verify cancelling nonexistent job returns False."""
        result = await queue.cancel_job("nonexistent")
        assert result is False


class TestJobQueueWorker:
    """Tests for background worker."""

    @pytest.fixture
    def queue(self):
        """Create fresh job queue."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_worker_processes_jobs(self, queue, tmp_path):
        """Verify worker processes queued jobs."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        processed_jobs = []

        async def mock_process(job):
            processed_jobs.append(job.job_id)

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        # Start worker
        await queue.start_worker(mock_process)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop worker
        await queue.stop_worker()

        assert job_id in processed_jobs
        status = queue.get_status()
        assert status["counts"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_worker_handles_errors(self, queue, tmp_path):
        """Verify worker handles processing errors gracefully."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        async def failing_process(job):
            raise RuntimeError("Processing failed")

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        await queue.start_worker(failing_process)
        await asyncio.sleep(0.2)
        await queue.stop_worker()

        status = queue.get_status()
        assert status["counts"]["failed"] == 1
        assert "Processing failed" in status["failed"][0]["error_message"]

    @pytest.mark.asyncio
    async def test_is_worker_running(self, queue):
        """Verify is_worker_running returns correct state."""
        async def mock_process(job):
            await asyncio.sleep(1)

        assert queue.is_worker_running() is False

        await queue.start_worker(mock_process)
        assert queue.is_worker_running() is True

        await queue.stop_worker()
        assert queue.is_worker_running() is False


class TestJobQueueStatus:
    """Tests for status reporting."""

    @pytest.fixture
    def queue(self):
        """Create fresh job queue."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_get_status_structure(self, queue):
        """Verify status has correct structure."""
        status = queue.get_status()

        assert "queued" in status
        assert "processing" in status
        assert "completed" in status
        assert "failed" in status
        assert "counts" in status
        assert all(
            key in status["counts"]
            for key in ["queued", "processing", "completed", "failed"]
        )

    @pytest.mark.asyncio
    async def test_status_callback(self, queue, tmp_path):
        """Verify status callback is invoked on changes."""
        audio_path = tmp_path / "test.wav"
        audio_path.touch()

        callback_count = 0

        async def status_callback(status):
            nonlocal callback_count
            callback_count += 1

        queue.set_status_callback(status_callback)

        await queue.add_job(
            audio_path=audio_path,
            original_filename="test.wav",
        )

        assert callback_count >= 1


class TestGlobalJobQueue:
    """Tests for global queue singleton."""

    def test_get_job_queue_returns_singleton(self):
        """Verify get_job_queue returns same instance."""
        reset_job_queue()

        queue1 = get_job_queue()
        queue2 = get_job_queue()

        assert queue1 is queue2

    def test_reset_job_queue(self):
        """Verify reset_job_queue clears singleton."""
        queue1 = get_job_queue()
        reset_job_queue()
        queue2 = get_job_queue()

        assert queue1 is not queue2
