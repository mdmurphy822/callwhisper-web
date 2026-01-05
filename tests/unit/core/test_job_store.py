"""
Tests for Job Store module.

Tests crash recovery checkpoint system:
- JobCheckpoint dataclass operations
- JobStore file I/O and state management
- Module-level convenience functions
"""

import pytest
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from callwhisper.core.job_store import (
    JobCheckpoint,
    JobStore,
    get_job_store,
    create_checkpoint,
    update_checkpoint,
)


class TestJobCheckpoint:
    """Tests for JobCheckpoint dataclass."""

    def test_to_dict_serialization(self):
        """Checkpoint serializes all fields to dict."""
        checkpoint = JobCheckpoint(
            job_id="test-123",
            audio_path="/path/to/audio.wav",
            status="processing",
            chunks_completed=5,
            total_chunks=10,
            partial_transcript="Hello world",
            error_message=None,
            device_name="VB-Cable",
            ticket_id="ticket-456",
        )
        result = checkpoint.to_dict()

        assert result["job_id"] == "test-123"
        assert result["audio_path"] == "/path/to/audio.wav"
        assert result["status"] == "processing"
        assert result["chunks_completed"] == 5
        assert result["total_chunks"] == 10
        assert result["partial_transcript"] == "Hello world"
        assert result["device_name"] == "VB-Cable"
        assert result["ticket_id"] == "ticket-456"
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict_deserialization(self):
        """Checkpoint deserializes from dict correctly."""
        data = {
            "job_id": "test-789",
            "audio_path": "/audio/file.wav",
            "status": "recording",
            "chunks_completed": 0,
            "total_chunks": 0,
            "partial_transcript": "",
            "error_message": None,
            "device_name": "Stereo Mix",
            "ticket_id": None,
            "created_at": 1000.0,
            "updated_at": 1001.0,
        }
        checkpoint = JobCheckpoint.from_dict(data)

        assert checkpoint.job_id == "test-789"
        assert checkpoint.audio_path == "/audio/file.wav"
        assert checkpoint.status == "recording"
        assert checkpoint.device_name == "Stereo Mix"
        assert checkpoint.created_at == 1000.0
        assert checkpoint.updated_at == 1001.0

    def test_is_complete_property_true(self):
        """is_complete returns True when status is complete."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="complete",
        )
        assert checkpoint.is_complete is True

    def test_is_complete_property_false(self):
        """is_complete returns False for other statuses."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="processing",
        )
        assert checkpoint.is_complete is False

    def test_is_failed_property_true(self):
        """is_failed returns True when status is failed."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="failed",
            error_message="Something went wrong",
        )
        assert checkpoint.is_failed is True

    def test_is_failed_property_false(self):
        """is_failed returns False for other statuses."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="complete",
        )
        assert checkpoint.is_failed is False

    def test_is_incomplete_property_true(self):
        """is_incomplete returns True for in-progress statuses."""
        for status in ["recording", "processing", "chunk_5"]:
            checkpoint = JobCheckpoint(
                job_id="job1",
                audio_path="/path",
                status=status,
            )
            assert checkpoint.is_incomplete is True, f"Failed for status: {status}"

    def test_is_incomplete_property_false(self):
        """is_incomplete returns False for terminal statuses."""
        for status in ["complete", "failed"]:
            checkpoint = JobCheckpoint(
                job_id="job1",
                audio_path="/path",
                status=status,
            )
            assert checkpoint.is_incomplete is False, f"Failed for status: {status}"

    def test_progress_percent_calculation(self):
        """progress_percent calculates percentage correctly."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="processing",
            chunks_completed=3,
            total_chunks=10,
        )
        assert checkpoint.progress_percent == 30.0

    def test_progress_percent_zero_chunks(self):
        """progress_percent handles division by zero."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="recording",
            chunks_completed=0,
            total_chunks=0,
        )
        assert checkpoint.progress_percent == 0.0

    def test_progress_percent_complete(self):
        """progress_percent returns 100 when all chunks done."""
        checkpoint = JobCheckpoint(
            job_id="job1",
            audio_path="/path",
            status="complete",
            chunks_completed=10,
            total_chunks=10,
        )
        assert checkpoint.progress_percent == 100.0


class TestJobStore:
    """Tests for JobStore class."""

    def test_init_creates_directories(self, tmp_path):
        """JobStore creates store and archive directories."""
        store_path = tmp_path / "jobs"
        store = JobStore(store_path=store_path)

        assert store_path.exists()
        assert (store_path / "archive").exists()

    def test_init_uses_default_path(self):
        """JobStore uses default path when none provided."""
        with patch.object(Path, "mkdir"):
            store = JobStore()
            assert ".callwhisper" in str(store.store_path)
            assert "jobs" in str(store.store_path)

    def test_save_checkpoint(self, tmp_path):
        """save_checkpoint writes JSON to disk."""
        store = JobStore(store_path=tmp_path)
        checkpoint = JobCheckpoint(
            job_id="save-test",
            audio_path="/audio.wav",
            status="recording",
        )

        store.save_checkpoint(checkpoint)

        saved_path = tmp_path / "save-test.json"
        assert saved_path.exists()

        with open(saved_path) as f:
            data = json.load(f)
        assert data["job_id"] == "save-test"
        assert data["status"] == "recording"

    def test_save_checkpoint_updates_timestamp(self, tmp_path):
        """save_checkpoint updates updated_at timestamp."""
        store = JobStore(store_path=tmp_path)
        checkpoint = JobCheckpoint(
            job_id="timestamp-test",
            audio_path="/audio.wav",
            status="recording",
            updated_at=1000.0,
        )

        before = time.time()
        store.save_checkpoint(checkpoint)
        after = time.time()

        assert checkpoint.updated_at >= before
        assert checkpoint.updated_at <= after

    def test_load_checkpoint_exists(self, tmp_path):
        """load_checkpoint loads existing checkpoint."""
        store = JobStore(store_path=tmp_path)
        data = {
            "job_id": "load-test",
            "audio_path": "/audio.wav",
            "status": "processing",
            "chunks_completed": 2,
            "total_chunks": 5,
            "partial_transcript": "test",
            "error_message": None,
            "device_name": None,
            "ticket_id": None,
            "created_at": 1000.0,
            "updated_at": 1001.0,
        }
        (tmp_path / "load-test.json").write_text(json.dumps(data))

        checkpoint = store.load_checkpoint("load-test")

        assert checkpoint is not None
        assert checkpoint.job_id == "load-test"
        assert checkpoint.status == "processing"
        assert checkpoint.chunks_completed == 2

    def test_load_checkpoint_not_exists(self, tmp_path):
        """load_checkpoint returns None when file missing."""
        store = JobStore(store_path=tmp_path)

        checkpoint = store.load_checkpoint("nonexistent")

        assert checkpoint is None

    def test_load_checkpoint_corrupted(self, tmp_path):
        """load_checkpoint returns None for malformed JSON."""
        store = JobStore(store_path=tmp_path)
        (tmp_path / "corrupted.json").write_text("not valid json {{{")

        checkpoint = store.load_checkpoint("corrupted")

        assert checkpoint is None

    def test_get_incomplete_jobs(self, tmp_path):
        """get_incomplete_jobs finds only incomplete jobs."""
        store = JobStore(store_path=tmp_path)

        # Create various checkpoints
        for job_id, status in [
            ("job1", "recording"),
            ("job2", "processing"),
            ("job3", "complete"),
            ("job4", "failed"),
            ("job5", "chunk_3"),
        ]:
            checkpoint = JobCheckpoint(
                job_id=job_id,
                audio_path=f"/{job_id}.wav",
                status=status,
            )
            store.save_checkpoint(checkpoint)

        incomplete = store.get_incomplete_jobs()

        job_ids = [c.job_id for c in incomplete]
        assert "job1" in job_ids  # recording
        assert "job2" in job_ids  # processing
        assert "job5" in job_ids  # chunk_3
        assert "job3" not in job_ids  # complete
        assert "job4" not in job_ids  # failed

    def test_get_incomplete_jobs_sorted(self, tmp_path):
        """get_incomplete_jobs returns jobs sorted by created_at desc."""
        store = JobStore(store_path=tmp_path)

        for i, job_id in enumerate(["old", "middle", "new"]):
            checkpoint = JobCheckpoint(
                job_id=job_id,
                audio_path=f"/{job_id}.wav",
                status="processing",
                created_at=1000.0 + i * 100,  # old=1000, middle=1100, new=1200
            )
            store.save_checkpoint(checkpoint)

        incomplete = store.get_incomplete_jobs()

        assert len(incomplete) == 3
        assert incomplete[0].job_id == "new"
        assert incomplete[1].job_id == "middle"
        assert incomplete[2].job_id == "old"

    def test_mark_complete(self, tmp_path):
        """mark_complete updates status and moves to archive."""
        store = JobStore(store_path=tmp_path)
        checkpoint = JobCheckpoint(
            job_id="complete-test",
            audio_path="/audio.wav",
            status="processing",
        )
        store.save_checkpoint(checkpoint)

        store.mark_complete("complete-test")

        # Original should be gone
        assert not (tmp_path / "complete-test.json").exists()
        # Should be in archive
        archive_path = tmp_path / "archive" / "complete-test.json"
        assert archive_path.exists()

        with open(archive_path) as f:
            data = json.load(f)
        assert data["status"] == "complete"

    def test_mark_complete_not_found(self, tmp_path):
        """mark_complete handles missing checkpoint gracefully."""
        store = JobStore(store_path=tmp_path)

        # Should not raise
        store.mark_complete("nonexistent")

    def test_mark_failed(self, tmp_path):
        """mark_failed sets status and error message."""
        store = JobStore(store_path=tmp_path)
        checkpoint = JobCheckpoint(
            job_id="fail-test",
            audio_path="/audio.wav",
            status="processing",
        )
        store.save_checkpoint(checkpoint)

        store.mark_failed("fail-test", "Out of memory")

        loaded = store.load_checkpoint("fail-test")
        assert loaded.status == "failed"
        assert loaded.error_message == "Out of memory"

    def test_delete_checkpoint(self, tmp_path):
        """delete_checkpoint removes file and returns True."""
        store = JobStore(store_path=tmp_path)
        checkpoint = JobCheckpoint(
            job_id="delete-test",
            audio_path="/audio.wav",
            status="processing",
        )
        store.save_checkpoint(checkpoint)

        result = store.delete_checkpoint("delete-test")

        assert result is True
        assert not (tmp_path / "delete-test.json").exists()

    def test_delete_checkpoint_not_found(self, tmp_path):
        """delete_checkpoint returns False when file missing."""
        store = JobStore(store_path=tmp_path)

        result = store.delete_checkpoint("nonexistent")

        assert result is False

    def test_cleanup_old_checkpoints(self, tmp_path):
        """cleanup_old_checkpoints removes old archived files."""
        store = JobStore(store_path=tmp_path)
        archive_path = tmp_path / "archive"

        # Create old and new archived checkpoints
        old_time = time.time() - (10 * 24 * 60 * 60)  # 10 days ago
        new_time = time.time() - (1 * 24 * 60 * 60)  # 1 day ago

        for job_id, updated_at in [("old-job", old_time), ("new-job", new_time)]:
            data = {
                "job_id": job_id,
                "audio_path": f"/{job_id}.wav",
                "status": "complete",
                "chunks_completed": 10,
                "total_chunks": 10,
                "partial_transcript": "",
                "error_message": None,
                "device_name": None,
                "ticket_id": None,
                "created_at": updated_at,
                "updated_at": updated_at,
            }
            (archive_path / f"{job_id}.json").write_text(json.dumps(data))

        removed = store.cleanup_old_checkpoints(max_age_days=7)

        assert removed == 1
        assert not (archive_path / "old-job.json").exists()
        assert (archive_path / "new-job.json").exists()

    def test_get_job_history(self, tmp_path):
        """get_job_history returns archived jobs sorted by updated_at."""
        store = JobStore(store_path=tmp_path)
        archive_path = tmp_path / "archive"

        for i, job_id in enumerate(["first", "second", "third"]):
            data = {
                "job_id": job_id,
                "audio_path": f"/{job_id}.wav",
                "status": "complete",
                "chunks_completed": 10,
                "total_chunks": 10,
                "partial_transcript": "",
                "error_message": None,
                "device_name": None,
                "ticket_id": None,
                "created_at": 1000.0 + i,
                "updated_at": 1000.0 + i,
            }
            (archive_path / f"{job_id}.json").write_text(json.dumps(data))

        history = store.get_job_history(limit=10)

        assert len(history) == 3
        assert history[0].job_id == "third"  # Most recent
        assert history[2].job_id == "first"  # Oldest

    def test_get_job_history_limit(self, tmp_path):
        """get_job_history respects limit parameter."""
        store = JobStore(store_path=tmp_path)
        archive_path = tmp_path / "archive"

        for i in range(10):
            data = {
                "job_id": f"job-{i}",
                "audio_path": f"/job-{i}.wav",
                "status": "complete",
                "chunks_completed": 10,
                "total_chunks": 10,
                "partial_transcript": "",
                "error_message": None,
                "device_name": None,
                "ticket_id": None,
                "created_at": 1000.0 + i,
                "updated_at": 1000.0 + i,
            }
            (archive_path / f"job-{i}.json").write_text(json.dumps(data))

        history = store.get_job_history(limit=3)

        assert len(history) == 3


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_job_store_singleton(self, tmp_path):
        """get_job_store returns singleton instance."""
        import callwhisper.core.job_store as job_store_module

        # Reset singleton
        job_store_module._job_store = None

        with patch.object(Path, "home", return_value=tmp_path):
            store1 = get_job_store()
            store2 = get_job_store()

        assert store1 is store2

        # Clean up
        job_store_module._job_store = None

    def test_create_checkpoint(self, tmp_path):
        """create_checkpoint creates and saves new checkpoint."""
        import callwhisper.core.job_store as job_store_module

        # Use custom store
        test_store = JobStore(store_path=tmp_path)
        job_store_module._job_store = test_store

        try:
            checkpoint = create_checkpoint(
                job_id="new-job",
                audio_path="/path/to/audio.wav",
                device_name="VB-Cable",
                ticket_id="ticket-123",
            )

            assert checkpoint.job_id == "new-job"
            assert checkpoint.status == "recording"
            assert checkpoint.device_name == "VB-Cable"

            # Verify saved to disk
            assert (tmp_path / "new-job.json").exists()
        finally:
            job_store_module._job_store = None

    def test_update_checkpoint(self, tmp_path):
        """update_checkpoint updates existing checkpoint fields."""
        import callwhisper.core.job_store as job_store_module

        test_store = JobStore(store_path=tmp_path)
        job_store_module._job_store = test_store

        try:
            # Create initial checkpoint
            initial = JobCheckpoint(
                job_id="update-test",
                audio_path="/audio.wav",
                status="recording",
            )
            test_store.save_checkpoint(initial)

            # Update it
            updated = update_checkpoint(
                job_id="update-test",
                status="processing",
                chunks_completed=5,
                total_chunks=10,
                partial_transcript="Hello",
            )

            assert updated is not None
            assert updated.status == "processing"
            assert updated.chunks_completed == 5
            assert updated.total_chunks == 10
            assert updated.partial_transcript == "Hello"
        finally:
            job_store_module._job_store = None

    def test_update_checkpoint_not_found(self, tmp_path):
        """update_checkpoint returns None when checkpoint missing."""
        import callwhisper.core.job_store as job_store_module

        test_store = JobStore(store_path=tmp_path)
        job_store_module._job_store = test_store

        try:
            result = update_checkpoint(
                job_id="nonexistent",
                status="processing",
            )

            assert result is None
        finally:
            job_store_module._job_store = None

    def test_update_checkpoint_partial_update(self, tmp_path):
        """update_checkpoint only updates provided fields."""
        import callwhisper.core.job_store as job_store_module

        test_store = JobStore(store_path=tmp_path)
        job_store_module._job_store = test_store

        try:
            # Create initial checkpoint
            initial = JobCheckpoint(
                job_id="partial-test",
                audio_path="/audio.wav",
                status="recording",
                partial_transcript="Original",
            )
            test_store.save_checkpoint(initial)

            # Update only chunks_completed
            updated = update_checkpoint(
                job_id="partial-test",
                chunks_completed=3,
            )

            assert updated is not None
            assert updated.status == "recording"  # Unchanged
            assert updated.partial_transcript == "Original"  # Unchanged
            assert updated.chunks_completed == 3  # Updated
        finally:
            job_store_module._job_store = None
