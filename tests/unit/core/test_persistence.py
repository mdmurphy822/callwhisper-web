"""
Tests for State Persistence & Checkpointing Module.

Tests checkpoint management:
- Checkpoint dataclass operations
- CheckpointManager file I/O
- Session recovery and cleanup
"""

import pytest
import json
from pathlib import Path
from datetime import datetime, timedelta

from callwhisper.core.persistence import (
    Checkpoint,
    CheckpointStage,
    CheckpointManager,
)


class TestCheckpointStage:
    """Tests for CheckpointStage enum."""

    def test_stage_values(self):
        """All expected stages exist with correct values."""
        assert CheckpointStage.STARTED.value == "started"
        assert CheckpointStage.RECORDING.value == "recording"
        assert CheckpointStage.STOPPED.value == "stopped"
        assert CheckpointStage.NORMALIZING.value == "normalizing"
        assert CheckpointStage.TRANSCRIBING.value == "transcribing"
        assert CheckpointStage.BUNDLING.value == "bundling"
        assert CheckpointStage.COMPLETED.value == "completed"
        assert CheckpointStage.FAILED.value == "failed"


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_to_dict_serialization(self):
        """Checkpoint serializes to dict correctly."""
        checkpoint = Checkpoint(
            session_id="test-session",
            stage=CheckpointStage.RECORDING,
            timestamp="2024-01-01T12:00:00",
            device_name="VB-Cable",
            ticket_id="ticket-123",
        )

        result = checkpoint.to_dict()

        assert result["session_id"] == "test-session"
        assert result["stage"] == "recording"  # Converted to string value
        assert result["device_name"] == "VB-Cable"
        assert result["ticket_id"] == "ticket-123"

    def test_from_dict_deserialization(self):
        """Checkpoint deserializes from dict correctly."""
        data = {
            "session_id": "test-session",
            "stage": "transcribing",
            "timestamp": "2024-01-01T12:00:00",
            "device_name": "Stereo Mix",
            "ticket_id": None,
            "output_folder": "/output",
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {"key": "value"},
        }

        checkpoint = Checkpoint.from_dict(data)

        assert checkpoint.session_id == "test-session"
        assert checkpoint.stage == CheckpointStage.TRANSCRIBING
        assert checkpoint.device_name == "Stereo Mix"
        assert checkpoint.metadata == {"key": "value"}

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata defaults to empty dict if None."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.STARTED,
            timestamp="2024-01-01T12:00:00",
            metadata=None,
        )

        assert checkpoint.metadata == {}


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create CheckpointManager with temp directory."""
        return CheckpointManager(checkpoint_dir=tmp_path)

    def test_init_creates_directory(self, tmp_path):
        """CheckpointManager creates checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        assert not checkpoint_dir.exists()

        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        assert checkpoint_dir.exists()

    def test_save_checkpoint(self, manager, tmp_path):
        """save_checkpoint writes checkpoint to disk."""
        checkpoint = manager.save_checkpoint(
            session_id="save-test",
            stage=CheckpointStage.RECORDING,
            device_name="VB-Cable",
        )

        assert checkpoint.session_id == "save-test"
        assert checkpoint.stage == CheckpointStage.RECORDING

        path = tmp_path / "save-test.checkpoint.json"
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["session_id"] == "save-test"
        assert data["stage"] == "recording"

    def test_load_checkpoint_exists(self, manager, tmp_path):
        """load_checkpoint loads existing checkpoint."""
        data = {
            "session_id": "load-test",
            "stage": "transcribing",
            "timestamp": "2024-01-01T12:00:00",
            "device_name": "Test Device",
            "ticket_id": None,
            "output_folder": None,
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {},
        }
        (tmp_path / "load-test.checkpoint.json").write_text(json.dumps(data))

        checkpoint = manager.load_checkpoint("load-test")

        assert checkpoint is not None
        assert checkpoint.session_id == "load-test"
        assert checkpoint.stage == CheckpointStage.TRANSCRIBING

    def test_load_checkpoint_not_exists(self, manager):
        """load_checkpoint returns None when file missing."""
        checkpoint = manager.load_checkpoint("nonexistent")
        assert checkpoint is None

    def test_load_checkpoint_corrupted(self, manager, tmp_path):
        """load_checkpoint returns None for malformed JSON."""
        (tmp_path / "corrupted.checkpoint.json").write_text("not valid json")

        checkpoint = manager.load_checkpoint("corrupted")
        assert checkpoint is None

    def test_update_checkpoint_existing(self, manager):
        """update_checkpoint updates existing checkpoint."""
        manager.save_checkpoint(
            session_id="update-test",
            stage=CheckpointStage.RECORDING,
            device_name="Device1",
        )

        updated = manager.update_checkpoint(
            session_id="update-test",
            stage=CheckpointStage.TRANSCRIBING,
            audio_file="/path/to/audio.wav",
        )

        assert updated.stage == CheckpointStage.TRANSCRIBING
        assert updated.device_name == "Device1"  # Preserved
        assert updated.audio_file == "/path/to/audio.wav"

    def test_update_checkpoint_creates_if_missing(self, manager):
        """update_checkpoint creates checkpoint if not exists."""
        updated = manager.update_checkpoint(
            session_id="new-session",
            stage=CheckpointStage.NORMALIZING,
        )

        assert updated is not None
        assert updated.stage == CheckpointStage.NORMALIZING

    def test_get_incomplete_sessions(self, manager):
        """get_incomplete_sessions finds incomplete sessions."""
        # Create various checkpoints
        manager.save_checkpoint("recording", CheckpointStage.RECORDING)
        manager.save_checkpoint("transcribing", CheckpointStage.TRANSCRIBING)
        manager.save_checkpoint("completed", CheckpointStage.COMPLETED)
        manager.save_checkpoint("failed", CheckpointStage.FAILED)

        incomplete = manager.get_incomplete_sessions()

        session_ids = [c.session_id for c in incomplete]
        assert "recording" in session_ids
        assert "transcribing" in session_ids
        assert "completed" not in session_ids
        assert "failed" not in session_ids

    def test_clear_checkpoint(self, manager, tmp_path):
        """clear_checkpoint removes checkpoint file."""
        manager.save_checkpoint("clear-test", CheckpointStage.COMPLETED)
        path = tmp_path / "clear-test.checkpoint.json"
        assert path.exists()

        result = manager.clear_checkpoint("clear-test")

        assert result is True
        assert not path.exists()

    def test_clear_checkpoint_not_found(self, manager):
        """clear_checkpoint returns False when file missing."""
        result = manager.clear_checkpoint("nonexistent")
        assert result is False

    def test_mark_completed(self, manager):
        """mark_completed updates stage to COMPLETED."""
        manager.save_checkpoint("complete-test", CheckpointStage.BUNDLING)

        result = manager.mark_completed("complete-test", bundle_file="/output/bundle.vtb")

        assert result.stage == CheckpointStage.COMPLETED
        assert result.bundle_file == "/output/bundle.vtb"

    def test_mark_failed(self, manager):
        """mark_failed updates stage to FAILED with error."""
        manager.save_checkpoint("fail-test", CheckpointStage.TRANSCRIBING)

        result = manager.mark_failed("fail-test", "Out of memory")

        assert result.stage == CheckpointStage.FAILED
        assert result.error_message == "Out of memory"

    def test_get_resumable_stage_stopped(self, manager):
        """get_resumable_stage returns NORMALIZING for STOPPED."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.STOPPED,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result == CheckpointStage.NORMALIZING

    def test_get_resumable_stage_transcribing(self, manager):
        """get_resumable_stage returns TRANSCRIBING for TRANSCRIBING."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.TRANSCRIBING,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result == CheckpointStage.TRANSCRIBING

    def test_get_resumable_stage_not_resumable(self, manager):
        """get_resumable_stage returns None for non-resumable stages."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.RECORDING,  # Can't resume recording
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result is None

    def test_cleanup_old_checkpoints(self, manager, tmp_path):
        """cleanup_old_checkpoints removes old completed checkpoints."""
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        new_time = (datetime.now() - timedelta(hours=1)).isoformat()

        # Create old completed checkpoint
        old_data = {
            "session_id": "old-session",
            "stage": "completed",
            "timestamp": old_time,
            "device_name": None,
            "ticket_id": None,
            "output_folder": None,
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {},
        }
        (tmp_path / "old-session.checkpoint.json").write_text(json.dumps(old_data))

        # Create new completed checkpoint
        new_data = {
            "session_id": "new-session",
            "stage": "completed",
            "timestamp": new_time,
            "device_name": None,
            "ticket_id": None,
            "output_folder": None,
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {},
        }
        (tmp_path / "new-session.checkpoint.json").write_text(json.dumps(new_data))

        removed = manager.cleanup_old_checkpoints(max_age_hours=24)

        assert removed == 1
        assert not (tmp_path / "old-session.checkpoint.json").exists()
        assert (tmp_path / "new-session.checkpoint.json").exists()

    def test_cleanup_old_checkpoints_keeps_incomplete(self, manager, tmp_path):
        """cleanup_old_checkpoints keeps incomplete sessions regardless of age."""
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()

        # Create old incomplete checkpoint
        data = {
            "session_id": "old-incomplete",
            "stage": "transcribing",  # Not completed/failed
            "timestamp": old_time,
            "device_name": None,
            "ticket_id": None,
            "output_folder": None,
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {},
        }
        (tmp_path / "old-incomplete.checkpoint.json").write_text(json.dumps(data))

        removed = manager.cleanup_old_checkpoints(max_age_hours=24)

        assert removed == 0
        assert (tmp_path / "old-incomplete.checkpoint.json").exists()


# ============================================================================
# Batch 7: Concurrent Access and Edge Case Tests
# ============================================================================

import threading
from concurrent.futures import ThreadPoolExecutor


class TestCheckpointConcurrency:
    """Tests for thread safety of checkpoint operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create CheckpointManager with temp directory."""
        return CheckpointManager(checkpoint_dir=tmp_path)

    def test_concurrent_save_different_sessions(self, manager):
        """Concurrent saves to different sessions are thread-safe."""
        num_threads = 10
        errors = []

        def save_checkpoint(session_num):
            try:
                for i in range(10):
                    manager.save_checkpoint(
                        session_id=f"session-{session_num}",
                        stage=CheckpointStage.RECORDING,
                        device_name=f"Device-{i}",
                    )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(save_checkpoint, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        assert len(errors) == 0

        # Verify all checkpoints saved correctly
        for i in range(num_threads):
            checkpoint = manager.load_checkpoint(f"session-{i}")
            assert checkpoint is not None
            assert checkpoint.stage == CheckpointStage.RECORDING

    def test_concurrent_read_write_same_session(self, manager):
        """Concurrent reads and writes to same session."""
        session_id = "concurrent-session"
        manager.save_checkpoint(session_id, CheckpointStage.STARTED)

        errors = []
        read_results = []

        def writer():
            try:
                for stage in [CheckpointStage.RECORDING, CheckpointStage.STOPPED,
                              CheckpointStage.NORMALIZING, CheckpointStage.TRANSCRIBING]:
                    manager.update_checkpoint(session_id, stage)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    checkpoint = manager.load_checkpoint(session_id)
                    if checkpoint:
                        read_results.append(checkpoint.stage)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(writer),
                executor.submit(reader),
                executor.submit(reader),
                executor.submit(reader),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0
        # All read stages should be valid CheckpointStage values
        for stage in read_results:
            assert isinstance(stage, CheckpointStage)

    def test_concurrent_cleanup_and_save(self, manager, tmp_path):
        """Cleanup and save running concurrently."""
        errors = []

        # Seed some old completed checkpoints
        old_time = (datetime.now() - timedelta(hours=48)).isoformat()
        for i in range(10):
            data = {
                "session_id": f"old-{i}",
                "stage": "completed",
                "timestamp": old_time,
                "device_name": None,
                "ticket_id": None,
                "output_folder": None,
                "audio_file": None,
                "normalized_file": None,
                "transcript_file": None,
                "bundle_file": None,
                "error_message": None,
                "metadata": {},
            }
            (tmp_path / f"old-{i}.checkpoint.json").write_text(json.dumps(data))

        def cleanup():
            try:
                manager.cleanup_old_checkpoints(max_age_hours=24)
            except Exception as e:
                errors.append(e)

        def save():
            try:
                for i in range(10):
                    manager.save_checkpoint(f"new-{i}", CheckpointStage.RECORDING)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(cleanup),
                executor.submit(save),
                executor.submit(cleanup),
                executor.submit(save),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0


class TestCheckpointEdgeCases:
    """Edge case tests for checkpoint operations."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create CheckpointManager with temp directory."""
        return CheckpointManager(checkpoint_dir=tmp_path)

    def test_metadata_with_complex_types(self, manager):
        """Checkpoint handles complex metadata structures."""
        complex_metadata = {
            "nested": {"key": "value", "list": [1, 2, 3]},
            "numbers": [1.5, 2.5, 3.5],
            "unicode": "日本語テスト",
            "bool": True,
            "null": None,
        }

        checkpoint = manager.save_checkpoint(
            session_id="complex-meta",
            stage=CheckpointStage.RECORDING,
            metadata=complex_metadata,
        )

        loaded = manager.load_checkpoint("complex-meta")
        assert loaded.metadata == complex_metadata

    def test_unicode_session_ids(self, manager):
        """Checkpoint handles unicode session IDs."""
        session_id = "日本語セッション"

        manager.save_checkpoint(session_id, CheckpointStage.RECORDING)
        loaded = manager.load_checkpoint(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id

    def test_very_long_session_id(self, manager):
        """Checkpoint handles very long session IDs."""
        session_id = "a" * 200

        manager.save_checkpoint(session_id, CheckpointStage.RECORDING)
        loaded = manager.load_checkpoint(session_id)

        assert loaded is not None
        assert loaded.session_id == session_id

    def test_special_characters_in_paths(self, manager):
        """Checkpoint handles special characters in file paths."""
        checkpoint = manager.save_checkpoint(
            session_id="special-chars",
            stage=CheckpointStage.STOPPED,
            audio_file="/path/with spaces/and 'quotes'/file.wav",
            output_folder="/path/with/émojis/🎤/",
        )

        loaded = manager.load_checkpoint("special-chars")
        assert loaded.audio_file == "/path/with spaces/and 'quotes'/file.wav"
        assert loaded.output_folder == "/path/with/émojis/🎤/"

    def test_empty_session_id(self, manager):
        """Checkpoint handles empty session ID."""
        manager.save_checkpoint("", CheckpointStage.RECORDING)
        loaded = manager.load_checkpoint("")

        assert loaded is not None
        assert loaded.session_id == ""

    def test_checkpoint_with_all_fields(self, manager):
        """Checkpoint saves and loads all fields correctly."""
        checkpoint = manager.save_checkpoint(
            session_id="full-checkpoint",
            stage=CheckpointStage.BUNDLING,
            device_name="VB-Cable",
            ticket_id="TICKET-12345",
            output_folder="/output/folder",
            audio_file="/output/audio.wav",
            normalized_file="/output/normalized.wav",
            transcript_file="/output/transcript.srt",
            bundle_file="/output/bundle.vtb",
            error_message=None,
            metadata={"key": "value"},
        )

        loaded = manager.load_checkpoint("full-checkpoint")

        assert loaded.session_id == "full-checkpoint"
        assert loaded.stage == CheckpointStage.BUNDLING
        assert loaded.device_name == "VB-Cable"
        assert loaded.ticket_id == "TICKET-12345"
        assert loaded.output_folder == "/output/folder"
        assert loaded.audio_file == "/output/audio.wav"
        assert loaded.normalized_file == "/output/normalized.wav"
        assert loaded.transcript_file == "/output/transcript.srt"
        assert loaded.bundle_file == "/output/bundle.vtb"
        assert loaded.metadata == {"key": "value"}

    def test_invalid_stage_in_file(self, manager, tmp_path):
        """load_checkpoint handles invalid stage value gracefully."""
        data = {
            "session_id": "invalid-stage",
            "stage": "nonexistent_stage",
            "timestamp": datetime.now().isoformat(),
            "device_name": None,
            "ticket_id": None,
            "output_folder": None,
            "audio_file": None,
            "normalized_file": None,
            "transcript_file": None,
            "bundle_file": None,
            "error_message": None,
            "metadata": {},
        }
        (tmp_path / "invalid-stage.checkpoint.json").write_text(json.dumps(data))

        # Should return None due to parsing error
        checkpoint = manager.load_checkpoint("invalid-stage")
        assert checkpoint is None

    def test_missing_fields_in_file(self, manager, tmp_path):
        """load_checkpoint handles missing fields gracefully."""
        # Minimal valid checkpoint
        data = {
            "session_id": "minimal",
            "stage": "recording",
            "timestamp": datetime.now().isoformat(),
        }
        (tmp_path / "minimal.checkpoint.json").write_text(json.dumps(data))

        # Should fail due to missing required fields
        checkpoint = manager.load_checkpoint("minimal")
        assert checkpoint is None

    def test_resumable_stage_normalizing(self, manager):
        """get_resumable_stage returns NORMALIZING for NORMALIZING stage."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.NORMALIZING,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result == CheckpointStage.NORMALIZING

    def test_resumable_stage_bundling(self, manager):
        """get_resumable_stage returns BUNDLING for BUNDLING stage."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.BUNDLING,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result == CheckpointStage.BUNDLING

    def test_resumable_stage_completed(self, manager):
        """get_resumable_stage returns None for COMPLETED stage."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.COMPLETED,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result is None

    def test_resumable_stage_failed(self, manager):
        """get_resumable_stage returns None for FAILED stage."""
        checkpoint = Checkpoint(
            session_id="test",
            stage=CheckpointStage.FAILED,
            timestamp="2024-01-01",
        )

        result = manager.get_resumable_stage(checkpoint)
        assert result is None

    def test_get_incomplete_ignores_corrupted_files(self, manager, tmp_path):
        """get_incomplete_sessions skips corrupted checkpoint files."""
        # Create valid checkpoint
        manager.save_checkpoint("valid", CheckpointStage.RECORDING)

        # Create corrupted checkpoint
        (tmp_path / "corrupted.checkpoint.json").write_text("not valid json {{{")

        incomplete = manager.get_incomplete_sessions()

        # Should only find the valid checkpoint
        assert len(incomplete) == 1
        assert incomplete[0].session_id == "valid"

    def test_cleanup_ignores_corrupted_files(self, manager, tmp_path):
        """cleanup_old_checkpoints skips corrupted files without crashing."""
        # Create corrupted checkpoint
        (tmp_path / "corrupted.checkpoint.json").write_text("not valid json")

        # Should not raise
        removed = manager.cleanup_old_checkpoints(max_age_hours=24)
        assert removed == 0
