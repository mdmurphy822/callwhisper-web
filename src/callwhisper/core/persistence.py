"""
State Persistence & Checkpointing Module

Based on LibV2 orchestrator-architecture course:
- Durable execution patterns
- Checkpoint after each stage completion
- Recovery from incomplete sessions on startup
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .logging_config import get_core_logger

logger = get_core_logger()


class CheckpointStage(str, Enum):
    """Stages in the recording/transcription workflow."""
    STARTED = "started"
    RECORDING = "recording"
    STOPPED = "stopped"
    NORMALIZING = "normalizing"
    TRANSCRIBING = "transcribing"
    BUNDLING = "bundling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Checkpoint:
    """Represents a workflow checkpoint."""
    session_id: str
    stage: CheckpointStage
    timestamp: str
    device_name: Optional[str] = None
    ticket_id: Optional[str] = None
    output_folder: Optional[str] = None
    audio_file: Optional[str] = None
    normalized_file: Optional[str] = None
    transcript_file: Optional[str] = None
    bundle_file: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert checkpoint to dictionary."""
        data = asdict(self)
        data['stage'] = self.stage.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        data['stage'] = CheckpointStage(data['stage'])
        return cls(**data)


class CheckpointManager:
    """
    Manages workflow checkpoints for crash recovery.

    Features:
    - Save checkpoint after each stage completion
    - Load checkpoints for recovery
    - Find incomplete sessions on startup
    - Clean up completed checkpoints
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("checkpoint_manager_initialized", checkpoint_dir=str(self.checkpoint_dir))

    def _get_checkpoint_path(self, session_id: str) -> Path:
        """Get path for a session's checkpoint file."""
        return self.checkpoint_dir / f"{session_id}.checkpoint.json"

    def save_checkpoint(
        self,
        session_id: str,
        stage: CheckpointStage,
        **kwargs
    ) -> Checkpoint:
        """
        Save a checkpoint for a session.

        Args:
            session_id: Unique session identifier
            stage: Current workflow stage
            **kwargs: Additional checkpoint data

        Returns:
            The saved Checkpoint object
        """
        checkpoint = Checkpoint(
            session_id=session_id,
            stage=stage,
            timestamp=datetime.now().isoformat(),
            **kwargs
        )

        path = self._get_checkpoint_path(session_id)
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2))

        logger.info(
            "checkpoint_saved",
            session_id=session_id,
            stage=stage.value,
            path=str(path)
        )

        return checkpoint

    def load_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        """
        Load a checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Checkpoint if exists, None otherwise
        """
        path = self._get_checkpoint_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            checkpoint = Checkpoint.from_dict(data)
            logger.debug("checkpoint_loaded", session_id=session_id, stage=checkpoint.stage.value)
            return checkpoint
        except Exception as e:
            logger.error("checkpoint_load_error", session_id=session_id, error=str(e))
            return None

    def update_checkpoint(
        self,
        session_id: str,
        stage: CheckpointStage,
        **kwargs
    ) -> Optional[Checkpoint]:
        """
        Update an existing checkpoint.

        Args:
            session_id: Session identifier
            stage: New workflow stage
            **kwargs: Fields to update

        Returns:
            Updated Checkpoint or None if not found
        """
        existing = self.load_checkpoint(session_id)
        if not existing:
            return self.save_checkpoint(session_id, stage, **kwargs)

        # Merge existing data with updates
        existing_dict = existing.to_dict()
        existing_dict['stage'] = stage.value
        existing_dict['timestamp'] = datetime.now().isoformat()
        for key, value in kwargs.items():
            if value is not None:
                existing_dict[key] = value

        path = self._get_checkpoint_path(session_id)
        path.write_text(json.dumps(existing_dict, indent=2))

        logger.info(
            "checkpoint_updated",
            session_id=session_id,
            stage=stage.value
        )

        return Checkpoint.from_dict(existing_dict)

    def get_incomplete_sessions(self) -> List[Checkpoint]:
        """
        Find all incomplete sessions (for recovery on startup).

        Returns:
            List of checkpoints for sessions that didn't complete
        """
        incomplete = []

        for path in self.checkpoint_dir.glob("*.checkpoint.json"):
            try:
                data = json.loads(path.read_text())
                checkpoint = Checkpoint.from_dict(data)

                # Check if session is incomplete
                if checkpoint.stage not in [CheckpointStage.COMPLETED, CheckpointStage.FAILED]:
                    incomplete.append(checkpoint)
                    logger.info(
                        "incomplete_session_found",
                        session_id=checkpoint.session_id,
                        stage=checkpoint.stage.value,
                        timestamp=checkpoint.timestamp
                    )
            except Exception as e:
                logger.error("checkpoint_parse_error", path=str(path), error=str(e))

        return incomplete

    def clear_checkpoint(self, session_id: str) -> bool:
        """
        Remove checkpoint after successful completion.

        Args:
            session_id: Session identifier

        Returns:
            True if checkpoint was removed, False if not found
        """
        path = self._get_checkpoint_path(session_id)
        if path.exists():
            path.unlink()
            logger.info("checkpoint_cleared", session_id=session_id)
            return True
        return False

    def mark_completed(self, session_id: str, bundle_file: Optional[str] = None) -> Optional[Checkpoint]:
        """
        Mark a session as completed.

        Args:
            session_id: Session identifier
            bundle_file: Path to the final bundle file

        Returns:
            Updated checkpoint
        """
        return self.update_checkpoint(
            session_id,
            CheckpointStage.COMPLETED,
            bundle_file=bundle_file
        )

    def mark_failed(self, session_id: str, error_message: str) -> Optional[Checkpoint]:
        """
        Mark a session as failed.

        Args:
            session_id: Session identifier
            error_message: Description of the failure

        Returns:
            Updated checkpoint
        """
        return self.update_checkpoint(
            session_id,
            CheckpointStage.FAILED,
            error_message=error_message
        )

    def get_resumable_stage(self, checkpoint: Checkpoint) -> Optional[CheckpointStage]:
        """
        Determine which stage to resume from based on checkpoint.

        Args:
            checkpoint: The checkpoint to analyze

        Returns:
            The stage to resume from, or None if not resumable
        """
        stage = checkpoint.stage

        # Determine resume point based on what was completed
        if stage == CheckpointStage.STOPPED:
            # Recording completed, resume from normalizing
            return CheckpointStage.NORMALIZING
        elif stage == CheckpointStage.NORMALIZING:
            # Normalizing in progress, restart normalizing
            return CheckpointStage.NORMALIZING
        elif stage == CheckpointStage.TRANSCRIBING:
            # Transcribing in progress, restart transcribing
            return CheckpointStage.TRANSCRIBING
        elif stage == CheckpointStage.BUNDLING:
            # Bundling in progress, restart bundling
            return CheckpointStage.BUNDLING

        # Other stages are not resumable
        return None

    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed/failed checkpoints.

        Args:
            max_age_hours: Maximum age of checkpoints to keep

        Returns:
            Number of checkpoints removed
        """
        from datetime import timedelta

        removed = 0
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        for path in self.checkpoint_dir.glob("*.checkpoint.json"):
            try:
                data = json.loads(path.read_text())
                checkpoint = Checkpoint.from_dict(data)

                if checkpoint.stage in [CheckpointStage.COMPLETED, CheckpointStage.FAILED]:
                    checkpoint_time = datetime.fromisoformat(checkpoint.timestamp)
                    if checkpoint_time < cutoff:
                        path.unlink()
                        removed += 1
                        logger.debug("old_checkpoint_removed", session_id=checkpoint.session_id)

            except Exception as e:
                logger.warning("checkpoint_cleanup_error", path=str(path), error=str(e))

        if removed > 0:
            logger.info("checkpoint_cleanup_completed", removed_count=removed)

        return removed
