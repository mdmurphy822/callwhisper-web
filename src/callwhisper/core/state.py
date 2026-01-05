"""Application state management."""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import asyncio

from .logging_config import get_logger

logger = get_logger(__name__)


class AppState(str, Enum):
    """Application states."""

    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class RecordingSession:
    """Active recording session data."""

    id: str
    device_name: str
    ticket_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output_folder: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CompletedRecording:
    """Completed recording metadata."""

    id: str
    ticket_id: Optional[str]
    created_at: datetime
    duration_seconds: float
    output_folder: str
    bundle_path: Optional[str] = None
    transcript_preview: Optional[str] = None


class ApplicationState:
    """Global application state manager."""

    def __init__(self):
        self.state: AppState = AppState.IDLE
        self.current_session: Optional[RecordingSession] = None
        self.completed_recordings: List[CompletedRecording] = []
        self.elapsed_seconds: int = 0
        self._timer_task: Optional[asyncio.Task] = None
        self._timer_lock: asyncio.Lock = asyncio.Lock()
        self._state_callbacks: List[callable] = []

    def add_state_callback(self, callback: callable):
        """Add callback for state changes."""
        self._state_callbacks.append(callback)

    async def _cancel_timer(self):
        """Safely cancel the timer task with proper cleanup."""
        async with self._timer_lock:
            if self._timer_task:
                self._timer_task.cancel()
                try:
                    await self._timer_task
                except asyncio.CancelledError:
                    pass
                finally:
                    self._timer_task = None

    async def _notify_state_change(self, data: Dict[str, Any]):
        """Notify all callbacks of state change."""
        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(
                    "state_callback_error",
                    callback=(
                        callback.__name__
                        if hasattr(callback, "__name__")
                        else str(callback)
                    ),
                    error=str(e),
                )

    async def start_recording(self, session: RecordingSession):
        """Transition to recording state."""
        self.state = AppState.RECORDING
        self.current_session = session
        self.current_session.start_time = datetime.now()
        self.elapsed_seconds = 0

        # Start timer
        self._timer_task = asyncio.create_task(self._run_timer())

        await self._notify_state_change(
            {
                "type": "state_change",
                "state": self.state.value,
                "recording_id": session.id,
            }
        )

    async def _run_timer(self):
        """Run the recording timer."""
        try:
            while self.state == AppState.RECORDING:
                await asyncio.sleep(1)
                self.elapsed_seconds += 1
                await self._notify_state_change(
                    {
                        "type": "timer",
                        "elapsed_seconds": self.elapsed_seconds,
                        "formatted": f"{self.elapsed_seconds // 60:02d}:{self.elapsed_seconds % 60:02d}",
                    }
                )
        except asyncio.CancelledError:
            pass

    async def stop_recording(self):
        """Transition to processing state."""
        await self._cancel_timer()

        if self.current_session:
            self.current_session.end_time = datetime.now()

        self.state = AppState.PROCESSING

        await self._notify_state_change(
            {
                "type": "state_change",
                "state": self.state.value,
                "recording_id": (
                    self.current_session.id if self.current_session else None
                ),
            }
        )

    async def processing_progress(self, percent: int, stage: str):
        """Report processing progress."""
        await self._notify_state_change(
            {
                "type": "processing_progress",
                "percent": percent,
                "stage": stage,
            }
        )

    async def partial_transcript(self, text: str, is_final: bool = False):
        """
        Broadcast partial transcript text during transcription.

        Used for real-time preview during processing.

        Args:
            text: The partial transcript text so far
            is_final: True when transcription is complete
        """
        await self._notify_state_change(
            {
                "type": "partial_transcript",
                "text": text,
                "is_final": is_final,
            }
        )

    async def complete_recording(self, recording: CompletedRecording):
        """Transition to done state."""
        self.state = AppState.DONE
        self.completed_recordings.insert(0, recording)
        self.current_session = None

        await self._notify_state_change(
            {
                "type": "recording_complete",
                "state": self.state.value,
                "recording_id": recording.id,
                "output_folder": recording.output_folder,
                "bundle_path": recording.bundle_path,
            }
        )

    async def set_error(self, error: str):
        """Transition to error state."""
        self.state = AppState.ERROR

        await self._cancel_timer()

        if self.current_session:
            self.current_session.error = error

        await self._notify_state_change(
            {
                "type": "error",
                "state": self.state.value,
                "message": error,
            }
        )

    async def reset(self):
        """Reset to idle state."""
        self.state = AppState.IDLE
        self.current_session = None
        self.elapsed_seconds = 0

        await self._notify_state_change(
            {
                "type": "state_change",
                "state": self.state.value,
            }
        )

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "state": self.state.value,
            "recording_id": self.current_session.id if self.current_session else None,
            "elapsed_seconds": self.elapsed_seconds,
            "elapsed_formatted": f"{self.elapsed_seconds // 60:02d}:{self.elapsed_seconds % 60:02d}",
        }


# Global application state instance
app_state = ApplicationState()
