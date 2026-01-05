"""
Call Detection Orchestrator.

Combines signals from:
- Tier 1: WASAPI audio session state (authoritative)
- Tier 2: Process lifecycle (required)
- Tier 3: Network hints (optional - not implemented)
- Tier 4: Window metadata (optional - not implemented)

State Machine:
    NO_CALL --[audio session active]--> CALL_STARTING
    CALL_STARTING --[confirmed 1s]--> CALL_ACTIVE (start recording)
    CALL_ACTIVE --[audio session inactive]--> CALL_ENDING
    CALL_ENDING --[confirmed 2s]--> NO_CALL (stop recording)
"""

import sys
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

if sys.platform != "win32":
    raise ImportError("CallDetector is only available on Windows")

from .windows_audio_monitor import (
    WindowsAudioSessionMonitor,
    AudioSessionEvent,
    AudioSessionState
)
from .process_monitor import ProcessMonitor, ProcessEvent
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class CallState(Enum):
    """States in the call detection state machine."""
    NO_CALL = "no_call"           # No call detected
    CALL_STARTING = "call_starting"  # Audio session detected, confirming
    CALL_ACTIVE = "call_active"      # Recording in progress
    CALL_ENDING = "call_ending"      # Audio stopped, confirming


@dataclass
class CallInfo:
    """Information about a detected call."""
    call_id: str
    process_name: str
    process_id: int
    state: CallState
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        return f"Call({self.call_id}: {self.state.value}, {self.process_name}[{self.process_id}])"


@dataclass
class CallDetectorConfig:
    """Configuration for call detection."""
    # Feature toggle
    enabled: bool = False

    # Target processes to monitor
    target_processes: List[str] = field(default_factory=lambda: ["CiscoJabber.exe"])

    # Browser processes for Finesse (optional)
    finesse_browsers: List[str] = field(
        default_factory=lambda: ["chrome.exe", "msedge.exe", "firefox.exe"]
    )
    finesse_url_pattern: str = "finesse"  # Match in window title

    # Timing
    call_start_confirm_seconds: float = 1.0   # Confirm audio active for 1s
    call_end_confirm_seconds: float = 2.0     # Confirm audio inactive for 2s
    audio_poll_interval: float = 0.5          # How often to poll audio sessions
    process_poll_interval: float = 2.0        # How often to poll processes

    # Safety
    max_call_duration_minutes: int = 180      # Auto-stop after 3 hours
    min_call_duration_seconds: int = 5        # Discard calls shorter than 5s


class CallDetector:
    """
    Orchestrates call detection from multiple signal sources.

    Architecture:
    - Tier 1 (WASAPI): Audio session monitoring is the authoritative signal
    - Tier 2 (Process): Process monitoring gates when to start/stop monitoring

    When a target process starts, we begin monitoring its audio sessions.
    When an audio session becomes active, we transition to CALL_STARTING.
    After confirmation period, we transition to CALL_ACTIVE and trigger recording.
    """

    def __init__(self):
        self._config: Optional[CallDetectorConfig] = None
        self._audio_monitor: Optional[WindowsAudioSessionMonitor] = None
        self._process_monitor: Optional[ProcessMonitor] = None
        self._running = False

        # State tracking
        self._current_state = CallState.NO_CALL
        self._current_call: Optional[CallInfo] = None
        self._state_change_time: float = 0.0
        self._call_counter = 0

        # Callbacks
        self._on_call_start: List[Callable[[CallInfo], None]] = []
        self._on_call_end: List[Callable[[CallInfo], None]] = []

        # Async event loop reference
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._confirmation_task: Optional[asyncio.Task] = None

    async def start(self, config: CallDetectorConfig) -> None:
        """Start call detection with the given configuration."""
        if self._running:
            logger.warning("call_detector_already_running")
            return

        if not config.enabled:
            logger.info("call_detector_disabled")
            return

        self._config = config
        self._running = True
        self._loop = asyncio.get_event_loop()

        # Combine target processes (Jabber + Finesse browsers)
        all_targets = config.target_processes.copy()

        logger.info(
            "call_detector_starting",
            targets=all_targets,
            confirm_start_s=config.call_start_confirm_seconds,
            confirm_end_s=config.call_end_confirm_seconds
        )

        # Initialize Tier 2: Process monitor
        self._process_monitor = ProcessMonitor(
            target_processes=all_targets,
            poll_interval=config.process_poll_interval
        )
        self._process_monitor.add_callback(self._handle_process_event)
        self._process_monitor.start()

        # Initialize Tier 1: Audio session monitor
        self._audio_monitor = WindowsAudioSessionMonitor(
            target_processes=all_targets,
            poll_interval=config.audio_poll_interval
        )
        self._audio_monitor.add_callback(self._handle_audio_event)
        self._audio_monitor.start()

        logger.info("call_detector_started")

    async def stop(self) -> None:
        """Stop call detection."""
        if not self._running:
            return

        self._running = False

        # Cancel any pending confirmation
        if self._confirmation_task:
            self._confirmation_task.cancel()
            self._confirmation_task = None

        # Stop monitors
        if self._audio_monitor:
            self._audio_monitor.stop()
            self._audio_monitor = None

        if self._process_monitor:
            self._process_monitor.stop()
            self._process_monitor = None

        # End any active call
        if self._current_call and self._current_state == CallState.CALL_ACTIVE:
            await self._end_call()

        self._current_state = CallState.NO_CALL
        self._current_call = None
        self._config = None

        logger.info("call_detector_stopped")

    def on_call_start(self, callback: Callable[[CallInfo], None]) -> None:
        """Register callback for call start events."""
        self._on_call_start.append(callback)

    def on_call_end(self, callback: Callable[[CallInfo], None]) -> None:
        """Register callback for call end events."""
        self._on_call_end.append(callback)

    def get_state(self) -> CallState:
        """Get current call state."""
        return self._current_state

    def get_current_call(self) -> Optional[CallInfo]:
        """Get information about the current call (if any)."""
        return self._current_call

    def get_status(self) -> Dict:
        """Get full status for API endpoints."""
        return {
            "enabled": self._running,
            "state": self._current_state.value,
            "current_call": {
                "call_id": self._current_call.call_id,
                "process_name": self._current_call.process_name,
                "process_id": self._current_call.process_id,
                "started_at": self._current_call.started_at.isoformat(),
                "duration_seconds": (
                    datetime.now() - self._current_call.started_at
                ).total_seconds()
            } if self._current_call else None,
            "monitors": {
                "audio": self._audio_monitor.is_running if self._audio_monitor else False,
                "process": self._process_monitor.is_running if self._process_monitor else False,
            },
            "config": {
                "target_processes": self._config.target_processes if self._config else [],
                "call_start_confirm_seconds": self._config.call_start_confirm_seconds if self._config else 0,
                "call_end_confirm_seconds": self._config.call_end_confirm_seconds if self._config else 0,
            } if self._config else None
        }

    def _handle_process_event(self, event: ProcessEvent) -> None:
        """Handle process start/stop events (Tier 2)."""
        logger.debug(
            "call_detector_process_event",
            event_type=event.event_type,
            process=event.process_name,
            pid=event.process_id
        )

        # Process events are primarily for logging/debugging
        # The audio monitor handles the actual call detection

    def _handle_audio_event(self, event: AudioSessionEvent) -> None:
        """Handle audio session events (Tier 1 - authoritative)."""
        logger.info(
            "call_detector_audio_event",
            process=event.process_name,
            pid=event.process_id,
            state=event.state.name,
            current_state=self._current_state.value
        )

        # State machine transitions
        if event.state == AudioSessionState.ACTIVE:
            self._on_audio_active(event)
        elif event.state in (AudioSessionState.INACTIVE, AudioSessionState.EXPIRED):
            self._on_audio_inactive(event)

    def _on_audio_active(self, event: AudioSessionEvent) -> None:
        """Handle audio session becoming active."""
        if self._current_state == CallState.NO_CALL:
            # Transition to CALL_STARTING
            self._transition_to(CallState.CALL_STARTING, event)
            self._schedule_confirmation(
                self._config.call_start_confirm_seconds,
                self._confirm_call_start
            )

        elif self._current_state == CallState.CALL_ENDING:
            # Call resuming - cancel end confirmation and go back to active
            self._cancel_confirmation()
            self._transition_to(CallState.CALL_ACTIVE)

    def _on_audio_inactive(self, event: AudioSessionEvent) -> None:
        """Handle audio session becoming inactive."""
        if self._current_state == CallState.CALL_STARTING:
            # Cancel start confirmation - false positive
            self._cancel_confirmation()
            self._transition_to(CallState.NO_CALL)

        elif self._current_state == CallState.CALL_ACTIVE:
            # Transition to CALL_ENDING
            self._transition_to(CallState.CALL_ENDING)
            self._schedule_confirmation(
                self._config.call_end_confirm_seconds,
                self._confirm_call_end
            )

    def _transition_to(
        self,
        new_state: CallState,
        event: Optional[AudioSessionEvent] = None
    ) -> None:
        """Transition to a new state."""
        old_state = self._current_state
        self._current_state = new_state
        self._state_change_time = time.time()

        logger.info(
            "call_state_transition",
            from_state=old_state.value,
            to_state=new_state.value,
            process=event.process_name if event else None
        )

        # Create call info on CALL_STARTING
        if new_state == CallState.CALL_STARTING and event:
            self._call_counter += 1
            self._current_call = CallInfo(
                call_id=f"call_{self._call_counter}_{int(time.time())}",
                process_name=event.process_name,
                process_id=event.process_id,
                state=new_state,
                started_at=datetime.now()
            )

    def _schedule_confirmation(
        self,
        delay_seconds: float,
        callback: Callable
    ) -> None:
        """Schedule a confirmation callback after delay."""
        self._cancel_confirmation()

        if self._loop:
            self._confirmation_task = self._loop.call_later(
                delay_seconds,
                lambda: asyncio.create_task(callback())
            )

    def _cancel_confirmation(self) -> None:
        """Cancel any pending confirmation."""
        if self._confirmation_task:
            if hasattr(self._confirmation_task, 'cancel'):
                self._confirmation_task.cancel()
            self._confirmation_task = None

    async def _confirm_call_start(self) -> None:
        """Confirm call start after debounce period."""
        if self._current_state != CallState.CALL_STARTING:
            return

        # Check if audio is still active
        if self._audio_monitor:
            active_sessions = self._audio_monitor.get_active_sessions()
            if not active_sessions:
                # Audio stopped during confirmation
                self._transition_to(CallState.NO_CALL)
                self._current_call = None
                return

        # Confirm call start
        self._transition_to(CallState.CALL_ACTIVE)

        if self._current_call:
            self._current_call.state = CallState.CALL_ACTIVE
            logger.info(
                "call_started",
                call_id=self._current_call.call_id,
                process=self._current_call.process_name
            )

            # Fire callbacks
            for callback in self._on_call_start:
                try:
                    callback(self._current_call)
                except Exception as e:
                    logger.error("call_start_callback_error", error=str(e))

    async def _confirm_call_end(self) -> None:
        """Confirm call end after debounce period."""
        if self._current_state != CallState.CALL_ENDING:
            return

        # Check if audio is still inactive
        if self._audio_monitor:
            active_sessions = self._audio_monitor.get_active_sessions()
            if active_sessions:
                # Audio became active again
                self._transition_to(CallState.CALL_ACTIVE)
                return

        # Confirm call end
        await self._end_call()

    async def _end_call(self) -> None:
        """End the current call and fire callbacks."""
        if not self._current_call:
            self._transition_to(CallState.NO_CALL)
            return

        self._current_call.ended_at = datetime.now()
        self._current_call.state = CallState.NO_CALL
        self._current_call.duration_seconds = (
            self._current_call.ended_at - self._current_call.started_at
        ).total_seconds()

        # Check minimum duration
        if self._config and self._current_call.duration_seconds < self._config.min_call_duration_seconds:
            logger.info(
                "call_discarded_too_short",
                call_id=self._current_call.call_id,
                duration=self._current_call.duration_seconds,
                min_duration=self._config.min_call_duration_seconds
            )
        else:
            logger.info(
                "call_ended",
                call_id=self._current_call.call_id,
                duration=self._current_call.duration_seconds
            )

            # Fire callbacks
            for callback in self._on_call_end:
                try:
                    callback(self._current_call)
                except Exception as e:
                    logger.error("call_end_callback_error", error=str(e))

        self._transition_to(CallState.NO_CALL)
        self._current_call = None

    @property
    def is_running(self) -> bool:
        """Check if call detector is running."""
        return self._running


# Global instance for the application
_call_detector: Optional[CallDetector] = None


def get_call_detector() -> CallDetector:
    """Get or create the global call detector instance."""
    global _call_detector
    if _call_detector is None:
        _call_detector = CallDetector()
    return _call_detector
