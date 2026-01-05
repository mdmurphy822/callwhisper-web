"""
Windows Audio Session Monitor using WASAPI/pycaw.

Monitors audio sessions for specific processes (CiscoJabber.exe)
and detects when they become active/inactive.

This is the authoritative trigger for call detection (Tier 1).
"""

import sys
import time
import threading
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

if sys.platform != "win32":
    raise ImportError("WindowsAudioSessionMonitor is only available on Windows")

import comtypes
from comtypes import GUID, COMMETHOD, HRESULT
from ctypes import POINTER, c_int, c_uint, c_wchar_p, c_float
from ctypes.wintypes import DWORD, LPCWSTR

# Initialize COM for the current thread
comtypes.CoInitialize()

from pycaw.pycaw import AudioUtilities, IAudioSessionControl2, ISimpleAudioVolume

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AudioSessionState(IntEnum):
    """Windows audio session states from audiosessiontypes.h."""

    INACTIVE = 0  # AudioSessionStateInactive - session idle
    ACTIVE = 1  # AudioSessionStateActive - session playing audio
    EXPIRED = 2  # AudioSessionStateExpired - session disconnected


@dataclass
class AudioSessionEvent:
    """Event fired when an audio session changes state."""

    process_name: str
    process_id: int
    state: AudioSessionState
    session_id: str
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"AudioSession({self.process_name}[{self.process_id}]: {self.state.name})"
        )


class WindowsAudioSessionMonitor:
    """
    Monitors WASAPI audio sessions for target processes.

    Uses polling-based detection to monitor audio sessions because
    IAudioSessionNotification requires COM apartment threading that
    doesn't play well with async frameworks.

    Detection logic:
    - Poll every 500ms for audio session changes
    - Track sessions by (process_id, session_id)
    - Fire callbacks on state transitions
    """

    def __init__(self, target_processes: List[str], poll_interval: float = 0.5):
        """
        Initialize the audio session monitor.

        Args:
            target_processes: List of process names to monitor (e.g., ["CiscoJabber.exe"])
            poll_interval: How often to poll for session changes (seconds)
        """
        self._targets = [p.lower() for p in target_processes]
        self._poll_interval = poll_interval
        self._callbacks: List[Callable[[AudioSessionEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Track known sessions: {(pid, session_id): AudioSessionState}
        self._known_sessions: Dict[tuple, AudioSessionState] = {}

        logger.info(
            "audio_monitor_initialized",
            targets=self._targets,
            poll_interval=poll_interval,
        )

    def add_callback(self, callback: Callable[[AudioSessionEvent], None]) -> None:
        """Register a callback for session events."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[AudioSessionEvent], None]) -> None:
        """Remove a registered callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self) -> None:
        """Start monitoring audio sessions."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, name="AudioSessionMonitor", daemon=True
        )
        self._thread.start()
        logger.info("audio_monitor_started")

    def stop(self) -> None:
        """Stop monitoring audio sessions."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._known_sessions.clear()
        logger.info("audio_monitor_stopped")

    def get_active_sessions(self) -> List[AudioSessionEvent]:
        """Get currently active sessions for target processes."""
        active = []
        sessions = self._enumerate_sessions()

        for session_info in sessions:
            if session_info["state"] == AudioSessionState.ACTIVE:
                active.append(
                    AudioSessionEvent(
                        process_name=session_info["process_name"],
                        process_id=session_info["process_id"],
                        state=AudioSessionState.ACTIVE,
                        session_id=session_info["session_id"],
                    )
                )

        return active

    def get_all_sessions(self) -> List[Dict]:
        """Get all audio sessions (for debugging)."""
        return self._enumerate_sessions(filter_targets=False)

    def _poll_loop(self) -> None:
        """Main polling loop running in a separate thread."""
        # Re-initialize COM for this thread
        comtypes.CoInitialize()

        try:
            while self._running:
                try:
                    self._check_sessions()
                except Exception as e:
                    logger.error("audio_session_poll_error", error=str(e))

                time.sleep(self._poll_interval)
        finally:
            comtypes.CoUninitialize()

    def _check_sessions(self) -> None:
        """Check for session state changes."""
        current_sessions = {}

        for session_info in self._enumerate_sessions():
            key = (session_info["process_id"], session_info["session_id"])
            state = session_info["state"]
            current_sessions[key] = (state, session_info)

        # Detect changes
        with self._lock:
            # Check for new or changed sessions
            for key, (state, info) in current_sessions.items():
                old_state = self._known_sessions.get(key)

                if old_state is None:
                    # New session
                    self._fire_event(
                        AudioSessionEvent(
                            process_name=info["process_name"],
                            process_id=info["process_id"],
                            state=state,
                            session_id=info["session_id"],
                        )
                    )
                elif old_state != state:
                    # State changed
                    self._fire_event(
                        AudioSessionEvent(
                            process_name=info["process_name"],
                            process_id=info["process_id"],
                            state=state,
                            session_id=info["session_id"],
                        )
                    )

            # Check for expired sessions
            expired_keys = set(self._known_sessions.keys()) - set(
                current_sessions.keys()
            )
            for key in expired_keys:
                pid, session_id = key
                # Fire expired event
                self._fire_event(
                    AudioSessionEvent(
                        process_name="unknown",  # Process may have exited
                        process_id=pid,
                        state=AudioSessionState.EXPIRED,
                        session_id=session_id,
                    )
                )

            # Update known sessions
            self._known_sessions = {k: v[0] for k, v in current_sessions.items()}

    def _enumerate_sessions(self, filter_targets: bool = True) -> List[Dict]:
        """
        Enumerate all audio sessions.

        Args:
            filter_targets: If True, only return sessions for target processes

        Returns:
            List of session info dicts
        """
        sessions = []

        try:
            all_sessions = AudioUtilities.GetAllSessions()

            for session in all_sessions:
                if session.Process is None:
                    continue

                process_name = session.Process.name()
                process_id = session.Process.pid

                # Filter by target processes if requested
                if filter_targets and process_name.lower() not in self._targets:
                    continue

                # Get session state
                try:
                    state_value = session._ctl.GetState()
                    state = AudioSessionState(state_value)
                except Exception:
                    state = AudioSessionState.INACTIVE

                # Generate a session ID from the session instance ID
                try:
                    session_id = session.Identifier or f"session_{process_id}"
                except Exception:
                    session_id = f"session_{process_id}"

                sessions.append(
                    {
                        "process_name": process_name,
                        "process_id": process_id,
                        "state": state,
                        "session_id": session_id,
                    }
                )

        except Exception as e:
            logger.error("enumerate_sessions_error", error=str(e))

        return sessions

    def _fire_event(self, event: AudioSessionEvent) -> None:
        """Fire event to all registered callbacks."""
        logger.debug(
            "audio_session_event",
            process=event.process_name,
            pid=event.process_id,
            state=event.state.name,
        )

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    "callback_error",
                    error=str(e),
                    callback=(
                        callback.__name__
                        if hasattr(callback, "__name__")
                        else str(callback)
                    ),
                )

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def target_processes(self) -> List[str]:
        """Get list of target processes."""
        return self._targets.copy()
