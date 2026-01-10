"""
Thread-Safe State Machine for CallWhisper

Based on LibV2 patterns:
- orchestrator-architecture: State persistence and workflow coordination
- advanced-react: useReducer patterns for centralized state logic

Key features:
- Thread-safe state transitions with Lock()
- Explicit transition validation
- Transition history for debugging
- Callback notifications for reactive updates
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, List, Callable, Optional, Any, Awaitable
from threading import Lock

from .exceptions import InvalidStateTransitionError
from .logging_config import get_core_logger

logger = get_core_logger()


class RecordingState(str, Enum):
    """
    Recording workflow states.

    State flow:
    IDLE → INITIALIZING → RECORDING → STOPPING → NORMALIZING → TRANSCRIBING → BUNDLING → COMPLETED
                    ↓           ↓          ↓           ↓             ↓            ↓
                  ERROR ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
                    ↓
                RECOVERING → (resume point)
    """

    IDLE = "idle"
    INITIALIZING = "initializing"
    RECORDING = "recording"
    STOPPING = "stopping"
    NORMALIZING = "normalizing"
    TRANSCRIBING = "transcribing"
    BUNDLING = "bundling"
    COMPLETED = "completed"
    ERROR = "error"
    RECOVERING = "recovering"


# Define valid state transitions
VALID_TRANSITIONS: Dict[RecordingState, Set[RecordingState]] = {
    RecordingState.IDLE: {RecordingState.INITIALIZING, RecordingState.RECOVERING},
    RecordingState.INITIALIZING: {RecordingState.RECORDING, RecordingState.ERROR},
    RecordingState.RECORDING: {RecordingState.STOPPING, RecordingState.ERROR},
    RecordingState.STOPPING: {RecordingState.NORMALIZING, RecordingState.ERROR},
    RecordingState.NORMALIZING: {RecordingState.TRANSCRIBING, RecordingState.ERROR},
    RecordingState.TRANSCRIBING: {
        RecordingState.BUNDLING,
        RecordingState.COMPLETED,  # If bundling disabled
        RecordingState.ERROR,
    },
    RecordingState.BUNDLING: {RecordingState.COMPLETED, RecordingState.ERROR},
    RecordingState.COMPLETED: {RecordingState.IDLE},
    RecordingState.ERROR: {RecordingState.IDLE, RecordingState.RECOVERING},
    RecordingState.RECOVERING: {
        RecordingState.IDLE,
        RecordingState.NORMALIZING,
        RecordingState.TRANSCRIBING,
        RecordingState.BUNDLING,
        RecordingState.ERROR,
    },
}


@dataclass
class StateTransitionEvent:
    """Record of a state transition."""

    event_id: str
    from_state: RecordingState
    to_state: RecordingState
    timestamp: str
    session_id: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


StateCallback = Callable[[StateTransitionEvent], Awaitable[None]]


class StateMachine:
    """
    Thread-safe state machine for recording workflow.

    Provides:
    - Atomic state transitions with Lock()
    - Transition validation
    - History tracking for debugging
    - Async callback notifications
    """

    def __init__(self, initial_state: RecordingState = RecordingState.IDLE):
        self._state = initial_state
        self._lock = Lock()
        self._callbacks: List[StateCallback] = []
        self._transition_history: List[StateTransitionEvent] = []
        self._current_session_id: Optional[str] = None
        self._event_counter = 0

    @property
    def state(self) -> RecordingState:
        """Current state (read-only)."""
        return self._state

    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._current_session_id

    @property
    def history(self) -> List[StateTransitionEvent]:
        """Transition history (read-only copy)."""
        return list(self._transition_history)

    def can_transition(self, to_state: RecordingState) -> bool:
        """Check if transition to given state is valid."""
        return to_state in VALID_TRANSITIONS.get(self._state, set())

    def get_valid_transitions(self) -> Set[RecordingState]:
        """Get set of valid next states."""
        return VALID_TRANSITIONS.get(self._state, set()).copy()

    def add_callback(self, callback: StateCallback) -> None:
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: StateCallback) -> None:
        """Remove a registered callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def transition(
        self,
        to_state: RecordingState,
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
        **metadata,
    ) -> StateTransitionEvent:
        """
        Perform atomic state transition.

        Args:
            to_state: Target state
            session_id: Session ID (updates current if provided)
            error_message: Error message (for ERROR state)
            **metadata: Additional metadata to record

        Returns:
            StateTransitionEvent record

        Raises:
            InvalidStateTransitionError: If transition is not valid
        """
        with self._lock:
            # Validate transition
            if not self.can_transition(to_state):
                valid = list(self.get_valid_transitions())
                logger.warning(
                    "invalid_state_transition_attempted",
                    from_state=self._state.value,
                    to_state=to_state.value,
                    valid_transitions=[s.value for s in valid],
                )
                raise InvalidStateTransitionError(
                    from_state=self._state.value,
                    to_state=to_state.value,
                    valid_transitions=[s.value for s in valid],
                )

            # Create event record
            self._event_counter += 1
            event = StateTransitionEvent(
                event_id=f"evt_{self._event_counter:06d}",
                from_state=self._state,
                to_state=to_state,
                timestamp=datetime.now().isoformat(),
                session_id=session_id or self._current_session_id,
                metadata=metadata,
                error_message=error_message,
            )

            # Update state
            old_state = self._state
            self._state = to_state

            # Update session ID if provided
            if session_id:
                self._current_session_id = session_id

            # Clear session on return to IDLE
            if to_state == RecordingState.IDLE:
                self._current_session_id = None

            # Record in history (keep last 100 events)
            self._transition_history.append(event)
            if len(self._transition_history) > 100:
                self._transition_history = self._transition_history[-100:]

            logger.info(
                "state_transition",
                from_state=old_state.value,
                to_state=to_state.value,
                session_id=event.session_id,
                event_id=event.event_id,
                **metadata,
            )

        # Notify callbacks outside the lock
        await self._notify_callbacks(event)

        return event

    async def _notify_callbacks(self, event: StateTransitionEvent) -> None:
        """Notify all registered callbacks of state change."""
        for callback in self._callbacks:
            try:
                await callback(event)
            except asyncio.CancelledError:
                # Callback cancellation shouldn't break the state machine
                logger.warning(
                    "state_callback_cancelled",
                    callback=(
                        callback.__name__
                        if hasattr(callback, "__name__")
                        else str(callback)
                    ),
                    event_id=event.event_id,
                )
            except Exception as e:
                logger.error(
                    "state_callback_error",
                    callback=(
                        callback.__name__
                        if hasattr(callback, "__name__")
                        else str(callback)
                    ),
                    error=str(e),
                    event_id=event.event_id,
                )

    async def reset(self) -> None:
        """Reset state machine to IDLE."""
        if self._state != RecordingState.IDLE:
            await self.transition(RecordingState.IDLE)

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for API responses."""
        return {
            "state": self._state.value,
            "session_id": self._current_session_id,
            "valid_transitions": [s.value for s in self.get_valid_transitions()],
        }

    def is_busy(self) -> bool:
        """Check if state machine is in a busy state (not IDLE or COMPLETED)."""
        return self._state not in {
            RecordingState.IDLE,
            RecordingState.COMPLETED,
            RecordingState.ERROR,
        }

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._state == RecordingState.RECORDING

    def is_processing(self) -> bool:
        """Check if in any processing state."""
        return self._state in {
            RecordingState.STOPPING,
            RecordingState.NORMALIZING,
            RecordingState.TRANSCRIBING,
            RecordingState.BUNDLING,
        }


# Global state machine instance
state_machine = StateMachine()
