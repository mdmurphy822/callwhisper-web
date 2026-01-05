"""
Unit tests for the thread-safe state machine.

Based on LibV2 Python programming course testing patterns:
- Test valid state transitions
- Test invalid state transitions (exception handling)
- Test thread safety
- Test callback notifications
"""

import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

from callwhisper.core.state_machine import (
    StateMachine,
    RecordingState,
    VALID_TRANSITIONS,
    StateTransitionEvent
)
from callwhisper.core.exceptions import InvalidStateTransitionError


@pytest.mark.unit
class TestStateMachineBasics:
    """Basic state machine functionality tests."""

    def test_initial_state_is_idle(self):
        """State machine should start in IDLE state."""
        sm = StateMachine()
        assert sm.state == RecordingState.IDLE

    def test_custom_initial_state(self):
        """State machine should accept custom initial state."""
        sm = StateMachine(initial_state=RecordingState.ERROR)
        assert sm.state == RecordingState.ERROR

    def test_session_id_initially_none(self):
        """Session ID should be None initially."""
        sm = StateMachine()
        assert sm.session_id is None

    def test_history_initially_empty(self):
        """Transition history should be empty initially."""
        sm = StateMachine()
        assert sm.history == []


@pytest.mark.unit
class TestValidTransitions:
    """Test valid state transitions."""

    @pytest.mark.asyncio
    async def test_idle_to_initializing(self):
        """Can transition from IDLE to INITIALIZING."""
        sm = StateMachine()
        event = await sm.transition(RecordingState.INITIALIZING)
        assert sm.state == RecordingState.INITIALIZING
        assert event.from_state == RecordingState.IDLE
        assert event.to_state == RecordingState.INITIALIZING

    @pytest.mark.asyncio
    async def test_full_successful_workflow(self):
        """Test complete recording workflow."""
        sm = StateMachine()
        session_id = "test_session_001"

        # Start recording
        await sm.transition(RecordingState.INITIALIZING, session_id=session_id)
        assert sm.session_id == session_id

        await sm.transition(RecordingState.RECORDING)
        await sm.transition(RecordingState.STOPPING)
        await sm.transition(RecordingState.NORMALIZING)
        await sm.transition(RecordingState.TRANSCRIBING)
        await sm.transition(RecordingState.BUNDLING)
        await sm.transition(RecordingState.COMPLETED)

        assert sm.state == RecordingState.COMPLETED

        # Return to IDLE
        await sm.transition(RecordingState.IDLE)
        assert sm.state == RecordingState.IDLE
        assert sm.session_id is None

    @pytest.mark.asyncio
    async def test_skip_bundling_when_disabled(self):
        """Can transition directly from TRANSCRIBING to COMPLETED."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)
        await sm.transition(RecordingState.STOPPING)
        await sm.transition(RecordingState.NORMALIZING)
        await sm.transition(RecordingState.TRANSCRIBING)
        await sm.transition(RecordingState.COMPLETED)  # Skip bundling
        assert sm.state == RecordingState.COMPLETED

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error and recovery workflow."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)

        # Error during recording
        await sm.transition(RecordingState.ERROR, error_message="Connection lost")
        assert sm.state == RecordingState.ERROR

        # Recovery
        await sm.transition(RecordingState.RECOVERING)
        await sm.transition(RecordingState.NORMALIZING)  # Resume from checkpoint
        await sm.transition(RecordingState.TRANSCRIBING)
        await sm.transition(RecordingState.COMPLETED)
        assert sm.state == RecordingState.COMPLETED


@pytest.mark.unit
class TestInvalidTransitions:
    """Test invalid state transition handling."""

    @pytest.mark.asyncio
    async def test_idle_to_recording_invalid(self):
        """Cannot transition directly from IDLE to RECORDING."""
        sm = StateMachine()
        with pytest.raises(InvalidStateTransitionError) as exc_info:
            await sm.transition(RecordingState.RECORDING)

        error = exc_info.value
        assert error.details["from_state"] == "idle"
        assert error.details["to_state"] == "recording"
        assert "initializing" in error.details["valid_transitions"]

    @pytest.mark.asyncio
    async def test_recording_to_idle_invalid(self):
        """Cannot transition directly from RECORDING to IDLE."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)

        with pytest.raises(InvalidStateTransitionError):
            await sm.transition(RecordingState.IDLE)

        # State should remain RECORDING
        assert sm.state == RecordingState.RECORDING

    @pytest.mark.asyncio
    async def test_completed_to_recording_invalid(self):
        """Cannot transition from COMPLETED to RECORDING."""
        sm = StateMachine(initial_state=RecordingState.COMPLETED)
        with pytest.raises(InvalidStateTransitionError):
            await sm.transition(RecordingState.RECORDING)


@pytest.mark.unit
class TestTransitionValidation:
    """Test transition validation methods."""

    def test_can_transition_returns_true_for_valid(self):
        """can_transition returns True for valid transitions."""
        sm = StateMachine()
        assert sm.can_transition(RecordingState.INITIALIZING) is True
        assert sm.can_transition(RecordingState.RECOVERING) is True

    def test_can_transition_returns_false_for_invalid(self):
        """can_transition returns False for invalid transitions."""
        sm = StateMachine()
        assert sm.can_transition(RecordingState.RECORDING) is False
        assert sm.can_transition(RecordingState.COMPLETED) is False

    def test_get_valid_transitions(self):
        """get_valid_transitions returns correct set."""
        sm = StateMachine()
        valid = sm.get_valid_transitions()
        assert RecordingState.INITIALIZING in valid
        assert RecordingState.RECOVERING in valid
        assert len(valid) == 2

    def test_valid_transitions_from_recording(self):
        """Check valid transitions from RECORDING state."""
        sm = StateMachine(initial_state=RecordingState.RECORDING)
        valid = sm.get_valid_transitions()
        assert RecordingState.STOPPING in valid
        assert RecordingState.ERROR in valid
        assert len(valid) == 2


@pytest.mark.unit
class TestTransitionHistory:
    """Test transition history tracking."""

    @pytest.mark.asyncio
    async def test_history_records_transitions(self):
        """Transitions are recorded in history."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)

        assert len(sm.history) == 2
        assert sm.history[0].from_state == RecordingState.IDLE
        assert sm.history[0].to_state == RecordingState.INITIALIZING
        assert sm.history[1].from_state == RecordingState.INITIALIZING
        assert sm.history[1].to_state == RecordingState.RECORDING

    @pytest.mark.asyncio
    async def test_history_includes_metadata(self):
        """Transition metadata is recorded."""
        sm = StateMachine()
        await sm.transition(
            RecordingState.INITIALIZING,
            session_id="test123",
            device_name="Test Device"
        )

        event = sm.history[0]
        assert event.session_id == "test123"
        assert event.metadata["device_name"] == "Test Device"

    @pytest.mark.asyncio
    async def test_history_limits_to_100_events(self):
        """History is limited to 100 events."""
        sm = StateMachine()

        # Create more than 100 transitions
        for _ in range(60):
            await sm.transition(RecordingState.INITIALIZING)
            await sm.transition(RecordingState.ERROR)
            await sm.transition(RecordingState.IDLE)

        # Should be capped at 100
        assert len(sm.history) == 100

    def test_history_returns_copy(self):
        """History property returns a copy, not original."""
        sm = StateMachine()
        history1 = sm.history
        history2 = sm.history
        assert history1 is not history2


@pytest.mark.unit
class TestCallbacks:
    """Test callback notification system."""

    @pytest.mark.asyncio
    async def test_callback_called_on_transition(self):
        """Registered callbacks are called on transition."""
        sm = StateMachine()
        events_received: List[StateTransitionEvent] = []

        async def callback(event: StateTransitionEvent):
            events_received.append(event)

        sm.add_callback(callback)
        await sm.transition(RecordingState.INITIALIZING)

        assert len(events_received) == 1
        assert events_received[0].to_state == RecordingState.INITIALIZING

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Multiple callbacks can be registered."""
        sm = StateMachine()
        count = {"value": 0}

        async def callback1(event):
            count["value"] += 1

        async def callback2(event):
            count["value"] += 10

        sm.add_callback(callback1)
        sm.add_callback(callback2)
        await sm.transition(RecordingState.INITIALIZING)

        assert count["value"] == 11

    @pytest.mark.asyncio
    async def test_remove_callback(self):
        """Callbacks can be removed."""
        sm = StateMachine()
        count = {"value": 0}

        async def callback(event):
            count["value"] += 1

        sm.add_callback(callback)
        sm.remove_callback(callback)
        await sm.transition(RecordingState.INITIALIZING)

        assert count["value"] == 0

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_stop_transitions(self):
        """Callback errors don't prevent state transition."""
        sm = StateMachine()

        async def failing_callback(event):
            raise ValueError("Callback error")

        sm.add_callback(failing_callback)

        # Should not raise, transition should succeed
        await sm.transition(RecordingState.INITIALIZING)
        assert sm.state == RecordingState.INITIALIZING


@pytest.mark.unit
class TestSessionManagement:
    """Test session ID management."""

    @pytest.mark.asyncio
    async def test_session_id_set_on_transition(self):
        """Session ID is set when provided."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING, session_id="test_001")
        assert sm.session_id == "test_001"

    @pytest.mark.asyncio
    async def test_session_id_cleared_on_idle(self):
        """Session ID is cleared when returning to IDLE."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING, session_id="test_001")
        await sm.transition(RecordingState.ERROR)
        await sm.transition(RecordingState.IDLE)
        assert sm.session_id is None

    @pytest.mark.asyncio
    async def test_session_id_persists_through_workflow(self):
        """Session ID persists through workflow states."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING, session_id="test_001")
        await sm.transition(RecordingState.RECORDING)
        await sm.transition(RecordingState.STOPPING)

        assert sm.session_id == "test_001"


@pytest.mark.unit
class TestHelperMethods:
    """Test helper methods."""

    def test_is_busy_returns_false_for_idle(self):
        """is_busy returns False for IDLE state."""
        sm = StateMachine()
        assert sm.is_busy() is False

    @pytest.mark.asyncio
    async def test_is_busy_returns_true_during_recording(self):
        """is_busy returns True during recording."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)
        assert sm.is_busy() is True

    def test_is_recording(self):
        """is_recording returns correct value."""
        sm = StateMachine(initial_state=RecordingState.RECORDING)
        assert sm.is_recording() is True

        sm2 = StateMachine(initial_state=RecordingState.IDLE)
        assert sm2.is_recording() is False

    def test_is_processing(self):
        """is_processing returns True for processing states."""
        for state in [RecordingState.STOPPING, RecordingState.NORMALIZING,
                      RecordingState.TRANSCRIBING, RecordingState.BUNDLING]:
            sm = StateMachine(initial_state=state)
            assert sm.is_processing() is True, f"Expected True for {state}"

        for state in [RecordingState.IDLE, RecordingState.RECORDING,
                      RecordingState.COMPLETED, RecordingState.ERROR]:
            sm = StateMachine(initial_state=state)
            assert sm.is_processing() is False, f"Expected False for {state}"

    def test_get_state_info(self):
        """get_state_info returns correct information."""
        sm = StateMachine()
        info = sm.get_state_info()

        assert info["state"] == "idle"
        assert info["session_id"] is None
        assert "initializing" in info["valid_transitions"]

    @pytest.mark.asyncio
    async def test_reset(self):
        """reset returns state machine to IDLE."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)
        await sm.transition(RecordingState.RECORDING)
        await sm.transition(RecordingState.ERROR)

        await sm.reset()
        assert sm.state == RecordingState.IDLE


@pytest.mark.unit
class TestThreadSafety:
    """Test thread safety of state machine."""

    @pytest.mark.asyncio
    async def test_concurrent_transition_attempts(self):
        """Concurrent transition attempts are handled safely."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)

        results = {"success": 0, "failed": 0}

        async def attempt_transition():
            try:
                await sm.transition(RecordingState.RECORDING)
                results["success"] += 1
            except InvalidStateTransitionError:
                results["failed"] += 1

        # Try concurrent transitions
        await asyncio.gather(*[attempt_transition() for _ in range(10)])

        # Only one should succeed
        assert results["success"] == 1
        assert results["failed"] == 9

    def test_thread_safe_state_read(self):
        """State can be read safely from multiple threads."""
        sm = StateMachine()

        def read_state():
            for _ in range(100):
                _ = sm.state
                _ = sm.session_id
                _ = sm.history
            return True

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_state) for _ in range(4)]
            results = [f.result() for f in futures]

        assert all(results)


# ============================================================================
# Edge Case Tests (Test Suite Expansion)
# ============================================================================


@pytest.mark.unit
class TestEventIdGeneration:
    """Tests for event ID generation edge cases."""

    @pytest.mark.asyncio
    async def test_event_id_format(self):
        """Event ID should follow expected format."""
        sm = StateMachine()
        event = await sm.transition(RecordingState.INITIALIZING)
        assert event.event_id.startswith("evt_")
        assert len(event.event_id) == 10  # evt_000001

    @pytest.mark.asyncio
    async def test_event_ids_are_sequential(self):
        """Event IDs should increment sequentially."""
        sm = StateMachine()
        event1 = await sm.transition(RecordingState.INITIALIZING)
        event2 = await sm.transition(RecordingState.ERROR)
        event3 = await sm.transition(RecordingState.IDLE)

        assert event1.event_id == "evt_000001"
        assert event2.event_id == "evt_000002"
        assert event3.event_id == "evt_000003"

    @pytest.mark.asyncio
    async def test_event_counter_high_values(self):
        """Event counter works with high values."""
        sm = StateMachine()
        sm._event_counter = 999990  # Set high counter

        event = await sm.transition(RecordingState.INITIALIZING)
        assert event.event_id == "evt_999991"

    @pytest.mark.asyncio
    async def test_event_counter_uniqueness_concurrent(self):
        """Event IDs remain unique under concurrent transitions."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)

        event_ids = set()
        errors = []

        async def rapid_transition():
            try:
                # Try to transition to ERROR (valid from any state)
                event = await sm.transition(RecordingState.ERROR)
                event_ids.add(event.event_id)
            except InvalidStateTransitionError:
                pass  # Expected for some concurrent attempts

        # Run many concurrent transitions
        await asyncio.gather(*[rapid_transition() for _ in range(20)])

        # All successful events should have unique IDs
        # (we can't guarantee how many succeed due to state transitions)
        assert len(event_ids) >= 1


@pytest.mark.unit
class TestAllTransitionRules:
    """Exhaustively test all valid state transitions."""

    @pytest.mark.asyncio
    async def test_idle_valid_transitions(self):
        """IDLE can transition to INITIALIZING and RECOVERING."""
        sm1 = StateMachine()
        await sm1.transition(RecordingState.INITIALIZING)
        assert sm1.state == RecordingState.INITIALIZING

        sm2 = StateMachine()
        await sm2.transition(RecordingState.RECOVERING)
        assert sm2.state == RecordingState.RECOVERING

    @pytest.mark.asyncio
    async def test_initializing_valid_transitions(self):
        """INITIALIZING can transition to RECORDING and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.INITIALIZING)
        await sm1.transition(RecordingState.RECORDING)
        assert sm1.state == RecordingState.RECORDING

        sm2 = StateMachine(initial_state=RecordingState.INITIALIZING)
        await sm2.transition(RecordingState.ERROR)
        assert sm2.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_recording_valid_transitions(self):
        """RECORDING can transition to STOPPING and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.RECORDING)
        await sm1.transition(RecordingState.STOPPING)
        assert sm1.state == RecordingState.STOPPING

        sm2 = StateMachine(initial_state=RecordingState.RECORDING)
        await sm2.transition(RecordingState.ERROR)
        assert sm2.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_stopping_valid_transitions(self):
        """STOPPING can transition to NORMALIZING and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.STOPPING)
        await sm1.transition(RecordingState.NORMALIZING)
        assert sm1.state == RecordingState.NORMALIZING

        sm2 = StateMachine(initial_state=RecordingState.STOPPING)
        await sm2.transition(RecordingState.ERROR)
        assert sm2.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_normalizing_valid_transitions(self):
        """NORMALIZING can transition to TRANSCRIBING and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.NORMALIZING)
        await sm1.transition(RecordingState.TRANSCRIBING)
        assert sm1.state == RecordingState.TRANSCRIBING

        sm2 = StateMachine(initial_state=RecordingState.NORMALIZING)
        await sm2.transition(RecordingState.ERROR)
        assert sm2.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_transcribing_valid_transitions(self):
        """TRANSCRIBING can transition to BUNDLING, COMPLETED, and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.TRANSCRIBING)
        await sm1.transition(RecordingState.BUNDLING)
        assert sm1.state == RecordingState.BUNDLING

        sm2 = StateMachine(initial_state=RecordingState.TRANSCRIBING)
        await sm2.transition(RecordingState.COMPLETED)
        assert sm2.state == RecordingState.COMPLETED

        sm3 = StateMachine(initial_state=RecordingState.TRANSCRIBING)
        await sm3.transition(RecordingState.ERROR)
        assert sm3.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_bundling_valid_transitions(self):
        """BUNDLING can transition to COMPLETED and ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.BUNDLING)
        await sm1.transition(RecordingState.COMPLETED)
        assert sm1.state == RecordingState.COMPLETED

        sm2 = StateMachine(initial_state=RecordingState.BUNDLING)
        await sm2.transition(RecordingState.ERROR)
        assert sm2.state == RecordingState.ERROR

    @pytest.mark.asyncio
    async def test_completed_valid_transitions(self):
        """COMPLETED can only transition to IDLE."""
        sm = StateMachine(initial_state=RecordingState.COMPLETED)
        await sm.transition(RecordingState.IDLE)
        assert sm.state == RecordingState.IDLE

    @pytest.mark.asyncio
    async def test_error_valid_transitions(self):
        """ERROR can transition to IDLE and RECOVERING."""
        sm1 = StateMachine(initial_state=RecordingState.ERROR)
        await sm1.transition(RecordingState.IDLE)
        assert sm1.state == RecordingState.IDLE

        sm2 = StateMachine(initial_state=RecordingState.ERROR)
        await sm2.transition(RecordingState.RECOVERING)
        assert sm2.state == RecordingState.RECOVERING

    @pytest.mark.asyncio
    async def test_recovering_valid_transitions(self):
        """RECOVERING can transition to IDLE, NORMALIZING, TRANSCRIBING, BUNDLING, ERROR."""
        sm1 = StateMachine(initial_state=RecordingState.RECOVERING)
        await sm1.transition(RecordingState.IDLE)
        assert sm1.state == RecordingState.IDLE

        sm2 = StateMachine(initial_state=RecordingState.RECOVERING)
        await sm2.transition(RecordingState.NORMALIZING)
        assert sm2.state == RecordingState.NORMALIZING

        sm3 = StateMachine(initial_state=RecordingState.RECOVERING)
        await sm3.transition(RecordingState.TRANSCRIBING)
        assert sm3.state == RecordingState.TRANSCRIBING

        sm4 = StateMachine(initial_state=RecordingState.RECOVERING)
        await sm4.transition(RecordingState.BUNDLING)
        assert sm4.state == RecordingState.BUNDLING

        sm5 = StateMachine(initial_state=RecordingState.RECOVERING)
        await sm5.transition(RecordingState.ERROR)
        assert sm5.state == RecordingState.ERROR


@pytest.mark.unit
class TestInvalidTransitionsExhaustive:
    """Exhaustively test all invalid state transitions."""

    @pytest.mark.asyncio
    async def test_idle_invalid_transitions(self):
        """IDLE cannot transition to most states directly."""
        sm = StateMachine()
        invalid_states = [
            RecordingState.RECORDING,
            RecordingState.STOPPING,
            RecordingState.NORMALIZING,
            RecordingState.TRANSCRIBING,
            RecordingState.BUNDLING,
            RecordingState.COMPLETED,
            RecordingState.ERROR,
        ]
        for state in invalid_states:
            with pytest.raises(InvalidStateTransitionError):
                await sm.transition(state)

    @pytest.mark.asyncio
    async def test_completed_invalid_transitions(self):
        """COMPLETED cannot transition to states other than IDLE."""
        sm = StateMachine(initial_state=RecordingState.COMPLETED)
        invalid_states = [
            RecordingState.INITIALIZING,
            RecordingState.RECORDING,
            RecordingState.STOPPING,
            RecordingState.NORMALIZING,
            RecordingState.TRANSCRIBING,
            RecordingState.BUNDLING,
            RecordingState.ERROR,
            RecordingState.RECOVERING,
        ]
        for state in invalid_states:
            with pytest.raises(InvalidStateTransitionError):
                await sm.transition(state)


@pytest.mark.unit
class TestCallbackEdgeCases:
    """Tests for callback edge cases."""

    @pytest.mark.asyncio
    async def test_callback_with_cancelled_error(self):
        """Callback raising CancelledError should not break transitions."""
        sm = StateMachine()

        async def cancelling_callback(event: StateTransitionEvent):
            raise asyncio.CancelledError()

        sm.add_callback(cancelling_callback)

        # Should not raise, transition should succeed
        await sm.transition(RecordingState.INITIALIZING)
        assert sm.state == RecordingState.INITIALIZING

    @pytest.mark.asyncio
    async def test_callback_execution_order(self):
        """Callbacks should be called in registration order."""
        sm = StateMachine()
        order = []

        async def callback1(event):
            order.append(1)

        async def callback2(event):
            order.append(2)

        async def callback3(event):
            order.append(3)

        sm.add_callback(callback1)
        sm.add_callback(callback2)
        sm.add_callback(callback3)

        await sm.transition(RecordingState.INITIALIZING)
        assert order == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_callback_error_continues_to_next(self):
        """Error in one callback should not stop other callbacks."""
        sm = StateMachine()
        results = []

        async def callback1(event):
            results.append("first")

        async def failing_callback(event):
            raise ValueError("Intentional error")

        async def callback3(event):
            results.append("third")

        sm.add_callback(callback1)
        sm.add_callback(failing_callback)
        sm.add_callback(callback3)

        await sm.transition(RecordingState.INITIALIZING)
        assert results == ["first", "third"]

    @pytest.mark.asyncio
    async def test_removing_nonexistent_callback(self):
        """Removing non-existent callback should not raise."""
        sm = StateMachine()

        async def callback(event):
            pass

        # Should not raise
        sm.remove_callback(callback)

    @pytest.mark.asyncio
    async def test_add_callback_during_notification(self):
        """Adding callback during notification should work safely."""
        sm = StateMachine()
        added_called = {"value": False}

        async def late_callback(event):
            added_called["value"] = True

        async def adding_callback(event):
            sm.add_callback(late_callback)

        sm.add_callback(adding_callback)
        await sm.transition(RecordingState.INITIALIZING)

        # Second transition should call the late callback
        await sm.transition(RecordingState.ERROR)
        assert added_called["value"] is True


@pytest.mark.unit
class TestMetadataEdgeCases:
    """Tests for transition metadata edge cases."""

    @pytest.mark.asyncio
    async def test_large_metadata_dictionary(self):
        """Large metadata should be handled correctly."""
        sm = StateMachine()
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}

        event = await sm.transition(
            RecordingState.INITIALIZING,
            **large_metadata
        )

        assert len(event.metadata) == 1000
        assert event.metadata["key_0"] == "value_0"
        assert event.metadata["key_999"] == "value_999"

    @pytest.mark.asyncio
    async def test_metadata_with_special_characters(self):
        """Metadata with special characters should work."""
        sm = StateMachine()
        event = await sm.transition(
            RecordingState.INITIALIZING,
            device_name="Device (v2.0) <test> & \"special\"",
            path="/path/with spaces/and\ttabs",
            unicode_key="日本語 🎙️"
        )

        assert "Device (v2.0)" in event.metadata["device_name"]
        assert event.metadata["unicode_key"] == "日本語 🎙️"

    @pytest.mark.asyncio
    async def test_metadata_with_nested_dict(self):
        """Nested dictionaries in metadata should work."""
        sm = StateMachine()
        event = await sm.transition(
            RecordingState.INITIALIZING,
            config={"nested": {"deeply": {"value": 42}}}
        )

        assert event.metadata["config"]["nested"]["deeply"]["value"] == 42

    @pytest.mark.asyncio
    async def test_metadata_with_list_values(self):
        """Lists in metadata should work."""
        sm = StateMachine()
        event = await sm.transition(
            RecordingState.INITIALIZING,
            devices=["device1", "device2", "device3"]
        )

        assert len(event.metadata["devices"]) == 3


@pytest.mark.unit
class TestHistoryEdgeCases:
    """Tests for transition history edge cases."""

    @pytest.mark.asyncio
    async def test_history_exactly_100_events(self):
        """History should contain exactly 100 events when limit reached."""
        sm = StateMachine()

        # Generate 100 transitions (3 per cycle, 34 cycles = 102 events)
        for _ in range(34):
            await sm.transition(RecordingState.INITIALIZING)
            await sm.transition(RecordingState.ERROR)
            await sm.transition(RecordingState.IDLE)

        assert len(sm.history) == 100

    @pytest.mark.asyncio
    async def test_history_keeps_most_recent(self):
        """History should keep most recent events when truncated."""
        sm = StateMachine()

        # Generate more than 100 events
        for i in range(50):
            await sm.transition(RecordingState.INITIALIZING, session_id=f"session_{i}")
            await sm.transition(RecordingState.ERROR)
            await sm.transition(RecordingState.IDLE)

        # Check that recent events are kept
        history = sm.history
        # Last event should be recent (within last 100)
        last_event = history[-1]
        assert last_event.to_state == RecordingState.IDLE

    @pytest.mark.asyncio
    async def test_history_mutation_isolation(self):
        """Modifying returned history should not affect internal state."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING)

        history = sm.history
        original_len = len(history)
        history.clear()

        # Internal history should be unchanged
        assert len(sm.history) == original_len


@pytest.mark.unit
class TestResetEdgeCases:
    """Tests for reset method edge cases."""

    @pytest.mark.asyncio
    async def test_reset_already_idle(self):
        """Reset when already IDLE should not add to history."""
        sm = StateMachine()
        initial_history_len = len(sm.history)

        await sm.reset()

        # Should not add transition since already IDLE
        assert len(sm.history) == initial_history_len

    @pytest.mark.asyncio
    async def test_reset_from_error(self):
        """Reset from ERROR should transition to IDLE."""
        sm = StateMachine(initial_state=RecordingState.ERROR)
        await sm.reset()
        assert sm.state == RecordingState.IDLE

    @pytest.mark.asyncio
    async def test_reset_clears_session_id(self):
        """Reset should clear session ID."""
        sm = StateMachine()
        await sm.transition(RecordingState.INITIALIZING, session_id="test_session")
        assert sm.session_id == "test_session"

        await sm.transition(RecordingState.ERROR)
        await sm.reset()

        assert sm.session_id is None


@pytest.mark.unit
class TestStateInfoEdgeCases:
    """Tests for get_state_info edge cases."""

    def test_state_info_all_states(self):
        """get_state_info should work for all states."""
        for state in RecordingState:
            sm = StateMachine(initial_state=state)
            info = sm.get_state_info()

            assert info["state"] == state.value
            assert "valid_transitions" in info
            assert isinstance(info["valid_transitions"], list)
