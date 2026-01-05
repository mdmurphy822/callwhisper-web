"""
Unit tests for custom exception hierarchy.

Based on LibV2 Python programming course patterns:
- Test exception inheritance
- Test error message formatting
- Test details and context
"""

import pytest

from callwhisper.core.exceptions import (
    CallWhisperError,
    DeviceError,
    DeviceNotFoundError,
    DeviceBlockedError,
    RecordingError,
    RecordingStartError,
    RecordingInProgressError,
    TranscriptionError,
    TranscriptionModelNotFoundError,
    TranscriptionProcessError,
    TranscriptionTimeoutError,
    BundleError,
    BundleValidationError,
    StateError,
    InvalidStateTransitionError,
    ProcessError,
    ProcessTimeoutError,
    ProcessNotFoundError,
    CircuitOpenError,
    AllHandlersFailedError,
)


@pytest.mark.unit
class TestExceptionHierarchy:
    """Test exception inheritance chain."""

    def test_all_exceptions_inherit_from_base(self):
        """All custom exceptions inherit from CallWhisperError."""
        exceptions = [
            DeviceError,
            DeviceNotFoundError,
            DeviceBlockedError,
            RecordingError,
            RecordingStartError,
            TranscriptionError,
            BundleError,
            StateError,
            ProcessError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, CallWhisperError)

    def test_device_exceptions_inherit_from_device_error(self):
        """Device exceptions inherit from DeviceError."""
        assert issubclass(DeviceNotFoundError, DeviceError)
        assert issubclass(DeviceBlockedError, DeviceError)

    def test_recording_exceptions_inherit_from_recording_error(self):
        """Recording exceptions inherit from RecordingError."""
        assert issubclass(RecordingStartError, RecordingError)
        assert issubclass(RecordingInProgressError, RecordingError)

    def test_transcription_exceptions_inherit_from_transcription_error(self):
        """Transcription exceptions inherit from TranscriptionError."""
        assert issubclass(TranscriptionModelNotFoundError, TranscriptionError)
        assert issubclass(TranscriptionProcessError, TranscriptionError)
        assert issubclass(TranscriptionTimeoutError, TranscriptionError)


@pytest.mark.unit
class TestCallWhisperError:
    """Test base exception class."""

    def test_basic_message(self):
        """Exception stores message correctly."""
        exc = CallWhisperError("Test error")
        assert exc.message == "Test error"
        assert str(exc) == "Test error"

    def test_with_details(self):
        """Exception includes details in string."""
        exc = CallWhisperError("Test error", details={"key": "value"})
        assert exc.details == {"key": "value"}
        assert "Details:" in str(exc)
        assert "key" in str(exc)

    def test_with_cause(self):
        """Exception includes cause in string."""
        cause = ValueError("Original error")
        exc = CallWhisperError("Wrapped error", cause=cause)
        assert exc.cause == cause
        assert "Caused by: ValueError" in str(exc)

    def test_all_fields(self):
        """Exception handles all fields together."""
        cause = OSError("Disk full")
        exc = CallWhisperError(
            "Write failed",
            details={"file": "/tmp/test"},
            cause=cause
        )
        s = str(exc)
        assert "Write failed" in s
        assert "file" in s
        assert "Caused by: OSError" in s


@pytest.mark.unit
class TestDeviceExceptions:
    """Test device-related exceptions."""

    def test_device_not_found_error(self):
        """DeviceNotFoundError includes device info."""
        exc = DeviceNotFoundError(
            "My Device",
            available_devices=["Device A", "Device B"]
        )
        assert exc.details["requested_device"] == "My Device"
        assert exc.details["available_devices"] == ["Device A", "Device B"]
        assert "My Device" in str(exc)

    def test_device_blocked_error(self):
        """DeviceBlockedError includes blocking reason."""
        exc = DeviceBlockedError(
            "Microphone",
            reason="Matches blocklist pattern",
            match_type="regex"
        )
        assert exc.details["device_name"] == "Microphone"
        assert exc.details["reason"] == "Matches blocklist pattern"
        assert exc.details["match_type"] == "regex"


@pytest.mark.unit
class TestRecordingExceptions:
    """Test recording-related exceptions."""

    def test_recording_start_error(self):
        """RecordingStartError includes device and reason."""
        exc = RecordingStartError(
            "Stereo Mix",
            reason="Device busy",
            cause=OSError("Resource unavailable")
        )
        assert exc.details["device_name"] == "Stereo Mix"
        assert exc.details["reason"] == "Device busy"
        assert "Caused by: OSError" in str(exc)

    def test_recording_in_progress_error(self):
        """RecordingInProgressError includes session ID."""
        exc = RecordingInProgressError("session_123")
        assert exc.details["current_session_id"] == "session_123"
        assert "already in progress" in str(exc).lower()


@pytest.mark.unit
class TestTranscriptionExceptions:
    """Test transcription-related exceptions."""

    def test_model_not_found_error(self):
        """TranscriptionModelNotFoundError includes paths."""
        exc = TranscriptionModelNotFoundError(
            "/path/to/model.bin",
            "/path/to/models"
        )
        assert exc.details["model_path"] == "/path/to/model.bin"
        assert exc.details["models_dir"] == "/path/to/models"
        assert "suggestion" in exc.details

    def test_process_error(self):
        """TranscriptionProcessError includes exit code and stderr."""
        exc = TranscriptionProcessError(
            "Whisper crashed",
            exit_code=1,
            stderr="Error: memory allocation failed"
        )
        assert exc.details["exit_code"] == 1
        assert "memory allocation" in exc.details["stderr"]

    def test_process_error_truncates_stderr(self):
        """TranscriptionProcessError truncates long stderr."""
        long_stderr = "x" * 1000
        exc = TranscriptionProcessError("Error", stderr=long_stderr)
        assert len(exc.details["stderr"]) == 500

    def test_timeout_error(self):
        """TranscriptionTimeoutError includes timing info."""
        exc = TranscriptionTimeoutError(300.0, audio_duration=120.5)
        assert exc.details["timeout_seconds"] == 300.0
        assert exc.details["audio_duration"] == 120.5


@pytest.mark.unit
class TestBundleExceptions:
    """Test bundle-related exceptions."""

    def test_validation_error(self):
        """BundleValidationError includes failed files."""
        exc = BundleValidationError(
            "/path/bundle.vtb",
            {"audio.wav": True, "transcript.txt": False, "meta.json": True}
        )
        assert exc.details["bundle_path"] == "/path/bundle.vtb"
        assert "transcript.txt" in exc.details["failed_files"]
        assert "audio.wav" not in exc.details["failed_files"]


@pytest.mark.unit
class TestStateExceptions:
    """Test state-related exceptions."""

    def test_invalid_transition_error(self):
        """InvalidStateTransitionError includes transition info."""
        exc = InvalidStateTransitionError(
            from_state="idle",
            to_state="recording",
            valid_transitions=["initializing", "recovering"]
        )
        assert exc.details["from_state"] == "idle"
        assert exc.details["to_state"] == "recording"
        assert "initializing" in exc.details["valid_transitions"]
        assert "idle → recording" in str(exc)


@pytest.mark.unit
class TestProcessExceptions:
    """Test process-related exceptions."""

    def test_process_timeout_error(self):
        """ProcessTimeoutError includes process and timeout."""
        exc = ProcessTimeoutError("ffmpeg timed out", process_name="ffmpeg", timeout_seconds=120.0)
        assert exc.details["process_name"] == "ffmpeg"
        assert exc.details["timeout_seconds"] == 120.0

    def test_process_not_found_error(self):
        """ProcessNotFoundError includes path info."""
        exc = ProcessNotFoundError("whisper-cli", "/usr/bin/whisper-cli")
        assert exc.details["executable_name"] == "whisper-cli"
        assert exc.details["expected_path"] == "/usr/bin/whisper-cli"
        assert "suggestion" in exc.details

    def test_circuit_open_error(self):
        """CircuitOpenError includes circuit state."""
        exc = CircuitOpenError("transcription", failure_count=5, cooldown_remaining=45.5)
        assert exc.details["handler_name"] == "transcription"
        assert exc.details["failure_count"] == 5
        assert exc.details["cooldown_remaining_seconds"] == 45.5

    def test_all_handlers_failed_error(self):
        """AllHandlersFailedError includes attempt history."""
        attempts = [
            {"handler": "large", "error": "timeout"},
            {"handler": "medium", "error": "memory"},
            {"handler": "small", "error": "crash"},
        ]
        exc = AllHandlersFailedError("transcription", attempts)
        assert exc.details["task_type"] == "transcription"
        assert len(exc.details["attempts"]) == 3


@pytest.mark.unit
class TestExceptionCatching:
    """Test exception catching behavior."""

    def test_catch_by_base_type(self):
        """Can catch specific exceptions by base type."""
        exc = DeviceBlockedError("Mic", "blocked", "exact")

        with pytest.raises(CallWhisperError):
            raise exc

    def test_catch_by_intermediate_type(self):
        """Can catch by intermediate type in hierarchy."""
        exc = DeviceBlockedError("Mic", "blocked", "exact")

        with pytest.raises(DeviceError):
            raise exc

    def test_catch_by_specific_type(self):
        """Can catch by specific exception type."""
        exc = DeviceBlockedError("Mic", "blocked", "exact")

        with pytest.raises(DeviceBlockedError) as exc_info:
            raise exc

        assert exc_info.value.details["device_name"] == "Mic"
