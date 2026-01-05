"""
Custom Exceptions for CallWhisper

Based on LibV2 Python course patterns:
- Use specific exception types
- Provide meaningful error context
- Enable proper error handling chains
"""

from typing import Optional, Dict, Any


class CallWhisperError(Exception):
    """Base exception for all CallWhisper errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        base = self.message
        if self.details:
            base += f" | Details: {self.details}"
        if self.cause:
            base += f" | Caused by: {type(self.cause).__name__}: {self.cause}"
        return base


# Device-related exceptions
class DeviceError(CallWhisperError):
    """Base exception for device-related errors."""

    pass


class DeviceNotFoundError(DeviceError):
    """Raised when a requested device is not found."""

    def __init__(self, device_name: str, available_devices: Optional[list] = None):
        super().__init__(
            f"Device not found: '{device_name}'",
            details={
                "requested_device": device_name,
                "available_devices": available_devices or [],
            },
        )


class DeviceBlockedError(DeviceError):
    """Raised when attempting to record from a blocked device (e.g., microphone)."""

    def __init__(self, device_name: str, reason: str, match_type: str):
        super().__init__(
            f"Device blocked: '{device_name}'",
            details={
                "device_name": device_name,
                "reason": reason,
                "match_type": match_type,
            },
        )


class DeviceEnumerationError(DeviceError):
    """Raised when device enumeration fails."""

    pass


# Recording-related exceptions
class RecordingError(CallWhisperError):
    """Base exception for recording-related errors."""

    pass


class RecordingStartError(RecordingError):
    """Raised when recording fails to start."""

    def __init__(
        self, device_name: str, reason: str, cause: Optional[Exception] = None
    ):
        super().__init__(
            f"Failed to start recording on '{device_name}': {reason}",
            details={"device_name": device_name, "reason": reason},
            cause=cause,
        )


class RecordingStopError(RecordingError):
    """Raised when recording fails to stop gracefully."""

    pass


class RecordingInProgressError(RecordingError):
    """Raised when attempting to start recording while already recording."""

    def __init__(self, current_session_id: str):
        super().__init__(
            "Recording already in progress",
            details={"current_session_id": current_session_id},
        )


# Transcription-related exceptions
class TranscriptionError(CallWhisperError):
    """Base exception for transcription-related errors."""

    pass


class TranscriptionModelNotFoundError(TranscriptionError):
    """Raised when the transcription model is not found."""

    def __init__(self, model_path: str, models_dir: str):
        super().__init__(
            f"Transcription model not found: '{model_path}'",
            details={
                "model_path": model_path,
                "models_dir": models_dir,
                "suggestion": "Place a whisper model (e.g., ggml-medium.en.bin) in the models/ directory",
            },
        )


class TranscriptionProcessError(TranscriptionError):
    """Raised when the transcription process fails."""

    def __init__(
        self, reason: str, exit_code: Optional[int] = None, stderr: Optional[str] = None
    ):
        super().__init__(
            f"Transcription failed: {reason}",
            details={
                "exit_code": exit_code,
                "stderr": stderr[:500] if stderr else None,  # Truncate long stderr
            },
        )


class TranscriptionTimeoutError(TranscriptionError):
    """Raised when transcription exceeds timeout."""

    def __init__(self, timeout_seconds: float, audio_duration: Optional[float] = None):
        super().__init__(
            f"Transcription timed out after {timeout_seconds}s",
            details={
                "timeout_seconds": timeout_seconds,
                "audio_duration": audio_duration,
            },
        )


# Bundle-related exceptions
class BundleError(CallWhisperError):
    """Base exception for bundle-related errors."""

    pass


class BundleCreationError(BundleError):
    """Raised when bundle creation fails."""

    pass


class BundleExtractionError(BundleError):
    """Raised when bundle extraction fails."""

    pass


class BundleValidationError(BundleError):
    """Raised when bundle validation fails (e.g., hash mismatch)."""

    def __init__(self, bundle_path: str, failed_files: Dict[str, bool]):
        super().__init__(
            f"Bundle validation failed: '{bundle_path}'",
            details={
                "bundle_path": bundle_path,
                "failed_files": [f for f, valid in failed_files.items() if not valid],
            },
        )


# State-related exceptions
class StateError(CallWhisperError):
    """Base exception for state-related errors."""

    pass


class InvalidStateTransitionError(StateError):
    """Raised when attempting an invalid state transition."""

    def __init__(self, from_state: str, to_state: str, valid_transitions: list):
        super().__init__(
            f"Invalid state transition: {from_state} → {to_state}",
            details={
                "from_state": from_state,
                "to_state": to_state,
                "valid_transitions": valid_transitions,
            },
        )


class StateRecoveryError(StateError):
    """Raised when state recovery fails."""

    pass


# Process-related exceptions
class ProcessError(CallWhisperError):
    """Base exception for subprocess-related errors."""

    pass


class ProcessTimeoutError(ProcessError):
    """Raised when a subprocess times out or request deadline exceeded."""

    def __init__(
        self,
        message: str,
        process_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        task_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        details = {}
        if process_name:
            details["process_name"] = process_name
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if task_type:
            details["task_type"] = task_type
        if correlation_id:
            details["correlation_id"] = correlation_id

        super().__init__(message, details=details)


class ProcessNotFoundError(ProcessError):
    """Raised when a required executable is not found."""

    def __init__(self, executable_name: str, expected_path: str):
        super().__init__(
            f"Executable not found: '{executable_name}'",
            details={
                "executable_name": executable_name,
                "expected_path": expected_path,
                "suggestion": f"Place {executable_name} in the vendor/ directory",
            },
        )


class CircuitOpenError(ProcessError):
    """Raised when circuit breaker is open and requests are rejected."""

    def __init__(
        self, handler_name: str, failure_count: int, cooldown_remaining: float
    ):
        super().__init__(
            f"Circuit breaker open for '{handler_name}'",
            details={
                "handler_name": handler_name,
                "failure_count": failure_count,
                "cooldown_remaining_seconds": cooldown_remaining,
            },
        )


class AllHandlersFailedError(ProcessError):
    """Raised when all handlers in a fallback chain fail."""

    def __init__(
        self, task_type: str, attempts: list, correlation_id: Optional[str] = None
    ):
        details = {"task_type": task_type, "attempts": attempts}
        if correlation_id:
            details["correlation_id"] = correlation_id

        super().__init__(
            f"All handlers failed for task type '{task_type}'", details=details
        )


# Audio processing exceptions
class AudioDurationError(CallWhisperError):
    """Raised when audio duration cannot be determined."""

    def __init__(self, audio_path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Could not determine audio duration for: '{audio_path}'",
            details={"audio_path": audio_path},
            cause=cause,
        )


# Persistence exceptions
class MetricsPersistenceError(CallWhisperError):
    """Raised when metrics cannot be saved or loaded."""

    def __init__(self, operation: str, path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Metrics {operation} failed: '{path}'",
            details={"operation": operation, "path": path},
            cause=cause,
        )


class CheckpointCorruptedError(CallWhisperError):
    """Raised when a checkpoint file is corrupted or malformed."""

    def __init__(self, path: str, cause: Optional[Exception] = None):
        super().__init__(
            f"Corrupted checkpoint file: '{path}'", details={"path": path}, cause=cause
        )


# Security exceptions
class PathTraversalError(CallWhisperError):
    """Raised when a path traversal attack is detected."""

    def __init__(self, value: str, reason: str):
        super().__init__(
            f"Path traversal detected: {reason}",
            details={"value": value, "reason": reason},
        )
