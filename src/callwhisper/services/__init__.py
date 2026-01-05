"""
CallWhisper Services Module.

Contains services for automatic call detection:
- WindowsAudioMonitor: WASAPI audio session monitoring
- ProcessMonitor: WMI process lifecycle tracking
- CallDetector: Orchestrates call detection from multiple signals
"""

import sys

# Only expose Windows-specific services on Windows
if sys.platform == "win32":
    from .windows_audio_monitor import (
        WindowsAudioSessionMonitor,
        AudioSessionState,
        AudioSessionEvent,
    )
    from .process_monitor import ProcessMonitor, ProcessEvent
    from .call_detector import CallDetector, CallState

    __all__ = [
        "WindowsAudioSessionMonitor",
        "AudioSessionState",
        "AudioSessionEvent",
        "ProcessMonitor",
        "ProcessEvent",
        "CallDetector",
        "CallState",
    ]
else:
    # Provide stub classes for non-Windows platforms
    __all__ = []
