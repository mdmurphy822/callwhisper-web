"""
Platform Detection Utilities.

Provides cross-platform abstractions for audio backend selection
and platform-specific behavior.

Supported platforms:
- Windows: DirectShow (dshow)
- macOS: AVFoundation (avfoundation)
- Linux: PulseAudio (pulse) or ALSA (alsa)
"""

import sys
import shutil
from functools import lru_cache
from typing import Literal

AudioBackend = Literal["dshow", "avfoundation", "pulse", "alsa"]
Platform = Literal["windows", "macos", "linux"]


@lru_cache
def get_platform() -> Platform:
    """
    Return normalized platform name.

    Returns:
        "windows", "macos", or "linux"
    """
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    return "linux"


@lru_cache
def get_audio_backend() -> AudioBackend:
    """
    Return appropriate FFmpeg audio backend for current platform.

    On Linux, prefers PulseAudio if available, falls back to ALSA.

    Returns:
        FFmpeg input format: "dshow", "avfoundation", "pulse", or "alsa"
    """
    if sys.platform == "win32":
        return "dshow"
    elif sys.platform == "darwin":
        return "avfoundation"
    else:
        # Linux: prefer PulseAudio, fall back to ALSA
        if shutil.which("pactl"):
            return "pulse"
        return "alsa"


def is_windows() -> bool:
    """Check if running on Windows."""
    return get_platform() == "windows"


def is_macos() -> bool:
    """Check if running on macOS."""
    return get_platform() == "macos"


def is_linux() -> bool:
    """Check if running on Linux."""
    return get_platform() == "linux"


def has_pulseaudio() -> bool:
    """Check if PulseAudio is available on the system."""
    return shutil.which("pactl") is not None


def has_pipewire() -> bool:
    """Check if PipeWire is available on the system."""
    return shutil.which("pw-cli") is not None


def get_audio_system_info() -> dict:
    """
    Get information about the audio system.

    Returns:
        Dict with platform, backend, and availability info.
    """
    return {
        "platform": get_platform(),
        "audio_backend": get_audio_backend(),
        "has_pulseaudio": has_pulseaudio(),
        "has_pipewire": has_pipewire(),
    }
