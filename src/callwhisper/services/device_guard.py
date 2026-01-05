"""
Device Guard - Microphone Recording Prevention

CRITICAL SAFETY MODULE

This module ensures the application NEVER records from a microphone.
Only approved output/loopback devices are allowed.

Supports Windows (VB-Cable, Stereo Mix) and Linux (PulseAudio monitors, PipeWire).
"""

import re
from typing import Dict, Any, List

from ..core.config import DeviceGuardConfig
from ..utils.platform import get_platform


# Windows safe device patterns (case-insensitive)
SAFE_PATTERNS_WINDOWS = [
    r"vb-?cable",
    r"cable\s*output",
    r"virtual\s*(audio\s*)?cable",
    r"stereo\s*mix",
    r"what\s*u\s*hear",
    r"voicemeeter",
    r"jabber",
    r"finesse",
    r"cisco",
    r"loopback",
    r"speakers?\s*\(",  # "Speakers (Device)" type names with output indicator
]

# Linux safe device patterns (case-insensitive)
SAFE_PATTERNS_LINUX = [
    r"\.monitor$",  # PulseAudio monitor sinks (ends with .monitor)
    r"monitor$",  # Monitor suffix
    r"null[-_]?sink",  # Null sink
    r"loopback",  # Loopback device
    r"pipewire",  # PipeWire devices
    r"output.*monitor",  # Output monitors
    r"alsa_output.*monitor",  # ALSA output monitors
]


def _get_safe_patterns() -> List[str]:
    """Get safe device patterns for current platform."""
    platform = get_platform()
    if platform == "windows":
        return SAFE_PATTERNS_WINDOWS
    return SAFE_PATTERNS_LINUX


# Combined for backward compatibility
DEFAULT_SAFE_PATTERNS = SAFE_PATTERNS_WINDOWS + SAFE_PATTERNS_LINUX

# Dangerous device patterns (case-insensitive) - MUST BE BLOCKED
# Uses negative lookahead to allow "output" variations like "Microphone Output"
DANGEROUS_PATTERNS = [
    r"\bmicrophone\b(?!\s*out)",  # "microphone" but NOT "microphone output"
    r"\bmic\b(?!\s*out)",  # "mic" but NOT "mic out" or "mic output"
    r"^mic\s(?!out)",  # starts with "mic " but not "mic out"
    r"webcam",
    r"\bcamera\b",
    r"headset\s*mic(?!\s*out)",  # "headset mic" but NOT "headset mic output"
    r"built-?in\s*(mic|audio\s*input)",
    r"internal\s*mic(?!\s*out)",
    r"array\s*mic(?!\s*out)",
    r"\binput\s*device\b",
    r"realtek.*\binput\b",
    r"\brecording\s*device\b",
]


def is_device_safe(device_name: str, config: DeviceGuardConfig) -> bool:
    """
    Check if a device is safe to record from.

    SAFETY LOGIC:
    1. Check explicit blocklist first (always blocked)
    2. Check against dangerous patterns (always blocked)
    3. Check explicit allowlist (allowed if match)
    4. Check against safe patterns (allowed if match)
    5. Default: BLOCKED (fail-safe)

    Args:
        device_name: Name of the audio device
        config: Device guard configuration

    Returns:
        True if device is safe to record from
    """
    if not config.enabled:
        return True  # Guard disabled (not recommended)

    name_lower = device_name.lower()

    # 1. Check explicit blocklist with word boundary matching
    for blocked in config.blocklist:
        blocked_lower = blocked.lower()
        # Use word boundary match to avoid false positives
        # e.g., "microphone" in blocklist should not block "microphone output"
        pattern = r"\b" + re.escape(blocked_lower) + r"\b(?!\s*out)"
        if re.search(pattern, name_lower, re.IGNORECASE):
            return False

    # 2. Check dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return False

    # 3. Check explicit allowlist
    for allowed in config.allowlist:
        if allowed.lower() in name_lower:
            return True

    # 4. Check safe patterns (platform-specific)
    safe_patterns = _get_safe_patterns()
    for pattern in safe_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return True

    # 5. Default: BLOCKED (fail-safe)
    return False


def get_device_status(device_name: str, config: DeviceGuardConfig) -> Dict[str, Any]:
    """
    Get detailed safety status for a device.

    Args:
        device_name: Name of the audio device
        config: Device guard configuration

    Returns:
        Dict with 'safe' boolean and 'reason' explanation
    """
    if not config.enabled:
        return {
            "safe": True,
            "reason": "Device guard disabled",
            "match_type": "disabled",
        }

    name_lower = device_name.lower()

    # Check explicit blocklist with word boundary matching
    for blocked in config.blocklist:
        blocked_lower = blocked.lower()
        pattern = r"\b" + re.escape(blocked_lower) + r"\b(?!\s*out)"
        if re.search(pattern, name_lower, re.IGNORECASE):
            return {
                "safe": False,
                "reason": f"Blocked by config: '{blocked}'",
                "match_type": "blocklist",
            }

    # Check dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return {
                "safe": False,
                "reason": "Matches dangerous pattern: microphone/input device",
                "match_type": "dangerous_pattern",
            }

    # Check explicit allowlist
    for allowed in config.allowlist:
        if allowed.lower() in name_lower:
            return {
                "safe": True,
                "reason": f"Allowed by config: '{allowed}'",
                "match_type": "allowlist",
            }

    # Check safe patterns (platform-specific)
    safe_patterns = _get_safe_patterns()
    for pattern in safe_patterns:
        if re.search(pattern, name_lower, re.IGNORECASE):
            return {
                "safe": True,
                "reason": "Matches safe output device pattern",
                "match_type": "safe_pattern",
            }

    # Default: blocked
    platform = get_platform()
    if platform == "windows":
        suggestion = "Add to allowlist in config or install VB-Cable."
    else:
        suggestion = "Add to allowlist in config or use a PulseAudio monitor sink."

    return {
        "safe": False,
        "reason": f"Not recognized as safe output device. {suggestion}",
        "match_type": "default_block",
    }


def validate_device_for_recording(device_name: str, config: DeviceGuardConfig) -> None:
    """
    Validate a device is safe for recording.

    Raises:
        ValueError: If device is not safe to record from

    Args:
        device_name: Name of the audio device
        config: Device guard configuration
    """
    status = get_device_status(device_name, config)

    if not status["safe"]:
        raise ValueError(
            f"BLOCKED: Cannot record from '{device_name}'. "
            f"Reason: {status['reason']}. "
            f"This application only records from approved output devices, "
            f"never from microphones."
        )


def get_safe_devices(devices: List[str], config: DeviceGuardConfig) -> List[str]:
    """
    Filter a list of devices to only safe ones.

    Args:
        devices: List of device names
        config: Device guard configuration

    Returns:
        List of safe device names
    """
    return [d for d in devices if is_device_safe(d, config)]


def log_device_decision(device_name: str, config: DeviceGuardConfig) -> str:
    """
    Generate a log message for device selection decision.

    Args:
        device_name: Name of the audio device
        config: Device guard configuration

    Returns:
        Log message describing the decision
    """
    status = get_device_status(device_name, config)
    action = "ALLOWED" if status["safe"] else "BLOCKED"

    return (
        f"Device Guard: {action} - '{device_name}' "
        f"(match_type: {status['match_type']}, reason: {status['reason']})"
    )
