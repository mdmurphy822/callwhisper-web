"""
Virtual Audio Device Detector

Detects virtual audio devices for first-run setup and health checks.
Supports Windows (VB-Cable, Stereo Mix) and Linux (PulseAudio monitors, PipeWire).

Based on LibV2 orchestrator-architecture patterns:
- Health Checks: Verification at each installation phase
- Capability Detection: Runtime feature availability

Key concepts:
- Virtual audio cables capture system audio for transcription
- Required for Jabber/Finesse audio capture
- Detection uses platform-specific enumeration
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .device_enum import list_audio_devices
from ..core.logging_config import get_service_logger
from ..utils.platform import get_platform

logger = get_service_logger()

# Platform-specific virtual audio patterns
VIRTUAL_AUDIO_PATTERNS_WINDOWS = [
    "vb-cable",
    "vb-audio",
    "cable output",
    "cable input",
    "virtual",
    "voicemeeter",
    "stereo mix",
    "what u hear",
    "wave out mix",
    "loopback",
]

VIRTUAL_AUDIO_PATTERNS_LINUX = [
    "monitor",           # PulseAudio monitor sinks
    ".monitor",          # PulseAudio monitor suffix
    "null",              # Null sink
    "loopback",          # ALSA/Pulse loopback
    "pipewire",          # PipeWire
    "virtual",           # Generic virtual device
]

# Combined patterns (used for backward compatibility)
VIRTUAL_AUDIO_PATTERNS = VIRTUAL_AUDIO_PATTERNS_WINDOWS + VIRTUAL_AUDIO_PATTERNS_LINUX

# Platform-specific recommended devices
RECOMMENDED_DEVICES_WINDOWS = [
    {
        "name": "VB-Cable",
        "pattern": "cable",
        "website": "https://vb-audio.com/Cable/",
        "description": "Free virtual audio cable driver",
        "installation": "Download and run installer (requires admin)",
    },
    {
        "name": "VoiceMeeter",
        "pattern": "voicemeeter",
        "website": "https://vb-audio.com/Voicemeeter/",
        "description": "Advanced audio mixer with virtual cables",
        "installation": "Download and run installer (requires admin)",
    },
]

RECOMMENDED_DEVICES_LINUX = [
    {
        "name": "PulseAudio Monitor",
        "pattern": "monitor",
        "website": "https://www.freedesktop.org/wiki/Software/PulseAudio/",
        "description": "Built-in system audio capture via monitor sinks",
        "installation": "Usually pre-installed. Use 'pactl list sources' to find monitors.",
    },
    {
        "name": "PipeWire",
        "pattern": "pipewire",
        "website": "https://pipewire.org/",
        "description": "Modern Linux audio system with loopback support",
        "installation": "Install pipewire package from your distribution",
    },
]


def _get_virtual_audio_patterns() -> List[str]:
    """Get virtual audio patterns for current platform."""
    platform = get_platform()
    if platform == "windows":
        return VIRTUAL_AUDIO_PATTERNS_WINDOWS
    return VIRTUAL_AUDIO_PATTERNS_LINUX


def _get_recommended_devices() -> List[Dict[str, str]]:
    """Get recommended devices for current platform."""
    platform = get_platform()
    if platform == "windows":
        return RECOMMENDED_DEVICES_WINDOWS
    return RECOMMENDED_DEVICES_LINUX


# Keep for backward compatibility
RECOMMENDED_DEVICES = _get_recommended_devices()


@dataclass
class VirtualAudioDevice:
    """Detected virtual audio device."""
    name: str
    device_type: str  # "vb-cable", "stereo-mix", "voicemeeter", "other"
    is_recommended: bool


@dataclass
class SetupStatus:
    """First-run setup status."""
    virtual_audio_detected: bool
    detected_devices: List[VirtualAudioDevice]
    all_audio_devices: List[str]
    recommended_action: Optional[str]
    setup_complete: bool


def detect_virtual_audio_devices() -> List[VirtualAudioDevice]:
    """
    Detect installed virtual audio devices.

    Uses platform-specific enumeration and pattern matching.

    Returns:
        List of detected virtual audio devices.
    """
    try:
        all_devices = list_audio_devices()
    except Exception as e:
        logger.error("virtual_audio_detection_error", error=str(e))
        return []

    virtual_devices = []
    platform = get_platform()
    patterns = _get_virtual_audio_patterns()

    for device in all_devices:
        device_lower = device.lower()

        # Check if this is a virtual audio device
        for pattern in patterns:
            if pattern in device_lower:
                # Determine device type based on platform
                device_type, is_recommended = _classify_device(device_lower, platform)

                virtual_devices.append(VirtualAudioDevice(
                    name=device,
                    device_type=device_type,
                    is_recommended=is_recommended,
                ))

                logger.debug(
                    "virtual_audio_device_found",
                    device=device,
                    device_type=device_type,
                    is_recommended=is_recommended,
                    platform=platform,
                )
                break  # Don't match multiple patterns for same device

    logger.info(
        "virtual_audio_detection_complete",
        total_devices=len(all_devices),
        virtual_devices=len(virtual_devices),
        platform=platform,
    )

    return virtual_devices


def _classify_device(device_lower: str, platform: str) -> tuple:
    """
    Classify a virtual audio device by type and recommendation.

    Returns:
        Tuple of (device_type, is_recommended)
    """
    if platform == "windows":
        if "cable" in device_lower or "vb-" in device_lower:
            return "vb-cable", True
        elif "voicemeeter" in device_lower:
            return "voicemeeter", True
        elif "stereo mix" in device_lower or "wave out mix" in device_lower:
            return "stereo-mix", False
        elif "virtual" in device_lower or "loopback" in device_lower:
            return "virtual", False
    else:  # Linux
        if ".monitor" in device_lower or device_lower.endswith("monitor"):
            return "pulse-monitor", True
        elif "pipewire" in device_lower:
            return "pipewire", True
        elif "null" in device_lower:
            return "null-sink", True
        elif "loopback" in device_lower:
            return "loopback", True
        elif "virtual" in device_lower:
            return "virtual", False

    return "other", False


def is_virtual_cable_available() -> bool:
    """
    Check if any virtual audio cable is installed.

    Returns:
        True if at least one virtual audio device is detected.
    """
    devices = detect_virtual_audio_devices()
    return len(devices) > 0


def is_recommended_device_available() -> bool:
    """
    Check if a recommended virtual audio device is available.

    Recommended devices (VB-Cable, VoiceMeeter) provide better
    reliability than generic Stereo Mix.

    Returns:
        True if VB-Cable or VoiceMeeter is detected.
    """
    devices = detect_virtual_audio_devices()
    return any(d.is_recommended for d in devices)


def get_setup_status() -> Dict[str, Any]:
    """
    Get comprehensive setup status for UI display.

    Returns:
        Dict with setup status information for the first-run wizard.
    """
    try:
        all_devices = list_audio_devices()
    except Exception as e:
        logger.error("setup_status_error", error=str(e))
        all_devices = []

    virtual_devices = detect_virtual_audio_devices()
    has_virtual = len(virtual_devices) > 0
    has_recommended = any(d.is_recommended for d in virtual_devices)

    # Determine recommended action (platform-aware)
    platform = get_platform()
    recommended_action = None
    if not has_virtual:
        if platform == "windows":
            recommended_action = "install_vbcable"
        else:
            recommended_action = "configure_monitor"  # Linux: use PulseAudio monitor
    elif not has_recommended:
        recommended_action = "consider_upgrade"  # Has basic device but not recommended

    # Check if setup was previously skipped
    setup_skipped = _check_setup_skipped()

    return {
        "virtual_audio_detected": has_virtual,
        "recommended_device_available": has_recommended,
        "detected_devices": [
            {
                "name": d.name,
                "type": d.device_type,
                "recommended": d.is_recommended,
            }
            for d in virtual_devices
        ],
        "all_audio_devices": all_devices,
        "recommended_action": recommended_action,
        "setup_complete": has_virtual or setup_skipped,
        "setup_skipped": setup_skipped,
        "recommendations": _get_recommended_devices(),
        "platform": platform,
    }


def _check_setup_skipped() -> bool:
    """
    Check if user has previously skipped VB-Cable setup.

    Reads from config file to see if setup was skipped.
    """
    from ..utils.paths import get_config_path

    try:
        config_path = get_config_path()
        if not config_path.exists():
            return False

        import json
        with open(config_path, "r") as f:
            config = json.load(f)

        return config.get("setup_skipped", False)

    except Exception as e:
        logger.warning(
            "setup_status_check_failed",
            error=str(e),
            error_type=type(e).__name__
        )
        return False


def mark_setup_complete(skipped: bool = False) -> bool:
    """
    Mark first-run setup as complete.

    Args:
        skipped: True if user chose to skip VB-Cable installation.

    Returns:
        True if successfully saved.
    """
    from ..utils.paths import get_config_path

    try:
        config_path = get_config_path()
        config = {}

        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                config = json.load(f)

        config["setup_complete"] = True
        config["setup_skipped"] = skipped
        config["setup_timestamp"] = __import__("datetime").datetime.now().isoformat()

        import json
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(
            "setup_marked_complete",
            skipped=skipped,
        )

        return True

    except Exception as e:
        logger.error("setup_mark_complete_error", error=str(e))
        return False


def get_recommended_device_info() -> List[Dict[str, str]]:
    """
    Get information about recommended virtual audio devices.

    Returns platform-specific recommendations.

    Returns:
        List of recommended device info with download links.
    """
    return _get_recommended_devices()


def get_best_audio_device() -> Optional[str]:
    """
    Get the best available audio device for recording.

    Priority:
    1. VB-Cable / VoiceMeeter (recommended)
    2. Any virtual audio device
    3. None (user needs to install VB-Cable)

    Returns:
        Device name or None if no suitable device found.
    """
    devices = detect_virtual_audio_devices()

    if not devices:
        return None

    # Prefer recommended devices
    recommended = [d for d in devices if d.is_recommended]
    if recommended:
        # Prefer VB-Cable over VoiceMeeter (simpler)
        vb_cable = [d for d in recommended if d.device_type == "vb-cable"]
        if vb_cable:
            return vb_cable[0].name
        return recommended[0].name

    # Fall back to any virtual device
    return devices[0].name
