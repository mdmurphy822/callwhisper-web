"""Cross-platform audio device enumeration."""

import subprocess
import re
import time
import threading
from typing import List, Optional, Tuple
from pathlib import Path

from ..core.logging_config import get_service_logger
from ..utils.paths import get_ffmpeg_path
from ..utils.platform import get_audio_backend

logger = get_service_logger()

# Device cache with TTL and thread-safe access
_device_cache: Tuple[List[str], float] = ([], 0)
_cache_lock = threading.Lock()
CACHE_TTL_SECONDS = 30


def list_audio_devices(use_cache: bool = True) -> List[str]:
    """
    List available audio devices for the current platform.

    Thread-safe implementation with caching.
    Supports Windows (DirectShow), Linux (PulseAudio/ALSA).

    Args:
        use_cache: If True, return cached results if available and fresh.

    Returns:
        List of audio device names available for recording.
    """
    global _device_cache

    # Check cache first (with lock)
    with _cache_lock:
        if use_cache:
            devices, cached_at = _device_cache
            if devices and (time.time() - cached_at) < CACHE_TTL_SECONDS:
                logger.debug("device_cache_hit", device_count=len(devices))
                return devices.copy()

    backend = get_audio_backend()

    if backend == "dshow":
        devices = _list_dshow_devices()
    elif backend == "pulse":
        devices = _list_pulse_devices()
    elif backend == "alsa":
        devices = _list_alsa_devices()
    else:
        logger.warning("unsupported_audio_backend", backend=backend)
        devices = []

    # Update cache (with lock)
    with _cache_lock:
        _device_cache = (devices, time.time())
    logger.debug("device_list_refreshed", device_count=len(devices), backend=backend)

    return devices


def _list_dshow_devices() -> List[str]:
    """List audio devices using Windows DirectShow."""
    ffmpeg_path = get_ffmpeg_path()

    if not ffmpeg_path.exists():
        ffmpeg_path = Path("ffmpeg")

    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return _parse_dshow_output(result.stderr)

    except subprocess.TimeoutExpired:
        logger.warning(
            "device_enumeration_timeout", backend="dshow", timeout_seconds=10
        )
        return []
    except FileNotFoundError:
        logger.error("ffmpeg_not_found", path=str(ffmpeg_path))
        return []
    except Exception as e:
        logger.error("device_enumeration_failed", backend="dshow", error=str(e))
        return []


def _list_pulse_devices() -> List[str]:
    """List audio devices using PulseAudio."""
    try:
        # Use pactl to list sources (recording devices)
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning("pactl_failed", stderr=result.stderr)
            return []

        return _parse_pulse_output(result.stdout)

    except FileNotFoundError:
        logger.error("pactl_not_found")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(
            "device_enumeration_timeout", backend="pulse", timeout_seconds=10
        )
        return []
    except Exception as e:
        logger.error("device_enumeration_failed", backend="pulse", error=str(e))
        return []


def _list_alsa_devices() -> List[str]:
    """List audio devices using ALSA."""
    try:
        # Use arecord to list capture devices
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning("arecord_failed", stderr=result.stderr)
            return []

        return _parse_alsa_output(result.stdout)

    except FileNotFoundError:
        logger.error("arecord_not_found")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("device_enumeration_timeout", backend="alsa", timeout_seconds=10)
        return []
    except Exception as e:
        logger.error("device_enumeration_failed", backend="alsa", error=str(e))
        return []


def _parse_dshow_output(ffmpeg_output: str) -> List[str]:
    """
    Parse FFmpeg's DirectShow device list output.

    FFmpeg output format:
    [dshow @ ...] DirectShow video devices
    [dshow @ ...]  "Device Name"
    [dshow @ ...] DirectShow audio devices
    [dshow @ ...]  "Audio Device Name"
    """
    devices = []
    in_audio_section = False

    for line in ffmpeg_output.split("\n"):
        line = line.strip()

        if "DirectShow audio devices" in line:
            in_audio_section = True
            continue

        if in_audio_section and ("DirectShow video devices" in line or not line):
            if "DirectShow video devices" in line:
                in_audio_section = False
            continue

        if in_audio_section:
            match = re.search(r'"([^"]+)"', line)
            if match:
                device_name = match.group(1)
                if "Alternative name" not in line:
                    devices.append(device_name)

    return devices


def _parse_pulse_output(pactl_output: str) -> List[str]:
    """
    Parse pactl list sources output.

    Output format:
    <index>	<name>	<module>	<sample_spec>	<state>
    0	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	...
    1	alsa_input.pci-0000_00_1f.3.analog-stereo	...
    """
    devices = []

    for line in pactl_output.strip().split("\n"):
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) >= 2:
            device_name = parts[1]
            devices.append(device_name)

    return devices


def _parse_alsa_output(arecord_output: str) -> List[str]:
    """
    Parse arecord -l output.

    Output format:
    **** List of CAPTURE Hardware Devices ****
    card 0: PCH [HDA Intel PCH], device 0: ALC287 Analog [ALC287 Analog]
      Subdevices: 1/1
      Subdevice #0: subdevice #0
    """
    devices = []

    # Match "card N: Name [Description], device M: ..."
    pattern = r"card (\d+):.*\[([^\]]+)\], device (\d+):"

    for line in arecord_output.split("\n"):
        match = re.search(pattern, line)
        if match:
            card_num = match.group(1)
            card_name = match.group(2)
            device_num = match.group(3)
            # Format as ALSA device specifier: hw:card,device
            device_id = f"hw:{card_num},{device_num}"
            # Include friendly name for display
            devices.append(f"{device_id} ({card_name})")

    return devices


# Keep old function name for backwards compatibility
def parse_device_list(ffmpeg_output: str) -> List[str]:
    """Parse FFmpeg's device list output (DirectShow format)."""
    return _parse_dshow_output(ffmpeg_output)


def get_device_by_name(name: str) -> Optional[str]:
    """
    Find a device by partial name match.

    Args:
        name: Partial device name to search for

    Returns:
        Full device name if found, None otherwise
    """
    devices = list_audio_devices()
    name_lower = name.lower()

    for device in devices:
        if name_lower in device.lower():
            return device

    return None


def device_exists(name: str) -> bool:
    """
    Check if a device with the given name exists.

    Args:
        name: Device name to check

    Returns:
        True if device exists
    """
    devices = list_audio_devices()
    return name in devices


async def list_audio_devices_async(use_cache: bool = True) -> List[str]:
    """
    Async version of list_audio_devices using asyncio subprocess.

    Should be used when called from async context to avoid blocking.

    Args:
        use_cache: If True, return cached results if available and fresh.

    Returns:
        List of audio device names available for recording.
    """
    import asyncio

    global _device_cache

    # Check cache first (with lock)
    with _cache_lock:
        if use_cache:
            devices, cached_at = _device_cache
            if devices and (time.time() - cached_at) < CACHE_TTL_SECONDS:
                logger.debug("device_cache_hit_async", device_count=len(devices))
                return devices.copy()

    backend = get_audio_backend()

    if backend == "dshow":
        devices = await _list_dshow_devices_async()
    elif backend == "pulse":
        devices = await _list_pulse_devices_async()
    elif backend == "alsa":
        devices = await _list_alsa_devices_async()
    else:
        logger.warning("unsupported_audio_backend_async", backend=backend)
        devices = []

    # Update cache (with lock)
    with _cache_lock:
        _device_cache = (devices, time.time())
    logger.debug(
        "device_list_refreshed_async", device_count=len(devices), backend=backend
    )

    return devices


async def _list_dshow_devices_async() -> List[str]:
    """Async version of DirectShow device enumeration."""
    import asyncio

    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path.exists():
        ffmpeg_path = Path("ffmpeg")

    try:
        proc = await asyncio.create_subprocess_exec(
            str(ffmpeg_path),
            "-list_devices",
            "true",
            "-f",
            "dshow",
            "-i",
            "dummy",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
        output = stderr.decode(errors="replace")

        return _parse_dshow_output(output)

    except asyncio.TimeoutError:
        logger.warning(
            "device_enumeration_timeout_async", backend="dshow", timeout_seconds=10
        )
        return []
    except FileNotFoundError:
        logger.error("ffmpeg_not_found_async", path=str(ffmpeg_path))
        return []
    except Exception as e:
        logger.error("device_enumeration_failed_async", backend="dshow", error=str(e))
        return []


async def _list_pulse_devices_async() -> List[str]:
    """Async version of PulseAudio device enumeration."""
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            "list",
            "sources",
            "short",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        if proc.returncode != 0:
            logger.warning("pactl_failed_async", stderr=stderr.decode(errors="replace"))
            return []

        return _parse_pulse_output(stdout.decode(errors="replace"))

    except asyncio.TimeoutError:
        logger.warning(
            "device_enumeration_timeout_async", backend="pulse", timeout_seconds=10
        )
        return []
    except FileNotFoundError:
        logger.error("pactl_not_found_async")
        return []
    except Exception as e:
        logger.error("device_enumeration_failed_async", backend="pulse", error=str(e))
        return []


async def _list_alsa_devices_async() -> List[str]:
    """Async version of ALSA device enumeration."""
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            "arecord",
            "-l",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        if proc.returncode != 0:
            logger.warning(
                "arecord_failed_async", stderr=stderr.decode(errors="replace")
            )
            return []

        return _parse_alsa_output(stdout.decode(errors="replace"))

    except asyncio.TimeoutError:
        logger.warning(
            "device_enumeration_timeout_async", backend="alsa", timeout_seconds=10
        )
        return []
    except FileNotFoundError:
        logger.error("arecord_not_found_async")
        return []
    except Exception as e:
        logger.error("device_enumeration_failed_async", backend="alsa", error=str(e))
        return []
