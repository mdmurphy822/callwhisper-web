"""
FFmpeg Audio Recorder Service

Handles cross-platform audio recording using FFmpeg subprocess.
Supports Windows (DirectShow), Linux (PulseAudio/ALSA).

Based on LibV2 patterns:
- Structured logging with context
- Explicit timeouts for subprocess operations
- Resource cleanup in finally blocks
"""

import subprocess
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import signal
import os

from ..core.config import Settings
from ..core.state import RecordingSession
from ..core.logging_config import get_service_logger
from ..core.exceptions import RecordingError
from ..utils.paths import get_ffmpeg_path, get_output_dir
from ..utils.platform import get_audio_backend

logger = get_service_logger()


# Global reference to FFmpeg process
# Protected by _state_lock to prevent race conditions
_state_lock = threading.Lock()
_ffmpeg_process: Optional[subprocess.Popen] = None
_current_output_folder: Optional[Path] = None


def _get_input_args(backend: str, device_name: str) -> List[str]:
    """
    Get platform-specific FFmpeg input arguments.

    Args:
        backend: Audio backend (dshow, pulse, alsa)
        device_name: Device name or identifier

    Returns:
        List of FFmpeg arguments for audio input
    """
    if backend == "dshow":
        # Windows DirectShow: -f dshow -i audio="Device Name"
        return ["-f", "dshow", "-i", f"audio={device_name}"]
    elif backend == "pulse":
        # Linux PulseAudio: -f pulse -i device_name
        return ["-f", "pulse", "-i", device_name]
    elif backend == "alsa":
        # Linux ALSA: -f alsa -i hw:card,device
        # For ALSA, device_name might be "hw:0,0 (Card Name)" - extract hw:X,Y
        alsa_device = device_name.split()[0] if " " in device_name else device_name
        return ["-f", "alsa", "-i", alsa_device]
    elif backend == "avfoundation":
        # macOS AVFoundation: -f avfoundation -i ":device_index"
        return ["-f", "avfoundation", "-i", f":{device_name}"]
    else:
        logger.warning("unknown_audio_backend", backend=backend)
        # Fall back to DirectShow format
        return ["-f", "dshow", "-i", f"audio={device_name}"]


async def start_recording(session: RecordingSession, settings: Settings) -> Path:
    """
    Start recording audio from the specified device.

    Args:
        session: Recording session with device info
        settings: Application settings

    Returns:
        Path to the output folder

    Raises:
        RuntimeError: If recording fails to start
    """
    global _ffmpeg_process, _current_output_folder

    with _state_lock:
        if _ffmpeg_process is not None and _ffmpeg_process.poll() is None:
            raise RecordingError("Recording already in progress")

    ffmpeg_path = get_ffmpeg_path()

    # Check if ffmpeg exists
    if not ffmpeg_path.exists():
        # Try system PATH as fallback
        ffmpeg_path = Path("ffmpeg")

    # Create output folder
    output_base = get_output_dir()
    folder_name = session.id
    output_folder = output_base / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)

    _current_output_folder = output_folder

    # Output file path
    raw_audio_path = output_folder / "audio_raw.wav"
    log_path = output_folder / "ffmpeg.log"

    # Build FFmpeg command with platform-specific input
    backend = get_audio_backend()
    input_args = _get_input_args(backend, session.device_name)

    cmd = [
        str(ffmpeg_path),
        "-y",  # Overwrite output
        *input_args,
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(settings.audio.sample_rate),  # Sample rate
        "-ac", str(settings.audio.channels),  # Channels
        str(raw_audio_path),
    ]

    # Log the command
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Recording started: {datetime.now().isoformat()}\n")
        f.write(f"Device: {session.device_name}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("-" * 50 + "\n")

    try:
        logger.info(
            "starting_ffmpeg_recording",
            device=session.device_name,
            output_folder=str(output_folder),
            session_id=session.id
        )

        # Start FFmpeg process
        with open(log_path, "a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            with _state_lock:
                _ffmpeg_process = process

        # Wait briefly to check if process started successfully
        await asyncio.sleep(0.5)

        if _ffmpeg_process.poll() is not None:
            # Process already exited - read error
            with open(log_path, "r", encoding="utf-8") as f:
                error_log = f.read()
            logger.error("ffmpeg_start_failed", error=error_log)
            raise RecordingError(f"FFmpeg failed to start: {error_log}")

        logger.info("ffmpeg_recording_started", pid=_ffmpeg_process.pid)
        session.output_folder = str(output_folder)
        return output_folder

    except RecordingError:
        with _state_lock:
            _ffmpeg_process = None
        raise
    except Exception as e:
        with _state_lock:
            _ffmpeg_process = None
        logger.error("recording_start_error", error=str(e))
        raise RecordingError(f"Failed to start recording: {e}")


async def stop_recording() -> Optional[Path]:
    """
    Stop the current recording.

    Returns:
        Path to the raw audio file, or None if no recording active
    """
    global _ffmpeg_process

    with _state_lock:
        if _ffmpeg_process is None:
            return None
        process = _ffmpeg_process
        output_folder = _current_output_folder

    raw_audio_path = output_folder / "audio_raw.wav" if output_folder else None

    try:
        logger.info("stopping_ffmpeg_recording")

        # Send 'q' to FFmpeg to gracefully quit
        if process.stdin:
            try:
                process.stdin.write(b'q')
                process.stdin.flush()
            except Exception:
                pass

        # Wait for process to finish (with timeout)
        try:
            process.wait(timeout=5)
            logger.info("ffmpeg_stopped_gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if not responding
            logger.warning("ffmpeg_graceful_stop_timeout", timeout_seconds=5)
            process.terminate()
            try:
                process.wait(timeout=2)
                logger.info("ffmpeg_terminated")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("ffmpeg_force_killed")

    except Exception as e:
        logger.error("ffmpeg_stop_error", error=str(e))

    finally:
        with _state_lock:
            _ffmpeg_process = None

    # Log completion
    if output_folder:
        log_path = output_folder / "ffmpeg.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("-" * 50 + "\n")
            f.write(f"Recording stopped: {datetime.now().isoformat()}\n")

    return raw_audio_path


async def finalize_recording(session: RecordingSession) -> Path:
    """
    Finalize recording: stop FFmpeg and return output folder.

    Args:
        session: Recording session

    Returns:
        Path to the output folder
    """
    await stop_recording()

    output_folder = Path(session.output_folder) if session.output_folder else _current_output_folder

    if not output_folder:
        raise RuntimeError("No output folder found")

    return output_folder


def is_recording() -> bool:
    """Check if currently recording."""
    with _state_lock:
        return _ffmpeg_process is not None and _ffmpeg_process.poll() is None


def get_recording_stats() -> Optional[dict]:
    """
    Get current recording statistics.

    Returns:
        Dict with recording stats, or None if not recording
    """
    with _state_lock:
        if _ffmpeg_process is None or _ffmpeg_process.poll() is not None:
            return None
        output_folder = _current_output_folder

    if not output_folder:
        return None

    raw_audio_path = output_folder / "audio_raw.wav"

    stats = {
        "output_folder": str(output_folder),
        "bytes_written": 0,
    }

    if raw_audio_path.exists():
        stats["bytes_written"] = raw_audio_path.stat().st_size

    return stats
