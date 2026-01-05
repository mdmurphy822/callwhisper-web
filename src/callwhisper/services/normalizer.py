"""
Audio Normalization Service

Converts audio to whisper-compatible format: 16kHz, mono, PCM WAV.

Based on LibV2 patterns:
- Structured logging with context
- Explicit timeouts for subprocess operations
- Specific exception handling
"""

import subprocess
import asyncio
from pathlib import Path
from typing import Optional
import os

from ..core.logging_config import get_service_logger
from ..core.exceptions import ProcessTimeoutError
from ..utils.paths import get_ffmpeg_path

logger = get_service_logger()

# FFmpeg operation timeouts
NORMALIZATION_TIMEOUT_SECONDS = 120
CONVERSION_TIMEOUT_SECONDS = 120


async def normalize_audio(
    input_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """
    Normalize audio file for transcription.

    Converts to:
    - 16kHz sample rate (whisper default)
    - Mono channel
    - PCM 16-bit WAV

    Args:
        input_path: Path to input audio file
        output_path: Path for normalized output (default: audio_16k.wav in same dir)
        sample_rate: Target sample rate (default: 16000)
        channels: Target channels (default: 1 for mono)

    Returns:
        Path to normalized audio file

    Raises:
        RuntimeError: If normalization fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / "audio_16k.wav"

    ffmpeg_path = get_ffmpeg_path()

    # Fallback to system PATH
    if not ffmpeg_path.exists():
        ffmpeg_path = Path("ffmpeg")

    # FFmpeg normalization command
    cmd = [
        str(ffmpeg_path),
        "-y",  # Overwrite output
        "-i",
        str(input_path),  # Input file
        "-ar",
        str(sample_rate),  # Sample rate
        "-ac",
        str(channels),  # Mono
        "-acodec",
        "pcm_s16le",  # PCM 16-bit
        str(output_path),
    ]

    try:
        logger.info(
            "starting_audio_normalization",
            input=str(input_path),
            output=str(output_path),
            sample_rate=sample_rate,
            channels=channels,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # Apply timeout to normalization
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=NORMALIZATION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(
                "normalization_timeout", timeout_seconds=NORMALIZATION_TIMEOUT_SECONDS
            )
            raise ProcessTimeoutError(
                f"Audio normalization timed out after {NORMALIZATION_TIMEOUT_SECONDS} seconds"
            )

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            logger.error("ffmpeg_normalization_failed", error=error_msg)
            raise RuntimeError(f"FFmpeg normalization failed: {error_msg}")

        if not output_path.exists():
            raise RuntimeError("Normalized audio file was not created")

        logger.info("audio_normalization_completed", output=str(output_path))
        return output_path

    except ProcessTimeoutError:
        raise
    except Exception as e:
        logger.error("audio_normalization_error", error=str(e))
        raise RuntimeError(f"Audio normalization error: {e}")


async def convert_to_opus(
    input_path: Path,
    output_path: Optional[Path] = None,
    bitrate: str = "64k",
) -> Path:
    """
    Convert audio to Opus format for efficient storage.

    Args:
        input_path: Path to input audio file
        output_path: Path for opus output (default: recording.opus in same dir)
        bitrate: Target bitrate (default: 64k for speech)

    Returns:
        Path to opus audio file
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path is None:
        output_path = input_path.parent / "recording.opus"

    ffmpeg_path = get_ffmpeg_path()

    if not ffmpeg_path.exists():
        ffmpeg_path = Path("ffmpeg")

    # FFmpeg opus conversion command
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(input_path),
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        "-ar",
        "48000",  # Opus preferred sample rate
        "-ac",
        "1",  # Mono
        str(output_path),
    ]

    try:
        logger.info(
            "starting_opus_conversion",
            input=str(input_path),
            output=str(output_path),
            bitrate=bitrate,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # Apply timeout to conversion
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=CONVERSION_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(
                "opus_conversion_timeout", timeout_seconds=CONVERSION_TIMEOUT_SECONDS
            )
            raise ProcessTimeoutError(
                f"Opus conversion timed out after {CONVERSION_TIMEOUT_SECONDS} seconds"
            )

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            logger.error("opus_conversion_failed", error=error_msg)
            raise RuntimeError(f"Opus conversion failed: {error_msg}")

        logger.info("opus_conversion_completed", output=str(output_path))
        return output_path

    except ProcessTimeoutError:
        raise
    except Exception as e:
        logger.error("opus_conversion_error", error=str(e))
        raise RuntimeError(f"Opus conversion error: {e}")


def get_audio_duration(file_path: Path) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        file_path: Path to audio file

    Returns:
        Duration in seconds
    """
    import wave

    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / rate
        except Exception as e:
            logger.warning(
                "wav_duration_read_failed", file=str(file_path), error=str(e)
            )

    # Fallback: use ffprobe if available
    return 0.0
