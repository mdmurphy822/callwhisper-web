"""
Whisper Transcription Service

Transcribes audio using whisper.cpp CLI.

Based on LibV2 patterns:
- Structured logging with context
- Explicit timeouts for subprocess operations
- Specific exception handling
"""

import subprocess
import asyncio
import re
from pathlib import Path
from typing import Optional, Callable, Awaitable, List
import os
import shutil

from ..core.config import Settings
from ..core.logging_config import get_service_logger
from ..core.exceptions import TranscriptionError, ProcessTimeoutError
from ..core.job_store import update_checkpoint
from ..utils.paths import get_whisper_path, get_models_dir, get_ffmpeg_path
from .normalizer import normalize_audio
from .srt_merger import merge_srt_segments

logger = get_service_logger()

# Default transcription timeout (10 minutes)
TRANSCRIPTION_TIMEOUT_SECONDS = 600

# Adaptive timeout parameters
MIN_TIMEOUT_SECONDS = 120  # 2 minutes minimum
MAX_TIMEOUT_SECONDS = 7200  # 2 hours maximum
TIMEOUT_MULTIPLIER = 3  # 3x audio duration


async def get_audio_duration_seconds(audio_path: Path) -> float:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If duration cannot be determined
    """
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = ffmpeg_path.parent / (
        "ffprobe.exe" if os.name == "nt" else "ffprobe"
    )

    if not ffprobe_path.exists():
        ffprobe_path = Path("ffprobe")

    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {stderr.decode()}")

        return float(stdout.decode().strip())

    except ValueError as e:
        raise RuntimeError(f"Could not parse audio duration: {e}")


def calculate_adaptive_timeout(audio_duration_seconds: float) -> int:
    """
    Calculate an appropriate timeout based on audio duration.

    Uses TIMEOUT_MULTIPLIER * audio_duration, clamped to min/max bounds.
    This allows longer files more time while preventing infinite hangs.

    Args:
        audio_duration_seconds: Duration of audio file

    Returns:
        Timeout in seconds
    """
    timeout = int(audio_duration_seconds * TIMEOUT_MULTIPLIER)
    return max(MIN_TIMEOUT_SECONDS, min(MAX_TIMEOUT_SECONDS, timeout))


async def transcribe_audio(
    output_folder: Path,
    settings: Settings,
    progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
    partial_transcript_callback: Optional[
        Callable[[str, bool], Awaitable[None]]
    ] = None,
) -> Path:
    """
    Transcribe audio in the output folder.

    Pipeline:
    1. Normalize audio to 16kHz mono WAV
    2. Run whisper.cpp transcription
    3. Save transcript files

    Args:
        output_folder: Folder containing audio_raw.wav
        settings: Application settings
        progress_callback: Optional async callback(percent, stage) for progress updates
        partial_transcript_callback: Optional async callback(text, is_final) for real-time preview

    Returns:
        Path to transcript.txt

    Raises:
        RuntimeError: If transcription fails
    """
    raw_audio = output_folder / "audio_raw.wav"

    if not raw_audio.exists():
        raise FileNotFoundError(f"Raw audio not found: {raw_audio}")

    # Step 1: Normalize audio
    normalized_audio = await normalize_audio(raw_audio)

    # Calculate adaptive timeout based on audio duration
    try:
        audio_duration = await get_audio_duration_seconds(normalized_audio)
        timeout = calculate_adaptive_timeout(audio_duration)
        logger.info(
            "adaptive_timeout_calculated",
            audio_seconds=audio_duration,
            timeout_seconds=timeout,
        )
    except Exception as e:
        logger.warning("duration_detection_failed", error=str(e))
        timeout = TRANSCRIPTION_TIMEOUT_SECONDS  # Fallback to default

    # Step 2: Transcribe with whisper.cpp
    whisper_path = get_whisper_path()
    model_path = get_models_dir() / settings.transcription.model

    # Fallback to system whisper if not found
    if not whisper_path.exists():
        whisper_path = Path("whisper-cli")

    if not model_path.exists():
        # Try alternative model names
        models_dir = get_models_dir()
        if models_dir.exists():
            models = list(models_dir.glob("*.bin"))
            if models:
                model_path = models[0]
            else:
                raise FileNotFoundError(
                    f"Whisper model not found: {model_path}. "
                    f"Please place a whisper model in the models/ directory."
                )
        else:
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # whisper.cpp command - use absolute paths for cross-platform compatibility
    cmd = [
        str(whisper_path.resolve()),
        "-m",
        str(model_path.resolve()),
        "-f",
        str(normalized_audio.resolve()),
        "-otxt",  # Output text
        "-osrt",  # Output SRT subtitles
        "-pp",  # Print progress for real-time updates
        "--language",
        settings.transcription.language,
        "--beam-size",
        str(settings.transcription.beam_size),
        "--best-of",
        str(settings.transcription.best_of),
    ]

    log_path = output_folder / "whisper.log"

    try:
        logger.info(
            "starting_transcription",
            audio_file=str(normalized_audio),
            model=str(model_path.name),
            language=settings.transcription.language,
        )

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Whisper command: {' '.join(cmd)}\n")
            log_file.write("-" * 50 + "\n")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(output_folder),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # Stream stderr for progress updates while respecting timeout
        stderr_lines = []
        stdout_data = b""
        last_progress = -1

        async def read_stderr_with_progress():
            nonlocal last_progress
            buffer = ""
            while True:
                try:
                    # Read small chunks instead of waiting for newlines
                    # whisper uses \r for inline progress updates without newlines
                    chunk = await asyncio.wait_for(
                        process.stderr.read(256), timeout=timeout
                    )
                    if not chunk:
                        break
                    chunk_text = chunk.decode("utf-8", errors="replace")
                    buffer += chunk_text
                    stderr_lines.append(chunk_text)

                    # Parse progress from buffer (format: "progress =  XX%")
                    if progress_callback:
                        matches = re.findall(r"(\d+)\s*%", buffer)
                        if matches:
                            whisper_percent = int(matches[-1])  # Use most recent
                            # Only update if progress increased
                            if whisper_percent > last_progress:
                                last_progress = whisper_percent
                                # Map whisper's 0-100% to overall 30-80%
                                mapped_percent = 30 + int(whisper_percent * 0.5)
                                await progress_callback(
                                    mapped_percent,
                                    f"Transcribing... {whisper_percent}%",
                                )
                            # Keep tail of buffer for partial matches
                            buffer = buffer[-50:]
                except asyncio.TimeoutError:
                    raise

        try:
            # Read stderr with progress updates
            await read_stderr_with_progress()
            # Read any remaining stdout
            stdout_data = await process.stdout.read()
            # Wait for process to finish
            await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error("transcription_timeout", timeout_seconds=timeout)
            raise ProcessTimeoutError(
                f"Transcription timed out after {timeout} seconds"
            )

        # Log output
        stderr_text = "".join(stderr_lines)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(stdout_data.decode("utf-8", errors="replace"))
            log_file.write(stderr_text)

        if process.returncode != 0:
            logger.error("whisper_transcription_failed", error=stderr_text)
            raise TranscriptionError(f"Whisper transcription failed: {stderr_text}")

        logger.info("transcription_completed")

    except FileNotFoundError:
        logger.error("whisper_not_found", path=str(whisper_path))
        raise TranscriptionError(
            f"whisper-cli not found at {whisper_path}. "
            f"Please place whisper-cli.exe in the vendor/ directory."
        )
    except (ProcessTimeoutError, TranscriptionError):
        raise
    except Exception as e:
        logger.error("transcription_error", error=str(e))
        raise TranscriptionError(f"Transcription error: {e}")

    # Step 3: Rename whisper outputs to standard names
    # whisper.cpp outputs: audio_16k.txt, audio_16k.srt
    whisper_txt = output_folder / "audio_16k.txt"
    whisper_srt = output_folder / "audio_16k.srt"

    transcript_txt = output_folder / "transcript.txt"
    transcript_srt = output_folder / "transcript.srt"

    if whisper_txt.exists():
        shutil.copy(whisper_txt, transcript_txt)
    else:
        # Check for alternative output names
        for alt in ["audio_16k.wav.txt", "transcript.txt"]:
            alt_path = output_folder / alt
            if alt_path.exists():
                if alt_path != transcript_txt:
                    shutil.copy(alt_path, transcript_txt)
                break

    if whisper_srt.exists():
        shutil.copy(whisper_srt, transcript_srt)
    else:
        for alt in ["audio_16k.wav.srt", "transcript.srt"]:
            alt_path = output_folder / alt
            if alt_path.exists():
                if alt_path != transcript_srt:
                    shutil.copy(alt_path, transcript_srt)
                break

    # Step 4: Merge SRT segments for better readability
    if transcript_srt.exists():
        merge_srt_segments(transcript_srt)
        logger.info("srt_segments_merged", path=str(transcript_srt))

    # Verify transcript was created
    if not transcript_txt.exists():
        # Create empty transcript if whisper produced no output
        transcript_txt.write_text("[No speech detected]", encoding="utf-8")

    # Broadcast final transcript for real-time preview
    if partial_transcript_callback:
        final_text = transcript_txt.read_text(encoding="utf-8")
        await partial_transcript_callback(final_text, is_final=True)

    return transcript_txt


def srt_to_vtt(srt_path: Path, vtt_path: Optional[Path] = None) -> Path:
    """
    Convert SRT subtitles to WebVTT format.

    Args:
        srt_path: Path to SRT file
        vtt_path: Output VTT path (default: same name with .vtt)

    Returns:
        Path to VTT file
    """
    if vtt_path is None:
        vtt_path = srt_path.with_suffix(".vtt")

    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    with open(srt_path, "r", encoding="utf-8") as f:
        srt_content = f.read()

    # Convert SRT to VTT
    # SRT format: 00:00:00,000 --> 00:00:01,000
    # VTT format: 00:00:00.000 --> 00:00:01.000

    vtt_content = "WEBVTT\n\n"

    # Replace comma with dot in timestamps
    import re

    vtt_content += re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt_content)

    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write(vtt_content)

    return vtt_path


async def transcribe_chunk(
    chunk_path: Path,
    settings: Settings,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Transcribe a single audio chunk.

    Args:
        chunk_path: Path to the chunk WAV file
        settings: Application settings
        output_dir: Output directory for whisper (defaults to chunk's parent)

    Returns:
        Transcript text for this chunk
    """
    if output_dir is None:
        output_dir = chunk_path.parent

    whisper_path = get_whisper_path()
    model_path = get_models_dir() / settings.transcription.model

    if not whisper_path.exists():
        whisper_path = Path("whisper-cli")

    if not model_path.exists():
        models_dir = get_models_dir()
        if models_dir.exists():
            models = list(models_dir.glob("*.bin"))
            if models:
                model_path = models[0]
            else:
                raise FileNotFoundError("No whisper model found")
        else:
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Output file path (whisper.cpp adds .txt automatically)
    output_base = chunk_path.stem

    cmd = [
        str(whisper_path.resolve()),
        "-m",
        str(model_path.resolve()),
        "-f",
        str(chunk_path.resolve()),
        "-otxt",
        "--language",
        settings.transcription.language,
        "--beam-size",
        str(settings.transcription.beam_size),
        "--best-of",
        str(settings.transcription.best_of),
    ]

    # Calculate timeout for this chunk
    try:
        chunk_duration = await get_audio_duration_seconds(chunk_path)
        timeout = calculate_adaptive_timeout(chunk_duration)
    except Exception as e:
        logger.warning(
            "timeout_calculation_fallback",
            chunk_path=str(chunk_path),
            fallback_timeout=MIN_TIMEOUT_SECONDS,
            error=str(e),
            error_type=type(e).__name__,
        )
        timeout = MIN_TIMEOUT_SECONDS

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(output_dir),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ProcessTimeoutError(f"Chunk transcription timed out after {timeout}s")

    if proc.returncode != 0:
        raise TranscriptionError(f"Chunk transcription failed: {stderr.decode()}")

    # Read transcript output
    txt_path = output_dir / f"{output_base}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()

    # Try alternative naming
    alt_path = output_dir / f"{chunk_path.name}.txt"
    if alt_path.exists():
        return alt_path.read_text(encoding="utf-8").strip()

    return ""


async def transcribe_audio_chunked(
    output_folder: Path,
    settings: Settings,
    job_id: Optional[str] = None,
    start_from_chunk: int = 0,
    progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
    partial_transcript_callback: Optional[
        Callable[[str, bool], Awaitable[None]]
    ] = None,
) -> Path:
    """
    Transcribe audio using chunks for crash recovery.

    This method splits audio into chunks and transcribes each separately,
    checkpointing progress after each chunk. If interrupted, transcription
    can resume from the last completed chunk.

    Args:
        output_folder: Folder containing normalized audio (audio_16k.wav)
        settings: Application settings
        job_id: Job ID for checkpoint updates (defaults to folder name)
        start_from_chunk: Resume from this chunk index (0-based)
        progress_callback: Optional async callback(percent, stage)
        partial_transcript_callback: Optional async callback(text, is_final) for real-time preview

    Returns:
        Path to final transcript.txt
    """
    from .audio_chunker import ensure_chunks_exist, merge_chunk_transcripts

    normalized_audio = output_folder / "audio_16k.wav"
    if not normalized_audio.exists():
        raise FileNotFoundError(f"Normalized audio not found: {normalized_audio}")

    job_id = job_id or output_folder.name

    # Ensure chunks exist (creates them if needed, reuses if already split)
    manifest = await ensure_chunks_exist(normalized_audio, output_folder)
    total_chunks = manifest.chunk_count

    logger.info(
        "chunked_transcription_starting",
        job_id=job_id,
        total_chunks=total_chunks,
        start_from_chunk=start_from_chunk,
    )

    # Load existing transcripts if resuming
    transcripts: List[str] = []
    partial_file = output_folder / "partial_transcript.txt"

    if start_from_chunk > 0 and partial_file.exists():
        partial_content = partial_file.read_text(encoding="utf-8")
        transcripts = partial_content.split("\n---CHUNK_BOUNDARY---\n")
        logger.info("loaded_partial_transcript", chunks_loaded=len(transcripts))

    # Transcribe remaining chunks
    for chunk in manifest.chunks[start_from_chunk:]:
        chunk_path = Path(chunk.chunk_path)
        chunk_idx = chunk.index

        logger.info(
            "transcribing_chunk",
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )

        # Transcribe this chunk
        try:
            chunk_transcript = await transcribe_chunk(chunk_path, settings)
            transcripts.append(chunk_transcript)
        except Exception as e:
            logger.error("chunk_transcription_failed", chunk=chunk_idx, error=str(e))
            # Save partial progress before re-raising
            partial_file.write_text(
                "\n---CHUNK_BOUNDARY---\n".join(transcripts), encoding="utf-8"
            )
            raise

        # Broadcast partial transcript for real-time preview
        if partial_transcript_callback:
            combined_text = " ".join(transcripts)
            await partial_transcript_callback(combined_text, is_final=False)

        # Save checkpoint after each chunk
        update_checkpoint(
            job_id=job_id,
            status=f"chunk_{chunk_idx}",
            chunks_completed=chunk_idx + 1,
            total_chunks=total_chunks,
            partial_transcript="\n---CHUNK_BOUNDARY---\n".join(transcripts),
        )

        # Save partial transcript file
        partial_file.write_text(
            "\n---CHUNK_BOUNDARY---\n".join(transcripts), encoding="utf-8"
        )

        # Update progress
        if progress_callback:
            pct = 30 + int(((chunk_idx + 1) / total_chunks) * 50)
            await progress_callback(
                pct, f"Transcribing chunk {chunk_idx + 1}/{total_chunks}"
            )

    # Merge all transcripts
    logger.info("merging_chunk_transcripts", chunk_count=len(transcripts))
    final_transcript = merge_chunk_transcripts(transcripts, manifest.overlap_seconds)

    # Write final transcript
    transcript_path = output_folder / "transcript.txt"
    transcript_path.write_text(final_transcript, encoding="utf-8")

    # Broadcast final transcript
    if partial_transcript_callback:
        await partial_transcript_callback(final_transcript, is_final=True)

    # Clean up partial file
    if partial_file.exists():
        partial_file.unlink()

    logger.info(
        "chunked_transcription_complete",
        job_id=job_id,
        chunks_processed=total_chunks,
        transcript_length=len(final_transcript),
    )

    return transcript_path
