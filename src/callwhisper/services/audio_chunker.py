"""
Audio Chunking Service for Resumable Transcription

Splits audio files into chunks for crash-resistant transcription.
Each chunk is transcribed separately and checkpointed, allowing
resume from the last completed chunk after a crash.
"""

import asyncio
import json
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

from ..core.logging_config import get_service_logger
from ..utils.paths import get_ffmpeg_path

logger = get_service_logger()

# Default chunking parameters
CHUNK_DURATION_SECONDS = 300  # 5 minutes per chunk
CHUNK_OVERLAP_SECONDS = 5     # 5 seconds overlap for context continuity


@dataclass
class AudioChunk:
    """Represents a single audio chunk for transcription."""
    index: int
    chunk_path: str
    start_time: float
    end_time: float
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AudioChunk":
        return cls(**data)


@dataclass
class ChunkManifest:
    """Manifest tracking all chunks for an audio file."""
    audio_path: str
    total_duration: float
    chunk_count: int
    chunk_duration: int
    overlap_seconds: int
    chunks: List[AudioChunk]

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "total_duration": self.total_duration,
            "chunk_count": self.chunk_count,
            "chunk_duration": self.chunk_duration,
            "overlap_seconds": self.overlap_seconds,
            "chunks": [c.to_dict() for c in self.chunks]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkManifest":
        chunks = [AudioChunk.from_dict(c) for c in data["chunks"]]
        return cls(
            audio_path=data["audio_path"],
            total_duration=data["total_duration"],
            chunk_count=data["chunk_count"],
            chunk_duration=data["chunk_duration"],
            overlap_seconds=data["overlap_seconds"],
            chunks=chunks
        )

    def save(self, manifest_path: Path) -> None:
        """Save manifest to JSON file."""
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, manifest_path: Path) -> "ChunkManifest":
        """Load manifest from JSON file."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


async def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds

    Raises:
        RuntimeError: If ffprobe fails
    """
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = ffmpeg_path.parent / ("ffprobe.exe" if os.name == 'nt' else "ffprobe")

    if not ffprobe_path.exists():
        ffprobe_path = Path("ffprobe")

    cmd = [
        str(ffprobe_path),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {stderr.decode()}")

        duration = float(stdout.decode().strip())
        logger.debug("audio_duration_detected", path=str(audio_path), duration=duration)
        return duration

    except ValueError as e:
        raise RuntimeError(f"Could not parse audio duration: {e}")


async def split_audio_to_chunks(
    audio_path: Path,
    output_dir: Path,
    chunk_duration: int = CHUNK_DURATION_SECONDS,
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS,
) -> ChunkManifest:
    """
    Split audio into chunks for resumable transcription.

    Args:
        audio_path: Path to source audio file
        output_dir: Directory to store chunk files
        chunk_duration: Duration of each chunk in seconds
        overlap_seconds: Overlap between chunks for context

    Returns:
        ChunkManifest with all chunk information
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_duration = await get_audio_duration(audio_path)
    ffmpeg_path = get_ffmpeg_path()

    if not ffmpeg_path.exists():
        ffmpeg_path = Path("ffmpeg")

    chunks = []
    chunk_idx = 0
    start = 0.0

    logger.info(
        "splitting_audio",
        audio_path=str(audio_path),
        total_duration=total_duration,
        chunk_duration=chunk_duration
    )

    while start < total_duration:
        end = min(start + chunk_duration, total_duration)
        actual_duration = end - start
        chunk_path = output_dir / f"chunk_{chunk_idx:04d}.wav"

        # FFmpeg command to extract chunk
        cmd = [
            str(ffmpeg_path),
            "-y",                   # Overwrite output
            "-ss", str(start),      # Seek to start position
            "-i", str(audio_path),  # Input file
            "-t", str(actual_duration),  # Duration
            "-ar", "16000",         # 16kHz sample rate (whisper requirement)
            "-ac", "1",             # Mono
            "-acodec", "pcm_s16le", # 16-bit PCM
            str(chunk_path)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("chunk_creation_failed", chunk=chunk_idx, error=stderr.decode())
            raise RuntimeError(f"Failed to create chunk {chunk_idx}: {stderr.decode()}")

        chunk = AudioChunk(
            index=chunk_idx,
            chunk_path=str(chunk_path),
            start_time=start,
            end_time=end,
            duration=actual_duration
        )
        chunks.append(chunk)

        logger.debug(
            "chunk_created",
            chunk_index=chunk_idx,
            start=start,
            end=end,
            duration=actual_duration
        )

        # Move start position with overlap for context
        start = end - overlap_seconds if end < total_duration else total_duration
        chunk_idx += 1

    manifest = ChunkManifest(
        audio_path=str(audio_path),
        total_duration=total_duration,
        chunk_count=len(chunks),
        chunk_duration=chunk_duration,
        overlap_seconds=overlap_seconds,
        chunks=chunks
    )

    logger.info(
        "audio_split_complete",
        chunk_count=len(chunks),
        total_duration=total_duration
    )

    return manifest


def merge_chunk_transcripts(
    transcripts: List[str],
    overlap_seconds: int = CHUNK_OVERLAP_SECONDS
) -> str:
    """
    Merge transcripts from multiple chunks, handling overlaps.

    This function attempts to deduplicate text that appears in the
    overlap regions between chunks.

    Args:
        transcripts: List of transcript texts from each chunk
        overlap_seconds: Overlap duration (used to estimate dedup window)

    Returns:
        Merged transcript text
    """
    if not transcripts:
        return ""

    if len(transcripts) == 1:
        return transcripts[0].strip()

    merged_parts = [transcripts[0].strip()]

    for i in range(1, len(transcripts)):
        current = transcripts[i].strip()
        if not current:
            continue

        # Simple deduplication: look for common suffix/prefix overlap
        prev_words = merged_parts[-1].split()
        curr_words = current.split()

        if not prev_words or not curr_words:
            merged_parts.append(current)
            continue

        # Look for overlapping words at boundary (check last N words of prev)
        # Typical overlap is ~5 seconds, so check last 10-20 words
        max_overlap_words = min(20, len(prev_words), len(curr_words))
        best_overlap = 0

        for overlap_len in range(1, max_overlap_words + 1):
            prev_end = prev_words[-overlap_len:]
            curr_start = curr_words[:overlap_len]

            if prev_end == curr_start:
                best_overlap = overlap_len

        # Skip overlapping words from current transcript
        if best_overlap > 0:
            current = " ".join(curr_words[best_overlap:])
            logger.debug("transcript_overlap_removed", words_removed=best_overlap)

        if current:
            merged_parts.append(current)

    return " ".join(merged_parts)


def get_manifest_path(output_dir: Path) -> Path:
    """Get the standard manifest file path for a chunk directory."""
    return output_dir / "chunks" / "manifest.json"


async def ensure_chunks_exist(
    audio_path: Path,
    output_dir: Path,
    chunk_duration: int = CHUNK_DURATION_SECONDS,
) -> ChunkManifest:
    """
    Ensure chunks exist for an audio file, creating them if needed.

    This is the main entry point for chunking. It:
    1. Checks if a manifest already exists
    2. If so, returns the existing manifest
    3. If not, splits the audio and creates a new manifest

    Args:
        audio_path: Path to source audio file
        output_dir: Base output directory
        chunk_duration: Duration of each chunk

    Returns:
        ChunkManifest with chunk information
    """
    chunks_dir = output_dir / "chunks"
    manifest_path = chunks_dir / "manifest.json"

    # Check for existing manifest
    if manifest_path.exists():
        try:
            manifest = ChunkManifest.load(manifest_path)
            # Verify chunks still exist
            all_exist = all(
                Path(c.chunk_path).exists()
                for c in manifest.chunks
            )
            if all_exist:
                logger.debug("using_existing_chunks", chunk_count=manifest.chunk_count)
                return manifest
            else:
                logger.warning("chunks_missing_recreating")
        except Exception as e:
            logger.warning("manifest_corrupted_recreating", error=str(e))

    # Create new chunks
    manifest = await split_audio_to_chunks(
        audio_path,
        chunks_dir,
        chunk_duration=chunk_duration
    )

    # Save manifest
    manifest.save(manifest_path)

    return manifest
