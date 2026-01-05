"""
VTB (Voice Transcript Bundle) Format Handler

Creates ZIP-based .vtb bundles containing audio recordings and transcripts.
"""

import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from ..core.config import Settings
from ..core.state import RecordingSession
from ..utils.hashing import compute_sha256, compute_sha256_bytes
from .normalizer import convert_to_opus, get_audio_duration


# Constants
VTB_VERSION = "1.0.0"
VTB_MIMETYPE = "application/x-vtb"
VTB_EXTENSION = ".vtb"


@dataclass
class VTBManifest:
    """Manifest metadata for VTB bundle."""
    version: str
    format: str
    created: str
    generator_name: str
    generator_version: str
    recording_id: str
    ticket_id: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: float
    device_name: str
    audio_format: str
    transcript_word_count: int
    files: List[Dict[str, Any]]


def get_content_type(filename: str) -> str:
    """Get MIME content type for a file."""
    ext = Path(filename).suffix.lower()
    content_types = {
        ".opus": "audio/opus",
        ".wav": "audio/wav",
        ".txt": "text/plain; charset=utf-8",
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".json": "application/json",
    }
    return content_types.get(ext, "application/octet-stream")


def get_compression(filename: str) -> int:
    """Get appropriate ZIP compression method."""
    ext = Path(filename).suffix.lower()
    # Already compressed formats - store without compression
    if ext in {'.opus', '.mp3', '.ogg', '.m4a', '.png', '.jpg'}:
        return zipfile.ZIP_STORED
    # Everything else gets deflate
    return zipfile.ZIP_DEFLATED


async def create_vtb_bundle(
    output_folder: Path,
    session: RecordingSession,
    settings: Settings,
) -> Path:
    """
    Create a VTB bundle from recording artifacts.

    Bundle structure:
    - mimetype (uncompressed, first)
    - META-INF/manifest.json
    - META-INF/hashes.json
    - audio/recording.opus (or .wav)
    - transcript/transcript.txt

    Args:
        output_folder: Folder containing recording artifacts
        session: Recording session metadata
        settings: Application settings

    Returns:
        Path to created .vtb bundle
    """
    from .. import __version__, __app_name__

    # Determine audio source and format
    raw_audio = output_folder / "audio_raw.wav"
    normalized_audio = output_folder / "audio_16k.wav"
    transcript_txt = output_folder / "transcript.txt"
    transcript_srt = output_folder / "transcript.srt"

    # Use normalized audio if available, otherwise raw
    source_audio = normalized_audio if normalized_audio.exists() else raw_audio

    if not source_audio.exists():
        raise FileNotFoundError(f"No audio file found in {output_folder}")

    # Convert to opus for bundle (smaller size)
    audio_format = settings.output.audio_format
    if audio_format == "opus":
        bundle_audio = output_folder / "recording.opus"
        await convert_to_opus(source_audio, bundle_audio)
        audio_archive_path = "audio/recording.opus"
    else:
        bundle_audio = source_audio
        audio_archive_path = f"audio/recording{source_audio.suffix}"

    # Get audio duration
    duration = get_audio_duration(source_audio)
    if duration == 0 and session.start_time and session.end_time:
        duration = (session.end_time - session.start_time).total_seconds()

    # Read transcript for word count
    word_count = 0
    if transcript_txt.exists():
        with open(transcript_txt, 'r', encoding='utf-8') as f:
            word_count = len(f.read().split())

    # Build file list with hashes
    files_to_add = []  # (archive_path, source_path)
    file_hashes = {}
    file_entries = []

    # Add audio
    files_to_add.append((audio_archive_path, bundle_audio))
    file_hashes[audio_archive_path] = compute_sha256(bundle_audio)
    file_entries.append({
        "path": audio_archive_path,
        "size_bytes": bundle_audio.stat().st_size,
        "content_type": get_content_type(str(bundle_audio)),
        "required": True,
    })

    # Add transcript.txt
    if transcript_txt.exists():
        files_to_add.append(("transcript/transcript.txt", transcript_txt))
        file_hashes["transcript/transcript.txt"] = compute_sha256(transcript_txt)
        file_entries.append({
            "path": "transcript/transcript.txt",
            "size_bytes": transcript_txt.stat().st_size,
            "content_type": "text/plain; charset=utf-8",
            "required": True,
        })

    # Add transcript.srt if exists
    if transcript_srt.exists():
        files_to_add.append(("transcript/transcript.srt", transcript_srt))
        file_hashes["transcript/transcript.srt"] = compute_sha256(transcript_srt)
        file_entries.append({
            "path": "transcript/transcript.srt",
            "size_bytes": transcript_srt.stat().st_size,
            "content_type": "application/x-subrip",
            "required": False,
        })

    # Build manifest
    now = datetime.now(timezone.utc).isoformat()

    manifest = VTBManifest(
        version=VTB_VERSION,
        format="vtb",
        created=now,
        generator_name=__app_name__,
        generator_version=__version__,
        recording_id=session.id,
        ticket_id=session.ticket_id,
        start_time=session.start_time.isoformat() if session.start_time else None,
        end_time=session.end_time.isoformat() if session.end_time else None,
        duration_seconds=duration,
        device_name=session.device_name,
        audio_format=audio_format,
        transcript_word_count=word_count,
        files=file_entries,
    )

    manifest_json = json.dumps(asdict(manifest), indent=2)
    manifest_hash = compute_sha256_bytes(manifest_json.encode('utf-8'))

    # Build hashes document
    hashes_doc = {
        "version": VTB_VERSION,
        "algorithm": "sha256",
        "generated": now,
        "files": file_hashes,
        "manifest_hash": manifest_hash,
    }

    # Create bundle path
    bundle_name = f"{session.id}{VTB_EXTENSION}"
    bundle_path = output_folder / bundle_name

    # Create the ZIP archive
    with zipfile.ZipFile(bundle_path, 'w') as zf:
        # 1. Write mimetype FIRST, uncompressed
        zf.writestr(
            "mimetype",
            VTB_MIMETYPE.encode('ascii'),
            compress_type=zipfile.ZIP_STORED
        )

        # 2. Write manifest
        zf.writestr(
            "META-INF/manifest.json",
            manifest_json,
            compress_type=zipfile.ZIP_DEFLATED
        )

        # 3. Write hashes
        zf.writestr(
            "META-INF/hashes.json",
            json.dumps(hashes_doc, indent=2),
            compress_type=zipfile.ZIP_DEFLATED
        )

        # 4. Write content files
        for archive_path, source_path in files_to_add:
            compression = get_compression(str(source_path))
            zf.write(source_path, archive_path, compress_type=compression)

    return bundle_path


def extract_vtb(bundle_path: Path, destination: Path) -> Path:
    """
    Extract a VTB bundle to a directory.

    Args:
        bundle_path: Path to .vtb file
        destination: Extraction destination

    Returns:
        Path to extraction directory
    """
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        zf.extractall(destination)

    return destination


def verify_vtb(bundle_path: Path) -> Dict[str, bool]:
    """
    Verify integrity of a VTB bundle.

    Args:
        bundle_path: Path to .vtb file

    Returns:
        Dict mapping file paths to verification status
    """
    results = {}

    with zipfile.ZipFile(bundle_path, 'r') as zf:
        # Load stored hashes
        try:
            hashes_data = json.loads(zf.read("META-INF/hashes.json"))
            stored_hashes = hashes_data.get("files", {})
        except KeyError:
            return {"error": False}

        # Verify each file
        for file_path, expected_hash in stored_hashes.items():
            try:
                content = zf.read(file_path)
                actual_hash = compute_sha256_bytes(content)
                results[file_path] = (actual_hash == expected_hash)
            except KeyError:
                results[file_path] = False

    return results


def get_vtb_info(bundle_path: Path) -> Dict[str, Any]:
    """
    Get information about a VTB bundle.

    Args:
        bundle_path: Path to .vtb file

    Returns:
        Dict with bundle information
    """
    with zipfile.ZipFile(bundle_path, 'r') as zf:
        manifest_data = json.loads(zf.read("META-INF/manifest.json"))

    return {
        "path": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "recording_id": manifest_data.get("recording_id"),
        "ticket_id": manifest_data.get("ticket_id"),
        "duration_seconds": manifest_data.get("duration_seconds"),
        "created": manifest_data.get("created"),
        "word_count": manifest_data.get("transcript_word_count"),
    }
