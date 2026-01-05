"""
Folder Scanner Service

Scans directories for audio files suitable for batch transcription.
"""

from pathlib import Path
from typing import List, Set
from dataclasses import dataclass
from datetime import datetime

from ..core.logging_config import get_logger

logger = get_logger(__name__)


# Supported audio file extensions
SUPPORTED_EXTENSIONS: Set[str] = {
    '.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac', '.wma', '.opus'
}

# Maximum file size for batch processing (500MB)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024


@dataclass
class ScannedFile:
    """Information about a scanned audio file."""
    path: Path
    filename: str
    size_bytes: int
    modified_at: float
    extension: str

    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return self.size_bytes / (1024 * 1024)


def is_audio_file(path: Path) -> bool:
    """Check if a file is a supported audio format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def scan_folder(
    folder_path: Path,
    recursive: bool = False,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> List[ScannedFile]:
    """
    Scan folder for audio files.

    Args:
        folder_path: Directory to scan
        recursive: If True, scan subdirectories
        max_size_bytes: Maximum file size to include

    Returns:
        List of ScannedFile objects sorted by filename
    """
    if not folder_path.exists():
        raise ValueError(f"Path does not exist: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    pattern = "**/*" if recursive else "*"
    files: List[ScannedFile] = []
    skipped_count = 0
    too_large_count = 0

    for ext in SUPPORTED_EXTENSIONS:
        for path in folder_path.glob(f"{pattern}{ext}"):
            if not path.is_file():
                continue

            try:
                stat = path.stat()
                size = stat.st_size

                if size > max_size_bytes:
                    too_large_count += 1
                    logger.debug(
                        "file_too_large_skipped",
                        path=str(path),
                        size_mb=size / (1024 * 1024)
                    )
                    continue

                if size == 0:
                    skipped_count += 1
                    continue

                files.append(ScannedFile(
                    path=path,
                    filename=path.name,
                    size_bytes=size,
                    modified_at=stat.st_mtime,
                    extension=path.suffix.lower(),
                ))
            except (OSError, PermissionError) as e:
                skipped_count += 1
                logger.warning(
                    "file_scan_error",
                    path=str(path),
                    error=str(e)
                )

    # Sort by filename (case-insensitive)
    files.sort(key=lambda f: f.filename.lower())

    logger.info(
        "folder_scan_complete",
        folder=str(folder_path),
        recursive=recursive,
        files_found=len(files),
        skipped=skipped_count,
        too_large=too_large_count
    )

    return files


def scan_folder_paths(
    folder_path: Path,
    recursive: bool = False,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> List[Path]:
    """
    Scan folder and return just the paths.

    Convenience wrapper around scan_folder for simpler use cases.

    Args:
        folder_path: Directory to scan
        recursive: If True, scan subdirectories
        max_size_bytes: Maximum file size to include

    Returns:
        List of Path objects sorted by filename
    """
    scanned = scan_folder(folder_path, recursive, max_size_bytes)
    return [f.path for f in scanned]


def get_folder_stats(folder_path: Path, recursive: bool = False) -> dict:
    """
    Get statistics about audio files in a folder.

    Args:
        folder_path: Directory to analyze
        recursive: If True, include subdirectories

    Returns:
        Dictionary with folder statistics
    """
    files = scan_folder(folder_path, recursive)

    if not files:
        return {
            "total_files": 0,
            "total_size_mb": 0,
            "extensions": {},
            "oldest_file": None,
            "newest_file": None,
        }

    total_size = sum(f.size_bytes for f in files)
    extensions: dict = {}
    for f in files:
        ext = f.extension
        if ext not in extensions:
            extensions[ext] = {"count": 0, "size_bytes": 0}
        extensions[ext]["count"] += 1
        extensions[ext]["size_bytes"] += f.size_bytes

    oldest = min(files, key=lambda f: f.modified_at)
    newest = max(files, key=lambda f: f.modified_at)

    return {
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "extensions": {
            ext: {
                "count": data["count"],
                "size_mb": round(data["size_bytes"] / (1024 * 1024), 2)
            }
            for ext, data in extensions.items()
        },
        "oldest_file": {
            "filename": oldest.filename,
            "modified_at": datetime.fromtimestamp(oldest.modified_at).isoformat(),
        },
        "newest_file": {
            "filename": newest.filename,
            "modified_at": datetime.fromtimestamp(newest.modified_at).isoformat(),
        },
    }
