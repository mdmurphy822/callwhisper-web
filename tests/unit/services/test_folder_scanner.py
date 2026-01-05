"""
Tests for folder scanner service.

Tests directory scanning for audio files:
- Finding supported audio formats
- Recursive scanning
- File size limits
- Folder statistics
"""

import pytest
from pathlib import Path

from callwhisper.services.folder_scanner import (
    is_audio_file,
    scan_folder,
    scan_folder_paths,
    get_folder_stats,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)


class TestIsAudioFile:
    """Tests for is_audio_file function."""

    def test_recognizes_wav(self, tmp_path):
        """Recognizes .wav files."""
        path = tmp_path / "test.wav"
        assert is_audio_file(path) is True

    def test_recognizes_mp3(self, tmp_path):
        """Recognizes .mp3 files."""
        path = tmp_path / "test.mp3"
        assert is_audio_file(path) is True

    def test_recognizes_m4a(self, tmp_path):
        """Recognizes .m4a files."""
        path = tmp_path / "test.m4a"
        assert is_audio_file(path) is True

    def test_case_insensitive(self, tmp_path):
        """Extension matching is case-insensitive."""
        path = tmp_path / "test.WAV"
        assert is_audio_file(path) is True

    def test_rejects_text_file(self, tmp_path):
        """Rejects non-audio files."""
        path = tmp_path / "test.txt"
        assert is_audio_file(path) is False

    def test_rejects_video_file(self, tmp_path):
        """Rejects video files."""
        path = tmp_path / "test.mp4"
        assert is_audio_file(path) is False


class TestScanFolder:
    """Tests for scan_folder function."""

    def test_finds_audio_files(self, tmp_path):
        """Finds audio files in directory."""
        # Create test audio files
        (tmp_path / "audio1.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "audio2.mp3").write_bytes(b"ID3" + b"\x00" * 100)
        (tmp_path / "not_audio.txt").write_text("text file")

        files = scan_folder(tmp_path)

        assert len(files) == 2
        filenames = [f.filename for f in files]
        assert "audio1.wav" in filenames
        assert "audio2.mp3" in filenames
        assert "not_audio.txt" not in filenames

    def test_recursive_scanning(self, tmp_path):
        """Finds files in subdirectories when recursive=True."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "top.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (subdir / "nested.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        # Non-recursive should only find top level
        files_flat = scan_folder(tmp_path, recursive=False)
        assert len(files_flat) == 1
        assert files_flat[0].filename == "top.wav"

        # Recursive should find both
        files_recursive = scan_folder(tmp_path, recursive=True)
        assert len(files_recursive) == 2

    def test_skips_empty_files(self, tmp_path):
        """Skips zero-byte files."""
        (tmp_path / "empty.wav").write_bytes(b"")
        (tmp_path / "valid.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        files = scan_folder(tmp_path)

        assert len(files) == 1
        assert files[0].filename == "valid.wav"

    def test_respects_max_size(self, tmp_path):
        """Skips files exceeding max size."""
        # Create a small file
        (tmp_path / "small.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        # Scan with tiny max size
        files = scan_folder(tmp_path, max_size_bytes=50)

        assert len(files) == 0

    def test_raises_for_missing_path(self, tmp_path):
        """Raises ValueError for non-existent path."""
        with pytest.raises(ValueError, match="does not exist"):
            scan_folder(tmp_path / "nonexistent")

    def test_raises_for_file_path(self, tmp_path):
        """Raises ValueError when path is a file."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="Not a directory"):
            scan_folder(file_path)

    def test_sorts_by_filename(self, tmp_path):
        """Results are sorted by filename."""
        (tmp_path / "zebra.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "alpha.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "Beta.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        files = scan_folder(tmp_path)

        assert files[0].filename == "alpha.wav"
        assert files[1].filename == "Beta.wav"
        assert files[2].filename == "zebra.wav"


class TestScanFolderPaths:
    """Tests for scan_folder_paths function."""

    def test_returns_path_objects(self, tmp_path):
        """Returns list of Path objects."""
        (tmp_path / "test.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        paths = scan_folder_paths(tmp_path)

        assert len(paths) == 1
        assert isinstance(paths[0], Path)
        assert paths[0].name == "test.wav"


class TestGetFolderStats:
    """Tests for get_folder_stats function."""

    def test_returns_stats_dict(self, tmp_path):
        """Returns statistics dictionary."""
        (tmp_path / "test1.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "test2.mp3").write_bytes(b"ID3" + b"\x00" * 200)

        stats = get_folder_stats(tmp_path)

        assert stats["total_files"] == 2
        assert stats["total_size_mb"] >= 0  # Small files round to 0
        assert ".wav" in stats["extensions"]
        assert ".mp3" in stats["extensions"]
        assert stats["oldest_file"] is not None
        assert stats["newest_file"] is not None

    def test_empty_folder_stats(self, tmp_path):
        """Returns zeroed stats for empty folder."""
        stats = get_folder_stats(tmp_path)

        assert stats["total_files"] == 0
        assert stats["total_size_mb"] == 0
        assert stats["extensions"] == {}
        assert stats["oldest_file"] is None
        assert stats["newest_file"] is None
