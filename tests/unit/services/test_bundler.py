"""
Tests for VTB (Voice Transcript Bundle) bundler service.

Tests bundle creation, extraction, and verification:
- Bundle structure (mimetype, manifest, files)
- SHA256 hash verification
- Extraction and info retrieval
"""

import pytest
import json
import zipfile
import wave
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

from callwhisper.services.bundler import (
    create_vtb_bundle,
    extract_vtb,
    verify_vtb,
    get_vtb_info,
    get_content_type,
    get_compression,
    VTB_VERSION,
    VTB_MIMETYPE,
    VTB_EXTENSION,
)
from callwhisper.core.state import RecordingSession


class TestConstants:
    """Tests for VTB format constants."""

    def test_vtb_version(self):
        """Verify VTB version is set."""
        assert VTB_VERSION == "1.0.0"

    def test_vtb_mimetype(self):
        """Verify VTB MIME type is set."""
        assert VTB_MIMETYPE == "application/x-vtb"

    def test_vtb_extension(self):
        """Verify VTB extension is set."""
        assert VTB_EXTENSION == ".vtb"


class TestGetContentType:
    """Tests for content type detection."""

    def test_opus_content_type(self):
        """Verify opus files get correct MIME type."""
        assert get_content_type("audio.opus") == "audio/opus"

    def test_wav_content_type(self):
        """Verify WAV files get correct MIME type."""
        assert get_content_type("audio.wav") == "audio/wav"

    def test_txt_content_type(self):
        """Verify text files get correct MIME type."""
        assert get_content_type("transcript.txt") == "text/plain; charset=utf-8"

    def test_srt_content_type(self):
        """Verify SRT files get correct MIME type."""
        assert get_content_type("transcript.srt") == "application/x-subrip"

    def test_unknown_content_type(self):
        """Verify unknown files get octet-stream type."""
        assert get_content_type("file.xyz") == "application/octet-stream"


class TestGetCompression:
    """Tests for compression method selection."""

    def test_opus_no_compression(self):
        """Verify opus files are stored without compression."""
        assert get_compression("audio.opus") == zipfile.ZIP_STORED

    def test_wav_with_compression(self):
        """Verify WAV files are compressed."""
        assert get_compression("audio.wav") == zipfile.ZIP_DEFLATED

    def test_txt_with_compression(self):
        """Verify text files are compressed."""
        assert get_compression("transcript.txt") == zipfile.ZIP_DEFLATED


class TestCreateVTBBundle:
    """Tests for bundle creation."""

    @pytest.fixture
    def output_folder(self, tmp_path):
        """Create output folder with required files."""
        folder = tmp_path / "recording"
        folder.mkdir()

        # Create audio file
        audio_path = folder / "audio_raw.wav"
        self._create_wav_file(audio_path)

        # Create transcript
        (folder / "transcript.txt").write_text("Hello world transcript")
        (folder / "transcript.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nHello world\n"
        )

        return folder

    def _create_wav_file(self, path: Path, duration: float = 1.0):
        """Create a minimal valid WAV file."""
        sample_rate = 44100
        with wave.open(str(path), 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            frames = int(sample_rate * duration)
            for i in range(frames):
                value = int(32767 * 0.5)
                wav.writeframes(struct.pack('<h', value))

    @pytest.fixture
    def mock_session(self):
        """Create mock recording session."""
        session = MagicMock(spec=RecordingSession)
        session.id = "20241229_120000_TEST"
        session.ticket_id = "TEST-123"
        session.device_name = "Test Device"
        session.start_time = datetime(2024, 12, 29, 12, 0, 0, tzinfo=timezone.utc)
        session.end_time = datetime(2024, 12, 29, 12, 1, 0, tzinfo=timezone.utc)
        return session

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.output.audio_format = "wav"  # Skip opus conversion for tests
        return settings

    @pytest.mark.asyncio
    async def test_creates_vtb_file(self, output_folder, mock_session, mock_settings):
        """Verify VTB bundle file is created."""
        bundle_path = await create_vtb_bundle(
            output_folder, mock_session, mock_settings
        )

        assert bundle_path.exists()
        assert bundle_path.suffix == VTB_EXTENSION

    @pytest.mark.asyncio
    async def test_bundle_structure(self, output_folder, mock_session, mock_settings):
        """Verify VTB bundle has correct structure."""
        bundle_path = await create_vtb_bundle(
            output_folder, mock_session, mock_settings
        )

        with zipfile.ZipFile(bundle_path, 'r') as zf:
            names = zf.namelist()

            # Check required files
            assert "mimetype" in names
            assert "META-INF/manifest.json" in names
            assert "META-INF/hashes.json" in names
            assert any("audio/" in n for n in names)
            assert "transcript/transcript.txt" in names

    @pytest.mark.asyncio
    async def test_mimetype_first_and_uncompressed(
        self, output_folder, mock_session, mock_settings
    ):
        """Verify mimetype is first file and uncompressed."""
        bundle_path = await create_vtb_bundle(
            output_folder, mock_session, mock_settings
        )

        with zipfile.ZipFile(bundle_path, 'r') as zf:
            # First file should be mimetype
            assert zf.namelist()[0] == "mimetype"

            # Should be stored (not compressed)
            info = zf.getinfo("mimetype")
            assert info.compress_type == zipfile.ZIP_STORED

            # Should contain correct MIME type
            content = zf.read("mimetype").decode('ascii')
            assert content == VTB_MIMETYPE

    @pytest.mark.asyncio
    async def test_manifest_content(
        self, output_folder, mock_session, mock_settings
    ):
        """Verify manifest contains correct metadata."""
        bundle_path = await create_vtb_bundle(
            output_folder, mock_session, mock_settings
        )

        with zipfile.ZipFile(bundle_path, 'r') as zf:
            manifest = json.loads(zf.read("META-INF/manifest.json"))

            assert manifest["version"] == VTB_VERSION
            assert manifest["format"] == "vtb"
            assert manifest["recording_id"] == mock_session.id
            assert manifest["ticket_id"] == mock_session.ticket_id
            assert "files" in manifest
            assert manifest["transcript_word_count"] == 3  # "Hello world transcript"


class TestExtractVTB:
    """Tests for bundle extraction."""

    @pytest.fixture
    def sample_vtb(self, tmp_path):
        """Create a sample VTB file for testing."""
        vtb_path = tmp_path / "test.vtb"

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr(
                "META-INF/manifest.json",
                json.dumps({"version": "1.0.0", "recording_id": "test"})
            )
            zf.writestr("transcript/transcript.txt", "Test content")

        return vtb_path

    def test_extracts_files(self, sample_vtb, tmp_path):
        """Verify files are extracted correctly."""
        dest = tmp_path / "extracted"

        result = extract_vtb(sample_vtb, dest)

        assert result == dest
        assert (dest / "mimetype").exists()
        assert (dest / "META-INF/manifest.json").exists()
        assert (dest / "transcript/transcript.txt").exists()

    def test_creates_destination_directory(self, sample_vtb, tmp_path):
        """Verify destination directory is created if not exists."""
        dest = tmp_path / "new" / "nested" / "dir"

        extract_vtb(sample_vtb, dest)

        assert dest.exists()


class TestVerifyVTB:
    """Tests for bundle verification."""

    @pytest.fixture
    def vtb_with_hashes(self, tmp_path):
        """Create VTB with hash verification."""
        from callwhisper.utils.hashing import compute_sha256_bytes

        vtb_path = tmp_path / "test.vtb"
        transcript_content = b"Test transcript content"
        transcript_hash = compute_sha256_bytes(transcript_content)

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("transcript/transcript.txt", transcript_content)
            zf.writestr(
                "META-INF/hashes.json",
                json.dumps({
                    "version": "1.0.0",
                    "algorithm": "sha256",
                    "files": {
                        "transcript/transcript.txt": transcript_hash
                    }
                })
            )

        return vtb_path

    def test_verifies_valid_bundle(self, vtb_with_hashes):
        """Verify valid bundle passes verification."""
        result = verify_vtb(vtb_with_hashes)

        assert result["transcript/transcript.txt"] is True

    def test_detects_corrupted_file(self, tmp_path):
        """Verify corrupted files are detected."""
        vtb_path = tmp_path / "corrupted.vtb"

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("transcript/transcript.txt", "Actual content")
            zf.writestr(
                "META-INF/hashes.json",
                json.dumps({
                    "version": "1.0.0",
                    "algorithm": "sha256",
                    "files": {
                        "transcript/transcript.txt": "wrong_hash_value"
                    }
                })
            )

        result = verify_vtb(vtb_path)

        assert result["transcript/transcript.txt"] is False


class TestGetVTBInfo:
    """Tests for bundle info retrieval."""

    @pytest.fixture
    def sample_vtb(self, tmp_path):
        """Create sample VTB with manifest."""
        vtb_path = tmp_path / "info_test.vtb"

        manifest = {
            "version": "1.0.0",
            "recording_id": "rec_123",
            "ticket_id": "TICKET-456",
            "duration_seconds": 120.5,
            "created": "2024-12-29T12:00:00Z",
            "transcript_word_count": 150,
        }

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("META-INF/manifest.json", json.dumps(manifest))

        return vtb_path

    def test_returns_bundle_info(self, sample_vtb):
        """Verify bundle info is returned correctly."""
        info = get_vtb_info(sample_vtb)

        assert info["recording_id"] == "rec_123"
        assert info["ticket_id"] == "TICKET-456"
        assert info["duration_seconds"] == 120.5
        assert info["word_count"] == 150
        assert "path" in info
        assert "size_bytes" in info
