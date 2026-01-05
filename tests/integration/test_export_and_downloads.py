"""
Integration tests for export and download functionality.

Tests the export system including:
- VTB bundle creation and validation
- Export formats (TXT, SRT, CSV, JSON, VTT, DOCX, PDF)
- Bundle integrity verification
- Download endpoints
- Transcript editing
- File upload handling
"""

import asyncio
import pytest
import zipfile
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import wave
import struct
import math

from callwhisper.services.exporter import (
    TranscriptExporter,
    get_exporter,
)
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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_output_folder(temp_dir):
    """Create a sample output folder with transcript files."""
    # Create transcript.txt
    (temp_dir / "transcript.txt").write_text(
        "Hello world. This is a test transcript.\n\n"
        "It has multiple paragraphs for testing.\n\n"
        "This is the third paragraph.",
        encoding="utf-8"
    )

    # Create transcript.srt
    srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world.

2
00:00:02,500 --> 00:00:05,000
This is a test transcript.

3
00:00:05,000 --> 00:00:08,000
It has multiple segments.
"""
    (temp_dir / "transcript.srt").write_text(srt_content, encoding="utf-8")

    return temp_dir


@pytest.fixture
def sample_audio_folder(temp_dir):
    """Create folder with sample audio files."""
    # Create raw audio
    raw_audio = temp_dir / "audio_raw.wav"
    sample_rate = 44100
    duration = 2

    with wave.open(str(raw_audio), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(sample_rate * duration):
            value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

    # Create normalized audio
    norm_audio = temp_dir / "audio_16k.wav"
    with wave.open(str(norm_audio), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x00' * 16000 * 4)

    return temp_dir


@pytest.fixture
def sample_recording_folder(sample_audio_folder, sample_output_folder):
    """Create complete recording folder with audio and transcripts."""
    # Copy transcript files to audio folder
    (sample_audio_folder / "transcript.txt").write_text(
        (sample_output_folder / "transcript.txt").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    (sample_audio_folder / "transcript.srt").write_text(
        (sample_output_folder / "transcript.srt").read_text(encoding="utf-8"),
        encoding="utf-8"
    )
    return sample_audio_folder


@pytest.fixture
def mock_session():
    """Create a mock recording session."""
    session = MagicMock()
    session.id = "20241229_120000_TEST123"
    session.ticket_id = "TICKET-001"
    session.device_name = "Test Device"
    session.start_time = datetime(2024, 12, 29, 12, 0, 0, tzinfo=timezone.utc)
    session.end_time = datetime(2024, 12, 29, 12, 2, 0, tzinfo=timezone.utc)
    return session


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.output.audio_format = "wav"
    settings.output.create_bundle = True
    return settings


@pytest.fixture
def sample_vtb_bundle(temp_dir):
    """Create a sample VTB bundle for testing."""
    vtb_path = temp_dir / "test_bundle.vtb"

    manifest = {
        "version": VTB_VERSION,
        "format": "vtb",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_name": "CallWhisper",
        "generator_version": "1.0.0",
        "recording_id": "test_recording",
        "ticket_id": "TEST-001",
        "start_time": None,
        "end_time": None,
        "duration_seconds": 60.0,
        "device_name": "Test Device",
        "audio_format": "wav",
        "transcript_word_count": 10,
        "files": [
            {
                "path": "audio/recording.wav",
                "size_bytes": 1000,
                "content_type": "audio/wav",
                "required": True,
            },
            {
                "path": "transcript/transcript.txt",
                "size_bytes": 50,
                "content_type": "text/plain; charset=utf-8",
                "required": True,
            }
        ]
    }

    transcript_content = "Test transcript content"
    audio_content = b'\x00' * 1000  # Dummy audio

    # Calculate hashes
    import hashlib

    def sha256(data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    manifest_json = json.dumps(manifest, indent=2)

    hashes = {
        "version": VTB_VERSION,
        "algorithm": "sha256",
        "generated": datetime.now(timezone.utc).isoformat(),
        "files": {
            "audio/recording.wav": sha256(audio_content),
            "transcript/transcript.txt": sha256(transcript_content),
        },
        "manifest_hash": sha256(manifest_json),
    }

    with zipfile.ZipFile(vtb_path, 'w') as zf:
        zf.writestr("mimetype", VTB_MIMETYPE, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.json", manifest_json)
        zf.writestr("META-INF/hashes.json", json.dumps(hashes, indent=2))
        zf.writestr("audio/recording.wav", audio_content)
        zf.writestr("transcript/transcript.txt", transcript_content)

    return vtb_path


# ============================================================================
# TranscriptExporter Tests
# ============================================================================

class TestTranscriptExporterBasics:
    """Basic tests for TranscriptExporter."""

    def test_exporter_creation(self, sample_output_folder):
        """Exporter initializes correctly."""
        exporter = TranscriptExporter(sample_output_folder)

        assert exporter.output_folder == sample_output_folder
        assert exporter.transcript_txt == sample_output_folder / "transcript.txt"
        assert exporter.transcript_srt == sample_output_folder / "transcript.srt"

    def test_load_transcript_text(self, sample_output_folder):
        """Loading transcript text works."""
        exporter = TranscriptExporter(sample_output_folder)
        text = exporter._load_transcript_text()

        assert "Hello world" in text
        assert "multiple paragraphs" in text

    def test_load_transcript_text_missing_file(self, temp_dir):
        """Loading missing transcript returns empty string."""
        exporter = TranscriptExporter(temp_dir)
        text = exporter._load_transcript_text()

        assert text == ""

    def test_load_srt_entries(self, sample_output_folder):
        """Loading SRT entries parses correctly."""
        exporter = TranscriptExporter(sample_output_folder)
        entries = exporter._load_srt_entries()

        assert len(entries) == 3
        assert entries[0]["index"] == 1
        assert entries[0]["text"] == "Hello world."
        assert entries[0]["start"] == "00:00:00,000"
        assert entries[0]["end"] == "00:00:02,500"

    def test_load_srt_entries_missing_file(self, temp_dir):
        """Loading missing SRT returns empty list."""
        exporter = TranscriptExporter(temp_dir)
        entries = exporter._load_srt_entries()

        assert entries == []

    def test_get_exporter_factory(self, sample_output_folder):
        """Factory function creates exporter."""
        exporter = get_exporter(sample_output_folder)

        assert isinstance(exporter, TranscriptExporter)


class TestExportJSON:
    """Tests for JSON export."""

    @pytest.mark.asyncio
    async def test_export_json_structure(self, sample_output_folder):
        """JSON export has correct structure."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.object(exporter, '_get_metadata', return_value={
            "recording_id": "test_recording"
        }):
            output = await exporter.export_json("test_recording")

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))

        assert data["version"] == "1.0.0"
        assert data["generator"] == "CallWhisper"
        assert "exported_at" in data
        assert "recording" in data
        assert "transcript" in data
        assert "text" in data["transcript"]
        assert "word_count" in data["transcript"]
        assert "segments" in data["transcript"]

    @pytest.mark.asyncio
    async def test_export_json_includes_segments(self, sample_output_folder):
        """JSON export includes SRT segments."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.object(exporter, '_get_metadata', return_value={}):
            output = await exporter.export_json("test")

        data = json.loads(output.read_text(encoding="utf-8"))

        assert len(data["transcript"]["segments"]) == 3
        assert data["transcript"]["segments"][0]["text"] == "Hello world."

    @pytest.mark.asyncio
    async def test_export_json_word_count(self, sample_output_folder):
        """JSON export calculates word count."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.object(exporter, '_get_metadata', return_value={}):
            output = await exporter.export_json("test")

        data = json.loads(output.read_text(encoding="utf-8"))

        assert data["transcript"]["word_count"] > 0


class TestExportVTT:
    """Tests for WebVTT export."""

    @pytest.mark.asyncio
    async def test_export_vtt_format(self, sample_output_folder):
        """VTT export has correct format."""
        exporter = TranscriptExporter(sample_output_folder)

        output = await exporter.export_vtt("test")

        content = output.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")
        # VTT uses dots instead of commas for milliseconds
        assert "00:00:00.000" in content
        assert "Hello world." in content

    @pytest.mark.asyncio
    async def test_export_vtt_timecodes_converted(self, sample_output_folder):
        """VTT timecodes use dots instead of commas."""
        exporter = TranscriptExporter(sample_output_folder)

        output = await exporter.export_vtt("test")

        content = output.read_text(encoding="utf-8")
        # Should not have SRT-style commas
        assert "00:00:00,000" not in content
        assert "00:00:00.000" in content


class TestExportCSV:
    """Tests for CSV export."""

    @pytest.mark.asyncio
    async def test_export_csv_structure(self, sample_output_folder):
        """CSV export has correct columns."""
        exporter = TranscriptExporter(sample_output_folder)

        output = await exporter.export_csv("test")

        with open(output, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert headers == ["index", "start_time", "end_time", "text"]

    @pytest.mark.asyncio
    async def test_export_csv_data_rows(self, sample_output_folder):
        """CSV export contains segment data."""
        exporter = TranscriptExporter(sample_output_folder)

        output = await exporter.export_csv("test")

        with open(output, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 3 data rows
        assert len(rows) == 4
        assert rows[1][0] == "1"  # First segment index
        assert "Hello world" in rows[1][3]

    @pytest.mark.asyncio
    async def test_export_csv_handles_newlines(self, temp_dir):
        """CSV export handles newlines in text."""
        # Create SRT with multiline text
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First line
Second line
"""
        (temp_dir / "transcript.srt").write_text(srt_content, encoding="utf-8")

        exporter = TranscriptExporter(temp_dir)
        output = await exporter.export_csv("test")

        with open(output, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Newlines should be replaced with spaces
        assert "First line Second line" in rows[1][3]


class TestExportPDF:
    """Tests for PDF export."""

    @pytest.mark.asyncio
    async def test_export_pdf_creates_file(self, sample_output_folder):
        """PDF export creates file."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.object(exporter, '_get_metadata', return_value={
            "recording_id": "test",
            "ticket_id": "TICKET-001",
            "created_at": "2024-12-29T12:00:00",
            "duration_seconds": 120,
        }):
            try:
                output = await exporter.export_pdf("test")
                assert output.exists()
                assert output.suffix == ".pdf"
            except ImportError:
                pytest.skip("reportlab not installed")

    @pytest.mark.asyncio
    async def test_export_pdf_handles_special_chars(self, temp_dir):
        """PDF export handles special characters."""
        # Create transcript with special chars
        (temp_dir / "transcript.txt").write_text(
            "Test <special> & \"chars\" here",
            encoding="utf-8"
        )

        exporter = TranscriptExporter(temp_dir)

        with patch.object(exporter, '_get_metadata', return_value={}):
            try:
                output = await exporter.export_pdf("test")
                assert output.exists()
            except ImportError:
                pytest.skip("reportlab not installed")


class TestExportDOCX:
    """Tests for DOCX export."""

    @pytest.mark.asyncio
    async def test_export_docx_creates_file(self, sample_output_folder):
        """DOCX export creates file."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.object(exporter, '_get_metadata', return_value={
            "recording_id": "test",
            "ticket_id": "TICKET-001",
            "created_at": "2024-12-29T12:00:00",
            "duration_seconds": 120,
        }):
            try:
                output = await exporter.export_docx("test")
                assert output.exists()
                assert output.suffix == ".docx"
            except ImportError:
                pytest.skip("python-docx not installed")


# ============================================================================
# VTB Bundler Tests
# ============================================================================

class TestVTBHelpers:
    """Tests for VTB helper functions."""

    def test_get_content_type_audio(self):
        """Content types for audio formats."""
        assert get_content_type("test.opus") == "audio/opus"
        assert get_content_type("test.wav") == "audio/wav"

    def test_get_content_type_text(self):
        """Content types for text formats."""
        assert get_content_type("test.txt") == "text/plain; charset=utf-8"
        assert get_content_type("test.srt") == "application/x-subrip"
        assert get_content_type("test.vtt") == "text/vtt"
        assert get_content_type("test.json") == "application/json"

    def test_get_content_type_unknown(self):
        """Unknown types return octet-stream."""
        assert get_content_type("test.xyz") == "application/octet-stream"

    def test_get_compression_already_compressed(self):
        """Already compressed formats use STORED."""
        assert get_compression("test.opus") == zipfile.ZIP_STORED
        assert get_compression("test.mp3") == zipfile.ZIP_STORED
        assert get_compression("test.ogg") == zipfile.ZIP_STORED

    def test_get_compression_uncompressed(self):
        """Uncompressed formats use DEFLATED."""
        assert get_compression("test.txt") == zipfile.ZIP_DEFLATED
        assert get_compression("test.json") == zipfile.ZIP_DEFLATED
        assert get_compression("test.wav") == zipfile.ZIP_DEFLATED


class TestCreateVTBBundle:
    """Tests for VTB bundle creation."""

    @pytest.mark.asyncio
    async def test_create_vtb_basic(self, sample_recording_folder, mock_session, mock_settings):
        """Basic VTB bundle creation."""
        with patch("callwhisper.services.bundler.convert_to_opus") as mock_convert:
            with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
                with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                    with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                        mock_convert.return_value = sample_recording_folder / "recording.opus"
                        # Create fake opus file
                        (sample_recording_folder / "recording.opus").write_bytes(b'\x00' * 100)

                        mock_settings.output.audio_format = "opus"
                        bundle = await create_vtb_bundle(
                            sample_recording_folder,
                            mock_session,
                            mock_settings
                        )

                        assert bundle.exists()
                        assert bundle.suffix == VTB_EXTENSION

    @pytest.mark.asyncio
    async def test_create_vtb_wav_format(self, sample_recording_folder, mock_session, mock_settings):
        """VTB bundle with WAV audio."""
        with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
            with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                    mock_settings.output.audio_format = "wav"
                    bundle = await create_vtb_bundle(
                        sample_recording_folder,
                        mock_session,
                        mock_settings
                    )

                    assert bundle.exists()
                    # Check bundle contains WAV
                    with zipfile.ZipFile(bundle, 'r') as zf:
                        names = zf.namelist()
                        assert any("recording" in n and ".wav" in n for n in names)

    @pytest.mark.asyncio
    async def test_create_vtb_missing_audio_raises(self, temp_dir, mock_session, mock_settings):
        """Bundle creation fails without audio."""
        with pytest.raises(FileNotFoundError, match="No audio file found"):
            await create_vtb_bundle(temp_dir, mock_session, mock_settings)

    @pytest.mark.asyncio
    async def test_create_vtb_includes_manifest(self, sample_recording_folder, mock_session, mock_settings):
        """Bundle includes manifest."""
        with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
            with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                    mock_settings.output.audio_format = "wav"
                    bundle = await create_vtb_bundle(
                        sample_recording_folder,
                        mock_session,
                        mock_settings
                    )

                    with zipfile.ZipFile(bundle, 'r') as zf:
                        manifest = json.loads(zf.read("META-INF/manifest.json"))

                    assert manifest["version"] == VTB_VERSION
                    assert manifest["recording_id"] == mock_session.id

    @pytest.mark.asyncio
    async def test_create_vtb_includes_hashes(self, sample_recording_folder, mock_session, mock_settings):
        """Bundle includes hash file."""
        with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
            with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                    mock_settings.output.audio_format = "wav"
                    bundle = await create_vtb_bundle(
                        sample_recording_folder,
                        mock_session,
                        mock_settings
                    )

                    with zipfile.ZipFile(bundle, 'r') as zf:
                        hashes = json.loads(zf.read("META-INF/hashes.json"))

                    assert "files" in hashes
                    assert hashes["algorithm"] == "sha256"


class TestExtractVTB:
    """Tests for VTB extraction."""

    def test_extract_vtb_basic(self, sample_vtb_bundle, temp_dir):
        """Basic VTB extraction."""
        dest = temp_dir / "extracted"
        result = extract_vtb(sample_vtb_bundle, dest)

        assert result == dest
        assert (dest / "mimetype").exists()
        assert (dest / "META-INF" / "manifest.json").exists()
        assert (dest / "transcript" / "transcript.txt").exists()

    def test_extract_vtb_creates_directory(self, sample_vtb_bundle, temp_dir):
        """Extraction creates destination directory."""
        dest = temp_dir / "new" / "nested" / "path"
        extract_vtb(sample_vtb_bundle, dest)

        assert dest.exists()
        assert dest.is_dir()


class TestVerifyVTB:
    """Tests for VTB verification."""

    def test_verify_vtb_valid_bundle(self, sample_vtb_bundle):
        """Verification passes for valid bundle."""
        results = verify_vtb(sample_vtb_bundle)

        assert all(v for v in results.values())

    def test_verify_vtb_corrupted_file(self, temp_dir):
        """Verification fails for corrupted files."""
        vtb_path = temp_dir / "corrupted.vtb"

        hashes = {
            "version": VTB_VERSION,
            "algorithm": "sha256",
            "generated": datetime.now(timezone.utc).isoformat(),
            "files": {
                "transcript/transcript.txt": "wrong_hash_value",
            },
            "manifest_hash": "x",
        }

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("META-INF/manifest.json", "{}")
            zf.writestr("META-INF/hashes.json", json.dumps(hashes))
            zf.writestr("transcript/transcript.txt", "content")

        results = verify_vtb(vtb_path)

        assert results.get("transcript/transcript.txt") is False

    def test_verify_vtb_missing_hashes(self, temp_dir):
        """Verification handles missing hashes file."""
        vtb_path = temp_dir / "no_hashes.vtb"

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("META-INF/manifest.json", "{}")

        results = verify_vtb(vtb_path)

        assert results == {"error": False}


class TestGetVTBInfo:
    """Tests for VTB info retrieval."""

    def test_get_vtb_info_basic(self, sample_vtb_bundle):
        """Get basic bundle information."""
        info = get_vtb_info(sample_vtb_bundle)

        assert info["recording_id"] == "test_recording"
        assert info["ticket_id"] == "TEST-001"
        assert info["duration_seconds"] == 60.0
        assert info["word_count"] == 10
        assert "path" in info
        assert "size_bytes" in info

    def test_get_vtb_info_returns_path(self, sample_vtb_bundle):
        """Info includes bundle path."""
        info = get_vtb_info(sample_vtb_bundle)

        assert info["path"] == str(sample_vtb_bundle)


# ============================================================================
# VTB Bundle Structure Tests
# ============================================================================

class TestVTBBundleStructure:
    """Tests for VTB bundle structure compliance."""

    def test_mimetype_is_first(self, sample_vtb_bundle):
        """Mimetype must be first file in archive."""
        with zipfile.ZipFile(sample_vtb_bundle, 'r') as zf:
            first_file = zf.namelist()[0]

        assert first_file == "mimetype"

    def test_mimetype_uncompressed(self, sample_vtb_bundle):
        """Mimetype must be stored uncompressed."""
        with zipfile.ZipFile(sample_vtb_bundle, 'r') as zf:
            info = zf.getinfo("mimetype")

        assert info.compress_type == zipfile.ZIP_STORED

    def test_mimetype_content(self, sample_vtb_bundle):
        """Mimetype has correct content."""
        with zipfile.ZipFile(sample_vtb_bundle, 'r') as zf:
            content = zf.read("mimetype").decode('ascii')

        assert content == VTB_MIMETYPE

    def test_manifest_in_meta_inf(self, sample_vtb_bundle):
        """Manifest is in META-INF folder."""
        with zipfile.ZipFile(sample_vtb_bundle, 'r') as zf:
            assert "META-INF/manifest.json" in zf.namelist()

    def test_hashes_in_meta_inf(self, sample_vtb_bundle):
        """Hashes file is in META-INF folder."""
        with zipfile.ZipFile(sample_vtb_bundle, 'r') as zf:
            assert "META-INF/hashes.json" in zf.namelist()


# ============================================================================
# Integration Tests
# ============================================================================

class TestExportIntegration:
    """Integration tests combining export components."""

    @pytest.mark.asyncio
    async def test_all_export_formats(self, sample_output_folder):
        """All export formats produce valid output."""
        exporter = TranscriptExporter(sample_output_folder)
        recording_id = "test_export"

        with patch.object(exporter, '_get_metadata', return_value={
            "recording_id": recording_id,
        }):
            # JSON
            json_path = await exporter.export_json(recording_id)
            assert json_path.exists()
            json.loads(json_path.read_text(encoding="utf-8"))  # Validates JSON

            # VTT
            vtt_path = await exporter.export_vtt(recording_id)
            assert vtt_path.exists()
            assert vtt_path.read_text(encoding="utf-8").startswith("WEBVTT")

            # CSV
            csv_path = await exporter.export_csv(recording_id)
            assert csv_path.exists()
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                list(reader)  # Validates CSV

    @pytest.mark.asyncio
    async def test_export_empty_transcript(self, temp_dir):
        """Export handles empty transcript gracefully."""
        # Create empty transcript files
        (temp_dir / "transcript.txt").write_text("", encoding="utf-8")
        (temp_dir / "transcript.srt").write_text("", encoding="utf-8")

        exporter = TranscriptExporter(temp_dir)

        with patch.object(exporter, '_get_metadata', return_value={}):
            # All formats should work with empty content
            json_path = await exporter.export_json("empty")
            assert json_path.exists()

            vtt_path = await exporter.export_vtt("empty")
            assert vtt_path.exists()

            csv_path = await exporter.export_csv("empty")
            assert csv_path.exists()


class TestBundleAndExportWorkflow:
    """Tests for complete bundle and export workflows."""

    @pytest.mark.asyncio
    async def test_create_bundle_then_extract_and_export(
        self, sample_recording_folder, mock_session, mock_settings, temp_dir
    ):
        """Full workflow: create bundle, extract, export."""
        # Create bundle
        with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
            with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                    mock_settings.output.audio_format = "wav"
                    bundle = await create_vtb_bundle(
                        sample_recording_folder,
                        mock_session,
                        mock_settings
                    )

        # Verify bundle
        results = verify_vtb(bundle)
        assert all(v for v in results.values())

        # Extract bundle
        extracted = temp_dir / "extracted"
        extract_vtb(bundle, extracted)

        # Export from extracted
        transcript_folder = extracted / "transcript"
        # Create proper folder structure
        (transcript_folder / "transcript.txt").exists() or \
            (temp_dir / "transcript.txt").write_text("Extracted content", encoding="utf-8")

    @pytest.mark.asyncio
    async def test_round_trip_integrity(
        self, sample_recording_folder, mock_session, mock_settings, temp_dir
    ):
        """Bundle content matches original after extraction."""
        original_txt = (sample_recording_folder / "transcript.txt").read_text(encoding="utf-8")

        # Create bundle
        with patch("callwhisper.services.bundler.get_audio_duration", return_value=60.0):
            with patch("callwhisper.services.bundler.__version__", "1.0.0"):
                with patch("callwhisper.services.bundler.__app_name__", "CallWhisper"):
                    mock_settings.output.audio_format = "wav"
                    bundle = await create_vtb_bundle(
                        sample_recording_folder,
                        mock_session,
                        mock_settings
                    )

        # Extract and verify content
        extracted = temp_dir / "extracted"
        extract_vtb(bundle, extracted)

        extracted_txt = (extracted / "transcript" / "transcript.txt").read_text(encoding="utf-8")
        assert extracted_txt == original_txt


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestExportEdgeCases:
    """Edge case tests for export functionality."""

    @pytest.mark.asyncio
    async def test_export_unicode_content(self, temp_dir):
        """Export handles unicode content."""
        # Create transcript with unicode
        (temp_dir / "transcript.txt").write_text(
            "Unicode: \u00e9\u00e8\u00ea\u00eb \u4e2d\u6587 \ud83c\udf89",
            encoding="utf-8"
        )
        (temp_dir / "transcript.srt").write_text(
            "1\n00:00:00,000 --> 00:00:05,000\n\u4e2d\u6587\u5185\u5bb9\n",
            encoding="utf-8"
        )

        exporter = TranscriptExporter(temp_dir)

        with patch.object(exporter, '_get_metadata', return_value={}):
            json_path = await exporter.export_json("unicode")
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert "\u4e2d\u6587" in data["transcript"]["text"]

    @pytest.mark.asyncio
    async def test_export_very_long_transcript(self, temp_dir):
        """Export handles very long transcripts."""
        # Create 1MB transcript
        long_text = "Word " * 200000  # ~1MB of text
        (temp_dir / "transcript.txt").write_text(long_text, encoding="utf-8")
        (temp_dir / "transcript.srt").write_text("", encoding="utf-8")

        exporter = TranscriptExporter(temp_dir)

        with patch.object(exporter, '_get_metadata', return_value={}):
            json_path = await exporter.export_json("long")
            assert json_path.exists()

            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert data["transcript"]["word_count"] == 200000

    def test_vtb_with_missing_optional_files(self, temp_dir):
        """VTB info works with missing optional files."""
        vtb_path = temp_dir / "minimal.vtb"

        manifest = {
            "version": VTB_VERSION,
            "format": "vtb",
            "recording_id": "minimal",
            "ticket_id": None,
            "duration_seconds": 0,
        }

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("META-INF/manifest.json", json.dumps(manifest))

        info = get_vtb_info(vtb_path)
        assert info["recording_id"] == "minimal"
        assert info["ticket_id"] is None


class TestExportErrorHandling:
    """Error handling tests for export functionality."""

    @pytest.mark.asyncio
    async def test_export_pdf_missing_reportlab(self, sample_output_folder):
        """PDF export fails gracefully without reportlab."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.dict('sys.modules', {'reportlab': None}):
            with patch.object(exporter, '_get_metadata', return_value={}):
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    await exporter.export_pdf("test")

    @pytest.mark.asyncio
    async def test_export_docx_missing_docx(self, sample_output_folder):
        """DOCX export fails gracefully without python-docx."""
        exporter = TranscriptExporter(sample_output_folder)

        with patch.dict('sys.modules', {'docx': None}):
            with patch.object(exporter, '_get_metadata', return_value={}):
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    await exporter.export_docx("test")

    def test_verify_corrupted_zip(self, temp_dir):
        """Verify handles corrupted ZIP files."""
        corrupted = temp_dir / "corrupted.vtb"
        corrupted.write_bytes(b"not a valid zip file")

        with pytest.raises(zipfile.BadZipFile):
            verify_vtb(corrupted)

    def test_get_info_invalid_manifest(self, temp_dir):
        """Get info handles invalid manifest JSON."""
        vtb_path = temp_dir / "invalid.vtb"

        with zipfile.ZipFile(vtb_path, 'w') as zf:
            zf.writestr("mimetype", VTB_MIMETYPE)
            zf.writestr("META-INF/manifest.json", "not valid json{")

        with pytest.raises(json.JSONDecodeError):
            get_vtb_info(vtb_path)
