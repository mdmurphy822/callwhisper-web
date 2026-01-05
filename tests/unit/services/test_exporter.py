"""
Tests for transcript export service.

Tests all export formats:
- JSON (structured with metadata)
- VTT (WebVTT subtitle format)
- CSV (tabular segment data)
- PDF (formatted document)
- DOCX (Word document)
"""

import pytest
import json
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from callwhisper.services.exporter import TranscriptExporter, get_exporter


class TestTranscriptExporterInit:
    """Tests for TranscriptExporter initialization."""

    def test_init_sets_paths(self, tmp_path):
        """Verify initialization sets correct file paths."""
        exporter = TranscriptExporter(tmp_path)

        assert exporter.output_folder == tmp_path
        assert exporter.transcript_txt == tmp_path / "transcript.txt"
        assert exporter.transcript_srt == tmp_path / "transcript.srt"

    def test_get_exporter_factory(self, tmp_path):
        """Verify factory function creates exporter."""
        exporter = get_exporter(tmp_path)

        assert isinstance(exporter, TranscriptExporter)
        assert exporter.output_folder == tmp_path


class TestLoadTranscriptText:
    """Tests for _load_transcript_text method."""

    def test_loads_existing_text(self, tmp_path):
        """Verify text file is loaded correctly."""
        transcript_content = "Hello world, this is a test transcript."
        (tmp_path / "transcript.txt").write_text(transcript_content)

        exporter = TranscriptExporter(tmp_path)
        result = exporter._load_transcript_text()

        assert result == transcript_content

    def test_returns_empty_for_missing_file(self, tmp_path):
        """Verify empty string returned when file doesn't exist."""
        exporter = TranscriptExporter(tmp_path)
        result = exporter._load_transcript_text()

        assert result == ""


class TestLoadSRTEntries:
    """Tests for _load_srt_entries method."""

    def test_parses_srt_correctly(self, tmp_path):
        """Verify SRT file is parsed into entries."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a test"""
        (tmp_path / "transcript.srt").write_text(srt_content)

        exporter = TranscriptExporter(tmp_path)
        entries = exporter._load_srt_entries()

        assert len(entries) == 2
        assert entries[0]["index"] == 1
        assert entries[0]["start"] == "00:00:00,000"
        assert entries[0]["end"] == "00:00:02,500"
        assert entries[0]["text"] == "Hello world"
        assert entries[1]["index"] == 2
        assert entries[1]["text"] == "This is a test"

    def test_returns_empty_for_missing_srt(self, tmp_path):
        """Verify empty list returned when SRT doesn't exist."""
        exporter = TranscriptExporter(tmp_path)
        entries = exporter._load_srt_entries()

        assert entries == []

    def test_handles_multiline_text(self, tmp_path):
        """Verify SRT entries with multiple lines are handled."""
        srt_content = """1
00:00:00,000 --> 00:00:03,000
Line one
Line two"""
        (tmp_path / "transcript.srt").write_text(srt_content)

        exporter = TranscriptExporter(tmp_path)
        entries = exporter._load_srt_entries()

        assert len(entries) == 1
        assert entries[0]["text"] == "Line one\nLine two"


class TestJSONExport:
    """Tests for JSON export format."""

    @pytest.fixture
    def exporter_with_transcript(self, tmp_path):
        """Create exporter with sample transcript files."""
        (tmp_path / "transcript.txt").write_text("Hello world")
        (tmp_path / "transcript.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nHello world\n"
        )
        return TranscriptExporter(tmp_path)

    @pytest.mark.asyncio
    async def test_json_export_structure(self, exporter_with_transcript):
        """Verify JSON export has required structure."""
        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_json("test-id")

            assert result.exists()
            data = json.loads(result.read_text())

            assert data["version"] == "1.0.0"
            assert data["generator"] == "CallWhisper"
            assert "exported_at" in data
            assert "recording" in data
            assert "transcript" in data
            assert "text" in data["transcript"]
            assert "word_count" in data["transcript"]
            assert "segments" in data["transcript"]

    @pytest.mark.asyncio
    async def test_json_export_word_count(self, exporter_with_transcript):
        """Verify word count is calculated correctly."""
        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_json("test-id")

            data = json.loads(result.read_text())
            assert data["transcript"]["word_count"] == 2  # "Hello world"

    @pytest.mark.asyncio
    async def test_json_export_includes_segments(self, exporter_with_transcript):
        """Verify SRT segments are included in JSON."""
        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_json("test-id")

            data = json.loads(result.read_text())
            assert len(data["transcript"]["segments"]) == 1
            assert data["transcript"]["segments"][0]["text"] == "Hello world"


class TestVTTExport:
    """Tests for WebVTT subtitle export."""

    @pytest.fixture
    def exporter_with_srt(self, tmp_path):
        """Create exporter with SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
Test subtitle"""
        (tmp_path / "transcript.srt").write_text(srt_content)
        return TranscriptExporter(tmp_path)

    @pytest.mark.asyncio
    async def test_vtt_has_header(self, exporter_with_srt):
        """Verify VTT file starts with WEBVTT header."""
        result = await exporter_with_srt.export_vtt("test-id")

        content = result.read_text()
        assert content.startswith("WEBVTT")

    @pytest.mark.asyncio
    async def test_vtt_timecode_conversion(self, exporter_with_srt):
        """Verify SRT commas are converted to VTT periods."""
        result = await exporter_with_srt.export_vtt("test-id")

        content = result.read_text()
        # VTT uses periods instead of commas in timecodes
        assert "00:00:00.000 --> 00:00:02.500" in content
        assert "," not in content.split("\n")[2]  # First timecode line

    @pytest.mark.asyncio
    async def test_vtt_contains_all_entries(self, exporter_with_srt):
        """Verify all SRT entries are converted."""
        result = await exporter_with_srt.export_vtt("test-id")

        content = result.read_text()
        assert "Hello world" in content
        assert "Test subtitle" in content


class TestCSVExport:
    """Tests for CSV export format."""

    @pytest.fixture
    def exporter_with_srt(self, tmp_path):
        """Create exporter with SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
Test line"""
        (tmp_path / "transcript.srt").write_text(srt_content)
        return TranscriptExporter(tmp_path)

    @pytest.mark.asyncio
    async def test_csv_has_header_row(self, exporter_with_srt):
        """Verify CSV has proper header row."""
        result = await exporter_with_srt.export_csv("test-id")

        with open(result, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["index", "start_time", "end_time", "text"]

    @pytest.mark.asyncio
    async def test_csv_contains_data_rows(self, exporter_with_srt):
        """Verify CSV contains all data rows."""
        result = await exporter_with_srt.export_csv("test-id")

        with open(result, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 2 data rows
        assert len(rows) == 3
        assert rows[1][0] == "1"  # Index
        assert rows[1][3] == "Hello world"  # Text


class TestPDFExport:
    """Tests for PDF export."""

    @pytest.fixture
    def exporter_with_transcript(self, tmp_path):
        """Create exporter with transcript text."""
        (tmp_path / "transcript.txt").write_text("This is a test transcript.")
        return TranscriptExporter(tmp_path)

    @pytest.mark.asyncio
    async def test_pdf_is_created(self, exporter_with_transcript):
        """Verify PDF file is created."""
        pytest.importorskip("reportlab")

        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_pdf("test-id")

            assert result.exists()
            assert result.suffix == ".pdf"

    @pytest.mark.asyncio
    async def test_pdf_is_valid(self, exporter_with_transcript):
        """Verify PDF file starts with PDF header."""
        pytest.importorskip("reportlab")

        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_pdf("test-id")

            # PDF files start with %PDF
            with open(result, 'rb') as f:
                header = f.read(4)
            assert header == b"%PDF"


class TestDOCXExport:
    """Tests for Word document export."""

    @pytest.fixture
    def exporter_with_transcript(self, tmp_path):
        """Create exporter with transcript text."""
        (tmp_path / "transcript.txt").write_text("This is a test transcript.")
        return TranscriptExporter(tmp_path)

    @pytest.mark.asyncio
    async def test_docx_is_created(self, exporter_with_transcript):
        """Verify DOCX file is created."""
        pytest.importorskip("docx")

        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_docx("test-id")

            assert result.exists()
            assert result.suffix == ".docx"

    @pytest.mark.asyncio
    async def test_docx_contains_transcript(self, exporter_with_transcript):
        """Verify DOCX contains transcript text."""
        docx = pytest.importorskip("docx")
        Document = docx.Document

        with patch.object(exporter_with_transcript, '_get_metadata') as mock_meta:
            mock_meta.return_value = {"recording_id": "test-id"}

            result = await exporter_with_transcript.export_docx("test-id")

            doc = Document(str(result))
            full_text = "\n".join([p.text for p in doc.paragraphs])

            assert "This is a test transcript" in full_text
            assert "Call Transcript" in full_text
