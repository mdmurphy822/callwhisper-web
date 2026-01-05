"""
Tests for SRT segment merger service.

Tests subtitle merging functionality:
- Parsing SRT files
- Time conversion functions
- Segment merging logic
- SRT file writing
"""

import pytest
from pathlib import Path

from callwhisper.services.srt_merger import (
    SrtSegment,
    _time_to_seconds,
    _seconds_to_time,
    parse_srt,
    write_srt,
    merge_srt_segments,
)


class TestTimeConversion:
    """Tests for time conversion functions."""

    def test_time_to_seconds_basic(self):
        """Converts basic SRT time to seconds."""
        assert _time_to_seconds("00:00:05,000") == 5.0

    def test_time_to_seconds_with_hours(self):
        """Converts time with hours correctly."""
        assert _time_to_seconds("01:30:45,500") == 5445.5

    def test_time_to_seconds_milliseconds(self):
        """Handles milliseconds correctly."""
        assert _time_to_seconds("00:00:00,123") == 0.123

    def test_seconds_to_time_basic(self):
        """Converts seconds to SRT format."""
        assert _seconds_to_time(5.0) == "00:00:05,000"

    def test_seconds_to_time_with_hours(self):
        """Converts large seconds value with hours."""
        result = _seconds_to_time(5445.5)
        assert result == "01:30:45,500"

    def test_seconds_to_time_milliseconds(self):
        """Preserves milliseconds in output."""
        result = _seconds_to_time(0.123)
        assert result == "00:00:00,123"


class TestSrtSegment:
    """Tests for SrtSegment dataclass."""

    def test_start_seconds(self):
        """Calculates start time in seconds."""
        seg = SrtSegment(1, "00:00:10,000", "00:00:15,000", "Hello")
        assert seg.start_seconds() == 10.0

    def test_end_seconds(self):
        """Calculates end time in seconds."""
        seg = SrtSegment(1, "00:00:10,000", "00:00:15,000", "Hello")
        assert seg.end_seconds() == 15.0

    def test_duration(self):
        """Calculates segment duration."""
        seg = SrtSegment(1, "00:00:10,000", "00:00:15,500", "Hello")
        assert seg.duration() == 5.5


class TestParseSrt:
    """Tests for parse_srt function."""

    def test_parses_single_segment(self, tmp_path):
        """Parses single segment SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello world"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = parse_srt(srt_path)

        assert len(segments) == 1
        assert segments[0].index == 1
        assert segments[0].start_time == "00:00:00,000"
        assert segments[0].end_time == "00:00:02,500"
        assert segments[0].text == "Hello world"

    def test_parses_multiple_segments(self, tmp_path):
        """Parses multi-segment SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Hello

2
00:00:02,500 --> 00:00:05,000
World"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = parse_srt(srt_path)

        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[1].text == "World"

    def test_handles_multiline_text(self, tmp_path):
        """Handles segments with multiline text."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Line one
Line two"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        segments = parse_srt(srt_path)

        assert len(segments) == 1
        assert "Line one\nLine two" == segments[0].text

    def test_handles_empty_file(self, tmp_path):
        """Handles empty SRT file."""
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("")

        segments = parse_srt(srt_path)

        assert segments == []


class TestWriteSrt:
    """Tests for write_srt function."""

    def test_writes_single_segment(self, tmp_path):
        """Writes single segment to SRT format."""
        segments = [SrtSegment(1, "00:00:00,000", "00:00:02,500", "Hello")]
        output_path = tmp_path / "output.srt"

        write_srt(segments, output_path)

        content = output_path.read_text()
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Hello" in content

    def test_renumbers_segments(self, tmp_path):
        """Renumbers segments starting from 1."""
        segments = [
            SrtSegment(5, "00:00:00,000", "00:00:02,500", "First"),
            SrtSegment(10, "00:00:02,500", "00:00:05,000", "Second"),
        ]
        output_path = tmp_path / "output.srt"

        write_srt(segments, output_path)

        content = output_path.read_text()
        lines = content.split('\n')
        assert lines[0] == "1"
        assert "2" in content


class TestMergeSrtSegments:
    """Tests for merge_srt_segments function."""

    def test_merges_short_segments(self, tmp_path):
        """Merges consecutive short segments."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,000 --> 00:00:02,000
world"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        merge_srt_segments(srt_path)

        segments = parse_srt(srt_path)
        assert len(segments) == 1
        assert "Hello world" in segments[0].text

    def test_respects_sentence_endings(self, tmp_path):
        """Stops merging at sentence-ending punctuation."""
        srt_content = """1
00:00:00,000 --> 00:00:01,000
Hello.

2
00:00:01,000 --> 00:00:02,000
World"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        merge_srt_segments(srt_path)

        segments = parse_srt(srt_path)
        assert len(segments) == 2

    def test_respects_max_duration(self, tmp_path):
        """Respects maximum segment duration limit."""
        srt_content = """1
00:00:00,000 --> 00:00:20,000
Part one

2
00:00:20,000 --> 00:00:40,000
Part two"""
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content)

        # Max 30 seconds - should not merge 40 second total
        merge_srt_segments(srt_path, max_duration_sec=30.0)

        segments = parse_srt(srt_path)
        assert len(segments) == 2

    def test_handles_empty_srt(self, tmp_path):
        """Handles empty SRT file gracefully."""
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("")

        result = merge_srt_segments(srt_path)

        assert result == srt_path
