"""
SRT Segment Merger

Merges consecutive SRT segments into consolidated blocks for better readability.
Whisper segments based on audio pauses, which can create many short segments.
This post-processor merges them based on sentence boundaries and duration limits.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class SrtSegment:
    """Represents a single SRT subtitle segment."""
    index: int
    start_time: str  # Format: HH:MM:SS,mmm
    end_time: str
    text: str

    def start_seconds(self) -> float:
        """Convert start time to seconds."""
        return _time_to_seconds(self.start_time)

    def end_seconds(self) -> float:
        """Convert end time to seconds."""
        return _time_to_seconds(self.end_time)

    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_seconds() - self.start_seconds()


def _time_to_seconds(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to seconds."""
    # Handle both comma and period as decimal separator
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _seconds_to_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_srt(srt_path: Path) -> List[SrtSegment]:
    """Parse SRT file into list of segments."""
    content = srt_path.read_text(encoding='utf-8')
    segments = []

    # Split by double newline (segment separator)
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_line = lines[1]
                time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)
                if time_match:
                    start_time = time_match.group(1)
                    end_time = time_match.group(2)
                    text = '\n'.join(lines[2:])
                    segments.append(SrtSegment(index, start_time, end_time, text))
            except (ValueError, IndexError):
                continue

    return segments


def write_srt(segments: List[SrtSegment], output_path: Path) -> None:
    """Write segments to SRT file."""
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{seg.start_time} --> {seg.end_time}")
        lines.append(seg.text)
        lines.append('')  # Empty line between segments

    output_path.write_text('\n'.join(lines), encoding='utf-8')


def merge_srt_segments(
    srt_path: Path,
    max_duration_sec: float = 30.0,
    sentence_endings: str = '.!?'
) -> Path:
    """
    Merge consecutive SRT segments into consolidated blocks.

    Segments are merged until either:
    - Combined duration exceeds max_duration_sec
    - Text ends with sentence-ending punctuation

    Args:
        srt_path: Path to input SRT file
        max_duration_sec: Maximum duration for merged segment (default 30s)
        sentence_endings: Characters that end sentences

    Returns:
        Path to merged SRT file (overwrites original)
    """
    segments = parse_srt(srt_path)

    if not segments:
        return srt_path

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        combined_duration = next_seg.end_seconds() - current.start_seconds()
        current_text = current.text.strip()

        # Check if we should start a new segment
        ends_sentence = current_text and current_text[-1] in sentence_endings
        too_long = combined_duration > max_duration_sec

        if ends_sentence or too_long:
            # Finalize current segment
            merged.append(current)
            current = next_seg
        else:
            # Merge: extend current segment
            current = SrtSegment(
                index=current.index,
                start_time=current.start_time,
                end_time=next_seg.end_time,
                text=current.text.rstrip() + ' ' + next_seg.text.lstrip()
            )

    # Don't forget the last segment
    merged.append(current)

    # Write merged segments back
    write_srt(merged, srt_path)

    return srt_path
