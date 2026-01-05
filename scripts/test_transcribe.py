#!/usr/bin/env python3
"""
Test transcription with a sample audio file.

Usage:
    python scripts/test_transcribe.py <audio_file>
    python scripts/test_transcribe.py  # uses default test file
"""
import asyncio
import sys
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from callwhisper.services.normalizer import normalize_audio
from callwhisper.services.transcriber import transcribe_audio
from callwhisper.utils.paths import get_models_dir
from callwhisper.core.config import get_settings


async def test_transcribe(audio_path: Path):
    """Test transcription pipeline with a given audio file."""
    print(f"\n{'='*60}")
    print(f"Testing Transcription Pipeline")
    print(f"{'='*60}")
    print(f"Input: {audio_path}")
    print(f"Model dir: {get_models_dir()}")

    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return

    # Create output folder
    output_folder = audio_path.parent / "test_output"
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)
    print(f"Output: {output_folder}")

    # Copy audio to expected location
    raw_audio = output_folder / "audio_raw.wav"
    print(f"\n[1/3] Copying audio to {raw_audio.name}...")
    shutil.copy(audio_path, raw_audio)
    print(f"      Size: {raw_audio.stat().st_size / 1024 / 1024:.2f} MB")

    # Normalize to 16kHz mono
    print(f"\n[2/3] Normalizing audio to 16kHz mono...")
    try:
        normalized = await normalize_audio(raw_audio)
        print(f"      Normalized: {normalized}")
        print(f"      Size: {normalized.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"      ERROR: Normalization failed: {e}")
        return

    # Transcribe
    print(f"\n[3/3] Transcribing with Whisper...")
    print(f"      This may take a few minutes...")
    try:
        settings = get_settings()
        transcript_path = await transcribe_audio(output_folder, settings)
        print(f"      Transcript: {transcript_path}")
    except Exception as e:
        print(f"      ERROR: Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Show results
    transcript_txt = output_folder / "transcript.txt"
    if transcript_txt.exists():
        print(f"\n{'='*60}")
        print("TRANSCRIPT:")
        print(f"{'='*60}")
        content = transcript_txt.read_text()
        print(content if content.strip() else "(empty transcript)")
        print(f"{'='*60}")
        print(f"Length: {len(content)} characters")
    else:
        print("\nWARNING: transcript.txt not created")

    # Check for SRT
    transcript_srt = output_folder / "transcript.srt"
    if transcript_srt.exists():
        print(f"\nSRT file created: {transcript_srt}")
        print(f"SRT size: {transcript_srt.stat().st_size} bytes")

    print(f"\nDone! Output files in: {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
    else:
        # Default test file
        audio_file = Path(__file__).parent.parent / "test_audio" / "speech_sample.mp3"

    asyncio.run(test_transcribe(audio_file))
