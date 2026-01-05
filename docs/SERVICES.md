# CallWhisper Services Reference

Comprehensive documentation of all service modules in CallWhisper.

---

## Table of Contents

1. [Overview](#overview)
2. [Audio Services](#audio-services)
3. [Transcription Services](#transcription-services)
4. [Output Services](#output-services)
5. [Device Services](#device-services)
6. [Orchestration Services](#orchestration-services)
7. [Call Detection Services](#call-detection-services-windows)

---

## Overview

The Services Layer contains business logic modules organized by domain:

```
services/                        # 18 modules
├── Audio Services
│   ├── recorder.py             # FFmpeg audio capture
│   ├── normalizer.py           # Audio resampling to 16kHz
│   ├── audio_chunker.py        # Split audio for parallel processing
│   └── audio_detector.py       # Voice activity detection
│
├── Transcription Services
│   ├── transcriber.py          # Whisper.cpp orchestration
│   ├── audio_pipeline.py       # Full processing pipeline
│   └── srt_merger.py           # Merge SRT segments
│
├── Output Services
│   ├── bundler.py              # VTB bundle creation
│   └── exporter.py             # Multi-format export
│
├── Device Services
│   ├── device_enum.py          # Platform-specific device listing
│   └── device_guard.py         # Microphone protection
│
├── Orchestration Services
│   ├── process_orchestrator.py # End-to-end workflow
│   ├── job_queue.py            # Batch processing queue
│   └── folder_scanner.py       # Folder import scanning
│
└── Call Detection Services (Windows only)
    ├── call_detector.py        # Automatic call detection FSM
    ├── windows_audio_monitor.py # WASAPI audio session monitoring
    └── process_monitor.py      # Process existence monitoring
```

---

## Audio Services

### recorder.py - FFmpeg Audio Capture

Manages FFmpeg subprocess for cross-platform audio recording.

#### Key Functions

```python
async def start_recording(
    session: RecordingSession,
    settings: Settings
) -> Path:
    """
    Start recording audio from specified device.

    Args:
        session: Recording session with device info
        settings: Application settings

    Returns:
        Path to the output folder

    Raises:
        RecordingError: If recording fails to start
    """
```

```python
async def stop_recording() -> Optional[Path]:
    """
    Stop the current recording.

    Returns:
        Path to raw audio file, or None if no recording active
    """
```

```python
def is_recording() -> bool:
    """Check if currently recording."""
```

#### Platform Support

| Platform | Backend | Device Format |
|----------|---------|---------------|
| Windows | DirectShow | `audio=Device Name` |
| Linux | PulseAudio | `device_name` |
| Linux | ALSA | `hw:card,device` |
| macOS | AVFoundation | `:device_index` |

#### Output Files

```
output/{session_id}/
├── audio_raw.wav      # Raw PCM recording (44.1kHz stereo)
└── ffmpeg.log         # FFmpeg command and stderr
```

#### Example Usage

```python
from callwhisper.services.recorder import start_recording, stop_recording
from callwhisper.core.state import RecordingSession
from callwhisper.core.config import get_settings

session = RecordingSession(
    id="20241229_120000_ABC",
    device_name="VB-Cable",
    ticket_id="TICKET-123"
)
settings = get_settings()

# Start recording
output_folder = await start_recording(session, settings)

# ... recording happens ...

# Stop recording
raw_audio_path = await stop_recording()
```

---

### normalizer.py - Audio Normalization

Converts audio to whisper-compatible format (16kHz mono WAV).

#### Key Functions

```python
async def normalize_audio(
    input_path: Path,
    output_path: Optional[Path] = None
) -> Path:
    """
    Normalize audio to 16kHz mono WAV.

    Args:
        input_path: Path to input audio file
        output_path: Optional output path (default: audio_16k.wav)

    Returns:
        Path to normalized audio file

    Raises:
        RuntimeError: If normalization fails
    """
```

```python
async def convert_to_opus(
    input_path: Path,
    output_path: Path,
    bitrate: int = 32000
) -> Path:
    """
    Convert audio to Opus format for bundle storage.

    Args:
        input_path: Input audio file
        output_path: Output Opus file
        bitrate: Target bitrate (default 32kbps)

    Returns:
        Path to Opus file
    """
```

```python
async def get_audio_duration(audio_path: Path) -> float:
    """
    Get audio duration in seconds.

    Returns:
        Duration in seconds
    """
```

#### FFmpeg Parameters

**Normalization:**
```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

**Opus Encoding:**
```bash
ffmpeg -i input.wav -c:a libopus -b:a 32k output.opus
```

---

### audio_chunker.py - Audio Segmentation

Splits long audio files for parallel transcription.

#### Key Functions

```python
async def chunk_audio(
    audio_path: Path,
    chunk_duration_seconds: float = 30.0,
    output_dir: Optional[Path] = None
) -> List[AudioChunk]:
    """
    Split audio into chunks for parallel processing.

    Args:
        audio_path: Path to audio file
        chunk_duration_seconds: Duration of each chunk
        output_dir: Directory for chunk files

    Returns:
        List of AudioChunk objects with paths and timings
    """
```

```python
@dataclass
class AudioChunk:
    """Represents a single audio chunk."""
    index: int
    path: Path
    start_seconds: float
    end_seconds: float
    duration_seconds: float
```

#### Chunking Strategy

```
Original Audio: [=====================================] 90 seconds
                    │
                    ▼
Chunk 1:        [==========] 0:00 - 0:30
Chunk 2:        [==========] 0:30 - 1:00
Chunk 3:        [==========] 1:00 - 1:30
```

**Benefits:**
- Parallel transcription across multiple chunks
- Progress reporting per chunk
- Crash recovery (resume from last completed chunk)

---

### audio_detector.py - Voice Activity Detection

Detects audio activity for call detection.

#### Key Functions

```python
def detect_audio_activity(
    audio_data: bytes,
    sample_rate: int = 16000,
    threshold_db: float = -40.0
) -> bool:
    """
    Detect if audio data contains voice activity.

    Args:
        audio_data: Raw PCM audio bytes
        sample_rate: Sample rate in Hz
        threshold_db: RMS threshold in dB

    Returns:
        True if audio activity detected
    """
```

```python
def calculate_rms_db(samples: np.ndarray) -> float:
    """
    Calculate RMS level in decibels.

    Returns:
        RMS level in dB (negative values, -inf to 0)
    """
```

---

## Transcription Services

### transcriber.py - Whisper Integration

Orchestrates whisper.cpp subprocess for transcription.

#### Key Functions

```python
async def transcribe_audio(
    output_folder: Path,
    settings: Settings,
    progress_callback: Optional[Callable] = None,
    partial_transcript_callback: Optional[Callable] = None,
) -> Path:
    """
    Transcribe audio in the output folder.

    Pipeline:
    1. Normalize audio to 16kHz mono
    2. Run whisper.cpp transcription
    3. Save transcript files

    Args:
        output_folder: Folder containing audio_raw.wav
        settings: Application settings
        progress_callback: async callback(percent, stage)
        partial_transcript_callback: async callback(text, is_final)

    Returns:
        Path to transcript.txt

    Raises:
        TranscriptionError: If transcription fails
        ProcessTimeoutError: If transcription times out
    """
```

```python
def calculate_adaptive_timeout(audio_duration_seconds: float) -> int:
    """
    Calculate appropriate timeout based on audio duration.

    Formula: timeout = audio_duration * 3, clamped to [120, 7200] seconds

    Args:
        audio_duration_seconds: Duration of audio

    Returns:
        Timeout in seconds
    """
```

#### Whisper CLI Parameters

```bash
whisper-cli \
    -m models/ggml-medium.en.bin \
    -f audio_16k.wav \
    -l en \
    -bs 5 \
    -bo 5 \
    -osrt \
    -otxt \
    -oj
```

| Parameter | Description |
|-----------|-------------|
| `-m` | Model file path |
| `-f` | Input audio file |
| `-l` | Language code |
| `-bs` | Beam search size |
| `-bo` | Best of N candidates |
| `-osrt` | Output SRT subtitles |
| `-otxt` | Output plain text |
| `-oj` | Output JSON |

#### Output Files

```
output/{session_id}/
├── audio_16k.wav         # Normalized audio
├── transcript.txt        # Plain text transcript
├── transcript.srt        # SRT subtitles
└── transcript.json       # Whisper JSON output (if enabled)
```

#### Timeout Calculation

```
Audio Duration    Calculated Timeout    Clamped Result
30 seconds        90 seconds            120 seconds (min)
5 minutes         15 minutes            15 minutes
30 minutes        90 minutes            90 minutes
60 minutes        180 minutes           120 minutes (max)
```

---

### srt_merger.py - Subtitle Merging

Merges multiple SRT segments from chunked transcription.

#### Key Functions

```python
def merge_srt_segments(
    segment_paths: List[Path],
    output_path: Path,
    chunk_offsets: List[float]
) -> Path:
    """
    Merge multiple SRT files into single file.

    Adjusts timestamps based on chunk offsets.

    Args:
        segment_paths: List of SRT segment files
        output_path: Output merged SRT file
        chunk_offsets: Start time offset for each segment

    Returns:
        Path to merged SRT file
    """
```

#### Merging Process

```
Chunk 1 (0:00-0:30):           Chunk 2 (0:30-1:00):
1                              1
00:00:05,000 --> 00:00:10,000  00:00:05,000 --> 00:00:12,000
Hello world.                   This is the second part.

                    ↓ Merge with offset adjustment

Merged Output:
1
00:00:05,000 --> 00:00:10,000
Hello world.

2
00:00:35,000 --> 00:00:42,000   ← 30s offset added
This is the second part.
```

---

## Output Services

### bundler.py - VTB Bundle Creation

Creates ZIP-based `.vtb` bundles containing recordings and transcripts.

#### Key Functions

```python
async def create_vtb_bundle(
    output_folder: Path,
    session: RecordingSession,
    settings: Settings,
) -> Path:
    """
    Create a VTB bundle from recording artifacts.

    Args:
        output_folder: Folder with recording files
        session: Recording session metadata
        settings: Application settings

    Returns:
        Path to created .vtb bundle
    """
```

```python
def extract_vtb_bundle(
    bundle_path: Path,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Extract VTB bundle contents.

    Returns:
        Dict mapping file type to extracted path
    """
```

#### VTB Bundle Structure

```
recording.vtb (ZIP format)
├── mimetype                    # "application/x-vtb" (uncompressed, first)
├── META-INF/
│   ├── manifest.json           # Recording metadata
│   └── hashes.json             # SHA-256 integrity checksums
├── audio/
│   └── recording.opus          # Opus-compressed audio (or .wav)
└── transcript/
    ├── transcript.txt          # Plain text transcript
    └── transcript.srt          # SRT subtitles
```

#### Manifest Schema

```json
{
    "version": "1.0.0",
    "format": "vtb",
    "created": "2024-12-29T12:00:00Z",
    "generator_name": "CallWhisper",
    "generator_version": "1.0.0",
    "recording_id": "20241229_120000_ABC123",
    "ticket_id": "TICKET-123",
    "start_time": "2024-12-29T12:00:00Z",
    "end_time": "2024-12-29T12:05:30Z",
    "duration_seconds": 330.5,
    "device_name": "VB-Cable",
    "audio_format": "opus",
    "transcript_word_count": 450,
    "files": [
        {"path": "audio/recording.opus", "size": 52480, "type": "audio/opus"},
        {"path": "transcript/transcript.txt", "size": 2048, "type": "text/plain"}
    ]
}
```

---

### exporter.py - Multi-Format Export

Exports transcripts to various formats.

#### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON | `.json` | Structured with metadata |
| VTT | `.vtt` | WebVTT subtitles |
| CSV | `.csv` | Tabular data |
| PDF | `.pdf` | Formatted document |
| DOCX | `.docx` | Word document |

#### Key Functions

```python
class TranscriptExporter:
    """Export transcripts to various formats."""

    async def export_json(self, recording_id: str) -> Path:
        """Export as structured JSON."""

    async def export_vtt(self, recording_id: str) -> Path:
        """Export as WebVTT subtitle file."""

    async def export_csv(self, recording_id: str) -> Path:
        """Export as CSV with timestamps."""

    async def export_pdf(self, recording_id: str) -> Path:
        """Export as formatted PDF document."""

    async def export_docx(self, recording_id: str) -> Path:
        """Export as Word document."""
```

#### JSON Export Structure

```json
{
    "version": "1.0.0",
    "generator": "CallWhisper",
    "exported_at": "2024-12-29T12:30:00Z",
    "recording": {
        "recording_id": "20241229_120000_ABC123",
        "ticket_id": "TICKET-123",
        "created_at": "2024-12-29T12:00:00Z",
        "duration_seconds": 330.5
    },
    "transcript": {
        "text": "Full transcript text...",
        "word_count": 450,
        "segments": [
            {"start": "00:00:00,000", "end": "00:00:05,000", "text": "Hello..."}
        ]
    }
}
```

---

## Device Services

### device_enum.py - Device Enumeration

Platform-specific audio device listing.

#### Key Functions

```python
async def enumerate_audio_devices(
    timeout_seconds: float = 10.0
) -> List[AudioDevice]:
    """
    List available audio devices.

    Uses platform-specific method:
    - Windows: DirectShow via FFmpeg
    - Linux: PulseAudio/ALSA via pactl/arecord
    - macOS: AVFoundation via FFmpeg

    Returns:
        List of AudioDevice objects
    """
```

```python
@dataclass
class AudioDevice:
    """Represents an audio device."""
    name: str
    device_id: str
    is_input: bool
    is_output: bool
    platform_backend: str  # dshow, pulse, alsa, avfoundation
```

#### Platform Detection

```python
from callwhisper.services.device_enum import enumerate_audio_devices

devices = await enumerate_audio_devices()

# Windows example output:
# [
#     AudioDevice(name="VB-Cable", device_id="VB-Cable", is_output=True, ...),
#     AudioDevice(name="Stereo Mix (Realtek)", device_id="Stereo Mix (Realtek)", ...),
# ]

# Linux example output:
# [
#     AudioDevice(name="alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", ...),
#     AudioDevice(name="bluez_sink.XX_XX_XX_XX.monitor", ...),
# ]
```

---

### device_guard.py - Microphone Protection

**CRITICAL SAFETY MODULE** - Ensures the application never records from microphones.

#### Safety Logic

```
Device Name Input
        │
        ▼
┌───────────────────────┐
│ 1. Explicit Blocklist │  → Match? → BLOCKED
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 2. Dangerous Patterns │  → Match? → BLOCKED
│    (microphone, mic,  │
│     webcam, camera)   │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 3. Explicit Allowlist │  → Match? → ALLOWED
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 4. Safe Patterns      │  → Match? → ALLOWED
│    (VB-Cable, monitor,│
│     Stereo Mix)       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ 5. Default: BLOCKED   │  → Fail-safe
└───────────────────────┘
```

#### Key Functions

```python
def is_device_safe(
    device_name: str,
    config: DeviceGuardConfig
) -> bool:
    """
    Check if device is safe to record from.

    Returns:
        True if device is safe (approved output device)
    """
```

```python
def get_device_status(
    device_name: str,
    config: DeviceGuardConfig
) -> Dict[str, Any]:
    """
    Get detailed safety status.

    Returns:
        {
            "safe": bool,
            "reason": str,
            "match_type": str  # blocklist, dangerous_pattern, allowlist, etc.
        }
    """
```

```python
def validate_device_for_recording(
    device_name: str,
    config: DeviceGuardConfig
) -> None:
    """
    Validate device is safe.

    Raises:
        ValueError: If device is not safe
    """
```

#### Safe Patterns by Platform

**Windows:**
- `vb-?cable`
- `cable\s*output`
- `stereo\s*mix`
- `voicemeeter`
- `jabber`, `finesse`, `cisco`

**Linux:**
- `\.monitor$` (PulseAudio monitor sinks)
- `loopback`
- `pipewire`
- `null[-_]?sink`

#### Dangerous Patterns (Always Blocked)

```python
DANGEROUS_PATTERNS = [
    r"\bmicrophone\b(?!\s*out)",  # "microphone" but NOT "microphone output"
    r"\bmic\b(?!\s*out)",          # "mic" but NOT "mic output"
    r"webcam",
    r"\bcamera\b",
    r"headset\s*mic",
    r"built-?in\s*mic",
]
```

---

## Orchestration Services

### process_orchestrator.py - Workflow Orchestration

Coordinates the end-to-end recording and transcription workflow.

#### Key Functions

```python
async def process_recording(
    session: RecordingSession,
    settings: Settings,
    progress_callback: Optional[Callable] = None
) -> CompletedRecording:
    """
    Process a completed recording through the full pipeline.

    Pipeline:
    1. Normalize audio (16kHz mono)
    2. Transcribe with whisper.cpp
    3. Create VTB bundle
    4. Update application state

    Args:
        session: Recording session
        settings: Application settings
        progress_callback: Progress updates

    Returns:
        CompletedRecording with all artifacts
    """
```

```python
async def process_uploaded_file(
    file_path: Path,
    ticket_id: Optional[str],
    settings: Settings,
    progress_callback: Optional[Callable] = None
) -> CompletedRecording:
    """
    Process an uploaded audio file.

    Creates session, runs transcription pipeline.
    """
```

#### Progress Stages

| Stage | Percent | Description |
|-------|---------|-------------|
| Initializing | 0-5% | Setting up session |
| Normalizing | 5-20% | Converting audio format |
| Transcribing | 20-80% | Running whisper.cpp |
| Bundling | 80-95% | Creating VTB bundle |
| Completing | 95-100% | Finalizing |

---

### job_queue.py - Batch Processing

Manages queue for batch transcription jobs.

#### Key Functions

```python
def queue_job(
    file_path: Path,
    ticket_id: Optional[str] = None,
    priority: int = 0
) -> str:
    """
    Add job to processing queue.

    Args:
        file_path: Path to audio file
        ticket_id: Optional ticket ID
        priority: Higher = processed first

    Returns:
        Job ID
    """
```

```python
async def process_queue(
    settings: Settings,
    progress_callback: Optional[Callable] = None
) -> None:
    """
    Process all jobs in queue (FIFO with priority).
    """
```

```python
def get_queue_status() -> Dict[str, Any]:
    """
    Get current queue statistics.

    Returns:
        {
            "queued": int,
            "processing": int,
            "completed": int,
            "failed": int,
            "jobs": [...]
        }
    """
```

#### Job States

```
QUEUED → PROCESSING → COMPLETED
                   ↘
                    FAILED
```

---

### folder_scanner.py - Folder Import

Scans folders for audio files to queue.

#### Key Functions

```python
async def scan_folder(
    folder_path: Path,
    recursive: bool = True,
    ticket_prefix: Optional[str] = None
) -> List[str]:
    """
    Scan folder for audio files and queue them.

    Args:
        folder_path: Folder to scan
        recursive: Include subfolders
        ticket_prefix: Prefix for generated ticket IDs

    Returns:
        List of job IDs queued
    """
```

#### Supported Audio Formats

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`
- `.ogg`
- `.opus`
- `.wma`

---

## Call Detection Services (Windows)

### call_detector.py - Automatic Call Detection

State machine for detecting Cisco Jabber/Finesse calls.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Call Detector State Machine                            │
│                                                                             │
│                          ┌─────────┐                                       │
│               ┌──────────│  IDLE   │──────────┐                           │
│               │          └────┬────┘          │                           │
│               │               │               │                           │
│               │ process       │ audio         │ audio                     │
│               │ not found     │ detected      │ + process                 │
│               │               │               │                           │
│               ▼               ▼               ▼                           │
│         ┌──────────┐   ┌──────────┐   ┌──────────────┐                   │
│         │MONITORING│   │ DEBOUNCE │   │ CALL_ACTIVE  │                   │
│         │_PROCESS  │   │ _START   │   │              │                   │
│         └────┬─────┘   └────┬─────┘   └──────┬───────┘                   │
│              │              │                 │                           │
│              │              │ confirmed       │ audio                     │
│              │              │                 │ stopped                   │
│              │              ▼                 │                           │
│              │        ┌──────────────┐       ▼                           │
│              │        │ CALL_ACTIVE  │  ┌──────────┐                     │
│              │        │              │  │ DEBOUNCE │                     │
│              │        └──────────────┘  │ _END     │                     │
│              │                          └────┬─────┘                     │
│              │                               │ confirmed                  │
│              │                               ▼                            │
│              └───────────────────────────► IDLE                          │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Functions

```python
class CallDetector:
    """Automatic call detection for Windows."""

    async def start(self, config: CallDetectorConfig) -> None:
        """Start call detection monitoring."""

    async def stop(self) -> None:
        """Stop call detection."""

    def get_status(self) -> Dict[str, Any]:
        """Get current detection status."""
```

#### Configuration

```json
{
    "call_detector": {
        "enabled": true,
        "target_processes": ["CiscoJabber.exe", "CiscoCollabHost.exe"],
        "finesse_browsers": ["chrome.exe", "msedge.exe", "firefox.exe"],
        "finesse_url_pattern": "finesse",
        "call_start_confirm_seconds": 1.0,
        "call_end_confirm_seconds": 2.0,
        "max_call_duration_minutes": 180,
        "min_call_duration_seconds": 5
    }
}
```

---

### windows_audio_monitor.py - WASAPI Audio Monitoring

Monitors Windows audio sessions for active audio.

#### Key Functions

```python
class WindowsAudioMonitor:
    """Monitor audio sessions using Windows Core Audio API (WASAPI)."""

    def get_active_audio_sessions(self) -> List[AudioSessionInfo]:
        """
        Get list of processes with active audio sessions.

        Returns:
            List of AudioSessionInfo with process ID and name
        """

    def is_process_producing_audio(self, process_name: str) -> bool:
        """
        Check if specific process is currently producing audio.

        Uses COM interface to query audio session state.
        """
```

#### Requirements

- Windows 10/11
- Python packages: `pycaw`, `comtypes`

---

### process_monitor.py - Process Monitoring

Monitors process existence using Windows WMI.

#### Key Functions

```python
class ProcessMonitor:
    """Monitor process existence using WMI."""

    def is_process_running(self, process_name: str) -> bool:
        """Check if a process is currently running."""

    def get_process_window_title(self, process_name: str) -> Optional[str]:
        """
        Get window title of a process.

        Used to detect Finesse in browser windows.
        """
```

#### Finesse Detection

For Cisco Finesse (web-based), detection checks:
1. Browser process is running (chrome.exe, msedge.exe, firefox.exe)
2. Window title contains "finesse" pattern

```python
# Example: Detect Finesse in Chrome
if process_monitor.is_process_running("chrome.exe"):
    title = process_monitor.get_process_window_title("chrome.exe")
    if title and "finesse" in title.lower():
        # Finesse detected
```

---

## Service Dependencies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Service Dependency Graph                             │
│                                                                             │
│   process_orchestrator                                                      │
│         │                                                                   │
│         ├──► recorder                                                       │
│         │         │                                                         │
│         │         └──► device_enum                                          │
│         │         └──► device_guard                                         │
│         │                                                                   │
│         ├──► normalizer                                                     │
│         │                                                                   │
│         ├──► audio_chunker ───► normalizer                                 │
│         │                                                                   │
│         ├──► transcriber                                                    │
│         │         │                                                         │
│         │         └──► normalizer                                           │
│         │         └──► srt_merger                                           │
│         │                                                                   │
│         ├──► bundler ───► normalizer (for opus conversion)                  │
│         │                                                                   │
│         └──► exporter                                                       │
│                                                                             │
│   call_detector                                                             │
│         │                                                                   │
│         ├──► windows_audio_monitor                                          │
│         │                                                                   │
│         ├──► process_monitor                                                │
│         │                                                                   │
│         └──► process_orchestrator (triggers recording)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

All services use the custom exception hierarchy from `core/exceptions.py`:

| Exception | Service | Description |
|-----------|---------|-------------|
| `RecordingError` | recorder | FFmpeg capture failures |
| `TranscriptionError` | transcriber | Whisper failures |
| `ProcessTimeoutError` | transcriber | Transcription timeout |
| `BundleError` | bundler | VTB creation failures |
| `DeviceBlockedError` | device_guard | Blocked device access |
| `DeviceEnumerationError` | device_enum | Device listing failures |

---

## Logging

All services use structured logging via `structlog`:

```python
from callwhisper.core.logging_config import get_service_logger

logger = get_service_logger()

# Log with context
logger.info(
    "transcription_started",
    session_id="abc123",
    model="medium.en",
    audio_duration_seconds=120.5
)
```
