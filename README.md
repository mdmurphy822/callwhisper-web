# CallWhisper

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/callwhisper/callwhisper-web)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Offline Mode](https://img.shields.io/badge/mode-100%25%20offline-green.svg)](#security)

**Local-first voice transcription for Cisco Jabber and Finesse phone calls.**

CallWhisper records audio from virtual audio devices and transcribes speech to text using whisper.cpp - all processing happens locally on your machine with zero external cloud dependencies.

---

## Features

| Feature | Description |
|---------|-------------|
| **Web Interface** | Browser-based UI with real-time status updates via WebSocket |
| **Safe Recording** | Device Guard ensures only approved output devices are recorded (never microphones) |
| **Local Transcription** | Whisper.cpp for offline speech-to-text - no data leaves your machine |
| **VTB Bundles** | Packages recordings and transcripts into portable `.vtb` files |
| **Batch Processing** | Upload multiple files or scan folders for batch transcription |
| **Multiple Export Formats** | Export as TXT, SRT, VTT, CSV, PDF, or DOCX |
| **Auto Call Detection** | Windows-only feature to automatically start/stop recording on Jabber calls |
| **Crash Recovery** | Checkpoints allow resuming interrupted transcriptions |
| **Fully Accessible** | WCAG 2.2 AA compliant with full keyboard navigation |

---

## Quick Start (5 Minutes)

### Option 1: Windows Portable (Recommended)

1. **Download** the latest `CallWhisper_Windows_x64.zip` from releases
2. **Extract** to any folder
3. **Run** `CallWhisper.exe`
4. **Open** http://localhost:8765 in your browser

### Option 2: Development Setup

```bash
# Clone repository
git clone https://github.com/callwhisper/callwhisper-web.git
cd callwhisper-web

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run development server
./scripts/dev_run.sh   # Linux/macOS
# .\scripts\dev_run.ps1  # Windows

# Open http://localhost:8765
```

---

## Installation

### Prerequisites

| Component | Development | Production |
|-----------|-------------|------------|
| Python | 3.11+ | Not required (bundled) |
| FFmpeg | Required | Bundled in vendor/ |
| Whisper Model | Required | Bundled in models/ |
| VB-Cable | Recommended | Required for call recording |

### Windows Production Build

```powershell
# 1. Build executable
.\scripts\build.ps1

# 2. Download vendor binaries (if not present)
.\scripts\download-vendor.ps1

# 3. Run the application
.\dist\CallWhisper\CallWhisper.exe
```

### Linux Development

```bash
# Install system dependencies
sudo apt install ffmpeg python3.11 python3.11-venv

# Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download whisper model
mkdir -p models
wget -O models/ggml-medium.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin

# Run
PYTHONPATH=src python -m callwhisper
```

---

## Usage

### Recording a Call

1. **Setup Audio Routing**: Route Jabber/Finesse audio through VB-Cable or Stereo Mix
2. **Select Device**: Choose your virtual audio device from the dropdown
3. **Start Recording**: Click the record button when the call begins
4. **Stop & Transcribe**: Click stop when finished - transcription starts automatically
5. **Download**: Get your `.vtb` bundle containing audio and transcript

### Uploading Audio Files

```
POST /api/recordings/upload
Content-Type: multipart/form-data

file: <audio_file>
ticket_id: TICKET-123 (optional)
```

Or use the web UI to drag-and-drop audio files for transcription.

### Batch Processing

```bash
# Via API - scan folder
curl -X POST http://localhost:8765/api/queue/import-folder \
  -F "folder_path=/path/to/audio/files" \
  -F "recursive=true" \
  -F "ticket_prefix=BATCH"

# Via API - multiple uploads
curl -X POST http://localhost:8765/api/recordings/batch-upload \
  -F "files=@file1.wav" \
  -F "files=@file2.mp3" \
  -F "ticket_prefix=CALL"
```

### Automatic Call Detection (Windows)

Enable in `config/config.json`:

```json
{
  "call_detector": {
    "enabled": true,
    "target_processes": ["CiscoJabber.exe"]
  }
}
```

Or enable via API:

```bash
curl -X POST http://localhost:8765/api/call-detection/enable
```

---

## Configuration

Configuration is stored in `config/config.json`:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": true
  },
  "transcription": {
    "model": "ggml-medium.en.bin",
    "language": "en"
  },
  "device_guard": {
    "enabled": true,
    "allowlist": ["VB-Cable", "CABLE Output", "Stereo Mix"],
    "blocklist": ["Microphone", "Mic", "Webcam"]
  },
  "security": {
    "debug_endpoints_enabled": false,
    "rate_limit_enabled": true
  },
  "call_detector": {
    "enabled": false,
    "target_processes": ["CiscoJabber.exe"]
  }
}
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete reference.

---

## VTB Bundle Format

CallWhisper packages recordings in `.vtb` format - a ZIP-based container with integrity verification:

```
recording.vtb
├── mimetype                    # application/x-vtb
├── META-INF/
│   ├── manifest.json           # Recording metadata
│   └── hashes.json             # SHA-256 integrity hashes
├── audio/
│   └── recording.opus          # Opus-compressed audio
└── transcript/
    ├── transcript.txt          # Plain text transcript
    └── transcript.srt          # SRT subtitles (if available)
```

Extract with any ZIP tool or use the web UI download button.

---

## API Reference

Base URL: `http://localhost:8765/api`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/devices` | List audio devices |
| POST | `/recording/start` | Start recording |
| POST | `/recording/stop` | Stop and transcribe |
| GET | `/recordings` | List all recordings |
| POST | `/recordings/upload` | Upload audio file |
| GET | `/recordings/{id}/transcript` | Get transcript |
| GET | `/recordings/{id}/export/{format}` | Export (txt/srt/pdf/docx) |
| WS | `/ws` | Real-time status updates |

See [docs/API.md](docs/API.md) for complete API reference with examples.

---

## Security

### Offline-Only Operation

CallWhisper operates **100% offline**:

- **Network Guard**: Blocks all external network connections at the socket level
- **Local Processing**: All transcription uses local whisper.cpp
- **No Telemetry**: Zero data collection or external reporting

Verify offline mode:

```bash
curl http://localhost:8765/api/health
# {"mode": "offline", "network_guard": "enabled", "external_api_calls": "none"}
```

### Device Guard

Multi-layer protection prevents recording from physical microphones:

1. **Explicit Blocklist**: Blocks devices containing "Microphone", "Mic", "Webcam"
2. **Allowlist Approval**: Only explicitly allowed devices (VB-Cable, Stereo Mix)
3. **Fail-Safe Default**: Unknown devices are BLOCKED by default

### Security Hardening for Production

```json
{
  "security": {
    "debug_endpoints_enabled": false,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:8765"],
    "rate_limit_enabled": true,
    "rate_limit_rpm": 60
  }
}
```

See [SECURITY.md](SECURITY.md) for complete security documentation.

---

## Project Structure

```
callwhisper-web/
├── src/callwhisper/          # Python backend (50 files)
│   ├── api/                  # REST & WebSocket endpoints
│   │   ├── routes.py         # All API endpoints
│   │   └── websocket.py      # Real-time communication
│   ├── core/                 # Core infrastructure (21 modules)
│   │   ├── config.py         # Configuration management
│   │   ├── state.py          # Application state
│   │   ├── bulkhead.py       # Thread pool isolation
│   │   └── network_guard.py  # Network isolation
│   ├── services/             # Business logic (18 modules)
│   │   ├── recorder.py       # FFmpeg audio capture
│   │   ├── transcriber.py    # Whisper transcription
│   │   ├── bundler.py        # VTB bundle creation
│   │   ├── device_guard.py   # Microphone protection
│   │   └── call_detector.py  # Automatic call detection
│   └── utils/                # Helpers
│
├── static/                   # Web frontend
│   ├── index.html            # Main HTML
│   ├── css/styles.css        # Styling (2,274 lines)
│   └── js/                   # JavaScript modules (5,600 lines)
│
├── tests/                    # Test suite (54 files)
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
│
├── vendor/                   # Third-party binaries
├── models/                   # Whisper ML models
├── config/                   # Configuration files
├── output/                   # Recording output directory
├── docs/                     # Documentation
└── scripts/                  # Build & dev scripts
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/callwhisper --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No audio devices found" | Check FFmpeg is installed and accessible |
| "Device blocked by guard" | Add device to allowlist in config.json |
| "Transcription timeout" | Increase `timeouts.transcription_max_seconds` |
| "Model not found" | Download model to `models/ggml-medium.en.bin` |
| Port 8765 in use | Change port in config.json or kill existing process |

### Debug Mode

Enable debug endpoints for troubleshooting:

```json
{
  "security": {
    "debug_endpoints_enabled": true
  }
}
```

Then access:
- `GET /api/debug/state` - Full application state
- `GET /api/debug/paths` - Path configuration
- `GET /api/debug/network` - Network guard status

### Log Files

Logs are written to stdout by default. For file logging, redirect output:

```bash
./CallWhisper.exe > callwhisper.log 2>&1
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [API.md](docs/API.md) | Complete API reference |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Configuration options |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design & patterns |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Development setup |
| [SERVICES.md](docs/SERVICES.md) | Service module reference |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Deployment guide |
| [SECURITY.md](SECURITY.md) | Security documentation |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Run linting (`black src/ && ruff check src/`)
6. Commit your changes
7. Push to the branch
8. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Local transcription engine
- [FFmpeg](https://ffmpeg.org/) - Audio capture and processing
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [VB-Cable](https://vb-audio.com/Cable/) - Virtual audio device
