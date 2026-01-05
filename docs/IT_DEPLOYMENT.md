# CallWhisper IT Deployment Guide

## Executive Summary

CallWhisper is a **fully offline** voice transcription application designed for enterprise Windows environments. It records Cisco Jabber/Finesse phone calls and transcribes them locally using whisper.cpp.

**Key Points for IT:**
- **ZERO external network dependencies** - No cloud APIs, no telemetry, no internet required
- **All processing is local** - Audio and transcription never leave the machine
- **Portable or MSI deployment** - Choose based on your environment
- **Minimal system requirements** - Runs on standard Windows 10/11 workstations

---

## Network Requirements

### Inbound Connections
| Port | Protocol | Purpose |
|------|----------|---------|
| 8765 | TCP | Local web UI (localhost only) |

**Note:** The application binds to `127.0.0.1:8765` by default. No external inbound connections are accepted.

### Outbound Connections
| Destination | Purpose |
|-------------|---------|
| **None** | No outbound connections required |

**CallWhisper makes ZERO external network calls.** The application includes a network guard that actively blocks any accidental external connections.

### Firewall Rules
No firewall rules are required. The application operates entirely on localhost.

---

## Installation Methods

### Method 1: Portable Deployment (Recommended for Pilots)

1. Extract `CallWhisper_Windows_x64.zip` to desired location
2. Ensure vendor binaries are present:
   - `vendor/ffmpeg.exe`
   - `vendor/whisper-cli.exe`
3. Ensure whisper model is present:
   - `models/ggml-medium.en.bin`
4. Run `CallWhisper.exe`

**Advantages:**
- No installation required
- Can run from USB drive or network share
- Easy to test before enterprise rollout
- User-level permissions sufficient

### Method 2: MSI Installer (Recommended for Enterprise)

**Silent Installation:**
```powershell
msiexec /i CallWhisper-Setup.msi /qn
```

**Silent Installation with Logging:**
```powershell
msiexec /i CallWhisper-Setup.msi /qn /l*v C:\Logs\CallWhisper-install.log
```

**SCCM/GPO Deployment:**
- Deploy as standard MSI package
- No special parameters required
- Supports per-machine installation

**Uninstall:**
```powershell
msiexec /x CallWhisper-Setup.msi /qn
```

---

## VB-Cable Deployment (Prerequisite)

CallWhisper requires a virtual audio driver to capture call audio. We recommend VB-Cable.

> **Important:** VB-Cable is a third-party driver that requires administrator rights to install. It is NOT bundled with CallWhisper for security reasons (driver installation should be explicit and auditable).

### First-Run Experience

When a user first launches CallWhisper without VB-Cable installed:

1. The application detects no virtual audio devices
2. A setup wizard prompts the user to install VB-Cable
3. User clicks "Download VB-Cable" → opens https://vb-audio.com/Cable/
4. User downloads and installs VB-Cable (requires admin)
5. User restarts CallWhisper
6. Application detects "CABLE Output" device and is ready to use

### Option A: User Self-Install (Small Deployments)

For smaller deployments or pilots, users can install VB-Cable themselves:

1. User launches CallWhisper
2. Setup wizard detects missing virtual audio
3. User clicks "Download VB-Cable"
4. User installs with admin credentials
5. User restarts or rechecks in wizard

**Advantages:**
- No IT intervention required
- Works with standard user + admin elevation
- Audit trail (user explicitly installs driver)

### Option B: IT Pre-Deployment (Enterprise)

For large enterprise deployments, pre-install VB-Cable before deploying CallWhisper.

**SCCM/Intune Deployment:**
```powershell
# Download VB-Cable installer
# https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip

# Extract and run silent install
msiexec /i VBCable_Setup_x64.exe /qn
```

**GPO Deployment:**
1. Download VB-Cable driver package
2. Add to Group Policy Software Installation
3. Target to workstations that will run CallWhisper
4. Deploy before CallWhisper MSI

### Verification

After VB-Cable is installed, verify with PowerShell:

```powershell
# Check for VB-Cable device
Get-WmiObject Win32_SoundDevice | Where-Object { $_.Name -like "*VB*" }

# Or via CallWhisper API
Invoke-RestMethod -Uri "http://localhost:8765/api/setup/status"
```

Expected output includes:
```json
{
    "virtual_audio_detected": true,
    "detected_devices": [
        {"name": "CABLE Output", "type": "vb-cable", "recommended": true}
    ]
}
```

### Alternative Virtual Audio Drivers

VB-Cable is recommended but not required. These alternatives also work:

| Driver | Free | Notes |
|--------|------|-------|
| **VB-Cable** | Yes | Recommended. Simple, reliable |
| **VoiceMeeter** | Yes | Advanced. Includes multiple virtual cables |
| **Stereo Mix** | Built-in | Windows feature. May not work on all hardware |
| **Virtual Audio Cable (VAC)** | No | Commercial alternative |

### Softphone Configuration

After VB-Cable is installed, configure your softphone (Jabber/Finesse) to output audio to "CABLE Input":

1. Open Jabber/Finesse audio settings
2. Set **Speaker** to "CABLE Input (VB-Audio Virtual Cable)"
3. Keep **Microphone** as default (your headset)
4. Test a call - caller audio goes to VB-Cable, your voice stays on headset

CallWhisper will automatically select "CABLE Output" for recording.

---

## Directory Structure

### Portable Mode
```
CallWhisper/
    CallWhisper.exe       # Main application
    version.json          # Version info (enables portable mode detection)
    vendor/               # External binaries
        ffmpeg.exe        # Audio processing
        whisper-cli.exe   # Transcription engine
    models/               # Whisper models
        ggml-medium.en.bin
    data/                 # User data (recordings, config, logs)
        config/
            config.json
        output/           # Recorded audio and transcripts
        logs/             # Application logs
        checkpoints/      # Crash recovery data
```

### MSI Installation
```
C:\Program Files\CallWhisper\
    CallWhisper.exe
    version.json
    vendor/
    models/
    data/
        config/
        output/
        logs/
        checkpoints/
```

---

## Configuration

### Configuration File Location
- Portable: `<install_dir>/data/config/config.json`
- MSI: `C:\Program Files\CallWhisper\data\config\config.json`

### Default Configuration
```json
{
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": true
    },
    "audio": {
        "sample_rate": 44100,
        "channels": 2
    },
    "transcription": {
        "model": "ggml-medium.en.bin",
        "language": "en"
    },
    "device_guard": {
        "enabled": true,
        "allowlist": ["VB-Cable", "CABLE Output", "Stereo Mix", "Jabber", "Finesse"]
    }
}
```

### Important Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `server.host` | Bind address (keep as localhost) | `127.0.0.1` |
| `server.port` | Local web UI port | `8765` |
| `device_guard.enabled` | Prevent recording from physical mics | `true` |
| `device_guard.allowlist` | Allowed virtual audio devices | VB-Cable, etc. |

---

## Security Considerations

### Network Isolation
- Application binds only to localhost (127.0.0.1)
- Network guard blocks all external socket connections
- No DNS lookups for external hosts
- No telemetry or analytics

### Microphone Protection
- Device guard prevents recording from physical microphones
- Only virtual audio cables (VB-Cable, etc.) are allowed by default
- Protects against accidental recording of ambient conversations

### Data Storage
- All recordings stored locally in `data/output/`
- No cloud storage integration
- No data leaves the machine

### Verification Endpoints
IT can verify offline mode by checking:
```
GET http://localhost:8765/api/health
```

Response:
```json
{
    "status": "ok",
    "mode": "offline",
    "network_guard": "enabled",
    "external_api_calls": "none",
    "transcription_engine": "whisper.cpp (local)"
}
```

---

## System Requirements

### Minimum Requirements
- Windows 10 (64-bit) or Windows 11
- 4 GB RAM
- 2 GB disk space (plus space for recordings)
- x64 processor

### Recommended Requirements
- Windows 10/11 (64-bit)
- 8 GB RAM
- SSD storage
- Multi-core processor (faster transcription)

### Software Dependencies
- None (all dependencies bundled)
- No .NET Framework required
- No Visual C++ Redistributable required

---

## Troubleshooting

### Application Won't Start
1. Verify all vendor binaries exist in `vendor/`
2. Check Windows Event Viewer for errors
3. Run from command line to see error output:
   ```cmd
   CallWhisper.exe
   ```

### No Audio Devices Listed
1. Verify virtual audio cable is installed (VB-Cable recommended)
2. Check device_guard allowlist includes your device
3. Verify device appears in Windows Sound settings

### Transcription Fails
1. Verify whisper model exists in `models/`
2. Check available disk space
3. Review logs in `data/logs/`

### Health Check Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8765/api/health"
```

### Readiness Check Endpoint
```powershell
Invoke-RestMethod -Uri "http://localhost:8765/api/health/ready"
```

### Debug Endpoints
```powershell
# Application state
Invoke-RestMethod -Uri "http://localhost:8765/api/debug/state"

# Network isolation status
Invoke-RestMethod -Uri "http://localhost:8765/api/debug/network"

# Path information
Invoke-RestMethod -Uri "http://localhost:8765/api/debug/paths"
```

---

## Logging

### Log Location
- Portable: `<install_dir>/data/logs/`
- MSI: `C:\Program Files\CallWhisper\data\logs\`

### Log Format
Structured JSON logging for easy parsing:
```json
{
    "timestamp": "2024-01-15T10:30:45.123Z",
    "level": "info",
    "event": "recording_started",
    "session_id": "20240115_103045",
    "device": "CABLE Output"
}
```

---

## Backup and Recovery

### Data to Backup
- `data/config/config.json` - User configuration
- `data/output/` - Recordings and transcripts

### Crash Recovery
- Application uses checkpointing for crash recovery
- Incomplete sessions stored in `data/checkpoints/`
- On startup, incomplete sessions are detected and can be resumed

---

## Support Information

### Version Check
```powershell
Get-Content "C:\Program Files\CallWhisper\version.json" | ConvertFrom-Json
```

### Collecting Diagnostics
```powershell
# Create diagnostic bundle
$diagPath = "$env:TEMP\callwhisper-diag"
New-Item -ItemType Directory -Force -Path $diagPath

# Copy logs
Copy-Item "C:\Program Files\CallWhisper\data\logs\*" $diagPath

# Get health status
Invoke-RestMethod -Uri "http://localhost:8765/api/health/ready" |
    ConvertTo-Json | Out-File "$diagPath\health.json"

# Get debug info
Invoke-RestMethod -Uri "http://localhost:8765/api/debug/state" |
    ConvertTo-Json | Out-File "$diagPath\debug-state.json"

# Compress
Compress-Archive -Path $diagPath -DestinationPath "$env:TEMP\callwhisper-diag.zip"
```

---

## FAQ

**Q: Does CallWhisper connect to the internet?**
A: No. CallWhisper has zero external network dependencies and includes an active network guard that blocks all non-localhost connections.

**Q: Does it use Claude, OpenAI, or any cloud AI service?**
A: No. Transcription is performed locally using whisper.cpp, an open-source speech recognition engine that runs entirely on the local machine.

**Q: What audio sources can it record?**
A: By default, only virtual audio cables (VB-Cable, etc.) are allowed to prevent accidental recording from physical microphones.

**Q: Where is data stored?**
A: All data is stored locally in the `data/` subdirectory next to the application. No cloud storage is used.

**Q: Can it be deployed via SCCM/GPO?**
A: Yes. The MSI installer supports standard enterprise deployment methods including silent installation.
