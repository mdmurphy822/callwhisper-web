# CallWhisper Deployment Guide

Complete guide for deploying CallWhisper in production environments.

---

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Windows Deployment](#windows-deployment)
4. [Linux Deployment](#linux-deployment)
5. [Configuration for Production](#configuration-for-production)
6. [Security Hardening](#security-hardening)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Backup and Recovery](#backup-and-recovery)
9. [Troubleshooting](#troubleshooting)

---

## Deployment Options

| Option | Platform | Best For |
|--------|----------|----------|
| **Portable ZIP** | Windows | Individual users, USB distribution |
| **MSI Installer** | Windows | Enterprise deployment via GPO |
| **Python + venv** | Linux/macOS | Development, custom installations |
| **systemd Service** | Linux | Server deployment |

### Resource Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 3 GB | 10+ GB |
| OS | Windows 10, Ubuntu 20.04 | Latest |

---

## Pre-Deployment Checklist

### Required Components

- [ ] Python 3.11+ (development only - bundled in Windows build)
- [ ] FFmpeg binaries (`ffmpeg.exe`, `ffprobe.exe`)
- [ ] whisper.cpp binary (`whisper-cli.exe`)
- [ ] Whisper model file (`ggml-medium.en.bin`)
- [ ] Virtual audio device (VB-Cable or Stereo Mix)

### Verify Offline Operation

After deployment, verify the application operates fully offline:

```bash
# Check health endpoint
curl http://localhost:8765/api/health

# Expected response:
# {
#     "status": "healthy",
#     "mode": "offline",
#     "network_guard": "enabled",
#     "external_api_calls": "none"
# }
```

### Network Requirements

| Direction | Requirement |
|-----------|-------------|
| Inbound | Port 8765 (localhost only) |
| Outbound | **NONE** - fully offline |

---

## Windows Deployment

### Option 1: Portable ZIP (Recommended)

**Step 1: Download or Build**

Download the latest release:
```
CallWhisper_Windows_x64.zip
```

Or build from source:
```powershell
# Clone repository
git clone https://github.com/callwhisper/callwhisper-web.git
cd callwhisper-web

# Download vendor binaries
.\scripts\download-vendor.ps1

# Build portable package
.\scripts\build.ps1
```

**Step 2: Extract**

Extract to any location:
```
C:\CallWhisper\
├── CallWhisper.exe
├── vendor\
│   ├── ffmpeg.exe
│   ├── ffprobe.exe
│   └── whisper-cli.exe
├── models\
│   └── ggml-medium.en.bin
├── static\
├── config\
│   └── config.json
└── output\
```

**Step 3: First Run**

1. Double-click `CallWhisper.exe`
2. Browser opens to http://localhost:8765
3. Verify health status shows "offline" mode

**Step 4: Create Desktop Shortcut (Optional)**

```powershell
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\CallWhisper.lnk")
$Shortcut.TargetPath = "C:\CallWhisper\CallWhisper.exe"
$Shortcut.WorkingDirectory = "C:\CallWhisper"
$Shortcut.Save()
```

### Option 2: MSI Installer (Enterprise)

Build the MSI installer:
```powershell
# Requires WiX Toolset
.\scripts\build-msi.ps1
```

Deploy via Group Policy:
1. Copy MSI to network share
2. Create GPO with software installation
3. Target appropriate OUs

### File Permissions

Ensure the following permissions:

| Path | Permission |
|------|------------|
| `CallWhisper.exe` | Execute |
| `vendor/` | Read + Execute |
| `models/` | Read |
| `config/` | Read + Write |
| `output/` | Read + Write |
| `checkpoints/` | Read + Write |

### Windows Firewall (Optional)

If needed (localhost binding should not require):

```powershell
# Allow inbound on port 8765 (localhost only)
New-NetFirewallRule -DisplayName "CallWhisper" `
    -Direction Inbound `
    -LocalPort 8765 `
    -Protocol TCP `
    -Action Allow `
    -LocalAddress 127.0.0.1
```

---

## Linux Deployment

### Option 1: Python Virtual Environment

**Step 1: Install System Dependencies**

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv ffmpeg
```

Fedora/RHEL:
```bash
sudo dnf install python3.11 ffmpeg
```

**Step 2: Create Installation Directory**

```bash
sudo mkdir -p /opt/callwhisper
sudo chown $USER:$USER /opt/callwhisper
cd /opt/callwhisper
```

**Step 3: Setup Application**

```bash
# Clone or copy files
git clone https://github.com/callwhisper/callwhisper-web.git .

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Step 4: Download Whisper Model**

```bash
mkdir -p models
wget -O models/ggml-medium.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
```

**Step 5: Build whisper.cpp**

```bash
# Clone whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp.git /tmp/whisper.cpp
cd /tmp/whisper.cpp

# Build
make

# Copy binary
cp main /opt/callwhisper/vendor/whisper-cli
```

**Step 6: Create Configuration**

```bash
mkdir -p /opt/callwhisper/config
cat > /opt/callwhisper/config/config.json << 'EOF'
{
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": false
    },
    "security": {
        "debug_endpoints_enabled": false,
        "rate_limit_enabled": true
    }
}
EOF
```

**Step 7: Test Installation**

```bash
cd /opt/callwhisper
source .venv/bin/activate
PYTHONPATH=src python -m callwhisper
```

### Option 2: systemd Service

**Step 1: Create Service File**

```bash
sudo cat > /etc/systemd/system/callwhisper.service << 'EOF'
[Unit]
Description=CallWhisper Voice Transcription Service
After=network.target pulseaudio.service

[Service]
Type=simple
User=callwhisper
Group=callwhisper
WorkingDirectory=/opt/callwhisper
Environment="PYTHONPATH=/opt/callwhisper/src"
ExecStart=/opt/callwhisper/.venv/bin/python -m callwhisper
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/callwhisper/output /opt/callwhisper/checkpoints /opt/callwhisper/config
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
EOF
```

**Step 2: Create Service User**

```bash
sudo useradd -r -s /bin/false -d /opt/callwhisper callwhisper
sudo chown -R callwhisper:callwhisper /opt/callwhisper
```

**Step 3: Enable and Start**

```bash
sudo systemctl daemon-reload
sudo systemctl enable callwhisper
sudo systemctl start callwhisper
```

**Step 4: Check Status**

```bash
sudo systemctl status callwhisper
journalctl -u callwhisper -f
```

### Linux Audio Setup

For recording system audio on Linux, configure PulseAudio monitor:

```bash
# List available monitors
pactl list short sources | grep monitor

# Example output:
# 0    alsa_output.pci-0000_00_1f.3.analog-stereo.monitor

# Add to config.json allowlist:
# "device_guard": {
#     "allowlist": ["monitor"]
# }
```

---

## Configuration for Production

### Recommended Production Configuration

Create `config/config.json`:

```json
{
    "version": "1.0.0",

    "server": {
        "host": "127.0.0.1",
        "port": 8765,
        "open_browser": false
    },

    "transcription": {
        "model": "ggml-medium.en.bin",
        "language": "en",
        "beam_size": 5
    },

    "output": {
        "directory": "output",
        "create_bundle": true,
        "audio_format": "opus"
    },

    "device_guard": {
        "enabled": true,
        "allowlist": ["VB-Cable", "CABLE Output", "Stereo Mix"],
        "blocklist": ["Microphone", "Mic", "Webcam"]
    },

    "security": {
        "debug_endpoints_enabled": false,
        "cors_enabled": true,
        "allowed_origins": ["http://localhost:8765", "http://127.0.0.1:8765"],
        "rate_limit_enabled": true,
        "rate_limit_rpm": 60,
        "rate_limit_burst": 10
    },

    "performance": {
        "max_concurrent_transcriptions": 4,
        "cache_enabled": true,
        "cache_ttl_seconds": 3600,
        "cache_max_entries": 100
    },

    "timeouts": {
        "transcription_max_seconds": 600,
        "normalization_seconds": 120
    }
}
```

### Environment-Specific Settings

**Development:**
```json
{
    "security": {
        "debug_endpoints_enabled": true,
        "rate_limit_enabled": false
    },
    "device_guard": {
        "enabled": false
    }
}
```

**Production:**
```json
{
    "security": {
        "debug_endpoints_enabled": false,
        "rate_limit_enabled": true
    },
    "device_guard": {
        "enabled": true
    }
}
```

**Enterprise (Cisco Jabber):**
```json
{
    "call_detector": {
        "enabled": true,
        "target_processes": ["CiscoJabber.exe", "CiscoCollabHost.exe"]
    },
    "device_guard": {
        "allowlist": ["VB-Cable", "Stereo Mix", "Jabber"]
    }
}
```

---

## Security Hardening

### Security Checklist

- [ ] **Bind to localhost only** (`host: "127.0.0.1"`)
- [ ] **Disable debug endpoints** (`debug_endpoints_enabled: false`)
- [ ] **Enable rate limiting** (`rate_limit_enabled: true`)
- [ ] **Enable device guard** (`device_guard.enabled: true`)
- [ ] **Review allowlist** - remove unnecessary devices
- [ ] **Set file permissions** - restrict config.json access
- [ ] **Verify network guard** - check health endpoint

### File Permissions

**Windows:**
```powershell
# Restrict config.json to administrators
icacls "C:\CallWhisper\config\config.json" /inheritance:r
icacls "C:\CallWhisper\config\config.json" /grant:r "BUILTIN\Administrators:(R,W)"
icacls "C:\CallWhisper\config\config.json" /grant:r "SYSTEM:(R)"
```

**Linux:**
```bash
# Restrict config.json
chmod 640 /opt/callwhisper/config/config.json
chown callwhisper:callwhisper /opt/callwhisper/config/config.json
```

### Verify Security Settings

```bash
# Check all security features
curl http://localhost:8765/api/health/ready

# Expected output:
# {
#     "ready": true,
#     "checks": {
#         "network_guard": true,      # External connections blocked
#         "device_guard": true,       # Microphone protection enabled
#         "rate_limiter": true,       # Rate limiting active
#         "model_available": true     # Whisper model loaded
#     }
# }
```

---

## Monitoring and Logging

### Log Output

CallWhisper logs to stdout by default. Redirect to file:

**Windows:**
```powershell
.\CallWhisper.exe > callwhisper.log 2>&1
```

**Linux (systemd):**
```bash
# View logs
journalctl -u callwhisper -f

# Export to file
journalctl -u callwhisper > /var/log/callwhisper.log
```

### Health Checks

**Basic Health:**
```bash
curl http://localhost:8765/api/health
```

**Readiness Check (all components):**
```bash
curl http://localhost:8765/api/health/ready
```

**Detailed Metrics:**
```bash
curl http://localhost:8765/api/health/metrics
```

### Prometheus Metrics (if enabled)

```bash
curl http://localhost:8765/api/health/metrics

# Output:
# {
#     "recordings_total": 42,
#     "transcriptions_total": 42,
#     "transcription_duration_seconds": {
#         "avg": 45.2,
#         "min": 12.0,
#         "max": 180.0
#     },
#     "queue_depth": 0,
#     "cache_hit_ratio": 0.85
# }
```

### Alert Conditions

| Metric | Warning | Critical |
|--------|---------|----------|
| `bulkhead_utilization` | > 0.8 | > 0.95 |
| `cache_hit_ratio` | < 0.5 | < 0.2 |
| `queue_depth` | > 10 | > 50 |
| `error_rate` | > 0.05 | > 0.10 |

---

## Backup and Recovery

### Data to Backup

| Data | Location | Priority |
|------|----------|----------|
| Configuration | `config/config.json` | High |
| Recordings | `output/` | High |
| Checkpoints | `checkpoints/` | Medium |

### Backup Script (Windows)

```powershell
$BackupDir = "C:\Backups\CallWhisper"
$Date = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupPath = "$BackupDir\callwhisper_$Date"

New-Item -ItemType Directory -Path $BackupPath -Force

# Backup config
Copy-Item "C:\CallWhisper\config" "$BackupPath\config" -Recurse

# Backup recordings
Copy-Item "C:\CallWhisper\output" "$BackupPath\output" -Recurse

Write-Host "Backup completed: $BackupPath"
```

### Backup Script (Linux)

```bash
#!/bin/bash
BACKUP_DIR="/var/backups/callwhisper"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/callwhisper_$DATE"

mkdir -p "$BACKUP_PATH"

# Backup config
cp -r /opt/callwhisper/config "$BACKUP_PATH/"

# Backup recordings
cp -r /opt/callwhisper/output "$BACKUP_PATH/"

# Compress
tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "callwhisper_$DATE"
rm -rf "$BACKUP_PATH"

echo "Backup completed: $BACKUP_PATH.tar.gz"
```

### Recovery from Checkpoint

If transcription was interrupted:

1. Check for incomplete sessions:
   ```bash
   ls checkpoints/*.checkpoint.json
   ```

2. Restart application - it will detect incomplete sessions
3. Use API to recover or discard:
   ```bash
   # List incomplete jobs
   curl http://localhost:8765/api/jobs/incomplete

   # Recover specific job
   curl -X POST http://localhost:8765/api/jobs/{job_id}/recover
   ```

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 8765
# Windows:
netstat -ano | findstr :8765
taskkill /PID <pid> /F

# Linux:
lsof -i :8765
kill -9 <pid>
```

#### FFmpeg Not Found

```
Error: FFmpeg not found at vendor/ffmpeg.exe
```

**Solution:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Place in `vendor/` directory
3. Verify with: `vendor\ffmpeg.exe -version`

#### Model Not Found

```
Error: Whisper model not found: models/ggml-medium.en.bin
```

**Solution:**
```bash
# Download model
wget -O models/ggml-medium.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
```

#### Device Not Appearing

```
Error: Device 'My Device' is blocked by device guard
```

**Solution:**
1. Check exact device name: `curl http://localhost:8765/api/devices`
2. Add to allowlist in config.json:
   ```json
   {
       "device_guard": {
           "allowlist": ["My Device"]
       }
   }
   ```
3. Restart application

#### Transcription Timeout

```
Error: Transcription timeout after 600 seconds
```

**Solution:**
Increase timeout in config.json:
```json
{
    "timeouts": {
        "transcription_max_seconds": 1800
    }
}
```

#### No Audio Devices (Linux)

```
Error: No audio devices found
```

**Solution:**
1. Check PulseAudio is running: `pulseaudio --check`
2. List available sources: `pactl list short sources`
3. Ensure monitors are available

### Debug Mode

Enable debug endpoints temporarily:

```json
{
    "security": {
        "debug_endpoints_enabled": true
    }
}
```

Then check:
- `GET /api/debug/state` - Full application state
- `GET /api/debug/paths` - Path configuration
- `GET /api/debug/network` - Network guard status
- `GET /api/debug/bulkhead` - Thread pool metrics

### Support

If issues persist:
1. Check logs for error messages
2. Enable debug mode and capture state
3. Open issue at https://github.com/callwhisper/callwhisper-web/issues
