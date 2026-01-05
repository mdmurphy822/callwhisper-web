# CallWhisper Configuration Guide

Complete reference for all configuration options in CallWhisper v1.0.0.

---

## Configuration File Location

Configuration is stored in `config/config.json` relative to the application directory:

| Deployment | Location |
|------------|----------|
| Development | `./config/config.json` |
| Portable (Windows) | `CallWhisper/config/config.json` |
| Installed | `%APPDATA%/CallWhisper/config/config.json` |

If the file doesn't exist, default values are used.

---

## Complete Configuration Reference

```json
{
  "version": "1.0.0",

  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": true
  },

  "audio": {
    "sample_rate": 44100,
    "channels": 2,
    "format": "pcm_s16le"
  },

  "transcription": {
    "model": "ggml-medium.en.bin",
    "language": "en",
    "beam_size": 5,
    "best_of": 5
  },

  "output": {
    "directory": "output",
    "create_bundle": true,
    "audio_format": "opus"
  },

  "device_guard": {
    "enabled": true,
    "allowlist": ["VB-Cable", "CABLE Output", "Stereo Mix"],
    "blocklist": ["Microphone", "Mic", "Webcam", "Camera", "Headset"]
  },

  "security": {
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:8765", "http://127.0.0.1:8765"],
    "allow_credentials": true,
    "rate_limit_enabled": true,
    "rate_limit_rpm": 60,
    "rate_limit_burst": 10,
    "rate_limit_excluded": ["/api/health", "/api/health/ready", "/api/health/metrics"],
    "debug_endpoints_enabled": false
  },

  "performance": {
    "max_concurrent_transcriptions": 4,
    "chunk_size_seconds": 30.0,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600,
    "cache_max_entries": 100,
    "audio_pool_size": 2,
    "transcription_pool_size": 2,
    "io_pool_size": 4
  },

  "timeouts": {
    "device_enumeration_seconds": 10.0,
    "transcription_min_seconds": 30.0,
    "transcription_max_seconds": 600.0,
    "transcription_ratio": 3.0,
    "health_check_seconds": 5.0,
    "normalization_seconds": 120.0
  },

  "call_detector": {
    "enabled": false,
    "target_processes": ["CiscoJabber.exe"],
    "finesse_browsers": ["chrome.exe", "msedge.exe", "firefox.exe"],
    "finesse_url_pattern": "finesse",
    "call_start_confirm_seconds": 1.0,
    "call_end_confirm_seconds": 2.0,
    "audio_poll_interval": 0.5,
    "process_poll_interval": 2.0,
    "max_call_duration_minutes": 180,
    "min_call_duration_seconds": 5
  }
}
```

---

## Section Reference

### Server Configuration

Controls the HTTP server binding and startup behavior.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | string | "127.0.0.1" | IP address to bind. Use "0.0.0.0" to allow external access (not recommended) |
| `port` | int | 8765 | Port number |
| `open_browser` | bool | true | Automatically open browser on startup |

**Example - Change port:**
```json
{
  "server": {
    "port": 9000
  }
}
```

**Security Note:** Always bind to `127.0.0.1` (localhost) unless you have a specific need for network access and have implemented authentication.

---

### Audio Configuration

Controls audio capture format (passed to FFmpeg).

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `sample_rate` | int | 44100 | Sample rate in Hz |
| `channels` | int | 2 | Number of audio channels (1=mono, 2=stereo) |
| `format` | string | "pcm_s16le" | Audio format (PCM 16-bit little-endian) |

**Note:** These settings affect the raw recording. Audio is normalized to 16kHz mono before transcription.

---

### Transcription Configuration

Controls whisper.cpp transcription settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | "ggml-medium.en.bin" | Whisper model filename in models/ directory |
| `language` | string | "en" | Language code (en, es, fr, de, etc.) |
| `beam_size` | int | 5 | Beam search width (higher = more accurate, slower) |
| `best_of` | int | 5 | Number of candidates to sample |

**Available Models:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| ggml-tiny.en.bin | 75 MB | Fastest | Low | Testing, short clips |
| ggml-base.en.bin | 142 MB | Fast | Medium | General use |
| ggml-small.en.bin | 466 MB | Medium | Good | Recommended for most |
| ggml-medium.en.bin | 1.5 GB | Slow | High | Default, best accuracy |
| ggml-large-v3.bin | 3 GB | Slowest | Highest | Maximum quality |

**Example - Use smaller model:**
```json
{
  "transcription": {
    "model": "ggml-small.en.bin",
    "beam_size": 3
  }
}
```

---

### Output Configuration

Controls recording output format and location.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `directory` | string | "output" | Output directory for recordings |
| `create_bundle` | bool | true | Create .vtb bundle files |
| `audio_format` | string | "opus" | Compressed audio format in bundle |

**Audio Formats:**
- `opus` - Best compression, good quality (recommended)
- `mp3` - Widely compatible
- `wav` - Uncompressed (large files)

---

### Device Guard Configuration

Controls microphone protection settings.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | true | Enable device safety checking |
| `allowlist` | string[] | Platform-specific | Devices explicitly allowed |
| `blocklist` | string[] | Common mic names | Devices explicitly blocked |

**Default Allowlist (Windows):**
```json
["VB-Cable", "CABLE Output", "Virtual Cable", "Stereo Mix", "What U Hear", "Jabber", "Finesse"]
```

**Default Allowlist (Linux):**
```json
["Monitor", "monitor", "Loopback", "loopback", "null", "pipewire"]
```

**Default Blocklist (All platforms):**
```json
["Microphone", "Mic", "Webcam", "Camera", "Headset"]
```

**How Device Guard Works:**

1. Check blocklist - if device name contains any blocklist term, BLOCK
2. Check allowlist - if device name contains any allowlist term, ALLOW
3. Default - if not matched, BLOCK (fail-safe)

**Example - Add custom virtual device:**
```json
{
  "device_guard": {
    "allowlist": ["VB-Cable", "CABLE Output", "My Custom Loopback"]
  }
}
```

---

### Security Configuration

Controls security features for production deployment.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cors_enabled` | bool | true | Enable CORS headers |
| `allowed_origins` | string[] | localhost only | Allowed CORS origins |
| `allow_credentials` | bool | true | Allow credentials in CORS |
| `rate_limit_enabled` | bool | true | Enable request rate limiting |
| `rate_limit_rpm` | int | 60 | Max requests per minute per IP |
| `rate_limit_burst` | int | 10 | Burst allowance |
| `rate_limit_excluded` | string[] | Health endpoints | Paths excluded from rate limiting |
| `debug_endpoints_enabled` | bool | false | Enable /api/debug/* endpoints |

**Production Hardening:**
```json
{
  "security": {
    "debug_endpoints_enabled": false,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:8765"],
    "rate_limit_enabled": true,
    "rate_limit_rpm": 30
  }
}
```

**Development Settings:**
```json
{
  "security": {
    "debug_endpoints_enabled": true,
    "rate_limit_enabled": false
  }
}
```

---

### Performance Configuration

Controls concurrency, caching, and resource pools.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_concurrent_transcriptions` | int | 4 | Max parallel transcription chunks |
| `chunk_size_seconds` | float | 30.0 | Audio chunk size for parallel processing |
| `cache_enabled` | bool | true | Enable transcription result caching |
| `cache_ttl_seconds` | int | 3600 | Cache entry lifetime (1 hour) |
| `cache_max_entries` | int | 100 | Maximum cached entries |
| `audio_pool_size` | int | 2 | Thread pool for audio operations |
| `transcription_pool_size` | int | 2 | Thread pool for transcription |
| `io_pool_size` | int | 4 | Thread pool for I/O operations |

**Low-Resource System:**
```json
{
  "performance": {
    "max_concurrent_transcriptions": 2,
    "audio_pool_size": 1,
    "transcription_pool_size": 1,
    "cache_max_entries": 50
  }
}
```

**High-Performance System:**
```json
{
  "performance": {
    "max_concurrent_transcriptions": 8,
    "audio_pool_size": 4,
    "transcription_pool_size": 4,
    "io_pool_size": 8,
    "cache_max_entries": 500
  }
}
```

---

### Timeout Configuration

Controls operation timeouts.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device_enumeration_seconds` | float | 10.0 | Timeout for listing audio devices |
| `transcription_min_seconds` | float | 30.0 | Minimum transcription timeout |
| `transcription_max_seconds` | float | 600.0 | Maximum transcription timeout (10 min) |
| `transcription_ratio` | float | 3.0 | Timeout = audio_duration * ratio |
| `health_check_seconds` | float | 5.0 | Health check timeout |
| `normalization_seconds` | float | 120.0 | Audio normalization timeout |

**Transcription Timeout Formula:**
```
timeout = max(min_seconds, min(max_seconds, audio_duration * ratio))
```

For a 2-minute recording:
```
timeout = max(30, min(600, 120 * 3)) = max(30, min(600, 360)) = 360 seconds
```

**Long Recording Support:**
```json
{
  "timeouts": {
    "transcription_max_seconds": 1800,
    "normalization_seconds": 300
  }
}
```

---

### Call Detector Configuration (Windows Only)

Controls automatic call detection for Cisco Jabber/Finesse.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | false | Enable automatic call detection |
| `target_processes` | string[] | ["CiscoJabber.exe"] | Processes to monitor |
| `finesse_browsers` | string[] | Chrome, Edge, Firefox | Browser processes for Finesse |
| `finesse_url_pattern` | string | "finesse" | Pattern to match in window title |
| `call_start_confirm_seconds` | float | 1.0 | Debounce for call start |
| `call_end_confirm_seconds` | float | 2.0 | Debounce for call end |
| `audio_poll_interval` | float | 0.5 | Audio session polling interval |
| `process_poll_interval` | float | 2.0 | Process polling interval |
| `max_call_duration_minutes` | int | 180 | Auto-stop after 3 hours |
| `min_call_duration_seconds` | int | 5 | Discard calls shorter than 5s |

**Enable Call Detection:**
```json
{
  "call_detector": {
    "enabled": true,
    "target_processes": ["CiscoJabber.exe", "CiscoCollabHost.exe"]
  }
}
```

**Requirements:**
- Windows 10/11 only
- Python packages: pycaw, wmi, pywin32, comtypes
- Jabber or Finesse must be using audio

---

## Example Configurations

### Development Configuration

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": true
  },
  "transcription": {
    "model": "ggml-small.en.bin"
  },
  "security": {
    "debug_endpoints_enabled": true,
    "rate_limit_enabled": false
  },
  "device_guard": {
    "enabled": false
  }
}
```

### Production Configuration

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": false
  },
  "transcription": {
    "model": "ggml-medium.en.bin"
  },
  "security": {
    "debug_endpoints_enabled": false,
    "cors_enabled": true,
    "allowed_origins": ["http://localhost:8765"],
    "rate_limit_enabled": true,
    "rate_limit_rpm": 60
  },
  "device_guard": {
    "enabled": true
  }
}
```

### Enterprise Configuration

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8765,
    "open_browser": false
  },
  "transcription": {
    "model": "ggml-medium.en.bin",
    "beam_size": 5
  },
  "output": {
    "directory": "C:/CallWhisper/Recordings",
    "create_bundle": true
  },
  "security": {
    "debug_endpoints_enabled": false,
    "rate_limit_enabled": true,
    "rate_limit_rpm": 30
  },
  "device_guard": {
    "enabled": true,
    "allowlist": ["VB-Cable", "CABLE Output", "Stereo Mix"]
  },
  "call_detector": {
    "enabled": true,
    "target_processes": ["CiscoJabber.exe"]
  },
  "performance": {
    "max_concurrent_transcriptions": 4,
    "cache_enabled": true,
    "cache_ttl_seconds": 7200
  }
}
```

---

## Security Hardening Checklist

Before deploying to production:

- [ ] Set `debug_endpoints_enabled: false`
- [ ] Enable rate limiting
- [ ] Bind to localhost only (`host: "127.0.0.1"`)
- [ ] Enable device guard
- [ ] Review allowlist - remove unnecessary devices
- [ ] Set appropriate file permissions on config.json
- [ ] Verify network guard is enabled (check /api/health)
- [ ] Test with `/api/health/ready` endpoint

---

## Reloading Configuration

Configuration is loaded once at startup. To apply changes:

1. Stop the application
2. Edit `config/config.json`
3. Restart the application

There is no hot-reload feature to prevent configuration changes during active recordings.

---

## Troubleshooting Configuration Issues

### Config file not found

The application will use default values. Create the config directory and file:

```bash
mkdir -p config
echo '{}' > config/config.json
```

### Invalid JSON

Check for syntax errors:
```bash
python -c "import json; json.load(open('config/config.json'))"
```

### Unknown options ignored

Unknown configuration keys are silently ignored. Check spelling if an option isn't working.

### Device not appearing in allowlist

Device names must be substring matches. Check exact device name with:
```bash
curl http://localhost:8765/api/devices
```

Then add the exact name or a unique substring to the allowlist.
