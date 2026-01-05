# CallWhisper API Reference

Complete API documentation for CallWhisper v1.0.0.

---

## Overview

| Property | Value |
|----------|-------|
| Base URL | `http://localhost:8765/api` |
| Protocol | HTTP/1.1, WebSocket |
| Authentication | None (localhost only) |
| Content-Type | `application/json` |
| Rate Limiting | 60 requests/minute (configurable) |

### Response Format

All endpoints return JSON with consistent structure:

**Success Response:**
```json
{
  "status": "ok",
  "data": { ... }
}
```

**Error Response:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 403 | Forbidden - Device blocked by guard |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |
| 501 | Not Implemented - Feature unavailable on this platform |
| 503 | Service Unavailable - Rate limited |

---

## Health Endpoints

### GET /health

Basic health check - is the service running?

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "mode": "offline",
  "network_guard": "enabled",
  "external_api_calls": "none",
  "transcription_engine": "whisper.cpp (local)"
}
```

---

### GET /health/ready

Readiness probe - can the service accept work?

**Response:**
```json
{
  "ready": true,
  "checks": [
    {"name": "ffmpeg", "ready": true, "details": "/path/to/ffmpeg"},
    {"name": "whisper", "ready": true, "details": "/path/to/whisper-cli"},
    {"name": "disk_space", "ready": true, "details": "45.23 GB free"},
    {"name": "app_state", "ready": true, "details": "idle"},
    {"name": "bulkhead", "ready": true, "details": "All pools healthy"},
    {"name": "network_guard", "ready": true, "details": "External connections blocked"}
  ]
}
```

---

### GET /health/detailed

Detailed health check for pre-recording validation.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| device | string | No | Audio device name to validate |

**Response:**
```json
{
  "healthy": true,
  "checks": [
    {
      "name": "ffmpeg",
      "healthy": true,
      "message": "FFmpeg available",
      "details": {"path": "/path/to/ffmpeg", "version": "6.0"}
    },
    {
      "name": "disk_space",
      "healthy": true,
      "message": "Sufficient disk space",
      "details": {"free_gb": 45.23, "required_gb": 1.0}
    }
  ],
  "timestamp": 1704412800.0
}
```

---

### GET /health/metrics

Application metrics for monitoring.

**Response:**
```json
{
  "uptime_seconds": 3600.5,
  "operations": {
    "recordings_started": 15,
    "recordings_completed": 14,
    "transcriptions_completed": 14
  },
  "circuit_breakers": {
    "ffmpeg": {"state": "closed", "failures": 0},
    "whisper": {"state": "closed", "failures": 0}
  },
  "bulkhead_pools": {
    "audio": {"active": 0, "queued": 0, "max_size": 2},
    "transcription": {"active": 1, "queued": 0, "max_size": 2}
  },
  "cache": {
    "hits": 5,
    "misses": 10,
    "entries": 10,
    "max_entries": 100
  },
  "active_recording": false,
  "completed_recordings_count": 14
}
```

---

## Device Endpoints

### GET /devices

List available audio devices with safety status.

**Response:**
```json
{
  "devices": [
    {
      "name": "CABLE Output (VB-Audio Virtual Cable)",
      "safe": true,
      "reason": null
    },
    {
      "name": "Microphone (Realtek Audio)",
      "safe": false,
      "reason": "Blocked: matches blocklist pattern 'Microphone'"
    },
    {
      "name": "Stereo Mix (Realtek Audio)",
      "safe": true,
      "reason": null
    }
  ]
}
```

---

## Recording Endpoints

### POST /recording/start

Start a new recording session.

**Request Body:**
```json
{
  "device": "CABLE Output (VB-Audio Virtual Cable)",
  "ticket_id": "TICKET-123"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| device | string | Yes | Audio device name from /devices |
| ticket_id | string | No | Optional ticket/case ID (max 50 chars) |

**Response:**
```json
{
  "recording_id": "20240105_143022_TICKET-123",
  "started_at": "2024-01-05T14:30:22.123456"
}
```

**Errors:**
- 400: Already recording
- 400: Processing in progress
- 403: Device blocked by guard

---

### POST /recording/stop

Stop the current recording and start transcription.

**Response:**
```json
{
  "recording_id": "20240105_143022_TICKET-123",
  "state": "processing"
}
```

**Errors:**
- 400: Not currently recording

---

## State Endpoints

### GET /state

Get current application state.

**Response:**
```json
{
  "state": "idle",
  "recording_id": null,
  "elapsed_seconds": 0,
  "elapsed_formatted": "00:00"
}
```

**States:**
| State | Description |
|-------|-------------|
| idle | Ready to record |
| recording | Recording in progress |
| processing | Transcribing audio |
| error | Error occurred |

---

### POST /reset

Reset application to idle state.

**Response:**
```json
{
  "status": "ok"
}
```

---

## Recording Access Endpoints

### GET /recordings

List all completed recordings.

**Response:**
```json
{
  "recordings": [
    {
      "id": "20240105_143022_TICKET-123",
      "ticket_id": "TICKET-123",
      "created_at": "2024-01-05T14:30:22",
      "duration_seconds": 125.5,
      "output_folder": "/path/to/output/20240105_143022_TICKET-123",
      "bundle_path": "/path/to/output/20240105_143022_TICKET-123/recording.vtb"
    }
  ]
}
```

---

### GET /recordings/search

Search and filter recordings with pagination.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | string | null | Full-text search in ID, ticket, transcript |
| date_from | string | null | Filter from date (YYYY-MM-DD) |
| date_to | string | null | Filter to date (YYYY-MM-DD) |
| ticket_prefix | string | null | Filter by ticket ID prefix |
| sort | string | "newest" | Sort order: newest, oldest, duration |
| page | int | 1 | Page number (1-indexed) |
| page_size | int | 20 | Results per page (max 100) |

**Example:**
```
GET /recordings/search?query=call&date_from=2024-01-01&sort=duration&page=1&page_size=10
```

**Response:**
```json
{
  "recordings": [...],
  "total": 45,
  "page": 1,
  "page_size": 10,
  "total_pages": 5
}
```

---

### GET /recordings/{recording_id}/download

Download the VTB bundle for a recording.

**Response:** Binary file download (`application/x-vtb`)

---

### GET /recordings/{recording_id}/transcript

Get transcript text and metadata.

**Response:**
```json
{
  "id": "20240105_143022_TICKET-123",
  "text": "Hello, thank you for calling support...",
  "srt": "1\n00:00:00,000 --> 00:00:02,500\nHello, thank you for calling support...",
  "duration_seconds": 125.5,
  "ticket_id": "TICKET-123",
  "word_count": 523
}
```

---

### PUT /recordings/{recording_id}/transcript

Update transcript text (for corrections).

**Request Body:**
```json
{
  "text": "Hello, thank you for calling support..."
}
```

**Response:**
```json
{
  "status": "ok",
  "recording_id": "20240105_143022_TICKET-123",
  "word_count": 523
}
```

---

### GET /recordings/{recording_id}/export/{format}

Export transcript in specified format.

**Formats:**
| Format | Content-Type | Description |
|--------|--------------|-------------|
| json | application/json | Structured JSON with metadata |
| vtt | text/vtt | WebVTT subtitle format |
| csv | text/csv | Tabular segment data |
| pdf | application/pdf | Formatted PDF document |
| docx | application/vnd.openxmlformats... | Word document |

**Example:**
```
GET /recordings/20240105_143022/export/pdf
```

**Response:** Binary file download

---

### GET /recordings/{recording_id}/export-formats

Get available export formats for a recording.

**Response:**
```json
{
  "recording_id": "20240105_143022",
  "formats": {
    "json": {"available": true, "description": "Structured JSON with metadata"},
    "vtt": {"available": true, "description": "WebVTT subtitle format"},
    "csv": {"available": true, "description": "Tabular segment data"},
    "pdf": {"available": true, "description": "PDF document"},
    "docx": {"available": true, "description": "Word document"}
  }
}
```

---

### POST /recordings/{recording_id}/open-folder

Open the output folder in system file manager.

**Response:**
```json
{
  "status": "ok",
  "folder": "/path/to/output/20240105_143022"
}
```

---

## File Upload Endpoints

### POST /recordings/upload

Upload a single audio file for transcription.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file | Yes | Audio file (WAV, MP3, OGG, M4A, FLAC) |
| ticket_id | string | No | Optional ticket ID |

**Limits:**
- Max file size: 500 MB
- Supported formats: WAV, MP3, OGG, OPUS, M4A, FLAC, WebM

**Example:**
```bash
curl -X POST http://localhost:8765/api/recordings/upload \
  -F "file=@recording.wav" \
  -F "ticket_id=TICKET-456"
```

**Response:**
```json
{
  "recording_id": "upload_20240105_143022_abc123",
  "status": "processing",
  "message": "Processing recording.wav"
}
```

---

### POST /recordings/batch-upload

Upload multiple audio files for batch transcription.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| files | file[] | Yes | Multiple audio files (max 20) |
| ticket_prefix | string | No | Prefix for ticket IDs (becomes PREFIX-1, PREFIX-2...) |

**Example:**
```bash
curl -X POST http://localhost:8765/api/recordings/batch-upload \
  -F "files=@call1.wav" \
  -F "files=@call2.wav" \
  -F "ticket_prefix=BATCH"
```

**Response:**
```json
{
  "status": "queued",
  "jobs_queued": 2,
  "job_ids": ["batch_20240105_143022_abc123", "batch_20240105_143022_def456"]
}
```

---

## Queue Management Endpoints

### GET /queue/status

Get current job queue status.

**Response:**
```json
{
  "queued": [
    {
      "job_id": "batch_20240105_143022_abc123",
      "original_filename": "call1.wav",
      "ticket_id": "BATCH-1",
      "status": "queued",
      "priority": 0,
      "progress": 0,
      "error_message": null,
      "created_at": 1704412800.0,
      "started_at": null,
      "completed_at": null
    }
  ],
  "processing": {
    "job_id": "batch_20240105_143022_def456",
    "original_filename": "call2.wav",
    "status": "processing",
    "progress": 45
  },
  "completed": [...],
  "failed": [...],
  "counts": {
    "queued": 1,
    "processing": 1,
    "completed": 5,
    "failed": 0
  }
}
```

---

### DELETE /queue/jobs/{job_id}

Cancel a queued job (cannot cancel jobs already processing).

**Response:**
```json
{
  "status": "cancelled",
  "job_id": "batch_20240105_143022_abc123"
}
```

**Errors:**
- 400: Cannot cancel job that is currently processing
- 404: Job not found in queue

---

### POST /queue/clear-history

Clear completed and failed job history.

**Response:**
```json
{
  "status": "ok",
  "message": "Queue history cleared"
}
```

---

### POST /queue/scan-folder

Scan a folder for audio files (preview before import).

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| folder_path | string | Yes | Path to folder |
| recursive | boolean | No | Include subdirectories (default: false) |

**Response:**
```json
{
  "total_files": 15,
  "total_size_mb": 234.5,
  "extensions": {
    ".wav": {"count": 10, "size_mb": 200.0},
    ".mp3": {"count": 5, "size_mb": 34.5}
  },
  "oldest_file": {"name": "call_001.wav", "date": "2024-01-01"},
  "newest_file": {"name": "call_015.wav", "date": "2024-01-05"}
}
```

---

### POST /queue/import-folder

Import all audio files from a folder into the queue.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| folder_path | string | Yes | Path to folder |
| recursive | boolean | No | Include subdirectories |
| ticket_prefix | string | No | Prefix for ticket IDs |

**Limits:** Max 100 files per import

**Response:**
```json
{
  "status": "queued",
  "jobs_queued": 15,
  "job_ids": ["import_...", "import_...", ...]
}
```

---

## Job Recovery Endpoints

### GET /jobs/incomplete

Get list of incomplete jobs for crash recovery.

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "20240105_143022",
      "audio_path": "/path/to/audio.wav",
      "status": "interrupted",
      "chunks_completed": 3,
      "total_chunks": 10,
      "progress_percent": 30.0,
      "device_name": "VB-Cable",
      "ticket_id": "TICKET-123",
      "created_at": 1704412800.0,
      "updated_at": 1704413100.0
    }
  ],
  "count": 1
}
```

---

### POST /jobs/{job_id}/resume

Resume an incomplete job from last checkpoint.

**Response:**
```json
{
  "status": "resume_started",
  "job_id": "20240105_143022",
  "from_chunk": 3,
  "total_chunks": 10
}
```

**Errors:**
- 400: Job already complete
- 400: Recording/processing in progress
- 404: Job not found
- 404: Audio file no longer exists

---

### DELETE /jobs/{job_id}

Discard an incomplete job and its checkpoint.

**Response:**
```json
{
  "status": "ok",
  "job_id": "20240105_143022",
  "deleted": true
}
```

---

### GET /jobs/history

Get recent job history.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Max results to return |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "20240105_143022",
      "status": "completed",
      "chunks_completed": 10,
      "total_chunks": 10,
      "ticket_id": "TICKET-123",
      "created_at": 1704412800.0,
      "updated_at": 1704413100.0
    }
  ],
  "count": 25
}
```

---

## Transcription Metrics Endpoints

### GET /transcriptions/summary

Get aggregate transcription statistics.

**Response:**
```json
{
  "total_transcriptions": 150,
  "total_audio_hours": 45.5,
  "total_processing_hours": 12.3,
  "avg_processing_speed": 3.7,
  "success_rate": 0.9867,
  "last_7_days": [
    {
      "date": "2024-01-05",
      "transcription_count": 15,
      "total_audio_seconds": 5400.0,
      "total_processing_seconds": 1800.0,
      "success_count": 15,
      "failure_count": 0,
      "success_rate": 1.0
    }
  ]
}
```

---

### GET /transcriptions/recent

Get recent transcription records.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Max results to return |

**Response:**
```json
{
  "transcriptions": [
    {
      "job_id": "20240105_143022",
      "audio_duration_seconds": 125.5,
      "transcription_duration_seconds": 45.2,
      "processing_speed": 2.78,
      "model_used": "ggml-medium.en.bin",
      "success": true,
      "error_message": null,
      "device_name": "VB-Cable",
      "ticket_id": "TICKET-123",
      "timestamp": 1704412800.0
    }
  ],
  "count": 50
}
```

---

### POST /transcriptions/export

Export transcription metrics to CSV.

**Response:** CSV file download

---

## Setup Endpoints

### GET /setup/status

Check first-run setup status.

**Response:**
```json
{
  "virtual_audio_detected": true,
  "recommended_device_available": true,
  "detected_devices": [
    {"name": "VB-Cable", "type": "virtual_cable"}
  ],
  "all_audio_devices": ["VB-Cable", "Microphone", "Speakers"],
  "recommended_action": null,
  "setup_complete": true,
  "setup_skipped": false,
  "recommendations": []
}
```

---

### POST /setup/complete

Mark first-run setup as complete.

**Request Body:**
```json
{
  "skipped": false
}
```

**Response:**
```json
{
  "status": "ok",
  "skipped": false
}
```

---

## Call Detection Endpoints (Windows Only)

### GET /call-detection/status

Get current call detection status.

**Response:**
```json
{
  "enabled": true,
  "state": "no_call",
  "current_call": null,
  "monitors": {
    "audio": true,
    "process": true
  },
  "config": {
    "target_processes": ["CiscoJabber.exe"],
    "call_start_confirm_seconds": 1.0,
    "call_end_confirm_seconds": 2.0
  },
  "platform_supported": true
}
```

**States:**
| State | Description |
|-------|-------------|
| no_call | No call detected |
| call_starting | Audio session active, confirming |
| call_active | Recording in progress |
| call_ending | Audio stopped, confirming |
| unsupported | Not on Windows |
| unavailable | Dependencies not installed |

---

### POST /call-detection/enable

Enable automatic call detection.

**Request Body (optional):**
```json
{
  "enabled": true,
  "target_processes": ["CiscoJabber.exe"],
  "call_start_confirm_seconds": 1.0,
  "call_end_confirm_seconds": 2.0
}
```

**Response:**
```json
{
  "status": "enabled",
  "config": {
    "target_processes": ["CiscoJabber.exe"],
    "call_start_confirm_seconds": 1.0,
    "call_end_confirm_seconds": 2.0
  }
}
```

**Errors:**
- 501: Call detection only available on Windows
- 501: Dependencies not installed (pycaw, wmi, pywin32)

---

### POST /call-detection/disable

Disable automatic call detection.

**Response:**
```json
{
  "status": "disabled"
}
```

---

### GET /call-detection/processes

List running target processes.

**Response:**
```json
{
  "processes": [
    {"name": "ciscojabber.exe", "pids": [1234, 5678]}
  ]
}
```

---

### GET /call-detection/audio-sessions

List all Windows audio sessions (debug endpoint, requires `debug_endpoints_enabled: true`).

**Response:**
```json
{
  "sessions": [
    {
      "process_name": "CiscoJabber.exe",
      "process_id": 1234,
      "state": "ACTIVE",
      "session_id": "session_1234"
    }
  ]
}
```

---

## Debug Endpoints

**Note:** All debug endpoints require `security.debug_endpoints_enabled: true` in configuration. They return 404 when disabled.

### GET /debug/state

Get full internal application state.

**Response:**
```json
{
  "request_id": "abc123",
  "current_state": "idle",
  "current_session": null,
  "completed_recordings_count": 14,
  "circuit_breakers": {...},
  "metrics_summary": {...}
}
```

---

### GET /debug/cache

Get transcription cache statistics.

**Response:**
```json
{
  "request_id": "abc123",
  "cache": {
    "hits": 5,
    "misses": 10,
    "entries": 10,
    "max_entries": 100,
    "ttl_seconds": 3600
  }
}
```

---

### POST /debug/cache/clear

Clear the transcription cache.

**Response:**
```json
{
  "status": "ok",
  "entries_cleared": 10
}
```

---

### GET /debug/capabilities

Get registered capability handlers.

**Response:**
```json
{
  "request_id": "abc123",
  "registry": {
    "total_capabilities": 15,
    "total_handlers": 20
  },
  "types": {
    "audio_format": {
      "handlers": ["wav", "mp3", "ogg"],
      "default": "wav"
    }
  }
}
```

---

### GET /debug/network

Get network isolation status.

**Response:**
```json
{
  "request_id": "abc123",
  "network": {
    "guard_enabled": true,
    "blocked_connections": 0
  },
  "cloud_dependencies": [],
  "external_api_calls": "none",
  "transcription_engine": "whisper.cpp (local)",
  "offline_verified": true
}
```

---

### GET /debug/paths

Get installation paths and deployment mode.

**Response:**
```json
{
  "request_id": "abc123",
  "mode": "development",
  "base_dir": "/path/to/callwhisper-web",
  "data_dir": "/path/to/callwhisper-web",
  "static_dir": "/path/to/callwhisper-web/static",
  "output_dir": "/path/to/callwhisper-web/output",
  "models_dir": "/path/to/callwhisper-web/models",
  "ffmpeg_path": "/path/to/ffmpeg",
  "whisper_path": "/path/to/whisper-cli"
}
```

---

### POST /debug/reset-metrics

Reset all collected metrics.

**Response:**
```json
{
  "status": "ok",
  "message": "Metrics reset"
}
```

---

### POST /debug/reset-circuits

Reset all circuit breakers.

**Response:**
```json
{
  "status": "ok",
  "message": "Circuit breakers reset"
}
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8765/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('State update:', data);
};
```

### Message Types

**State Update:**
```json
{
  "type": "state_update",
  "state": "recording",
  "recording_id": "20240105_143022",
  "elapsed_seconds": 45,
  "elapsed_formatted": "00:45"
}
```

**Progress Update:**
```json
{
  "type": "progress",
  "percent": 45,
  "stage": "Transcribing audio..."
}
```

**Partial Transcript:**
```json
{
  "type": "partial_transcript",
  "text": "Hello, thank you for calling...",
  "is_final": false
}
```

**Recording Complete:**
```json
{
  "type": "recording_complete",
  "recording": {
    "id": "20240105_143022",
    "ticket_id": "TICKET-123",
    "duration_seconds": 125.5,
    "transcript_preview": "Hello, thank you for calling..."
  }
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Transcription failed: timeout"
}
```

---

## Rate Limiting

| Setting | Default | Description |
|---------|---------|-------------|
| Requests per minute | 60 | Max requests per IP per minute |
| Burst size | 10 | Max burst before limiting |
| Excluded paths | /health, /health/ready | Not rate limited |

When rate limited, the API returns:

```
HTTP 503 Service Unavailable
Retry-After: 60

{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

---

## OpenAPI/Swagger

Interactive API documentation is available at:

- **Swagger UI:** http://localhost:8765/docs
- **ReDoc:** http://localhost:8765/redoc
- **OpenAPI JSON:** http://localhost:8765/openapi.json
