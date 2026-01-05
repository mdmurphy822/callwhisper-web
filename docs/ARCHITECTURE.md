# CallWhisper Architecture

Comprehensive system architecture documentation for CallWhisper v1.0.0.

---

## Table of Contents

1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Design Principles](#design-principles)
4. [System Architecture](#system-architecture)
5. [Layer Responsibilities](#layer-responsibilities)
6. [Audio Processing Pipeline](#audio-processing-pipeline)
7. [State Machine](#state-machine)
8. [Design Patterns](#design-patterns)
9. [Threading Model](#threading-model)
10. [Network Isolation](#network-isolation)
11. [Error Handling](#error-handling)
12. [Persistence and Recovery](#persistence-and-recovery)
13. [WebSocket Protocol](#websocket-protocol)
14. [Security Architecture](#security-architecture)
15. [Performance Considerations](#performance-considerations)
16. [File Locations](#file-locations)

---

## Overview

CallWhisper is a local-first voice transcription application designed for privacy-focused call recording and transcription. All processing happens locally on the user's machine - no audio data is ever sent to external servers.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           CallWhisper v1.0.0                               │
│                                                                            │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│   │  Web Browser │◄──►│  FastAPI     │◄──►│  whisper.cpp             │   │
│   │  (Frontend)  │    │  (Backend)   │    │  (Local Transcription)   │   │
│   └──────────────┘    └──────────────┘    └──────────────────────────┘   │
│          │                   │                        │                   │
│          │ HTTP/WS          │ subprocess             │ local model       │
│          ▼                   ▼                        ▼                   │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    NO EXTERNAL CONNECTIONS                        │   │
│   │                 Network Guard Active - Offline Only               │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend Framework** | FastAPI (Python 3.11+) | Async REST API & WebSocket |
| **Frontend** | Vanilla JavaScript | Browser UI (no framework) |
| **Transcription** | whisper.cpp | Local inference using Whisper models |
| **Real-time Communication** | WebSocket | Live UI updates |
| **Audio Processing** | FFmpeg | Capture, normalization, encoding |
| **Configuration** | JSON | config/config.json settings |
| **Persistence** | JSON files | Checkpoints, job store, event store |

---

## Design Principles

1. **Privacy First**
   - All processing happens locally
   - No cloud APIs required
   - Network guard blocks all external connections

2. **Crash Recovery**
   - Checkpointing after each stage
   - Automatic recovery of interrupted sessions
   - Resumable transcription from last completed chunk

3. **Fault Tolerance**
   - Bulkhead pattern for resource isolation
   - Circuit breakers for graceful degradation
   - Timeout cascades prevent hanging operations

4. **Accessibility**
   - WCAG 2.2 AA compliant UI
   - Full keyboard navigation
   - Screen reader support

5. **Offline Capable**
   - Works without internet after initial setup
   - All dependencies bundled

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend Layer                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  app.js  │  │  ui.js   │  │websocket │  │ keyboard │  │recorder  │      │
│  │ (state)  │  │ (render) │  │  .js     │  │   .js    │  │   .js    │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┘  └──────────┘      │
│       │             │             │                                          │
└───────┼─────────────┼─────────────┼──────────────────────────────────────────┘
        │ HTTP        │ DOM         │ WebSocket
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        routes.py (47 endpoints)                       │   │
│  │  /api/health  /api/devices  /api/recording  /api/queue  /api/debug  │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         websocket.py                                  │   │
│  │              Real-time state updates & progress                       │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────────────┐
│                            Service Layer (18 modules)                        │
│                                  │                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Audio Services │  │ Transcription  │  │ Output Services│                 │
│  │  recorder.py   │  │  transcriber   │  │  bundler.py    │                 │
│  │  normalizer.py │  │  audio_chunker │  │  exporter.py   │                 │
│  │  audio_detect. │  │  srt_merger.py │  │                │                 │
│  └────────┬───────┘  └───────┬────────┘  └───────┬────────┘                 │
│           │                  │                    │                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Device Services│  │ Orchestration  │  │ Call Detection │                 │
│  │  device_enum   │  │  orchestrator  │  │ call_detector  │                 │
│  │  device_guard  │  │  job_queue.py  │  │ win_audio_mon  │                 │
│  │                │  │  folder_scan   │  │ process_mon    │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────────────┐
│                            Core Layer (21 modules)                           │
│                                  │                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Infrastructure                                                        │   │
│  │  config.py      state.py       state_machine.py    connection_mgr    │   │
│  │  logging_conf   exceptions.py  job_store.py        metrics.py        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Resilience Patterns                                                   │   │
│  │  bulkhead.py    rate_limiter    degradation.py     timeout_cascade   │   │
│  │  cache.py       idempotency     resource_manager   network_guard     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ Observability                                                         │   │
│  │  tracing.py     health.py      event_store.py      persistence.py    │   │
│  │  capability_registry.py                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼───────────────────────────────────────────┐
│                         External Processes                                   │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────────────────────┐  │
│  │    FFmpeg     │  │   whisper.cpp   │  │        File System            │  │
│  │   (capture)   │  │  (transcribe)   │  │  (recordings, bundles, logs)  │  │
│  └───────────────┘  └─────────────────┘  └───────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Responsibilities

### API Layer (`src/callwhisper/api/`)

| Module | Responsibility |
|--------|----------------|
| `routes.py` | REST API endpoints (47 total) |
| `websocket.py` | WebSocket connection management |

**Key responsibilities:**
- Request validation and serialization
- Route handling and response formatting
- Authentication (localhost-only)
- Rate limit enforcement

### Core Layer (`src/callwhisper/core/`)

| Module | Pattern | Responsibility |
|--------|---------|----------------|
| `config.py` | Singleton | Configuration management with Pydantic |
| `state.py` | Global State | Application-wide state container |
| `state_machine.py` | State Machine | Recording workflow state management |
| `bulkhead.py` | Bulkhead | Isolated thread pools |
| `rate_limiter.py` | Token Bucket | Request rate limiting |
| `cache.py` | LRU Cache | Transcription result caching |
| `network_guard.py` | Proxy | Network isolation enforcement |
| `persistence.py` | Checkpoint | Crash recovery checkpoints |
| `job_store.py` | Repository | Job persistence |
| `metrics.py` | Observer | Prometheus-style metrics |
| `health.py` | Health Check | System health monitoring |
| `tracing.py` | Decorator | Distributed tracing |
| `event_store.py` | Event Sourcing | Audit trail events |
| `degradation.py` | Circuit Breaker | Graceful degradation |
| `timeout_cascade.py` | Timeout | Cascading timeout management |
| `exceptions.py` | - | Custom exception hierarchy |
| `logging_config.py` | - | Structured logging setup |
| `connection_manager.py` | - | WebSocket connection pool |
| `idempotency.py` | Idempotency Key | Duplicate request prevention |
| `resource_manager.py` | Resource Pool | External resource management |
| `capability_registry.py` | Feature Flags | Runtime feature detection |

### Services Layer (`src/callwhisper/services/`)

| Module | Responsibility |
|--------|----------------|
| `recorder.py` | FFmpeg audio capture subprocess |
| `normalizer.py` | Audio resampling (16kHz mono) |
| `transcriber.py` | whisper.cpp subprocess orchestration |
| `audio_chunker.py` | Audio segmentation for parallel processing |
| `audio_detector.py` | Voice activity detection |
| `audio_pipeline.py` | Full audio processing pipeline |
| `bundler.py` | VTB bundle creation (ZIP format) |
| `exporter.py` | Multi-format export (TXT, SRT, PDF, DOCX) |
| `device_enum.py` | Platform-specific device enumeration |
| `device_guard.py` | Microphone protection validation |
| `process_orchestrator.py` | End-to-end workflow orchestration |
| `job_queue.py` | Batch processing queue |
| `folder_scanner.py` | Folder import scanning |
| `srt_merger.py` | SRT subtitle merging |
| `call_detector.py` | Automatic call detection (Windows) |
| `windows_audio_monitor.py` | WASAPI audio session monitoring |
| `process_monitor.py` | Process existence monitoring |

### Utils Layer (`src/callwhisper/utils/`)

| Module | Responsibility |
|--------|----------------|
| `paths.py` | Path resolution utilities |
| `validation.py` | Input validation helpers |
| `time_utils.py` | Time formatting utilities |

---

## Audio Processing Pipeline

### Recording to Transcription Flow

```
                              RECORDING PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  User clicks   ─►  Device Guard   ─►  FFmpeg Start   ─►  Raw Audio         │
│  "Start"           validates          subprocess         capture           │
│                    device                                                   │
│                       │                                                     │
│                       ▼                                                     │
│               Blocklist check ─► Allowlist check ─► Default: BLOCK         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                             PROCESSING PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  User clicks   ─►  FFmpeg Stop   ─►  audio_raw.wav saved                   │
│  "Stop"            subprocess                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            NORMALIZATION PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  normalizer.py   ─►  FFmpeg converts   ─►  audio_16k.wav                   │
│  invoked             to 16kHz mono         (whisper-ready)                 │
│                      PCM WAV                                                │
│                                                                             │
│  Input: audio_raw.wav (44.1kHz stereo)                                     │
│  Output: audio_16k.wav (16kHz mono)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                            CHUNKING PHASE (if enabled)
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  audio_chunker.py   ─►  Split audio   ─►  chunks/chunk_001.wav            │
│  invoked                by duration       chunks/chunk_002.wav             │
│                         (30s default)     chunks/chunk_003.wav             │
│                                           ...                               │
│                                                                             │
│  Purpose:                                                                   │
│  - Parallel transcription (max_concurrent_transcriptions)                  │
│  - Progress reporting (per-chunk completion)                               │
│  - Crash recovery (resume from last chunk)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           TRANSCRIPTION PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  transcriber.py   ─►  whisper-cli    ─►  JSON output   ─►  SRT segments   │
│  orchestrates         subprocess         per chunk         merged          │
│                                                                             │
│  Whisper Parameters:                                                        │
│  - model: ggml-medium.en.bin (configurable)                                │
│  - language: en (configurable)                                             │
│  - beam_size: 5 (configurable)                                             │
│  - best_of: 5 (configurable)                                               │
│                                                                             │
│  Output:                                                                    │
│  - transcript.txt (plain text)                                             │
│  - transcript.srt (subtitles with timestamps)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                             BUNDLING PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  bundler.py   ─►  Create VTB   ─►  ZIP structure   ─►  {id}.vtb           │
│  invoked          bundle                                                    │
│                                                                             │
│  VTB Bundle Structure:                                                      │
│  recording.vtb/                                                             │
│  ├── mimetype                  # "application/x-vtb"                       │
│  ├── META-INF/                                                             │
│  │   ├── manifest.json         # Recording metadata                        │
│  │   └── hashes.json           # SHA-256 integrity hashes                  │
│  ├── audio/                                                                │
│  │   └── recording.opus        # Compressed audio (configurable)           │
│  └── transcript/                                                           │
│      ├── transcript.txt        # Plain text                                │
│      └── transcript.srt        # SRT subtitles                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              COMPLETION
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  WebSocket   ─►  "recording_complete"   ─►  UI updates   ─►  Download     │
│  broadcast       message sent                                available      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## State Machine

### Recording Workflow States

The application uses a strict state machine to prevent invalid operations:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Recording State Machine                              │
│                                                                              │
│                                ┌─────────┐                                  │
│                   ┌───────────►│  IDLE   │◄──────────────────┐              │
│                   │            └────┬────┘                   │              │
│                   │                 │ start                  │ complete     │
│                   │                 ▼                        │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │ INITIALIZING │                  │              │
│                   │        └───────┬──────┘                  │              │
│                   │                │ initialized             │              │
│        reset      │                ▼                         │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │  RECORDING   │                  │              │
│                   │        └───────┬──────┘                  │              │
│                   │                │ stop                    │              │
│                   │                ▼                         │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │   STOPPING   │                  │              │
│                   │        └───────┬──────┘                  │              │
│                   │                │ stopped                 │              │
│                   │                ▼                         │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │ NORMALIZING  │                  │              │
│                   │        └───────┬──────┘                  │              │
│                   │                │ normalized              │              │
│                   │                ▼                         │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │ TRANSCRIBING │                  │              │
│                   │        └───────┬──────┘                  │              │
│                   │                │ transcribed             │              │
│                   │                ▼                         │              │
│                   │        ┌──────────────┐                  │              │
│                   │        │   BUNDLING   │──────────────────┘              │
│                   │        └───────┬──────┘                                 │
│                   │                │ error (any state)                      │
│                   │                ▼                                        │
│                   │        ┌──────────────┐                                 │
│                   └────────│    ERROR     │                                 │
│                            └───────┬──────┘                                 │
│                                    │ recover                                │
│                                    ▼                                        │
│                            ┌──────────────┐                                 │
│                            │  RECOVERING  │────► resume point               │
│                            └──────────────┘                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### State Transition Matrix

| From State | Valid Transitions |
|------------|-------------------|
| IDLE | INITIALIZING, RECOVERING |
| INITIALIZING | RECORDING, ERROR |
| RECORDING | STOPPING, ERROR |
| STOPPING | NORMALIZING, ERROR |
| NORMALIZING | TRANSCRIBING, ERROR |
| TRANSCRIBING | BUNDLING, COMPLETED, ERROR |
| BUNDLING | COMPLETED, ERROR |
| COMPLETED | IDLE |
| ERROR | IDLE, RECOVERING |
| RECOVERING | IDLE, NORMALIZING, TRANSCRIBING, BUNDLING, ERROR |

---

## Design Patterns

CallWhisper implements 10+ enterprise design patterns:

### 1. Bulkhead Pattern (`core/bulkhead.py`)

Isolates failures by using separate thread pools for different operations.

```python
# Isolated pools prevent cascade failures
pools = {
    "audio": ThreadPoolExecutor(max_workers=2),      # Audio operations
    "transcription": ThreadPoolExecutor(max_workers=2),  # Whisper tasks
    "io": ThreadPoolExecutor(max_workers=4),         # File I/O
    "general": ThreadPoolExecutor(max_workers=4),    # Other tasks
}
```

**Benefits:**
- Transcription failures don't affect audio recording
- I/O bottlenecks don't block processing
- Queue depth limits prevent memory exhaustion

### 2. State Machine Pattern (`core/state_machine.py`)

Thread-safe state transitions with validation and history.

```python
# Invalid transitions raise exception
await state_machine.transition(RecordingState.RECORDING)  # Only valid from INITIALIZING

# Thread-safe with Lock()
with self._lock:
    if not self.can_transition(to_state):
        raise InvalidStateTransitionError(...)
```

### 3. Checkpoint Pattern (`core/persistence.py`)

Durable execution with checkpoints after each stage.

```python
# Save checkpoint after normalization
checkpoint_manager.update_checkpoint(
    session_id,
    CheckpointStage.NORMALIZING,
    audio_file=audio_path
)

# On startup, find incomplete sessions
incomplete = checkpoint_manager.get_incomplete_sessions()
for checkpoint in incomplete:
    resume_stage = checkpoint_manager.get_resumable_stage(checkpoint)
```

### 4. Circuit Breaker Pattern (`core/degradation.py`)

Prevents repeated failures from overwhelming the system.

```
        CLOSED                OPEN                 HALF-OPEN
    (normal operation)   (requests blocked)    (testing recovery)

         ┌──────┐            ┌──────┐             ┌──────┐
     ───►│      │  failure   │      │   timeout   │      │
         │CLOSED│───────────►│ OPEN │────────────►│ HALF │
         │      │  threshold │      │             │ OPEN │
         └──────┘            └──────┘             └──────┘
             ▲                                        │
             │              success                   │
             └────────────────────────────────────────┘
```

### 5. Proxy Pattern (`core/network_guard.py`)

Socket-level interception to enforce offline mode.

```python
# Override socket functions
socket.create_connection = _guarded_create_connection
socket.getaddrinfo = _guarded_getaddrinfo

# Only localhost allowed
ALLOWED_HOSTS = frozenset(['127.0.0.1', 'localhost', '::1', '0.0.0.0'])
```

### 6. Observer Pattern (`api/websocket.py`)

Real-time notifications via WebSocket subscriptions.

```python
# Broadcast state changes to all clients
await connection_manager.broadcast({
    "type": "state_change",
    "state": "transcribing"
})
```

### 7. Token Bucket / Sliding Window (`core/rate_limiter.py`)

Request rate limiting with burst allowance.

```python
# Sliding window rate limiter
config = RateLimitConfig(
    requests_per_minute=60,
    burst_size=10,
    excluded_paths=["/api/health"]
)
```

### 8. Repository Pattern (`core/job_store.py`)

Abstracted persistence for job management.

```python
# Job store interface
job_store.create_job(job_id, metadata)
job_store.update_status(job_id, "completed")
job_store.get_job(job_id)
```

### 9. Event Sourcing (`core/event_store.py`)

Immutable audit trail of all events.

```python
# Record events for auditing
event_store.append(EventRecord(
    event_type="recording_started",
    session_id=session_id,
    timestamp=datetime.now().isoformat(),
    metadata={"device": device_name}
))
```

### 10. Singleton Pattern (`core/config.py`, `core/state.py`)

Global instances with lazy initialization.

```python
# Thread-safe singleton
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

### 11. Decorator Pattern (`core/tracing.py`)

Cross-cutting concerns via middleware.

```python
# Tracing middleware adds request IDs
app.add_middleware(TracingMiddleware)

# All requests get X-Request-ID header
response.headers["X-Request-ID"] = request_id
```

---

## Threading Model

### Thread Pool Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BulkheadExecutor                                     │
│                                                                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │   Audio Pool      │  │ Transcription Pool│  │    I/O Pool       │       │
│  │   max_workers: 2  │  │   max_workers: 2  │  │   max_workers: 4  │       │
│  │   queue_size: 100 │  │   queue_size: 100 │  │   queue_size: 100 │       │
│  │                   │  │                   │  │                   │       │
│  │  ┌─────┐ ┌─────┐ │  │  ┌─────┐ ┌─────┐ │  │  ┌───┐┌───┐┌───┐┌───┐     │
│  │  │  T1 │ │  T2 │ │  │  │  T1 │ │  T2 │ │  │  │T1 ││T2 ││T3 ││T4 │     │
│  │  └─────┘ └─────┘ │  │  └─────┘ └─────┘ │  │  └───┘└───┘└───┘└───┘     │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                             │
│  ┌───────────────────┐                                                     │
│  │   General Pool    │  Metrics tracked per pool:                          │
│  │   max_workers: 4  │  - active_tasks                                     │
│  │   queue_size: 100 │  - completed_tasks                                  │
│  │                   │  - failed_tasks                                     │
│  │  ┌───┐┌───┐┌───┐┌───┐  - rejected_tasks                                │
│  │  │T1 ││T2 ││T3 ││T4 │  - avg_execution_time_ms                         │
│  │  └───┘└───┘└───┘└───┘  - utilization                                   │
│  └───────────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Async/Sync Boundary

```python
# FastAPI endpoint (async)
@router.post("/recording/start")
async def start_recording(device: str):
    # Run blocking FFmpeg operation in audio pool
    result = await executor.run_audio_task(
        recorder.start,  # Sync function
        device
    )
    return result

# Automatic thread pool execution
async def run_in_pool(self, pool_type, func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        self._pools[pool_type],
        lambda: func(*args, **kwargs)
    )
```

---

## Network Isolation

### Network Guard Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Network Guard                                        │
│                                                                             │
│   Application Code                                                          │
│        │                                                                    │
│        │ socket.create_connection("api.example.com", 443)                  │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │              _guarded_create_connection()                    │          │
│   │                                                             │          │
│   │   if host not in ALLOWED_HOSTS:                             │          │
│   │       raise ConnectionRefusedError(                         │          │
│   │           "External connections blocked by network guard"    │          │
│   │       )                                                     │          │
│   │                                                             │          │
│   │   ALLOWED_HOSTS = {                                         │          │
│   │       '127.0.0.1',                                          │          │
│   │       'localhost',                                          │          │
│   │       '::1',                                                │          │
│   │       '0.0.0.0'                                             │          │
│   │   }                                                         │          │
│   └─────────────────────────────────────────────────────────────┘          │
│        │                                                                    │
│        ▼                                                                    │
│   ConnectionRefusedError: "External connections blocked..."                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Blocked Operations

| Operation | Result |
|-----------|--------|
| `socket.create_connection("8.8.8.8", 53)` | ConnectionRefusedError |
| `socket.getaddrinfo("google.com", 443)` | socket.gaierror (EAI_NONAME) |
| `requests.get("https://api.openai.com")` | Blocked at socket level |
| `socket.create_connection("127.0.0.1", 8765)` | Allowed |
| `socket.create_connection("localhost", 8765)` | Allowed |

---

## Error Handling

### Exception Hierarchy

```
CallWhisperError (base)
├── InvalidStateTransitionError
│   └── Recording/transcription state violations
├── DeviceError
│   ├── DeviceNotFoundError
│   ├── DeviceBlockedError
│   └── DeviceEnumerationError
├── TranscriptionError
│   ├── ModelNotFoundError
│   ├── TranscriptionTimeoutError
│   └── ChunkProcessingError
├── BundleError
│   ├── BundleCreationError
│   └── BundleExtractionError
├── ConfigurationError
│   └── Invalid configuration
└── RecoveryError
    └── Checkpoint recovery failures
```

### Error Response Format

```json
{
    "error": "device_blocked",
    "message": "Device 'Microphone' is blocked by device guard",
    "details": {
        "device_name": "Microphone",
        "reason": "blocklist_match",
        "suggestion": "Use an allowed device like VB-Cable or Stereo Mix"
    },
    "request_id": "req_abc123",
    "recoverable": true
}
```

---

## Persistence and Recovery

### Checkpoint Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Checkpoint Recovery Flow                              │
│                                                                             │
│   Normal Operation:                                                         │
│                                                                             │
│   START ──► checkpoint(STARTED) ──► RECORDING ──► checkpoint(STOPPED)      │
│                                                          │                  │
│              ──► NORMALIZING ──► checkpoint(NORMALIZING) │                  │
│                                                          │                  │
│              ──► TRANSCRIBING ──► checkpoint(TRANSCRIBING)                  │
│                                                          │                  │
│              ──► BUNDLING ──► checkpoint(COMPLETED) ──► clear_checkpoint    │
│                                                                             │
│   On Crash:                                                                 │
│                                                                             │
│   Application restarts                                                      │
│        │                                                                    │
│        ▼                                                                    │
│   checkpoint_manager.get_incomplete_sessions()                              │
│        │                                                                    │
│        ▼                                                                    │
│   Found: session_123 at stage TRANSCRIBING                                  │
│        │                                                                    │
│        ▼                                                                    │
│   Resume from TRANSCRIBING with saved audio_file path                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Checkpoint File Structure

```json
{
    "session_id": "20241229_120000_ABC123",
    "stage": "transcribing",
    "timestamp": "2024-12-29T12:05:23.456789",
    "device_name": "VB-Cable",
    "ticket_id": "TICKET-123",
    "output_folder": "/output/20241229_120000_ABC123",
    "audio_file": "/output/.../audio_raw.wav",
    "normalized_file": "/output/.../audio_16k.wav",
    "transcript_file": null,
    "bundle_file": null,
    "error_message": null,
    "metadata": {
        "chunks_completed": 3,
        "total_chunks": 5
    }
}
```

---

## WebSocket Protocol

### Message Types

#### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `state_change` | `{state: string}` | State machine transition |
| `timer` | `{formatted: string, seconds: number}` | Recording timer update |
| `processing_progress` | `{percent: number, stage: string}` | Progress update |
| `partial_transcript` | `{text: string, is_final: boolean}` | Live transcript preview |
| `recording_complete` | `{recording_id: string, preview: string}` | Recording finished |
| `error` | `{message: string, recoverable: boolean}` | Error notification |
| `queue_status` | `{counts: object}` | Queue statistics |
| `call_detected` | `{process: string, started: boolean}` | Call detection event |

#### Client → Server

| Type | Payload | Description |
|------|---------|-------------|
| `ping` | `{}` | Keep-alive |
| `subscribe` | `{topics: string[]}` | Topic subscription |

### Connection Lifecycle

```
Client                                          Server
   │                                               │
   │──────── WS handshake /ws ────────────────────►│
   │                                               │
   │◄─────── connection_established ───────────────│
   │         {type: "state_change", state: "idle"} │
   │                                               │
   │◄─────── state updates (as needed) ────────────│
   │                                               │
   │──────── {type: "ping"} ──────────────────────►│  every 30s
   │◄─────── {type: "pong"} ───────────────────────│
   │                                               │
   │◄─────── {type: "recording_complete"} ─────────│
   │                                               │
   │──────── connection close ────────────────────►│
   │                                               │
```

---

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Security Layers                                     │
│                                                                             │
│   Layer 1: Network Binding                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Server binds to 127.0.0.1 only (localhost)                         │  │
│   │  No external network access                                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│   Layer 2: Network Guard                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Socket-level blocking of all external connections                  │  │
│   │  DNS resolution blocked for non-localhost                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│   Layer 3: CORS                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Only allowed_origins can make requests                             │  │
│   │  Default: ["http://localhost:8765", "http://127.0.0.1:8765"]        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│   Layer 4: Rate Limiting                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Sliding window: 60 requests/minute per IP                          │  │
│   │  Burst allowance: 10 requests                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│   Layer 5: Device Guard                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Blocklist → Allowlist → Default BLOCK                              │  │
│   │  Prevents microphone recording                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│   Layer 6: Input Validation                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  Ticket ID validation: [A-Za-z0-9\-_]+                              │  │
│   │  File type validation: supported audio formats only                 │  │
│   │  Path sanitization: prevent directory traversal                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Considerations

### Bottleneck Analysis

| Operation | Typical Duration | Bottleneck | Mitigation |
|-----------|------------------|------------|------------|
| Audio capture | Real-time | FFmpeg subprocess | Separate thread pool |
| Normalization | 1-5 seconds | FFmpeg CPU | Timeout (120s) |
| Transcription | 0.5-3x audio duration | whisper.cpp CPU | Chunking, parallelism |
| Bundling | 1-10 seconds | Opus encoding | Configurable format |

### Memory Management

```python
# Chunked processing for large files
chunk_size_seconds = 30.0  # Process in 30-second chunks

# History limits
_transition_history = _transition_history[-100:]  # Keep last 100 state events

# Cache limits
cache_max_entries = 100  # LRU eviction
cache_ttl_seconds = 3600  # 1-hour expiry
```

### WebSocket Efficiency

```python
# Heartbeat interval
HEARTBEAT_INTERVAL = 30  # seconds

# Progress update throttling
MIN_PROGRESS_INTERVAL = 0.1  # 100ms between updates (max 10/sec)

# Connection pooling
connection_manager.active_connections  # All connected clients
```

---

## File Locations

### Default Paths

| Data Type | Default Path | Configurable |
|-----------|--------------|--------------|
| Configuration | `./config/config.json` | No |
| Recordings | `./output/` | Yes (output.directory) |
| Whisper Models | `./models/` | No |
| FFmpeg/whisper-cli | `./vendor/` | No |
| Checkpoints | `./checkpoints/` | No |
| Job Store | `./checkpoints/jobs.json` | No |
| Static Files | `./static/` | No |
| Logs | stdout | No (redirect to file) |

### Recording Output Structure

```
output/
└── 20241229_120000_ABC123/
    ├── audio_raw.wav         # Original recording
    ├── audio_16k.wav         # Normalized for whisper
    ├── transcript.txt        # Plain text transcript
    ├── transcript.srt        # SRT subtitles
    └── 20241229_120000_ABC123.vtb  # Bundle file
```

---

## Future Considerations

### Planned Enhancements

- ML/NLP post-processing (keyword extraction, NER)
- Full-text search indexing
- Speaker diarization
- Multi-language support with automatic detection

### Technical Improvements

- Consider SQLite for job persistence (vs JSON files)
- GPU acceleration for whisper inference
- Streaming transcription (real-time preview)
- Plugin architecture for custom exporters
