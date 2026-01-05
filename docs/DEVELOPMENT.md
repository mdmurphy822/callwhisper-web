# CallWhisper Development Guide

Complete guide for setting up a development environment and contributing to CallWhisper.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Running the Development Server](#running-the-development-server)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Adding New Features](#adding-new-features)
8. [Debugging](#debugging)
9. [Contributing](#contributing)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime environment |
| FFmpeg | 4.0+ | Audio capture and processing |
| Git | Any recent | Version control |

### Platform-Specific Requirements

**Windows:**
```powershell
# Python 3.11+ from python.org
# FFmpeg can be downloaded via download-vendor.ps1
# Or install via chocolatey:
choco install python311 ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv ffmpeg
```

**Linux (Fedora):**
```bash
sudo dnf install python3.11 ffmpeg
```

**macOS:**
```bash
brew install python@3.11 ffmpeg
```

### Whisper Model

Download a Whisper model to `models/`:

```bash
# Create models directory
mkdir -p models

# Download medium.en model (1.5GB - recommended for development)
wget -O models/ggml-medium.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin

# Or use smaller model for faster iteration (466MB)
wget -O models/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/callwhisper/callwhisper-web.git
cd callwhisper-web
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
# Install all dependencies (including dev dependencies)
pip install -r requirements.txt

# Or install with pyproject.toml
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check FFmpeg
ffmpeg -version

# Run tests to verify setup
pytest tests/ -v --tb=short
```

### 5. Configure Development Settings

Create `config/config.json` for development:

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

---

## Project Structure

```
callwhisper-web/
├── src/callwhisper/          # Main Python package (50 modules)
│   ├── __init__.py           # Package metadata, version
│   ├── __main__.py           # Entry point: python -m callwhisper
│   ├── main.py               # FastAPI application factory
│   │
│   ├── api/                  # REST & WebSocket layer
│   │   ├── routes.py         # 47 API endpoints
│   │   └── websocket.py      # WebSocket handler
│   │
│   ├── core/                 # Core infrastructure (21 modules)
│   │   ├── config.py         # Pydantic settings
│   │   ├── state.py          # Global application state
│   │   ├── state_machine.py  # Recording workflow FSM
│   │   ├── bulkhead.py       # Thread pool isolation
│   │   ├── network_guard.py  # Network isolation
│   │   ├── persistence.py    # Checkpoint management
│   │   ├── job_store.py      # Job persistence
│   │   ├── metrics.py        # Prometheus-style metrics
│   │   ├── health.py         # Health checks
│   │   ├── cache.py          # LRU cache
│   │   ├── rate_limiter.py   # Rate limiting
│   │   ├── tracing.py        # Request tracing
│   │   ├── exceptions.py     # Custom exceptions
│   │   ├── logging_config.py # Structured logging
│   │   └── ...               # Additional core modules
│   │
│   ├── services/             # Business logic (18 modules)
│   │   ├── recorder.py       # FFmpeg audio capture
│   │   ├── normalizer.py     # Audio resampling
│   │   ├── transcriber.py    # Whisper integration
│   │   ├── bundler.py        # VTB bundle creation
│   │   ├── exporter.py       # Multi-format export
│   │   ├── device_guard.py   # Microphone protection
│   │   ├── device_enum.py    # Device enumeration
│   │   ├── job_queue.py      # Batch processing
│   │   ├── process_orchestrator.py
│   │   ├── call_detector.py  # Auto call detection (Windows)
│   │   └── ...               # Additional services
│   │
│   └── utils/                # Utility functions
│       ├── paths.py          # Path resolution
│       ├── validation.py     # Input validation
│       └── time_utils.py     # Time formatting
│
├── static/                   # Web frontend
│   ├── index.html            # Main HTML file
│   ├── css/
│   │   └── styles.css        # Stylesheet (2,274 lines)
│   └── js/                   # JavaScript modules (5,600 lines)
│       ├── app.js            # Application state
│       ├── ui.js             # DOM manipulation
│       ├── websocket.js      # WebSocket client
│       ├── recorder.js       # Recording controls
│       ├── keyboard.js       # Keyboard shortcuts
│       ├── exporter.js       # Export functionality
│       └── ...               # Additional modules
│
├── tests/                    # Test suite (54 files)
│   ├── conftest.py           # Shared fixtures
│   ├── fixtures/             # Test data
│   ├── unit/                 # Unit tests
│   │   ├── core/             # Core module tests
│   │   └── services/         # Service tests
│   └── integration/          # Integration tests
│
├── vendor/                   # Third-party binaries
│   ├── ffmpeg.exe            # Windows FFmpeg
│   └── whisper-cli/          # Whisper executable
│
├── models/                   # Whisper ML models
│   └── ggml-medium.en.bin    # Default model
│
├── config/                   # Configuration
│   └── config.json           # Runtime config
│
├── output/                   # Recording output directory
├── docs/                     # Documentation
├── scripts/                  # Build & dev scripts
│
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
└── README.md                 # Project README
```

---

## Running the Development Server

### Option 1: Using Development Script

**Linux/macOS:**
```bash
./scripts/dev_run.sh
```

**Windows:**
```powershell
.\scripts\dev_run.ps1
```

### Option 2: Direct Python Invocation

```bash
# From project root with virtual environment activated
PYTHONPATH=src python -m callwhisper
```

### Option 3: Using Uvicorn Directly

```bash
# For hot-reload during development
PYTHONPATH=src uvicorn callwhisper.main:app --reload --host 127.0.0.1 --port 8765
```

### Accessing the Application

Once running, open your browser to:
- **Main UI:** http://localhost:8765
- **API Docs (Swagger):** http://localhost:8765/docs
- **API Docs (ReDoc):** http://localhost:8765/redoc

---

## Testing

### Test Structure

```
tests/
├── conftest.py               # Shared fixtures (328 lines)
├── fixtures/                 # Test data files
│   ├── sample.wav
│   └── sample.srt
├── unit/                     # Unit tests
│   ├── core/                 # Core module tests
│   │   ├── test_state_machine.py
│   │   ├── test_bulkhead.py
│   │   ├── test_persistence.py
│   │   ├── test_metrics.py
│   │   └── ...
│   └── services/             # Service tests
│       ├── test_recorder.py
│       ├── test_transcriber.py
│       ├── test_bundler.py
│       ├── test_exporter.py
│       └── ...
└── integration/              # Integration tests
    ├── test_api_health.py
    ├── test_api_recording.py
    ├── test_websocket.py
    └── ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src/callwhisper --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/core/test_state_machine.py

# Run specific test function
pytest tests/unit/core/test_state_machine.py::test_valid_transitions

# Run tests matching a pattern
pytest -k "test_bulkhead"

# Run with short traceback
pytest --tb=short

# Run in parallel (if pytest-xdist installed)
pytest -n auto
```

### Test Markers

```bash
# Run slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run only async tests
pytest -m asyncio
```

### Writing Tests

**Unit Test Example:**

```python
# tests/unit/core/test_example.py
import pytest
from callwhisper.core.state_machine import StateMachine, RecordingState


@pytest.mark.asyncio
async def test_valid_state_transition():
    """Test that valid state transitions work correctly."""
    sm = StateMachine()

    # Initial state should be IDLE
    assert sm.state == RecordingState.IDLE

    # Transition to INITIALIZING
    await sm.transition(RecordingState.INITIALIZING)
    assert sm.state == RecordingState.INITIALIZING


@pytest.mark.asyncio
async def test_invalid_transition_raises():
    """Test that invalid transitions raise exception."""
    from callwhisper.core.exceptions import InvalidStateTransitionError

    sm = StateMachine()

    # Cannot go directly from IDLE to RECORDING
    with pytest.raises(InvalidStateTransitionError):
        await sm.transition(RecordingState.RECORDING)
```

**Integration Test Example:**

```python
# tests/integration/test_api_health.py
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Test health endpoint returns OK."""
    response = await client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "offline"
```

### Test Fixtures

The `tests/conftest.py` provides common fixtures:

| Fixture | Purpose |
|---------|---------|
| `temp_dir` | Temporary directory for test outputs |
| `temp_checkpoint_dir` | Checkpoint directory |
| `temp_output_dir` | Recording output directory |
| `mock_settings` | Mock configuration settings |
| `sample_session` | Sample recording session |
| `mock_subprocess` | Mock for subprocess calls |
| `mock_subprocess_failure` | Mock failing subprocess |
| `app` | FastAPI application |
| `client` | Async HTTP client |
| `reset_app_state` | Reset application state |
| `sample_audio_file` | Generate test WAV file |
| `sample_vtb_bundle` | Generate test VTB bundle |

### Coverage Requirements

Aim for:
- **Unit tests:** 80%+ coverage
- **Integration tests:** All API endpoints covered
- **Critical paths:** 100% coverage

Generate coverage report:
```bash
pytest --cov=src/callwhisper --cov-report=html --cov-report=term-missing
# Open htmlcov/index.html in browser
```

---

## Code Style

### Formatting with Black

```bash
# Format all Python files
black src/ tests/

# Check formatting without changes
black --check src/ tests/

# Show diff of proposed changes
black --diff src/ tests/
```

### Linting with Ruff

```bash
# Run linter
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

### Type Checking with mypy

```bash
# Run type checker
mypy src/callwhisper/
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

### Style Guidelines

1. **Imports:** Group imports in order: stdlib, third-party, local
2. **Type Hints:** Use type hints for function signatures
3. **Docstrings:** Use Google-style docstrings
4. **Line Length:** 88 characters (Black default)
5. **Naming:** snake_case for functions/variables, PascalCase for classes

**Example:**

```python
"""
Module docstring describing the purpose.

Based on LibV2 patterns:
- Pattern 1
- Pattern 2
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from fastapi import APIRouter

from .exceptions import CallWhisperError
from .logging_config import get_logger


logger = get_logger(__name__)


@dataclass
class ExampleConfig:
    """Configuration for the example feature.

    Attributes:
        name: The feature name.
        enabled: Whether the feature is enabled.
    """
    name: str
    enabled: bool = True


def process_data(data: Dict[str, Any], config: Optional[ExampleConfig] = None) -> str:
    """Process input data according to configuration.

    Args:
        data: Input data dictionary.
        config: Optional configuration overrides.

    Returns:
        Processed data as string.

    Raises:
        CallWhisperError: If processing fails.
    """
    if config is None:
        config = ExampleConfig(name="default")

    logger.info("processing_data", name=config.name)

    # Implementation...
    return str(data)
```

---

## Adding New Features

### Step 1: Plan the Feature

1. Identify which layer(s) the feature touches:
   - **API Layer:** New endpoints needed?
   - **Service Layer:** New business logic?
   - **Core Layer:** New infrastructure patterns?

2. Review existing patterns in similar modules

### Step 2: Write Tests First (TDD)

```python
# tests/unit/services/test_new_feature.py
import pytest


@pytest.mark.asyncio
async def test_new_feature_basic():
    """Test basic functionality of new feature."""
    # Write test before implementation
    from callwhisper.services.new_feature import process

    result = await process("input")
    assert result == "expected_output"
```

### Step 3: Implement the Feature

**Service Module Template:**

```python
# src/callwhisper/services/new_feature.py
"""
New Feature Service

Brief description of what this service does.
"""

from typing import Optional
from ..core.logging_config import get_logger
from ..core.exceptions import CallWhisperError


logger = get_logger(__name__)


class NewFeatureError(CallWhisperError):
    """Raised when new feature fails."""
    pass


async def process(input_data: str) -> str:
    """Process input data.

    Args:
        input_data: Data to process.

    Returns:
        Processed result.

    Raises:
        NewFeatureError: If processing fails.
    """
    logger.info("new_feature_processing", input=input_data)

    try:
        # Implementation
        result = input_data.upper()
        logger.info("new_feature_complete", result=result)
        return result
    except Exception as e:
        logger.error("new_feature_failed", error=str(e))
        raise NewFeatureError(f"Processing failed: {e}")
```

### Step 4: Add API Endpoint (if needed)

```python
# Add to src/callwhisper/api/routes.py

@router.post("/new-feature", tags=["new-feature"])
async def new_feature_endpoint(data: NewFeatureRequest) -> NewFeatureResponse:
    """
    Process data with new feature.

    - **data**: Input data to process
    """
    from ..services.new_feature import process

    result = await process(data.input)
    return NewFeatureResponse(output=result)
```

### Step 5: Update Documentation

1. Add endpoint to `docs/API.md`
2. Update `docs/SERVICES.md` if new service
3. Add configuration to `docs/CONFIGURATION.md` if needed

---

## Debugging

### Debug Endpoints

Enable debug endpoints in config:

```json
{
  "security": {
    "debug_endpoints_enabled": true
  }
}
```

Available debug endpoints:
- `GET /api/debug/state` - Full application state
- `GET /api/debug/paths` - Path configuration
- `GET /api/debug/network` - Network guard status
- `GET /api/debug/bulkhead` - Thread pool metrics
- `GET /api/debug/checkpoints` - Checkpoint status
- `GET /api/debug/memory` - Memory usage

### Logging

CallWhisper uses structured logging with `structlog`:

```python
from callwhisper.core.logging_config import get_logger

logger = get_logger(__name__)

# Log with context
logger.info(
    "operation_started",
    session_id="abc123",
    device="VB-Cable"
)

# Log error with exception
try:
    risky_operation()
except Exception as e:
    logger.error(
        "operation_failed",
        error=str(e),
        exc_info=True
    )
```

### Environment Variables

```bash
# Enable debug logging
export CALLWHISPER_LOG_LEVEL=DEBUG

# Disable network guard (testing only)
export CALLWHISPER_NETWORK_GUARD_DISABLED=true

# Override config file
export CALLWHISPER_CONFIG_PATH=/custom/config.json
```

### Common Issues

**Issue: Port already in use**
```bash
# Find process using port 8765
lsof -i :8765  # Linux/macOS
netstat -ano | findstr :8765  # Windows

# Kill the process or change port in config
```

**Issue: FFmpeg not found**
```bash
# Verify FFmpeg is in PATH
ffmpeg -version

# Or set explicit path in config
```

**Issue: Import errors**
```bash
# Ensure PYTHONPATH includes src directory
export PYTHONPATH=src

# Or run from project root
python -m callwhisper
```

**Issue: Test fixtures not found**
```bash
# Ensure conftest.py adds src to path
# Run pytest from project root
cd /path/to/callwhisper-web
pytest
```

---

## Contributing

### Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b feature/amazing-feature`
4. Set up development environment (see above)

### Development Workflow

1. **Write tests** for your feature/fix
2. **Implement** the feature
3. **Run tests:** `pytest`
4. **Format code:** `black src/ tests/`
5. **Lint code:** `ruff check src/ tests/`
6. **Commit** with descriptive message
7. **Push** to your fork
8. **Open Pull Request**

### Commit Message Format

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(api): add batch upload endpoint

- Supports multiple file uploads in single request
- Returns job IDs for tracking

Closes #123
```

```
fix(transcriber): handle timeout gracefully

Previously, transcription timeout would cause crash.
Now properly catches asyncio.TimeoutError and reports to user.
```

### Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black --check src/ tests/`)
- [ ] No linting errors (`ruff check src/ tests/`)
- [ ] Documentation updated (if applicable)
- [ ] Changelog updated (if applicable)
- [ ] PR description explains the changes

### Security

- Never commit secrets or credentials
- Don't disable security features (network guard, device guard) in production code
- Report security issues privately to maintainers

---

## Useful Commands Reference

```bash
# Development server
PYTHONPATH=src python -m callwhisper

# Run tests
pytest
pytest -v --tb=short
pytest --cov=src/callwhisper --cov-report=html

# Code quality
black src/ tests/
ruff check src/ tests/
mypy src/callwhisper/

# Build Windows executable
.\scripts\build.ps1

# Download vendor binaries
.\scripts\download-vendor.ps1

# Check dependencies for vulnerabilities
pip-audit
safety check
```
