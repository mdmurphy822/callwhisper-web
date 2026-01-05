"""
CallWhisper - Main FastAPI Application

Web-based voice transcriber for recording Cisco Jabber/Finesse phone calls.

Based on LibV2 patterns:
- Structured logging with context
- Distributed tracing middleware
- Graceful startup/shutdown with recovery checks
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, __app_name__
from .core.config import get_settings
from .core.state import app_state
from .core.logging_config import get_logger, configure_logging
from .core.tracing import TracingMiddleware
from .core.persistence import CheckpointManager
from .core.bulkhead import get_executor, shutdown_executor
from .core.rate_limiter import (
    RateLimitMiddleware,
    SlidingWindowRateLimiter,
    RateLimitConfig,
)
from .core.cache import get_cache, CacheConfig
from .core.network_guard import enable_network_guard, is_guard_enabled
from .api.routes import router as api_router
from .api.websocket import router as ws_router
from .utils.paths import get_static_dir, get_data_dir

# Configure logging on module load
configure_logging()
logger = get_logger(__name__)


def print_startup_banner(version: str) -> None:
    """Print offline mode confirmation banner to console for IT verification."""
    banner = f"""
================================================================================
  CallWhisper v{version}
  Mode: FULLY OFFLINE
  External network connections: BLOCKED
  Transcription engine: Local whisper.cpp

  This application has ZERO external cloud dependencies.
  All audio processing and transcription occurs locally.
================================================================================
"""
    print(banner)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with recovery checks."""
    settings = get_settings()

    # Enable network guard FIRST - blocks all external connections
    enable_network_guard()

    # Print startup banner for IT verification
    print_startup_banner(__version__)

    # Ensure output directory exists
    output_dir = get_data_dir() / settings.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize checkpoint manager and check for incomplete sessions
    checkpoint_dir = get_data_dir() / "checkpoints"
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    incomplete_sessions = checkpoint_manager.get_incomplete_sessions()
    if incomplete_sessions:
        logger.warning(
            "incomplete_sessions_found",
            count=len(incomplete_sessions),
            sessions=[s.session_id for s in incomplete_sessions]
        )

    # Initialize bulkhead executor for isolated thread pools
    executor = get_executor()
    logger.info(
        "bulkhead_executor_ready",
        pools=list(executor.get_all_metrics().keys())
    )

    # Initialize transcription cache
    cache_config = CacheConfig(
        max_entries=settings.performance.cache_max_entries,
        ttl_seconds=settings.performance.cache_ttl_seconds,
        enabled=settings.performance.cache_enabled,
    )
    cache = get_cache(cache_config)
    logger.info(
        "transcription_cache_ready",
        max_entries=cache_config.max_entries,
        ttl_seconds=cache_config.ttl_seconds
    )

    logger.info(
        "application_starting",
        app_name=__app_name__,
        version=__version__,
        host=settings.server.host,
        port=settings.server.port,
        mode="OFFLINE",
        network_guard="enabled" if is_guard_enabled() else "disabled",
        external_connections="blocked",
        transcription_engine="whisper.cpp (local)",
        cors_enabled=settings.security.cors_enabled,
        rate_limit_enabled=settings.security.rate_limit_enabled
    )

    # Initialize call detector on Windows if enabled
    call_detector = None
    if sys.platform == "win32" and settings.call_detector.enabled:
        try:
            from .services.call_detector import get_call_detector, CallDetectorConfig

            call_detector = get_call_detector()
            detector_config = CallDetectorConfig(
                enabled=True,
                target_processes=settings.call_detector.target_processes,
                finesse_browsers=settings.call_detector.finesse_browsers,
                call_start_confirm_seconds=settings.call_detector.call_start_confirm_seconds,
                call_end_confirm_seconds=settings.call_detector.call_end_confirm_seconds,
                max_call_duration_minutes=settings.call_detector.max_call_duration_minutes,
                min_call_duration_seconds=settings.call_detector.min_call_duration_seconds,
            )
            await call_detector.start(detector_config)
            logger.info(
                "call_detector_started",
                targets=settings.call_detector.target_processes
            )
        except ImportError as e:
            logger.warning(
                "call_detector_unavailable",
                error=str(e),
                message="Install pycaw, wmi, and pywin32 to enable call detection"
            )
        except Exception as e:
            logger.error("call_detector_init_failed", error=str(e))

    yield

    # Cleanup on shutdown
    logger.info("application_shutting_down", app_name=__app_name__)

    # Stop call detector if running
    if call_detector is not None:
        try:
            await call_detector.stop()
            logger.info("call_detector_stopped")
        except Exception as e:
            logger.error("call_detector_stop_error", error=str(e))

    # Shutdown bulkhead executor gracefully
    shutdown_executor(wait=True)

    # Cleanup old checkpoints
    checkpoint_manager.cleanup_old_checkpoints(max_age_hours=48)


app = FastAPI(
    title=__app_name__,
    version=__version__,
    description="""
## CallWhisper - Voice Transcription Service

Secure, offline voice transcription for call recordings.

### Features
- **Record**: Capture audio from virtual audio devices (VB-Cable, Stereo Mix)
- **Transcribe**: Local whisper.cpp transcription (no cloud)
- **Export**: Download transcripts as TXT, SRT, or VTB bundles

### Security
- Fully offline - no external API calls
- Device Guard prevents microphone recording
- Network isolation enforced

### API Usage
Use these endpoints to integrate with other systems or automate transcription workflows.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health checks and readiness probes"},
        {"name": "devices", "description": "Audio device enumeration and validation"},
        {"name": "recording", "description": "Start/stop recording operations"},
        {"name": "transcription", "description": "Transcription and job management"},
        {"name": "recordings", "description": "Access completed recordings"},
        {"name": "call-detection", "description": "Automatic call detection (Windows only)"},
        {"name": "debug", "description": "Debugging and metrics endpoints"},
    ],
)

# Get settings for middleware configuration
_settings = get_settings()

# Add CORS middleware if enabled
if _settings.security.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.security.allowed_origins,
        allow_credentials=_settings.security.allow_credentials,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-Request-ID", "Content-Type"],
    )

# Add rate limiting middleware if enabled
if _settings.security.rate_limit_enabled:
    rate_limit_config = RateLimitConfig(
        requests_per_minute=_settings.security.rate_limit_rpm,
        burst_size=_settings.security.rate_limit_burst,
        enabled=True,
        excluded_paths=_settings.security.rate_limit_excluded,
    )
    rate_limiter = SlidingWindowRateLimiter(rate_limit_config)
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Add tracing middleware for distributed tracing
app.add_middleware(TracingMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)

# Mount static files
static_dir = get_static_dir()
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = get_static_dir() / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": f"{__app_name__} API", "version": __version__}


def main():
    """Entry point for running the application."""
    import uvicorn

    settings = get_settings()

    # Open browser if configured
    if settings.server.open_browser:
        import webbrowser
        import threading

        def open_browser():
            import time
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f"http://{settings.server.host}:{settings.server.port}")

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
