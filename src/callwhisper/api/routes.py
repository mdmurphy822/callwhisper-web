"""REST API routes."""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import (
    APIRouter,
    HTTPException,
    BackgroundTasks,
    File,
    UploadFile,
    Form,
    Depends,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
import uuid
import aiofiles

from .. import __version__
from ..core.config import get_settings
from ..core.state import app_state, AppState, RecordingSession, CompletedRecording
from ..core.logging_config import get_api_logger
from ..core.metrics import metrics
from ..core.tracing import get_request_id
from ..services.device_enum import list_audio_devices
from ..services.device_guard import is_device_safe, get_device_status
from ..services.audio_detector import get_setup_status, mark_setup_complete
from ..services.recorder import start_recording, stop_recording
from ..services.process_orchestrator import process_orchestrator
from ..core.bulkhead import get_executor
from ..core.cache import get_cache
from ..core.capability_registry import get_registry
from ..core.network_guard import is_guard_enabled, get_guard_status
from ..core.health import get_health_checker, HealthStatus as HealthCheckStatus
from ..core.job_store import get_job_store, JobCheckpoint
from ..core.metrics import get_transcription_store
from ..services.job_queue import get_job_queue, QueuedJob
from ..services.folder_scanner import scan_folder, scan_folder_paths, get_folder_stats
from ..utils.paths import (
    get_output_dir,
    get_ffmpeg_path,
    get_whisper_path,
    get_path_info,
    get_models_dir,
    get_data_dir,
    sanitize_path_component,
    validate_path_within_directory,
)

logger = get_api_logger()
router = APIRouter()


# Dependency to check if debug endpoints are enabled
async def require_debug_enabled():
    """
    Dependency that returns 404 if debug endpoints are disabled.

    This protects sensitive debug endpoints in production environments.
    Set security.debug_endpoints_enabled=true in config to enable.
    """
    settings = get_settings()
    if not settings.security.debug_endpoints_enabled:
        raise HTTPException(
            status_code=404, detail="Debug endpoints are disabled in production"
        )


# Request/Response Models


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str = "offline"
    network_guard: str = "enabled"
    external_api_calls: str = "none"
    transcription_engine: str = "whisper.cpp (local)"


class ReadinessCheck(BaseModel):
    """Individual readiness check result."""

    name: str
    ready: bool
    details: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool
    checks: List[ReadinessCheck]


class DetailedHealthCheck(BaseModel):
    """Individual detailed health check result."""

    name: str
    healthy: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class DetailedHealthResponse(BaseModel):
    """Detailed health check response for pre-recording validation."""

    healthy: bool
    checks: List[DetailedHealthCheck]
    timestamp: float


class IncompleteJob(BaseModel):
    """Incomplete job information for recovery."""

    job_id: str
    audio_path: str
    status: str
    chunks_completed: int
    total_chunks: int
    progress_percent: float
    device_name: Optional[str]
    ticket_id: Optional[str]
    created_at: float
    updated_at: float


class IncompleteJobsResponse(BaseModel):
    """Response with list of incomplete jobs."""

    jobs: List[IncompleteJob]
    count: int


class TranscriptionMetricItem(BaseModel):
    """Individual transcription metric."""

    job_id: str
    audio_duration_seconds: float
    transcription_duration_seconds: float
    processing_speed: float
    model_used: str
    success: bool
    error_message: Optional[str]
    device_name: Optional[str]
    ticket_id: Optional[str]
    timestamp: float


class DailyStatsItem(BaseModel):
    """Daily statistics."""

    date: str
    transcription_count: int
    total_audio_seconds: float
    total_processing_seconds: float
    success_count: int
    failure_count: int
    success_rate: float


class TranscriptionSummaryResponse(BaseModel):
    """Transcription metrics summary."""

    total_transcriptions: int
    total_audio_hours: float
    total_processing_hours: float
    avg_processing_speed: float
    success_rate: float
    last_7_days: List[DailyStatsItem]


class MetricsResponse(BaseModel):
    """Application metrics response."""

    uptime_seconds: float
    operations: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    bulkhead_pools: Dict[str, Any]
    cache: Dict[str, Any]
    active_recording: bool
    completed_recordings_count: int


class DebugStateResponse(BaseModel):
    """Debug state response."""

    request_id: str
    current_state: str
    current_session: Optional[Dict[str, Any]]
    completed_recordings_count: int
    circuit_breakers: Dict[str, Any]
    metrics_summary: Dict[str, Any]


class SetupStatusResponse(BaseModel):
    """First-run setup status response."""

    virtual_audio_detected: bool
    recommended_device_available: bool
    detected_devices: List[Dict[str, Any]]
    all_audio_devices: List[str]
    recommended_action: Optional[str]
    setup_complete: bool
    setup_skipped: bool
    recommendations: List[Dict[str, str]]


class SetupCompleteRequest(BaseModel):
    """Request to mark setup as complete."""

    skipped: bool = False


class DeviceInfo(BaseModel):
    name: str
    safe: bool
    reason: Optional[str] = None


class DevicesResponse(BaseModel):
    devices: List[DeviceInfo]


class StateResponse(BaseModel):
    state: str
    recording_id: Optional[str] = None
    elapsed_seconds: int = 0
    elapsed_formatted: str = "00:00"


class StartRecordingRequest(BaseModel):
    """
    Request model for starting a recording.

    Based on LibV2 Pydantic patterns:
    - Explicit validation with Field constraints
    - Custom validators for complex rules
    - Immutable after creation (frozen)
    """

    device: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Audio device name from /api/devices",
    )
    ticket_id: Optional[str] = Field(
        None, max_length=50, description="Optional ticket/case ID for the recording"
    )

    @field_validator("device")
    @classmethod
    def validate_device_name(cls, v: str) -> str:
        """Validate device name doesn't contain suspicious characters."""
        # Allow alphanumeric, spaces, parentheses, hyphens, underscores
        # This prevents command injection via device name
        if not re.match(r"^[\w\s\(\)\-\.\,\'\"]+$", v, re.UNICODE):
            raise ValueError("Device name contains invalid characters")
        return v.strip()

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate ticket ID format with path traversal protection."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Use centralized sanitization to prevent path traversal
        try:
            return sanitize_path_component(v, max_length=50)
        except ValueError as e:
            raise ValueError(f"Invalid ticket ID: {e}")

    model_config = {
        "frozen": True,  # Immutable after creation
        "str_strip_whitespace": True,
    }


class StartRecordingResponse(BaseModel):
    recording_id: str
    started_at: str


class StopRecordingResponse(BaseModel):
    recording_id: str
    state: str


class RecordingInfo(BaseModel):
    id: str
    ticket_id: Optional[str]
    created_at: str
    duration_seconds: float
    output_folder: str
    bundle_path: Optional[str] = None


class UpdateTranscriptRequest(BaseModel):
    """Request model for updating a transcript."""

    text: str = Field(
        ..., min_length=0, max_length=500000, description="The updated transcript text"
    )


class RecordingsResponse(BaseModel):
    recordings: List[RecordingInfo]


# Routes


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Liveness probe - is the service running?

    Returns 200 OK if the service is alive.
    Includes offline mode confirmation for enterprise IT verification.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        mode="offline",
        network_guard="enabled" if is_guard_enabled() else "disabled",
        external_api_calls="none",
        transcription_engine="whisper.cpp (local)",
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness probe - can the service accept work?

    Checks:
    - FFmpeg availability
    - Whisper model availability
    - Disk space
    - Application state
    """
    checks = []

    # Check FFmpeg
    ffmpeg_path = get_ffmpeg_path()
    ffmpeg_ready = ffmpeg_path.exists() or shutil.which("ffmpeg") is not None
    checks.append(
        ReadinessCheck(
            name="ffmpeg",
            ready=ffmpeg_ready,
            details=str(ffmpeg_path) if ffmpeg_ready else "FFmpeg not found",
        )
    )

    # Check Whisper
    whisper_path = get_whisper_path()
    whisper_ready = whisper_path.exists() or shutil.which("whisper-cli") is not None
    checks.append(
        ReadinessCheck(
            name="whisper",
            ready=whisper_ready,
            details=str(whisper_path) if whisper_ready else "Whisper not found",
        )
    )

    # Check disk space (require at least 1GB free)
    output_dir = get_output_dir()
    try:
        disk_usage = shutil.disk_usage(output_dir)
        free_gb = disk_usage.free / (1024**3)
        disk_ready = free_gb >= 1.0
        checks.append(
            ReadinessCheck(
                name="disk_space", ready=disk_ready, details=f"{free_gb:.2f} GB free"
            )
        )
    except Exception as e:
        checks.append(ReadinessCheck(name="disk_space", ready=False, details=str(e)))

    # Check application state
    state_ready = app_state.state != AppState.ERROR
    checks.append(
        ReadinessCheck(
            name="app_state", ready=state_ready, details=app_state.state.value
        )
    )

    # Check bulkhead executor health
    executor = get_executor()
    bulkhead_healthy = executor.is_healthy()
    checks.append(
        ReadinessCheck(
            name="bulkhead",
            ready=bulkhead_healthy,
            details=(
                "All pools healthy"
                if bulkhead_healthy
                else "One or more pools overloaded"
            ),
        )
    )

    # Check network guard (should be enabled for enterprise deployment)
    network_guard_enabled = is_guard_enabled()
    checks.append(
        ReadinessCheck(
            name="network_guard",
            ready=network_guard_enabled,
            details=(
                "External connections blocked"
                if network_guard_enabled
                else "WARNING: External connections allowed"
            ),
        )
    )

    all_ready = all(check.ready for check in checks)

    return ReadinessResponse(ready=all_ready, checks=checks)


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(device: Optional[str] = None):
    """
    Detailed health check for pre-recording validation.

    Runs comprehensive checks on system resources and dependencies.
    Call this before starting a recording to ensure everything is ready.

    Args:
        device: Optional audio device name to validate

    Returns:
        Detailed health status with individual check results
    """
    from ..core.health import configure_health_checker

    # Configure health checker with current paths
    checker = configure_health_checker(
        ffmpeg_path=str(get_ffmpeg_path()),
        models_dir=get_models_dir(),
        min_disk_gb=1.0,
        min_memory_mb=500.0,
    )

    # Run all health checks
    status = await checker.run_all_checks(
        device_name=device, recordings_dir=get_output_dir()
    )

    # Convert to response model
    checks = [
        DetailedHealthCheck(
            name=c.name, healthy=c.healthy, message=c.message, details=c.details
        )
        for c in status.checks
    ]

    return DetailedHealthResponse(
        healthy=status.healthy, checks=checks, timestamp=status.timestamp
    )


@router.get("/jobs/incomplete", response_model=IncompleteJobsResponse)
async def get_incomplete_jobs():
    """
    Get list of incomplete jobs for recovery.

    Called on app startup to check if there are jobs
    that need to be resumed or discarded.
    """
    store = get_job_store()
    incomplete = store.get_incomplete_jobs()

    jobs = [
        IncompleteJob(
            job_id=j.job_id,
            audio_path=j.audio_path,
            status=j.status,
            chunks_completed=j.chunks_completed,
            total_chunks=j.total_chunks,
            progress_percent=j.progress_percent,
            device_name=j.device_name,
            ticket_id=j.ticket_id,
            created_at=j.created_at,
            updated_at=j.updated_at,
        )
        for j in incomplete
    ]

    return IncompleteJobsResponse(jobs=jobs, count=len(jobs))


@router.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str, background_tasks: BackgroundTasks):
    """
    Resume an incomplete job.

    Attempts to continue transcription from the last checkpoint.
    Uses chunked transcription to resume from the last completed chunk.
    """
    store = get_job_store()
    checkpoint = store.load_checkpoint(job_id)

    if not checkpoint:
        raise HTTPException(status_code=404, detail="Job not found")

    if checkpoint.is_complete:
        raise HTTPException(status_code=400, detail="Job already complete")

    # Verify audio file still exists
    audio_path = Path(checkpoint.audio_path)
    if not audio_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file no longer exists. Job cannot be resumed.",
        )

    # Check app state - can't resume if already busy
    if app_state.state == AppState.RECORDING:
        raise HTTPException(status_code=400, detail="Recording in progress")
    if app_state.state == AppState.PROCESSING:
        raise HTTPException(status_code=400, detail="Another job is processing")

    # Queue background task to resume transcription
    background_tasks.add_task(resume_transcription, checkpoint=checkpoint)

    logger.info(
        "job_resume_started",
        job_id=job_id,
        from_chunk=checkpoint.chunks_completed,
        total_chunks=checkpoint.total_chunks,
    )

    return {
        "status": "resume_started",
        "job_id": job_id,
        "from_chunk": checkpoint.chunks_completed,
        "total_chunks": checkpoint.total_chunks,
    }


@router.delete("/jobs/{job_id}")
async def discard_job(job_id: str):
    """
    Discard an incomplete job.

    Deletes the checkpoint and any partial data.
    """
    store = get_job_store()

    if not store.delete_checkpoint(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    logger.info("job_discarded", job_id=job_id)

    return {"status": "ok", "job_id": job_id, "deleted": True}


@router.get("/jobs/history")
async def get_job_history(limit: int = 50):
    """
    Get recent job history from archive.

    Returns completed jobs for reporting/analytics.
    """
    store = get_job_store()
    history = store.get_job_history(limit=limit)

    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "chunks_completed": j.chunks_completed,
                "total_chunks": j.total_chunks,
                "ticket_id": j.ticket_id,
                "created_at": j.created_at,
                "updated_at": j.updated_at,
            }
            for j in history
        ],
        "count": len(history),
    }


@router.get("/transcriptions/summary", response_model=TranscriptionSummaryResponse)
async def get_transcription_summary():
    """
    Get transcription metrics summary.

    Returns aggregate statistics for the metrics dashboard.
    """
    store = get_transcription_store()
    summary = store.get_summary()

    return TranscriptionSummaryResponse(
        total_transcriptions=summary.total_transcriptions,
        total_audio_hours=round(summary.total_audio_hours, 2),
        total_processing_hours=round(summary.total_processing_hours, 2),
        avg_processing_speed=round(summary.avg_processing_speed, 2),
        success_rate=round(summary.success_rate, 4),
        last_7_days=[
            DailyStatsItem(
                date=d.date,
                transcription_count=d.transcription_count,
                total_audio_seconds=round(d.total_audio_seconds, 2),
                total_processing_seconds=round(d.total_processing_seconds, 2),
                success_count=d.success_count,
                failure_count=d.failure_count,
                success_rate=round(d.success_rate, 4),
            )
            for d in summary.last_7_days
        ],
    )


@router.get("/transcriptions/recent")
async def get_recent_transcriptions(limit: int = 50):
    """
    Get recent transcription records.

    Returns individual transcription metrics for the dashboard.
    """
    store = get_transcription_store()
    recent = store.get_recent(limit=limit)

    return {
        "transcriptions": [
            {
                "job_id": m.job_id,
                "audio_duration_seconds": round(m.audio_duration_seconds, 2),
                "transcription_duration_seconds": round(
                    m.transcription_duration_seconds, 2
                ),
                "processing_speed": round(m.processing_speed, 2),
                "model_used": m.model_used,
                "success": m.success,
                "error_message": m.error_message,
                "device_name": m.device_name,
                "ticket_id": m.ticket_id,
                "timestamp": m.timestamp,
            }
            for m in recent
        ],
        "count": len(recent),
    }


@router.post("/transcriptions/export")
async def export_transcription_metrics():
    """
    Export transcription metrics to CSV.

    Returns the CSV file for download.
    """
    from tempfile import NamedTemporaryFile

    store = get_transcription_store()

    # Create temp file for CSV
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = Path(f.name)

    count = store.export_csv(temp_path)

    if count == 0:
        temp_path.unlink()
        raise HTTPException(status_code=404, detail="No metrics to export")

    return FileResponse(
        temp_path,
        filename=f"transcription_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        media_type="text/csv",
    )


@router.get("/setup/status", response_model=SetupStatusResponse)
async def setup_status():
    """
    Check first-run setup status.

    Used by the setup wizard to determine if VB-Cable or other
    virtual audio devices are installed.

    Returns:
        Setup status including detected virtual audio devices
        and recommended actions.
    """
    status = get_setup_status()
    logger.debug(
        "setup_status_check",
        request_id=get_request_id(),
        virtual_audio_detected=status["virtual_audio_detected"],
    )
    return SetupStatusResponse(**status)


@router.post("/setup/complete")
async def setup_complete(request: SetupCompleteRequest):
    """
    Mark first-run setup as complete.

    Called when user completes or skips the setup wizard.

    Args:
        request: Contains 'skipped' flag indicating if user skipped setup

    Returns:
        Success status
    """
    success = mark_setup_complete(skipped=request.skipped)

    logger.info(
        "setup_complete",
        request_id=get_request_id(),
        skipped=request.skipped,
        success=success,
    )

    return {
        "status": "ok" if success else "error",
        "skipped": request.skipped,
    }


@router.get("/health/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Application metrics endpoint.

    Returns operation counts, durations, error rates, and circuit breaker states.
    """
    all_metrics = metrics.get_all_metrics()

    return MetricsResponse(
        uptime_seconds=all_metrics.get("uptime_seconds", 0),
        operations=all_metrics.get("operations", {}),
        circuit_breakers=process_orchestrator.get_circuit_status(),
        bulkhead_pools=get_executor().get_all_metrics(),
        cache=get_cache().get_stats(),
        active_recording=app_state.state == AppState.RECORDING,
        completed_recordings_count=len(app_state.completed_recordings),
    )


@router.get(
    "/debug/state",
    response_model=DebugStateResponse,
    dependencies=[Depends(require_debug_enabled)],
)
async def debug_state():
    """
    Get detailed application state for debugging.

    Includes current session info, circuit breaker states, and metrics summary.
    """
    current_session = None
    if app_state.current_session:
        session = app_state.current_session
        current_session = {
            "id": session.id,
            "device_name": session.device_name,
            "ticket_id": session.ticket_id,
            "start_time": (
                session.start_time.isoformat() if session.start_time else None
            ),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "output_folder": session.output_folder,
        }

    return DebugStateResponse(
        request_id=get_request_id(),
        current_state=app_state.state.value,
        current_session=current_session,
        completed_recordings_count=len(app_state.completed_recordings),
        circuit_breakers=process_orchestrator.get_circuit_status(),
        metrics_summary=metrics.get_all_metrics(),
    )


@router.post("/debug/reset-metrics", dependencies=[Depends(require_debug_enabled)])
async def reset_metrics():
    """Reset all collected metrics (for testing)."""
    metrics.reset()
    logger.info("metrics_reset", request_id=get_request_id())
    return {"status": "ok", "message": "Metrics reset"}


@router.post("/debug/reset-circuits", dependencies=[Depends(require_debug_enabled)])
async def reset_circuits():
    """Reset all circuit breakers (for testing/recovery)."""
    for handler in list(process_orchestrator._circuit_breakers.keys()):
        process_orchestrator.reset_circuit(handler)
    logger.info("circuits_reset", request_id=get_request_id())
    return {"status": "ok", "message": "Circuit breakers reset"}


@router.get("/debug/cache", dependencies=[Depends(require_debug_enabled)])
async def get_cache_stats():
    """Get transcription cache statistics."""
    cache = get_cache()
    return {"request_id": get_request_id(), "cache": cache.get_stats()}


@router.post("/debug/cache/clear", dependencies=[Depends(require_debug_enabled)])
async def clear_cache():
    """Clear the transcription cache."""
    cache = get_cache()
    cleared = cache.clear()
    logger.info("cache_cleared", request_id=get_request_id(), entries=cleared)
    return {"status": "ok", "entries_cleared": cleared}


@router.get("/debug/capabilities", dependencies=[Depends(require_debug_enabled)])
async def get_capabilities():
    """Get registered capability handlers."""
    registry = get_registry()
    return {
        "request_id": get_request_id(),
        "registry": registry.get_registry_stats(),
        "types": {
            cap_type: registry.get_capability_info(cap_type)
            for cap_type in registry.get_all_types()
        },
    }


@router.get("/debug/network", dependencies=[Depends(require_debug_enabled)])
async def get_network_status():
    """
    Get network isolation status.

    Confirms that external connections are blocked for enterprise IT.
    """
    return {
        "request_id": get_request_id(),
        "network": get_guard_status(),
        "cloud_dependencies": [],
        "external_api_calls": "none",
        "transcription_engine": "whisper.cpp (local)",
        "offline_verified": is_guard_enabled(),
    }


@router.get("/debug/paths", dependencies=[Depends(require_debug_enabled)])
async def get_paths():
    """
    Get installation paths and deployment mode.

    Useful for IT administrators to verify installation location
    and data directory paths.
    """
    path_info = get_path_info()
    return {"request_id": get_request_id(), **path_info}


@router.get("/devices", response_model=DevicesResponse)
async def get_devices():
    """List available audio devices."""
    settings = get_settings()
    devices = list_audio_devices()

    device_list = []
    for name in devices:
        status = get_device_status(name, settings.device_guard)
        device_list.append(
            DeviceInfo(
                name=name,
                safe=status["safe"],
                reason=status.get("reason"),
            )
        )

    return DevicesResponse(devices=device_list)


@router.get("/state", response_model=StateResponse)
async def get_state():
    """Get current application state."""
    info = app_state.get_state_info()
    return StateResponse(**info)


@router.post("/recording/start", response_model=StartRecordingResponse)
async def start_recording_endpoint(
    request: StartRecordingRequest, background_tasks: BackgroundTasks
):
    """Start a new recording."""
    settings = get_settings()

    # Check if already recording
    if app_state.state == AppState.RECORDING:
        raise HTTPException(status_code=400, detail="Already recording")

    if app_state.state == AppState.PROCESSING:
        raise HTTPException(status_code=400, detail="Processing in progress")

    # Validate device safety
    if settings.device_guard.enabled:
        status = get_device_status(request.device, settings.device_guard)
        if not status["safe"]:
            raise HTTPException(
                status_code=403,
                detail=f"Device blocked: {status.get('reason', 'Not on allowlist')}",
            )

    # Generate recording ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_id = (
        f"{timestamp}_{request.ticket_id}" if request.ticket_id else timestamp
    )

    # Create session
    session = RecordingSession(
        id=recording_id,
        device_name=request.device,
        ticket_id=request.ticket_id,
    )

    # Start recording in background
    try:
        await start_recording(session, settings)
        await app_state.start_recording(session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StartRecordingResponse(
        recording_id=recording_id,
        started_at=session.start_time.isoformat() if session.start_time else "",
    )


@router.post("/recording/stop", response_model=StopRecordingResponse)
async def stop_recording_endpoint(background_tasks: BackgroundTasks):
    """Stop the current recording and start transcription."""
    if app_state.state != AppState.RECORDING:
        raise HTTPException(status_code=400, detail="Not currently recording")

    recording_id = app_state.current_session.id if app_state.current_session else None

    # Stop recording and start processing
    await app_state.stop_recording()

    # Add transcription to background tasks
    background_tasks.add_task(process_recording)

    return StopRecordingResponse(
        recording_id=recording_id or "",
        state=app_state.state.value,
    )


async def process_recording():
    """Background task to process recording (transcribe, bundle)."""
    from ..services.recorder import finalize_recording
    from ..services.transcriber import transcribe_audio
    from ..services.bundler import create_vtb_bundle

    settings = get_settings()
    session = app_state.current_session

    if not session:
        await app_state.set_error("No recording session found")
        return

    try:
        # Stop FFmpeg and get output path
        await app_state.processing_progress(10, "Finalizing recording")
        output_folder = await finalize_recording(session)

        # Normalize and transcribe with progress updates
        await app_state.processing_progress(30, "Transcribing audio")

        async def transcription_progress(percent: int, stage: str):
            await app_state.processing_progress(percent, stage)

        async def partial_transcript_callback(text: str, is_final: bool):
            await app_state.partial_transcript(text, is_final)

        transcript_path = await transcribe_audio(
            output_folder,
            settings,
            progress_callback=transcription_progress,
            partial_transcript_callback=partial_transcript_callback,
        )

        # Create bundle if enabled
        bundle_path = None
        if settings.output.create_bundle:
            await app_state.processing_progress(80, "Creating bundle")
            bundle_path = await create_vtb_bundle(output_folder, session, settings)

        # Complete
        await app_state.processing_progress(100, "Complete")

        # Read transcript preview
        transcript_preview = None
        if transcript_path and transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_preview = f.read()[:500]

        # Calculate duration
        duration = 0.0
        if session.start_time and session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()

        completed = CompletedRecording(
            id=session.id,
            ticket_id=session.ticket_id,
            created_at=session.start_time or datetime.now(),
            duration_seconds=duration,
            output_folder=str(output_folder),
            bundle_path=str(bundle_path) if bundle_path else None,
            transcript_preview=transcript_preview,
        )

        await app_state.complete_recording(completed)

    except Exception as e:
        await app_state.set_error(str(e))


async def resume_transcription(checkpoint: JobCheckpoint):
    """
    Background task to resume an interrupted transcription.

    Uses chunked transcription to continue from the last completed chunk.
    """
    from ..services.transcriber import transcribe_audio_chunked
    from ..services.bundler import create_vtb_bundle

    settings = get_settings()
    output_folder = Path(checkpoint.audio_path).parent
    store = get_job_store()

    try:
        # Set app state to processing
        await app_state.processing_progress(
            5, f"Resuming job from chunk {checkpoint.chunks_completed + 1}"
        )

        # Progress callback for UI updates
        async def progress_cb(pct: int, stage: str):
            await app_state.processing_progress(pct, stage)

        async def partial_transcript_cb(text: str, is_final: bool):
            await app_state.partial_transcript(text, is_final)

        # Run chunked transcription with resume
        transcript_path = await transcribe_audio_chunked(
            output_folder,
            settings,
            job_id=checkpoint.job_id,
            start_from_chunk=checkpoint.chunks_completed,
            progress_callback=progress_cb,
            partial_transcript_callback=partial_transcript_cb,
        )

        # Get duration info
        duration = 0.0
        try:
            from ..services.transcriber import get_audio_duration_seconds

            audio_path = output_folder / "audio_16k.wav"
            if audio_path.exists():
                duration = await get_audio_duration_seconds(audio_path)
        except Exception as e:
            logger.warning(
                "duration_detection_failed",
                job_id=checkpoint.job_id,
                audio_path=str(audio_path) if "audio_path" in dir() else None,
                error=str(e),
                error_type=type(e).__name__,
            )

        # Create bundle if enabled
        bundle_path = None
        if settings.output.create_bundle:
            await app_state.processing_progress(85, "Creating bundle")
            # Create a mock session for the bundler
            session = RecordingSession(
                id=checkpoint.job_id,
                device_name=checkpoint.device_name or "resumed_job",
                ticket_id=checkpoint.ticket_id,
            )
            session.start_time = datetime.fromtimestamp(checkpoint.created_at)
            session.end_time = datetime.now()
            session.output_folder = str(output_folder)

            bundle_path = await create_vtb_bundle(output_folder, session, settings)

        # Mark job complete in store
        store.mark_complete(checkpoint.job_id)

        # Read transcript preview
        transcript_preview = None
        if transcript_path and transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_preview = f.read()[:500]

        # Create completed recording entry
        completed = CompletedRecording(
            id=checkpoint.job_id,
            ticket_id=checkpoint.ticket_id,
            created_at=datetime.fromtimestamp(checkpoint.created_at),
            duration_seconds=duration,
            output_folder=str(output_folder),
            bundle_path=str(bundle_path) if bundle_path else None,
            transcript_preview=transcript_preview,
        )

        await app_state.processing_progress(100, "Complete")
        await app_state.complete_recording(completed)

        logger.info(
            "resume_transcription_complete",
            job_id=checkpoint.job_id,
            duration_seconds=duration,
        )

    except Exception as e:
        logger.error(
            "resume_transcription_failed", job_id=checkpoint.job_id, error=str(e)
        )
        store.mark_failed(checkpoint.job_id, str(e))
        await app_state.set_error(f"Resume failed: {str(e)}")


@router.get("/recordings", response_model=RecordingsResponse)
async def list_recordings():
    """List completed recordings."""
    recordings = [
        RecordingInfo(
            id=r.id,
            ticket_id=r.ticket_id,
            created_at=r.created_at.isoformat(),
            duration_seconds=r.duration_seconds,
            output_folder=r.output_folder,
            bundle_path=r.bundle_path,
        )
        for r in app_state.completed_recordings
    ]
    return RecordingsResponse(recordings=recordings)


class SearchRecordingsResponse(BaseModel):
    """Response for recordings search."""

    recordings: List[RecordingInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get(
    "/recordings/search", response_model=SearchRecordingsResponse, tags=["recordings"]
)
async def search_recordings(
    query: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    ticket_prefix: Optional[str] = None,
    sort: str = "newest",
    page: int = 1,
    page_size: int = 20,
):
    """
    Search and filter recordings with pagination.

    Args:
        query: Full-text search in recording ID, ticket ID, or transcript
        date_from: Filter recordings from this date (YYYY-MM-DD)
        date_to: Filter recordings until this date (YYYY-MM-DD)
        ticket_prefix: Filter recordings by ticket ID prefix
        sort: Sort order - 'newest', 'oldest', or 'duration'
        page: Page number (1-indexed)
        page_size: Number of results per page (max 100)

    Returns:
        Paginated list of matching recordings
    """
    from datetime import datetime as dt

    # Validate parameters
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100

    # Start with all recordings
    results = list(app_state.completed_recordings)

    # Apply filters
    if query:
        query_lower = query.lower()
        filtered = []
        for r in results:
            # Search in ID, ticket ID, and transcript preview
            searchable = [
                r.id.lower(),
                (r.ticket_id or "").lower(),
                (r.transcript_preview or "").lower(),
            ]
            if any(query_lower in s for s in searchable):
                filtered.append(r)
        results = filtered

    if date_from:
        try:
            from_date = dt.strptime(date_from, "%Y-%m-%d")
            results = [r for r in results if r.created_at >= from_date]
        except ValueError:
            pass  # Invalid date format, skip filter

    if date_to:
        try:
            to_date = dt.strptime(date_to, "%Y-%m-%d")
            # Include the entire "to" day by adding a day
            to_date = to_date.replace(hour=23, minute=59, second=59)
            results = [r for r in results if r.created_at <= to_date]
        except ValueError:
            pass  # Invalid date format, skip filter

    if ticket_prefix:
        prefix_lower = ticket_prefix.lower()
        results = [
            r
            for r in results
            if r.ticket_id and r.ticket_id.lower().startswith(prefix_lower)
        ]

    # Sort results
    if sort == "oldest":
        results.sort(key=lambda r: r.created_at)
    elif sort == "duration":
        results.sort(key=lambda r: r.duration_seconds, reverse=True)
    else:  # newest (default)
        results.sort(key=lambda r: r.created_at, reverse=True)

    # Calculate pagination
    total = len(results)
    total_pages = max(1, (total + page_size - 1) // page_size)

    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_results = results[start_idx:end_idx]

    # Convert to response format
    recordings = [
        RecordingInfo(
            id=r.id,
            ticket_id=r.ticket_id,
            created_at=r.created_at.isoformat(),
            duration_seconds=r.duration_seconds,
            output_folder=r.output_folder,
            bundle_path=r.bundle_path,
        )
        for r in paginated_results
    ]

    return SearchRecordingsResponse(
        recordings=recordings,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/recordings/{recording_id}/download")
async def download_bundle(recording_id: str):
    """Download the VTB bundle for a recording."""
    # Find recording
    recording = next(
        (r for r in app_state.completed_recordings if r.id == recording_id), None
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if not recording.bundle_path:
        raise HTTPException(status_code=404, detail="Bundle not available")

    bundle_path = Path(recording.bundle_path)
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail="Bundle file not found")

    return FileResponse(
        bundle_path,
        filename=bundle_path.name,
        media_type="application/x-vtb",
    )


@router.get("/recordings/{recording_id}/transcript")
async def get_transcript(recording_id: str):
    """Get transcript text for a recording."""
    # Find recording
    recording = next(
        (r for r in app_state.completed_recordings if r.id == recording_id), None
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if not recording.output_folder:
        raise HTTPException(status_code=404, detail="Output folder not available")

    output_folder = Path(recording.output_folder)
    transcript_txt = output_folder / "transcript.txt"
    transcript_srt = output_folder / "transcript.srt"

    # Read transcript text
    text = ""
    if transcript_txt.exists():
        text = transcript_txt.read_text(encoding="utf-8")
    else:
        text = "[No transcript available]"

    # Read SRT if available
    srt = ""
    if transcript_srt.exists():
        srt = transcript_srt.read_text(encoding="utf-8")

    return {
        "id": recording.id,
        "text": text,
        "srt": srt,
        "duration_seconds": recording.duration_seconds,
        "ticket_id": recording.ticket_id,
        "word_count": len(text.split()) if text else 0,
    }


@router.put("/recordings/{recording_id}/transcript", tags=["recordings"])
async def update_transcript(recording_id: str, request: UpdateTranscriptRequest):
    """
    Update transcript text for a recording.

    Overwrites the transcript.txt file with the new text and updates
    the recording's transcript preview in memory.

    Args:
        recording_id: The ID of the recording to update
        request: Contains the new transcript text

    Returns:
        Status with word count
    """
    # Find recording
    recording = next(
        (r for r in app_state.completed_recordings if r.id == recording_id), None
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    if not recording.output_folder:
        raise HTTPException(status_code=404, detail="Output folder not available")

    output_folder = Path(recording.output_folder)
    transcript_txt = output_folder / "transcript.txt"

    if not output_folder.exists():
        raise HTTPException(status_code=404, detail="Recording files not found")

    # Write updated transcript
    try:
        transcript_txt.write_text(request.text, encoding="utf-8")
    except Exception as e:
        logger.error(
            "transcript_update_failed", recording_id=recording_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to save transcript: {str(e)}"
        )

    # Update the in-memory transcript preview
    recording.transcript_preview = request.text[:500] if request.text else None

    word_count = len(request.text.split()) if request.text else 0

    logger.info("transcript_updated", recording_id=recording_id, word_count=word_count)

    return {
        "status": "ok",
        "recording_id": recording_id,
        "word_count": word_count,
    }


# ============================================================================
# Export Endpoints
# ============================================================================

# Supported export formats
EXPORT_FORMATS = {"json", "vtt", "csv", "pdf", "docx"}


@router.get("/recordings/{recording_id}/export/{format}", tags=["recordings"])
async def export_transcript(recording_id: str, format: str):
    """
    Export transcript in specified format.

    Supported formats:
    - json: Structured JSON with metadata and segments
    - vtt: WebVTT subtitle format
    - csv: Tabular segment data
    - pdf: Formatted PDF document
    - docx: Word document

    Returns the exported file for download.
    """
    format = format.lower()

    if format not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: {', '.join(sorted(EXPORT_FORMATS))}",
        )

    # Find recording
    recording = next(
        (r for r in app_state.completed_recordings if r.id == recording_id), None
    )
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    output_folder = Path(recording.output_folder)
    if not output_folder.exists():
        raise HTTPException(status_code=404, detail="Recording files not found")

    from ..services.exporter import get_exporter

    exporter = get_exporter(output_folder)

    # Export based on format
    try:
        if format == "json":
            export_path = await exporter.export_json(recording_id)
            media_type = "application/json"
        elif format == "vtt":
            export_path = await exporter.export_vtt(recording_id)
            media_type = "text/vtt"
        elif format == "csv":
            export_path = await exporter.export_csv(recording_id)
            media_type = "text/csv"
        elif format == "pdf":
            export_path = await exporter.export_pdf(recording_id)
            media_type = "application/pdf"
        elif format == "docx":
            export_path = await exporter.export_docx(recording_id)
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    except ImportError as e:
        logger.error("export_missing_dependency", format=format, error=str(e))
        raise HTTPException(
            status_code=501,
            detail=f"Export format '{format}' requires additional dependencies: {str(e)}",
        )
    except Exception as e:
        logger.error(
            "export_failed", recording_id=recording_id, format=format, error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    logger.info(
        "export_completed",
        recording_id=recording_id,
        format=format,
        path=str(export_path),
    )

    return FileResponse(
        export_path,
        filename=export_path.name,
        media_type=media_type,
    )


@router.get("/recordings/{recording_id}/export-formats", tags=["recordings"])
async def get_export_formats(recording_id: str):
    """
    Get available export formats for a recording.

    All formats are available by default.
    """
    # Verify recording exists
    recording = next(
        (r for r in app_state.completed_recordings if r.id == recording_id), None
    )
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    formats = {
        "json": {"available": True, "description": "Structured JSON with metadata"},
        "vtt": {"available": True, "description": "WebVTT subtitle format"},
        "csv": {"available": True, "description": "Tabular segment data"},
        "pdf": {"available": True, "description": "PDF document"},
        "docx": {"available": True, "description": "Word document"},
    }

    return {"recording_id": recording_id, "formats": formats}


@router.post("/reset")
async def reset_state():
    """Reset application to idle state."""
    await app_state.reset()
    return {"status": "ok"}


@router.post("/recordings/{recording_id}/open-folder")
async def open_recording_folder(recording_id: str):
    """Open the output folder for a recording in the system file manager."""
    recording = recording_manager.get_recording(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    folder_path = Path(recording.output_folder)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    # Platform-specific folder opening
    import sys

    if sys.platform == "linux":
        subprocess.Popen(["xdg-open", str(folder_path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder_path)])
    elif sys.platform == "win32":
        subprocess.Popen(["explorer", str(folder_path)])

    logger.info("open_folder", recording_id=recording_id, folder=str(folder_path))
    return {"status": "ok", "folder": str(folder_path)}


# File Upload Constants
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/opus",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
    "audio/x-flac",
    "audio/webm",
}
# File extension to MIME type mapping (fallback when MIME is generic)
ALLOWED_EXTENSIONS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class UploadResponse(BaseModel):
    """Response for file upload."""

    recording_id: str
    status: str
    message: str


class BatchUploadResponse(BaseModel):
    """Response for batch file upload."""

    status: str
    jobs_queued: int
    job_ids: List[str]


class QueueJobInfo(BaseModel):
    """Information about a queued job."""

    job_id: str
    original_filename: str
    ticket_id: Optional[str]
    status: str
    priority: int
    progress: int
    error_message: Optional[str]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]


class QueueStatusResponse(BaseModel):
    """Response for queue status."""

    queued: List[QueueJobInfo]
    processing: Optional[QueueJobInfo]
    completed: List[QueueJobInfo]
    failed: List[QueueJobInfo]
    counts: Dict[str, int]


class FolderScanResponse(BaseModel):
    """Response for folder scan."""

    total_files: int
    total_size_mb: float
    extensions: Dict[str, Dict[str, Any]]
    oldest_file: Optional[Dict[str, str]]
    newest_file: Optional[Dict[str, str]]


@router.post("/recordings/upload", response_model=UploadResponse)
async def upload_audio_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ticket_id: Optional[str] = Form(None),
):
    """
    Upload an audio file for transcription.

    Accepts audio files (WAV, MP3, OGG, M4A, FLAC, etc.) up to 500MB.
    The file will be processed through the same transcription pipeline
    as live recordings.

    Args:
        file: Audio file to transcribe
        ticket_id: Optional ticket/case ID

    Returns:
        Recording ID for tracking progress via WebSocket
    """
    # Validate content type - check MIME type or fall back to file extension
    content_type = file.content_type
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    # If MIME type is generic or unknown, use file extension
    if content_type in (None, "", "application/octet-stream", "binary/octet-stream"):
        if ext in ALLOWED_EXTENSIONS:
            content_type = ALLOWED_EXTENSIONS[ext]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {ext}. Supported: WAV, MP3, OGG, M4A, FLAC",
            )
    elif content_type not in ALLOWED_AUDIO_TYPES and not content_type.startswith(
        "audio/"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {content_type}. Supported: WAV, MP3, OGG, M4A, FLAC",
        )

    # Read file and check size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // 1024 // 1024}MB",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate and sanitize ticket_id if provided
    if ticket_id:
        ticket_id = ticket_id.strip()
        try:
            ticket_id = sanitize_path_component(ticket_id)
        except ValueError as e:
            logger.warning("invalid_ticket_id", ticket_id=ticket_id, error=str(e))
            raise HTTPException(
                status_code=400,
                detail="Invalid ticket ID: contains disallowed characters",
            )

    # Check if busy
    if app_state.state == AppState.RECORDING:
        raise HTTPException(status_code=400, detail="Recording in progress")
    if app_state.state == AppState.PROCESSING:
        raise HTTPException(status_code=400, detail="Processing in progress")

    # Generate recording ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    recording_id = f"upload_{timestamp}_{short_uuid}"
    if ticket_id:
        recording_id = f"upload_{timestamp}_{ticket_id}"

    # Create output folder with path validation
    output_folder = get_output_dir() / recording_id
    if not validate_path_within_directory(output_folder, get_output_dir()):
        logger.error("path_traversal_attempt", recording_id=recording_id)
        raise HTTPException(status_code=400, detail="Invalid recording path")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Determine extension from original filename
    original_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if not original_ext:
        original_ext = ".wav"

    # Save uploaded file
    raw_audio_path = output_folder / f"audio_raw{original_ext}"
    async with aiofiles.open(raw_audio_path, "wb") as f:
        await f.write(contents)

    logger.info(
        "upload_received",
        recording_id=recording_id,
        filename=file.filename,
        size_bytes=len(contents),
        content_type=file.content_type,
        ticket_id=ticket_id,
    )

    # Queue background processing
    background_tasks.add_task(
        process_uploaded_file,
        recording_id=recording_id,
        audio_path=raw_audio_path,
        output_folder=output_folder,
        ticket_id=ticket_id,
        original_filename=file.filename,
    )

    return UploadResponse(
        recording_id=recording_id,
        status="processing",
        message=f"Processing {file.filename or 'uploaded file'}",
    )


async def process_uploaded_file(
    recording_id: str,
    audio_path: Path,
    output_folder: Path,
    ticket_id: Optional[str],
    original_filename: Optional[str],
):
    """
    Background task to process uploaded audio file.

    Reuses the existing transcription pipeline:
    1. Normalize audio to 16kHz mono WAV
    2. Run whisper.cpp transcription
    3. Create VTB bundle
    """
    from ..services.normalizer import normalize_audio
    from ..services.transcriber import transcribe_audio
    from ..services.bundler import create_vtb_bundle

    settings = get_settings()

    try:
        # Broadcast processing started
        await app_state.processing_progress(5, "Starting transcription...")

        # Step 1: Normalize audio (handles any format via FFmpeg)
        await app_state.processing_progress(10, "Normalizing audio...")
        logger.info("upload_normalizing", recording_id=recording_id)

        # The normalizer expects audio_raw.wav in the output folder
        # Copy/rename if needed
        expected_raw = output_folder / "audio_raw.wav"
        if audio_path != expected_raw and audio_path.suffix.lower() != ".wav":
            # FFmpeg will handle conversion during normalization
            pass

        normalized_path = await normalize_audio(audio_path)

        # Step 2: Transcribe with progress updates
        await app_state.processing_progress(30, "Transcribing audio...")
        logger.info("upload_transcribing", recording_id=recording_id)

        async def transcription_progress(percent: int, stage: str):
            await app_state.processing_progress(percent, stage)

        async def partial_transcript_callback(text: str, is_final: bool):
            await app_state.partial_transcript(text, is_final)

        transcript_path = await transcribe_audio(
            output_folder,
            settings,
            progress_callback=transcription_progress,
            partial_transcript_callback=partial_transcript_callback,
        )

        # Step 3: Get duration from normalized audio
        duration = 0.0
        try:
            from ..services.normalizer import get_audio_duration

            duration = get_audio_duration(normalized_path)
        except Exception as e:
            logger.warning(
                "duration_detection_failed",
                recording_id=recording_id,
                audio_path=str(normalized_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback: estimate from file size (16-bit PCM, 16kHz mono)
            try:
                file_size = normalized_path.stat().st_size
                duration = file_size / (16000 * 2)  # samples * bytes_per_sample
            except OSError:
                pass

        # Step 4: Create bundle
        await app_state.processing_progress(80, "Creating bundle...")
        logger.info("upload_bundling", recording_id=recording_id)

        # Create a mock session for the bundler
        session = RecordingSession(
            id=recording_id,
            device_name="file_upload",
            ticket_id=ticket_id,
        )
        session.start_time = datetime.now()
        session.end_time = datetime.now()
        session.output_folder = str(output_folder)

        bundle_path = None
        if settings.output.create_bundle:
            bundle_path = await create_vtb_bundle(output_folder, session, settings)

        # Read transcript preview
        transcript_preview = None
        if transcript_path and transcript_path.exists():
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript_preview = f.read()[:500]

        # Create completed recording entry
        completed = CompletedRecording(
            id=recording_id,
            ticket_id=ticket_id,
            created_at=datetime.now(),
            duration_seconds=duration,
            output_folder=str(output_folder),
            bundle_path=str(bundle_path) if bundle_path else None,
            transcript_preview=transcript_preview,
        )

        # Complete
        await app_state.processing_progress(100, "Complete")
        await app_state.complete_recording(completed)

        logger.info(
            "upload_complete",
            recording_id=recording_id,
            duration_seconds=duration,
            transcript_length=len(transcript_preview) if transcript_preview else 0,
        )

    except Exception as e:
        logger.error(
            "upload_processing_failed",
            recording_id=recording_id,
            error=str(e),
        )
        await app_state.set_error(f"Processing failed: {str(e)}")


# ============================================================================
# Batch Processing Endpoints
# ============================================================================


async def save_upload_to_temp(file: UploadFile) -> Path:
    """Save an uploaded file to a temporary location."""
    temp_dir = get_data_dir() / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    temp_path = temp_dir / f"batch_{timestamp}_{short_uuid}{ext}"

    # Save file
    contents = await file.read()
    async with aiofiles.open(temp_path, "wb") as f:
        await f.write(contents)

    return temp_path


async def process_queued_job(job: QueuedJob):
    """
    Process a single queued job - called by the queue worker.

    This reuses the existing transcription pipeline.
    """
    from ..services.normalizer import normalize_audio
    from ..services.transcriber import transcribe_audio
    from ..services.bundler import create_vtb_bundle

    settings = get_settings()
    queue = get_job_queue()

    # Create output folder
    output_dir = get_output_dir() / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio to output folder
    ext = job.audio_path.suffix.lower()
    raw_audio = output_dir / f"audio_raw{ext}"
    shutil.copy(job.audio_path, raw_audio)

    logger.info(
        "batch_job_processing",
        job_id=job.job_id,
        filename=job.original_filename,
    )

    # Step 1: Normalize audio
    await queue.update_progress(job.job_id, 10)
    normalized_path = await normalize_audio(raw_audio)

    # Step 2: Transcribe
    await queue.update_progress(job.job_id, 30)

    async def progress_cb(pct: int, stage: str):
        # Map transcription progress (30-80%)
        mapped = 30 + int((pct / 100) * 50)
        await queue.update_progress(job.job_id, mapped)

    async def partial_transcript_cb(text: str, is_final: bool):
        # Broadcast partial transcript for batch jobs too
        await app_state.partial_transcript(text, is_final)

    transcript_path = await transcribe_audio(
        output_dir,
        settings,
        progress_callback=progress_cb,
        partial_transcript_callback=partial_transcript_cb,
    )

    # Step 3: Get duration
    duration = 0.0
    try:
        from ..services.normalizer import get_audio_duration

        duration = get_audio_duration(normalized_path)
    except Exception as e:
        logger.warning(
            "duration_detection_failed",
            job_id=job.job_id,
            audio_path=str(normalized_path),
            error=str(e),
            error_type=type(e).__name__,
        )

    # Step 4: Create bundle
    await queue.update_progress(job.job_id, 85)

    session = RecordingSession(
        id=job.job_id,
        device_name="batch_upload",
        ticket_id=job.ticket_id,
    )
    session.start_time = datetime.fromtimestamp(job.created_at)
    session.end_time = datetime.now()
    session.output_folder = str(output_dir)

    bundle_path = None
    if settings.output.create_bundle:
        bundle_path = await create_vtb_bundle(output_dir, session, settings)

    # Read transcript preview
    transcript_preview = None
    if transcript_path and transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_preview = f.read()[:500]

    # Create completed recording entry
    completed = CompletedRecording(
        id=job.job_id,
        ticket_id=job.ticket_id,
        created_at=datetime.fromtimestamp(job.created_at),
        duration_seconds=duration,
        output_folder=str(output_dir),
        bundle_path=str(bundle_path) if bundle_path else None,
        transcript_preview=transcript_preview,
    )

    await queue.update_progress(job.job_id, 100)

    # Add to completed recordings list
    app_state.completed_recordings.append(completed)

    # Broadcast completion via WebSocket
    from .websocket import broadcast_state

    await broadcast_state(
        {
            "type": "recording_complete",
            "recording": {
                "id": completed.id,
                "ticket_id": completed.ticket_id,
                "created_at": completed.created_at.isoformat(),
                "duration_seconds": completed.duration_seconds,
                "output_folder": completed.output_folder,
                "bundle_path": completed.bundle_path,
                "transcript_preview": completed.transcript_preview,
            },
        }
    )

    # Cleanup temp file
    if job.audio_path.exists():
        try:
            job.audio_path.unlink()
        except Exception as e:
            logger.warning("batch_temp_cleanup_failed", job_id=job.job_id, error=str(e))

    logger.info(
        "batch_job_complete",
        job_id=job.job_id,
        duration_seconds=duration,
    )


@router.post(
    "/recordings/batch-upload",
    response_model=BatchUploadResponse,
    tags=["transcription"],
)
async def batch_upload(
    files: List[UploadFile] = File(...),
    ticket_prefix: Optional[str] = Form(None),
):
    """
    Upload multiple audio files for batch transcription.

    Files are queued and processed sequentially. Use the /api/queue/status
    endpoint to monitor progress.

    Args:
        files: List of audio files to transcribe (max 20 files)
        ticket_prefix: Optional prefix for ticket IDs (will be appended with -1, -2, etc.)

    Returns:
        List of job IDs for tracking
    """
    # Limit batch size
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch upload")

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # Validate and sanitize ticket_prefix if provided
    if ticket_prefix:
        ticket_prefix = ticket_prefix.strip()
        try:
            ticket_prefix = sanitize_path_component(ticket_prefix, max_length=50)
        except ValueError as e:
            logger.warning(
                "invalid_ticket_prefix", ticket_prefix=ticket_prefix, error=str(e)
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid ticket prefix: contains disallowed characters",
            )

    queue = get_job_queue()
    job_ids = []

    for i, file in enumerate(files):
        # Validate file type
        content_type = file.content_type
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()

        if content_type in (
            None,
            "",
            "application/octet-stream",
            "binary/octet-stream",
        ):
            if ext not in ALLOWED_EXTENSIONS:
                logger.warning(
                    "batch_file_skipped",
                    filename=filename,
                    reason="unsupported_extension",
                )
                continue
        elif content_type not in ALLOWED_AUDIO_TYPES and not content_type.startswith(
            "audio/"
        ):
            logger.warning(
                "batch_file_skipped",
                filename=filename,
                reason="unsupported_content_type",
            )
            continue

        # Save file to temp location
        temp_path = await save_upload_to_temp(file)

        # Generate ticket ID with prefix if provided
        ticket_id = f"{ticket_prefix}-{i+1}" if ticket_prefix else None

        job_id = await queue.add_job(
            audio_path=temp_path,
            original_filename=file.filename or f"file_{i+1}",
            ticket_id=ticket_id,
        )
        job_ids.append(job_id)

    if not job_ids:
        raise HTTPException(status_code=400, detail="No valid audio files in upload")

    # Ensure worker is running
    if not queue.is_worker_running():
        await queue.start_worker(process_queued_job)

    logger.info(
        "batch_upload_queued",
        files_count=len(job_ids),
        ticket_prefix=ticket_prefix,
    )

    return BatchUploadResponse(
        status="queued",
        jobs_queued=len(job_ids),
        job_ids=job_ids,
    )


@router.get("/queue/status", response_model=QueueStatusResponse, tags=["transcription"])
async def get_queue_status():
    """
    Get current job queue status.

    Returns the list of queued, processing, completed, and failed jobs.
    """
    queue = get_job_queue()
    status = queue.get_status()

    def to_job_info(job_dict: Dict[str, Any]) -> QueueJobInfo:
        return QueueJobInfo(**job_dict)

    return QueueStatusResponse(
        queued=[to_job_info(j) for j in status["queued"]],
        processing=to_job_info(status["processing"]) if status["processing"] else None,
        completed=[to_job_info(j) for j in status["completed"]],
        failed=[to_job_info(j) for j in status["failed"]],
        counts=status["counts"],
    )


@router.delete("/queue/jobs/{job_id}", tags=["transcription"])
async def cancel_queued_job(job_id: str):
    """
    Cancel a queued job.

    Cannot cancel jobs that are currently processing.
    """
    queue = get_job_queue()
    cancelled = await queue.cancel_job(job_id)

    if cancelled:
        return {"status": "cancelled", "job_id": job_id}

    # Check if it's processing
    status = queue.get_status()
    if status["processing"] and status["processing"]["job_id"] == job_id:
        raise HTTPException(
            status_code=400, detail="Cannot cancel job that is currently processing"
        )

    raise HTTPException(status_code=404, detail="Job not found in queue")


@router.post("/queue/clear-history", tags=["transcription"])
async def clear_queue_history():
    """
    Clear completed and failed job history from the queue.

    Does not affect queued or processing jobs.
    """
    queue = get_job_queue()
    await queue.clear_completed()
    return {"status": "ok", "message": "Queue history cleared"}


@router.post(
    "/queue/scan-folder", response_model=FolderScanResponse, tags=["transcription"]
)
async def scan_audio_folder(
    folder_path: str = Form(...),
    recursive: bool = Form(False),
):
    """
    Scan a folder for audio files.

    Returns statistics about found audio files. Use this to preview
    what files would be imported before running batch-import.

    Args:
        folder_path: Path to the folder to scan
        recursive: If True, scan subdirectories

    Returns:
        Statistics about audio files in the folder
    """
    path = Path(folder_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        stats = get_folder_stats(path, recursive=recursive)
        return FolderScanResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/queue/import-folder", response_model=BatchUploadResponse, tags=["transcription"]
)
async def import_folder(
    folder_path: str = Form(...),
    recursive: bool = Form(False),
    ticket_prefix: Optional[str] = Form(None),
):
    """
    Import all audio files from a folder into the queue.

    Args:
        folder_path: Path to the folder containing audio files
        recursive: If True, include files from subdirectories
        ticket_prefix: Optional prefix for ticket IDs

    Returns:
        List of job IDs for the imported files
    """
    path = Path(folder_path)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    # Validate and sanitize ticket_prefix if provided
    if ticket_prefix:
        ticket_prefix = ticket_prefix.strip()
        try:
            ticket_prefix = sanitize_path_component(ticket_prefix, max_length=50)
        except ValueError as e:
            logger.warning(
                "invalid_ticket_prefix", ticket_prefix=ticket_prefix, error=str(e)
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid ticket prefix: contains disallowed characters",
            )

    try:
        audio_files = scan_folder_paths(path, recursive=recursive)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not audio_files:
        raise HTTPException(status_code=404, detail="No audio files found in folder")

    # Limit to 100 files per import
    if len(audio_files) > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(audio_files)}). Maximum 100 files per import.",
        )

    queue = get_job_queue()
    job_ids = []

    for i, audio_path in enumerate(audio_files):
        ticket_id = f"{ticket_prefix}-{i+1}" if ticket_prefix else None

        job_id = await queue.add_job(
            audio_path=audio_path,
            original_filename=audio_path.name,
            ticket_id=ticket_id,
        )
        job_ids.append(job_id)

    # Ensure worker is running
    if not queue.is_worker_running():
        await queue.start_worker(process_queued_job)

    logger.info(
        "folder_import_queued",
        folder=str(path),
        files_count=len(job_ids),
        recursive=recursive,
    )

    return BatchUploadResponse(
        status="queued",
        jobs_queued=len(job_ids),
        job_ids=job_ids,
    )


# ============================================================================
# Call Detection Endpoints (Windows only)
# ============================================================================


class CallDetectionStatusResponse(BaseModel):
    """Call detection status response."""

    enabled: bool
    state: str
    current_call: Optional[Dict[str, Any]] = None
    monitors: Dict[str, bool] = {}
    config: Optional[Dict[str, Any]] = None
    platform_supported: bool = True


class CallDetectionConfigRequest(BaseModel):
    """Request to update call detection configuration."""

    enabled: bool = Field(..., description="Enable or disable call detection")
    target_processes: List[str] = Field(
        default=["CiscoJabber.exe"], description="Process names to monitor"
    )
    call_start_confirm_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Seconds to wait before confirming call start",
    )
    call_end_confirm_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Seconds to wait before confirming call end",
    )


class AudioSessionInfo(BaseModel):
    """Information about an active audio session."""

    process_name: str
    process_id: int
    state: str
    session_id: str


class ProcessInfo(BaseModel):
    """Information about a running process."""

    name: str
    pids: List[int]


@router.get(
    "/call-detection/status",
    response_model=CallDetectionStatusResponse,
    tags=["call-detection"],
)
async def get_call_detection_status():
    """
    Get current call detection status.

    Returns the state of the call detector, current call info if any,
    and the status of underlying monitors.
    """
    import sys

    if sys.platform != "win32":
        return CallDetectionStatusResponse(
            enabled=False,
            state="unsupported",
            platform_supported=False,
            monitors={},
            config=None,
        )

    try:
        from ..services.call_detector import get_call_detector

        detector = get_call_detector()
        status = detector.get_status()

        return CallDetectionStatusResponse(
            enabled=status["enabled"],
            state=status["state"],
            current_call=status["current_call"],
            monitors=status["monitors"],
            config=status["config"],
            platform_supported=True,
        )
    except ImportError as e:
        logger.warning("call_detection_import_error", error=str(e))
        return CallDetectionStatusResponse(
            enabled=False,
            state="unavailable",
            platform_supported=True,
            monitors={},
            config=None,
        )


@router.post("/call-detection/enable", tags=["call-detection"])
async def enable_call_detection(config: Optional[CallDetectionConfigRequest] = None):
    """
    Enable automatic call detection.

    On Windows, this starts monitoring for Cisco Jabber/Finesse calls
    using WASAPI audio session detection.

    Args:
        config: Optional configuration overrides

    Returns:
        Status of the call detector after enabling
    """
    import sys

    if sys.platform != "win32":
        raise HTTPException(
            status_code=501, detail="Call detection is only available on Windows"
        )

    try:
        from ..services.call_detector import get_call_detector, CallDetectorConfig

        detector = get_call_detector()

        # Build config from request or use defaults
        if config:
            detector_config = CallDetectorConfig(
                enabled=True,
                target_processes=config.target_processes,
                call_start_confirm_seconds=config.call_start_confirm_seconds,
                call_end_confirm_seconds=config.call_end_confirm_seconds,
            )
        else:
            settings = get_settings()
            detector_config = CallDetectorConfig(
                enabled=True,
                target_processes=settings.call_detector.target_processes,
                call_start_confirm_seconds=settings.call_detector.call_start_confirm_seconds,
                call_end_confirm_seconds=settings.call_detector.call_end_confirm_seconds,
            )

        await detector.start(detector_config)

        logger.info("call_detection_enabled", targets=detector_config.target_processes)

        return {
            "status": "enabled",
            "config": {
                "target_processes": detector_config.target_processes,
                "call_start_confirm_seconds": detector_config.call_start_confirm_seconds,
                "call_end_confirm_seconds": detector_config.call_end_confirm_seconds,
            },
        }

    except ImportError as e:
        logger.error("call_detection_import_error", error=str(e))
        raise HTTPException(
            status_code=501, detail=f"Call detection dependencies not available: {e}"
        )
    except Exception as e:
        logger.error("call_detection_enable_error", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to enable call detection: {e}"
        )


@router.post("/call-detection/disable", tags=["call-detection"])
async def disable_call_detection():
    """
    Disable automatic call detection.

    Stops all monitoring and cleans up resources.
    """
    import sys

    if sys.platform != "win32":
        return {
            "status": "disabled",
            "message": "Call detection not available on this platform",
        }

    try:
        from ..services.call_detector import get_call_detector

        detector = get_call_detector()
        await detector.stop()

        logger.info("call_detection_disabled")

        return {"status": "disabled"}

    except ImportError:
        return {"status": "disabled", "message": "Call detection not installed"}
    except Exception as e:
        logger.error("call_detection_disable_error", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Failed to disable call detection: {e}"
        )


@router.get("/call-detection/processes", tags=["call-detection"])
async def get_monitored_processes():
    """
    List currently detected target processes.

    Returns the running processes that match the target process list.
    """
    import sys

    if sys.platform != "win32":
        raise HTTPException(
            status_code=501, detail="Process monitoring is only available on Windows"
        )

    try:
        from ..services.call_detector import get_call_detector

        detector = get_call_detector()

        if not detector.is_running:
            return {"processes": [], "message": "Call detection not running"}

        # Get running processes from the process monitor
        if detector._process_monitor:
            running = detector._process_monitor.get_running_processes()
            processes = [
                {"name": name, "pids": pids} for name, pids in running.items() if pids
            ]
            return {"processes": processes}

        return {"processes": []}

    except ImportError:
        raise HTTPException(
            status_code=501, detail="Call detection dependencies not available"
        )


@router.get(
    "/call-detection/audio-sessions",
    tags=["call-detection"],
    dependencies=[Depends(require_debug_enabled)],
)
async def get_audio_sessions():
    """
    List all active audio sessions (debug endpoint).

    Returns all Windows audio sessions, not just for target processes.
    Useful for discovering which processes to monitor.

    Note: This is a debug endpoint and must be enabled in config.
    """
    import sys

    if sys.platform != "win32":
        raise HTTPException(
            status_code=501,
            detail="Audio session monitoring is only available on Windows",
        )

    try:
        from ..services.call_detector import get_call_detector

        detector = get_call_detector()

        if detector._audio_monitor:
            sessions = detector._audio_monitor.get_all_sessions()
            return {
                "sessions": [
                    {
                        "process_name": s["process_name"],
                        "process_id": s["process_id"],
                        "state": s["state"].name,
                        "session_id": s["session_id"],
                    }
                    for s in sessions
                ]
            }

        # If detector not running, create a temporary monitor
        from ..services.windows_audio_monitor import WindowsAudioSessionMonitor

        temp_monitor = WindowsAudioSessionMonitor(target_processes=[])
        sessions = temp_monitor.get_all_sessions()

        return {
            "sessions": [
                {
                    "process_name": s["process_name"],
                    "process_id": s["process_id"],
                    "state": s["state"].name,
                    "session_id": s["session_id"],
                }
                for s in sessions
            ],
            "note": "Showing all sessions (call detection not running)",
        }

    except ImportError as e:
        raise HTTPException(
            status_code=501, detail=f"Call detection dependencies not available: {e}"
        )
