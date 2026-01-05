"""Configuration management for CallWhisper."""

import sys
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from ..utils.paths import get_data_dir


def _get_default_allowlist() -> List[str]:
    """Get platform-specific default allowlist for device guard."""
    if sys.platform == "win32":
        return [
            "VB-Cable",
            "CABLE Output",
            "Virtual Cable",
            "Stereo Mix",
            "What U Hear",
            "Jabber",
            "Finesse",
        ]
    else:
        # Linux: PulseAudio/PipeWire monitor sinks
        return [
            "Monitor",
            "monitor",
            "Loopback",
            "loopback",
            "null",
            "pipewire",
        ]


def _get_default_blocklist() -> List[str]:
    """Get platform-specific default blocklist for device guard."""
    # Blocklist is the same for all platforms - these are input devices
    return [
        "Microphone",
        "Mic",
        "Webcam",
        "Camera",
        "Headset",
    ]


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = 44100
    channels: int = 2
    format: str = "pcm_s16le"


class TranscriptionConfig(BaseModel):
    """Transcription configuration."""

    model: str = "ggml-medium.en.bin"
    language: str = "en"
    beam_size: int = 5
    best_of: int = 5


class OutputConfig(BaseModel):
    """Output configuration."""

    directory: str = "output"
    create_bundle: bool = True
    audio_format: str = "opus"


class DeviceGuardConfig(BaseModel):
    """Device guard configuration for mic protection."""

    enabled: bool = True
    allowlist: List[str] = None  # Will use platform-specific defaults
    blocklist: List[str] = None  # Will use platform-specific defaults

    def __init__(self, **data):
        # Set platform-specific defaults if not provided
        if data.get("allowlist") is None:
            data["allowlist"] = _get_default_allowlist()
        if data.get("blocklist") is None:
            data["blocklist"] = _get_default_blocklist()
        super().__init__(**data)


class SecurityConfig(BaseModel):
    """Security configuration."""

    # CORS settings
    cors_enabled: bool = True
    allowed_origins: List[str] = ["http://localhost:8765", "http://127.0.0.1:8765"]
    allow_credentials: bool = True

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_rpm: int = 60  # Requests per minute
    rate_limit_burst: int = 10

    # Excluded paths from rate limiting
    rate_limit_excluded: List[str] = [
        "/api/health",
        "/api/health/ready",
        "/api/health/metrics",
    ]

    # Debug endpoints - DISABLE IN PRODUCTION
    # When False, /api/debug/* endpoints return 404
    debug_endpoints_enabled: bool = False


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    # Parallel processing
    max_concurrent_transcriptions: int = 4
    chunk_size_seconds: float = 30.0

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_max_entries: int = 100

    # Bulkhead pool sizes
    audio_pool_size: int = 2
    transcription_pool_size: int = 2
    io_pool_size: int = 4


class TimeoutConfig(BaseModel):
    """Timeout configuration for various operations."""

    # Device enumeration
    device_enumeration_seconds: float = 10.0

    # Transcription timeouts
    transcription_min_seconds: float = 30.0
    transcription_max_seconds: float = 600.0
    transcription_ratio: float = 3.0  # Multiplier of audio duration

    # Health checks
    health_check_seconds: float = 5.0

    # Audio normalization
    normalization_seconds: float = 120.0


class CallDetectorConfig(BaseModel):
    """
    Configuration for automatic call detection (Windows only).

    Uses WASAPI audio session monitoring to detect when Cisco Jabber
    or Finesse calls start/stop, then triggers recording automatically.
    """

    # Feature toggle - disabled by default
    enabled: bool = False

    # Target processes to monitor for audio sessions
    target_processes: List[str] = ["CiscoJabber.exe"]

    # Browser processes for Cisco Finesse (web-based softphone)
    finesse_browsers: List[str] = ["chrome.exe", "msedge.exe", "firefox.exe"]
    finesse_url_pattern: str = "finesse"  # Match in window title

    # Timing: debounce periods to avoid false triggers
    call_start_confirm_seconds: float = 1.0  # Confirm audio active for 1s
    call_end_confirm_seconds: float = 2.0  # Confirm audio inactive for 2s
    audio_poll_interval: float = 0.5  # How often to poll audio sessions
    process_poll_interval: float = 2.0  # How often to poll processes

    # Safety limits
    max_call_duration_minutes: int = 180  # Auto-stop after 3 hours
    min_call_duration_seconds: int = 5  # Discard calls shorter than 5s


class Settings(BaseModel):
    """Application settings."""

    version: str = "1.0.0"
    server: ServerConfig = ServerConfig()
    audio: AudioConfig = AudioConfig()
    transcription: TranscriptionConfig = TranscriptionConfig()
    output: OutputConfig = OutputConfig()
    device_guard: DeviceGuardConfig = DeviceGuardConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    timeouts: TimeoutConfig = TimeoutConfig()
    call_detector: CallDetectorConfig = CallDetectorConfig()


def load_config_file() -> dict:
    """Load configuration from config.json file."""
    config_path = get_data_dir() / "config" / "config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@lru_cache
def get_settings() -> Settings:
    """Get application settings (cached)."""
    config_data = load_config_file()
    return Settings(**config_data)


def reload_settings() -> Settings:
    """Reload settings from disk (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
