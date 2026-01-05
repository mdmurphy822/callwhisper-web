"""
Health Check Module

Based on LibV2 orchestrator-architecture course:
- Pre-flight checks before operations
- Readiness probes for service availability
- Graceful handling of missing dependencies

Supports Windows (DirectShow) and Linux (PulseAudio/ALSA).
"""

import time
import shutil
import psutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .logging_config import get_logger
from ..utils.platform import get_audio_backend

logger = get_logger(__name__)


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    healthy: bool
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "healthy": self.healthy, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class HealthStatus:
    """Aggregate health status from all checks."""

    healthy: bool
    checks: List[CheckResult]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """
    System health checker for pre-recording validation.

    Verifies that all required dependencies and resources
    are available before starting a recording session.
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        whisper_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        min_disk_gb: float = 1.0,
        min_memory_mb: float = 500.0,
    ):
        """
        Initialize health checker.

        Args:
            ffmpeg_path: Path to ffmpeg executable
            whisper_path: Path to whisper.cpp executable (optional)
            models_dir: Path to models directory
            min_disk_gb: Minimum required disk space in GB
            min_memory_mb: Minimum required available memory in MB
        """
        self.ffmpeg_path = ffmpeg_path
        self.whisper_path = whisper_path
        self.models_dir = models_dir
        self.min_disk_gb = min_disk_gb
        self.min_memory_mb = min_memory_mb

    def check_ffmpeg(self) -> CheckResult:
        """Verify ffmpeg executable is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split("\n")[0]
                return CheckResult(
                    name="ffmpeg",
                    healthy=True,
                    message="ffmpeg available",
                    details={"version": version_line},
                )
            else:
                return CheckResult(
                    name="ffmpeg",
                    healthy=False,
                    message="ffmpeg returned error",
                    details={
                        "stderr": result.stderr[:200],
                        "expected_path": "vendor/ffmpeg.exe",
                        "download_url": "https://github.com/BtbN/FFmpeg-Builds/releases",
                        "instructions": "Run scripts/download-vendor.ps1 or download manually",
                    },
                )

        except FileNotFoundError:
            return CheckResult(
                name="ffmpeg",
                healthy=False,
                message="FFmpeg not found - required for audio processing",
                details={
                    "path": self.ffmpeg_path,
                    "expected_path": "vendor/ffmpeg.exe",
                    "download_url": "https://github.com/BtbN/FFmpeg-Builds/releases",
                    "instructions": "Run scripts/download-vendor.ps1 or download manually",
                },
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                name="ffmpeg",
                healthy=False,
                message="ffmpeg check timed out",
                details={
                    "expected_path": "vendor/ffmpeg.exe",
                    "download_url": "https://github.com/BtbN/FFmpeg-Builds/releases",
                },
            )
        except Exception as e:
            return CheckResult(
                name="ffmpeg",
                healthy=False,
                message=f"ffmpeg check failed: {str(e)}",
                details={
                    "expected_path": "vendor/ffmpeg.exe",
                    "download_url": "https://github.com/BtbN/FFmpeg-Builds/releases",
                },
            )

    def check_whisper_model(self) -> CheckResult:
        """Verify whisper model file exists."""
        if not self.models_dir:
            return CheckResult(
                name="whisper_model",
                healthy=True,
                message="Models directory not configured (skipped)",
                details={"skipped": True},
            )

        models_path = Path(self.models_dir)

        if not models_path.exists():
            return CheckResult(
                name="whisper_model",
                healthy=False,
                message="Models directory not found",
                details={
                    "path": str(models_path),
                    "expected_path": "models/",
                    "download_url": "https://huggingface.co/ggerganov/whisper.cpp/tree/main",
                    "recommended_model": "ggml-medium.en.bin",
                    "instructions": "Run scripts/download-vendor.ps1 or download ggml-medium.en.bin manually",
                },
            )

        # Look for .bin model files
        model_files = list(models_path.glob("*.bin"))

        if not model_files:
            return CheckResult(
                name="whisper_model",
                healthy=False,
                message="No transcription model found - required for speech-to-text",
                details={
                    "path": str(models_path),
                    "expected_path": "models/ggml-medium.en.bin",
                    "download_url": "https://huggingface.co/ggerganov/whisper.cpp/tree/main",
                    "recommended_model": "ggml-medium.en.bin (1.5 GB)",
                    "alternative_models": [
                        "ggml-small.en.bin (466 MB) - faster, less accurate",
                        "ggml-large-v3.bin (3 GB) - slower, more accurate",
                    ],
                    "instructions": "Run scripts/download-vendor.ps1 or download manually",
                },
            )

        # Report found models
        model_info = [
            {"name": m.name, "size_mb": round(m.stat().st_size / (1024 * 1024), 1)}
            for m in model_files
        ]

        return CheckResult(
            name="whisper_model",
            healthy=True,
            message=f"Found {len(model_files)} model(s)",
            details={"models": model_info},
        )

    def check_disk_space(self, path: Optional[Path] = None) -> CheckResult:
        """Ensure sufficient disk space for recording."""
        check_path = path or Path.home()

        try:
            usage = shutil.disk_usage(check_path)
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_percent = (usage.used / usage.total) * 100

            healthy = free_gb >= self.min_disk_gb

            return CheckResult(
                name="disk_space",
                healthy=healthy,
                message=(
                    f"{free_gb:.1f} GB free"
                    if healthy
                    else f"Low disk space: {free_gb:.1f} GB"
                ),
                details={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 1),
                    "min_required_gb": self.min_disk_gb,
                    "path": str(check_path),
                },
            )

        except Exception as e:
            return CheckResult(
                name="disk_space", healthy=False, message=f"Disk check failed: {str(e)}"
            )

    def check_memory(self) -> CheckResult:
        """Ensure sufficient memory available."""
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            total_mb = memory.total / (1024 * 1024)
            used_percent = memory.percent

            healthy = available_mb >= self.min_memory_mb

            return CheckResult(
                name="memory",
                healthy=healthy,
                message=(
                    f"{available_mb:.0f} MB available"
                    if healthy
                    else f"Low memory: {available_mb:.0f} MB"
                ),
                details={
                    "available_mb": round(available_mb, 0),
                    "total_mb": round(total_mb, 0),
                    "used_percent": round(used_percent, 1),
                    "min_required_mb": self.min_memory_mb,
                },
            )

        except Exception as e:
            return CheckResult(
                name="memory", healthy=False, message=f"Memory check failed: {str(e)}"
            )

    def check_audio_device(self, device_name: Optional[str] = None) -> CheckResult:
        """
        Verify audio device is available.

        Note: This is a basic check. Full device validation happens
        when actually attempting to record.

        Supports Windows (DirectShow), Linux (PulseAudio/ALSA).
        """
        if not device_name:
            return CheckResult(
                name="audio_device",
                healthy=True,
                message="No device specified (skipped)",
                details={"skipped": True},
            )

        backend = get_audio_backend()

        try:
            if backend == "dshow":
                # Windows: use ffmpeg to list DirectShow devices
                result = subprocess.run(
                    [
                        self.ffmpeg_path,
                        "-list_devices",
                        "true",
                        "-f",
                        "dshow",
                        "-i",
                        "dummy",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = result.stderr
            elif backend == "pulse":
                # Linux PulseAudio: use pactl
                result = subprocess.run(
                    ["pactl", "list", "sources", "short"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                output = result.stdout
            elif backend == "alsa":
                # Linux ALSA: use arecord
                result = subprocess.run(
                    ["arecord", "-l"], capture_output=True, text=True, timeout=10
                )
                output = result.stdout
            else:
                return CheckResult(
                    name="audio_device",
                    healthy=False,
                    message=f"Unknown audio backend: {backend}",
                    details={"backend": backend},
                )

            if device_name.lower() in output.lower():
                return CheckResult(
                    name="audio_device",
                    healthy=True,
                    message=f"Device '{device_name}' found",
                    details={"device": device_name, "backend": backend},
                )
            else:
                return CheckResult(
                    name="audio_device",
                    healthy=False,
                    message=f"Device '{device_name}' not found",
                    details={"device": device_name, "backend": backend},
                )

        except FileNotFoundError as e:
            tool = (
                "pactl"
                if backend == "pulse"
                else "arecord" if backend == "alsa" else "ffmpeg"
            )
            return CheckResult(
                name="audio_device",
                healthy=False,
                message=f"Audio tool '{tool}' not found",
                details={"backend": backend, "error": str(e)},
            )
        except Exception as e:
            return CheckResult(
                name="audio_device",
                healthy=False,
                message=f"Device check failed: {str(e)}",
                details={"backend": backend},
            )

    def check_recordings_dir(
        self, recordings_dir: Optional[Path] = None
    ) -> CheckResult:
        """Verify recordings directory exists and is writable."""
        if not recordings_dir:
            recordings_dir = Path.home() / ".callwhisper" / "recordings"

        try:
            # Create if doesn't exist
            recordings_dir.mkdir(parents=True, exist_ok=True)

            # Test write permission
            test_file = recordings_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            return CheckResult(
                name="recordings_dir",
                healthy=True,
                message="Recordings directory writable",
                details={"path": str(recordings_dir)},
            )

        except PermissionError:
            return CheckResult(
                name="recordings_dir",
                healthy=False,
                message="Recordings directory not writable",
                details={"path": str(recordings_dir)},
            )
        except Exception as e:
            return CheckResult(
                name="recordings_dir",
                healthy=False,
                message=f"Directory check failed: {str(e)}",
                details={"path": str(recordings_dir)},
            )

    async def run_all_checks(
        self, device_name: Optional[str] = None, recordings_dir: Optional[Path] = None
    ) -> HealthStatus:
        """
        Run all health checks and return aggregate status.

        Args:
            device_name: Optional audio device to verify
            recordings_dir: Optional recordings directory to check

        Returns:
            HealthStatus with all check results
        """
        logger.debug("running_health_checks")

        checks = [
            self.check_ffmpeg(),
            self.check_whisper_model(),
            self.check_disk_space(recordings_dir),
            self.check_memory(),
            self.check_recordings_dir(recordings_dir),
        ]

        # Only check audio device if specified
        if device_name:
            checks.append(self.check_audio_device(device_name))

        # Overall health is True only if all non-skipped checks pass
        healthy = all(
            c.healthy or (c.details and c.details.get("skipped")) for c in checks
        )

        status = HealthStatus(healthy=healthy, checks=checks)

        logger.info(
            "health_check_complete",
            healthy=healthy,
            checks_passed=sum(1 for c in checks if c.healthy),
            checks_total=len(checks),
        )

        return status

    def run_quick_check(self) -> bool:
        """
        Run minimal health check for fast validation.

        Returns:
            True if system appears ready, False otherwise
        """
        try:
            # Just check ffmpeg and memory
            ffmpeg_ok = self.check_ffmpeg().healthy
            memory_ok = self.check_memory().healthy
            return ffmpeg_ok and memory_ok
        except Exception as e:
            logger.warning(
                "quick_health_check_failed", error=str(e), error_type=type(e).__name__
            )
            return False


# Default health checker instance
_default_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create default health checker instance."""
    global _default_checker
    if _default_checker is None:
        _default_checker = HealthChecker()
    return _default_checker


def configure_health_checker(
    ffmpeg_path: str = "ffmpeg",
    models_dir: Optional[Path] = None,
    min_disk_gb: float = 1.0,
    min_memory_mb: float = 500.0,
) -> HealthChecker:
    """Configure the default health checker."""
    global _default_checker
    _default_checker = HealthChecker(
        ffmpeg_path=ffmpeg_path,
        models_dir=models_dir,
        min_disk_gb=min_disk_gb,
        min_memory_mb=min_memory_mb,
    )
    return _default_checker
