"""
Tests for Health Check Module.

Tests system health validation:
- Individual check functions
- Aggregate status
- Error handling
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

from callwhisper.core.health import (
    CheckResult,
    HealthStatus,
    HealthChecker,
    get_health_checker,
    configure_health_checker,
)

# Skip Windows-specific dshow tests on Linux
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != 'win32', reason="Windows dshow format")


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_to_dict(self):
        """to_dict includes all fields."""
        result = CheckResult(
            name="test_check",
            healthy=True,
            message="All good",
            details={"key": "value"},
        )

        d = result.to_dict()

        assert d["name"] == "test_check"
        assert d["healthy"] is True
        assert d["message"] == "All good"
        assert d["details"] == {"key": "value"}

    def test_to_dict_without_details(self):
        """to_dict excludes None details."""
        result = CheckResult(name="test", healthy=False, message="Failed")

        d = result.to_dict()

        assert "details" not in d


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_to_dict(self):
        """to_dict includes all checks."""
        status = HealthStatus(
            healthy=True,
            checks=[
                CheckResult("check1", True, "OK"),
                CheckResult("check2", True, "OK"),
            ],
        )

        d = status.to_dict()

        assert d["healthy"] is True
        assert len(d["checks"]) == 2
        assert "timestamp" in d


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create HealthChecker with defaults."""
        return HealthChecker()

    @patch("subprocess.run")
    def test_check_ffmpeg_success(self, mock_run, checker):
        """check_ffmpeg succeeds when ffmpeg is available."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 5.0 Copyright (c) 2000-2022",
            stderr="",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is True
        assert "ffmpeg available" in result.message
        assert "version" in result.details

    @patch("subprocess.run")
    def test_check_ffmpeg_not_found(self, mock_run, checker):
        """check_ffmpeg fails when ffmpeg is not found."""
        mock_run.side_effect = FileNotFoundError()

        result = checker.check_ffmpeg()

        assert result.healthy is False
        assert "not found" in result.message.lower()

    @patch("subprocess.run")
    def test_check_ffmpeg_timeout(self, mock_run, checker):
        """check_ffmpeg fails on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 5)

        result = checker.check_ffmpeg()

        assert result.healthy is False
        assert "timed out" in result.message.lower()

    @patch("subprocess.run")
    def test_check_ffmpeg_error_returncode(self, mock_run, checker):
        """check_ffmpeg fails with non-zero return code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error message",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is False
        assert "error" in result.message.lower()

    def test_check_whisper_model_no_config(self, checker):
        """check_whisper_model skips when not configured."""
        checker.models_dir = None

        result = checker.check_whisper_model()

        assert result.healthy is True
        assert result.details.get("skipped") is True

    def test_check_whisper_model_dir_missing(self, tmp_path, checker):
        """check_whisper_model fails when dir is missing."""
        checker.models_dir = tmp_path / "nonexistent"

        result = checker.check_whisper_model()

        assert result.healthy is False
        assert "not found" in result.message.lower()

    def test_check_whisper_model_no_files(self, tmp_path, checker):
        """check_whisper_model fails when no .bin files exist."""
        checker.models_dir = tmp_path
        tmp_path.mkdir(exist_ok=True)

        result = checker.check_whisper_model()

        assert result.healthy is False
        assert "no" in result.message.lower() and "model" in result.message.lower()

    def test_check_whisper_model_success(self, tmp_path, checker):
        """check_whisper_model succeeds when model files exist."""
        checker.models_dir = tmp_path
        (tmp_path / "ggml-medium.en.bin").write_bytes(b"x" * 1000)

        result = checker.check_whisper_model()

        assert result.healthy is True
        assert "1 model" in result.message

    @patch("shutil.disk_usage")
    def test_check_disk_space_sufficient(self, mock_usage, checker):
        """check_disk_space succeeds with enough space."""
        # 10 GB free
        mock_usage.return_value = MagicMock(
            free=10 * 1024**3,
            total=100 * 1024**3,
            used=90 * 1024**3,
        )

        result = checker.check_disk_space()

        assert result.healthy is True
        assert "GB free" in result.message

    @patch("shutil.disk_usage")
    def test_check_disk_space_low(self, mock_usage, checker):
        """check_disk_space fails with low space."""
        checker.min_disk_gb = 5.0
        # Only 1 GB free
        mock_usage.return_value = MagicMock(
            free=1 * 1024**3,
            total=100 * 1024**3,
            used=99 * 1024**3,
        )

        result = checker.check_disk_space()

        assert result.healthy is False
        assert "low" in result.message.lower()

    @patch("psutil.virtual_memory")
    def test_check_memory_sufficient(self, mock_memory, checker):
        """check_memory succeeds with enough memory."""
        mock_memory.return_value = MagicMock(
            available=2000 * 1024 * 1024,  # 2000 MB
            total=8000 * 1024 * 1024,
            percent=75.0,
        )

        result = checker.check_memory()

        assert result.healthy is True
        assert "available" in result.message

    @patch("psutil.virtual_memory")
    def test_check_memory_low(self, mock_memory, checker):
        """check_memory fails with low memory."""
        checker.min_memory_mb = 1000.0
        mock_memory.return_value = MagicMock(
            available=200 * 1024 * 1024,  # 200 MB
            total=8000 * 1024 * 1024,
            percent=97.5,
        )

        result = checker.check_memory()

        assert result.healthy is False
        assert "low" in result.message.lower()

    def test_check_audio_device_no_device(self, checker):
        """check_audio_device skips when no device specified."""
        result = checker.check_audio_device(None)

        assert result.healthy is True
        assert result.details.get("skipped") is True

    @WINDOWS_ONLY
    @patch("subprocess.run")
    def test_check_audio_device_found(self, mock_run, checker):
        """check_audio_device succeeds when device is found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "VB-Cable" (audio)\n',
        )

        result = checker.check_audio_device("VB-Cable")

        assert result.healthy is True
        assert "found" in result.message.lower()

    @patch("subprocess.run")
    def test_check_audio_device_not_found(self, mock_run, checker):
        """check_audio_device fails when device is not found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "Microphone" (audio)\n',
        )

        result = checker.check_audio_device("VB-Cable")

        assert result.healthy is False
        assert "not found" in result.message.lower()

    def test_check_recordings_dir_creates(self, tmp_path, checker):
        """check_recordings_dir creates directory if missing."""
        new_dir = tmp_path / "recordings"

        result = checker.check_recordings_dir(new_dir)

        assert result.healthy is True
        assert new_dir.exists()

    def test_check_recordings_dir_writable(self, tmp_path, checker):
        """check_recordings_dir succeeds when writable."""
        result = checker.check_recordings_dir(tmp_path)

        assert result.healthy is True
        assert "writable" in result.message.lower()

    @pytest.mark.asyncio
    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_whisper_model")
    @patch.object(HealthChecker, "check_disk_space")
    @patch.object(HealthChecker, "check_memory")
    @patch.object(HealthChecker, "check_recordings_dir")
    async def test_run_all_checks_all_pass(
        self, mock_rec, mock_mem, mock_disk, mock_model, mock_ffmpeg, checker
    ):
        """run_all_checks returns healthy when all pass."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", True, "OK")
        mock_model.return_value = CheckResult("model", True, "OK")
        mock_disk.return_value = CheckResult("disk", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")
        mock_rec.return_value = CheckResult("recordings", True, "OK")

        status = await checker.run_all_checks()

        assert status.healthy is True
        assert len(status.checks) == 5

    @pytest.mark.asyncio
    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_whisper_model")
    @patch.object(HealthChecker, "check_disk_space")
    @patch.object(HealthChecker, "check_memory")
    @patch.object(HealthChecker, "check_recordings_dir")
    async def test_run_all_checks_one_fails(
        self, mock_rec, mock_mem, mock_disk, mock_model, mock_ffmpeg, checker
    ):
        """run_all_checks returns unhealthy when one fails."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", False, "Failed")
        mock_model.return_value = CheckResult("model", True, "OK")
        mock_disk.return_value = CheckResult("disk", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")
        mock_rec.return_value = CheckResult("recordings", True, "OK")

        status = await checker.run_all_checks()

        assert status.healthy is False

    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_memory")
    def test_run_quick_check_success(self, mock_mem, mock_ffmpeg, checker):
        """run_quick_check returns True when essential checks pass."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")

        result = checker.run_quick_check()

        assert result is True

    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_memory")
    def test_run_quick_check_failure(self, mock_mem, mock_ffmpeg, checker):
        """run_quick_check returns False when essential checks fail."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", False, "Failed")
        mock_mem.return_value = CheckResult("memory", True, "OK")

        result = checker.run_quick_check()

        assert result is False


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_health_checker_singleton(self):
        """get_health_checker returns same instance."""
        import callwhisper.core.health as health_module
        health_module._default_checker = None

        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2

        health_module._default_checker = None

    def test_configure_health_checker(self):
        """configure_health_checker creates with custom settings."""
        import callwhisper.core.health as health_module
        health_module._default_checker = None

        checker = configure_health_checker(
            ffmpeg_path="/custom/ffmpeg",
            min_disk_gb=5.0,
            min_memory_mb=1000.0,
        )

        assert checker.ffmpeg_path == "/custom/ffmpeg"
        assert checker.min_disk_gb == 5.0
        assert checker.min_memory_mb == 1000.0

        health_module._default_checker = None


# ============================================================================
# Edge Case Tests (Test Suite Expansion)
# ============================================================================


class TestFFmpegVersionParsing:
    """Tests for FFmpeg version string parsing edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    @patch("subprocess.run")
    def test_ffmpeg_prerelease_version(self, mock_run, checker):
        """Handle pre-release version strings."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version N-110234-g8d3c2e3d8e-20230915 Copyright (c) 2000-2023",
            stderr="",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is True
        assert "N-110234" in result.details["version"]

    @patch("subprocess.run")
    def test_ffmpeg_custom_build_version(self, mock_run, checker):
        """Handle custom build version strings."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 6.1-full_build-www.gyan.dev built with gcc 12.2.0",
            stderr="",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is True
        assert "6.1-full_build" in result.details["version"]

    @patch("subprocess.run")
    def test_ffmpeg_multiline_output(self, mock_run, checker):
        """Handle multi-line FFmpeg output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 5.0\nlibavutil 57.17.100\nlibavcodec 59.18.100",
            stderr="",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is True
        # Should only include first line
        assert result.details["version"] == "ffmpeg version 5.0"

    @patch("subprocess.run")
    def test_ffmpeg_empty_version(self, mock_run, checker):
        """Handle empty FFmpeg output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        result = checker.check_ffmpeg()

        assert result.healthy is True
        assert result.details["version"] == ""

    @patch("subprocess.run")
    def test_ffmpeg_permission_denied(self, mock_run, checker):
        """Handle permission denied error."""
        mock_run.side_effect = PermissionError("Permission denied")

        result = checker.check_ffmpeg()

        assert result.healthy is False
        assert "Permission denied" in result.message

    @patch("subprocess.run")
    def test_ffmpeg_stderr_truncation(self, mock_run, checker):
        """Long stderr should be truncated."""
        long_error = "X" * 500
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr=long_error,
        )

        result = checker.check_ffmpeg()

        assert result.healthy is False
        # stderr should be truncated to 200 chars
        assert len(result.details["stderr"]) <= 200


class TestWhisperModelEdgeCases:
    """Tests for whisper model detection edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    def test_multiple_model_files(self, tmp_path, checker):
        """Handle multiple .bin model files."""
        checker.models_dir = tmp_path
        (tmp_path / "ggml-small.bin").write_bytes(b"x" * 500_000)
        (tmp_path / "ggml-medium.bin").write_bytes(b"x" * 1_500_000)
        (tmp_path / "ggml-large.bin").write_bytes(b"x" * 3_000_000)

        result = checker.check_whisper_model()

        assert result.healthy is True
        assert "3 model" in result.message
        assert len(result.details["models"]) == 3

    def test_model_file_with_special_name(self, tmp_path, checker):
        """Handle model files with special characters in name."""
        checker.models_dir = tmp_path
        (tmp_path / "ggml-medium.en (v2).bin").write_bytes(b"x" * 1000)

        result = checker.check_whisper_model()

        assert result.healthy is True
        assert "1 model" in result.message

    def test_model_directory_is_file(self, tmp_path, checker):
        """Handle case where models_dir is a file, not directory."""
        file_path = tmp_path / "not_a_directory"
        file_path.write_text("I'm a file")
        checker.models_dir = file_path

        result = checker.check_whisper_model()

        assert result.healthy is False

    def test_symlink_to_models_dir(self, tmp_path, checker):
        """Handle symlink to models directory."""
        actual_dir = tmp_path / "actual_models"
        actual_dir.mkdir()
        (actual_dir / "model.bin").write_bytes(b"x" * 1000)

        symlink_dir = tmp_path / "models_link"
        symlink_dir.symlink_to(actual_dir)

        checker.models_dir = symlink_dir

        result = checker.check_whisper_model()

        assert result.healthy is True

    def test_empty_bin_file(self, tmp_path, checker):
        """Handle empty .bin file."""
        checker.models_dir = tmp_path
        (tmp_path / "empty.bin").write_bytes(b"")

        result = checker.check_whisper_model()

        assert result.healthy is True
        assert result.details["models"][0]["size_mb"] == 0.0


class TestAudioDeviceEdgeCases:
    """Tests for audio device detection edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    @WINDOWS_ONLY
    @patch("subprocess.run")
    def test_device_name_case_insensitive(self, mock_run, checker):
        """Device name matching should be case-insensitive."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "VB-CABLE Output" (audio)\n',
        )

        result = checker.check_audio_device("vb-cable output")

        assert result.healthy is True

    @WINDOWS_ONLY
    @patch("subprocess.run")
    def test_device_partial_match(self, mock_run, checker):
        """Partial device name should match."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "VB-Audio Virtual Cable" (audio)\n',
        )

        result = checker.check_audio_device("VB-Audio")

        assert result.healthy is True

    @WINDOWS_ONLY
    @patch("subprocess.run")
    def test_device_unicode_name(self, mock_run, checker):
        """Handle Unicode device names."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "マイク (Realtek)" (audio)\n',
        )

        result = checker.check_audio_device("マイク")

        assert result.healthy is True

    @WINDOWS_ONLY
    @patch("subprocess.run")
    def test_device_special_characters(self, mock_run, checker):
        """Handle special characters in device names."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr='[dshow] "Audio (Device <v2.0>)" (audio)\n',
        )

        result = checker.check_audio_device("Audio (Device <v2.0>)")

        assert result.healthy is True

    @patch("subprocess.run")
    def test_device_empty_output(self, mock_run, checker):
        """Handle empty device list output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="",
        )

        result = checker.check_audio_device("SomeDevice")

        assert result.healthy is False
        assert "not found" in result.message.lower()

    @patch("subprocess.run")
    def test_device_ffmpeg_timeout(self, mock_run, checker):
        """Handle device check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 10)

        result = checker.check_audio_device("VB-Cable")

        assert result.healthy is False
        assert "timed out" in result.message.lower() or "failed" in result.message.lower()


class TestDiskSpaceEdgeCases:
    """Tests for disk space check edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    @patch("shutil.disk_usage")
    def test_disk_space_boundary_exactly_minimum(self, mock_usage, checker):
        """Disk space exactly at minimum should pass."""
        checker.min_disk_gb = 1.0
        mock_usage.return_value = MagicMock(
            free=1 * 1024**3,  # Exactly 1 GB
            total=100 * 1024**3,
            used=99 * 1024**3,
        )

        result = checker.check_disk_space()

        assert result.healthy is True

    @patch("shutil.disk_usage")
    def test_disk_space_just_below_minimum(self, mock_usage, checker):
        """Disk space just below minimum should fail."""
        checker.min_disk_gb = 1.0
        mock_usage.return_value = MagicMock(
            free=int(0.99 * 1024**3),  # Just under 1 GB
            total=100 * 1024**3,
            used=int(99.01 * 1024**3),
        )

        result = checker.check_disk_space()

        assert result.healthy is False

    @patch("shutil.disk_usage")
    def test_disk_space_zero_free(self, mock_usage, checker):
        """Handle zero free space."""
        mock_usage.return_value = MagicMock(
            free=0,
            total=100 * 1024**3,
            used=100 * 1024**3,
        )

        result = checker.check_disk_space()

        assert result.healthy is False
        assert "0.0 GB" in result.message or "low" in result.message.lower()

    @patch("shutil.disk_usage")
    def test_disk_usage_oserror(self, mock_usage, checker):
        """Handle OS error during disk check."""
        mock_usage.side_effect = OSError("Disk not accessible")

        result = checker.check_disk_space()

        assert result.healthy is False
        assert "Disk not accessible" in result.message

    def test_disk_space_with_custom_path(self, tmp_path, checker):
        """Disk space check with custom path."""
        result = checker.check_disk_space(tmp_path)

        assert result.healthy is True  # Assuming tmp_path has space
        assert str(tmp_path) in result.details["path"]


class TestMemoryEdgeCases:
    """Tests for memory check edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    @patch("psutil.virtual_memory")
    def test_memory_exactly_minimum(self, mock_memory, checker):
        """Memory exactly at minimum should pass."""
        checker.min_memory_mb = 500.0
        mock_memory.return_value = MagicMock(
            available=500 * 1024 * 1024,  # Exactly 500 MB
            total=8000 * 1024 * 1024,
            percent=93.75,
        )

        result = checker.check_memory()

        assert result.healthy is True

    @patch("psutil.virtual_memory")
    def test_memory_psutil_exception(self, mock_memory, checker):
        """Handle psutil exception."""
        mock_memory.side_effect = Exception("Cannot read memory info")

        result = checker.check_memory()

        assert result.healthy is False
        assert "Cannot read memory" in result.message

    @patch("psutil.virtual_memory")
    def test_memory_very_low(self, mock_memory, checker):
        """Handle very low memory."""
        checker.min_memory_mb = 500.0
        mock_memory.return_value = MagicMock(
            available=10 * 1024 * 1024,  # Only 10 MB
            total=8000 * 1024 * 1024,
            percent=99.9,
        )

        result = checker.check_memory()

        assert result.healthy is False
        assert "10 MB" in result.message


class TestRecordingsDirEdgeCases:
    """Tests for recordings directory check edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    def test_recordings_dir_symlink(self, tmp_path, checker):
        """Handle symlink for recordings directory."""
        actual_dir = tmp_path / "actual_recordings"
        actual_dir.mkdir()

        symlink_dir = tmp_path / "recordings_link"
        symlink_dir.symlink_to(actual_dir)

        result = checker.check_recordings_dir(symlink_dir)

        assert result.healthy is True

    def test_recordings_dir_nested_creation(self, tmp_path, checker):
        """Create deeply nested recordings directory."""
        nested_dir = tmp_path / "a" / "b" / "c" / "recordings"

        result = checker.check_recordings_dir(nested_dir)

        assert result.healthy is True
        assert nested_dir.exists()

    def test_recordings_dir_already_exists(self, tmp_path, checker):
        """Existing directory should work."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        result = checker.check_recordings_dir(existing_dir)

        assert result.healthy is True

    def test_recordings_dir_special_chars(self, tmp_path, checker):
        """Handle directory with special characters."""
        special_dir = tmp_path / "recordings (v2.0)"

        result = checker.check_recordings_dir(special_dir)

        assert result.healthy is True


class TestRunAllChecksEdgeCases:
    """Tests for run_all_checks edge cases."""

    @pytest.fixture
    def checker(self):
        return HealthChecker()

    @pytest.mark.asyncio
    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_whisper_model")
    @patch.object(HealthChecker, "check_disk_space")
    @patch.object(HealthChecker, "check_memory")
    @patch.object(HealthChecker, "check_recordings_dir")
    @patch.object(HealthChecker, "check_audio_device")
    async def test_run_all_checks_with_device(
        self, mock_device, mock_rec, mock_mem, mock_disk, mock_model, mock_ffmpeg, checker
    ):
        """run_all_checks includes device check when device specified."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", True, "OK")
        mock_model.return_value = CheckResult("model", True, "OK")
        mock_disk.return_value = CheckResult("disk", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")
        mock_rec.return_value = CheckResult("recordings", True, "OK")
        mock_device.return_value = CheckResult("audio_device", True, "OK")

        status = await checker.run_all_checks(device_name="VB-Cable")

        assert status.healthy is True
        assert len(status.checks) == 6
        mock_device.assert_called_once_with("VB-Cable")

    @pytest.mark.asyncio
    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_whisper_model")
    @patch.object(HealthChecker, "check_disk_space")
    @patch.object(HealthChecker, "check_memory")
    @patch.object(HealthChecker, "check_recordings_dir")
    async def test_run_all_checks_skipped_checks_healthy(
        self, mock_rec, mock_mem, mock_disk, mock_model, mock_ffmpeg, checker
    ):
        """Skipped checks should not affect overall health."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", True, "OK")
        mock_model.return_value = CheckResult("model", True, "Skipped", {"skipped": True})
        mock_disk.return_value = CheckResult("disk", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")
        mock_rec.return_value = CheckResult("recordings", True, "OK")

        status = await checker.run_all_checks()

        assert status.healthy is True

    @pytest.mark.asyncio
    @patch.object(HealthChecker, "check_ffmpeg")
    @patch.object(HealthChecker, "check_whisper_model")
    @patch.object(HealthChecker, "check_disk_space")
    @patch.object(HealthChecker, "check_memory")
    @patch.object(HealthChecker, "check_recordings_dir")
    async def test_run_all_checks_has_timestamp(
        self, mock_rec, mock_mem, mock_disk, mock_model, mock_ffmpeg, checker
    ):
        """HealthStatus should include timestamp."""
        mock_ffmpeg.return_value = CheckResult("ffmpeg", True, "OK")
        mock_model.return_value = CheckResult("model", True, "OK")
        mock_disk.return_value = CheckResult("disk", True, "OK")
        mock_mem.return_value = CheckResult("memory", True, "OK")
        mock_rec.return_value = CheckResult("recordings", True, "OK")

        status = await checker.run_all_checks()

        assert status.timestamp > 0
        dict_result = status.to_dict()
        assert "timestamp" in dict_result


class TestHealthStatusSerialization:
    """Tests for HealthStatus serialization edge cases."""

    def test_health_status_with_empty_checks(self):
        """HealthStatus with no checks."""
        status = HealthStatus(healthy=True, checks=[])

        d = status.to_dict()

        assert d["healthy"] is True
        assert d["checks"] == []

    def test_health_status_with_none_details(self):
        """HealthStatus with checks that have no details."""
        status = HealthStatus(
            healthy=True,
            checks=[CheckResult("test", True, "OK", None)],
        )

        d = status.to_dict()

        assert "details" not in d["checks"][0]

    def test_check_result_with_complex_details(self):
        """CheckResult with nested details."""
        result = CheckResult(
            name="complex",
            healthy=True,
            message="OK",
            details={
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "number": 42,
            },
        )

        d = result.to_dict()

        assert d["details"]["nested"]["key"] == "value"
        assert d["details"]["list"] == [1, 2, 3]
