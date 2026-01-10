"""
Tests for device guard service - microphone protection.

Verifies that:
- Microphone devices are blocked
- Virtual audio cables are allowed
- Allowlist/blocklist configuration works
- Default blocking behavior is fail-safe
"""

import sys
import pytest
from unittest.mock import MagicMock

from callwhisper.services.device_guard import (
    is_device_safe,
    get_device_status,
    validate_device_for_recording,
    get_safe_devices,
    log_device_decision,
)
from callwhisper.core.config import DeviceGuardConfig

# Skip Windows-specific device tests on Linux
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific device names")


@pytest.fixture
def default_config():
    """Create default device guard config."""
    return DeviceGuardConfig()


@pytest.fixture
def disabled_config():
    """Create disabled device guard config."""
    return DeviceGuardConfig(enabled=False)


@pytest.fixture
def custom_config():
    """Create custom device guard config with specific lists."""
    return DeviceGuardConfig(
        enabled=True,
        allowlist=["Custom Audio Device"],
        blocklist=["Dangerous Device"]
    )


class TestIsDeviceSafe:
    """Tests for is_device_safe function."""

    def test_blocks_microphone(self, default_config):
        """Microphone devices are blocked."""
        assert is_device_safe("Microphone (Realtek Audio)", default_config) is False

    def test_blocks_mic(self, default_config):
        """Mic abbreviation is blocked."""
        assert is_device_safe("Internal Mic", default_config) is False

    def test_blocks_webcam(self, default_config):
        """Webcam audio is blocked."""
        assert is_device_safe("HD Webcam C270", default_config) is False

    @WINDOWS_ONLY
    def test_allows_stereo_mix(self, default_config):
        """Stereo Mix is allowed."""
        assert is_device_safe("Stereo Mix (Realtek Audio)", default_config) is True

    @WINDOWS_ONLY
    def test_allows_vb_cable(self, default_config):
        """VB-Cable is allowed."""
        assert is_device_safe("VB-Cable Output", default_config) is True

    @WINDOWS_ONLY
    def test_allows_voicemeeter(self, default_config):
        """VoiceMeeter is allowed."""
        assert is_device_safe("VoiceMeeter Input", default_config) is True

    def test_blocks_unknown_device(self, default_config):
        """Unknown devices are blocked by default (fail-safe)."""
        assert is_device_safe("Random Unknown Device XYZ", default_config) is False

    def test_disabled_allows_all(self, disabled_config):
        """Disabled guard allows all devices."""
        assert is_device_safe("Microphone (Realtek Audio)", disabled_config) is True

    def test_custom_allowlist(self, custom_config):
        """Custom allowlist entries work."""
        assert is_device_safe("Custom Audio Device", custom_config) is True

    def test_custom_blocklist(self, custom_config):
        """Custom blocklist entries work."""
        assert is_device_safe("Dangerous Device", custom_config) is False


class TestGetDeviceStatus:
    """Tests for get_device_status function."""

    def test_status_includes_reason(self, default_config):
        """Status includes reason for decision."""
        status = get_device_status("Microphone", default_config)
        assert "reason" in status
        assert "safe" in status
        assert status["safe"] is False

    def test_status_match_type_blocklist(self):
        """Status shows blocklist match type."""
        config = DeviceGuardConfig(blocklist=["CustomBlock"])
        status = get_device_status("CustomBlock Device", config)
        assert status["match_type"] == "blocklist"

    def test_status_match_type_allowlist(self):
        """Status shows allowlist match type."""
        config = DeviceGuardConfig(allowlist=["CustomAllow"])
        status = get_device_status("CustomAllow Device", config)
        assert status["match_type"] == "allowlist"

    @WINDOWS_ONLY
    def test_status_match_type_safe_pattern(self):
        """Status shows safe pattern match type."""
        # Use empty allowlist to test safe_pattern matching specifically
        config = DeviceGuardConfig(allowlist=[])
        status = get_device_status("Stereo Mix", config)
        assert status["match_type"] == "safe_pattern"

    def test_status_match_type_disabled(self, disabled_config):
        """Status shows disabled match type when guard off."""
        status = get_device_status("Microphone", disabled_config)
        assert status["match_type"] == "disabled"
        assert status["safe"] is True


class TestValidateDeviceForRecording:
    """Tests for validate_device_for_recording function."""

    def test_raises_for_unsafe_device(self, default_config):
        """Raises ValueError for unsafe devices."""
        with pytest.raises(ValueError) as exc_info:
            validate_device_for_recording("Microphone", default_config)
        assert "BLOCKED" in str(exc_info.value)

    @WINDOWS_ONLY
    def test_no_raise_for_safe_device(self, default_config):
        """Does not raise for safe devices."""
        # Should not raise
        validate_device_for_recording("Stereo Mix", default_config)


class TestGetSafeDevices:
    """Tests for get_safe_devices function."""

    @WINDOWS_ONLY
    def test_filters_unsafe_devices(self, default_config):
        """Filters out unsafe devices from list."""
        devices = [
            "Microphone (Realtek)",
            "Stereo Mix (Realtek)",
            "VB-Cable Output",
            "Webcam Audio",
        ]
        safe = get_safe_devices(devices, default_config)

        assert len(safe) == 2
        assert "Stereo Mix (Realtek)" in safe
        assert "VB-Cable Output" in safe
        assert "Microphone (Realtek)" not in safe

    def test_empty_list(self, default_config):
        """Handles empty device list."""
        assert get_safe_devices([], default_config) == []


class TestLogDeviceDecision:
    """Tests for log_device_decision function."""

    @WINDOWS_ONLY
    def test_log_format_allowed(self, default_config):
        """Log format for allowed device."""
        log = log_device_decision("Stereo Mix", default_config)
        assert "ALLOWED" in log
        assert "Stereo Mix" in log

    def test_log_format_blocked(self, default_config):
        """Log format for blocked device."""
        log = log_device_decision("Microphone", default_config)
        assert "BLOCKED" in log
        assert "Microphone" in log
