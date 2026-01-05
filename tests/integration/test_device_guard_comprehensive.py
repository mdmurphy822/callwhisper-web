"""
Comprehensive Device Guard Integration Tests

SAFETY-CRITICAL TESTS for microphone prevention.

These tests verify that the device guard:
1. Blocks all microphone/input devices
2. Allows approved virtual audio cables
3. Integrates correctly with the API
4. Handles edge cases and unicode device names
5. Maintains fail-safe behavior
"""

import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Platform markers for Windows-specific tests
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific device names")
from httpx import AsyncClient, ASGITransport

from callwhisper.services.device_guard import (
    is_device_safe,
    get_device_status,
    validate_device_for_recording,
    get_safe_devices,
    log_device_decision,
    DEFAULT_SAFE_PATTERNS,
    DANGEROUS_PATTERNS,
)
from callwhisper.core.config import DeviceGuardConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config():
    """Default device guard config with standard lists."""
    return DeviceGuardConfig()


@pytest.fixture
def strict_config():
    """Strict config with minimal allowlist."""
    return DeviceGuardConfig(
        enabled=True,
        allowlist=["VB-Cable"],
        blocklist=["Microphone", "Mic", "Webcam", "Camera", "Headset"],
    )


@pytest.fixture
def relaxed_config():
    """Relaxed config with expanded allowlist."""
    return DeviceGuardConfig(
        enabled=True,
        allowlist=[
            "VB-Cable",
            "Stereo Mix",
            "VoiceMeeter",
            "Custom Safe Device",
            "Test Output",
        ],
        blocklist=["Microphone"],
    )


# ============================================================================
# Microphone Blocking Tests (SAFETY-CRITICAL)
# ============================================================================


@pytest.mark.integration
class TestMicrophoneBlocking:
    """Verify all microphone variants are blocked."""

    @pytest.mark.parametrize("device_name", [
        "Microphone",
        "Microphone (Realtek Audio)",
        "Microphone (Realtek High Definition Audio)",
        "Internal Microphone",
        "Built-in Microphone",
        "Microphone Array",
        "Microphone Array (Intel SST)",
        "Front Microphone",
        "Rear Microphone",
        "USB Microphone",
        "Blue Yeti Microphone",
        "AT2020 USB Microphone",
    ])
    def test_blocks_microphone_variants(self, default_config, device_name):
        """All microphone name variants must be blocked."""
        assert is_device_safe(device_name, default_config) is False, \
            f"SAFETY FAILURE: Device '{device_name}' was not blocked!"

    @pytest.mark.parametrize("device_name", [
        "Mic",
        "Internal Mic",
        "Built-in Mic",
        "Front Mic",
        "USB Mic",
        "Headset Mic",
        "Array Mic",
    ])
    def test_blocks_mic_abbreviations(self, default_config, device_name):
        """All 'Mic' abbreviations must be blocked."""
        assert is_device_safe(device_name, default_config) is False, \
            f"SAFETY FAILURE: Device '{device_name}' was not blocked!"

    @pytest.mark.parametrize("device_name", [
        "Webcam",
        "HD Webcam C270",
        "Logitech Webcam C920",
        "USB Video Webcam",
        "Integrated Webcam",
    ])
    def test_blocks_webcam_audio(self, default_config, device_name):
        """Webcam audio must be blocked."""
        assert is_device_safe(device_name, default_config) is False, \
            f"SAFETY FAILURE: Device '{device_name}' was not blocked!"

    @pytest.mark.parametrize("device_name", [
        "Camera",
        "HD Camera",
        "USB Camera",
        "Integrated Camera",
        "Camera Audio",
    ])
    def test_blocks_camera_audio(self, default_config, device_name):
        """Camera audio must be blocked."""
        assert is_device_safe(device_name, default_config) is False, \
            f"SAFETY FAILURE: Device '{device_name}' was not blocked!"

    @pytest.mark.parametrize("device_name", [
        "Input Device",
        "Recording Device",
        "Realtek Audio Input",
    ])
    def test_blocks_input_devices(self, default_config, device_name):
        """Input/recording devices must be blocked."""
        assert is_device_safe(device_name, default_config) is False, \
            f"SAFETY FAILURE: Device '{device_name}' was not blocked!"


# ============================================================================
# Allowed Device Tests
# ============================================================================


@pytest.mark.integration
@WINDOWS_ONLY
class TestAllowedDevices:
    """Verify approved devices are allowed (Windows-specific device names)."""

    @pytest.mark.parametrize("device_name", [
        "VB-Cable Output",
        "CABLE Output (VB-Audio)",
        "VB-Audio Virtual Cable",
        "VB Cable",
        "VBCable",
    ])
    def test_allows_vb_cable_variants(self, default_config, device_name):
        """VB-Cable variants must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"

    @pytest.mark.parametrize("device_name", [
        "Stereo Mix",
        "Stereo Mix (Realtek Audio)",
        "Stereo Mix (Realtek High Definition Audio)",
    ])
    def test_allows_stereo_mix_variants(self, default_config, device_name):
        """Stereo Mix variants must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"

    @pytest.mark.parametrize("device_name", [
        "What U Hear",
        "What U Hear (Realtek)",
        "What U Hear (Creative)",
    ])
    def test_allows_what_u_hear(self, default_config, device_name):
        """'What U Hear' must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"

    @pytest.mark.parametrize("device_name", [
        "VoiceMeeter",
        "VoiceMeeter Input",
        "VoiceMeeter Output",
        "VoiceMeeter Aux",
        "VoiceMeeter VAIO",
    ])
    def test_allows_voicemeeter(self, default_config, device_name):
        """VoiceMeeter variants must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"

    @pytest.mark.parametrize("device_name", [
        "Jabber",
        "Cisco Jabber",
        "Cisco Jabber (Audio)",
        "Finesse",
        "Cisco Finesse",
    ])
    def test_allows_cisco_devices(self, default_config, device_name):
        """Cisco call center devices must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"

    @pytest.mark.parametrize("device_name", [
        "Loopback",
        "Audio Loopback",
        "Loopback Device",
    ])
    def test_allows_loopback_devices(self, default_config, device_name):
        """Loopback devices must be allowed."""
        assert is_device_safe(device_name, default_config) is True, \
            f"Device '{device_name}' should be allowed!"


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_device_name(self, default_config):
        """Empty device name should be blocked."""
        assert is_device_safe("", default_config) is False

    def test_whitespace_device_name(self, default_config):
        """Whitespace-only device name should be blocked."""
        assert is_device_safe("   ", default_config) is False

    def test_case_insensitive_blocking(self, default_config):
        """Blocking should be case-insensitive."""
        assert is_device_safe("MICROPHONE", default_config) is False
        assert is_device_safe("microphone", default_config) is False
        assert is_device_safe("MiCrOpHoNe", default_config) is False

    @WINDOWS_ONLY
    def test_case_insensitive_allowing(self, default_config):
        """Allowing should be case-insensitive (Windows-specific device names)."""
        assert is_device_safe("STEREO MIX", default_config) is True
        assert is_device_safe("stereo mix", default_config) is True
        assert is_device_safe("Stereo Mix", default_config) is True

    @WINDOWS_ONLY
    def test_microphone_output_allowed(self, default_config):
        """Microphone Output (loopback) should be allowed (Windows-specific)."""
        # This is a special case - routing microphone to output
        assert is_device_safe("Microphone Output", default_config) is True
        assert is_device_safe("Mic Output", default_config) is True

    @WINDOWS_ONLY
    def test_speakers_allowed(self, default_config):
        """Speakers with output indicator should be allowed (Windows-specific)."""
        assert is_device_safe("Speakers (Realtek Audio)", default_config) is True
        assert is_device_safe("Speaker (USB)", default_config) is True


@pytest.mark.integration
class TestUnicodeDeviceNames:
    """Test Unicode and international device names."""

    @pytest.mark.parametrize("device_name,should_block", [
        # Japanese microphone names - should be blocked if they contain known patterns
        ("マイク", True),  # Japanese for "microphone" - unknown, default blocked
        ("VB-Cable マイク出力", True),  # Has VB-Cable but also マイク
        # Chinese microphone names
        ("麦克风", True),  # Chinese for "microphone" - unknown, default blocked
        # German
        ("Mikrofon", True),  # Unknown, default blocked
        # Safe unicode names
        ("VB-Cable Output (日本語)", True),  # VB-Cable with Japanese
        ("Stereo Mix (中文)", True),  # Stereo Mix with Chinese
    ])
    def test_unicode_device_names(self, default_config, device_name, should_block):
        """Unicode device names should be handled safely."""
        # Default behavior: unknown devices are blocked
        result = is_device_safe(device_name, default_config)
        # We just verify it doesn't crash on unicode
        assert isinstance(result, bool)

    def test_unicode_in_allowlist(self):
        """Unicode device names can be added to allowlist."""
        config = DeviceGuardConfig(allowlist=["マイク出力", "麦克风输出"])
        assert is_device_safe("マイク出力 Device", config) is True
        assert is_device_safe("麦克风输出", config) is True


@pytest.mark.integration
class TestSpecialCharacters:
    """Test device names with special characters."""

    @pytest.mark.parametrize("device_name", [
        "Device (v2.0)",
        "Audio & Video Out",
        "Device [USB]",
        "Device {Test}",
        "Device <Output>",
        "Audio/Video Device",
        "Device @ Location",
        "Device #1",
        "Device $pecial",
        "Device %20",
        "Device^Audio",
        "Device+Output",
        "Device=Audio",
    ])
    def test_special_characters_handled(self, default_config, device_name):
        """Devices with special characters should not crash."""
        # Should not raise any exceptions
        result = is_device_safe(device_name, default_config)
        assert isinstance(result, bool)

        status = get_device_status(device_name, default_config)
        assert "safe" in status
        assert "reason" in status


# ============================================================================
# Configuration Override Tests
# ============================================================================


@pytest.mark.integration
class TestConfigurationOverrides:
    """Test allowlist/blocklist configuration overrides."""

    def test_custom_allowlist_overrides_default(self):
        """Custom allowlist can add new safe devices."""
        config = DeviceGuardConfig(allowlist=["My Custom Device"])
        assert is_device_safe("My Custom Device ABC", config) is True

    def test_custom_blocklist_blocks_even_safe_patterns(self):
        """Custom blocklist can block normally-safe devices."""
        config = DeviceGuardConfig(blocklist=["Stereo"])
        # Stereo Mix would normally be allowed, but "Stereo" is now blocked
        assert is_device_safe("Stereo Mix", config) is False

    def test_blocklist_checked_before_allowlist(self):
        """Blocklist takes precedence over allowlist."""
        config = DeviceGuardConfig(
            allowlist=["AudioDevice"],
            blocklist=["AudioDevice"],
        )
        # If both lists match, blocklist wins
        assert is_device_safe("AudioDevice", config) is False

    @WINDOWS_ONLY
    def test_empty_allowlist_uses_default_patterns(self):
        """Empty allowlist still uses default safe patterns (Windows-specific)."""
        config = DeviceGuardConfig(allowlist=[])
        # Default patterns should still match
        assert is_device_safe("VB-Cable Output", config) is True

    def test_empty_blocklist_uses_dangerous_patterns(self):
        """Empty blocklist still uses dangerous patterns."""
        config = DeviceGuardConfig(blocklist=[])
        # Dangerous patterns should still match
        assert is_device_safe("Microphone", config) is False


@pytest.mark.integration
class TestDisabledGuard:
    """Test behavior when guard is disabled."""

    def test_disabled_allows_microphones(self):
        """Disabled guard allows microphone devices."""
        config = DeviceGuardConfig(enabled=False)
        assert is_device_safe("Microphone (Realtek)", config) is True

    def test_disabled_status_shows_correct_type(self):
        """Disabled guard status shows 'disabled' match type."""
        config = DeviceGuardConfig(enabled=False)
        status = get_device_status("Any Device", config)
        assert status["match_type"] == "disabled"
        assert status["safe"] is True


# ============================================================================
# Fail-Safe Behavior Tests
# ============================================================================


@pytest.mark.integration
class TestFailSafeBehavior:
    """Test fail-safe (default block) behavior."""

    @pytest.mark.parametrize("device_name", [
        "Unknown Device XYZ",
        "Random Audio Device",
        "Unnamed Device",
        "Device 12345",
        "Test Device",
        "My Personal Device",
    ])
    def test_unknown_devices_blocked(self, default_config, device_name):
        """Unknown devices must be blocked by default."""
        assert is_device_safe(device_name, default_config) is False

    def test_status_shows_default_block(self, default_config):
        """Status shows default_block for unknown devices."""
        status = get_device_status("Unknown Device ABC", default_config)
        assert status["match_type"] == "default_block"
        assert status["safe"] is False


# ============================================================================
# Validation Function Tests
# ============================================================================


@pytest.mark.integration
class TestValidateDeviceForRecording:
    """Test validate_device_for_recording function."""

    def test_raises_for_microphone(self, default_config):
        """Raises ValueError with detailed message for microphone."""
        with pytest.raises(ValueError) as exc_info:
            validate_device_for_recording("Microphone (Realtek)", default_config)

        error_message = str(exc_info.value)
        assert "BLOCKED" in error_message
        assert "Microphone (Realtek)" in error_message
        assert "approved output devices" in error_message.lower()

    @WINDOWS_ONLY
    def test_no_raise_for_safe_device(self, default_config):
        """Does not raise for approved devices (Windows-specific)."""
        # Should not raise
        validate_device_for_recording("VB-Cable Output", default_config)
        validate_device_for_recording("Stereo Mix", default_config)


# ============================================================================
# Device Filtering Tests
# ============================================================================


@pytest.mark.integration
@WINDOWS_ONLY
class TestGetSafeDevices:
    """Test get_safe_devices filtering function (Windows-specific device names)."""

    def test_filters_mixed_device_list(self, default_config):
        """Correctly filters a mixed list of devices."""
        devices = [
            "Microphone (Realtek)",
            "VB-Cable Output",
            "Webcam Audio",
            "Stereo Mix (Realtek)",
            "Internal Mic",
            "VoiceMeeter Input",
            "Camera",
            "What U Hear",
        ]

        safe = get_safe_devices(devices, default_config)

        assert "VB-Cable Output" in safe
        assert "Stereo Mix (Realtek)" in safe
        assert "VoiceMeeter Input" in safe
        assert "What U Hear" in safe

        assert "Microphone (Realtek)" not in safe
        assert "Webcam Audio" not in safe
        assert "Internal Mic" not in safe
        assert "Camera" not in safe

    def test_preserves_order(self, default_config):
        """Preserves order of safe devices."""
        devices = [
            "VB-Cable Output",
            "Unknown Device",
            "Stereo Mix",
            "Microphone",
            "VoiceMeeter",
        ]

        safe = get_safe_devices(devices, default_config)

        assert safe == ["VB-Cable Output", "Stereo Mix", "VoiceMeeter"]

    def test_handles_empty_list(self, default_config):
        """Handles empty device list."""
        assert get_safe_devices([], default_config) == []

    def test_handles_all_unsafe_list(self, default_config):
        """Returns empty when all devices are unsafe."""
        devices = ["Microphone", "Webcam", "Camera"]
        assert get_safe_devices(devices, default_config) == []

    def test_handles_all_safe_list(self, default_config):
        """Returns all when all devices are safe."""
        devices = ["VB-Cable Output", "Stereo Mix", "VoiceMeeter"]
        safe = get_safe_devices(devices, default_config)
        assert len(safe) == 3


# ============================================================================
# Logging Tests
# ============================================================================


@pytest.mark.integration
class TestLogDeviceDecision:
    """Test log_device_decision function."""

    def test_log_contains_device_name(self, default_config):
        """Log message contains device name."""
        log = log_device_decision("Test Device", default_config)
        assert "Test Device" in log

    @WINDOWS_ONLY
    def test_log_shows_allowed(self, default_config):
        """Log shows ALLOWED for safe devices (Windows-specific)."""
        log = log_device_decision("VB-Cable Output", default_config)
        assert "ALLOWED" in log

    def test_log_shows_blocked(self, default_config):
        """Log shows BLOCKED for unsafe devices."""
        log = log_device_decision("Microphone", default_config)
        assert "BLOCKED" in log

    def test_log_includes_match_type(self, default_config):
        """Log includes match type."""
        log = log_device_decision("VB-Cable Output", default_config)
        assert "match_type" in log

    def test_log_includes_reason(self, default_config):
        """Log includes reason."""
        log = log_device_decision("Unknown Device", default_config)
        assert "reason" in log


# ============================================================================
# API Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIDeviceGuardIntegration:
    """Test device guard integration with API endpoints."""

    @pytest.fixture
    async def client(self):
        """Create async HTTP client for testing."""
        from callwhisper.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_recording_start_validates_device(self, client):
        """POST /api/recording/start validates device safety."""
        # Attempt to start recording with microphone should fail
        response = await client.post(
            "/api/recording/start",
            json={"device_name": "Microphone (Realtek)", "ticket_id": "TEST123"}
        )

        # Should be rejected (either 400 or validation error)
        assert response.status_code in [400, 422]

    async def test_devices_endpoint_filters_unsafe(self, client):
        """GET /api/devices filters out unsafe devices."""
        # This test verifies the API returns filtered devices
        response = await client.get("/api/devices")

        # The endpoint should work
        assert response.status_code == 200

        # Response should have devices key
        if response.json().get("devices"):
            devices = response.json()["devices"]
            # None of the returned devices should be obvious microphones
            for device in devices:
                name_lower = device.get("name", "").lower()
                # Basic sanity check - returned devices shouldn't be obvious mics
                # (Full check is done by device guard)
                assert "microphone" not in name_lower or "output" in name_lower


# ============================================================================
# Pattern Verification Tests
# ============================================================================


@pytest.mark.integration
@WINDOWS_ONLY
class TestPatternCoverage:
    """Verify pattern coverage is comprehensive (Windows-specific device patterns)."""

    def test_all_default_safe_patterns_work(self, default_config):
        """Each safe pattern should match at least one device."""
        test_devices = {
            r"vb-?cable": "VB-Cable Output",
            r"cable\s*output": "CABLE Output",
            r"virtual\s*(audio\s*)?cable": "Virtual Audio Cable",
            r"stereo\s*mix": "Stereo Mix",
            r"what\s*u\s*hear": "What U Hear",
            r"voicemeeter": "VoiceMeeter Input",
            r"jabber": "Cisco Jabber",
            r"finesse": "Finesse Audio",
            r"cisco": "Cisco Audio",
            r"loopback": "Audio Loopback",
            r"speakers?\s*\(": "Speakers (Realtek)",
        }

        for pattern, device in test_devices.items():
            assert is_device_safe(device, default_config) is True, \
                f"Pattern '{pattern}' did not match device '{device}'"

    def test_dangerous_patterns_comprehensive(self, default_config):
        """All dangerous patterns should block appropriate devices."""
        dangerous_devices = [
            "Microphone (Realtek)",
            "Internal Mic",
            "Webcam Audio",
            "USB Camera",
            "Headset Mic (USB)",
            "Built-in Microphone",
            "Internal Microphone",
            "Array Mic",
            "Input Device",
            "Realtek Audio Input",
            "Recording Device",
        ]

        for device in dangerous_devices:
            assert is_device_safe(device, default_config) is False, \
                f"Dangerous device '{device}' was not blocked!"
