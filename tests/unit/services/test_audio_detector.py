"""
Tests for Virtual Audio Device Detector.

Tests device detection:
- Virtual audio device pattern matching
- Device type classification
- Setup status generation
- Best device selection
- Platform-specific behavior (Windows vs Linux)
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from callwhisper.services.audio_detector import (
    VirtualAudioDevice,
    SetupStatus,
    detect_virtual_audio_devices,
    is_virtual_cable_available,
    is_recommended_device_available,
    get_setup_status,
    mark_setup_complete,
    get_recommended_device_info,
    get_best_audio_device,
    VIRTUAL_AUDIO_PATTERNS,
)


class TestDetectVirtualAudioDevices:
    """Tests for detect_virtual_audio_devices function."""

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_vb_cable(self, mock_list, mock_platform):
        """Detects VB-Cable as recommended device (Windows)."""
        mock_list.return_value = ["CABLE Output (VB-Audio Virtual Cable)"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "vb-cable"
        assert devices[0].is_recommended is True

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_voicemeeter(self, mock_list, mock_platform):
        """Detects VoiceMeeter as recommended device (Windows)."""
        mock_list.return_value = ["VoiceMeeter Input"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "voicemeeter"
        assert devices[0].is_recommended is True

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_stereo_mix(self, mock_list, mock_platform):
        """Detects Stereo Mix as non-recommended device (Windows)."""
        mock_list.return_value = ["Stereo Mix (Realtek Audio)"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "stereo-mix"
        assert devices[0].is_recommended is False

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_virtual_loopback_windows(self, mock_list, mock_platform):
        """Detects virtual/loopback devices (Windows)."""
        mock_list.return_value = ["Virtual Audio Loopback"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "virtual"
        assert devices[0].is_recommended is False

    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_ignores_non_virtual_devices(self, mock_list):
        """Ignores regular audio devices."""
        mock_list.return_value = [
            "Microphone (Realtek Audio)",
            "Speakers (Realtek Audio)",
            "Headphones",
        ]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 0

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_multiple_devices(self, mock_list, mock_platform):
        """Detects multiple virtual devices (Windows)."""
        mock_list.return_value = [
            "CABLE Output (VB-Audio Virtual Cable)",
            "Stereo Mix (Realtek Audio)",
            "VoiceMeeter Input",
        ]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 3
        types = [d.device_type for d in devices]
        assert "vb-cable" in types
        assert "stereo-mix" in types
        assert "voicemeeter" in types

    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_handles_list_error(self, mock_list):
        """Returns empty list when device enumeration fails."""
        mock_list.side_effect = Exception("FFmpeg not found")

        devices = detect_virtual_audio_devices()

        assert devices == []

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_case_insensitive_matching(self, mock_list, mock_platform):
        """Pattern matching is case insensitive (Windows)."""
        mock_list.return_value = ["VB-CABLE Output", "STEREO MIX"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 2

    # Linux-specific tests
    @patch("callwhisper.services.audio_detector.get_platform", return_value="linux")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_pulse_monitor_linux(self, mock_list, mock_platform):
        """Detects PulseAudio monitor sink as recommended device (Linux)."""
        mock_list.return_value = ["alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "pulse-monitor"
        assert devices[0].is_recommended is True

    @patch("callwhisper.services.audio_detector.get_platform", return_value="linux")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_null_sink_linux(self, mock_list, mock_platform):
        """Detects null sink as recommended device (Linux)."""
        mock_list.return_value = ["null-sink"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "null-sink"
        assert devices[0].is_recommended is True

    @patch("callwhisper.services.audio_detector.get_platform", return_value="linux")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_detects_loopback_linux(self, mock_list, mock_platform):
        """Detects loopback device as recommended (Linux)."""
        mock_list.return_value = ["loopback"]

        devices = detect_virtual_audio_devices()

        assert len(devices) == 1
        assert devices[0].device_type == "loopback"
        assert devices[0].is_recommended is True


class TestIsVirtualCableAvailable:
    """Tests for is_virtual_cable_available function."""

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_returns_true_when_devices_found(self, mock_detect):
        """Returns True when virtual devices exist."""
        mock_detect.return_value = [
            VirtualAudioDevice("Test", "vb-cable", True)
        ]

        assert is_virtual_cable_available() is True

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_returns_false_when_no_devices(self, mock_detect):
        """Returns False when no virtual devices."""
        mock_detect.return_value = []

        assert is_virtual_cable_available() is False


class TestIsRecommendedDeviceAvailable:
    """Tests for is_recommended_device_available function."""

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_returns_true_for_recommended(self, mock_detect):
        """Returns True when recommended device exists."""
        mock_detect.return_value = [
            VirtualAudioDevice("VB-Cable", "vb-cable", True)
        ]

        assert is_recommended_device_available() is True

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_returns_false_for_non_recommended(self, mock_detect):
        """Returns False when only non-recommended devices exist."""
        mock_detect.return_value = [
            VirtualAudioDevice("Stereo Mix", "stereo-mix", False)
        ]

        assert is_recommended_device_available() is False


class TestGetSetupStatus:
    """Tests for get_setup_status function."""

    @patch("callwhisper.services.audio_detector._check_setup_skipped")
    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_status_with_recommended_device(self, mock_list, mock_detect, mock_skipped):
        """Returns complete status with recommended device."""
        mock_list.return_value = ["CABLE Output", "Microphone"]
        mock_detect.return_value = [
            VirtualAudioDevice("CABLE Output", "vb-cable", True)
        ]
        mock_skipped.return_value = False

        status = get_setup_status()

        assert status["virtual_audio_detected"] is True
        assert status["recommended_device_available"] is True
        assert status["recommended_action"] is None
        assert status["setup_complete"] is True

    @patch("callwhisper.services.audio_detector.get_platform", return_value="windows")
    @patch("callwhisper.services.audio_detector._check_setup_skipped")
    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_status_needs_vbcable(self, mock_list, mock_detect, mock_skipped, mock_platform):
        """Recommends installing VB-Cable when no virtual devices (Windows)."""
        mock_list.return_value = ["Microphone"]
        mock_detect.return_value = []
        mock_skipped.return_value = False

        status = get_setup_status()

        assert status["virtual_audio_detected"] is False
        assert status["recommended_action"] == "install_vbcable"
        assert status["setup_complete"] is False

    @patch("callwhisper.services.audio_detector.get_platform", return_value="linux")
    @patch("callwhisper.services.audio_detector._check_setup_skipped")
    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_status_needs_monitor_linux(self, mock_list, mock_detect, mock_skipped, mock_platform):
        """Recommends configuring monitor when no virtual devices (Linux)."""
        mock_list.return_value = ["hw:0,0"]
        mock_detect.return_value = []
        mock_skipped.return_value = False

        status = get_setup_status()

        assert status["virtual_audio_detected"] is False
        assert status["recommended_action"] == "configure_monitor"
        assert status["setup_complete"] is False

    @patch("callwhisper.services.audio_detector._check_setup_skipped")
    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_status_consider_upgrade(self, mock_list, mock_detect, mock_skipped):
        """Suggests upgrade when only stereo mix available."""
        mock_list.return_value = ["Stereo Mix"]
        mock_detect.return_value = [
            VirtualAudioDevice("Stereo Mix", "stereo-mix", False)
        ]
        mock_skipped.return_value = False

        status = get_setup_status()

        assert status["virtual_audio_detected"] is True
        assert status["recommended_device_available"] is False
        assert status["recommended_action"] == "consider_upgrade"

    @patch("callwhisper.services.audio_detector._check_setup_skipped")
    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    @patch("callwhisper.services.audio_detector.list_audio_devices")
    def test_status_skipped_setup(self, mock_list, mock_detect, mock_skipped):
        """Setup is complete when user skipped it."""
        mock_list.return_value = []
        mock_detect.return_value = []
        mock_skipped.return_value = True

        status = get_setup_status()

        assert status["setup_complete"] is True
        assert status["setup_skipped"] is True


class TestMarkSetupComplete:
    """Tests for mark_setup_complete function."""

    def test_creates_config_file(self, tmp_path):
        """Creates config file and marks setup complete."""
        config_file = tmp_path / "config.json"

        with patch("callwhisper.utils.paths.get_config_path", return_value=config_file):
            result = mark_setup_complete(skipped=False)

        assert result is True
        assert config_file.exists()

        with open(config_file) as f:
            config = json.load(f)
        assert config["setup_complete"] is True
        assert config["setup_skipped"] is False

    def test_marks_skipped(self, tmp_path):
        """Records when setup was skipped."""
        config_file = tmp_path / "config.json"

        with patch("callwhisper.utils.paths.get_config_path", return_value=config_file):
            result = mark_setup_complete(skipped=True)

        assert result is True
        with open(config_file) as f:
            config = json.load(f)
        assert config["setup_skipped"] is True

    def test_preserves_existing_config(self, tmp_path):
        """Preserves other config values."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"other_setting": "value"}')

        with patch("callwhisper.utils.paths.get_config_path", return_value=config_file):
            mark_setup_complete()

        with open(config_file) as f:
            config = json.load(f)
        assert config["other_setting"] == "value"
        assert config["setup_complete"] is True


class TestGetBestAudioDevice:
    """Tests for get_best_audio_device function."""

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_prefers_vb_cable(self, mock_detect):
        """Prefers VB-Cable over other devices."""
        mock_detect.return_value = [
            VirtualAudioDevice("Stereo Mix", "stereo-mix", False),
            VirtualAudioDevice("VB-Cable Output", "vb-cable", True),
            VirtualAudioDevice("VoiceMeeter", "voicemeeter", True),
        ]

        result = get_best_audio_device()

        assert result == "VB-Cable Output"

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_falls_back_to_voicemeeter(self, mock_detect):
        """Falls back to VoiceMeeter if no VB-Cable."""
        mock_detect.return_value = [
            VirtualAudioDevice("Stereo Mix", "stereo-mix", False),
            VirtualAudioDevice("VoiceMeeter Input", "voicemeeter", True),
        ]

        result = get_best_audio_device()

        assert result == "VoiceMeeter Input"

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_uses_any_virtual_if_no_recommended(self, mock_detect):
        """Uses any virtual device if no recommended available."""
        mock_detect.return_value = [
            VirtualAudioDevice("Stereo Mix", "stereo-mix", False),
        ]

        result = get_best_audio_device()

        assert result == "Stereo Mix"

    @patch("callwhisper.services.audio_detector.detect_virtual_audio_devices")
    def test_returns_none_if_no_devices(self, mock_detect):
        """Returns None if no virtual devices available."""
        mock_detect.return_value = []

        result = get_best_audio_device()

        assert result is None


class TestGetRecommendedDeviceInfo:
    """Tests for get_recommended_device_info function."""

    def test_returns_recommendations_list(self):
        """Returns list of recommended devices."""
        result = get_recommended_device_info()

        assert isinstance(result, list)
        assert len(result) >= 2  # At least VB-Cable and VoiceMeeter

    def test_recommendations_have_required_fields(self):
        """Each recommendation has required fields."""
        result = get_recommended_device_info()

        for device in result:
            assert "name" in device
            assert "pattern" in device
            assert "website" in device
            assert "description" in device
