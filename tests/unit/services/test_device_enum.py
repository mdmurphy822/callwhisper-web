"""
Tests for device enumeration service.

Tests FFmpeg audio device listing:
- Parsing FFmpeg device output
- Device caching behavior
- Device lookup by name
"""

import sys
import pytest
from unittest.mock import patch, MagicMock
import time

from callwhisper.services.device_enum import (
    parse_device_list,
    list_audio_devices,
    get_device_by_name,
    device_exists,
    _device_cache,
    CACHE_TTL_SECONDS,
)

# Skip Windows-specific dshow tests on Linux
WINDOWS_ONLY = pytest.mark.skipif(sys.platform != 'win32', reason="Windows dshow format")


class TestParseDeviceList:
    """Tests for parse_device_list function."""

    def test_parses_audio_devices(self):
        """Parses audio devices from FFmpeg output."""
        ffmpeg_output = """
[dshow @ 0x12345] DirectShow video devices
[dshow @ 0x12345]  "HD Webcam C270"
[dshow @ 0x12345] DirectShow audio devices
[dshow @ 0x12345]  "Microphone (Realtek Audio)"
[dshow @ 0x12345]  "Stereo Mix (Realtek Audio)"
[dshow @ 0x12345]  "VB-Cable Output"
"""
        devices = parse_device_list(ffmpeg_output)

        assert len(devices) == 3
        assert "Microphone (Realtek Audio)" in devices
        assert "Stereo Mix (Realtek Audio)" in devices
        assert "VB-Cable Output" in devices
        # Video device should not be included
        assert "HD Webcam C270" not in devices

    def test_skips_alternative_names(self):
        """Skips 'Alternative name' entries."""
        ffmpeg_output = """
[dshow @ 0x12345] DirectShow audio devices
[dshow @ 0x12345]  "Stereo Mix (Realtek Audio)"
[dshow @ 0x12345]     Alternative name "@device_cm_{...}"
"""
        devices = parse_device_list(ffmpeg_output)

        assert len(devices) == 1
        assert "Stereo Mix (Realtek Audio)" in devices

    def test_handles_empty_output(self):
        """Handles empty FFmpeg output."""
        devices = parse_device_list("")
        assert devices == []

    def test_handles_no_audio_section(self):
        """Handles output with no audio devices section."""
        ffmpeg_output = """
[dshow @ 0x12345] DirectShow video devices
[dshow @ 0x12345]  "HD Webcam C270"
"""
        devices = parse_device_list(ffmpeg_output)
        assert devices == []


class TestListAudioDevices:
    """Tests for list_audio_devices function."""

    @pytest.fixture(autouse=True)
    def reset_cache(self):
        """Reset device cache before each test."""
        import callwhisper.services.device_enum as device_enum
        device_enum._device_cache = ([], 0)
        yield

    @WINDOWS_ONLY
    def test_returns_devices_on_success(self):
        """Returns parsed devices on successful FFmpeg run."""
        mock_result = MagicMock()
        mock_result.stderr = """
[dshow @ 0x12345] DirectShow audio devices
[dshow @ 0x12345]  "Test Device"
"""
        with patch("subprocess.run", return_value=mock_result):
            with patch("callwhisper.services.device_enum.get_ffmpeg_path") as mock_path:
                mock_path.return_value = MagicMock(exists=lambda: True)
                devices = list_audio_devices(use_cache=False)

        assert "Test Device" in devices

    def test_returns_empty_on_timeout(self):
        """Returns empty list on FFmpeg timeout."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10)):
            with patch("callwhisper.services.device_enum.get_ffmpeg_path") as mock_path:
                mock_path.return_value = MagicMock(exists=lambda: True)
                devices = list_audio_devices(use_cache=False)

        assert devices == []

    def test_returns_empty_on_ffmpeg_not_found(self):
        """Returns empty list when FFmpeg not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with patch("callwhisper.services.device_enum.get_ffmpeg_path") as mock_path:
                mock_path.return_value = MagicMock(exists=lambda: False)
                devices = list_audio_devices(use_cache=False)

        assert devices == []


class TestGetDeviceByName:
    """Tests for get_device_by_name function."""

    def test_finds_device_by_partial_name(self):
        """Finds device using partial name match."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = [
                "Microphone (Realtek Audio)",
                "Stereo Mix (Realtek Audio)",
            ]
            result = get_device_by_name("Stereo")

        assert result == "Stereo Mix (Realtek Audio)"

    def test_case_insensitive_match(self):
        """Name matching is case-insensitive."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = ["VB-Cable Output"]
            result = get_device_by_name("vb-cable")

        assert result == "VB-Cable Output"

    def test_returns_none_when_not_found(self):
        """Returns None when device not found."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = ["Stereo Mix"]
            result = get_device_by_name("Nonexistent Device")

        assert result is None


class TestDeviceExists:
    """Tests for device_exists function."""

    def test_returns_true_for_existing_device(self):
        """Returns True for existing device."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = ["Stereo Mix", "VB-Cable"]
            assert device_exists("Stereo Mix") is True

    def test_returns_false_for_missing_device(self):
        """Returns False for missing device."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = ["Stereo Mix"]
            assert device_exists("Nonexistent") is False

    def test_exact_match_required(self):
        """Requires exact device name match."""
        with patch("callwhisper.services.device_enum.list_audio_devices") as mock_list:
            mock_list.return_value = ["Stereo Mix (Realtek Audio)"]
            # Partial match should return False
            assert device_exists("Stereo Mix") is False
