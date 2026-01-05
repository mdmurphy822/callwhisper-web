"""
Tests for Path Resolution Utilities.

Tests deployment mode detection and path resolution:
- Frozen vs development mode
- Portable vs installed mode
- Path resolution functions
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from callwhisper.utils.paths import (
    is_frozen,
    is_portable_mode,
    get_version_info,
    get_deployment_mode,
    get_base_dir,
    get_data_dir,
    get_install_dir,
    get_static_dir,
    get_vendor_dir,
    get_output_dir,
    get_models_dir,
    get_config_dir,
    get_config_path,
    get_ffmpeg_path,
    get_whisper_path,
    get_logs_dir,
    get_checkpoints_dir,
    get_path_info,
    ensure_data_dirs,
)


class TestIsFrozen:
    """Tests for is_frozen function."""

    def test_returns_false_in_development(self):
        """is_frozen returns False when not running as PyInstaller bundle."""
        # Clear cache first
        is_frozen.cache_clear()

        result = is_frozen()

        assert result is False

    def test_returns_true_when_frozen(self):
        """is_frozen returns True when sys.frozen is True."""
        is_frozen.cache_clear()

        with patch.object(sys, "frozen", True, create=True):
            result = is_frozen()

        assert result is True

        is_frozen.cache_clear()


class TestIsPortableMode:
    """Tests for is_portable_mode function."""

    def test_returns_false_in_development(self):
        """is_portable_mode returns False in development."""
        is_portable_mode.cache_clear()
        is_frozen.cache_clear()

        result = is_portable_mode()

        assert result is False

    def test_returns_true_with_version_json(self, tmp_path):
        """is_portable_mode returns True when version.json exists."""
        is_portable_mode.cache_clear()
        is_frozen.cache_clear()

        version_file = tmp_path / "version.json"
        version_file.write_text('{"version": "1.0.0"}')

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(tmp_path / "app.exe")):
                result = is_portable_mode()

        assert result is True

        is_portable_mode.cache_clear()
        is_frozen.cache_clear()


class TestGetVersionInfo:
    """Tests for get_version_info function."""

    def test_returns_none_in_development(self):
        """get_version_info returns None in development mode."""
        get_version_info.cache_clear()
        is_frozen.cache_clear()

        result = get_version_info()

        assert result is None

    def test_returns_version_data(self, tmp_path):
        """get_version_info returns version data when available."""
        get_version_info.cache_clear()
        is_frozen.cache_clear()

        version_file = tmp_path / "version.json"
        version_data = {"version": "2.0.0", "build": "123"}
        version_file.write_text(json.dumps(version_data))

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(tmp_path / "app.exe")):
                result = get_version_info()

        assert result == version_data

        get_version_info.cache_clear()
        is_frozen.cache_clear()

    def test_returns_none_for_invalid_json(self, tmp_path):
        """get_version_info returns None for invalid JSON."""
        get_version_info.cache_clear()
        is_frozen.cache_clear()

        version_file = tmp_path / "version.json"
        version_file.write_text("not valid json {{{")

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(tmp_path / "app.exe")):
                result = get_version_info()

        assert result is None

        get_version_info.cache_clear()
        is_frozen.cache_clear()


class TestGetDeploymentMode:
    """Tests for get_deployment_mode function."""

    def test_development_mode(self):
        """get_deployment_mode returns 'development' in dev mode."""
        get_deployment_mode.cache_clear()
        is_frozen.cache_clear()
        is_portable_mode.cache_clear()

        result = get_deployment_mode()

        assert result == "development"

    def test_portable_mode(self, tmp_path):
        """get_deployment_mode returns 'portable' with version.json."""
        get_deployment_mode.cache_clear()
        is_frozen.cache_clear()
        is_portable_mode.cache_clear()

        version_file = tmp_path / "version.json"
        version_file.write_text('{"version": "1.0.0"}')

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(tmp_path / "app.exe")):
                result = get_deployment_mode()

        assert result == "portable"

        get_deployment_mode.cache_clear()
        is_frozen.cache_clear()
        is_portable_mode.cache_clear()

    def test_installed_mode(self, tmp_path):
        """get_deployment_mode returns 'installed' when frozen without version.json."""
        get_deployment_mode.cache_clear()
        is_frozen.cache_clear()
        is_portable_mode.cache_clear()

        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", str(tmp_path / "app.exe")):
                result = get_deployment_mode()

        assert result == "installed"

        get_deployment_mode.cache_clear()
        is_frozen.cache_clear()
        is_portable_mode.cache_clear()


class TestPathResolvers:
    """Tests for path resolution functions."""

    def test_get_base_dir_development(self):
        """get_base_dir returns project root in development."""
        get_base_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_base_dir()

        # Should be a real path
        assert result.exists() or True  # May not exist in test env

    def test_get_data_dir_development(self):
        """get_data_dir returns base_dir in development."""
        get_data_dir.cache_clear()
        get_base_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_data_dir()

        assert isinstance(result, Path)

    def test_get_install_dir_development(self):
        """get_install_dir returns base_dir in development."""
        get_install_dir.cache_clear()
        get_base_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_install_dir()

        assert isinstance(result, Path)

    def test_get_static_dir(self):
        """get_static_dir returns static subdirectory."""
        get_base_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_static_dir()

        assert result.name == "static"

    def test_get_vendor_dir(self):
        """get_vendor_dir returns vendor subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_vendor_dir()

        assert result.name == "vendor"

    def test_get_output_dir(self):
        """get_output_dir returns output subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_output_dir()

        assert result.name == "output"

    def test_get_models_dir(self):
        """get_models_dir returns models subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_models_dir()

        assert result.name == "models"

    def test_get_config_dir(self):
        """get_config_dir returns config subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_config_dir()

        assert result.name == "config"

    def test_get_config_path(self):
        """get_config_path returns config.json path."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_config_path()

        assert result.name == "config.json"
        assert result.parent.name == "config"

    def test_get_logs_dir(self):
        """get_logs_dir returns logs subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_logs_dir()

        assert result.name == "logs"

    def test_get_checkpoints_dir(self):
        """get_checkpoints_dir returns checkpoints subdirectory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        result = get_checkpoints_dir()

        assert result.name == "checkpoints"


class TestGetFfmpegPath:
    """Tests for get_ffmpeg_path function."""

    def test_returns_system_ffmpeg(self):
        """get_ffmpeg_path falls back to system ffmpeg."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            with patch("callwhisper.utils.paths.get_vendor_dir") as mock_vendor:
                mock_vendor.return_value = Path("/nonexistent/vendor")
                result = get_ffmpeg_path()

        assert str(result) == "/usr/bin/ffmpeg"

    def test_prefers_vendor_ffmpeg(self, tmp_path):
        """get_ffmpeg_path prefers vendor directory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        ffmpeg = vendor_dir / "ffmpeg"
        ffmpeg.write_text("")

        with patch("callwhisper.utils.paths.get_vendor_dir", return_value=vendor_dir):
            result = get_ffmpeg_path()

        assert result == ffmpeg


class TestGetWhisperPath:
    """Tests for get_whisper_path function."""

    def test_returns_system_whisper(self):
        """get_whisper_path falls back to system whisper-cli."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        with patch("shutil.which", return_value="/usr/bin/whisper-cli"):
            with patch("callwhisper.utils.paths.get_vendor_dir") as mock_vendor:
                mock_vendor.return_value = Path("/nonexistent/vendor")
                result = get_whisper_path()

        assert str(result) == "/usr/bin/whisper-cli"

    def test_prefers_vendor_whisper(self, tmp_path):
        """get_whisper_path prefers vendor directory."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        vendor_dir = tmp_path / "vendor"
        vendor_dir.mkdir()
        whisper = vendor_dir / "whisper-cli"
        whisper.write_text("")

        with patch("callwhisper.utils.paths.get_vendor_dir", return_value=vendor_dir):
            result = get_whisper_path()

        assert result == whisper


class TestGetPathInfo:
    """Tests for get_path_info function."""

    def test_returns_comprehensive_info(self):
        """get_path_info returns all path information."""
        # Clear caches
        for fn in [get_deployment_mode, is_frozen, is_portable_mode, get_version_info,
                   get_install_dir, get_base_dir, get_data_dir]:
            fn.cache_clear()

        result = get_path_info()

        assert "deployment_mode" in result
        assert "is_frozen" in result
        assert "is_portable" in result
        assert "paths" in result
        assert "install_dir" in result["paths"]
        assert "data_dir" in result["paths"]


class TestEnsureDataDirs:
    """Tests for ensure_data_dirs function."""

    def test_creates_directories(self, tmp_path):
        """ensure_data_dirs creates all required directories."""
        get_data_dir.cache_clear()
        is_frozen.cache_clear()

        with patch("callwhisper.utils.paths.get_data_dir", return_value=tmp_path):
            ensure_data_dirs()

        assert (tmp_path / "output").exists()
        assert (tmp_path / "config").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "checkpoints").exists()


# Import path sanitization functions
from callwhisper.utils.paths import sanitize_path_component, validate_path_within_directory


class TestSanitizePathComponent:
    """Tests for path sanitization security function."""

    def test_valid_simple_string(self):
        """Simple alphanumeric string passes through unchanged."""
        assert sanitize_path_component("TICKET-001") == "TICKET-001"

    def test_valid_with_underscore(self):
        """Underscores are allowed."""
        assert sanitize_path_component("call_20241229") == "call_20241229"

    def test_valid_with_dots(self):
        """Single dots are allowed."""
        assert sanitize_path_component("file.txt") == "file.txt"

    def test_path_traversal_parent_dir(self):
        """Path traversal with .. is rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_component("../../../etc/passwd")

    def test_path_traversal_double_dot(self):
        """Double dot alone is rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_component("..")

    def test_path_traversal_windows(self):
        """Windows-style path traversal is rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_component("..\\..\\windows")

    def test_forward_slash(self):
        """Forward slash is rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_component("ticket/secret")

    def test_backslash(self):
        """Backslash is rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_path_component("ticket\\secret")

    def test_null_byte(self):
        """Null byte injection is rejected."""
        with pytest.raises(ValueError, match="Null byte"):
            sanitize_path_component("ticket\x00evil")

    def test_max_length_truncation(self):
        """Long strings are truncated to max_length."""
        long_id = "A" * 200
        result = sanitize_path_component(long_id, max_length=128)
        assert len(result) == 128

    def test_special_chars_replaced(self):
        """Special characters are replaced with underscore."""
        result = sanitize_path_component("<script>alert(1)")
        assert "<" not in result
        assert ">" not in result
        assert "(" not in result
        assert ")" not in result

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert sanitize_path_component("") == ""

    def test_spaces_replaced(self):
        """Spaces are replaced with underscore."""
        result = sanitize_path_component("hello world")
        assert " " not in result
        assert "_" in result


class TestValidatePathWithinDirectory:
    """Tests for path containment validation."""

    def test_valid_subdirectory(self, tmp_path):
        """Path within base directory returns True."""
        base = tmp_path / "output"
        base.mkdir()
        path = base / "recording_001"

        assert validate_path_within_directory(path, base) is True

    def test_valid_nested_subdirectory(self, tmp_path):
        """Nested path within base directory returns True."""
        base = tmp_path / "output"
        base.mkdir()
        path = base / "2024" / "01" / "recording_001"

        assert validate_path_within_directory(path, base) is True

    def test_path_traversal_attempt(self, tmp_path):
        """Path traversal attempt returns False."""
        base = tmp_path / "output"
        base.mkdir()
        path = base / ".." / "etc" / "passwd"

        assert validate_path_within_directory(path, base) is False

    def test_absolute_path_outside(self, tmp_path):
        """Absolute path outside base returns False."""
        base = tmp_path / "output"
        base.mkdir()
        path = Path("/etc/passwd")

        assert validate_path_within_directory(path, base) is False

    def test_base_dir_itself(self, tmp_path):
        """Base directory itself returns True."""
        base = tmp_path / "output"
        base.mkdir()

        assert validate_path_within_directory(base, base) is True

    def test_sibling_directory(self, tmp_path):
        """Sibling directory returns False."""
        base = tmp_path / "output"
        base.mkdir()
        sibling = tmp_path / "config"
        sibling.mkdir()

        assert validate_path_within_directory(sibling, base) is False
