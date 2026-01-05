"""
Path resolution utilities for PyInstaller compatibility.

Supports three deployment modes:
1. Development: Running from source code
2. Portable: Running from PyInstaller bundle (ZIP extraction or folder)
3. Installed: Running from MSI-installed location

For enterprise Windows deployments, the portable and installed modes
store all data next to the executable for easy backup and GPO management.
"""

import re
import sys
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any


@lru_cache
def is_frozen() -> bool:
    """Check if running as PyInstaller bundle."""
    return getattr(sys, "frozen", False)


@lru_cache
def is_portable_mode() -> bool:
    """
    Check if running from portable bundle.

    Portable mode is detected when:
    1. Running as frozen PyInstaller bundle, AND
    2. version.json exists next to the executable

    This distinguishes portable ZIP distribution from MSI installation.
    """
    if not is_frozen():
        return False

    exe_dir = Path(sys.executable).parent
    version_file = exe_dir / "version.json"
    return version_file.exists()


@lru_cache
def get_version_info() -> Optional[Dict[str, Any]]:
    """
    Get version info from version.json if in portable mode.

    Returns:
        Dictionary with version info or None if not in portable mode.
    """
    if not is_frozen():
        return None

    exe_dir = Path(sys.executable).parent
    version_file = exe_dir / "version.json"

    if version_file.exists():
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    return None


@lru_cache
def get_deployment_mode() -> str:
    """
    Determine the current deployment mode.

    Returns:
        One of: "development", "portable", "installed"
    """
    if not is_frozen():
        return "development"

    if is_portable_mode():
        return "portable"

    return "installed"


@lru_cache
def get_base_dir() -> Path:
    """
    Get base directory for bundled resources.

    When running as PyInstaller bundle, returns _MEIPASS (temp extraction dir).
    When running in development, returns project root.
    """
    if is_frozen():
        return Path(sys._MEIPASS)
    else:
        # Development: go up from utils -> callwhisper -> src -> project root
        return Path(__file__).parent.parent.parent.parent


@lru_cache
def get_data_dir() -> Path:
    """
    Get directory for user data (config, output, models).

    Portable/Installed mode: Uses 'data' subdirectory next to executable.
    Development mode: Uses project root.

    This ensures all user data is stored in a predictable location
    for enterprise IT backup and management.
    """
    if is_frozen():
        # Both portable and installed modes store data next to executable
        return Path(sys.executable).parent / "data"
    else:
        return get_base_dir()


@lru_cache
def get_install_dir() -> Path:
    """
    Get the installation directory.

    For frozen apps, this is where the executable is located.
    For development, this is the project root.
    """
    if is_frozen():
        return Path(sys.executable).parent
    else:
        return get_base_dir()


def get_static_dir() -> Path:
    """Get static files directory."""
    return get_base_dir() / "static"


def get_vendor_dir() -> Path:
    """Get vendor binaries directory (ffmpeg, whisper)."""
    return get_data_dir() / "vendor"


def get_output_dir() -> Path:
    """Get output directory for recordings."""
    return get_data_dir() / "output"


def get_models_dir() -> Path:
    """Get models directory for whisper models."""
    return get_data_dir() / "models"


def get_config_dir() -> Path:
    """Get config directory."""
    return get_data_dir() / "config"


def get_config_path() -> Path:
    """Get path to main config file."""
    return get_config_dir() / "config.json"


def get_ffmpeg_path() -> Path:
    """Get path to ffmpeg executable.

    On Windows, returns vendor/ffmpeg.exe.
    On Linux/Mac, returns system PATH ffmpeg or vendor path.
    """
    import shutil
    import os

    # Check vendor directory first (Windows deployment)
    vendor_ffmpeg = get_vendor_dir() / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if vendor_ffmpeg.exists():
        return vendor_ffmpeg

    # Fall back to system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return Path(system_ffmpeg)

    # Return expected vendor path for error messages
    return get_vendor_dir() / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def get_whisper_path() -> Path:
    """Get path to whisper-cli executable.

    On Windows, returns vendor/whisper-cli.exe.
    On Linux/Mac, returns vendor/whisper-cli or system PATH.
    """
    import shutil
    import os

    # Check vendor directory first
    exe_name = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    vendor_whisper = get_vendor_dir() / exe_name
    if vendor_whisper.exists():
        return vendor_whisper

    # Fall back to system PATH (Linux/Mac)
    system_whisper = shutil.which("whisper-cli")
    if system_whisper:
        return Path(system_whisper)

    # Return expected vendor path for error messages
    return vendor_whisper


def get_logs_dir() -> Path:
    """Get logs directory."""
    return get_data_dir() / "logs"


def get_checkpoints_dir() -> Path:
    """Get checkpoints directory for crash recovery."""
    return get_data_dir() / "checkpoints"


def get_path_info() -> Dict[str, Any]:
    """
    Get comprehensive path information for debugging and IT verification.

    Returns:
        Dictionary with all path locations and deployment mode.
    """
    return {
        "deployment_mode": get_deployment_mode(),
        "is_frozen": is_frozen(),
        "is_portable": is_portable_mode(),
        "version_info": get_version_info(),
        "paths": {
            "install_dir": str(get_install_dir()),
            "base_dir": str(get_base_dir()),
            "data_dir": str(get_data_dir()),
            "static_dir": str(get_static_dir()),
            "vendor_dir": str(get_vendor_dir()),
            "output_dir": str(get_output_dir()),
            "models_dir": str(get_models_dir()),
            "config_dir": str(get_config_dir()),
            "logs_dir": str(get_logs_dir()),
            "ffmpeg": str(get_ffmpeg_path()),
            "whisper": str(get_whisper_path()),
        },
    }


def ensure_data_dirs() -> None:
    """
    Ensure all required data directories exist.

    Called during application startup to create necessary directories.
    """
    dirs = [
        get_data_dir(),
        get_output_dir(),
        get_config_dir(),
        get_logs_dir(),
        get_checkpoints_dir(),
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


# Path Sanitization Utilities

# Safe characters for path components: alphanumeric, dash, underscore, dot
_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def sanitize_path_component(value: str, max_length: int = 128) -> str:
    """
    Sanitize a string for safe use in path construction.

    Prevents path traversal attacks by rejecting dangerous patterns
    and replacing unsafe characters.

    Args:
        value: The string to sanitize
        max_length: Maximum allowed length (default 128)

    Returns:
        Sanitized string safe for use in paths

    Raises:
        ValueError: If the value contains path traversal attempts or null bytes
    """
    if not value:
        return value

    # Check for path traversal patterns
    if ".." in value:
        raise ValueError("Path traversal detected: contains '..'")

    if "/" in value:
        raise ValueError("Path traversal detected: contains '/'")

    if "\\" in value:
        raise ValueError("Path traversal detected: contains '\\'")

    # Check for null bytes
    if "\x00" in value:
        raise ValueError("Null byte detected in path component")

    # Truncate to max length
    value = value[:max_length]

    # Replace unsafe characters with underscore
    if not _SAFE_PATH_COMPONENT_PATTERN.match(value):
        value = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", value)

    return value


def validate_path_within_directory(path: Path, base_dir: Path) -> bool:
    """
    Validate that a path is within the expected base directory.

    Resolves symlinks and relative paths before checking containment.

    Args:
        path: The path to validate (will be resolved)
        base_dir: The directory the path must be within

    Returns:
        True if path is safely within base_dir, False otherwise
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        # Check if resolved path starts with base (using parts to handle edge cases)
        return resolved_path.parts[: len(resolved_base.parts)] == resolved_base.parts
    except (OSError, ValueError):
        return False
