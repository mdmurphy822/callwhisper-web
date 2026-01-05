"""
Tests for configuration management.

Tests settings loading and validation:
- Default values
- Config file loading
- Settings caching
- Individual config sections
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from callwhisper.core.config import (
    Settings,
    ServerConfig,
    AudioConfig,
    TranscriptionConfig,
    OutputConfig,
    DeviceGuardConfig,
    SecurityConfig,
    PerformanceConfig,
    load_config_file,
    get_settings,
    reload_settings,
)


class TestServerConfig:
    """Tests for ServerConfig defaults."""

    def test_default_host(self):
        """Default host is localhost."""
        config = ServerConfig()
        assert config.host == "127.0.0.1"

    def test_default_port(self):
        """Default port is 8765."""
        config = ServerConfig()
        assert config.port == 8765

    def test_default_open_browser(self):
        """Default opens browser."""
        config = ServerConfig()
        assert config.open_browser is True


class TestAudioConfig:
    """Tests for AudioConfig defaults."""

    def test_default_sample_rate(self):
        """Default sample rate is 44100."""
        config = AudioConfig()
        assert config.sample_rate == 44100

    def test_default_channels(self):
        """Default channels is stereo."""
        config = AudioConfig()
        assert config.channels == 2


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig defaults."""

    def test_default_model(self):
        """Default model is medium.en."""
        config = TranscriptionConfig()
        assert "medium" in config.model

    def test_default_language(self):
        """Default language is English."""
        config = TranscriptionConfig()
        assert config.language == "en"


class TestOutputConfig:
    """Tests for OutputConfig defaults."""

    def test_default_creates_bundle(self):
        """Default creates VTB bundle."""
        config = OutputConfig()
        assert config.create_bundle is True

    def test_default_audio_format(self):
        """Default audio format is opus."""
        config = OutputConfig()
        assert config.audio_format == "opus"


class TestDeviceGuardConfig:
    """Tests for DeviceGuardConfig defaults."""

    def test_enabled_by_default(self):
        """Device guard is enabled by default."""
        config = DeviceGuardConfig()
        assert config.enabled is True

    def test_has_default_allowlist(self):
        """Has default safe devices in allowlist."""
        import sys
        config = DeviceGuardConfig()
        # Allowlist varies by platform
        if sys.platform == "win32":
            assert "VB-Cable" in config.allowlist
            assert "Stereo Mix" in config.allowlist
        else:
            # Linux uses monitor/loopback devices
            assert any(
                x in config.allowlist
                for x in ["Monitor", "monitor", "Loopback", "loopback", "pipewire"]
            )

    def test_has_default_blocklist(self):
        """Has microphone in default blocklist."""
        config = DeviceGuardConfig()
        assert "Microphone" in config.blocklist


class TestSecurityConfig:
    """Tests for SecurityConfig defaults."""

    def test_cors_enabled_by_default(self):
        """CORS is enabled by default."""
        config = SecurityConfig()
        assert config.cors_enabled is True

    def test_rate_limiting_enabled(self):
        """Rate limiting is enabled by default."""
        config = SecurityConfig()
        assert config.rate_limit_enabled is True


class TestPerformanceConfig:
    """Tests for PerformanceConfig defaults."""

    def test_cache_enabled_by_default(self):
        """Caching is enabled by default."""
        config = PerformanceConfig()
        assert config.cache_enabled is True

    def test_has_pool_sizes(self):
        """Has pool size configuration."""
        config = PerformanceConfig()
        assert config.audio_pool_size > 0
        assert config.transcription_pool_size > 0


class TestSettings:
    """Tests for main Settings class."""

    def test_has_all_sections(self):
        """Settings includes all config sections."""
        settings = Settings()
        assert hasattr(settings, 'server')
        assert hasattr(settings, 'audio')
        assert hasattr(settings, 'transcription')
        assert hasattr(settings, 'output')
        assert hasattr(settings, 'device_guard')
        assert hasattr(settings, 'security')
        assert hasattr(settings, 'performance')

    def test_has_version(self):
        """Settings has version number."""
        settings = Settings()
        assert settings.version == "1.0.0"

    def test_custom_values(self):
        """Custom values override defaults."""
        settings = Settings(
            server=ServerConfig(port=9000),
            transcription=TranscriptionConfig(language="de")
        )
        assert settings.server.port == 9000
        assert settings.transcription.language == "de"


class TestLoadConfigFile:
    """Tests for load_config_file function."""

    def test_returns_empty_dict_when_no_file(self):
        """Returns empty dict when config file doesn't exist."""
        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = Path("/nonexistent")
            result = load_config_file()
        assert result == {}

    def test_loads_valid_json(self, tmp_path):
        """Loads valid JSON config file."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{"version": "2.0.0"}')

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            result = load_config_file()

        assert result == {"version": "2.0.0"}


class TestGetSettings:
    """Tests for get_settings function."""

    def test_returns_settings_object(self):
        """Returns Settings instance."""
        with patch("callwhisper.core.config.load_config_file") as mock_load:
            mock_load.return_value = {}
            # Clear cache first
            get_settings.cache_clear()
            settings = get_settings()

        assert isinstance(settings, Settings)


class TestReloadSettings:
    """Tests for reload_settings function."""

    @pytest.mark.skip(reason="Test incompatible with test settings patch in conftest.py")
    def test_clears_cache(self):
        """Reload clears settings cache."""
        with patch("callwhisper.core.config.load_config_file") as mock_load:
            mock_load.return_value = {}
            get_settings.cache_clear()

            # First call
            settings1 = get_settings()

            # Modify mock return
            mock_load.return_value = {"version": "3.0.0"}

            # Regular get_settings should return cached
            # But reload should get new value
            settings2 = reload_settings()

            # reload_settings should trigger fresh load
            assert mock_load.call_count >= 2


# ============================================================================
# Validation Edge Case Tests (Test Suite Expansion)
# ============================================================================


class TestServerConfigValidation:
    """Tests for ServerConfig input validation edge cases."""

    def test_port_zero(self):
        """Port zero should be accepted (ephemeral port)."""
        config = ServerConfig(port=0)
        assert config.port == 0

    def test_port_max_valid(self):
        """Maximum valid port (65535) should work."""
        config = ServerConfig(port=65535)
        assert config.port == 65535

    def test_port_negative_rejected(self):
        """Negative port numbers should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ServerConfig(port=-1)

    def test_port_too_large_rejected(self):
        """Port numbers > 65535 should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ServerConfig(port=65536)

    def test_port_string_coercion(self):
        """String port should be coerced to int."""
        config = ServerConfig(port="8080")
        assert config.port == 8080
        assert isinstance(config.port, int)

    def test_port_invalid_string_rejected(self):
        """Non-numeric string port should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ServerConfig(port="invalid")

    def test_host_empty_string(self):
        """Empty host string should be accepted."""
        config = ServerConfig(host="")
        assert config.host == ""


class TestAudioConfigValidation:
    """Tests for AudioConfig input validation edge cases."""

    def test_sample_rate_zero(self):
        """Zero sample rate - Pydantic accepts but may be invalid for use."""
        config = AudioConfig(sample_rate=0)
        assert config.sample_rate == 0

    def test_sample_rate_negative_rejected(self):
        """Negative sample rate should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=-44100)

    def test_channels_zero(self):
        """Zero channels - accepted by Pydantic."""
        config = AudioConfig(channels=0)
        assert config.channels == 0

    def test_channels_negative_rejected(self):
        """Negative channels should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AudioConfig(channels=-1)

    def test_sample_rate_very_high(self):
        """Very high sample rate should be accepted."""
        config = AudioConfig(sample_rate=192000)
        assert config.sample_rate == 192000

    def test_sample_rate_string_coercion(self):
        """String sample rate should be coerced to int."""
        config = AudioConfig(sample_rate="48000")
        assert config.sample_rate == 48000

    def test_format_arbitrary_string(self):
        """Arbitrary format string should be accepted (no enum)."""
        config = AudioConfig(format="custom_format_xyz")
        assert config.format == "custom_format_xyz"


class TestTranscriptionConfigValidation:
    """Tests for TranscriptionConfig input validation edge cases."""

    def test_language_arbitrary_code(self):
        """Arbitrary language code should be accepted."""
        config = TranscriptionConfig(language="xyz")
        assert config.language == "xyz"

    def test_language_empty_string(self):
        """Empty language code should be accepted."""
        config = TranscriptionConfig(language="")
        assert config.language == ""

    def test_beam_size_zero(self):
        """Beam size zero - accepted by Pydantic."""
        config = TranscriptionConfig(beam_size=0)
        assert config.beam_size == 0

    def test_beam_size_negative_rejected(self):
        """Negative beam size should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TranscriptionConfig(beam_size=-1)

    def test_best_of_negative_rejected(self):
        """Negative best_of should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TranscriptionConfig(best_of=-5)

    def test_model_path_with_special_chars(self):
        """Model path with special characters should be accepted."""
        config = TranscriptionConfig(model="models/ggml-base.en (v2).bin")
        assert "v2" in config.model


class TestPerformanceConfigValidation:
    """Tests for PerformanceConfig input validation edge cases."""

    def test_pool_size_zero(self):
        """Zero pool size - accepted by Pydantic."""
        config = PerformanceConfig(audio_pool_size=0)
        assert config.audio_pool_size == 0

    def test_pool_size_negative_rejected(self):
        """Negative pool size should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceConfig(transcription_pool_size=-1)

    def test_cache_ttl_zero(self):
        """Zero cache TTL should be accepted."""
        config = PerformanceConfig(cache_ttl_seconds=0)
        assert config.cache_ttl_seconds == 0

    def test_cache_ttl_negative_rejected(self):
        """Negative cache TTL should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceConfig(cache_ttl_seconds=-3600)

    def test_cache_max_entries_zero(self):
        """Zero max entries should be accepted."""
        config = PerformanceConfig(cache_max_entries=0)
        assert config.cache_max_entries == 0

    def test_cache_max_entries_negative_rejected(self):
        """Negative max entries should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceConfig(cache_max_entries=-10)

    def test_chunk_size_zero(self):
        """Zero chunk size should be accepted."""
        config = PerformanceConfig(chunk_size_seconds=0.0)
        assert config.chunk_size_seconds == 0.0

    def test_chunk_size_negative_rejected(self):
        """Negative chunk size should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceConfig(chunk_size_seconds=-30.0)

    def test_max_concurrent_negative_rejected(self):
        """Negative max concurrent transcriptions rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerformanceConfig(max_concurrent_transcriptions=-4)


class TestSecurityConfigValidation:
    """Tests for SecurityConfig input validation edge cases."""

    def test_rate_limit_rpm_zero(self):
        """Zero requests per minute should be accepted."""
        config = SecurityConfig(rate_limit_rpm=0)
        assert config.rate_limit_rpm == 0

    def test_rate_limit_rpm_negative_rejected(self):
        """Negative rate limit RPM should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SecurityConfig(rate_limit_rpm=-60)

    def test_rate_limit_burst_zero(self):
        """Zero burst size should be accepted."""
        config = SecurityConfig(rate_limit_burst=0)
        assert config.rate_limit_burst == 0

    def test_rate_limit_burst_negative_rejected(self):
        """Negative burst size should be rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SecurityConfig(rate_limit_burst=-10)

    def test_empty_allowed_origins(self):
        """Empty allowed origins list should be accepted."""
        config = SecurityConfig(allowed_origins=[])
        assert config.allowed_origins == []

    def test_empty_excluded_paths(self):
        """Empty excluded paths list should be accepted."""
        config = SecurityConfig(rate_limit_excluded=[])
        assert config.rate_limit_excluded == []


class TestDeviceGuardConfigValidation:
    """Tests for DeviceGuardConfig input validation edge cases."""

    def test_empty_allowlist(self):
        """Empty allowlist should be accepted."""
        config = DeviceGuardConfig(allowlist=[])
        assert config.allowlist == []

    def test_empty_blocklist(self):
        """Empty blocklist should be accepted."""
        config = DeviceGuardConfig(blocklist=[])
        assert config.blocklist == []

    def test_unicode_device_names(self):
        """Unicode device names should be accepted."""
        config = DeviceGuardConfig(
            allowlist=["マイク", "Mikrofon", "麦克风"]
        )
        assert "マイク" in config.allowlist
        assert "Mikrofon" in config.allowlist

    def test_special_chars_in_device_names(self):
        """Special characters in device names should be accepted."""
        config = DeviceGuardConfig(
            allowlist=["Device (v2.0)", "Mic & Speaker", "Audio<>Out"]
        )
        assert len(config.allowlist) == 3


class TestLoadConfigFileEdgeCases:
    """Tests for load_config_file edge cases."""

    def test_malformed_json_raises_error(self, tmp_path):
        """Malformed JSON should raise JSONDecodeError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{"version": "1.0.0",}')  # Trailing comma

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            with pytest.raises(json.JSONDecodeError):
                load_config_file()

    def test_incomplete_json_raises_error(self, tmp_path):
        """Incomplete JSON should raise JSONDecodeError."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{"version": ')  # Incomplete

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            with pytest.raises(json.JSONDecodeError):
                load_config_file()

    def test_config_with_unknown_fields(self, tmp_path):
        """Config with unknown top-level fields should not break loading."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{"version": "2.0.0", "unknown_field": "value"}')

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            result = load_config_file()

        assert result["version"] == "2.0.0"
        assert result["unknown_field"] == "value"

    def test_empty_config_file(self, tmp_path):
        """Empty JSON object should work."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text('{}')

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            result = load_config_file()

        assert result == {}

    def test_config_with_nested_unknown_fields(self, tmp_path):
        """Config with unknown nested fields should still load."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_data = {
            "server": {"host": "0.0.0.0", "unknown_nested": True}
        }
        config_file.write_text(json.dumps(config_data))

        with patch("callwhisper.core.config.get_data_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            result = load_config_file()

        assert result["server"]["host"] == "0.0.0.0"


class TestSettingsWithPartialConfig:
    """Tests for Settings loading with partial configuration."""

    def test_partial_server_config(self):
        """Settings with partial server config should use defaults for rest."""
        settings = Settings(server=ServerConfig(port=9000))
        assert settings.server.port == 9000
        assert settings.server.host == "127.0.0.1"  # Default
        assert settings.audio.sample_rate == 44100  # Default

    def test_partial_nested_override(self):
        """Can override some nested values while keeping others default."""
        settings = Settings(
            performance=PerformanceConfig(cache_enabled=False)
        )
        assert settings.performance.cache_enabled is False
        assert settings.performance.cache_ttl_seconds == 3600  # Default
        assert settings.performance.audio_pool_size == 2  # Default

    def test_all_defaults(self):
        """All defaults should work without any config."""
        settings = Settings()
        assert settings.version == "1.0.0"
        assert settings.server.port == 8765
        assert settings.audio.sample_rate == 44100
        assert settings.transcription.language == "en"
        assert settings.device_guard.enabled is True
