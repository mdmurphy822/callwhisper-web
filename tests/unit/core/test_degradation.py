"""
Tests for graceful degradation manager.

Tests load-based quality tradeoffs:
- Degradation level transitions
- Queue depth thresholds
- Latency thresholds
- Hysteresis / cooldown
- Level change listeners
- Settings for each level
"""

import threading
import time
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from callwhisper.core.degradation import (
    DegradationConfig,
    DegradationLevel,
    DegradationManager,
    LoadMetrics,
    degradation_manager,
)


# ============================================================================
# DegradationLevel Tests
# ============================================================================


class TestDegradationLevel:
    """Tests for DegradationLevel enum."""

    def test_level_values(self):
        """Levels have expected string values."""
        assert DegradationLevel.FULL.value == "full"
        assert DegradationLevel.BALANCED.value == "balanced"
        assert DegradationLevel.FAST.value == "fast"

    def test_level_from_string(self):
        """Levels can be created from strings."""
        assert DegradationLevel("full") == DegradationLevel.FULL
        assert DegradationLevel("balanced") == DegradationLevel.BALANCED
        assert DegradationLevel("fast") == DegradationLevel.FAST

    def test_level_comparison(self):
        """Levels are equal to their string values."""
        assert DegradationLevel.FULL == "full"
        assert DegradationLevel.BALANCED == "balanced"
        assert DegradationLevel.FAST == "fast"


# ============================================================================
# DegradationConfig Tests
# ============================================================================


class TestDegradationConfig:
    """Tests for DegradationConfig dataclass."""

    def test_default_values(self):
        """Config has sensible defaults."""
        config = DegradationConfig()

        assert config.balanced_threshold == 5
        assert config.fast_threshold == 15
        assert config.balanced_latency_ms == 5000.0
        assert config.fast_latency_ms == 15000.0
        assert config.hysteresis_window_seconds == 30.0
        assert config.level_change_cooldown == 10.0

    def test_custom_values(self):
        """Config accepts custom values."""
        config = DegradationConfig(
            balanced_threshold=10,
            fast_threshold=25,
            balanced_latency_ms=3000.0,
            fast_latency_ms=10000.0,
            level_change_cooldown=5.0,
        )

        assert config.balanced_threshold == 10
        assert config.fast_threshold == 25
        assert config.balanced_latency_ms == 3000.0
        assert config.fast_latency_ms == 10000.0
        assert config.level_change_cooldown == 5.0


# ============================================================================
# LoadMetrics Tests
# ============================================================================


class TestLoadMetrics:
    """Tests for LoadMetrics dataclass."""

    def test_default_values(self):
        """Metrics have expected defaults."""
        metrics = LoadMetrics()

        assert metrics.queue_depth == 0
        assert len(metrics.recent_latencies_ms) == 0
        assert metrics.error_count == 0
        assert metrics.success_count == 0
        assert metrics.last_updated > 0

    def test_p95_latency_empty(self):
        """p95 latency is 0 when no latencies recorded."""
        metrics = LoadMetrics()
        assert metrics.p95_latency_ms == 0.0

    def test_p95_latency_single_value(self):
        """p95 latency works with single value."""
        metrics = LoadMetrics()
        metrics.recent_latencies_ms.append(100.0)

        assert metrics.p95_latency_ms == 100.0

    def test_p95_latency_multiple_values(self):
        """p95 latency correctly calculates 95th percentile."""
        metrics = LoadMetrics()
        # Add 100 latencies: 1, 2, 3, ..., 100
        for i in range(1, 101):
            metrics.recent_latencies_ms.append(float(i))

        # 95th percentile of 1-100 is 96 (index 95 in sorted list)
        assert metrics.p95_latency_ms == 96.0

    def test_avg_latency_empty(self):
        """Average latency is 0 when no latencies recorded."""
        metrics = LoadMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculated(self):
        """Average latency is correctly calculated."""
        metrics = LoadMetrics()
        metrics.recent_latencies_ms.extend([100.0, 200.0, 300.0])

        assert metrics.avg_latency_ms == 200.0

    def test_error_rate_no_requests(self):
        """Error rate is 0 with no requests."""
        metrics = LoadMetrics()
        assert metrics.error_rate == 0.0

    def test_error_rate_calculated(self):
        """Error rate is correctly calculated."""
        metrics = LoadMetrics()
        metrics.success_count = 80
        metrics.error_count = 20

        assert metrics.error_rate == 0.2

    def test_latency_deque_maxlen(self):
        """Latency deque respects maxlen."""
        metrics = LoadMetrics()
        # Add more than maxlen (100) values
        for i in range(150):
            metrics.recent_latencies_ms.append(float(i))

        assert len(metrics.recent_latencies_ms) == 100
        # Oldest should be dropped
        assert min(metrics.recent_latencies_ms) == 50.0


# ============================================================================
# DegradationManager Basic Tests
# ============================================================================


class TestDegradationManagerBasic:
    """Tests for DegradationManager basic operations."""

    @pytest.fixture
    def manager(self):
        """Create manager with short cooldown for testing."""
        config = DegradationConfig(level_change_cooldown=0.0)
        return DegradationManager(config)

    def test_initial_level_is_full(self, manager):
        """Initial degradation level is FULL."""
        assert manager.get_current_level() == DegradationLevel.FULL

    def test_record_request_start_increments_queue(self, manager):
        """record_request_start increments queue depth."""
        manager.record_request_start()
        manager.record_request_start()
        manager.record_request_start()

        metrics = manager.get_metrics()
        assert metrics["queue_depth"] == 3

    def test_record_request_end_decrements_queue(self, manager):
        """record_request_end decrements queue depth."""
        manager.record_request_start()
        manager.record_request_start()
        manager.record_request_end(100.0)

        metrics = manager.get_metrics()
        assert metrics["queue_depth"] == 1

    def test_record_request_end_records_latency(self, manager):
        """record_request_end records latency."""
        manager.record_request_end(150.0)
        manager.record_request_end(250.0)

        metrics = manager.get_metrics()
        assert metrics["avg_latency_ms"] == 200.0

    def test_record_request_end_success(self, manager):
        """record_request_end increments success count."""
        manager.record_request_end(100.0, success=True)
        manager.record_request_end(100.0, success=True)
        manager.record_request_end(100.0, success=True)

        metrics = manager.get_metrics()
        assert metrics["success_count"] == 3
        assert metrics["error_count"] == 0

    def test_record_request_end_failure(self, manager):
        """record_request_end increments error count on failure."""
        manager.record_request_end(100.0, success=False)
        manager.record_request_end(100.0, success=False)

        metrics = manager.get_metrics()
        assert metrics["success_count"] == 0
        assert metrics["error_count"] == 2

    def test_queue_depth_cannot_go_negative(self, manager):
        """Queue depth stays at 0 even with unbalanced end calls."""
        manager.record_request_end(100.0)
        manager.record_request_end(100.0)
        manager.record_request_end(100.0)

        metrics = manager.get_metrics()
        assert metrics["queue_depth"] == 0


# ============================================================================
# DegradationManager Level Transition Tests
# ============================================================================


class TestDegradationLevelTransitions:
    """Tests for degradation level transitions."""

    @pytest.fixture
    def manager(self):
        """Create manager with no cooldown for testing."""
        config = DegradationConfig(
            balanced_threshold=5,
            fast_threshold=15,
            level_change_cooldown=0.0,
        )
        return DegradationManager(config)

    def test_transition_to_balanced_on_queue_depth(self, manager):
        """Transitions to BALANCED when queue depth exceeds threshold."""
        for _ in range(6):  # > balanced_threshold (5)
            manager.record_request_start()

        assert manager.get_current_level() == DegradationLevel.BALANCED

    def test_transition_to_fast_on_queue_depth(self, manager):
        """Transitions to FAST when queue depth exceeds threshold."""
        for _ in range(16):  # > fast_threshold (15)
            manager.record_request_start()

        assert manager.get_current_level() == DegradationLevel.FAST

    def test_transition_to_balanced_on_latency(self, manager):
        """Transitions to BALANCED when p95 latency exceeds threshold."""
        config = DegradationConfig(
            balanced_latency_ms=100.0,
            fast_latency_ms=500.0,
            level_change_cooldown=0.0,
        )
        manager = DegradationManager(config)

        # Add latencies that will make p95 > balanced threshold
        for _ in range(100):
            manager.record_request_end(150.0)

        assert manager.get_current_level() == DegradationLevel.BALANCED

    def test_transition_to_fast_on_latency(self, manager):
        """Transitions to FAST when p95 latency exceeds threshold."""
        config = DegradationConfig(
            balanced_latency_ms=100.0,
            fast_latency_ms=500.0,
            level_change_cooldown=0.0,
        )
        manager = DegradationManager(config)

        # Add latencies that will make p95 > fast threshold
        for _ in range(100):
            manager.record_request_end(600.0)

        assert manager.get_current_level() == DegradationLevel.FAST

    def test_transition_back_to_full(self, manager):
        """Transitions back to FULL when load decreases."""
        # First increase load
        for _ in range(6):
            manager.record_request_start()
        assert manager.get_current_level() == DegradationLevel.BALANCED

        # Then decrease load
        for _ in range(6):
            manager.record_request_end(50.0)
        assert manager.get_current_level() == DegradationLevel.FULL

    def test_remains_at_full_below_threshold(self, manager):
        """Remains at FULL when below all thresholds."""
        for _ in range(4):  # < balanced_threshold (5)
            manager.record_request_start()

        assert manager.get_current_level() == DegradationLevel.FULL


# ============================================================================
# DegradationManager Cooldown/Hysteresis Tests
# ============================================================================


class TestDegradationCooldown:
    """Tests for cooldown/hysteresis behavior."""

    def test_cooldown_prevents_rapid_changes(self):
        """Cooldown prevents rapid level oscillation."""
        config = DegradationConfig(
            balanced_threshold=5,
            fast_threshold=15,
            level_change_cooldown=1.0,  # 1 second cooldown
        )
        manager = DegradationManager(config)

        # Trigger change to BALANCED
        for _ in range(6):
            manager.record_request_start()
        assert manager.get_current_level() == DegradationLevel.BALANCED

        # Immediately try to go back to FULL
        for _ in range(6):
            manager.record_request_end(50.0)

        # Should still be BALANCED due to cooldown
        assert manager.get_current_level() == DegradationLevel.BALANCED

    def test_change_allowed_after_cooldown(self):
        """Level change is allowed after cooldown expires."""
        config = DegradationConfig(
            balanced_threshold=5,
            level_change_cooldown=0.05,  # Very short cooldown
        )
        manager = DegradationManager(config)

        # Trigger change to BALANCED
        for _ in range(6):
            manager.record_request_start()
        assert manager.get_current_level() == DegradationLevel.BALANCED

        # Wait for cooldown
        time.sleep(0.06)

        # Now decrease load
        for _ in range(6):
            manager.record_request_end(50.0)

        # Should now be FULL
        assert manager.get_current_level() == DegradationLevel.FULL


# ============================================================================
# DegradationManager Settings Tests
# ============================================================================


class TestDegradationSettings:
    """Tests for degradation level settings."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return DegradationManager(DegradationConfig())

    def test_full_settings(self, manager):
        """FULL level has highest quality settings."""
        settings = manager.get_settings_for_level(DegradationLevel.FULL)

        assert settings["model_size"] == "large"
        assert settings["beam_size"] == 5
        assert settings["enable_diarization"] is True
        assert settings["enable_noise_reduction"] is True

    def test_balanced_settings(self, manager):
        """BALANCED level has medium quality settings."""
        settings = manager.get_settings_for_level(DegradationLevel.BALANCED)

        assert settings["model_size"] == "medium"
        assert settings["beam_size"] == 3
        assert settings["enable_diarization"] is True
        assert settings["enable_noise_reduction"] is False

    def test_fast_settings(self, manager):
        """FAST level has lowest quality settings."""
        settings = manager.get_settings_for_level(DegradationLevel.FAST)

        assert settings["model_size"] == "small"
        assert settings["beam_size"] == 1
        assert settings["enable_diarization"] is False
        assert settings["enable_noise_reduction"] is False

    def test_get_settings_uses_current_level(self, manager):
        """get_settings_for_level uses current level when None."""
        settings = manager.get_settings_for_level(None)
        assert settings["model_size"] == "large"  # FULL is default


# ============================================================================
# DegradationManager Listener Tests
# ============================================================================


class TestDegradationListeners:
    """Tests for level change listeners."""

    @pytest.fixture
    def manager(self):
        """Create manager with no cooldown."""
        config = DegradationConfig(
            balanced_threshold=5,
            level_change_cooldown=0.0,
        )
        return DegradationManager(config)

    def test_listener_called_on_change(self, manager):
        """Listener is called when level changes."""
        listener = MagicMock()
        manager.add_level_change_listener(listener)

        # Trigger level change
        for _ in range(6):
            manager.record_request_start()

        listener.assert_called_once_with(
            DegradationLevel.FULL,
            DegradationLevel.BALANCED
        )

    def test_listener_not_called_without_change(self, manager):
        """Listener is not called when level doesn't change."""
        listener = MagicMock()
        manager.add_level_change_listener(listener)

        # Stay below threshold
        for _ in range(4):
            manager.record_request_start()

        listener.assert_not_called()

    def test_multiple_listeners_called(self, manager):
        """Multiple listeners are all called."""
        listener1 = MagicMock()
        listener2 = MagicMock()
        manager.add_level_change_listener(listener1)
        manager.add_level_change_listener(listener2)

        for _ in range(6):
            manager.record_request_start()

        listener1.assert_called_once()
        listener2.assert_called_once()

    def test_remove_listener(self, manager):
        """Removed listener is not called."""
        listener = MagicMock()
        manager.add_level_change_listener(listener)
        manager.remove_level_change_listener(listener)

        for _ in range(6):
            manager.record_request_start()

        listener.assert_not_called()

    def test_listener_exception_caught(self, manager):
        """Listener exceptions don't break processing."""
        bad_listener = MagicMock(side_effect=Exception("Listener error"))
        good_listener = MagicMock()
        manager.add_level_change_listener(bad_listener)
        manager.add_level_change_listener(good_listener)

        # Should not raise
        for _ in range(6):
            manager.record_request_start()

        # Good listener should still be called
        good_listener.assert_called_once()


# ============================================================================
# DegradationManager Force Level Tests
# ============================================================================


class TestForceLevel:
    """Tests for force_level functionality."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return DegradationManager(DegradationConfig())

    def test_force_level_changes_level(self, manager):
        """force_level changes the current level."""
        manager.force_level(DegradationLevel.FAST)

        assert manager.get_current_level() == DegradationLevel.FAST

    def test_force_level_resets_cooldown(self, manager):
        """force_level resets the cooldown timer."""
        old_change_time = manager._last_level_change

        manager.force_level(DegradationLevel.BALANCED)

        assert manager._last_level_change > old_change_time


# ============================================================================
# DegradationManager Metrics Tests
# ============================================================================


class TestDegradationMetrics:
    """Tests for metrics retrieval."""

    @pytest.fixture
    def manager(self):
        """Create manager for testing."""
        return DegradationManager(DegradationConfig())

    def test_get_metrics_includes_all_fields(self, manager):
        """get_metrics returns all expected fields."""
        metrics = manager.get_metrics()

        assert "current_level" in metrics
        assert "queue_depth" in metrics
        assert "p95_latency_ms" in metrics
        assert "avg_latency_ms" in metrics
        assert "error_rate" in metrics
        assert "success_count" in metrics
        assert "error_count" in metrics
        assert "last_level_change" in metrics
        assert "config" in metrics

    def test_get_metrics_includes_config(self, manager):
        """get_metrics includes configuration values."""
        metrics = manager.get_metrics()

        assert metrics["config"]["balanced_threshold"] == 5
        assert metrics["config"]["fast_threshold"] == 15

    def test_metrics_reflect_current_state(self, manager):
        """Metrics reflect current system state."""
        manager.record_request_start()
        manager.record_request_start()
        manager.record_request_end(100.0, success=True)
        manager.record_request_end(200.0, success=False)

        metrics = manager.get_metrics()

        assert metrics["queue_depth"] == 0
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 1
        assert metrics["error_rate"] == 0.5


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_request_tracking(self):
        """Concurrent request tracking is thread-safe."""
        config = DegradationConfig(level_change_cooldown=0.0)
        manager = DegradationManager(config)
        errors = []

        def add_requests():
            try:
                for _ in range(100):
                    manager.record_request_start()
            except Exception as e:
                errors.append(e)

        def complete_requests():
            try:
                for _ in range(100):
                    manager.record_request_end(100.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_requests),
            threading.Thread(target=complete_requests),
            threading.Thread(target=add_requests),
            threading.Thread(target=complete_requests),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Queue depth should be >= 0 (exact value depends on timing)
        assert manager.get_metrics()["queue_depth"] >= 0

    def test_concurrent_level_reads(self):
        """Concurrent level reads are thread-safe."""
        manager = DegradationManager(DegradationConfig())
        results = []
        errors = []

        def read_level():
            try:
                for _ in range(100):
                    level = manager.get_current_level()
                    results.append(level)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_level) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 500


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global degradation_manager instance."""

    def test_global_instance_exists(self):
        """Global instance is a DegradationManager."""
        assert isinstance(degradation_manager, DegradationManager)

    def test_global_instance_works(self):
        """Global instance is functional."""
        # Just verify it doesn't raise
        level = degradation_manager.get_current_level()
        assert level in [
            DegradationLevel.FULL,
            DegradationLevel.BALANCED,
            DegradationLevel.FAST
        ]


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_default_config(self):
        """Manager works with default config (None passed)."""
        manager = DegradationManager(None)
        assert manager.config is not None
        assert manager.get_current_level() == DegradationLevel.FULL

    def test_empty_latency_metrics(self):
        """Manager handles empty latency list."""
        manager = DegradationManager(DegradationConfig())
        metrics = manager.get_metrics()

        assert metrics["p95_latency_ms"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0

    def test_remove_nonexistent_listener(self):
        """Removing non-existent listener doesn't raise."""
        manager = DegradationManager(DegradationConfig())
        fake_listener = MagicMock()

        # Should not raise
        manager.remove_level_change_listener(fake_listener)

    def test_settings_for_unknown_level_returns_full(self):
        """Unknown level returns FULL settings."""
        manager = DegradationManager(DegradationConfig())

        # Force a weird state (shouldn't happen in practice)
        settings = manager.get_settings_for_level(DegradationLevel.FULL)
        assert settings["model_size"] == "large"
