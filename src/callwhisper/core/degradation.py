"""
Graceful Degradation Module

Based on LibV2 orchestrator-architecture course:
- Explicit quality/speed tradeoffs under load
- Progressive feature reduction as load increases
- Load-based degradation level selection
"""

import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from collections import deque

from .logging_config import get_logger

logger = get_logger(__name__)


class DegradationLevel(str, Enum):
    """
    Degradation levels for quality/speed tradeoffs.

    Based on LibV2 orchestrator-architecture course:
    - FULL: All enhancements enabled, highest quality
    - BALANCED: Some enhancements disabled for performance
    - FAST: Minimal processing for maximum throughput
    """
    FULL = "full"
    BALANCED = "balanced"
    FAST = "fast"


@dataclass
class DegradationConfig:
    """Configuration for degradation thresholds."""
    # Queue depth thresholds
    balanced_threshold: int = 5   # Switch to BALANCED when queue > this
    fast_threshold: int = 15      # Switch to FAST when queue > this

    # Latency thresholds (ms)
    balanced_latency_ms: float = 5000.0   # Switch to BALANCED when p95 > this
    fast_latency_ms: float = 15000.0      # Switch to FAST when p95 > this

    # Hysteresis - prevents rapid oscillation
    hysteresis_window_seconds: float = 30.0  # Consider last N seconds
    level_change_cooldown: float = 10.0      # Minimum seconds between level changes


@dataclass
class LoadMetrics:
    """Real-time load metrics for degradation decisions."""
    queue_depth: int = 0
    recent_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def p95_latency_ms(self) -> float:
        """Calculate 95th percentile latency."""
        if not self.recent_latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.recent_latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if not self.recent_latencies_ms:
            return 0.0
        return sum(self.recent_latencies_ms) / len(self.recent_latencies_ms)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.error_count / total


class DegradationManager:
    """
    Manages degradation levels based on system load.

    Based on LibV2 orchestrator-architecture course:
    - Monitors queue depth and latency metrics
    - Applies hysteresis to prevent oscillation
    - Provides degradation-aware configuration to pipeline
    """

    def __init__(self, config: Optional[DegradationConfig] = None):
        """
        Initialize the degradation manager.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self.config = config or DegradationConfig()
        self._current_level = DegradationLevel.FULL
        self._last_level_change: float = 0.0
        self._metrics = LoadMetrics()
        self._lock = threading.Lock()
        self._listeners: list[Callable[[DegradationLevel, DegradationLevel], None]] = []

    def record_request_start(self) -> None:
        """Record a request entering the queue."""
        with self._lock:
            self._metrics.queue_depth += 1
            self._metrics.last_updated = time.time()
            self._maybe_update_level()

    def record_request_end(self, latency_ms: float, success: bool = True) -> None:
        """
        Record a request completion.

        Args:
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
        """
        with self._lock:
            self._metrics.queue_depth = max(0, self._metrics.queue_depth - 1)
            self._metrics.recent_latencies_ms.append(latency_ms)
            if success:
                self._metrics.success_count += 1
            else:
                self._metrics.error_count += 1
            self._metrics.last_updated = time.time()
            self._maybe_update_level()

    def _maybe_update_level(self) -> None:
        """Check if degradation level should change."""
        now = time.time()

        # Respect cooldown period
        if now - self._last_level_change < self.config.level_change_cooldown:
            return

        new_level = self._calculate_level()

        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            self._last_level_change = now

            logger.info(
                "degradation_level_changed",
                old_level=old_level.value,
                new_level=new_level.value,
                queue_depth=self._metrics.queue_depth,
                p95_latency_ms=round(self._metrics.p95_latency_ms, 2)
            )

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(old_level, new_level)
                except Exception as e:
                    logger.error("degradation_listener_error", error=str(e))

    def _calculate_level(self) -> DegradationLevel:
        """Calculate the appropriate degradation level."""
        queue_depth = self._metrics.queue_depth
        p95_latency = self._metrics.p95_latency_ms

        # Check FAST thresholds (highest load)
        if (queue_depth > self.config.fast_threshold or
            p95_latency > self.config.fast_latency_ms):
            return DegradationLevel.FAST

        # Check BALANCED thresholds (moderate load)
        if (queue_depth > self.config.balanced_threshold or
            p95_latency > self.config.balanced_latency_ms):
            return DegradationLevel.BALANCED

        # Default to FULL quality
        return DegradationLevel.FULL

    def get_current_level(self) -> DegradationLevel:
        """Get the current degradation level."""
        with self._lock:
            return self._current_level

    def get_settings_for_level(self, level: Optional[DegradationLevel] = None) -> Dict[str, Any]:
        """
        Get processing settings appropriate for degradation level.

        Args:
            level: Level to get settings for (uses current if None)

        Returns:
            Dictionary of processing settings
        """
        level = level or self.get_current_level()

        settings = {
            DegradationLevel.FULL: {
                "model_size": "large",
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "enable_diarization": True,
                "enable_noise_reduction": True,
                "compression_level": "high"
            },
            DegradationLevel.BALANCED: {
                "model_size": "medium",
                "beam_size": 3,
                "best_of": 3,
                "temperature": 0.0,
                "enable_diarization": True,
                "enable_noise_reduction": False,
                "compression_level": "medium"
            },
            DegradationLevel.FAST: {
                "model_size": "small",
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.0,
                "enable_diarization": False,
                "enable_noise_reduction": False,
                "compression_level": "low"
            }
        }

        return settings.get(level, settings[DegradationLevel.FULL])

    def add_level_change_listener(
        self,
        listener: Callable[[DegradationLevel, DegradationLevel], None]
    ) -> None:
        """
        Add a listener for degradation level changes.

        Args:
            listener: Callback function(old_level, new_level)
        """
        with self._lock:
            self._listeners.append(listener)

    def remove_level_change_listener(
        self,
        listener: Callable[[DegradationLevel, DegradationLevel], None]
    ) -> None:
        """Remove a level change listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def force_level(self, level: DegradationLevel) -> None:
        """
        Force a specific degradation level (for testing/override).

        Args:
            level: Level to force
        """
        with self._lock:
            old_level = self._current_level
            self._current_level = level
            self._last_level_change = time.time()

            logger.warning(
                "degradation_level_forced",
                old_level=old_level.value,
                new_level=level.value
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                "current_level": self._current_level.value,
                "queue_depth": self._metrics.queue_depth,
                "p95_latency_ms": round(self._metrics.p95_latency_ms, 2),
                "avg_latency_ms": round(self._metrics.avg_latency_ms, 2),
                "error_rate": round(self._metrics.error_rate, 4),
                "success_count": self._metrics.success_count,
                "error_count": self._metrics.error_count,
                "last_level_change": self._last_level_change,
                "config": {
                    "balanced_threshold": self.config.balanced_threshold,
                    "fast_threshold": self.config.fast_threshold,
                    "balanced_latency_ms": self.config.balanced_latency_ms,
                    "fast_latency_ms": self.config.fast_latency_ms
                }
            }


# Global degradation manager instance
degradation_manager = DegradationManager()
