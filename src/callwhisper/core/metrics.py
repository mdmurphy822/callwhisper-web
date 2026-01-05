"""
Metrics Collection Module

Based on LibV2 orchestrator-architecture course:
- Track operation durations and counts
- Calculate percentiles for latency monitoring
- Thread-safe metrics collection
- Persistent transcription history for analytics
"""

import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime, timedelta

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionMetric:
    """
    Record of a single transcription job.

    Persisted to disk for analytics and reporting.
    """

    job_id: str
    audio_duration_seconds: float
    transcription_duration_seconds: float
    model_used: str
    success: bool
    error_message: Optional[str] = None
    device_name: Optional[str] = None
    ticket_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionMetric":
        return cls(**data)

    @property
    def processing_speed(self) -> float:
        """Ratio of audio duration to processing time (higher = faster)."""
        if self.transcription_duration_seconds == 0:
            return 0.0
        return self.audio_duration_seconds / self.transcription_duration_seconds


@dataclass
class DailyStats:
    """Aggregated stats for a single day."""

    date: str
    transcription_count: int
    total_audio_seconds: float
    total_processing_seconds: float
    success_count: int
    failure_count: int

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class MetricsSummary:
    """Aggregate summary of transcription metrics."""

    total_transcriptions: int
    total_audio_hours: float
    total_processing_hours: float
    avg_processing_speed: float
    success_rate: float
    last_7_days: List[DailyStats]


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""

    count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    durations: List[float] = field(default_factory=list)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    last_operation_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.success_count / self.count

    @property
    def error_rate(self) -> float:
        if self.count == 0:
            return 0.0
        return self.error_count / self.count

    @property
    def avg_duration_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_duration_ms / self.count


class MetricsCollector:
    """
    Thread-safe metrics collector for application observability.

    Tracks:
    - Transcription operations (count, duration, success/failure)
    - Recording operations
    - API request latencies
    - Error rates by type
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._operations: Dict[str, OperationMetrics] = {}
        self._start_time = time.time()

    def _get_or_create_operation(self, operation_name: str) -> OperationMetrics:
        """Get or create metrics for an operation."""
        if operation_name not in self._operations:
            self._operations[operation_name] = OperationMetrics()
        return self._operations[operation_name]

    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate the Nth percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        # Use (len-1) for proper 0-indexed array access
        index = int((len(sorted_values) - 1) * percentile / 100)
        return sorted_values[index]

    def record_operation(
        self,
        operation_name: str,
        duration_ms: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """
        Record an operation completion.

        Args:
            operation_name: Name of the operation (e.g., 'transcription', 'recording')
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            error_type: Type of error if failed (e.g., 'TimeoutError', 'TranscriptionError')
        """
        with self._lock:
            op = self._get_or_create_operation(operation_name)
            op.count += 1
            op.total_duration_ms += duration_ms
            # Keep only the last 1000 durations for percentile calculations
            if len(op.durations) >= 1000:
                op.durations.pop(0)
            op.durations.append(duration_ms)
            op.last_operation_time = time.time()

            if success:
                op.success_count += 1
            else:
                op.error_count += 1
                if error_type:
                    op.errors_by_type[error_type] = (
                        op.errors_by_type.get(error_type, 0) + 1
                    )

    def start_timer(self) -> float:
        """Start a timer for measuring operation duration."""
        return time.time()

    def stop_timer(self, start_time: float) -> float:
        """Stop timer and return duration in milliseconds."""
        return (time.time() - start_time) * 1000

    def get_operation_metrics(self, operation_name: str) -> Optional[Dict]:
        """Get metrics for a specific operation. Returns None if operation unknown."""
        with self._lock:
            if operation_name not in self._operations:
                return None
            op = self._operations[operation_name]
            return {
                "count": op.count,
                "success_count": op.success_count,
                "error_count": op.error_count,
                "success_rate": round(op.success_rate, 4),
                "error_rate": round(op.error_rate, 4),
                "avg_duration_ms": round(op.avg_duration_ms, 2),
                "p50_duration_ms": round(
                    self._calculate_percentile(op.durations, 50), 2
                ),
                "p95_duration_ms": round(
                    self._calculate_percentile(op.durations, 95), 2
                ),
                "p99_duration_ms": round(
                    self._calculate_percentile(op.durations, 99), 2
                ),
                "errors_by_type": dict(op.errors_by_type),
                "last_operation_time": op.last_operation_time,
            }

    def get_all_metrics(self) -> Dict:
        """Get metrics for all operations."""
        with self._lock:
            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "operations": {
                    name: {
                        "count": op.count,
                        "success_count": op.success_count,
                        "error_count": op.error_count,
                        "success_rate": round(op.success_rate, 4),
                        "error_rate": round(op.error_rate, 4),
                        "avg_duration_ms": round(op.avg_duration_ms, 2),
                        "p95_duration_ms": round(
                            self._calculate_percentile(op.durations, 95), 2
                        ),
                        "p99_duration_ms": round(
                            self._calculate_percentile(op.durations, 99), 2
                        ),
                    }
                    for name, op in self._operations.items()
                },
            }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._operations.clear()
            self._start_time = time.time()


class TranscriptionMetricsStore:
    """
    Persistent storage for transcription metrics.

    Saves transcription history to disk for analytics
    and reporting across app restarts.
    """

    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize metrics store.

        Args:
            store_path: Path for metrics file.
                       Defaults to ~/.callwhisper/metrics.json
        """
        if store_path is None:
            store_path = Path.home() / ".callwhisper" / "metrics.json"

        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metrics: List[TranscriptionMetric] = []
        self._load()

    def _load(self) -> None:
        """Load metrics from disk."""
        if not self.store_path.exists():
            return

        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._metrics = [
                TranscriptionMetric.from_dict(m) for m in data.get("transcriptions", [])
            ]

        except Exception as e:
            logger.warning(
                "metrics_load_failed",
                path=str(self.store_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            # If load fails, start fresh
            self._metrics = []

    def _save(self) -> None:
        """Save metrics to disk."""
        try:
            data = {
                "transcriptions": [m.to_dict() for m in self._metrics],
                "last_updated": time.time(),
            }

            # Write to temp file first
            temp_path = self.store_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.rename(self.store_path)

        except Exception as e:
            logger.error(
                "metrics_save_failed",
                path=str(self.store_path),
                error=str(e),
                error_type=type(e).__name__,
            )
            # Don't crash on save failure, but log the error

    def record(self, metric: TranscriptionMetric) -> None:
        """Record a transcription metric."""
        with self._lock:
            self._metrics.append(metric)
            self._save()

    def get_recent(self, limit: int = 50) -> List[TranscriptionMetric]:
        """Get recent transcription records."""
        with self._lock:
            # Sort by timestamp descending
            sorted_metrics = sorted(
                self._metrics, key=lambda m: m.timestamp, reverse=True
            )
            return sorted_metrics[:limit]

    def get_summary(self) -> MetricsSummary:
        """Get aggregate metrics summary."""
        with self._lock:
            if not self._metrics:
                return MetricsSummary(
                    total_transcriptions=0,
                    total_audio_hours=0.0,
                    total_processing_hours=0.0,
                    avg_processing_speed=0.0,
                    success_rate=1.0,
                    last_7_days=[],
                )

            total = len(self._metrics)
            success_count = sum(1 for m in self._metrics if m.success)
            total_audio = sum(m.audio_duration_seconds for m in self._metrics)
            total_processing = sum(
                m.transcription_duration_seconds for m in self._metrics
            )

            avg_speed = total_audio / total_processing if total_processing > 0 else 0.0

            # Calculate last 7 days
            now = datetime.now()
            daily_stats = []

            for days_ago in range(7):
                day = now - timedelta(days=days_ago)
                day_str = day.strftime("%Y-%m-%d")
                day_start = datetime(day.year, day.month, day.day).timestamp()
                day_end = day_start + 86400

                day_metrics = [
                    m for m in self._metrics if day_start <= m.timestamp < day_end
                ]

                if day_metrics:
                    daily_stats.append(
                        DailyStats(
                            date=day_str,
                            transcription_count=len(day_metrics),
                            total_audio_seconds=sum(
                                m.audio_duration_seconds for m in day_metrics
                            ),
                            total_processing_seconds=sum(
                                m.transcription_duration_seconds for m in day_metrics
                            ),
                            success_count=sum(1 for m in day_metrics if m.success),
                            failure_count=sum(1 for m in day_metrics if not m.success),
                        )
                    )
                else:
                    daily_stats.append(
                        DailyStats(
                            date=day_str,
                            transcription_count=0,
                            total_audio_seconds=0.0,
                            total_processing_seconds=0.0,
                            success_count=0,
                            failure_count=0,
                        )
                    )

            return MetricsSummary(
                total_transcriptions=total,
                total_audio_hours=total_audio / 3600,
                total_processing_hours=total_processing / 3600,
                avg_processing_speed=avg_speed,
                success_rate=success_count / total if total > 0 else 1.0,
                last_7_days=daily_stats,
            )

    def export_csv(self, output_path: Path) -> int:
        """
        Export metrics to CSV for reporting.

        Returns number of records exported.
        """
        with self._lock:
            if not self._metrics:
                return 0

            import csv

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "job_id",
                        "timestamp",
                        "audio_duration_seconds",
                        "transcription_duration_seconds",
                        "processing_speed",
                        "model_used",
                        "success",
                        "error_message",
                        "device_name",
                        "ticket_id",
                    ]
                )

                # Data rows
                for m in sorted(self._metrics, key=lambda x: x.timestamp):
                    writer.writerow(
                        [
                            m.job_id,
                            datetime.fromtimestamp(m.timestamp).isoformat(),
                            round(m.audio_duration_seconds, 2),
                            round(m.transcription_duration_seconds, 2),
                            round(m.processing_speed, 2),
                            m.model_used,
                            m.success,
                            m.error_message or "",
                            m.device_name or "",
                            m.ticket_id or "",
                        ]
                    )

            return len(self._metrics)

    def cleanup_old(self, max_age_days: int = 90) -> int:
        """
        Remove metrics older than max_age_days.

        Returns number of records removed.
        """
        cutoff = time.time() - (max_age_days * 24 * 60 * 60)

        with self._lock:
            original_count = len(self._metrics)
            self._metrics = [m for m in self._metrics if m.timestamp >= cutoff]
            removed = original_count - len(self._metrics)

            if removed > 0:
                self._save()

            return removed


# Global metrics instances
metrics = MetricsCollector()

# Global transcription metrics store
_transcription_store: Optional[TranscriptionMetricsStore] = None


def get_transcription_store() -> TranscriptionMetricsStore:
    """Get or create global transcription metrics store."""
    global _transcription_store
    if _transcription_store is None:
        _transcription_store = TranscriptionMetricsStore()
    return _transcription_store


def record_transcription(
    job_id: str,
    audio_duration_seconds: float,
    transcription_duration_seconds: float,
    model_used: str,
    success: bool,
    error_message: Optional[str] = None,
    device_name: Optional[str] = None,
    ticket_id: Optional[str] = None,
) -> None:
    """
    Record a transcription completion.

    Convenience function for recording transcription metrics.
    """
    metric = TranscriptionMetric(
        job_id=job_id,
        audio_duration_seconds=audio_duration_seconds,
        transcription_duration_seconds=transcription_duration_seconds,
        model_used=model_used,
        success=success,
        error_message=error_message,
        device_name=device_name,
        ticket_id=ticket_id,
    )

    store = get_transcription_store()
    store.record(metric)
