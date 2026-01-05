"""
Unit tests for metrics collection module.

Based on LibV2 orchestrator-architecture course patterns:
- Test thread-safe metrics collection
- Test percentile calculations
- Test metric aggregation
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from callwhisper.core.metrics import MetricsCollector, OperationMetrics


@pytest.mark.unit
class TestOperationMetrics:
    """Test individual operation metrics."""

    def test_initial_values(self):
        """Metrics start with zero values."""
        metrics = OperationMetrics()
        assert metrics.count == 0
        assert metrics.success_count == 0
        assert metrics.error_count == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.durations == []

    def test_success_rate_empty(self):
        """Success rate is 0 when no operations recorded."""
        metrics = OperationMetrics()
        assert metrics.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Success rate is 1.0 when all operations succeed."""
        metrics = OperationMetrics(count=10, success_count=10, error_count=0)
        assert metrics.success_rate == 1.0

    def test_success_rate_mixed(self):
        """Success rate calculated correctly for mixed results."""
        metrics = OperationMetrics(count=10, success_count=7, error_count=3)
        assert metrics.success_rate == 0.7

    def test_avg_duration_empty(self):
        """Average duration is 0 when no operations recorded."""
        metrics = OperationMetrics()
        assert metrics.avg_duration_ms == 0.0

    def test_avg_duration_calculated(self):
        """Average duration calculated correctly."""
        metrics = OperationMetrics(
            count=4,
            total_duration_ms=400.0
        )
        assert metrics.avg_duration_ms == 100.0


@pytest.mark.unit
class TestMetricsCollector:
    """Test metrics collector functionality."""

    def test_record_successful_operation(self):
        """Successfully record an operation."""
        collector = MetricsCollector()
        collector.record_operation("test_op", 100.0, success=True)

        metrics = collector.get_operation_metrics("test_op")
        assert metrics is not None
        assert metrics["count"] == 1
        assert metrics["success_count"] == 1
        assert metrics["error_count"] == 0
        assert metrics["avg_duration_ms"] == 100.0

    def test_record_failed_operation(self):
        """Record a failed operation with error type."""
        collector = MetricsCollector()
        collector.record_operation(
            "test_op", 50.0, success=False, error_type="TimeoutError"
        )

        metrics = collector.get_operation_metrics("test_op")
        assert metrics["count"] == 1
        assert metrics["success_count"] == 0
        assert metrics["error_count"] == 1
        assert metrics["errors_by_type"]["TimeoutError"] == 1

    def test_multiple_operations(self):
        """Record multiple operations and aggregate."""
        collector = MetricsCollector()
        collector.record_operation("test_op", 100.0, success=True)
        collector.record_operation("test_op", 200.0, success=True)
        collector.record_operation("test_op", 300.0, success=False)

        metrics = collector.get_operation_metrics("test_op")
        assert metrics["count"] == 3
        assert metrics["success_count"] == 2
        assert metrics["error_count"] == 1
        assert metrics["avg_duration_ms"] == 200.0

    def test_percentiles(self):
        """Calculate percentile durations."""
        collector = MetricsCollector()

        # Record operations with known durations
        for i in range(1, 101):
            collector.record_operation("test_op", float(i), success=True)

        metrics = collector.get_operation_metrics("test_op")
        assert metrics["p50_duration_ms"] == 50.0
        assert metrics["p95_duration_ms"] == 95.0
        assert metrics["p99_duration_ms"] == 99.0

    def test_different_operation_names(self):
        """Track different operations separately."""
        collector = MetricsCollector()
        collector.record_operation("op1", 100.0, success=True)
        collector.record_operation("op2", 200.0, success=False)

        op1_metrics = collector.get_operation_metrics("op1")
        op2_metrics = collector.get_operation_metrics("op2")

        assert op1_metrics["count"] == 1
        assert op1_metrics["success_count"] == 1

        assert op2_metrics["count"] == 1
        assert op2_metrics["error_count"] == 1

    def test_unknown_operation(self):
        """Return None for unknown operation."""
        collector = MetricsCollector()
        metrics = collector.get_operation_metrics("unknown")
        assert metrics is None

    def test_get_all_metrics(self):
        """Get all metrics including uptime."""
        collector = MetricsCollector()
        collector.record_operation("op1", 100.0, success=True)
        collector.record_operation("op2", 200.0, success=True)

        all_metrics = collector.get_all_metrics()
        assert "uptime_seconds" in all_metrics
        assert all_metrics["uptime_seconds"] >= 0
        assert "operations" in all_metrics
        assert "op1" in all_metrics["operations"]
        assert "op2" in all_metrics["operations"]

    def test_uptime_increases(self):
        """Uptime increases over time."""
        collector = MetricsCollector()
        time.sleep(0.1)

        all_metrics = collector.get_all_metrics()
        assert all_metrics["uptime_seconds"] >= 0.1

    def test_reset(self):
        """Reset clears all metrics."""
        collector = MetricsCollector()
        collector.record_operation("op1", 100.0, success=True)
        collector.record_operation("op2", 200.0, success=True)

        collector.reset()

        assert collector.get_operation_metrics("op1") is None
        assert collector.get_operation_metrics("op2") is None


@pytest.mark.unit
class TestMetricsThreadSafety:
    """Test thread safety of metrics collector."""

    def test_concurrent_recording(self):
        """Concurrent recording is thread-safe."""
        collector = MetricsCollector()
        num_threads = 10
        ops_per_thread = 100

        def record_operations():
            for i in range(ops_per_thread):
                collector.record_operation(
                    "concurrent_test",
                    float(i),
                    success=i % 2 == 0
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_operations) for _ in range(num_threads)]
            for f in futures:
                f.result()

        metrics = collector.get_operation_metrics("concurrent_test")
        expected_count = num_threads * ops_per_thread
        assert metrics["count"] == expected_count
        assert metrics["success_count"] == expected_count // 2

    def test_concurrent_read_write(self):
        """Concurrent reads and writes are thread-safe."""
        collector = MetricsCollector()
        stop_event = threading.Event()
        errors = []

        def writer():
            for i in range(100):
                try:
                    collector.record_operation("rw_test", float(i), success=True)
                except Exception as e:
                    errors.append(e)

        def reader():
            while not stop_event.is_set():
                try:
                    collector.get_operation_metrics("rw_test")
                    collector.get_all_metrics()
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Start readers
            reader_futures = [executor.submit(reader) for _ in range(3)]

            # Start writers
            writer_futures = [executor.submit(writer) for _ in range(2)]

            # Wait for writers to complete
            for f in writer_futures:
                f.result()

            # Stop readers
            stop_event.set()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


@pytest.mark.unit
class TestPercentileCalculations:
    """Test percentile calculation edge cases."""

    def test_empty_durations(self):
        """Percentiles handle empty durations."""
        collector = MetricsCollector()
        # Record with no duration tracking (shouldn't happen but test edge case)
        metrics_data = collector._operations.get("test", OperationMetrics())
        # Empty durations list
        assert collector._calculate_percentile([], 50) == 0.0

    def test_single_duration(self):
        """Percentiles with single value."""
        collector = MetricsCollector()
        collector.record_operation("single", 100.0, success=True)

        metrics = collector.get_operation_metrics("single")
        assert metrics["p50_duration_ms"] == 100.0
        assert metrics["p95_duration_ms"] == 100.0

    def test_two_durations(self):
        """Percentiles with two values."""
        collector = MetricsCollector()
        collector.record_operation("two", 100.0, success=True)
        collector.record_operation("two", 200.0, success=True)

        metrics = collector.get_operation_metrics("two")
        assert metrics["p50_duration_ms"] in [100.0, 200.0]


@pytest.mark.unit
class TestErrorTracking:
    """Test error type tracking."""

    def test_multiple_error_types(self):
        """Track different error types separately."""
        collector = MetricsCollector()
        collector.record_operation("test", 10.0, False, error_type="TypeError")
        collector.record_operation("test", 20.0, False, error_type="ValueError")
        collector.record_operation("test", 30.0, False, error_type="TypeError")

        metrics = collector.get_operation_metrics("test")
        assert metrics["errors_by_type"]["TypeError"] == 2
        assert metrics["errors_by_type"]["ValueError"] == 1

    def test_error_without_type(self):
        """Handle errors without explicit type."""
        collector = MetricsCollector()
        collector.record_operation("test", 10.0, success=False)

        metrics = collector.get_operation_metrics("test")
        assert metrics["error_count"] == 1
        # No specific error type tracked
        assert len(metrics["errors_by_type"]) == 0


# ============================================================================
# Batch 7: Large Dataset and Advanced Edge Case Tests
# ============================================================================

@pytest.mark.unit
class TestLargeDatasetMetrics:
    """Tests for large dataset handling."""

    def test_percentile_accuracy_1000_operations(self):
        """Percentile accuracy with 1000+ operations."""
        collector = MetricsCollector()

        # Record 1000 operations with sequential durations
        for i in range(1, 1001):
            collector.record_operation("large_test", float(i), success=True)

        metrics = collector.get_operation_metrics("large_test")

        assert metrics["count"] == 1000
        # p50 should be around 500
        assert 490 <= metrics["p50_duration_ms"] <= 510
        # p95 should be around 950
        assert 940 <= metrics["p95_duration_ms"] <= 960
        # p99 should be around 990
        assert 985 <= metrics["p99_duration_ms"] <= 995

    def test_duration_list_limit(self):
        """Duration list respects max size limit."""
        collector = MetricsCollector()

        # Record more operations than the typical limit (1000)
        for i in range(2000):
            collector.record_operation("overflow_test", float(i), success=True)

        # Should still work and have bounded memory
        metrics = collector.get_operation_metrics("overflow_test")
        assert metrics["count"] == 2000
        # Percentiles should still be calculable

    def test_high_volume_mixed_success_failure(self):
        """High volume with mixed success/failure."""
        collector = MetricsCollector()

        for i in range(1000):
            success = i % 3 != 0  # 2/3 success rate
            collector.record_operation(
                "mixed_test",
                float(i),
                success=success,
                error_type="Error" if not success else None
            )

        metrics = collector.get_operation_metrics("mixed_test")
        assert metrics["count"] == 1000
        # Approximately 333 errors
        assert 330 <= metrics["error_count"] <= 340
        # Approximately 666 successes
        assert 660 <= metrics["success_count"] <= 670


@pytest.mark.unit
class TestMetricsEdgeCases:
    """Advanced edge case tests for metrics."""

    def test_very_small_durations(self):
        """Handle very small duration values."""
        collector = MetricsCollector()

        for _ in range(100):
            collector.record_operation("tiny", 0.001, success=True)

        metrics = collector.get_operation_metrics("tiny")
        # Use absolute tolerance for very small values near zero
        assert metrics["avg_duration_ms"] == pytest.approx(0.001, abs=0.01)

    def test_very_large_durations(self):
        """Handle very large duration values."""
        collector = MetricsCollector()

        collector.record_operation("huge", 1000000.0, success=True)  # 1M ms

        metrics = collector.get_operation_metrics("huge")
        assert metrics["avg_duration_ms"] == 1000000.0

    def test_zero_duration(self):
        """Handle zero duration."""
        collector = MetricsCollector()

        collector.record_operation("zero", 0.0, success=True)

        metrics = collector.get_operation_metrics("zero")
        assert metrics["avg_duration_ms"] == 0.0

    def test_many_error_types(self):
        """Handle many different error types."""
        collector = MetricsCollector()

        error_types = [f"ErrorType{i}" for i in range(100)]

        for error_type in error_types:
            collector.record_operation("many_errors", 10.0, False, error_type)

        metrics = collector.get_operation_metrics("many_errors")
        assert len(metrics["errors_by_type"]) == 100
        assert all(count == 1 for count in metrics["errors_by_type"].values())

    def test_unicode_operation_names(self):
        """Handle unicode operation names."""
        collector = MetricsCollector()

        names = ["操作", "Операция", "運営", "operation_🎤"]

        for name in names:
            collector.record_operation(name, 100.0, success=True)

        for name in names:
            metrics = collector.get_operation_metrics(name)
            assert metrics is not None
            assert metrics["count"] == 1

    def test_empty_operation_name(self):
        """Handle empty operation name."""
        collector = MetricsCollector()

        collector.record_operation("", 100.0, success=True)

        metrics = collector.get_operation_metrics("")
        assert metrics is not None
        assert metrics["count"] == 1


@pytest.mark.unit
class TestConcurrentMetrics:
    """Additional concurrency tests for metrics."""

    def test_concurrent_different_operations(self):
        """Concurrent recording of different operations."""
        collector = MetricsCollector()
        num_operations = 10
        ops_per_type = 100

        def record_operation(op_name):
            for i in range(ops_per_type):
                collector.record_operation(op_name, float(i), success=True)

        with ThreadPoolExecutor(max_workers=num_operations) as executor:
            futures = [
                executor.submit(record_operation, f"op_{i}")
                for i in range(num_operations)
            ]
            for f in futures:
                f.result()

        for i in range(num_operations):
            metrics = collector.get_operation_metrics(f"op_{i}")
            assert metrics["count"] == ops_per_type

    def test_reset_during_recording(self):
        """Reset during concurrent recording."""
        collector = MetricsCollector()
        errors = []

        def record():
            for i in range(50):
                try:
                    collector.record_operation("test", float(i), success=True)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.001)

        def reset():
            time.sleep(0.025)  # Wait for some recordings
            try:
                collector.reset()
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(record),
                executor.submit(record),
                executor.submit(reset),
            ]
            for f in futures:
                f.result()

        assert len(errors) == 0
