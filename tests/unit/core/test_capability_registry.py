"""
Tests for capability registry module.

Tests capability registration and selection:
- Handler registration and unregistration
- Priority-based selection
- Availability checking
- Fallback chains
"""

import threading
from unittest.mock import MagicMock

import pytest

from callwhisper.core.capability_registry import (
    Capability,
    CapabilityRegistry,
    CapabilityType,
    get_registry,
    register_capability,
)


# ============================================================================
# CapabilityType Tests
# ============================================================================


class TestCapabilityType:
    """Tests for CapabilityType enum."""

    def test_capability_types_defined(self):
        """Standard capability types are defined."""
        assert CapabilityType.TRANSCRIPTION == "transcription"
        assert CapabilityType.AUDIO_NORMALIZATION == "audio_normalization"
        assert CapabilityType.AUDIO_CONVERSION == "audio_conversion"
        assert CapabilityType.BUNDLING == "bundling"


# ============================================================================
# Capability Tests
# ============================================================================


class TestCapability:
    """Tests for Capability dataclass."""

    def test_capability_creation(self):
        """Capability can be created."""
        handler = MagicMock()
        cap = Capability(
            name="test-handler",
            handler=handler,
            priority=10,
        )

        assert cap.name == "test-handler"
        assert cap.handler == handler
        assert cap.priority == 10
        assert cap.is_available()  # Default is available

    def test_capability_with_availability_check(self):
        """Capability can have custom availability check."""
        available = [True]

        cap = Capability(
            name="conditional",
            handler=MagicMock(),
            is_available=lambda: available[0],
        )

        assert cap.is_available() is True
        available[0] = False
        assert cap.is_available() is False

    def test_capability_with_metadata(self):
        """Capability can include metadata."""
        cap = Capability(
            name="whisper",
            handler=MagicMock(),
            metadata={"model_size": "large", "language": "en"},
        )

        assert cap.metadata["model_size"] == "large"
        assert cap.metadata["language"] == "en"


# ============================================================================
# CapabilityRegistry Basic Tests
# ============================================================================


class TestCapabilityRegistryBasic:
    """Tests for basic registry operations."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        return CapabilityRegistry()

    def test_register_capability(self, registry):
        """Capability can be registered."""
        handler = MagicMock()
        cap = Capability(name="test", handler=handler)

        registry.register("transcription", cap)

        assert registry.get_handler("transcription") == handler

    def test_unregister_capability(self, registry):
        """Capability can be unregistered."""
        cap = Capability(name="removable", handler=MagicMock())
        registry.register("transcription", cap)

        result = registry.unregister("transcription", "removable")

        assert result is True
        assert registry.get_handler("transcription") is None

    def test_unregister_nonexistent(self, registry):
        """Unregistering nonexistent returns False."""
        result = registry.unregister("transcription", "nonexistent")
        assert result is False


# ============================================================================
# Priority and Fallback Tests
# ============================================================================


class TestPriorityAndFallback:
    """Tests for priority-based selection and fallback."""

    @pytest.fixture
    def registry(self):
        """Create registry with multiple handlers."""
        reg = CapabilityRegistry()

        # Register handlers with different priorities
        reg.register("transcription", Capability(
            name="low-priority",
            handler=lambda: "low",
            priority=1,
        ))
        reg.register("transcription", Capability(
            name="high-priority",
            handler=lambda: "high",
            priority=10,
        ))
        reg.register("transcription", Capability(
            name="medium-priority",
            handler=lambda: "medium",
            priority=5,
        ))

        return reg

    def test_highest_priority_selected(self, registry):
        """Highest priority handler is selected."""
        handler = registry.get_handler("transcription")
        assert handler() == "high"

    def test_fallback_when_unavailable(self):
        """Falls back to lower priority when higher is unavailable."""
        registry = CapabilityRegistry()

        registry.register("transcription", Capability(
            name="primary",
            handler=lambda: "primary",
            priority=10,
            is_available=lambda: False,  # Not available
        ))
        registry.register("transcription", Capability(
            name="fallback",
            handler=lambda: "fallback",
            priority=5,
            is_available=lambda: True,
        ))

        handler = registry.get_handler("transcription", require_available=True)
        assert handler() == "fallback"

    def test_get_all_handlers_in_order(self, registry):
        """get_all_handlers returns in priority order."""
        handlers = registry.get_all_handlers("transcription")

        assert len(handlers) == 3
        # Execute handlers to verify order
        results = [h() for h in handlers]
        assert results == ["high", "medium", "low"]


# ============================================================================
# Availability Tests
# ============================================================================


class TestAvailability:
    """Tests for availability checking."""

    @pytest.fixture
    def registry(self):
        """Create registry for testing."""
        return CapabilityRegistry()

    def test_is_capability_available(self, registry):
        """is_capability_available checks handler availability."""
        registry.register("transcription", Capability(
            name="available",
            handler=MagicMock(),
            is_available=lambda: True,
        ))

        assert registry.is_capability_available("transcription") is True

    def test_is_capability_unavailable(self, registry):
        """Returns False when no handlers available."""
        registry.register("transcription", Capability(
            name="unavailable",
            handler=MagicMock(),
            is_available=lambda: False,
        ))

        assert registry.is_capability_available("transcription") is False

    def test_get_handler_without_availability_check(self, registry):
        """Can get unavailable handler when check disabled."""
        handler = MagicMock()
        registry.register("transcription", Capability(
            name="test",
            handler=handler,
            is_available=lambda: False,
        ))

        result = registry.get_handler("transcription", require_available=False)

        assert result == handler

    def test_get_only_available_handlers(self, registry):
        """get_all_handlers can filter by availability."""
        registry.register("transcription", Capability(
            name="available",
            handler=lambda: "available",
            is_available=lambda: True,
        ))
        registry.register("transcription", Capability(
            name="unavailable",
            handler=lambda: "unavailable",
            is_available=lambda: False,
        ))

        handlers = registry.get_all_handlers("transcription", only_available=True)

        assert len(handlers) == 1
        assert handlers[0]() == "available"


# ============================================================================
# Registry Info Tests
# ============================================================================


class TestRegistryInfo:
    """Tests for registry information methods."""

    @pytest.fixture
    def registry(self):
        """Create populated registry."""
        reg = CapabilityRegistry()
        reg.register("transcription", Capability(
            name="whisper",
            handler=MagicMock(),
            priority=10,
            metadata={"model": "large"},
        ))
        reg.register("normalization", Capability(
            name="ffmpeg",
            handler=MagicMock(),
        ))
        return reg

    def test_get_capability_info(self, registry):
        """get_capability_info returns details."""
        info = registry.get_capability_info("transcription")

        assert len(info) == 1
        assert info[0]["name"] == "whisper"
        assert info[0]["priority"] == 10
        assert info[0]["available"] is True
        assert info[0]["metadata"]["model"] == "large"

    def test_get_all_types(self, registry):
        """get_all_types returns registered types."""
        types = registry.get_all_types()

        assert "transcription" in types
        assert "normalization" in types

    def test_get_registry_stats(self, registry):
        """get_registry_stats returns statistics."""
        stats = registry.get_registry_stats()

        assert stats["total_types"] == 2
        assert stats["total_capabilities"] == 2
        assert "transcription" in stats["by_type"]


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_registration(self):
        """Concurrent registrations are thread-safe."""
        registry = CapabilityRegistry()
        errors = []

        def register_handlers(prefix):
            try:
                for i in range(20):
                    registry.register(
                        "transcription",
                        Capability(
                            name=f"{prefix}-{i}",
                            handler=MagicMock(),
                            priority=i,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_handlers, args=(f"thread-{t}",))
            for t in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All registrations should have succeeded
        info = registry.get_capability_info("transcription")
        assert len(info) == 80  # 4 threads * 20 handlers


# ============================================================================
# Global Registry Tests
# ============================================================================


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_registry_creates_instance(self):
        """get_registry creates instance on first call."""
        import callwhisper.core.capability_registry as module
        original = module._registry
        module._registry = None

        try:
            registry = get_registry()
            assert isinstance(registry, CapabilityRegistry)
        finally:
            module._registry = original

    def test_get_registry_returns_same_instance(self):
        """get_registry returns same instance."""
        import callwhisper.core.capability_registry as module
        original = module._registry
        module._registry = None

        try:
            registry1 = get_registry()
            registry2 = get_registry()
            assert registry1 is registry2
        finally:
            module._registry = original


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_register_capability_function(self):
        """register_capability convenience function works."""
        import callwhisper.core.capability_registry as module
        original = module._registry
        module._registry = CapabilityRegistry()

        try:
            handler = MagicMock()
            register_capability(
                capability_type="transcription",
                name="test-handler",
                handler=handler,
                priority=5,
                model_size="large",  # metadata
            )

            reg = get_registry()
            cap = reg.get_capability("transcription")

            assert cap.name == "test-handler"
            assert cap.priority == 5
            assert cap.metadata["model_size"] == "large"
        finally:
            module._registry = original
