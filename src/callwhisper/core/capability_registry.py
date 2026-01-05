"""
Capability Registry Module

Based on LibV2 orchestrator-architecture patterns:
- Specialist selection with capability matching
- Process vs domain knowledge separation
- Fallback handler chains
- Dynamic capability registration
"""

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List, Any, TypeVar
from enum import Enum

from .logging_config import get_core_logger

logger = get_core_logger()

T = TypeVar("T")


class CapabilityType(str, Enum):
    """Types of capabilities in the system."""

    TRANSCRIPTION = "transcription"
    AUDIO_NORMALIZATION = "audio_normalization"
    AUDIO_CONVERSION = "audio_conversion"
    BUNDLING = "bundling"


@dataclass
class Capability:
    """
    Represents a capability with its handler.

    Attributes:
        name: Human-readable name for the capability
        handler: Callable that implements the capability
        priority: Higher priority = preferred (default 0)
        is_available: Function to check if handler is currently available
        metadata: Additional info about the capability
    """

    name: str
    handler: Callable
    priority: int = 0
    is_available: Callable[[], bool] = field(default_factory=lambda: lambda: True)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure is_available is callable
        if not callable(self.is_available):
            self.is_available = lambda: True


@dataclass
class CapabilityMatch:
    """Result of a capability lookup."""

    capability: Capability
    score: float  # 0.0 to 1.0, higher is better match


class CapabilityRegistry:
    """
    Registry for subprocess capabilities.

    Implements the orchestrator pattern of separating:
    - Process knowledge (orchestrator): when/what to execute
    - Domain knowledge (specialists): how to execute

    Features:
    - Register capabilities with priority
    - Fallback chain via priority ordering
    - Availability checking
    - Dynamic registration/deregistration
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._capabilities: Dict[str, List[Capability]] = {}
        self._fallback_enabled = True

    def register(self, capability_type: str, capability: Capability) -> None:
        """
        Register a capability handler.

        Args:
            capability_type: Type of capability (e.g., "transcription")
            capability: The capability to register
        """
        with self._lock:
            if capability_type not in self._capabilities:
                self._capabilities[capability_type] = []

            self._capabilities[capability_type].append(capability)

            # Sort by priority (highest first)
            self._capabilities[capability_type].sort(key=lambda c: -c.priority)

            logger.info(
                "capability_registered",
                capability_type=capability_type,
                name=capability.name,
                priority=capability.priority,
            )

    def unregister(self, capability_type: str, name: str) -> bool:
        """
        Unregister a capability by name.

        Returns:
            True if capability was found and removed.
        """
        with self._lock:
            if capability_type not in self._capabilities:
                return False

            original_len = len(self._capabilities[capability_type])
            self._capabilities[capability_type] = [
                c for c in self._capabilities[capability_type] if c.name != name
            ]

            removed = len(self._capabilities[capability_type]) < original_len

            if removed:
                logger.info(
                    "capability_unregistered",
                    capability_type=capability_type,
                    name=name,
                )

            return removed

    def get_handler(
        self, capability_type: str, require_available: bool = True
    ) -> Optional[Callable]:
        """
        Get the best available handler for a capability type.

        This implements the fallback pattern: returns highest priority
        handler that is currently available.

        Args:
            capability_type: Type of capability needed
            require_available: If True, only return available handlers

        Returns:
            Handler callable or None if no suitable handler found.
        """
        with self._lock:
            capabilities = self._capabilities.get(capability_type, [])

            for cap in capabilities:
                if not require_available or cap.is_available():
                    logger.debug(
                        "capability_selected",
                        capability_type=capability_type,
                        name=cap.name,
                        priority=cap.priority,
                    )
                    return cap.handler

            logger.warning(
                "no_capability_found",
                capability_type=capability_type,
                registered_count=len(capabilities),
                require_available=require_available,
            )
            return None

    def get_capability(
        self, capability_type: str, require_available: bool = True
    ) -> Optional[Capability]:
        """
        Get the best available capability (not just handler).

        Useful when you need access to capability metadata.
        """
        with self._lock:
            capabilities = self._capabilities.get(capability_type, [])

            for cap in capabilities:
                if not require_available or cap.is_available():
                    return cap

            return None

    def get_all_handlers(
        self, capability_type: str, only_available: bool = False
    ) -> List[Callable]:
        """
        Get all handlers for a capability type (for fallback chains).

        Returns handlers in priority order (highest first).
        """
        with self._lock:
            capabilities = self._capabilities.get(capability_type, [])

            if only_available:
                return [c.handler for c in capabilities if c.is_available()]
            return [c.handler for c in capabilities]

    def get_capability_info(self, capability_type: str) -> List[Dict[str, Any]]:
        """Get info about all registered capabilities of a type."""
        with self._lock:
            capabilities = self._capabilities.get(capability_type, [])

            return [
                {
                    "name": c.name,
                    "priority": c.priority,
                    "available": c.is_available(),
                    "metadata": c.metadata,
                }
                for c in capabilities
            ]

    def get_all_types(self) -> List[str]:
        """Get all registered capability types."""
        with self._lock:
            return list(self._capabilities.keys())

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        with self._lock:
            stats = {
                "total_types": len(self._capabilities),
                "total_capabilities": sum(
                    len(caps) for caps in self._capabilities.values()
                ),
                "by_type": {},
            }

            for cap_type, caps in self._capabilities.items():
                available = sum(1 for c in caps if c.is_available())
                stats["by_type"][cap_type] = {
                    "total": len(caps),
                    "available": available,
                }

            return stats

    def is_capability_available(self, capability_type: str) -> bool:
        """Check if any handler is available for a capability type."""
        return self.get_handler(capability_type, require_available=True) is not None


# Global registry instance (lazy initialization)
_registry: Optional[CapabilityRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> CapabilityRegistry:
    """Get or create the global capability registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = CapabilityRegistry()
    return _registry


def register_capability(
    capability_type: str,
    name: str,
    handler: Callable,
    priority: int = 0,
    is_available: Optional[Callable[[], bool]] = None,
    **metadata,
) -> None:
    """Convenience function to register a capability."""
    cap = Capability(
        name=name,
        handler=handler,
        priority=priority,
        is_available=is_available or (lambda: True),
        metadata=metadata,
    )
    get_registry().register(capability_type, cap)
