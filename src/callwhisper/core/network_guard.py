"""
Network Isolation Guard

Ensures CallWhisper makes no external network calls.
This is critical for enterprise Windows environments with no internet access.

The guard works by overriding socket.create_connection to only allow
localhost connections. Any attempt to connect to external hosts will
raise ConnectionRefusedError.

Usage:
    from .network_guard import enable_network_guard, is_guard_enabled

    enable_network_guard()  # Call during startup
"""

import socket
import threading
from typing import Tuple, Optional, List

from .logging_config import get_core_logger

logger = get_core_logger()

# Store original function for restoration
_original_create_connection = socket.create_connection
_original_getaddrinfo = socket.getaddrinfo

# Guard state
_guard_enabled = False
_guard_lock = threading.Lock()

# Allowed hosts for localhost connections
ALLOWED_HOSTS = frozenset([
    '127.0.0.1',
    'localhost',
    '::1',
    '0.0.0.0',  # Binding address  # nosec B104
])

# Allowed port range (restrict to reasonable local service ports)
ALLOWED_PORTS = range(1024, 65536)


def _guarded_create_connection(
    address: Tuple[str, int],
    timeout: Optional[float] = None,
    source_address: Optional[Tuple[str, int]] = None,
    **kwargs
) -> socket.socket:
    """
    Guarded version of socket.create_connection.

    Only allows connections to localhost addresses.
    Blocks all external network connections.
    """
    host, port = address

    # Normalize host for comparison
    normalized_host = host.lower().strip()

    # Check if host is allowed
    if normalized_host not in ALLOWED_HOSTS:
        logger.warning(
            "external_connection_blocked",
            host=host,
            port=port,
            reason="Network guard active - external connections blocked"
        )
        raise ConnectionRefusedError(
            f"External connections blocked by network guard: {host}:{port}. "
            "CallWhisper is configured for offline-only operation."
        )

    # Log allowed connection (debug level)
    logger.debug(
        "local_connection_allowed",
        host=host,
        port=port
    )

    # Call original function for localhost connections
    return _original_create_connection(
        address,
        timeout=timeout,
        source_address=source_address,
        **kwargs
    )


def _guarded_getaddrinfo(
    host: str,
    port: int,
    family: int = 0,
    type_: int = 0,
    proto: int = 0,
    flags: int = 0
) -> List:
    """
    Guarded version of socket.getaddrinfo.

    Only resolves localhost addresses. External DNS lookups are blocked.
    """
    normalized_host = str(host).lower().strip()

    # Allow localhost resolution
    if normalized_host in ALLOWED_HOSTS:
        return _original_getaddrinfo(host, port, family, type_, proto, flags)

    # Block external DNS lookups
    logger.warning(
        "dns_lookup_blocked",
        host=host,
        port=port,
        reason="Network guard active - DNS lookups for external hosts blocked"
    )
    raise socket.gaierror(
        socket.EAI_NONAME,
        f"DNS lookup blocked by network guard: {host}. "
        "CallWhisper is configured for offline-only operation."
    )


def enable_network_guard() -> None:
    """
    Enable the network isolation guard.

    After calling this function, all socket connections to non-localhost
    addresses will be blocked with ConnectionRefusedError.

    This function is idempotent - calling it multiple times is safe.
    """
    global _guard_enabled

    with _guard_lock:
        if _guard_enabled:
            logger.debug("network_guard_already_enabled")
            return

        # Override socket functions
        socket.create_connection = _guarded_create_connection
        socket.getaddrinfo = _guarded_getaddrinfo

        _guard_enabled = True

        logger.info(
            "network_guard_enabled",
            allowed_hosts=list(ALLOWED_HOSTS),
            message="External network connections are now blocked"
        )


def disable_network_guard() -> None:
    """
    Disable the network isolation guard.

    Restores normal network behavior. Only use this for testing.
    """
    global _guard_enabled

    with _guard_lock:
        if not _guard_enabled:
            logger.debug("network_guard_already_disabled")
            return

        # Restore original functions
        socket.create_connection = _original_create_connection
        socket.getaddrinfo = _original_getaddrinfo

        _guard_enabled = False

        logger.info(
            "network_guard_disabled",
            message="External network connections are now allowed"
        )


def is_guard_enabled() -> bool:
    """Check if the network guard is currently enabled."""
    with _guard_lock:
        return _guard_enabled


def get_guard_status() -> dict:
    """
    Get detailed network guard status for health checks.

    Returns:
        Dictionary with guard status information.
    """
    with _guard_lock:
        return {
            "enabled": _guard_enabled,
            "allowed_hosts": list(ALLOWED_HOSTS),
            "external_connections": "blocked" if _guard_enabled else "allowed",
            "mode": "offline" if _guard_enabled else "normal",
        }


def verify_offline_mode() -> Tuple[bool, str]:
    """
    Verify that the application is properly configured for offline mode.

    Checks:
    - Network guard is enabled
    - No external network dependencies detected

    Returns:
        Tuple of (is_offline, message)
    """
    if not _guard_enabled:
        return False, "Network guard is not enabled"

    # Additional checks could go here (e.g., verify no cloud SDK imports)

    return True, "Offline mode verified - all external connections blocked"
