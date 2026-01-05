"""
Tests for network isolation guard.

Tests offline-mode network blocking:
- Guard enable/disable
- Localhost connection allowance
- External connection blocking
- DNS lookup blocking
- Thread safety
- Status and verification
"""

import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from callwhisper.core.network_guard import (
    ALLOWED_HOSTS,
    ALLOWED_PORTS,
    _guarded_create_connection,
    _guarded_getaddrinfo,
    disable_network_guard,
    enable_network_guard,
    get_guard_status,
    is_guard_enabled,
    verify_offline_mode,
)


# ============================================================================
# Setup/Teardown Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_guard_state():
    """Reset guard state before and after each test."""
    # Ensure guard is disabled before test
    disable_network_guard()
    yield
    # Ensure guard is disabled after test
    disable_network_guard()


# ============================================================================
# ALLOWED_HOSTS and ALLOWED_PORTS Tests
# ============================================================================


class TestAllowedHosts:
    """Tests for allowed hosts configuration."""

    def test_localhost_ip_allowed(self):
        """127.0.0.1 is in allowed hosts."""
        assert '127.0.0.1' in ALLOWED_HOSTS

    def test_localhost_name_allowed(self):
        """'localhost' is in allowed hosts."""
        assert 'localhost' in ALLOWED_HOSTS

    def test_ipv6_localhost_allowed(self):
        """::1 is in allowed hosts."""
        assert '::1' in ALLOWED_HOSTS

    def test_bind_address_allowed(self):
        """0.0.0.0 is in allowed hosts."""
        assert '0.0.0.0' in ALLOWED_HOSTS

    def test_external_hosts_not_allowed(self):
        """External hosts are not in allowed list."""
        assert 'google.com' not in ALLOWED_HOSTS
        assert '8.8.8.8' not in ALLOWED_HOSTS
        assert 'api.openai.com' not in ALLOWED_HOSTS


class TestAllowedPorts:
    """Tests for allowed ports configuration."""

    def test_allowed_port_range(self):
        """Port range is 1024-65535."""
        assert 1024 in ALLOWED_PORTS
        assert 8080 in ALLOWED_PORTS
        assert 65535 in ALLOWED_PORTS

    def test_privileged_ports_not_allowed(self):
        """Privileged ports (< 1024) are not allowed."""
        assert 80 not in ALLOWED_PORTS
        assert 443 not in ALLOWED_PORTS
        assert 22 not in ALLOWED_PORTS


# ============================================================================
# Enable/Disable Guard Tests
# ============================================================================


class TestGuardEnableDisable:
    """Tests for enabling and disabling the guard."""

    def test_guard_initially_disabled(self):
        """Guard is disabled by default."""
        assert is_guard_enabled() is False

    def test_enable_guard(self):
        """enable_network_guard enables the guard."""
        enable_network_guard()

        assert is_guard_enabled() is True

    def test_disable_guard(self):
        """disable_network_guard disables the guard."""
        enable_network_guard()
        disable_network_guard()

        assert is_guard_enabled() is False

    def test_enable_idempotent(self):
        """Calling enable multiple times is safe."""
        enable_network_guard()
        enable_network_guard()
        enable_network_guard()

        assert is_guard_enabled() is True

    def test_disable_idempotent(self):
        """Calling disable multiple times is safe."""
        disable_network_guard()
        disable_network_guard()
        disable_network_guard()

        assert is_guard_enabled() is False


# ============================================================================
# Guarded Create Connection Tests
# ============================================================================


class TestGuardedCreateConnection:
    """Tests for _guarded_create_connection."""

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_localhost_allowed(self, mock_original):
        """Localhost connections are allowed."""
        mock_socket = MagicMock()
        mock_original.return_value = mock_socket

        result = _guarded_create_connection(('127.0.0.1', 8080))

        mock_original.assert_called_once()
        assert result == mock_socket

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_localhost_name_allowed(self, mock_original):
        """'localhost' hostname is allowed."""
        mock_socket = MagicMock()
        mock_original.return_value = mock_socket

        result = _guarded_create_connection(('localhost', 8080))

        mock_original.assert_called_once()
        assert result == mock_socket

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_ipv6_localhost_allowed(self, mock_original):
        """IPv6 localhost is allowed."""
        mock_socket = MagicMock()
        mock_original.return_value = mock_socket

        result = _guarded_create_connection(('::1', 8080))

        mock_original.assert_called_once()

    def test_external_host_blocked(self):
        """External hosts are blocked."""
        with pytest.raises(ConnectionRefusedError) as exc_info:
            _guarded_create_connection(('google.com', 443))

        assert 'External connections blocked' in str(exc_info.value)
        assert 'network guard' in str(exc_info.value)

    def test_external_ip_blocked(self):
        """External IP addresses are blocked."""
        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('8.8.8.8', 53))

    def test_case_insensitive_host_check(self):
        """Host check is case insensitive."""
        with patch('callwhisper.core.network_guard._original_create_connection') as mock:
            mock.return_value = MagicMock()

            # These should all be allowed
            _guarded_create_connection(('LOCALHOST', 8080))
            _guarded_create_connection(('LocalHost', 8080))
            _guarded_create_connection(('localhost', 8080))

            assert mock.call_count == 3

    def test_whitespace_stripped_from_host(self):
        """Whitespace is stripped from host names."""
        with patch('callwhisper.core.network_guard._original_create_connection') as mock:
            mock.return_value = MagicMock()

            _guarded_create_connection((' localhost ', 8080))
            _guarded_create_connection(('\tlocalhost\n', 8080))

            assert mock.call_count == 2

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_passes_timeout(self, mock_original):
        """Timeout parameter is passed through."""
        mock_original.return_value = MagicMock()

        _guarded_create_connection(('localhost', 8080), timeout=5.0)

        mock_original.assert_called_with(
            ('localhost', 8080),
            timeout=5.0,
            source_address=None
        )

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_passes_source_address(self, mock_original):
        """Source address parameter is passed through."""
        mock_original.return_value = MagicMock()

        _guarded_create_connection(
            ('localhost', 8080),
            source_address=('127.0.0.1', 0)
        )

        mock_original.assert_called_with(
            ('localhost', 8080),
            timeout=None,
            source_address=('127.0.0.1', 0)
        )


# ============================================================================
# Guarded GetAddrInfo Tests
# ============================================================================


class TestGuardedGetAddrInfo:
    """Tests for _guarded_getaddrinfo."""

    @patch('callwhisper.core.network_guard._original_getaddrinfo')
    def test_localhost_resolution_allowed(self, mock_original):
        """Localhost DNS resolution is allowed."""
        mock_original.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', 8080))
        ]

        result = _guarded_getaddrinfo('localhost', 8080)

        mock_original.assert_called_once()
        assert len(result) == 1

    @patch('callwhisper.core.network_guard._original_getaddrinfo')
    def test_ip_localhost_resolution_allowed(self, mock_original):
        """IP localhost resolution is allowed."""
        mock_original.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, '', ('127.0.0.1', 8080))
        ]

        result = _guarded_getaddrinfo('127.0.0.1', 8080)

        mock_original.assert_called_once()

    def test_external_dns_blocked(self):
        """External DNS lookups are blocked."""
        with pytest.raises(socket.gaierror) as exc_info:
            _guarded_getaddrinfo('google.com', 443)

        assert 'DNS lookup blocked' in str(exc_info.value)

    def test_external_dns_error_code(self):
        """Blocked DNS lookup returns proper error code."""
        with pytest.raises(socket.gaierror) as exc_info:
            _guarded_getaddrinfo('example.com', 80)

        assert exc_info.value.errno == socket.EAI_NONAME


# ============================================================================
# Guard Status Tests
# ============================================================================


class TestGuardStatus:
    """Tests for guard status functions."""

    def test_get_guard_status_disabled(self):
        """Status reflects disabled state."""
        status = get_guard_status()

        assert status['enabled'] is False
        assert status['external_connections'] == 'allowed'
        assert status['mode'] == 'normal'
        assert 'localhost' in status['allowed_hosts']

    def test_get_guard_status_enabled(self):
        """Status reflects enabled state."""
        enable_network_guard()

        status = get_guard_status()

        assert status['enabled'] is True
        assert status['external_connections'] == 'blocked'
        assert status['mode'] == 'offline'

    def test_verify_offline_mode_disabled(self):
        """Verification fails when guard is disabled."""
        is_offline, message = verify_offline_mode()

        assert is_offline is False
        assert 'not enabled' in message

    def test_verify_offline_mode_enabled(self):
        """Verification succeeds when guard is enabled."""
        enable_network_guard()

        is_offline, message = verify_offline_mode()

        assert is_offline is True
        assert 'verified' in message


# ============================================================================
# Integration Tests
# ============================================================================


class TestGuardIntegration:
    """Integration tests for network guard."""

    def test_socket_function_replaced_when_enabled(self):
        """Socket.create_connection is replaced when guard enabled."""
        original = socket.create_connection

        enable_network_guard()

        assert socket.create_connection != original
        assert socket.create_connection == _guarded_create_connection

    def test_socket_function_restored_when_disabled(self):
        """Socket.create_connection is restored when guard disabled."""
        enable_network_guard()
        disable_network_guard()

        # Should be the original function (stored in module)
        from callwhisper.core.network_guard import _original_create_connection
        assert socket.create_connection == _original_create_connection

    def test_getaddrinfo_replaced_when_enabled(self):
        """Socket.getaddrinfo is replaced when guard enabled."""
        enable_network_guard()

        assert socket.getaddrinfo == _guarded_getaddrinfo

    def test_getaddrinfo_restored_when_disabled(self):
        """Socket.getaddrinfo is restored when guard disabled."""
        enable_network_guard()
        disable_network_guard()

        from callwhisper.core.network_guard import _original_getaddrinfo
        assert socket.getaddrinfo == _original_getaddrinfo


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe guard operations."""

    def test_concurrent_enable_disable(self):
        """Concurrent enable/disable operations are thread-safe."""
        errors = []

        def toggle_guard():
            try:
                for _ in range(50):
                    enable_network_guard()
                    disable_network_guard()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_guard) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_status_check(self):
        """Concurrent status checks are thread-safe."""
        errors = []
        results = []

        def check_status():
            try:
                for _ in range(100):
                    status = is_guard_enabled()
                    results.append(status)
            except Exception as e:
                errors.append(e)

        # Enable guard mid-test
        def toggle_guard():
            import time
            time.sleep(0.001)
            enable_network_guard()

        threads = [
            threading.Thread(target=check_status),
            threading.Thread(target=check_status),
            threading.Thread(target=toggle_guard),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 200


# ============================================================================
# Error Message Tests
# ============================================================================


class TestErrorMessages:
    """Tests for error messages."""

    def test_blocked_connection_error_includes_host(self):
        """Blocked connection error includes the host attempted."""
        try:
            _guarded_create_connection(('api.example.com', 443))
        except ConnectionRefusedError as e:
            assert 'api.example.com' in str(e)
            assert '443' in str(e)

    def test_blocked_connection_error_mentions_offline(self):
        """Error message mentions offline operation."""
        try:
            _guarded_create_connection(('external.server', 80))
        except ConnectionRefusedError as e:
            assert 'offline' in str(e).lower()

    def test_blocked_dns_error_includes_host(self):
        """Blocked DNS error includes the host attempted."""
        try:
            _guarded_getaddrinfo('bad.host.com', 80)
        except socket.gaierror as e:
            assert 'bad.host.com' in str(e)

    def test_blocked_dns_error_mentions_offline(self):
        """DNS error message mentions offline operation."""
        try:
            _guarded_getaddrinfo('external.dns', 80)
        except socket.gaierror as e:
            assert 'offline' in str(e).lower()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_host_blocked(self):
        """Empty host string is blocked."""
        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('', 8080))

    def test_host_with_only_whitespace_blocked(self):
        """Host with only whitespace is blocked."""
        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('   ', 8080))

    @patch('callwhisper.core.network_guard._original_create_connection')
    def test_bind_address_allowed(self, mock_original):
        """0.0.0.0 bind address is allowed."""
        mock_original.return_value = MagicMock()

        _guarded_create_connection(('0.0.0.0', 8080))

        mock_original.assert_called_once()

    def test_numeric_host_blocked(self):
        """Numeric hosts that aren't localhost are blocked."""
        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('192.168.1.1', 80))

    def test_internal_ip_blocked(self):
        """Internal network IPs are blocked (not just external)."""
        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('10.0.0.1', 80))

        with pytest.raises(ConnectionRefusedError):
            _guarded_create_connection(('172.16.0.1', 80))


# ============================================================================
# Allowed Hosts Variations Tests
# ============================================================================


class TestAllowedHostVariations:
    """Tests for various localhost representations."""

    @pytest.fixture
    def mock_original(self):
        """Mock the original create_connection."""
        with patch('callwhisper.core.network_guard._original_create_connection') as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_127_0_0_1_allowed(self, mock_original):
        """127.0.0.1 is allowed."""
        _guarded_create_connection(('127.0.0.1', 8080))
        mock_original.assert_called()

    def test_localhost_allowed(self, mock_original):
        """'localhost' is allowed."""
        _guarded_create_connection(('localhost', 8080))
        mock_original.assert_called()

    def test_ipv6_loopback_allowed(self, mock_original):
        """IPv6 loopback ::1 is allowed."""
        _guarded_create_connection(('::1', 8080))
        mock_original.assert_called()

    def test_all_interfaces_allowed(self, mock_original):
        """0.0.0.0 (all interfaces) is allowed."""
        _guarded_create_connection(('0.0.0.0', 8080))
        mock_original.assert_called()
