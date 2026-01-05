"""
Tests for SHA-256 hashing utilities.

Tests hash computation:
- File hashing with chunked reads
- Bytes hashing
- String hashing with encoding
- Edge cases
"""

import pytest
import hashlib
from pathlib import Path

from callwhisper.utils.hashing import (
    compute_sha256,
    compute_sha256_bytes,
    compute_sha256_string,
)


class TestComputeSha256:
    """Tests for file hashing function."""

    def test_hash_small_file(self, tmp_path):
        """compute_sha256 hashes small file correctly."""
        test_file = tmp_path / "small.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)

        result = compute_sha256(test_file)

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_large_file(self, tmp_path):
        """compute_sha256 handles files larger than read buffer."""
        test_file = tmp_path / "large.bin"
        # Create file larger than 8192 byte buffer
        content = b"x" * 20000
        test_file.write_bytes(content)

        result = compute_sha256(test_file)

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_empty_file(self, tmp_path):
        """compute_sha256 handles empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")

        result = compute_sha256(test_file)

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_with_string_path(self, tmp_path):
        """compute_sha256 accepts string path."""
        test_file = tmp_path / "string_path.txt"
        test_file.write_bytes(b"test content")

        result = compute_sha256(str(test_file))

        assert len(result) == 64  # SHA256 hex length

    def test_hash_with_path_object(self, tmp_path):
        """compute_sha256 accepts Path object."""
        test_file = tmp_path / "path_obj.txt"
        test_file.write_bytes(b"test content")

        result = compute_sha256(test_file)

        assert len(result) == 64

    def test_hash_binary_content(self, tmp_path):
        """compute_sha256 handles binary content."""
        test_file = tmp_path / "binary.bin"
        content = bytes(range(256))  # All byte values
        test_file.write_bytes(content)

        result = compute_sha256(test_file)

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_hash_returns_lowercase_hex(self, tmp_path):
        """compute_sha256 returns lowercase hex string."""
        test_file = tmp_path / "case.txt"
        test_file.write_bytes(b"test")

        result = compute_sha256(test_file)

        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)


class TestComputeSha256Bytes:
    """Tests for bytes hashing function."""

    def test_hash_bytes(self):
        """compute_sha256_bytes hashes bytes correctly."""
        data = b"test data"

        result = compute_sha256_bytes(data)

        expected = hashlib.sha256(data).hexdigest()
        assert result == expected

    def test_hash_empty_bytes(self):
        """compute_sha256_bytes handles empty bytes."""
        result = compute_sha256_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_deterministic(self):
        """Same input produces same hash."""
        data = b"consistent data"

        result1 = compute_sha256_bytes(data)
        result2 = compute_sha256_bytes(data)

        assert result1 == result2

    def test_hash_different_for_different_input(self):
        """Different inputs produce different hashes."""
        result1 = compute_sha256_bytes(b"data1")
        result2 = compute_sha256_bytes(b"data2")

        assert result1 != result2


class TestComputeSha256String:
    """Tests for string hashing function."""

    def test_hash_string(self):
        """compute_sha256_string hashes string correctly."""
        text = "Hello, World!"

        result = compute_sha256_string(text)

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_hash_empty_string(self):
        """compute_sha256_string handles empty string."""
        result = compute_sha256_string("")

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_hash_with_utf8_encoding(self):
        """compute_sha256_string uses UTF-8 by default."""
        text = "Héllo Wörld"

        result = compute_sha256_string(text)

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_hash_with_custom_encoding(self):
        """compute_sha256_string respects custom encoding."""
        text = "Hello"

        result = compute_sha256_string(text, encoding="ascii")

        expected = hashlib.sha256(text.encode("ascii")).hexdigest()
        assert result == expected

    def test_hash_unicode_characters(self):
        """compute_sha256_string handles Unicode."""
        text = "日本語テスト 🎉"

        result = compute_sha256_string(text)

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected

    def test_hash_multiline_string(self):
        """compute_sha256_string handles multiline text."""
        text = "line1\nline2\nline3"

        result = compute_sha256_string(text)

        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result == expected
