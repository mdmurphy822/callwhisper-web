"""SHA-256 hashing utilities."""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex-encoded SHA-256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.

    Args:
        data: Bytes to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(data).hexdigest()


def compute_sha256_string(text: str, encoding: str = "utf-8") -> str:
    """
    Compute SHA-256 hash of a string.

    Args:
        text: String to hash
        encoding: Text encoding

    Returns:
        Hex-encoded SHA-256 hash
    """
    return compute_sha256_bytes(text.encode(encoding))
