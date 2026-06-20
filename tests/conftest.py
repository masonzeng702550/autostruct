"""Shared test fixtures: synthetic same-format binary samples."""

from __future__ import annotations

import struct

import numpy as np
import pytest


def _body(length: int, seed: int) -> bytes:
    return bytes((i + seed * 7) % 256 for i in range(length))


def make_format_samples(n: int = 5, magic: bytes = b"\x7fELF") -> list[bytes]:
    """magic(4) + version u1 (const) + length u4le + body(var, sized by length)."""
    samples: list[bytes] = []
    for k in range(n):
        body_len = 20 + 8 * k
        body = _body(body_len, k + 1)
        sample = magic + bytes([0x02]) + struct.pack("<I", body_len) + body
        samples.append(sample)
    return samples


@pytest.fixture
def format_samples() -> list[bytes]:
    return make_format_samples()


@pytest.fixture
def png_samples() -> list[bytes]:
    magic = b"\x89PNG\r\n\x1a\n"
    out = []
    for k in range(4):
        # IHDR-like chunk with constant length 0x0d and varying content
        chunk = struct.pack(">I", 13) + b"IHDR" + bytes((i + k) % 256 for i in range(13))
        out.append(magic + chunk)
    return out


@pytest.fixture
def repeat_samples() -> list[bytes]:
    """A header followed by repeated 16-byte records."""
    out = []
    for k in range(3):
        header = b"REC!" + bytes([0x01, 0x00, 0x00, 0x00])
        records = b"".join(
            bytes([r, k]) + bytes(range(14)) for r in range(6)
        )
        out.append(header + records)
    return out


def to_matrix(samples: list[bytes]) -> np.ndarray:
    L = min(len(s) for s in samples)
    return np.array([np.frombuffer(s[:L], np.uint8) for s in samples], dtype=np.uint8)
