"""Integer width + endianness scoring.

For a candidate integer span we decode N values for each (width, endian) and
score by "numeric plausibility": small concentrated values are preferred, and the
endianness whose high bytes are mostly zero wins (small integers are common in
real formats). Monotonic-across-samples values get a counter/offset bonus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntGuess:
    width: int          # 1 / 2 / 4
    endian: str         # "le" / "be"
    score: float        # 0..1
    values: list[int]
    monotonic: bool


def _decode(col_bytes: np.ndarray, width: int, endian: str) -> list[int]:
    """Decode each sample's ``width`` bytes (already sliced) into an int list."""
    order = "little" if endian == "le" else "big"
    return [int.from_bytes(bytes(row.tolist()), order) for row in col_bytes]


def score_span(matrix: np.ndarray, start: int, width: int) -> list[IntGuess]:
    """Score both endiannesses for a fixed-width integer at ``start``."""
    n = matrix.shape[0]
    span = matrix[:, start:start + width]
    maxval = float(256 ** width - 1)
    guesses: list[IntGuess] = []

    for endian in ("le", "be"):
        values = _decode(span, width, endian)
        arr = np.array(values, dtype=np.float64)

        # 1) Smallness: prefer values well below the type's max.
        smallness = 1.0 - float(np.mean(arr)) / maxval if maxval else 0.0

        # 2) High-byte-zero: fraction of samples whose top byte is 0.
        if width > 1:
            top_shift = 8 * (width - 1)
            top_zero = float(np.mean([(v >> top_shift) == 0 for v in values]))
        else:
            top_zero = 0.5  # neutral for u1

        # 3) Concentration: low spread (relative std) reads as a real field.
        mean = float(np.mean(arr)) or 1.0
        rel_std = float(np.std(arr)) / max(mean, 1.0)
        concentration = 1.0 / (1.0 + rel_std)

        monotonic = bool(np.all(np.diff(arr) >= 0) and np.any(np.diff(arr) > 0)) if n >= 2 else False
        mono_bonus = 0.1 if monotonic else 0.0

        score = 0.45 * smallness + 0.35 * top_zero + 0.20 * concentration + mono_bonus
        guesses.append(
            IntGuess(width=width, endian=endian, score=min(1.0, score),
                     values=values, monotonic=monotonic)
        )
    return guesses


def best_guess(matrix: np.ndarray, start: int, width: int) -> IntGuess:
    return max(score_span(matrix, start, width), key=lambda g: g.score)
