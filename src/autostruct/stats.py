"""Per-position cross-sample statistics over the aligned byte matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PositionStats:
    distinct: np.ndarray     # (L,) int   distinct byte values across samples
    constant: np.ndarray     # (L,) bool  distinct == 1
    low_variance: np.ndarray # (L,) bool  distinct small (enum / flag candidate)
    monotonic: np.ndarray    # (L,) bool  non-decreasing across samples (sorted-ish)
    n: int

    @property
    def length(self) -> int:
        return int(self.distinct.shape[0])


def compute(matrix: np.ndarray) -> PositionStats:
    """Compute distinct/constant/low-variance/monotonic masks per column."""
    matrix = np.asarray(matrix, dtype=np.uint8)
    n, length = matrix.shape

    if length == 0:
        empty_b = np.zeros(0, dtype=bool)
        return PositionStats(
            distinct=np.zeros(0, dtype=int),
            constant=empty_b,
            low_variance=empty_b,
            monotonic=empty_b,
            n=n,
        )

    # distinct values per column
    distinct = np.array(
        [np.unique(matrix[:, j]).size for j in range(length)], dtype=int
    )
    constant = distinct == 1
    # "low variance" ⇒ few distinct values relative to sample count (enum/flag).
    thresh = max(1, math.ceil(n * 0.25))
    low_variance = (distinct > 1) & (distinct <= thresh)

    # monotonic across samples: column is non-decreasing when samples are
    # considered in load order. Useful as a weak counter/offset hint with N≥2.
    if n >= 2:
        diffs = np.diff(matrix.astype(np.int16), axis=0)  # (N-1, L)
        monotonic = np.all(diffs >= 0, axis=0) & np.any(diffs > 0, axis=0)
    else:
        monotonic = np.zeros(length, dtype=bool)

    return PositionStats(
        distinct=distinct,
        constant=constant,
        low_variance=low_variance,
        monotonic=monotonic,
        n=n,
    )
