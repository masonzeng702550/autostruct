"""Repeated fixed-length record detection via autocorrelation.

We look for a stable period ``p`` in a representative sample's structured region.
If a span is an integer multiple of ``p`` with a strong autocorrelation peak, we
infer a fixed-length record array of size ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RepeatHit:
    start: int
    end: int
    period: int
    count: int
    score: float  # 0..1 normalised autocorrelation peak


def _autocorr(x: np.ndarray) -> np.ndarray:
    """Unbiased autocorrelation of a zero-mean signal, lags 0..len-1."""
    x = x.astype(np.float64)
    x = x - x.mean()
    n = x.size
    if n == 0 or np.allclose(x, 0):
        return np.zeros(n)
    full = np.correlate(x, x, mode="full")[n - 1:]
    norm = full[0] if full[0] != 0 else 1.0
    return full / norm


def find_period(
    data: np.ndarray,
    start: int,
    end: int,
    min_period: int = 4,
    max_period: int = 256,
    min_repeats: int = 3,
    min_score: float = 0.35,
) -> Optional[RepeatHit]:
    """Find the dominant record period within ``[start, end)`` of ``data``."""
    span = np.asarray(data[start:end], dtype=np.uint8)
    length = span.size
    if length < min_period * min_repeats:
        return None

    ac = _autocorr(span)
    hi = min(max_period, length // min_repeats)
    if hi <= min_period:
        return None

    lags = np.arange(min_period, hi + 1)
    scores = ac[min_period:hi + 1]
    if scores.size == 0:
        return None

    best_idx = int(np.argmax(scores))
    period = int(lags[best_idx])
    score = float(scores[best_idx])
    if score < min_score:
        return None

    count = length // period
    if count < min_repeats:
        return None

    return RepeatHit(
        start=start,
        end=start + count * period,
        period=period,
        count=count,
        score=round(score, 3),
    )


def find_period_scan(
    data: np.ndarray,
    start: int,
    end: int,
    max_phase: int = 64,
    **kw,
) -> Optional[RepeatHit]:
    """Find the best record period allowing the array to start *after* ``start``.

    Real record arrays usually follow a header, so autocorrelation over the whole
    region is diluted. We probe a handful of phase offsets and keep the strongest,
    widest hit.
    """
    best: Optional[RepeatHit] = None
    limit = min(end, start + max_phase + 1)
    span = end - start
    for s0 in range(start, limit):
        if end - s0 < 12:
            break
        hit = find_period(data, s0, end, **kw)
        if hit is None:
            continue
        if best is None or (hit.score, hit.end - hit.start) > (best.score, best.end - best.start):
            best = hit
    return best
