"""Sliding-window Shannon entropy and region segmentation.

High-entropy spans are treated as compressed / encrypted opaque blobs and are
excluded from field-level inference. Everything else is ``structured`` and flows
into the detectors.
"""

from __future__ import annotations

import numpy as np

from .model import REGION_HIGH_ENTROPY, REGION_STRUCTURED, Region

DEFAULT_WINDOW = 256
DEFAULT_STEP = 64
DEFAULT_THRESHOLD = 0.92


def _window_entropy(window: np.ndarray) -> float:
    """Shannon entropy of one byte window, normalised to 0..1 (divide by 8)."""
    if window.size == 0:
        return 0.0
    counts = np.bincount(window, minlength=256).astype(np.float64)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    h = -np.sum(nz * np.log2(nz))
    return float(h / 8.0)


def entropy_profile(
    data: np.ndarray,
    window: int = DEFAULT_WINDOW,
    step: int = DEFAULT_STEP,
) -> np.ndarray:
    """Per-byte normalised entropy profile.

    Each window's entropy is assigned to every byte it covers; overlapping windows
    are averaged so the profile is exactly ``len(data)`` long and smooth.
    """
    data = np.asarray(data, dtype=np.uint8)
    n = data.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    window = max(1, min(window, n))
    step = max(1, step)

    acc = np.zeros(n, dtype=np.float64)
    cov = np.zeros(n, dtype=np.float64)
    start = 0
    while start < n:
        end = min(start + window, n)
        h = _window_entropy(data[start:end])
        acc[start:end] += h
        cov[start:end] += 1.0
        if end == n:
            break
        start += step
    cov[cov == 0] = 1.0
    return acc / cov


def segment(
    profile: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    min_region: int = 8,
) -> list[Region]:
    """Split a profile into structured / high_entropy regions.

    Contiguous runs above ``threshold`` become ``high_entropy``; tiny regions are
    merged into their neighbour so we don't emit a flurry of 1-byte blobs.
    """
    n = profile.size
    if n == 0:
        return []
    high = profile >= threshold

    regions: list[Region] = []
    start = 0
    cur = bool(high[0])
    for i in range(1, n):
        if bool(high[i]) != cur:
            kind = REGION_HIGH_ENTROPY if cur else REGION_STRUCTURED
            regions.append(Region(start, i, kind))
            start = i
            cur = bool(high[i])
    regions.append(Region(start, n, REGION_HIGH_ENTROPY if cur else REGION_STRUCTURED))

    return _merge_small(regions, min_region)


def _merge_small(regions: list[Region], min_region: int) -> list[Region]:
    if len(regions) <= 1:
        return regions
    merged: list[Region] = [regions[0]]
    for r in regions[1:]:
        prev = merged[-1]
        if r.size < min_region:
            # absorb tiny region into the previous one, keeping prev's kind
            prev.end = r.end
        elif prev.kind == r.kind:
            prev.end = r.end
        else:
            merged.append(r)
    return merged
