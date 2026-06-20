"""Sample loading, glob expansion, and cross-sample alignment.

The MVP supports two alignment strategies: ``equal`` (require identical lengths)
and ``prefix`` (left-align, truncate to the common shortest length). An
``approximate`` strategy is reserved for later (insertion/deletion-tolerant).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np


class LoaderError(Exception):
    """Raised on missing samples or unreadable files (CLI maps this to exit 2)."""


@dataclass
class LoadedSamples:
    paths: list[str]
    samples: list[bytes]
    matrix: np.ndarray          # (N, L) uint8, aligned
    lengths: list[int]          # original per-sample byte length
    strategy: str

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def aligned_length(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size or self.matrix.ndim == 2 else 0


def expand_paths(patterns: list[str]) -> list[str]:
    """Expand globs, drop directories, de-duplicate while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        matches = glob.glob(pat)
        # A literal path that exists but doesn't match as a glob still counts.
        if not matches and os.path.exists(pat):
            matches = [pat]
        for m in sorted(matches):
            if os.path.isdir(m):
                continue
            ap = os.path.abspath(m)
            if ap not in seen:
                seen.add(ap)
                out.append(m)
    return out


def read_samples(paths: list[str]) -> list[bytes]:
    samples: list[bytes] = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                samples.append(fh.read())
        except OSError as exc:  # pragma: no cover - exercised via CLI
            raise LoaderError(f"cannot read sample {p!r}: {exc}") from exc
    return samples


def align(samples: list[bytes], strategy: str = "prefix") -> np.ndarray:
    """Build the aligned ``(N, L)`` uint8 matrix.

    ``equal``   — all samples must share one length, else :class:`LoaderError`.
    ``prefix``  — left-align and truncate every sample to the shortest length.
    """
    if not samples:
        raise LoaderError("no samples to align")

    lengths = [len(s) for s in samples]

    if strategy == "equal":
        if len(set(lengths)) != 1:
            raise LoaderError(
                "align=equal requires identical sample lengths; got "
                f"{sorted(set(lengths))}"
            )
        length = lengths[0]
    elif strategy == "prefix":
        length = min(lengths)
    else:
        raise LoaderError(f"unknown alignment strategy {strategy!r}")

    matrix = np.zeros((len(samples), length), dtype=np.uint8)
    for i, s in enumerate(samples):
        matrix[i] = np.frombuffer(s[:length], dtype=np.uint8)
    return matrix


def load(patterns: list[str], strategy: str = "prefix") -> LoadedSamples:
    paths = expand_paths(patterns)
    if not paths:
        raise LoaderError(f"no samples matched: {patterns!r}")
    samples = read_samples(paths)
    if all(len(s) == 0 for s in samples):
        raise LoaderError("all samples are empty")
    matrix = align(samples, strategy)
    return LoadedSamples(
        paths=paths,
        samples=samples,
        matrix=matrix,
        lengths=[len(s) for s in samples],
        strategy=strategy,
    )
