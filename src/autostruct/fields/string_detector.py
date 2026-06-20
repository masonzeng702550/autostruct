"""String-run detection: ASCII / UTF-16LE, NUL-terminated vs fixed length."""

from __future__ import annotations

import numpy as np

from ..model import KIND_STRING, Field
from . import is_printable_ascii, sample_weight

MIN_STRING = 4


def _ascii_fractions(matrix: np.ndarray, start: int, end: int) -> tuple[float, float]:
    """Return (printable-or-NUL fraction, printable fraction) for the span.

    Trailing NULs are valid ``strz`` padding, so they count as acceptable but not
    as printable characters — we require a real printable core as well.
    """
    span = matrix[:, start:end]
    if span.size == 0:
        return 0.0, 0.0
    printable = np.vectorize(is_printable_ascii)(span)
    ok = printable | (span == 0)
    return float(ok.mean()), float(printable.mean())


def _utf16_fraction(matrix: np.ndarray, start: int, end: int) -> float:
    """Heuristic for UTF-16LE: even bytes printable, odd bytes mostly NUL."""
    span = matrix[:, start:end]
    if span.shape[1] < 2 * MIN_STRING:
        return 0.0
    even = span[:, 0::2]
    odd = span[:, 1::2]
    even_ok = np.vectorize(is_printable_ascii)(even).mean()
    odd_nul = (odd == 0).mean()
    return float(min(even_ok, odd_nul))


def detect(matrix: np.ndarray, start: int, end: int, n: int) -> Field | None:
    """Classify a variable span as a string field, or return ``None``."""
    size = end - start
    if size < MIN_STRING:
        return None

    base = sample_weight(n)
    ok_frac, printable_frac = _ascii_fractions(matrix, start, end)
    utf16_frac = _utf16_fraction(matrix, start, end)

    if ok_frac >= 0.9 and printable_frac >= 0.4:
        ascii_frac = printable_frac
        # NUL terminator anywhere in the representative sample → strz.
        rep = matrix[0, start:end]
        has_nul = bool((rep == 0).any())
        encoding = "ASCII"
        if has_nul:
            type_name, size_attr = "strz", None
        else:
            type_name, size_attr = "str", size
        conf = round(min(0.9, base * (0.6 + 0.4 * ascii_frac)), 3)
        rationale = f"{encoding} string run ({ascii_frac:.0%} printable)"
    elif utf16_frac >= 0.85:
        type_name, size_attr = "str", size
        encoding = "UTF-16LE"
        conf = round(min(0.85, base * (0.55 + 0.4 * utf16_frac)), 3)
        rationale = f"{encoding} string run ({utf16_frac:.0%} match)"
    else:
        return None

    return Field(
        offset=start,
        size=size,
        name=f"str_{start:x}",
        kind=KIND_STRING,
        confidence=conf,
        rationale=rationale,
        type=type_name,
        extra={"encoding": encoding, "size_attr": size_attr},
    )
