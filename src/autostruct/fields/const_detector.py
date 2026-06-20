"""Constant / magic field detection.

Consecutive positions that are constant across all samples merge into one field.
A constant run at offset 0 of length 2..8 is promoted to ``magic``; printable
ASCII runs get a confidence bump and a note in the rationale.
"""

from __future__ import annotations

import numpy as np

from ..model import KIND_CONSTANT, KIND_MAGIC, Field
from . import is_printable_ascii, sample_weight


def detect(
    matrix: np.ndarray,
    start: int,
    end: int,
    n: int,
) -> Field:
    """Build a constant/magic field spanning ``[start, end)``.

    The caller guarantees every column in the range is constant across samples.
    """
    values = [int(matrix[0, j]) for j in range(start, end)]
    size = end - start
    base = sample_weight(n)

    printable = all(is_printable_ascii(b) for b in values)
    ascii_note = ""
    if printable:
        text = bytes(values).decode("ascii", "replace")
        ascii_note = f', printable ASCII "{text}"'

    is_magic = start == 0 and 2 <= size <= 8
    if is_magic:
        conf = min(0.99, base + (0.15 if printable else 0.05))
        kind = KIND_MAGIC
        name = "magic"
        rationale = f"constant {size}-byte run at offset 0{ascii_note}"
    else:
        conf = base * (1.05 if printable else 1.0)
        conf = min(0.97, conf)
        kind = KIND_CONSTANT
        name = f"const_{start:x}"
        rationale = f"constant across all {n} samples{ascii_note}"

    # Pick a natural integer type for short non-magic constants (version/flag).
    field_type = None
    if kind == KIND_CONSTANT and size in (1, 2, 4):
        field_type = {1: "u1", 2: "u2", 4: "u4"}[size]

    return Field(
        offset=start,
        size=size,
        name=name,
        kind=kind,
        confidence=round(conf, 3),
        rationale=rationale,
        type=field_type,
        contents=values,
    )
