"""Integer typing: turn a variable byte span into one or more integer fields.

Greedy width selection — prefer the widest natural width (4, then 2, then 1) that
divides the span and scores well via :mod:`endian_scorer`.
"""

from __future__ import annotations

import numpy as np

from ..model import KIND_INT, Field
from . import sample_weight
from .endian_scorer import best_guess


def type_span(matrix: np.ndarray, start: int, end: int, n: int) -> list[Field]:
    """Carve ``[start, end)`` into integer fields, widest natural width first."""
    fields: list[Field] = []
    pos = start
    base = sample_weight(n)

    while pos < end:
        remaining = end - pos
        width = 4 if remaining >= 4 else 2 if remaining >= 2 else 1
        # Align width to a divisor of the remaining span when possible.
        if width == 4 and remaining % 4 not in (0,) and remaining >= 2:
            width = 2 if remaining % 2 == 0 else 1
        guess = best_guess(matrix, pos, width)
        conf = round(min(0.95, base * (0.5 + 0.5 * guess.score)), 3)

        type_name = {1: "u1", 2: "u2", 4: "u4"}[width]
        name = f"int_{pos:x}"
        rationale = (
            f"integer {type_name}/{guess.endian} "
            f"(plausibility {guess.score:.2f}"
            + (", monotonic across samples → counter/offset?" if guess.monotonic else "")
            + ")"
        )
        fields.append(
            Field(
                offset=pos,
                size=width,
                name=name,
                kind=KIND_INT,
                confidence=conf,
                rationale=rationale,
                type=type_name,
                endianness=guess.endian,
                extra={"values": guess.values, "monotonic": guess.monotonic},
            )
        )
        pos += width
    return fields
