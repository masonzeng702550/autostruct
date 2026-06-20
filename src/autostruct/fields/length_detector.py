"""Length-prefix / size-field detection.

An integer field whose decoded value tracks the remaining length (or total file
length) across all samples is a strong length/size candidate. We test exact
equality and constant-offset linear relations.
"""

from __future__ import annotations

from typing import Optional

from ..model import KIND_INT, KIND_LENGTH, Field


def _linear_match(values: list[int], targets: list[int]) -> Optional[int]:
    """Return constant ``c`` if ``value + c == target`` for every sample, else None."""
    if len(values) != len(targets) or not values:
        return None
    deltas = {t - v for v, t in zip(values, targets)}
    if len(deltas) == 1:
        return deltas.pop()
    return None


def annotate(fields: list[Field], lengths: list[int]) -> list[Field]:
    """Mutate integer fields in place, promoting length/size candidates.

    ``lengths`` is the original per-sample byte length (pre-alignment).
    """
    n = len(lengths)
    if n < 2:
        return fields  # cannot correlate with a single sample

    for f in fields:
        if f.kind != KIND_INT or "values" not in f.extra:
            continue
        values = f.extra["values"]
        if len(set(values)) <= 1:
            continue  # constant value can't be a varying length

        remaining = [L - f.end for L in lengths]

        c_total = _linear_match(values, lengths)
        c_remain = _linear_match(values, remaining)

        if c_remain is not None:
            f.kind = KIND_LENGTH
            f.name = f"len_{f.offset:x}"
            rel = "remaining bytes" + (f" + {c_remain}" if c_remain else "")
            f.rationale = f"value tracks {rel} across all samples → length prefix"
            f.confidence = round(min(0.95, f.confidence + 0.2), 3)
            f.extra["sizes_following"] = True
            f.extra["length_offset"] = c_remain
        elif c_total is not None:
            f.kind = KIND_LENGTH
            f.name = f"len_{f.offset:x}"
            rel = "total file length" + (f" + {c_total}" if c_total else "")
            f.rationale = f"value tracks {rel} across all samples → size field"
            f.confidence = round(min(0.92, f.confidence + 0.18), 3)
            f.extra["length_offset"] = c_total
    return fields
