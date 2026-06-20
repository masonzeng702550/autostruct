"""Field detectors — pure functions over the byte matrix and position stats.

Each detector returns candidate :class:`~autostruct.model.Field` objects (or
scores) carrying a 0..1 confidence and a human-readable rationale. The pipeline
composes them; detectors never mutate shared state.
"""

from __future__ import annotations


def sample_weight(n: int) -> float:
    """Confidence weight from sample count: ``1 - 1/(n+1)``.

    A single sample caps at ~0.5 (we cannot cross-diff), and the weight rises
    toward 1.0 as more same-format samples corroborate a finding.
    """
    return 1.0 - 1.0 / (n + 1)


def is_printable_ascii(byte: int) -> bool:
    return 0x20 <= byte <= 0x7E


__all__ = ["sample_weight", "is_printable_ascii"]
