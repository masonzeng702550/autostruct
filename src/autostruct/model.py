"""Serialization-agnostic intermediate model produced by the inference core.

The whole pipeline funnels its result into a :class:`StructModel`. Emitters
(``.ksy``, HTML, terminal) and the validator consume *only* this model, never the
internal detector state, so adding an output format never touches inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Field kinds, ordered roughly from most to least confident interpretation.
KIND_MAGIC = "magic"
KIND_CONSTANT = "constant"
KIND_LENGTH = "length"
KIND_INT = "int"
KIND_STRING = "string"
KIND_REPEAT = "repeat"
KIND_OPAQUE = "opaque"
KIND_UNKNOWN = "unknown"

# Region kinds from entropy segmentation.
REGION_STRUCTURED = "structured"
REGION_HIGH_ENTROPY = "high_entropy"


@dataclass
class Field:
    """One inferred field in the structure draft.

    ``contents`` holds the concrete byte values for magic/constant fields (so the
    emitter can write ``contents: [...]``). ``extra`` carries detector-specific
    metadata, e.g. ``{"size_field": "length"}`` for a sized body or
    ``{"repeat": "eos", "record_size": 64}`` for a record array.
    """

    offset: int
    size: int
    name: str
    kind: str
    confidence: float
    rationale: str
    type: Optional[str] = None          # u1 / u2 / u4 / str / strz / bytes / None
    endianness: Optional[str] = None    # "le" / "be" / None
    contents: Optional[list[int]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.3 or self.kind == KIND_UNKNOWN


@dataclass
class Region:
    start: int
    end: int
    kind: str  # REGION_STRUCTURED / REGION_HIGH_ENTROPY

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class Failure:
    sample: str
    offset: int
    error: str


@dataclass
class Validation:
    passed: int
    total: int
    failures: list[Failure] = field(default_factory=list)
    method: str = "internal"  # "internal" re-parse or "kaitai" compiler

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class StructModel:
    meta: dict[str, Any]               # {"id": str, "endian": "le"/"be"}
    fields: list[Field] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    validation: Optional[Validation] = None
    # Cross-sample distinct-value count per aligned position (HTML heat panel).
    distinct: list[int] = field(default_factory=list)
    n_samples: int = 0
    aligned_length: int = 0

    def field_at(self, offset: int) -> Optional[Field]:
        for f in self.fields:
            if f.offset <= offset < f.end:
                return f
        return None
