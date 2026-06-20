"""End-to-end inference pipeline: bytes → StructModel.

    load → align → entropy segmentation → per-position stats → field detection
    → length/repeat post-processing → model assembly → closed-loop validation
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from . import entropy as entropy_mod
from . import loader, stats, validator
from .fields import const_detector, int_typer, repeat_detector, string_detector
from .fields.length_detector import annotate as annotate_lengths
from .model import (
    KIND_INT,
    KIND_OPAQUE,
    KIND_REPEAT,
    KIND_STRING,
    KIND_UNKNOWN,
    REGION_HIGH_ENTROPY,
    REGION_STRUCTURED,
    Field,
    Region,
    StructModel,
)


@dataclass
class InferConfig:
    window: int = entropy_mod.DEFAULT_WINDOW
    step: int = entropy_mod.DEFAULT_STEP
    entropy_threshold: float = entropy_mod.DEFAULT_THRESHOLD
    align: str = "prefix"
    min_confidence: float = 0.3
    name: str = "autostruct_fmt"
    do_validate: bool = True
    verbose: bool = False

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[autostruct] {msg}", file=sys.stderr)


def infer_paths(patterns: list[str], config: Optional[InferConfig] = None):
    """Load samples from globs and run inference. Returns (model, LoadedSamples)."""
    config = config or InferConfig()
    loaded = loader.load(patterns, config.align)
    model = infer_samples(loaded.samples, loaded.paths, config, matrix=loaded.matrix,
                          lengths=loaded.lengths)
    return model, loaded


def infer_samples(
    samples: list[bytes],
    paths: Optional[list[str]] = None,
    config: Optional[InferConfig] = None,
    matrix: Optional[np.ndarray] = None,
    lengths: Optional[list[int]] = None,
) -> StructModel:
    """Run the full inference pipeline over already-read samples."""
    config = config or InferConfig()
    paths = paths or [f"sample{i}" for i in range(len(samples))]
    lengths = lengths or [len(s) for s in samples]
    if matrix is None:
        matrix = loader.align(samples, config.align)

    n, L = matrix.shape
    config.log(f"{n} samples, aligned length {L} (strategy={config.align})")

    # Representative sample for entropy + repeat detection (first, full length).
    rep = np.frombuffer(samples[0], dtype=np.uint8) if samples[0] else np.zeros(0, np.uint8)
    profile = entropy_mod.entropy_profile(rep, config.window, config.step)
    regions = entropy_mod.segment(profile, config.entropy_threshold)
    config.log(f"{len(regions)} regions: " +
               ", ".join(f"{r.kind}[{r.start}:{r.end}]" for r in regions))

    st = stats.compute(matrix)
    # Single sample: every column is trivially "constant"; that is not evidence,
    # so suppress the constant mask and degrade to entropy + string inference.
    constant_mask = st.constant.copy()
    if n < 2:
        constant_mask[:] = False
        config.log("single sample → constant detection disabled (low-confidence mode)")

    fields = _detect_fields(matrix, rep, regions, constant_mask, L, n, config)
    fields = _fill_gaps(fields, L)
    fields = _append_trailer(fields, samples, L, profile, config)

    annotate_lengths(fields, lengths)
    _apply_min_confidence(fields, config.min_confidence)
    fields.sort(key=lambda f: f.offset)

    model = StructModel(
        meta={"id": config.name, "endian": _majority_endian(fields)},
        fields=fields,
        regions=regions,
        entropy=[round(float(x), 4) for x in profile.tolist()],
        distinct=st.distinct.tolist(),
        n_samples=n,
        aligned_length=L,
    )

    if config.do_validate:
        model.validation = validator.validate(model, samples, paths)
        config.log(f"closed-loop: {model.validation.passed}/{model.validation.total}")

    return model


# --------------------------------------------------------------------------- #
# Field detection
# --------------------------------------------------------------------------- #

def _detect_fields(matrix, rep, regions, constant_mask, L, n, config) -> list[Field]:
    fields: list[Field] = []
    for region in regions:
        if region.start >= L:
            break
        r_end = min(region.end, L)
        if region.kind == REGION_HIGH_ENTROPY:
            fields.append(Field(
                offset=region.start, size=r_end - region.start,
                name=f"opaque_{region.start:x}", kind=KIND_OPAQUE,
                confidence=0.4, rationale="high-entropy opaque blob (compressed/encrypted?)",
                type=None, extra={"eos": False},
            ))
            continue
        fields.extend(_detect_structured(matrix, rep, region.start, r_end,
                                          constant_mask, n, config))
    return fields


def _detect_structured(matrix, rep, start, end, constant_mask, n, config) -> list[Field]:
    """Detect fields in a structured region, trying repeated records first."""
    if end - start >= 48:
        hit = repeat_detector.find_period_scan(rep, start, end)
        if hit and hit.score >= 0.5 and (hit.end - hit.start) >= 0.6 * (end - start):
            config.log(f"repeat: period {hit.period} ×{hit.count} @ [{hit.start}:{hit.end}] "
                       f"score {hit.score}")
            out: list[Field] = []
            out += _segment_const_var(matrix, start, hit.start, constant_mask, n)
            out.append(Field(
                offset=hit.start, size=hit.end - hit.start,
                name=f"records_{hit.start:x}", kind=KIND_REPEAT,
                confidence=round(min(0.85, 0.5 + 0.4 * hit.score), 3),
                rationale=f"{hit.count} repeated {hit.period}-byte records "
                          f"(autocorrelation {hit.score:.2f})",
                type=None,
                extra={"record_size": hit.period, "count": hit.count},
            ))
            out += _segment_const_var(matrix, hit.end, end, constant_mask, n)
            return out
    return _segment_const_var(matrix, start, end, constant_mask, n)


def _segment_const_var(matrix, start, end, constant_mask, n) -> list[Field]:
    """Walk a span, grouping constant runs and interpreting variable runs."""
    fields: list[Field] = []
    pos = start
    while pos < end:
        if constant_mask[pos]:
            run_end = pos
            while run_end < end and constant_mask[run_end]:
                run_end += 1
            # A long constant run at offset 0 is magic (first ≤8 bytes) plus a
            # trailing constant block (e.g. a constant length / version word).
            if pos == 0 and run_end - pos > 8:
                fields.append(const_detector.detect(matrix, 0, 8, n))
                fields.append(const_detector.detect(matrix, 8, run_end, n))
            else:
                fields.append(const_detector.detect(matrix, pos, run_end, n))
            pos = run_end
        else:
            run_end = pos
            while run_end < end and not constant_mask[run_end]:
                run_end += 1
            fields.extend(_interpret_variable(matrix, pos, run_end, n))
            pos = run_end
    return fields


def _interpret_variable(matrix, start, end, n) -> list[Field]:
    """A variable run is either a string, or carved into integers."""
    string_field = string_detector.detect(matrix, start, end, n)
    if string_field is not None:
        return [string_field]
    return int_typer.type_span(matrix, start, end, n)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #

def _fill_gaps(fields: list[Field], L: int) -> list[Field]:
    """Ensure fields tile [0, L) contiguously; fill holes with unknown bytes."""
    fields = sorted(fields, key=lambda f: f.offset)
    out: list[Field] = []
    cursor = 0
    for f in fields:
        if f.offset > cursor:
            out.append(_unknown(cursor, f.offset - cursor))
        out.append(f)
        cursor = max(cursor, f.end)
    if cursor < L:
        out.append(_unknown(cursor, L - cursor))
    return out


def _unknown(offset: int, size: int) -> Field:
    return Field(
        offset=offset, size=size, name=f"unknown_{offset:x}", kind=KIND_UNKNOWN,
        confidence=0.1, rationale="unclassified bytes", type=None,
    )


def _append_trailer(fields, samples, L, profile, config) -> list[Field]:
    """If some samples are longer than the aligned length, add an eos trailer."""
    if not any(len(s) > L for s in samples):
        return fields
    tail_high = bool(profile[L:].mean() >= config.entropy_threshold) if profile.size > L else False
    kind = KIND_OPAQUE
    rationale = ("high-entropy trailing blob beyond aligned length"
                 if tail_high else "trailing bytes beyond aligned length (variable size)")
    fields.append(Field(
        offset=L, size=0, name="trailer", kind=kind,
        confidence=0.35 if tail_high else 0.2,
        rationale=rationale, type=None, extra={"eos": True},
    ))
    return fields


def _apply_min_confidence(fields: list[Field], min_conf: float) -> None:
    """Demote low-confidence non-anchor fields to unknown bytes."""
    for f in fields:
        if f.kind in (KIND_OPAQUE, KIND_REPEAT):
            continue
        if f.confidence < min_conf and f.kind != KIND_UNKNOWN:
            f.kind = KIND_UNKNOWN
            f.type = None
            f.endianness = None
            f.contents = None
            f.rationale = f"below min-confidence ({f.confidence:.2f}) → unknown bytes"
            f.name = f"unknown_{f.offset:x}"


def _majority_endian(fields: list[Field]) -> str:
    votes = [f.endianness for f in fields if f.endianness in ("le", "be")]
    if not votes:
        return "le"
    return max(("le", "be"), key=votes.count)
