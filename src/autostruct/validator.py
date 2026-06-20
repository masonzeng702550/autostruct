"""Closed-loop validation: re-parse the original samples against the model.

The default validator is dependency-free — it walks the inferred field layout
against each raw sample and checks that constants match and every field fits. If
``kaitai-struct-compiler`` is on ``PATH`` we additionally run it as a syntax
check on the emitted ``.ksy`` and surface compile errors.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .model import (
    KIND_CONSTANT,
    KIND_LENGTH,
    KIND_MAGIC,
    Failure,
    StructModel,
    Validation,
)


def _parse_one(model: StructModel, sample: bytes, name: str) -> Failure | None:
    """Re-parse a single sample; return the first :class:`Failure` or ``None``."""
    n = len(sample)
    for f in model.fields:
        # high-entropy / eos opaque fields legitimately run to end-of-stream
        is_eos = f.extra.get("eos") or f.kind == "opaque"
        end = f.end
        if end > n and not is_eos:
            return Failure(name, f.offset, f"field {f.name!r} ({end}B) exceeds sample ({n}B)")
        if f.offset > n:
            return Failure(name, f.offset, f"field {f.name!r} starts past end of sample")

        if f.kind in (KIND_MAGIC, KIND_CONSTANT) and f.contents is not None:
            actual = list(sample[f.offset:min(end, n)])
            if actual != f.contents:
                exp = bytes(f.contents).hex()
                got = bytes(actual).hex()
                return Failure(name, f.offset, f"constant {f.name!r} mismatch: exp {exp} got {got}")

        if f.kind == KIND_LENGTH and "values" in f.extra:
            # Sanity: the decoded length should not point outside the sample.
            order = "little" if (f.endianness or "le") == "le" else "big"
            raw = sample[f.offset:end]
            if len(raw) == f.size:
                val = int.from_bytes(raw, order)
                off = f.extra.get("length_offset", 0) or 0
                implied = val + off
                if implied < 0 or implied > n:
                    return Failure(
                        name, f.offset,
                        f"length {f.name!r} implies {implied}B but sample is {n}B",
                    )
    return None


def validate_internal(model: StructModel, samples: list[bytes], paths: list[str]) -> Validation:
    failures: list[Failure] = []
    for sample, path in zip(samples, paths):
        fail = _parse_one(model, sample, path)
        if fail is not None:
            failures.append(fail)
    passed = len(samples) - len(failures)
    return Validation(passed=passed, total=len(samples), failures=failures, method="internal")


def kaitai_syntax_check(ksy_text: str) -> str | None:
    """If ``kaitai-struct-compiler`` is available, return its error output (or None)."""
    ksc = shutil.which("kaitai-struct-compiler") or shutil.which("ksc")
    if not ksc:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        ksy_path = Path(tmp) / "fmt.ksy"
        ksy_path.write_text(ksy_text, encoding="utf-8")
        try:
            proc = subprocess.run(
                [ksc, "--target", "python", "--outdir", tmp, str(ksy_path)],
                capture_output=True, text=True, timeout=60,
            )
        except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
            return f"compiler invocation failed: {exc}"
        if proc.returncode != 0:
            return (proc.stderr or proc.stdout).strip()
    return None


def validate(model: StructModel, samples: list[bytes], paths: list[str]) -> Validation:
    """Run the closed loop and attach the result. Internal re-parse is always run."""
    result = validate_internal(model, samples, paths)
    return result
