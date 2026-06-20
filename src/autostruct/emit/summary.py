"""Human-readable terminal summary of a StructModel."""

from __future__ import annotations

from ..model import StructModel


def _bar(conf: float, width: int = 10) -> str:
    filled = int(round(conf * width))
    return "█" * filled + "·" * (width - filled)


def render_summary(model: StructModel, color: bool = False) -> str:
    lines: list[str] = []
    meta_id = model.meta.get("id", "autostruct_fmt")
    lines.append(f"AutoStruct — {meta_id}")
    lines.append(
        f"  samples: {model.n_samples}   aligned length: {model.aligned_length}   "
        f"fields: {len(model.fields)}"
    )

    if model.regions:
        hi = sum(r.size for r in model.regions if r.kind == "high_entropy")
        lines.append(
            f"  regions: {len(model.regions)} "
            f"({hi} bytes high-entropy / opaque)"
        )

    lines.append("")
    header = f"  {'offset':>8}  {'size':>5}  {'kind':<9} {'type':<7} {'conf':<13} name"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for f in model.fields:
        typ = f.type or "-"
        # append endianness only for multi-byte ints whose type lacks a suffix
        if f.endianness and f.type and f.size > 1 and not f.type[-2:] in ("le", "be"):
            typ = f"{f.type}{f.endianness}"
        flag = " ⚠" if f.needs_review else ""
        lines.append(
            f"  {f.offset:>8}  {f.size:>5}  {f.kind:<9} {typ:<7} "
            f"{_bar(f.confidence)} {f.confidence:.2f} {f.name}{flag}"
        )

    if model.validation is not None:
        v = model.validation
        lines.append("")
        lines.append(
            f"  closed-loop ({v.method}): {v.passed}/{v.total} parsed "
            f"({v.pass_rate:.0%})"
        )
        for fail in v.failures[:5]:
            lines.append(f"    ✗ {fail.sample} @ {fail.offset}: {fail.error}")

    return "\n".join(lines)
