"""Single-file HTML structure map: hex view + field table + entropy + heat.

No external assets — CSS and a tiny bit of JS are inlined so the report is one
portable file.
"""

from __future__ import annotations

import html as _html

from ..model import REGION_HIGH_ENTROPY, StructModel

# A small qualitative palette; fields cycle through it, opaque/unknown get greys.
_PALETTE = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _field_color(index: int, kind: str) -> str:
    if kind in ("opaque", "unknown"):
        return "#888888"
    return _PALETTE[index % len(_PALETTE)]


def _entropy_svg(entropy: list[float], regions, width: int = 760, height: int = 80) -> str:
    if not entropy:
        return ""
    n = len(entropy)
    step = width / max(1, n - 1) if n > 1 else width
    pts = " ".join(
        f"{i * step:.1f},{height - e * height:.1f}" for i, e in enumerate(entropy)
    )
    bands = []
    for r in regions:
        if r.kind == REGION_HIGH_ENTROPY:
            x = r.start / n * width
            w = max(1.0, r.size / n * width)
            bands.append(
                f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height}" '
                f'fill="#e15759" opacity="0.15"/>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="entropy" '
        f'preserveAspectRatio="none">'
        + "".join(bands)
        + f'<polyline points="{pts}" fill="none" stroke="#4e79a7" stroke-width="1"/>'
        + f'<line x1="0" y1="{height*0.08:.0f}" x2="{width}" y2="{height*0.08:.0f}" '
        f'stroke="#aaa" stroke-dasharray="3,3"/>'
        + "</svg>"
    )


def _hex_view(model: StructModel, sample: bytes, max_bytes: int = 4096) -> str:
    data = sample[:max_bytes]
    # offset -> (field_index, field) for colouring
    field_index = {id(f): i for i, f in enumerate(model.fields)}
    rows: list[str] = []
    for base in range(0, len(data), 16):
        chunk = data[base:base + 16]
        cells = []
        ascii_cells = []
        for k, byte in enumerate(chunk):
            off = base + k
            f = model.field_at(off)
            if f is not None:
                color = _field_color(field_index[id(f)], f.kind)
                title = _html.escape(
                    f"{f.name} · {f.kind} · {f.type or '-'} · "
                    f"conf {f.confidence:.2f} · {f.rationale}"
                )
                cells.append(
                    f'<span class="b" style="background:{color}33;border-bottom:2px solid {color}" '
                    f'title="{title}">{byte:02x}</span>'
                )
            else:
                cells.append(f'<span class="b">{byte:02x}</span>')
            ch = chr(byte) if 0x20 <= byte <= 0x7E else "."
            ascii_cells.append(_html.escape(ch))
        rows.append(
            f'<div class="row"><span class="off">{base:08x}</span>'
            f'<span class="hex">{"".join(cells)}</span>'
            f'<span class="ascii">{"".join(ascii_cells)}</span></div>'
        )
    truncated = ""
    if len(sample) > max_bytes:
        truncated = f'<div class="note">… truncated, showing first {max_bytes} of {len(sample)} bytes</div>'
    return f'<div class="hexview">{"".join(rows)}{truncated}</div>'


def _field_table(model: StructModel) -> str:
    rows = []
    for i, f in enumerate(model.fields):
        color = _field_color(i, f.kind)
        typ = f.type or "-"
        if f.endianness and f.type:
            typ = f"{f.type} ({f.endianness})"
        pct = int(round(f.confidence * 100))
        review = ' <span class="review">REVIEW</span>' if f.needs_review else ""
        rows.append(
            f"<tr>"
            f'<td><span class="swatch" style="background:{color}"></span>{_html.escape(f.name)}{review}</td>'
            f"<td class=num>{f.offset}</td><td class=num>{f.size}</td>"
            f"<td>{_html.escape(f.kind)}</td><td>{_html.escape(typ)}</td>"
            f'<td><div class="confbar"><div class="conffill" style="width:{pct}%"></div>'
            f"<span>{f.confidence:.2f}</span></div></td>"
            f"<td class=rat>{_html.escape(f.rationale)}</td>"
            f"</tr>"
        )
    return (
        "<table class=fields><thead><tr>"
        "<th>field</th><th>offset</th><th>size</th><th>kind</th>"
        "<th>type</th><th>confidence</th><th>rationale</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _distinct_heat(model: StructModel, width: int = 760, height: int = 24) -> str:
    if not model.distinct or model.n_samples < 2:
        return ""
    n = len(model.distinct)
    maxd = max(model.distinct) or 1
    step = width / n
    rects = []
    for i, d in enumerate(model.distinct):
        # constant (d==1) → dark, highly variable → light
        intensity = 1.0 - (d - 1) / max(1, maxd - 1)
        shade = int(40 + intensity * 180)
        rects.append(
            f'<rect x="{i*step:.2f}" y="0" width="{step+0.5:.2f}" height="{height}" '
            f'fill="rgb({60},{shade},{shade})"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="heat" preserveAspectRatio="none">'
        + "".join(rects) + "</svg>"
    )


def _validation_block(model: StructModel) -> str:
    v = model.validation
    if v is None:
        return '<p class="note">closed-loop validation skipped</p>'
    cls = "ok" if v.pass_rate >= 0.7 else ("warn" if v.pass_rate > 0 else "bad")
    fails = "".join(
        f"<li>{_html.escape(f.sample)} @ {f.offset}: {_html.escape(f.error)}</li>"
        for f in v.failures[:10]
    )
    fails_html = f"<ul class=fails>{fails}</ul>" if fails else ""
    return (
        f'<p class="valid {cls}">closed-loop ({v.method}): '
        f"{v.passed}/{v.total} samples parsed ({v.pass_rate:.0%})</p>{fails_html}"
    )


_CSS = """
:root{color-scheme:light dark}
body{font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:24px;max-width:880px}
h1{font-size:20px;margin:0 0 2px}h2{font-size:14px;margin:24px 0 8px;border-bottom:1px solid #8884;padding-bottom:4px}
.meta{color:#888;margin-bottom:8px}
.hexview{white-space:nowrap;overflow-x:auto;border:1px solid #8883;border-radius:6px;padding:8px;background:#0001}
.row{display:flex;gap:12px}
.off{color:#999}.hex .b{padding:0 1px}.b{display:inline-block;width:18px;text-align:center}
.ascii{color:#777;letter-spacing:1px}
table.fields{border-collapse:collapse;width:100%;font-size:12px}
.fields th,.fields td{border-bottom:1px solid #8883;padding:4px 6px;text-align:left;vertical-align:top}
.fields td.num{text-align:right}.fields td.rat{color:#888;max-width:280px;white-space:normal}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.confbar{position:relative;background:#8882;border-radius:3px;height:14px;width:90px}
.conffill{position:absolute;left:0;top:0;bottom:0;background:#59a14f;border-radius:3px;opacity:.5}
.confbar span{position:relative;padding-left:4px;font-size:11px}
.review{color:#e15759;font-weight:bold;font-size:10px}
.valid{padding:6px 10px;border-radius:6px;display:inline-block}
.valid.ok{background:#59a14f33}.valid.warn{background:#edc94833}.valid.bad{background:#e1575933}
.fails{color:#e15759;font-size:12px}
.entropy,.heat{width:100%;height:auto;display:block;border:1px solid #8883;border-radius:6px;background:#0001}
.note{color:#999;font-size:11px;margin-top:6px}
"""


def render_html(model: StructModel, sample: bytes, title: str | None = None) -> str:
    name = title or model.meta.get("id", "autostruct_fmt")
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>AutoStruct — {_html.escape(name)}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>AutoStruct structure map</h1>",
        f'<div class="meta">id <b>{_html.escape(name)}</b> · '
        f"{model.n_samples} sample(s) · aligned length {model.aligned_length} · "
        f"{len(model.fields)} fields</div>",
        "<h2>Validation</h2>",
        _validation_block(model),
        "<h2>Entropy</h2>",
        _entropy_svg(model.entropy, model.regions),
        "<h2>Cross-sample distinct heat (dark = constant)</h2>",
        _distinct_heat(model) or '<p class="note">needs ≥2 samples</p>',
        "<h2>Hex view</h2>",
        _hex_view(model, sample),
        "<h2>Fields</h2>",
        _field_table(model),
        '<p class="note">Draft — generated by AutoStruct, needs human review.</p>',
    ]
    return "\n".join(parts)
