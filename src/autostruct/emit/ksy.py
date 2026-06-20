"""Serialize a :class:`StructModel` to a Kaitai Struct ``.ksy`` draft.

Every field carries an inline ``# guessed: <rationale>, confidence <x>`` comment;
low-confidence fields are prefixed ``# REVIEW`` to flag them for a human.
"""

from __future__ import annotations

from ..model import (
    KIND_CONSTANT,
    KIND_INT,
    KIND_LENGTH,
    KIND_MAGIC,
    KIND_OPAQUE,
    KIND_REPEAT,
    KIND_STRING,
    KIND_UNKNOWN,
    Field,
    StructModel,
)

REVIEW_THRESHOLD = 0.3


def _majority_endian(model: StructModel) -> str:
    votes = [f.endianness for f in model.fields if f.endianness in ("le", "be")]
    if not votes:
        return model.meta.get("endian", "le")
    return max(("le", "be"), key=votes.count)


def _int_type(field: Field, meta_endian: str) -> str:
    """Explicit-endian integer type so the field never depends on meta/endian."""
    if field.type == "u1" or field.size == 1:
        return "u1"
    suffix = field.endianness or meta_endian
    base = {2: "u2", 4: "u4", 8: "u8"}.get(field.size, "u4")
    return f"{base}{suffix}"


def _comment(field: Field) -> str:
    tag = "# REVIEW " if field.confidence < REVIEW_THRESHOLD else "# guessed: "
    return f"{tag}{field.rationale}, confidence {field.confidence:.2f}"


def _emit_field(field: Field, meta_endian: str, sized_by: dict[int, str]) -> list[str]:
    lines: list[str] = [f"  - id: {field.name}"]
    comment = _comment(field)

    if field.kind in (KIND_MAGIC,) and field.contents is not None:
        contents = ", ".join(f"0x{b:02x}" for b in field.contents)
        lines.append(f"    contents: [{contents}]   {comment}")
        return lines

    if field.kind == KIND_CONSTANT and field.contents is not None:
        contents = ", ".join(f"0x{b:02x}" for b in field.contents)
        lines.append(f"    contents: [{contents}]   {comment}")
        return lines

    if field.kind in (KIND_INT, KIND_LENGTH):
        lines.append(f"    type: {_int_type(field, meta_endian)}   {comment}")
        return lines

    if field.kind == KIND_STRING:
        size_attr = field.extra.get("size_attr")
        encoding = field.extra.get("encoding", "ASCII")
        lines.append(f"    type: {field.type}   {comment}")
        if size_attr is not None:
            lines.append(f"    size: {size_attr}")
        lines.append(f"    encoding: {encoding}")
        return lines

    if field.kind == KIND_REPEAT:
        period = field.extra.get("record_size", field.size)
        lines = [f"  - id: {field.name}", f"    size: {period}   {comment}"]
        count_field = field.extra.get("count_field")
        if count_field:
            lines.append("    repeat: expr")
            lines.append(f"    repeat-expr: {count_field}")
        else:
            lines.append("    repeat: eos")
        return lines

    if field.kind == KIND_OPAQUE:
        if field.extra.get("eos"):
            lines.append(f"    size-eos: true   {comment}")
        else:
            sized = sized_by.get(field.offset)
            if sized:
                lines.append(f"    size: {sized}   {comment}")
            else:
                lines.append(f"    size: {field.size}   {comment}")
        return lines

    # unknown / fallback → raw bytes by size
    sized = sized_by.get(field.offset)
    if sized:
        lines.append(f"    size: {sized}   {comment}")
    else:
        lines.append(f"    size: {field.size}   {comment}")
    return lines


def render_ksy(model: StructModel) -> str:
    """Return the ``.ksy`` YAML draft as a string."""
    meta_endian = _majority_endian(model)
    out: list[str] = []
    out.append("# AutoStruct draft — generated, needs human review.")
    out.append("meta:")
    out.append(f"  id: {model.meta.get('id', 'autostruct_fmt')}")
    out.append(f"  endian: {meta_endian}")
    out.append("seq:")

    # Map an opaque/body field's offset to the name of the length field sizing it.
    sized_by: dict[int, str] = {}
    length_fields = [f for f in model.fields if f.kind == KIND_LENGTH]
    for lf in length_fields:
        if lf.extra.get("sizes_following"):
            # the field immediately following the length is sized by it
            for f in model.fields:
                if f.offset >= lf.end and f.kind in (KIND_OPAQUE, KIND_UNKNOWN):
                    sized_by[f.offset] = lf.name
                    break

    if not model.fields:
        out.append("  []  # no fields inferred (all high-entropy or empty)")

    for field in model.fields:
        out.extend(_emit_field(field, meta_endian, sized_by))

    return "\n".join(out) + "\n"
