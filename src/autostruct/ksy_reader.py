"""Minimal ``.ksy`` reader for the closed-loop ``validate`` command.

This is intentionally small: it understands the subset of Kaitai needed to
re-check a flat ``seq`` of fixed-layout fields — ``contents`` (magic/constant),
fixed integer ``type`` (u1/u2/u4 with optional le/be), fixed ``size``, and
``size-eos``. It is not a full Kaitai parser; nested types, ``instances``,
expressions, and ``switch-on`` are out of scope. If PyYAML is available it is
used; otherwise a line-based fallback handles AutoStruct's own output.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from .model import (
    KIND_CONSTANT,
    KIND_INT,
    KIND_OPAQUE,
    KIND_UNKNOWN,
    Field,
)

_INT_SIZES = {"u1": 1, "u2": 2, "u4": 4, "u8": 8, "s1": 1, "s2": 2, "s4": 4, "s8": 8}


def _type_to_size_endian(t: str) -> tuple[int | None, str | None]:
    m = re.fullmatch(r"([su][1248])(le|be)?", t)
    if not m:
        return None, None
    base, endian = m.group(1), m.group(2)
    return _INT_SIZES[base], endian


def _field_from_entry(entry: dict[str, Any], offset: int) -> Field:
    name = str(entry.get("id", f"field_{offset:x}"))
    if "contents" in entry:
        contents = entry["contents"]
        if isinstance(contents, str):
            contents = list(contents.encode("latin-1"))
        contents = [int(b) for b in contents]
        return Field(offset=offset, size=len(contents), name=name,
                     kind=KIND_CONSTANT, confidence=1.0, rationale="ksy contents",
                     contents=contents)
    if "type" in entry and entry.get("size") is None:
        raw_type = str(entry["type"])
        size, endian = _type_to_size_endian(raw_type)
        if size is not None:
            base = raw_type[:-2] if raw_type[-2:] in ("le", "be") else raw_type
            return Field(offset=offset, size=size, name=name, kind=KIND_INT,
                         confidence=1.0, rationale="ksy integer", type=base,
                         endianness=endian)
    if entry.get("size-eos"):
        return Field(offset=offset, size=0, name=name, kind=KIND_OPAQUE,
                     confidence=1.0, rationale="ksy size-eos", extra={"eos": True})
    if "size" in entry:
        try:
            size = int(entry["size"])
        except (TypeError, ValueError):
            size = 0  # expression-sized; treated as variable / skip strict check
        return Field(offset=offset, size=size, name=name, kind=KIND_OPAQUE,
                     confidence=1.0, rationale="ksy sized", extra={"eos": size == 0})
    return Field(offset=offset, size=0, name=name, kind=KIND_UNKNOWN,
                 confidence=0.0, rationale="ksy unrecognised entry")


def _parse_with_yaml(text: str):
    import yaml  # type: ignore

    doc = yaml.safe_load(text)
    meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
    seq = doc.get("seq", []) if isinstance(doc, dict) else []
    return meta, seq


def _parse_fallback(text: str):
    """Line parser for AutoStruct's own emitted .ksy (strips inline comments)."""
    meta: dict[str, Any] = {}
    seq: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("meta:"):
            section = "meta"
            continue
        if line.startswith("seq:"):
            section = "seq"
            continue

        if section == "meta" and re.match(r"\s+\w", line):
            k, _, v = line.strip().partition(":")
            meta[k.strip()] = v.strip()
        elif section == "seq":
            item = re.match(r"\s*-\s*id:\s*(.+)", line)
            if item:
                current = {"id": item.group(1).strip()}
                seq.append(current)
                continue
            if current is None:
                continue
            key, _, val = line.strip().partition(":")
            key, val = key.strip(), val.strip()
            if key == "contents":
                try:
                    current["contents"] = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    current["contents"] = []
            elif key == "size-eos":
                current["size-eos"] = val.lower() == "true"
            else:
                current[key] = val
    return meta, seq


def read_ksy_fields(path: str) -> tuple[list[Field], dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        meta, seq = _parse_with_yaml(text)
    except Exception:
        meta, seq = _parse_fallback(text)

    fields: list[Field] = []
    offset = 0
    for entry in seq:
        if not isinstance(entry, dict):
            continue
        f = _field_from_entry(entry, offset)
        fields.append(f)
        offset = f.end if f.size else offset
    return fields, {"id": meta.get("id", "fmt"), "endian": meta.get("endian", "le")}
