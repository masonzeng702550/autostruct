import os
import tempfile

from autostruct.emit import render_html, render_ksy, render_summary
from autostruct.ksy_reader import read_ksy_fields
from autostruct.model import (
    KIND_INT,
    KIND_MAGIC,
    Field,
    StructModel,
    Validation,
)


def _model():
    return StructModel(
        meta={"id": "demo_fmt", "endian": "le"},
        fields=[
            Field(0, 4, "magic", KIND_MAGIC, 0.9, "magic at 0",
                  contents=[0x7f, 0x45, 0x4c, 0x46]),
            Field(4, 4, "length", "length", 0.8, "length prefix",
                  type="u4", endianness="le",
                  extra={"sizes_following": True, "values": [1, 2]}),
        ],
        n_samples=3,
        aligned_length=8,
        validation=Validation(passed=3, total=3),
    )


def test_render_ksy_has_meta_and_contents():
    text = render_ksy(_model())
    assert "meta:" in text
    assert "id: demo_fmt" in text
    assert "contents: [0x7f, 0x45, 0x4c, 0x46]" in text
    assert "type: u4le" in text
    assert "# guessed:" in text


def test_render_summary_includes_passrate():
    out = render_summary(_model())
    assert "demo_fmt" in out
    assert "3/3" in out


def test_render_html_is_self_contained():
    html = render_html(_model(), b"\x7fELF\x01\x00\x00\x00", title="demo_fmt")
    assert "<style>" in html
    assert "structure map" in html
    assert "demo_fmt" in html
    assert "http://" not in html.replace("http://www.w3.org", "")  # no external assets


def test_ksy_roundtrip_through_reader():
    text = render_ksy(_model())
    with tempfile.NamedTemporaryFile("w", suffix=".ksy", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        fields, meta = read_ksy_fields(path)
        assert meta["id"] == "demo_fmt"
        kinds = [f.kind for f in fields]
        assert KIND_MAGIC in kinds or any(f.contents for f in fields)
    finally:
        os.unlink(path)
