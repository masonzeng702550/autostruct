import struct

import numpy as np

from autostruct.fields import (
    const_detector,
    endian_scorer,
    int_typer,
    repeat_detector,
    string_detector,
)
from autostruct.fields.length_detector import annotate
from autostruct.model import KIND_LENGTH, KIND_MAGIC, KIND_STRING


def test_const_detector_magic_at_offset_zero():
    matrix = np.array([[0x7f, 0x45, 0x4c, 0x46]] * 4, dtype=np.uint8)
    f = const_detector.detect(matrix, 0, 4, n=4)
    assert f.kind == KIND_MAGIC
    assert f.contents == [0x7f, 0x45, 0x4c, 0x46]
    assert f.confidence > 0.7


def test_const_detector_sample_weight_caps_single():
    matrix = np.array([[1, 2]], dtype=np.uint8)
    f = const_detector.detect(matrix, 0, 2, n=1)
    assert f.confidence <= 0.66  # 1 - 1/2 with small ASCII bump


def test_endian_scorer_prefers_le_for_small_values():
    # values 1,2,3 encoded little-endian u4 → high bytes zero
    rows = [struct.pack("<I", v) for v in (1, 2, 3)]
    matrix = np.array([np.frombuffer(r, np.uint8) for r in rows], dtype=np.uint8)
    guess = endian_scorer.best_guess(matrix, 0, 4)
    assert guess.endian == "le"
    assert guess.values == [1, 2, 3]
    assert guess.monotonic


def test_int_typer_carves_u4():
    rows = [struct.pack("<I", v) for v in (5, 9, 13, 30)]
    matrix = np.array([np.frombuffer(r, np.uint8) for r in rows], dtype=np.uint8)
    fields = int_typer.type_span(matrix, 0, 4, n=4)
    assert len(fields) == 1
    assert fields[0].size == 4
    assert fields[0].type == "u4"


def test_length_detector_flags_remaining_length():
    # u4 field value tracks remaining length (== total - 4) across samples
    lengths = [24, 32, 40]
    values = [20, 28, 36]  # == length - 4 (remaining after the 4-byte field)
    rows = [struct.pack("<I", v) for v in values]
    matrix = np.array([np.frombuffer(r, np.uint8) for r in rows], dtype=np.uint8)
    fields = int_typer.type_span(matrix, 0, 4, n=3)
    annotate(fields, lengths)
    lf = [f for f in fields if f.kind == KIND_LENGTH]
    assert lf, "expected a length field"
    assert lf[0].offset == 0


def test_string_detector_ascii_strz():
    rows = [b"hello\x00\x00\x00", b"world\x00\x00\x00", b"abcde\x00\x00\x00"]
    matrix = np.array([np.frombuffer(r, np.uint8) for r in rows], dtype=np.uint8)
    f = string_detector.detect(matrix, 0, 8, n=3)
    assert f is not None
    assert f.kind == KIND_STRING
    assert f.type == "strz"


def test_repeat_detector_finds_period():
    record = bytes(range(16))
    data = np.frombuffer(record * 8, dtype=np.uint8)
    hit = repeat_detector.find_period(data, 0, data.size)
    assert hit is not None
    assert hit.period == 16
    assert hit.count == 8
