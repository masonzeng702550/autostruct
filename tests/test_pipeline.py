import os

import numpy as np

from autostruct.model import KIND_LENGTH, KIND_MAGIC, KIND_REPEAT
from autostruct.pipeline import InferConfig, infer_samples

from conftest import make_format_samples


def test_pipeline_detects_magic_and_length(format_samples):
    model = infer_samples(format_samples, config=InferConfig(verbose=False))
    kinds = [f.kind for f in model.fields]
    assert KIND_MAGIC in kinds, "should detect 4-byte magic at offset 0"
    magic = next(f for f in model.fields if f.kind == KIND_MAGIC)
    assert magic.offset == 0
    # the constant version byte may merge into magic; the \x7fELF prefix is what matters
    assert magic.contents[:4] == list(b"\x7fELF")
    assert KIND_LENGTH in kinds, "should detect the u4 length field"


def test_pipeline_closed_loop_passes(format_samples):
    model = infer_samples(format_samples)
    assert model.validation is not None
    assert model.validation.pass_rate >= 0.7


def test_pipeline_fields_tile_contiguously(format_samples):
    model = infer_samples(format_samples)
    cursor = 0
    for f in sorted(model.fields, key=lambda x: x.offset):
        assert f.offset == cursor, f"gap/overlap at {f.offset}, expected {cursor}"
        cursor = f.end if f.size else cursor


def test_png_magic_detection(png_samples):
    model = infer_samples(png_samples, config=InferConfig(name="png"))
    magic = next((f for f in model.fields if f.kind == KIND_MAGIC), None)
    assert magic is not None
    assert magic.contents[:4] == [0x89, 0x50, 0x4e, 0x47]


def test_repeat_records_detected(repeat_samples):
    model = infer_samples(repeat_samples, config=InferConfig(entropy_threshold=0.99))
    assert any(f.kind == KIND_REPEAT for f in model.fields)


def test_single_sample_low_confidence():
    samples = make_format_samples(n=1)
    model = infer_samples(samples)
    # no cross-sample evidence → no magic, all confidences modest
    assert all(f.confidence <= 0.6 for f in model.fields)


def test_empty_and_tiny_do_not_crash():
    model = infer_samples([b"\x01\x02", b"\x01\x03"], config=InferConfig(do_validate=True))
    assert model.aligned_length == 2
    assert model.validation is not None
