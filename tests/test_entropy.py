import os

import numpy as np

from autostruct import entropy
from autostruct.model import REGION_HIGH_ENTROPY, REGION_STRUCTURED


def test_low_entropy_constant_buffer():
    data = np.zeros(1024, dtype=np.uint8)
    prof = entropy.entropy_profile(data)
    assert prof.shape == (1024,)
    assert float(prof.max()) == 0.0


def test_high_entropy_random_buffer():
    # A 256-byte window of random data has a finite-sampling entropy ceiling
    # (~0.91 normalised); it is still clearly high vs. structured data.
    rng = np.frombuffer(os.urandom(2048), dtype=np.uint8)
    prof = entropy.entropy_profile(rng)
    assert float(prof.mean()) > 0.85


def test_segmentation_splits_structured_and_high():
    structured = np.zeros(512, dtype=np.uint8)
    high = np.frombuffer(os.urandom(1024), dtype=np.uint8)
    data = np.concatenate([structured, high])
    prof = entropy.entropy_profile(data)
    regions = entropy.segment(prof, threshold=0.85)
    kinds = {r.kind for r in regions}
    assert REGION_STRUCTURED in kinds
    assert REGION_HIGH_ENTROPY in kinds
    # regions must tile the whole buffer contiguously
    assert regions[0].start == 0
    assert regions[-1].end == data.size
    for a, b in zip(regions, regions[1:]):
        assert a.end == b.start


def test_empty():
    assert entropy.entropy_profile(np.zeros(0, np.uint8)).size == 0
    assert entropy.segment(np.zeros(0)) == []
