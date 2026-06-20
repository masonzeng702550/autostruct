import numpy as np

from autostruct import stats


def test_constant_and_variable_columns():
    matrix = np.array([
        [0x7f, 0x45, 10, 1],
        [0x7f, 0x45, 20, 2],
        [0x7f, 0x45, 30, 3],
    ], dtype=np.uint8)
    st = stats.compute(matrix)
    assert list(st.constant) == [True, True, False, False]
    assert st.distinct[0] == 1
    assert st.distinct[2] == 3


def test_monotonic_detection():
    matrix = np.array([
        [1, 9],
        [2, 5],
        [3, 1],
    ], dtype=np.uint8)
    st = stats.compute(matrix)
    assert st.monotonic[0]       # 1,2,3 increasing
    assert not st.monotonic[1]   # 9,5,1 decreasing


def test_single_sample_no_monotonic():
    st = stats.compute(np.array([[1, 2, 3]], dtype=np.uint8))
    assert not st.monotonic.any()
    assert all(st.constant)  # trivially constant with one sample
