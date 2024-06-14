import numpy as np
from arraykit import is_sorted as ak_is_sorted
from performance.reference.util import DTYPE_STR_KINDS

def is_sorted_correct(arr: np.ndarray) -> bool:
    return np.all(np.sort(arr) == arr)

arr_non_contiguous = np.arange(25).reshape(5,5)[:, 0]
arr_contiguous = arr_non_contiguous.copy()

assert not arr_non_contiguous.flags.c_contiguous
assert arr_contiguous.flags.c_contiguous

for dtype in DTYPE_STR_KINDS:
    arr1 = arr_contiguous.astype(dtype)
    arr2 = arr_non_contiguous.astype(dtype)
    assert (arr1 == arr2).all()

    assert not is_sorted_correct(arr1)
    assert not is_sorted_correct(arr2)
    assert is_sorted_correct(np.sort(arr1))
    assert is_sorted_correct(np.sort(arr2))

    assert not ak_is_sorted(arr1)
    assert not ak_is_sorted(arr2)
    assert ak_is_sorted(np.sort(arr1))
    assert ak_is_sorted(np.sort(arr2))