#!/usr/bin/env python
from functools import partial

import numpy as np  # type: ignore
from arraykit import isin_array

funcTT = partial(isin_array, array_is_unique=True, other_is_unique=True)
funcTF = partial(isin_array, array_is_unique=True, other_is_unique=False)
funcFT = partial(isin_array, array_is_unique=False, other_is_unique=True)
funcFF = partial(isin_array, array_is_unique=False, other_is_unique=False)

class Obj:
    def __init__(self, value):
        self.v = value
    def __hash__(self):
        return hash(self.v)
    def __eq__(self, other):
        return self.v == other.v

arr1 = np.array([[Obj(1), Obj(2), Obj(3)], [Obj(4), Obj(5), Obj(9)]], dtype=object)
arr2 = np.array([Obj(1), Obj(4), Obj(7), Obj(9)], dtype=object)
post = funcTT(array=arr1, other=arr2)
print(post)

arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.int_)
arr2 = np.array([1, 4, 7, 9], dtype=np.int_)
post = funcFF(array=arr1, other=arr2)

arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.float_)
arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
post = funcFF(array=arr1, other=arr2)

arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=str)
arr2 = np.array([1, 4, 7, 9], dtype=str)
post = funcFF(array=arr1, other=arr2)

arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.complex_)
arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
post = funcFF(array=arr1, other=arr2)


def test_arrays(arr1, arr2, expected, func):
    post = func(array=arr1.astype(np.int_), other=arr2.astype(np.int_))
    assert np.array_equal(expected, post)

    post = func(array=arr1.astype(np.float_), other=arr2.astype(np.float_))
    assert np.array_equal(expected, post)

    post = func(array=arr1.astype(np.complex_), other=arr2.astype(np.complex_))
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        post = func(array=arr1.astype(f'datetime64[{freq}]'), other=arr2.astype(f'datetime64[{freq}]'))
        assert np.array_equal(expected, post)

        post = func(array=arr1.astype(f'timedelta64[{freq}]'), other=arr2.astype(f'timedelta64[{freq}]'))
        assert np.array_equal(expected, post)


# ------------------------------------------------------------------------------
# ------------------------------------- 1D -------------------------------------

def dtype_unique_1d(func):
    arr1 = np.array([1, 5, 2, 3, 4])
    arr2 = np.array([1, 4, 7, 9])
    expected = np.array([1, 0, 0, 0, 1], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


def dtype_arr1_non_unique_1d(func):
    arr1 = np.array([1, 5, 2, 3, 4, 5, 1])
    arr2 = np.array([1, 4, 7, 9])
    expected = np.array([1, 0, 0, 0, 1, 0, 1], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


def dtype_arr2_non_unique_1d(func):
    arr1 = np.array([1, 5, 2, 3, 4])
    arr2 = np.array([1, 9, 4, 7, 9, 1])
    expected = np.array([1, 0, 0, 0, 1], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


# ------------------------------------------------------------------------------
# ------------------------------------- 2D -------------------------------------

def dtype_unique_2d(func):
    arr1 = np.array([[1, 2, 3], [4, 5, 9]])
    arr2 = np.array([1, 4, 7, 9])
    expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


def dtype_arr2_non_unique_1d(func):
    arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]])
    arr2 = np.array([1, 4, 7, 9])
    expected = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


def dtype_arr2_non_unique_1d(func):
    arr1 = np.array([[1, 2, 3], [4, 5, 9]])
    arr2 = np.array([1, 9, 4, 7, 9, 1])
    expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.bool_)
    test_arrays(arr1, arr2, expected, func)


dtype_unique_1d(funcTT)
dtype_unique_1d(funcTF)
dtype_unique_1d(funcFT)
dtype_unique_1d(funcFF)
dtype_unique_2d(funcTT)
dtype_unique_2d(funcTF)
dtype_unique_2d(funcFT)
dtype_unique_2d(funcFF)

dtype_arr1_non_unique_1d(funcFT)
dtype_arr1_non_unique_1d(funcFF)
dtype_arr2_non_unique_1d(funcTF)
dtype_arr2_non_unique_1d(funcFF)

dtype_arr2_non_unique_1d(funcFT)
dtype_arr2_non_unique_1d(funcFF)
dtype_arr2_non_unique_1d(funcTF)
dtype_arr2_non_unique_1d(funcFF)
