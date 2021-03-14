#!/usr/bin/env python
from functools import partial

import numpy as np  # type: ignore
from arraykit import isin_array

funcTT = partial(isin_array, array_is_unique=True, other_is_unique=True)
funcTF = partial(isin_array, array_is_unique=True, other_is_unique=False)
funcFT = partial(isin_array, array_is_unique=False, other_is_unique=True)
funcFF = partial(isin_array, array_is_unique=False, other_is_unique=False)

# ------------------------------------------------------------------------------
# ------------------------------------- 1D -------------------------------------

def dtype_unique_1d(func):
    expected = np.array([1, 0, 0, 0, 1], dtype=np.bool_)
    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.int_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.float_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.complex_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([1, 5, 2, 3, 4], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([1, 5, 2, 3, 4], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_unique_1d(funcTT)
dtype_unique_1d(funcTF)
dtype_unique_1d(funcFT)
dtype_unique_1d(funcFF)

def dtype_arr1_non_unique_1d(func):
    expected = np.array([1, 0, 0, 0, 1, 0, 1], dtype=np.bool_)
    arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.int_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.float_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=np.complex_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_arr1_non_unique_1d(funcFT)
dtype_arr1_non_unique_1d(funcFF)

def dtype_arr2_non_unique_1d(func):
    expected = np.array([1, 0, 0, 0, 1], dtype=np.bool_)
    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.int_)
    arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.float_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([1, 5, 2, 3, 4], dtype=np.complex_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([1, 5, 2, 3, 4], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([1, 5, 2, 3, 4], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_arr2_non_unique_1d(funcTF)
dtype_arr2_non_unique_1d(funcFF)

# ------------------------------------------------------------------------------
# ------------------------------------- 2D -------------------------------------

def dtype_unique_2d(func):
    expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.bool_)
    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.int_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.float_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.complex_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_unique_2d(funcTT)
dtype_unique_2d(funcTF)
dtype_unique_2d(funcFT)
dtype_unique_2d(funcFF)

def dtype_arr2_non_unique_1d(func):
    expected = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=np.bool_)
    arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]], dtype=np.int_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]], dtype=np.float_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]], dtype=np.complex_)
    arr2 = np.array([1, 4, 7, 9], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([[9, 1, 2, 3], [4, 3, 5, 9]], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 4, 7, 9], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_arr2_non_unique_1d(funcFT)
dtype_arr2_non_unique_1d(funcFF)

def dtype_arr2_non_unique_1d(func):
    expected = np.array([[1, 0, 0], [1, 0, 1]], dtype=np.bool_)
    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.int_)
    arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=np.int_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.float_)
    arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=np.float_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=np.complex_)
    arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=np.complex_)
    post = func(array=arr1, other=arr2)
    assert np.array_equal(expected, post)

    for freq in 'DMY':
        arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=f'datetime64[{freq}]')
        arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=f'datetime64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

        arr1 = np.array([[1, 2, 3], [4, 5, 9]], dtype=f'timedelta64[{freq}]')
        arr2 = np.array([1, 9, 4, 7, 9, 1], dtype=f'timedelta64[{freq}]')
        post = func(array=arr1, other=arr2)
        assert np.array_equal(expected, post)

dtype_arr2_non_unique_1d(funcTF)
dtype_arr2_non_unique_1d(funcFF)
