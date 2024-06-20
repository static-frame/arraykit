import pytest
import collections
import datetime
import unittest
import warnings
from io import StringIO
import numpy as np  # type: ignore
# import pandas as pd # disable so as to compile 32 bit wheels for python 3.12

from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter
from arraykit import shape_filter
from arraykit import column_2d_filter
from arraykit import column_1d_filter
from arraykit import row_1d_filter
from arraykit import mloc
from arraykit import immutable_filter
from arraykit import array_deepcopy
from arraykit import isna_element
from arraykit import dtype_from_element
from arraykit import count_iteration
from arraykit import first_true_1d
from arraykit import first_true_2d
from arraykit import slice_to_ascending_slice
from arraykit import array2d_to_array1d
from arraykit import array2d_tuple_iter

from performance.reference.util import get_new_indexers_and_screen_ak as get_new_indexers_and_screen_full
from arraykit import get_new_indexers_and_screen

from performance.reference.util import mloc as mloc_ref
from performance.reference.util import slice_to_ascending_slice as slice_to_ascending_slice_ref


class TestUnit(unittest.TestCase):

    def test_mloc_a(self) -> None:
        a1 = np.arange(10)
        self.assertEqual(mloc(a1), mloc_ref(a1))

    def test_immutable_filter_a(self) -> None:
        a1 = np.arange(10)
        self.assertFalse(immutable_filter(a1).flags.writeable)

    def test_resolve_dtype_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(resolve_dtype(a1.dtype, a1.dtype), a1.dtype)

        self.assertEqual(resolve_dtype(a1.dtype, a2.dtype), np.object_)
        self.assertEqual(resolve_dtype(a2.dtype, a3.dtype), np.object_)
        self.assertEqual(resolve_dtype(a2.dtype, a4.dtype), np.object_)
        self.assertEqual(resolve_dtype(a3.dtype, a4.dtype), np.object_)
        self.assertEqual(resolve_dtype(a3.dtype, a6.dtype), np.object_)

        self.assertEqual(resolve_dtype(a1.dtype, a4.dtype), np.float64)
        self.assertEqual(resolve_dtype(a1.dtype, a6.dtype), np.float64)
        self.assertEqual(resolve_dtype(a4.dtype, a6.dtype), np.float64)

    def test_resolve_dtype_b(self) -> None:

        self.assertEqual(
                resolve_dtype(np.array('a').dtype, np.array('aaa').dtype),
                np.dtype(('U', 3))
                )

    def test_resolve_dtype_c(self) -> None:


        a1 = np.array(['2019-01', '2019-02'], dtype=np.datetime64)
        a2 = np.array(['2019-01-01', '2019-02-01'], dtype=np.datetime64)
        a3 = np.array([0, 1], dtype='datetime64[ns]')
        a4 = np.array([0, 1])

        self.assertEqual(str(resolve_dtype(a1.dtype, a2.dtype)),
                'datetime64[D]')
        self.assertEqual(resolve_dtype(a1.dtype, a3.dtype).kind, 'M')
        self.assertEqual(
                np.datetime_data(resolve_dtype(a1.dtype, a3.dtype)),
                ('ns', 1))
        self.assertEqual(resolve_dtype(a1.dtype, a4.dtype),
                np.dtype('O'))


    def test_resolve_dtype_d(self) -> None:
        dt1 = np.array(1).dtype
        dt2 = np.array(2.3).dtype
        assert resolve_dtype(dt1, dt2) == np.dtype(float)

    def test_resolve_dtype_e(self) -> None:
        dt1 = np.array(1, dtype='timedelta64[D]').dtype
        dt2 = np.array(2, dtype='timedelta64[Y]').dtype
        assert resolve_dtype(dt1, dt2) == np.dtype(object)
        assert resolve_dtype(dt1, dt1) == dt1


    #---------------------------------------------------------------------------
    def test_resolve_dtype_iter_a(self) -> None:

        a1 = np.array([1, 2, 3])
        a2 = np.array([False, True, False])
        a3 = np.array(['b', 'c', 'd'])
        a4 = np.array([2.3, 3.2])
        a5 = np.array(['test', 'test again'], dtype='S')
        a6 = np.array([2.3,5.4], dtype='float32')

        self.assertEqual(resolve_dtype_iter((a1.dtype, a1.dtype)), a1.dtype)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype)), a2.dtype)

        # boolean with mixed types
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a3.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a5.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a2.dtype, a2.dtype, a6.dtype)), np.object_)

        # numerical types go to float64
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype)), np.float64)

        # add in bool or str, goes to object
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a2.dtype)), np.object_)
        self.assertEqual(resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a5.dtype)), np.object_)

        # mixed strings go to the largest
        self.assertEqual(resolve_dtype_iter((a3.dtype, a5.dtype)).kind, 'U')
        self.assertEqual(resolve_dtype_iter((a3.dtype, a5.dtype)).itemsize, 40)

        with pytest.raises(TypeError):
            resolve_dtype_iter((a3.dtype, int))

        self.assertEqual(resolve_dtype_iter((a1.dtype,)), a1.dtype)

        with pytest.raises(ValueError):
            resolve_dtype_iter(())

    #---------------------------------------------------------------------------

    def test_shape_filter_a(self) -> None:

        a1 = np.arange(10)
        self.assertEqual(shape_filter(a1), (10, 1))
        self.assertEqual(shape_filter(a1.reshape(2, 5)), (2, 5))
        self.assertEqual(shape_filter(a1.reshape(1, 10)), (1, 10))
        self.assertEqual(shape_filter(a1.reshape(10, 1)), (10, 1))

        a2 = np.arange(4)
        self.assertEqual(shape_filter(a2), (4, 1))
        self.assertEqual(shape_filter(a2.reshape(2, 2)), (2, 2))

        with self.assertRaises(NotImplementedError):
            shape_filter(a1.reshape(1,2,5))

        with self.assertRaises(NotImplementedError):
            # zero dimension
            shape_filter(np.array(1))

    #---------------------------------------------------------------------------

    def test_column_2d_filter_a(self) -> None:

        a1 = np.arange(10)
        self.assertEqual(column_2d_filter(a1).shape, (10, 1))
        self.assertEqual(column_2d_filter(a1.reshape(2, 5)).shape, (2, 5))
        self.assertEqual(column_2d_filter(a1.reshape(1, 10)).shape, (1, 10))

        with self.assertRaises(NotImplementedError):
            column_2d_filter(a1.reshape(1,2,5))


    #---------------------------------------------------------------------------

    def test_column_1d_filter_a(self) -> None:

        a1 = np.arange(10)
        self.assertEqual(column_1d_filter(a1).shape, (10,))
        self.assertEqual(column_1d_filter(a1.reshape(10, 1)).shape, (10,))

        with self.assertRaises(ValueError):
            column_1d_filter(a1.reshape(2, 5))

        with self.assertRaises(NotImplementedError):
            column_1d_filter(a1.reshape(1,2,5))

    #---------------------------------------------------------------------------

    def test_row_1d_filter_a(self) -> None:

        a1 = np.arange(10)
        self.assertEqual(row_1d_filter(a1).shape, (10,))
        self.assertEqual(row_1d_filter(a1.reshape(1, 10)).shape, (10,))

        with self.assertRaises(ValueError):
            row_1d_filter(a1.reshape(2, 5))

        with self.assertRaises(NotImplementedError):
            row_1d_filter(a1.reshape(1,2,5))

    #---------------------------------------------------------------------------

    def test_array_deepcopy_a1(self) -> None:
        a1 = np.arange(10)
        memo = {}
        a2 = array_deepcopy(a1, memo)

        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertFalse(a2.flags.writeable)
        self.assertEqual(a1.dtype, a2.dtype)

    def test_array_deepcopy_a2(self) -> None:
        a1 = np.arange(10)
        memo = {}
        a2 = array_deepcopy(a1, memo)

        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertIn(id(a1), memo)
        self.assertEqual(memo[id(a1)].tolist(), a2.tolist())
        self.assertFalse(a2.flags.writeable)


    def test_array_deepcopy_b(self) -> None:
        a1 = np.arange(10)
        memo = {id(a1): a1}
        a2 = array_deepcopy(a1, memo)

        self.assertEqual(mloc(a1), mloc(a2))


    def test_array_deepcopy_c1(self) -> None:
        mutable = [np.nan]
        memo = {}

        a1 = np.array((None, 'foo', True, None))
        a1[3] = mutable
        a2 = array_deepcopy(a1, memo)

        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertIsNot(a1[3], a2[3])
        self.assertFalse(a2.flags.writeable)

    def test_array_deepcopy_c2(self) -> None:
        memo = {}
        mutable = [np.nan]
        a1 = np.array((None, 'foo', True, None))
        a1[3] = mutable
        a2 = array_deepcopy(a1, memo)
        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertIsNot(a1[3], a2[3])
        self.assertFalse(a2.flags.writeable)
        self.assertIn(id(a1), memo)

    def test_array_deepcopy_d(self) -> None:
        memo = {}
        mutable = [3, 4, 5]
        a1 = np.array((None, 'foo', True, None))
        a1[3] = mutable
        a2 = array_deepcopy(a1, memo=memo)
        self.assertIsNot(a1, a2)
        self.assertTrue(id(mutable) in memo)

    def test_array_deepcopy_e(self) -> None:
        a1 = np.array((3, 4, 5))
        with self.assertRaises(TypeError):
            a2 = array_deepcopy(a1, memo='')

    def test_array_deepcopy_f(self) -> None:
        a1 = np.array((3, 4, 5))
        a2 = array_deepcopy(a1)
        self.assertNotEqual(id(a1), id(a2))

    def test_array_deepcopy_g(self) -> None:
        a1 = np.arange(10)
        a2 = array_deepcopy(a1, None)
        self.assertNotEqual(mloc(a1), mloc(a2))

    def test_array_deepcopy_h(self) -> None:
        a1 = np.arange(10)
        with self.assertRaises(TypeError):
            a2 = array_deepcopy(a1, ())

    #---------------------------------------------------------------------------
    def test_array2d_to_array1d_dummy(self) -> None:
        a1 = np.arange(10)
        with self.assertRaises(NotImplementedError):
            # 1 dimensional
            _ = array2d_to_array1d(a1)

    def test_array2d_to_array1d_b(self) -> None:
        a1 = np.arange(10, dtype=np.int64).reshape(5, 2)
        result = array2d_to_array1d(a1)
        assert isinstance(result[0], tuple)
        assert result[0] == (0, 1)
        self.assertIs(type(result[0][0]), np.int64)
        self.assertFalse(result.flags.writeable)
        self.assertEqual(tuple(result), ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)))


    def test_array2d_to_array1d_c(self) -> None:
        a1 = np.array([["a", "b"], ["ccc", "ddd"], ["ee", "ff"]])
        a2 = array2d_to_array1d(a1)
        self.assertEqual(a2.tolist(), [('a', 'b'), ('ccc', 'ddd'), ('ee', 'ff')])

    def test_array2d_to_array1d_d(self) -> None:
        a1 = np.array([[3, 5], [10, 20], [7, 2]], dtype=np.uint8)
        a2 = array2d_to_array1d(a1)
        self.assertEqual(a2.tolist(), [(3, 5), (10, 20), (7, 2)])
        self.assertIs(type(a2[0][0]), np.uint8)

    def test_array2d_to_array1d_e(self) -> None:
        a1 = np.arange(20, dtype=np.int64).reshape(4, 5)
        result = array2d_to_array1d(a1)
        self.assertEqual(result.tolist(), [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)])

    #---------------------------------------------------------------------------
    def test_array2d_tuple_iter_a(self) -> None:
        a1 = np.arange(20, dtype=np.int64).reshape(4, 5)
        result = list(array2d_tuple_iter(a1))
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9), (10, 11, 12, 13, 14), (15, 16, 17, 18, 19)])

    def test_array2d_tuple_iter_b(self) -> None:
        a1 = np.arange(20, dtype=np.int64).reshape(10, 2)
        result = list(array2d_tuple_iter(a1))
        self.assertEqual(result, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19)])

    def test_array2d_tuple_iter_c(self) -> None:
        a1 = np.array([['aaa', 'bb'], ['c', 'dd'], ['ee', 'fffff']])
        it = array2d_tuple_iter(a1)
        self.assertEqual(it.__length_hint__(), 3)
        self.assertEqual(next(it), ('aaa', 'bb'))
        self.assertEqual(it.__length_hint__(), 2)
        self.assertEqual(next(it), ('c', 'dd'))
        self.assertEqual(it.__length_hint__(), 1)
        self.assertEqual(next(it), ('ee', 'fffff'))
        self.assertEqual(it.__length_hint__(), 0)
        with self.assertRaises(StopIteration):
            next(it)

    def test_array2d_tuple_iter_d(self) -> None:
        a1 = np.array([['aaa', 'bb'], ['c', 'dd'], ['ee', 'fffff']])
        it = array2d_tuple_iter(a1)
        # __reversed__ not implemented
        with self.assertRaises(TypeError):
            reversed(it)

    def test_array2d_tuple_iter_e(self) -> None:
        a1 = np.array([[None, 'bb'], [None, 'dd'], [3, None]])
        it = array2d_tuple_iter(a1)
        del a1
        self.assertEqual(list(it), [(None, 'bb'), (None, 'dd'), (3, None)])

    def test_array2d_tuple_iter_f(self) -> None:
        a1 = np.array([[None, 'bb'], [None, 'dd'], [3, None]])
        it1 = array2d_tuple_iter(a1)
        del a1
        it2 = iter(it1)
        self.assertEqual(list(it1), [(None, 'bb'), (None, 'dd'), (3, None)])
        self.assertEqual(list(it2), []) # expected behavior

    def test_array2d_tuple_iter_g(self) -> None:
        a1 = np.array([[None, 'bb'], [None, 'dd'], [3, None]])
        it1 = array2d_tuple_iter(a1)
        it2 = array2d_tuple_iter(a1)
        del a1
        self.assertEqual(list(it1), [(None, 'bb'), (None, 'dd'), (3, None)])
        self.assertEqual(list(it2), [(None, 'bb'), (None, 'dd'), (3, None)])

    #---------------------------------------------------------------------------

    def test_isna_element_a(self) -> None:
        class FloatSubclass(float): pass
        class ComplexSubclass(complex): pass

        self.assertTrue(isna_element(np.datetime64('NaT')))
        self.assertTrue(isna_element(np.timedelta64('NaT')))

        nan = np.nan
        complex_nans = [
                complex(nan, 0),
                complex(-nan, 0),
                complex(0, nan),
                complex(0, -nan),
        ]

        float_classes = [float, np.float16, np.float32, np.float64, FloatSubclass]
        if hasattr(np, 'float128'):
            float_classes.append(np.float128)

        cfloat_classes = [complex, np.complex64, np.complex128, ComplexSubclass]
        if hasattr(np, 'complex256'):
            cfloat_classes.append(np.complex256)

        for float_class in float_classes:
            self.assertTrue(isna_element(float_class(nan)))
            self.assertTrue(isna_element(float_class(-nan)))

        for cfloat_class in cfloat_classes:
            for complex_nan in complex_nans:
                self.assertTrue(isna_element(cfloat_class(complex_nan)))

        self.assertTrue(isna_element(float('NaN')))
        self.assertTrue(isna_element(-float('NaN')))
        self.assertTrue(isna_element(None))

    def test_isna_element_b(self) -> None:
        # Test a wide range of float values, with different precision, across types
        for val in (
                1e-1000, 1e-309, 1e-39, 1e-16, 1e-5, 0.1, 0., 1.0, 1e5, 1e16, 1e39, 1e309, 1e1000,
            ):
            for sign in (1, -1):
                for ctor in (np.float16, np.float32, np.float64, float):
                    self.assertFalse(isna_element(ctor(sign * val)))

                if hasattr(np, 'float128'):
                    self.assertFalse(isna_element(np.float128(sign * val)))

        self.assertFalse(isna_element(1))
        self.assertFalse(isna_element('str'))
        self.assertFalse(isna_element(np.datetime64('2020-12-31')))
        self.assertFalse(isna_element(datetime.date(2020, 12, 31)))
        self.assertFalse(isna_element(False))


    def test_isna_element_c(self) -> None:
        self.assertFalse(isna_element(None, include_none=False))
        self.assertTrue(isna_element(None, include_none=True))
        self.assertFalse(isna_element(None, False))
        self.assertTrue(isna_element(None, True))

    # def test_isna_element_d(self) -> None:
    #     ts = pd.Timestamp('nat')
    #     self.assertTrue(isna_element(ts))

    #     s1 = pd.Series((0,))
    #     self.assertFalse(isna_element(s1))


    def test_isna_element_e(self) -> None:
        from types import SimpleNamespace
        sn = SimpleNamespace()
        sn.to_numpy = None
        self.assertFalse(isna_element(sn))


    #---------------------------------------------------------------------------

    def test_dtype_from_element_core_dtypes(self) -> None:
        dtypes = [
                np.longlong,
                np.int_,
                np.intc,
                np.short,
                np.byte,
                np.ubyte,
                np.ushort,
                np.uintc,
                np.uint,
                np.ulonglong,
                np.half,
                np.single,
                np.float_,
                np.longfloat,
                np.csingle,
                np.complex_,
                np.clongfloat,
                np.bool_,
        ]
        for dtype in dtypes:
            self.assertEqual(dtype, dtype_from_element(dtype()))

    def test_dtype_from_element_str_and_misc_dtypes(self) -> None:
        dtype_obj_pairs = [
                (np.dtype('<U1'), np.str_('1')),
                (np.dtype('<U1'), np.unicode_('1')),
                (np.dtype('V1'), np.void(1)),
                (np.dtype('O'), object),
                (np.dtype('<M8'), np.datetime64('NaT')),
                (np.dtype('<m8'), np.timedelta64('NaT')),
                (np.float_, np.nan),
        ]
        for dtype, obj in dtype_obj_pairs:
            self.assertEqual(dtype, dtype_from_element(obj))

    def test_dtype_from_element_obj_dtypes(self) -> None:
        NT = collections.namedtuple('NT', tuple('abc'))

        dtype_obj_pairs = [
                (np.int64, 12),
                (np.float64, 12.0),
                (np.bool_, True),
                (np.dtype('O'), None),
                (np.float64, float('NaN')),
                (np.dtype('O'), object()),
                (np.dtype('O'), (1, 2, 3)),
                (np.dtype('O'), NT(1, 2, 3)),
                (np.dtype('O'), datetime.date(2020, 12, 31)),
                (np.dtype('O'), datetime.timedelta(14)),
        ]
        for dtype, obj in dtype_obj_pairs:
            self.assertEqual(dtype, dtype_from_element(obj))

    def test_dtype_from_element_time_dtypes(self) -> None:
        # Datetime & Timedelta
        for precision in ['ns', 'us', 'ms', 's', 'm', 'h', 'D', 'M', 'Y']:
            for kind, ctor in (('m', np.timedelta64), ('M', np.datetime64)):
                obj = ctor(12, precision)
                self.assertEqual(np.dtype(f'<{kind}8[{precision}]'), dtype_from_element(obj))

    def test_dtype_from_element_str_and_bytes_dtypes(self) -> None:
        for size in (1, 8, 16, 32, 64, 128, 256, 512):
            self.assertEqual(np.dtype(f'|S{size}'), dtype_from_element(bytes(size)))
            self.assertEqual(np.dtype(f'<U{size}'), dtype_from_element('x' * size))

    def test_dtype_from_element_int(self) -> None:
        # make sure all platforms give 64 bit int
        self.assertEqual(str(dtype_from_element(3)), 'int64')

    #---------------------------------------------------------------------------

    def test_get_new_indexers_and_screen_a(self) -> None:
        indexersA = np.array([9, 9, 9, 9, 0, 0, 1, 4, 5, 0, 0, 0, 1], dtype=np.int64)
        postA = get_new_indexers_and_screen_full(indexersA, np.arange(10, dtype=np.int64))
        assert indexersA.flags.c_contiguous
        assert indexersA.flags.f_contiguous
        assert tuple(map(list, postA)) == (
            [9, 0, 1, 4, 5],
            [0, 0, 0, 0, 1, 1, 2, 3, 4, 1, 1, 1, 2],
        )

        # Prove we can handle non-continuous arrays
        indexersB = np.full((len(indexersA), 3), -1, dtype=np.int64)
        indexersB[:,1] = indexersA.copy()
        assert not indexersB[:,1].flags.c_contiguous
        assert not indexersB[:,1].flags.f_contiguous
        postB = get_new_indexers_and_screen_full(indexersB[:,1], np.arange(10, dtype=np.int64))
        assert tuple(map(list, postA)) == tuple(map(list, postB))

        indexersC = np.array([9, 9, 9, 9, 0, 0, 1, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        postC = get_new_indexers_and_screen_full(indexersC, positions=np.arange(15, dtype=np.int64))
        assert tuple(map(list, postC)) == (
            [9, 0, 1, 4, 5, 2, 3, 6, 7, 8, 10],
            [0, 0, 0, 0, 1, 1, 2, 3, 4, 1, 1, 1, 2, 5, 6, 3, 4,7, 8, 9, 0, 10],
        )

        indexersD = np.array([2, 1, 0, 2, 0, 1, 1, 2, 0], dtype=np.int64)
        postD = get_new_indexers_and_screen_full(indexers=indexersD, positions=np.arange(3, dtype=np.int64))
        assert tuple(map(list, postD)) == (
            [0, 1, 2],
            [2, 1, 0, 2, 0, 1, 1, 2, 0],
        )

    def test_get_new_indexers_and_screen_b(self) -> None:
        indexersA = np.array([5], dtype=np.int64)

        with self.assertRaises(ValueError):
            get_new_indexers_and_screen(indexersA, np.arange(6, dtype=np.int64))

        with self.assertRaises(ValueError):
            get_new_indexers_and_screen(indexersA, np.arange(106, dtype=np.int64))

        with self.assertRaises(ValueError):
            get_new_indexers_and_screen(indexersA.astype(np.int32), np.arange(5))

        with self.assertRaises(ValueError):
            get_new_indexers_and_screen(indexersA, np.arange(5).astype(np.int32))

        indexersB = np.arange(25, dtype=np.int64)
        postB = get_new_indexers_and_screen(indexersB, indexersB)
        assert tuple(map(list, postB)) == (list(indexersB), list(indexersB))

    #---------------------------------------------------------------------------
    def test_count_iteration_a(self) -> None:
        post = count_iteration(('a', 'b', 'c', 'd'))
        self.assertEqual(post, 4)

    def test_count_iteration_b(self) -> None:
        s1 = StringIO(',1,a,b\n-,1,43,54\nX,2,1,3\nY,1,8,10\n-,2,6,20')
        post = count_iteration(s1)
        self.assertEqual(post, 5)

    #---------------------------------------------------------------------------
    def test_first_true_1d_a(self) -> None:
        a1 = np.arange(100) == 50
        post = first_true_1d(a1, forward=True)
        self.assertEqual(post, 50)

    def test_first_true_1d_b(self) -> None:
        with self.assertRaises(TypeError):
            a1 = [2, 4, 5,]
            first_true_1d(a1, forward=True)

    def test_first_true_1d_c(self) -> None:
        with self.assertRaises(ValueError):
            a1 = np.arange(100) == 50
            first_true_1d(a1, forward=a1)

    def test_first_true_1d_d(self) -> None:
        a1 = np.arange(100) < 0
        post = first_true_1d(a1, forward=True)
        self.assertEqual(post, -1)

    def test_first_true_1d_e(self) -> None:
        a1 = np.arange(100)
        # only a Boolean array
        with self.assertRaises(ValueError):
            post = first_true_1d(a1, forward=True)

    def test_first_true_1d_f(self) -> None:
        a1 = (np.arange(100) == 0)[:50:2]
        # only a contiguous array
        with self.assertRaises(ValueError):
            post = first_true_1d(a1, forward=True)

    def test_first_true_1d_g(self) -> None:
        a1 = (np.arange(100) == 0).reshape(10, 10)
        # only a contiguous array
        with self.assertRaises(ValueError):
            post = first_true_1d(a1, forward=True)

    def test_first_true_1d_reverse_a(self) -> None:
        a1 = np.arange(100) == 50
        post = first_true_1d(a1, forward=False)
        self.assertEqual(post, 50)

    def test_first_true_1d_reverse_b(self) -> None:
        a1 = np.arange(100) == 0
        post = first_true_1d(a1, forward=False)
        self.assertEqual(post, 0)

    def test_first_true_1d_reverse_c(self) -> None:
        a1 = np.arange(100) == -1
        post = first_true_1d(a1, forward=False)
        self.assertEqual(post, -1)

    def test_first_true_1d_reverse_d(self) -> None:
        a1 = np.arange(100) == 99
        post = first_true_1d(a1, forward=False)
        self.assertEqual(post, 99)

    def test_first_true_1d_multi_a(self) -> None:
        a1 = np.isin(np.arange(100), (50, 70, 90))
        self.assertEqual(first_true_1d(a1, forward=True), 50)
        self.assertEqual(first_true_1d(a1, forward=False), 90)

    def test_first_true_1d_multi_b(self) -> None:
        a1 = np.isin(np.arange(100), (10, 30, 50))
        self.assertEqual(first_true_1d(a1, forward=True), 10)
        self.assertEqual(first_true_1d(a1, forward=False), 50)


    #---------------------------------------------------------------------------
    def test_first_true_2d_a(self) -> None:
        a1 = np.isin(np.arange(100), (9, 19, 38, 68, 96)).reshape(5, 20)

        post1 = first_true_2d(a1, axis=1, forward=True)
        # NOTE: this is an axis 1 result by argmax
        self.assertEqual(post1.tolist(),
                [9, 18, -1, 8, 16]
                )
        post2 = first_true_2d(a1, axis=1, forward=False)
        self.assertEqual(post2.tolist(),
                [19, 18, -1, 8, 16]
                )

    def test_first_true_2d_b(self) -> None:
        a1 = np.isin(np.arange(20), (3, 7, 10, 15, 18)).reshape(5, 4)

        post1 = first_true_2d(a1, axis=1, forward=False)
        self.assertEqual(post1.tolist(),
                [3, 3, 2, 3, 2]
                )
        post2 = first_true_2d(a1, axis=1, forward=True)
        self.assertEqual(post2.tolist(),
                [3, 3, 2, 3, 2]
                )

        post3 = first_true_2d(a1, axis=0, forward=False)
        self.assertEqual(post3.tolist(),
                [-1, -1, 4, 3]
                )
        post4 = first_true_2d(a1, axis=0, forward=True)
        self.assertEqual(post4.tolist(),
                [-1, -1, 2, 0]
                )

    def test_first_true_2d_c(self) -> None:
        a1 = np.isin(np.arange(20), ()).reshape(5, 4)

        post1 = first_true_2d(a1, axis=1, forward=False)
        self.assertEqual(post1.tolist(),
                [-1, -1, -1, -1, -1]
                )
        post2 = first_true_2d(a1, axis=1, forward=True)
        self.assertEqual(post2.tolist(),
                [-1, -1, -1, -1, -1]
                )

        post3 = first_true_2d(a1, axis=0, forward=False)
        self.assertEqual(post3.tolist(),
                [-1, -1, -1, -1]
                )
        post4 = first_true_2d(a1, axis=0, forward=True)
        self.assertEqual(post4.tolist(),
                [-1, -1, -1, -1]
                )


    def test_first_true_2d_d(self) -> None:
        a1 = np.isin(np.arange(20), (0, 3, 4, 7, 8, 11, 12, 15, 16, 19)).reshape(5, 4)

        post1 = first_true_2d(a1, axis=1, forward=False)
        self.assertEqual(post1.tolist(),
                [3, 3, 3, 3, 3]
                )
        post2 = first_true_2d(a1, axis=1, forward=True)
        self.assertEqual(post2.tolist(),
                [0, 0, 0, 0, 0]
                )

        post3 = first_true_2d(a1, axis=0, forward=True)
        self.assertEqual(post3.tolist(),
                [0, -1, -1, 0]
                )
        post4 = first_true_2d(a1, axis=0, forward=False)
        self.assertEqual(post4.tolist(),
                [4, -1, -1, 4]
                )

    def test_first_true_2d_e(self) -> None:
        a1 = np.isin(np.arange(15), (2, 7, 12)).reshape(3, 5)

        post1 = first_true_2d(a1, axis=1, forward=False)
        self.assertEqual(post1.tolist(),
                [2, 2, 2]
                )
        post2 = first_true_2d(a1, axis=1, forward=True)
        self.assertEqual(post2.tolist(),
                [2, 2, 2]
                )

    def test_first_true_2d_f(self) -> None:
        a1 = np.isin(np.arange(15), (2, 7, 12)).reshape(3, 5)

        with self.assertRaises(ValueError):
            post1 = first_true_2d(a1, axis=-1)

        with self.assertRaises(ValueError):
            post1 = first_true_2d(a1, axis=2)


    def test_first_true_2d_f(self) -> None:
        a1 = np.isin(np.arange(15), (1, 7, 14)).reshape(3, 5)
        post1 = first_true_2d(a1, axis=0, forward=True)
        self.assertEqual(post1.tolist(), [-1, 0, 1, -1, 2])

        post2 = first_true_2d(a1, axis=0, forward=False)
        self.assertEqual(post2.tolist(), [-1, 0, 1, -1, 2])


    def test_first_true_2d_g(self) -> None:
        a1 = np.isin(np.arange(15), (1, 7, 14)).reshape(3, 5).T # force fortran ordering
        self.assertEqual(first_true_2d(a1, axis=0, forward=True).tolist(),
                [1, 2, 4])
        self.assertEqual(first_true_2d(a1, axis=0, forward=False).tolist(),
                [1, 2, 4])
        self.assertEqual(first_true_2d(a1, axis=1, forward=True).tolist(),
                [-1, 0, 1, -1, 2])
        self.assertEqual(first_true_2d(a1, axis=1, forward=False).tolist(),
                [-1, 0, 1, -1, 2])


    def test_first_true_2d_h(self) -> None:
        # force fortran ordering, non-contiguous, non-owned
        a1 = np.isin(np.arange(15), (1, 4, 5, 7, 8, 12, 15)).reshape(3, 5).T[:4]
        self.assertEqual(first_true_2d(a1, axis=0, forward=True).tolist(),
                [1, 0, 2])
        self.assertEqual(first_true_2d(a1, axis=0, forward=False).tolist(),
                [1, 3, 2])
        self.assertEqual(first_true_2d(a1, axis=1, forward=True).tolist(),
                [1, 0, 1, 1])
        self.assertEqual(first_true_2d(a1, axis=1, forward=False).tolist(),
                [1, 0, 2, 1])


    #---------------------------------------------------------------------------
    def test_slice_to_ascending_slice_a(self) -> None:
        self.assertEqual(slice_to_ascending_slice(
                slice(5, 2, -1), 6),
                slice(3, 6, None),
                )

    def test_slice_to_ascending_slice_b(self) -> None:
        self.assertEqual(slice_to_ascending_slice(
                slice(2, 5, 1), 6),
                slice(2, 5, 1),
                )

    def test_slice_to_ascending_slice_c(self) -> None:
        with self.assertRaises(TypeError):
            _ = slice_to_ascending_slice('a', 6)

        with self.assertRaises(TypeError):
            _ = slice_to_ascending_slice(slice(1, 4), 'x')

    def test_slice_to_ascending_slice_d(self) -> None:
        self.assertEqual(slice_to_ascending_slice(
                slice(10, 2, -2), 12),
                slice(4, 11, 2),
                )

    def test_slice_to_ascending_slice_e(self) -> None:
        for slc, size in (
                (slice(10, 2, -2), 12),
                (slice(12, 2, -3), 12),
                (slice(12, None, -4), 12),
                (slice(76, 12, -8), 100),
                (slice(81, 33, -12), 100),
                (slice(97, 6, -7), 101),
                ):
            self.assertEqual(
                slice_to_ascending_slice(slc, size),
                slice_to_ascending_slice_ref(slc, size),
                )

    def test_slice_to_ascending_slice_f(self) -> None:

        a1 = np.arange(10)

        def compare(slc: slice) -> None:
            slc_asc = slice_to_ascending_slice(slc, len(a1))
            self.assertEqual(sorted(a1[slc]), list(a1[slc_asc]))

        compare(slice(4,))
        compare(slice(6, 1, -1))
        compare(slice(6, 1, -2))
        compare(slice(6, None, -3))
        compare(slice(6, 2, -2))
        compare(slice(None, 1, -1))

    def test_slice_to_ascending_slice_g(self) -> None:
        self.assertEqual(
            slice_to_ascending_slice(slice(3, None, -1), 10),
            slice(0, 4, None)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(3, None, -3), 10),
            slice(0, 4, 3)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-3, 0, -1), 10),
            slice(1, 8, None)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-3, None, -1), 10),
            slice(0, 8, None)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-3, 0, -2), 10),
            slice(1, 8, 2)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-3, None, -2), 10),
            slice(1, 8, 2)
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-3, None, -6), 10),
            slice(1, 8, 6)
            )

    def test_slice_to_ascending_slice_h(self) -> None:
        self.assertEqual(
            slice_to_ascending_slice(slice(-9, -1, 1), 10),
            slice(-9, -1, 1) # ascenidng
            )
        self.assertEqual(
            slice_to_ascending_slice(slice(-9, -1, -1), 10),
            slice(2, 2, None) # ascending start stop, descending
            )

    def test_slice_to_ascending_slice_i(self) -> None:
        self.assertEqual(
            slice_to_ascending_slice(slice(1, -10, -1), 10), # [1]
            slice(1, 2, None)
            )




if __name__ == '__main__':
    unittest.main()
