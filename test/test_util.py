import pytest
import collections
import datetime
import unittest
import itertools

import numpy as np  # type: ignore

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

from performance.reference.util import mloc as mloc_ref


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
        a1 = np.array((None, 'foo', True, mutable))
        a2 = array_deepcopy(a1, memo)

        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertIsNot(a1[3], a2[3])
        self.assertFalse(a2.flags.writeable)

    def test_array_deepcopy_c2(self) -> None:
        memo = {}
        mutable = [np.nan]
        a1 = np.array((None, 'foo', True, mutable))
        a2 = array_deepcopy(a1, memo)
        self.assertIsNot(a1, a2)
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertIsNot(a1[3], a2[3])
        self.assertFalse(a2.flags.writeable)
        self.assertIn(id(a1), memo)

    def test_array_deepcopy_d(self) -> None:
        memo = {}
        mutable = [3, 4, 5]
        a1 = np.array((None, 'foo', True, mutable))
        a2 = array_deepcopy(a1, memo=memo)
        self.assertIsNot(a1, a2)
        self.assertTrue(id(mutable) in memo)

    def test_array_deepcopy_e(self) -> None:
        a1 = np.array((3, 4, 5))
        with self.assertRaises(TypeError):
            # memo argument must be a dictionary
            a2 = array_deepcopy(a1, memo=None)

    def test_array_deepcopy_f(self) -> None:
        a1 = np.array((3, 4, 5))
        a2 = array_deepcopy(a1)
        self.assertNotEqual(id(a1), id(a2))

    def test_isna_element_true(self) -> None:
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

    def test_isna_element_false(self) -> None:
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
                (np.dtype('O'), np.object()),
                (np.dtype('<M8'), np.datetime64('NaT')),
                (np.dtype('<m8'), np.timedelta64('NaT')),
                (np.float_, np.nan),
        ]
        for dtype, obj in dtype_obj_pairs:
            self.assertEqual(dtype, dtype_from_element(obj))

    def test_dtype_from_element_obj_dtypes(self) -> None:
        NT = collections.namedtuple('NT', tuple('abc'))

        dtype_obj_pairs = [
                (np.int_, 12),
                (np.float_, 12.0),
                (np.bool_, True),
                (np.dtype('O'), None),
                (np.float_, float('NaN')),
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


if __name__ == '__main__':
    unittest.main()
