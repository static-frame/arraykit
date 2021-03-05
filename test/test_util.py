import collections
import datetime
import unittest

import numpy as np  # type: ignore

from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter
from arraykit import shape_filter
from arraykit import column_2d_filter
from arraykit import column_1d_filter
from arraykit import row_1d_filter
from arraykit import mloc
from arraykit import immutable_filter
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
        self.assertEqual(resolve_dtype(a1.dtype, a3.dtype),
                np.dtype('<M8[ns]'))

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
        self.assertEqual(resolve_dtype_iter((a3.dtype, a5.dtype)), np.dtype('<U10'))

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


    def test_isna_element_true(self) -> None:
        self.assertTrue(isna_element(np.datetime64('NaT')))
        self.assertTrue(isna_element(np.timedelta64('NaT')))
        self.assertTrue(isna_element(float('NaN')))
        self.assertTrue(isna_element(np.nan))
        self.assertTrue(isna_element(None))

    def test_isna_element_false(self) -> None:
        # Test a wide range of float values, with different precision, across types
        for val in (
                1e-1000, 1e-309, 1e-39, 1e-16, 1e-5, 0.1, 0., 1.0, 1e5, 1e16, 1e39, 1e309, 1e1000,
            ):
            for sign in (1, -1):
                for ctor in (np.float16, np.float32, np.float64, float):
                    self.assertFalse(isna_element((ctor(sign * val))))

                if hasattr(np, 'float128'):
                    self.assertFalse(isna_element((np.float128(sign * val))))

        self.assertFalse(isna_element(1))
        self.assertFalse(isna_element('str'))
        self.assertFalse(isna_element(np.datetime64('2020-12-31')))
        self.assertFalse(isna_element(datetime.date(2020, 12, 31)))
        self.assertFalse(isna_element(False))

    def test_dtype_from_element(self) -> None:
        NT = collections.namedtuple('NT', tuple('abc'))

        dtype_val_pairs = [
                (np.longlong, -1),(np.int_, -1),(np.intc, -1),(np.short, -1),
                (np.byte, -1),(np.ubyte, 1),(np.ushort, 1),(np.uintc, 1),
                (np.uint, 1),(np.ulonglong, 1),(np.half, 1.0),(np.single, 1.0),
                (np.float_, 1.0),(np.longfloat, 1.0),(np.csingle, 1.0j),
                (np.complex_, 1.0j),(np.clongfloat, 1.0j),(np.bool_, 0),
        ]
        for dtype, val in dtype_val_pairs:
            obj = dtype(val)
            self.assertEqual(dtype, dtype_from_element(obj))

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

        # Datetime & Timedelta
        for precision in ['ns', 'us', 'ms', 's', 'm', 'h', 'D', 'M', 'Y']:
            for kind, ctor in (('m', np.timedelta64), ('M', np.datetime64)):
                obj = ctor(12, precision)
                self.assertEqual(np.dtype(f'<{kind}8[{precision}]'), dtype_from_element(obj))

        for size in (1, 8, 16, 32, 64, 128, 256, 512):
            self.assertEqual(np.dtype(f'|S{size}'), dtype_from_element(bytes(size)))
            self.assertEqual(np.dtype(f'<U{size}'), dtype_from_element('x' * size))


if __name__ == '__main__':
    unittest.main()
