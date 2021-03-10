from datetime import date
from functools import partial
import itertools
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

    def test_isin_1d(self) -> None:
        from performance.reference.util import isin_array

        T, F = True, False
        arr1 = np.array([1, 2, 3, 4, 5])

        expected = [
                (np.array([T, F, T, T, F]), [1, 3, 4]),
                (np.array([F, F, F, F, F]), [7, 8]),
                (np.array([T, T, T, T, T]), [1, 2, 3, 4, 5]),
        ]

        for expected_result, values in expected:
            for dtype in (int, object):
                arr2 = np.array(values, dtype=dtype)

                for aiu, oiu in itertools.product((T, F), (T, F)):
                    self.assertTrue(np.array_equal(expected_result, isin_array(
                            array=arr1,
                            array_is_unique=aiu,
                            other=arr2,
                            other_is_unique=oiu,
                    )))

    def test_isin_2d(self) -> None:
        from performance.reference.util import isin_array

        T, F = True, False
        arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        expected = [
                (np.array([[T, F, T], [T, F, F], [F, F, T]]), [1, 3, 4, 9]),
                (np.array([[F, F, F], [F, F, F], [F, F, F]]), [10, 11]),
                (np.array([[T, T, T], [T, T, T], [T, T, T]]), [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ]

        for expected_result, values in expected:
            for dtype in (int, object):
                arr2 = np.array(values, dtype=dtype)

                for aiu, oiu in itertools.product((T, F), (T, F)):
                    self.assertTrue(np.array_equal(expected_result, isin_array(
                            array=arr1,
                            array_is_unique=aiu,
                            other=arr2,
                            other_is_unique=oiu,
                    )))

    def test_risky(self) -> None:
        from arraykit import isin_array

        isin_array_func = partial(isin_array, array_is_unique=True, other_is_unique=True)

        e_1d = np.array([1, 0, 0, 1, 0], dtype=bool)
        e_2d = np.array([[1, 0, 0], [1, 0, 1]], dtype=bool)

        v_1d = [1, 2, 3, 4, 5]
        v_2d = [[1, 2, 3], [4, 5, 9]]

        w_1d = [1, 4, 7, 9]

        dtype_funcs = [
                (int, int),
                (float, float),
                (str, str),
                ('datetime64[D]', lambda x: date(2020, 1, x)),
        ]

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([dtype_func(v) for v in v_1d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertTrue(np.array_equal(e_1d, post), msg=(e_1d, post))

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([[dtype_func(x) for x in y] for y in v_2d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertFalse(np.array_equal(e_2d, post), msg=(e_2d, post))


if __name__ == '__main__':
    unittest.main()
