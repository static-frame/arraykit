from datetime import date, timedelta
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
from arraykit import array_deepcopy

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

    #---------------------------------------------------------------------------

    def test_array_deepcopy_a1(self) -> None:
        a1 = np.arange(10)
        memo = {}
        a2 = array_deepcopy(a1, memo)

        self.assertNotEqual(id(a1), id(a2))
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertFalse(a2.flags.writeable)
        self.assertEqual(a1.dtype, a2.dtype)

    def test_array_deepcopy_a2(self) -> None:
        a1 = np.arange(10)
        memo = {}
        a2 = array_deepcopy(a1, memo)

        self.assertNotEqual(id(a1), id(a2))
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

        self.assertNotEqual(id(a1), id(a2))
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertNotEqual(id(a1[3]), id(a2[3]))
        self.assertFalse(a2.flags.writeable)

    def test_array_deepcopy_c2(self) -> None:
        memo = {}
        mutable = [np.nan]
        a1 = np.array((None, 'foo', True, mutable))
        a2 = array_deepcopy(a1, memo)
        self.assertNotEqual(id(a1), id(a2))
        self.assertNotEqual(mloc(a1), mloc(a2))
        self.assertNotEqual(id(a1[3]), id(a2[3]))
        self.assertFalse(a2.flags.writeable)
        self.assertIn(id(a1), memo)

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

    def test_1d_2d_dtype_unique(self) -> None:
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
                ('timedelta64[D]', timedelta),
        ]

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([dtype_func(v) for v in v_1d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertTrue(np.array_equal(e_1d, post), msg=f'\n{dtype}\nExpected:\n{e_1d}\nActual:\n{post}')

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([[dtype_func(x) for x in y] for y in v_2d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertTrue(np.array_equal(e_2d, post), msg=f'\n{dtype}\nExpected:\n{e_2d}\nActual:\n{post}')

    def test_1d_2d_dtype_object_unique(self) -> None:
        from arraykit import isin_array

        e_1d = np.array([1, 0, 0, 1, 0], dtype=bool)
        e_2d = np.array([[1, 0, 0], [1, 0, 1]], dtype=bool)

        arr1_1d = np.array([1, 2, 3, 4, 5], dtype=object)
        arr1_2d = np.array([[1, 2, 3], [4, 5, 9]], dtype=object)

        arr2 = np.array([1, 4, 7, 9], dtype=object)

        post = isin_array(array=arr1_1d, array_is_unique=True, other=arr2, other_is_unique=True)
        self.assertTrue(np.array_equal(e_1d, post))

        post = isin_array(array=arr1_2d, array_is_unique=True, other=arr2, other_is_unique=True)
        self.assertTrue(np.array_equal(e_2d, post))

        class C:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                return self.val == other.val

            def __hash__(self):
                return hash(self.val)

        arr1 = np.array([C(1), C(2), C(3), C(4), C(5)])
        arr2 = np.array([C(1), C(4), C(9)])

        post = isin_array(array=arr1, array_is_unique=True, other=arr2, other_is_unique=True)
        self.assertTrue(np.array_equal(e_1d, post))

        arr1 = np.array([[C(1), C(2), C(3)], [C(4), C(5), C(9)]])

        post = isin_array(array=arr1, array_is_unique=True, other=arr2, other_is_unique=True)
        self.assertTrue(np.array_equal(e_2d, post))

    def test_1d_2d_dtype_object_non_unique(self) -> None:
        from arraykit import isin_array

        e_1d = np.array([1, 0, 0, 1, 0], dtype=bool)
        e_2d = np.array([[1, 0, 0], [1, 0, 1]], dtype=bool)

        arr1_1d = np.array([1, 2, 2, 4, 5], dtype=object)
        arr1_2d = np.array([[1, 2, 3], [4, 2, 9]], dtype=object)

        arr2 = np.array([1, 4, 4, 9], dtype=object)

        post = isin_array(array=arr1_1d, array_is_unique=False, other=arr2, other_is_unique=False)
        self.assertTrue(np.array_equal(e_1d, post))

        post = isin_array(array=arr1_2d, array_is_unique=False, other=arr2, other_is_unique=False)
        self.assertTrue(np.array_equal(e_2d, post))

    def test_1d_2d_dtype_non_unique(self) -> None:
        from arraykit import isin_array

        isin_array_func = partial(isin_array, array_is_unique=False, other_is_unique=False)

        e_1d = np.array([1, 0, 0, 0, 1, 0, 1], dtype=bool)
        e_2d = np.array([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=bool)

        v_1d = [1, 5, 2, 3, 4, 5, 1]
        v_2d = [[9, 1, 2, 3], [4, 3, 5, 9]]

        w_1d = [1, 4, 7, 9]

        dtype_funcs = [
                (int, int),
                (float, float),
                (str, str),
                ('datetime64[D]', lambda x: date(2020, 1, x)),
                ('timedelta64[D]', timedelta),
        ]

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([dtype_func(v) for v in v_1d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertTrue(np.array_equal(e_1d, post), msg=f'\n{dtype}\nExpected:\n{e_1d}\nActual:\n{post}')

        for dtype, dtype_func in dtype_funcs:
            arr1 = np.array([[dtype_func(x) for x in y] for y in v_2d], dtype=dtype)
            arr2 = np.array([dtype_func(v) for v in w_1d], dtype=dtype)

            post = isin_array_func(array=arr1, other=arr2)
            self.assertTrue(np.array_equal(e_2d, post), msg=f'\n{dtype}\nExpected:\n{e_2d}\nActual:\n{post}')


if __name__ == '__main__':
    unittest.main()
