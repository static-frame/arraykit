import typing as tp
import unittest

from automap import FrozenAutoMap
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
from arraykit import is_gen_copy_values

from performance.reference.util import mloc as mloc_ref

from arraykit import prepare_iter_for_array
#from performance.reference.util import prepare_iter_for_array


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

    def test_is_gen_copy_values(self) -> None:
        self.assertEqual((True, True), is_gen_copy_values((x for x in range(3))))

        l = [1, 2, 3]
        t = (1, 2, 3)
        fam = FrozenAutoMap((1, 2, 3))
        s = {1, 2, 3}
        d = {1:1, 2:2, 3:3}

        self.assertEqual((False, False), is_gen_copy_values(l))
        self.assertEqual((False, False), is_gen_copy_values(t))

        self.assertEqual((False, True), is_gen_copy_values(fam))
        self.assertEqual((False, True), is_gen_copy_values(d.keys()))
        self.assertEqual((False, True), is_gen_copy_values(d.values()))
        self.assertEqual((False, True), is_gen_copy_values(d))
        self.assertEqual((False, True), is_gen_copy_values(s))


class TestPrepareIterUnit(unittest.TestCase):
    def test_resolve_type_iter_a(self) -> None:
        is_object, has_tuple, values = prepare_iter_for_array(('a', 'b', 'c'))
        self.assertEqual(is_object, False)

        is_object, has_tuple, values = prepare_iter_for_array(('a', 'b', 3))
        self.assertEqual(is_object, True)

        is_object, has_tuple, values = prepare_iter_for_array(('a', 'b', (1, 2)))
        self.assertEqual(is_object, True)
        self.assertTrue(has_tuple)

        # TODO: This inexplicably fails on Python 3.6 -> why!!
        # https://github.com/InvestmentSystems/arraykit/pull/43/checks?check_run_id=2460697121#step:7:612
        #is_object, has_tuple, values = prepare_iter_for_array((1, 2, 4.3, 2))
        #self.assertEqual(is_object, False)

        is_object, has_tuple, values = prepare_iter_for_array((1, 2, 4.3, 2, None))
        self.assertEqual(is_object, False)

        is_object, has_tuple, values = prepare_iter_for_array((1, 2, 4.3, 2, 'g'))
        self.assertEqual(is_object, True)

        is_object, has_tuple, values = prepare_iter_for_array(())
        self.assertEqual(is_object, False)

    def test_resolve_type_iter_b(self) -> None:
        is_object, has_tuple, values = prepare_iter_for_array(iter(('a', 'b', 'c')))
        self.assertEqual(is_object, False)

        is_object, has_tuple, values = prepare_iter_for_array(iter(('a', 'b', 3)))
        self.assertEqual(is_object, True)

        is_object, has_tuple, values = prepare_iter_for_array(iter(('a', 'b', (1, 2))))
        self.assertEqual(is_object, True)
        self.assertTrue(has_tuple)

        is_object, has_tuple, values = prepare_iter_for_array(range(4))
        self.assertEqual(is_object, False)

    def test_resolve_type_iter_c(self) -> None:
        a = [True, False, True]
        is_object, has_tuple, values = prepare_iter_for_array(a)
        self.assertEqual(id(a), id(values))

        is_object, has_tuple, values = prepare_iter_for_array(iter(a))
        self.assertNotEqual(id(a), id(values))

        self.assertEqual(is_object, False)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_d(self) -> None:
        a = [3, 2, (3,4)]
        is_object, has_tuple, values = prepare_iter_for_array(a)
        self.assertEqual(id(a), id(values))
        self.assertTrue(has_tuple)

        is_object, has_tuple, values = prepare_iter_for_array(iter(a))
        self.assertNotEqual(id(a), id(values))

        self.assertEqual(is_object, True)
        self.assertEqual(has_tuple, True)

    def test_resolve_type_iter_e(self) -> None:
        a = [300000000000000002, 5000000000000000001]
        is_object, has_tuple, values = prepare_iter_for_array(a)
        self.assertEqual(id(a), id(values))

        is_object, has_tuple, values = prepare_iter_for_array(iter(a))
        self.assertNotEqual(id(a), id(values))
        self.assertEqual(is_object, False)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_f(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            for i in range(3):
                yield i
            yield None

        is_object, has_tuple, values = prepare_iter_for_array(a())
        self.assertEqual(values, [0, 1, 2, None])
        self.assertEqual(is_object, False)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_g(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            yield None
            for i in range(3):
                yield i

        is_object, has_tuple, values = prepare_iter_for_array(a())
        self.assertEqual(values, [None, 0, 1, 2])
        self.assertEqual(is_object, False)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_h(self) -> None:

        def a() -> tp.Iterator[tp.Any]:
            yield 10
            yield None
            for i in range(3):
                yield i
            yield (3,4)

        is_object, has_tuple, values = prepare_iter_for_array(a())
        self.assertEqual(values, [10, None, 0, 1, 2, (3,4)])
        self.assertEqual(is_object, True)
        # we stop evaluation after finding object
        self.assertEqual(has_tuple, True)

    def test_resolve_type_iter_i(self) -> None:
        a0 = range(3, 7)
        is_object, has_tuple, values = prepare_iter_for_array(a0)
        # a copy is not made
        self.assertEqual(id(a0), id(values))
        self.assertEqual(is_object, False)

    def test_resolve_type_iter_j(self) -> None:
        # this case was found through hypothesis
        a0 = [0.0, 36_028_797_018_963_969]
        is_object, has_tuple, values = prepare_iter_for_array(a0)
        self.assertEqual(is_object, True)

    def test_resolve_type_iter_k(self) -> None:
        is_object, has_tuple, values = prepare_iter_for_array((x for x in ())) #type: ignore
        self.assertEqual(is_object, False)
        self.assertEqual(len(values), 0)
        self.assertEqual(has_tuple, False)

    def test_resolve_type_iter_l(self) -> None:
        self.assertEqual((False, False, []), prepare_iter_for_array([], True))
        self.assertEqual((False, False, ()), prepare_iter_for_array((), True))
        self.assertEqual((False, False, {}), prepare_iter_for_array({}, True))
        self.assertEqual((False, False, set()), prepare_iter_for_array(set(), True))


if __name__ == '__main__':
    unittest.main()
