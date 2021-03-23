import unittest
import datetime
import numpy as np  # type: ignore

from arraykit import resolve_dtype
from arraykit import resolve_dtype_iter
from arraykit import shape_filter
from arraykit import column_2d_filter
from arraykit import column_1d_filter
from arraykit import row_1d_filter
from arraykit import mloc
from arraykit import immutable_filter
from arraykit import delimited_to_arrays
from arraykit import iterable_str_to_array_1d

from performance.reference.util import mloc as mloc_ref


class TestUnit(unittest.TestCase):



    # def test_sequence_str_to_array_1d_a(self) -> None:
    #     a1 = iterable_str_to_array_1d(['1', '3', '4'], int)
    #     self.assertEqual(a1.tolist(), [1, 3, 4])
    #     self.assertEqual(a1.dtype, np.dtype(int))

    #     a2 = iterable_str_to_array_1d(['1', '30', '4'], str)
    #     self.assertEqual(a2.tolist(), ['1', '30', '4'])
    #     self.assertEqual(a2.dtype, np.dtype('<U2'))

    #     # with dtype_discover set to True, this should return integers in an object array
    #     a3 = iterable_str_to_array_1d(['1', '3', '4'], object)
    #     self.assertEqual(a3.tolist(), ['1', '3', '4'])
    #     self.assertEqual(a3.dtype, np.dtype('O'))


    def test_sequence_str_to_array_1d_b1(self) -> None:
        a1 = iterable_str_to_array_1d(['true', 'false', 'TRUE', 'FALSE'], bool)
        self.assertEqual(a1.tolist(), [True, False, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)


    def test_sequence_str_to_array_1d_b2(self) -> None:
        a1 = iterable_str_to_array_1d(['true', 'True', 'TRUE', 't'], bool)
        self.assertEqual(a1.tolist(), [True, True, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)

    def test_sequence_str_to_array_1d_b3(self) -> None:
        a1 = iterable_str_to_array_1d(['sd', 'er', 'TRUE', 'twerwersdfsd'], bool)
        self.assertEqual(a1.tolist(), [False, False, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)


    # def test_sequence_str_to_array_1d_c(self) -> None:
    #     with self.assertRaises(ValueError):
    #         _ = iterable_str_to_array_1d(['3.2', 'fo', 'nan', 'inf', 'NaN'], float)

    #     a1 = iterable_str_to_array_1d(['3.2', 'nan', 'inf', 'NaN'], float)
    #     self.assertEqual(str(a1.tolist()), '[3.2, nan, inf, nan]')
    #     self.assertEqual(a1.dtype, np.dtype(float))


    # def test_sequence_str_to_array_1d_d1(self) -> None:
    #     a1 = iterable_str_to_array_1d(['(3+0j)', '(100+0j)'], complex)
    #     self.assertEqual(a1.dtype, np.dtype(complex))
    #     self.assertEqual(a1.tolist(), [(3+0j), (100+0j)])

    # def test_sequence_str_to_array_1d_d2(self) -> None:
    #     a1 = iterable_str_to_array_1d(['3+0j', '100+nanj'], complex)
    #     self.assertEqual(a1.dtype, np.dtype(complex))

    # def test_sequence_str_to_array_1d_d3(self) -> None:
    #     a1 = iterable_str_to_array_1d(['-2+1.2j', '1.5+4.2j'], complex)
    #     self.assertEqual(a1.dtype, np.dtype(complex))
    #     self.assertEqual(a1.tolist(), [(-2+1.2j), (1.5+4.2j)])

    # def test_sequence_str_to_array_1d_d4(self) -> None:
    #     with self.assertRaises(ValueError):
    #         a1 = iterable_str_to_array_1d(['-2+1.2j', '1.5+-4.2j'], complex)


    # def test_sequence_str_to_array_1d_e(self) -> None:

    #     a1 = iterable_str_to_array_1d(['2020-01-01', '2020-02-01'], np.datetime64)
    #     self.assertEqual(a1.dtype, np.dtype('<M8[D]'))
    #     self.assertEqual(a1.tolist(), [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)])
    #     # import ipdb; ipdb.set_trace()

    #---------------------------------------------------------------------------
    def test_test_a(self) -> None:
        from arraykit import _test
        post = _test(
                [['true', 'True', 'TRUE', 'FALSE', 'fAlse', 'tRUE'],
                ['g', 't', 'w', 'g', 'true', 'f']]
                )
        print(post)

    #---------------------------------------------------------------------------

    def test_delimited_to_arrays_a(self) -> None:

        msg = [
            'false,true,true,false',
            'true,f,f,true',
            'True,False,TRUE,FALSE',
        ]

        dtypes0 = [bool, np.dtype(bool), bool]
        post0 = delimited_to_arrays(msg, dtypes0, 0)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 3)
        self.assertTrue(all(len(e) == 4 for e in post0))

        dtypes1 = [bool, np.dtype(bool), bool, bool]
        post1 = delimited_to_arrays(msg, dtypes1, 1)
        self.assertTrue(isinstance(post1, list))
        self.assertEqual(len(post1), 4)
        self.assertTrue(all(len(e) == 3 for e in post1))



    def test_delimited_to_arrays_b(self) -> None:

        msg = [
            ','.join(['True', 'False'] * 20),
            ','.join(['True', 'True'] * 20),
            ','.join(['False', 'False'] * 20),
        ]

        dtypes0 = [bool] * 40
        post0 = delimited_to_arrays(msg, dtypes0, 1)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 40)
        self.assertTrue(all(len(e) == 3 for e in post0))



    #---------------------------------------------------------------------------

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

if __name__ == '__main__':
    unittest.main()



