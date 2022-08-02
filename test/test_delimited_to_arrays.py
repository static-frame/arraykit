import unittest
import datetime

import numpy as np

from hypothesis import strategies as st
from hypothesis import given

from arraykit import delimited_to_arrays
from arraykit import iterable_str_to_array_1d


class TestUnit(unittest.TestCase):



    # def test_iterable_str_to_array_1d_a(self) -> None:
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


    def test_iterable_str_to_array_1d_bool_1(self) -> None:
        a1 = iterable_str_to_array_1d(['true', 'false', 'TRUE', 'FALSE'], bool)
        self.assertEqual(a1.tolist(), [True, False, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_bool_2(self) -> None:
        src = ['true', 'True', 'TRUE', 't']
        a1 = iterable_str_to_array_1d(src, bool)
        self.assertEqual(a1.tolist(), [True, True, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)
        # same as genfromtxt
        self.assertEqual(a1.tolist(), np.genfromtxt(src, dtype=bool).tolist())


    def test_iterable_str_to_array_1d_bool_3(self) -> None:
        a1 = iterable_str_to_array_1d(['sd', 'er', 'TRUE', 'twerwersdfsd'], bool)
        self.assertEqual(a1.tolist(), [False, False, True, False])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_bool_4(self) -> None:
        a1 = iterable_str_to_array_1d(['false', '  true  ', '  false  ', 'true'], bool)
        self.assertEqual(a1.tolist(), [False, True, False, True])
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)



    def test_iterable_str_to_array_1d_int_1(self) -> None:
        # NOTE: floats will be truncated
        a1 = iterable_str_to_array_1d(['23', '-54', '  1000', '23  '], np.int64)
        self.assertEqual(a1.tolist(), [23, -54, 1000, 23])
        self.assertEqual(a1.dtype, np.dtype(np.int64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_2(self) -> None:
        # NOTE: empty strings get converted to zero
        a1 = iterable_str_to_array_1d(['23', '', '  -123000', '23'], np.int64)
        self.assertEqual(a1.tolist(), [23, 0, -123000, 23])
        self.assertEqual(a1.dtype, np.dtype(np.int64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_3a(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(9_223_372_036_854_775_807),
                '0',
                str(-9_223_372_036_854_775_808)], np.int64)
        self.assertEqual(a1.tolist(), [9223372036854775807, 0, -9223372036854775808])
        self.assertEqual(a1.dtype, np.dtype(np.int64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_3b(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(9_223_372_036_854_775_808),
                '0',
                str(-9_223_372_036_854_775_809)], np.int64)
        # NOTE: overflow may not be stable
        self.assertEqual(a1.tolist(), [0, 0, 0])
        self.assertEqual(a1.dtype, np.dtype(np.int64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_4(self) -> None:
        # NOTE: floats will be truncated
        a1 = iterable_str_to_array_1d(['23', '-54', '  1000', '23  '], np.int32)
        self.assertEqual(a1.tolist(), [23, -54, 1000, 23])
        self.assertEqual(a1.dtype, np.dtype(np.int32))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_5(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(2_147_483_647),
                '0',
                str(-2_147_483_648)], np.int32)
        self.assertEqual(a1.tolist(), [2147483647, 0, -2147483648])
        self.assertEqual(a1.dtype, np.dtype(np.int32))
        self.assertFalse(a1.flags.writeable)


    def test_iterable_str_to_array_1d_int_6(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(2_147_483_647_000),
                '0',
                str(-2_147_483_648_000)], np.int32)
        # NOTE: overflow characteristics may not be stable
        self.assertEqual(a1.tolist(), [-1000, 0, 0])
        self.assertEqual(a1.dtype, np.dtype(np.int32))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_7(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(32_767),
                '0',
                str(-32_768)], np.int16)
        self.assertEqual(a1.tolist(), [32767, 0, -32768])
        self.assertEqual(a1.dtype, np.dtype(np.int16))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_int_8(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(127),
                '0',
                str(-128)], np.int8)
        self.assertEqual(a1.tolist(), [127, 0, -128])
        self.assertEqual(a1.dtype, np.dtype(np.int8))
        self.assertFalse(a1.flags.writeable)



    def test_iterable_str_to_array_1d_uint_1(self) -> None:
        a1 = iterable_str_to_array_1d(['23', '54', '  1000', '23  '], np.uint64)
        self.assertEqual(a1.tolist(), [23, 54, 1000, 23])
        self.assertEqual(a1.dtype, np.dtype(np.uint64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_2a(self) -> None:
        a1 = iterable_str_to_array_1d([str(18_446_744_073_709_551_615), '0'], np.uint64)
        self.assertEqual(a1.tolist(), [18446744073709551615, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_2a(self) -> None:
        a1 = iterable_str_to_array_1d([str(18_446_744_073_709_551_616), '0'], np.uint64)
        # overflow for now simply returns 0
        self.assertEqual(a1.tolist(), [0, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_3(self) -> None:
        a1 = iterable_str_to_array_1d([
                str(18_446_744_073_709_551),
                str(18_446_744_073_709_551_6),
                str(18_446_744_073_709_551_61),
                '0'], np.uint64)
        self.assertEqual(a1.tolist(),
                [18446744073709551, 184467440737095516, 1844674407370955161, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_4(self) -> None:
        a1 = iterable_str_to_array_1d([str(4294967295), '0'], np.uint32)
        self.assertEqual(a1.tolist(), [4294967295, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint32))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_4(self) -> None:
        a1 = iterable_str_to_array_1d([str(65535), '0'], np.uint16)
        self.assertEqual(a1.tolist(), [65535, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint16))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_uint_4(self) -> None:
        a1 = iterable_str_to_array_1d([str(255), '0'], np.uint8)
        self.assertEqual(a1.tolist(), [255, 0])
        self.assertEqual(a1.dtype, np.dtype(np.uint8))
        self.assertFalse(a1.flags.writeable)



    def test_iterable_str_to_array_1d_float_1(self) -> None:
        a1 = iterable_str_to_array_1d(['23.1', '54.5', '1000.2', '23.'], float)
        self.assertEqual(a1.tolist(),[23.1, 54.5, 1000.2, 23.0])
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_2(self) -> None:
        a1 = iterable_str_to_array_1d(['23.1', '   54.5    ', '   1000.2', '23.   '], float)
        self.assertEqual(a1.tolist(),[23.1, 54.5, 1000.2, 23.0])
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_3(self) -> None:
        a1 = iterable_str_to_array_1d(['23.1', '   54', '   1000.2', '23'], float)
        self.assertEqual(a1.tolist(),[23.1, 54.0, 1000.2, 23.0])
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_4(self) -> None:
        a1 = iterable_str_to_array_1d(['23', '   54', '   1000', '23'], float)
        self.assertEqual(a1.tolist(),[23.0, 54.0, 1000.0, 23.0])
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_5(self) -> None:
        a1 = iterable_str_to_array_1d(['inf', '   nan', '   1e-200', '1.5e34'], float)
        self.assertEqual(str(a1.tolist()), '[inf, nan, 1e-200, 1.5e+34]')
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_6(self) -> None:
        if hasattr(np, 'float128'):
            a1 = iterable_str_to_array_1d(['3.2', '3', 'inf', '-inf'], np.float128)
            self.assertEqual(a1.tolist(), [3.2, 3, np.inf, -np.inf])
            self.assertEqual(a1.dtype, np.float128)
            self.assertFalse(a1.flags.writeable)




    def test_iterable_str_to_array_1d_str_1(self) -> None:
        a1 = iterable_str_to_array_1d(['    sdf  ', '  we', 'aaa', 'qqqqq '], str)
        self.assertEqual(a1.dtype.str, '<U9')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['    sdf  ', '  we', 'aaa', 'qqqqq '])

    def test_iterable_str_to_array_1d_str_2(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'cc', 'dddd '], str)
        self.assertEqual(a1.dtype.str, '<U5')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aa', 'bbb', 'cc', 'dddd '])

    def test_iterable_str_to_array_1d_str_3(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb'], str)
        self.assertEqual(a1.dtype.str, '<U3')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aa', 'bbb'])

    def test_iterable_str_to_array_1d_str_4(self) -> None:
        a1 = iterable_str_to_array_1d(['aaaaaaaaaa', 'bbb'], str)
        self.assertEqual(a1.dtype.str, '<U10')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aaaaaaaaaa', 'bbb'])



    def test_iterable_str_to_array_1d_str_5(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', ' dddd '], np.dtype('<U2'))
        self.assertEqual(a1.dtype.str, '<U2')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aa', 'bb', 'cc', ' d'])

    def test_iterable_str_to_array_1d_str_6(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', ' dddd '], np.dtype('<U4'))
        self.assertEqual(a1.dtype.str, '<U4')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aa', 'bbb', 'cccc', ' ddd'])

    def test_iterable_str_to_array_1d_str_7(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', ' dddd ', ''], np.dtype('<U8'))
        self.assertEqual(a1.dtype.str, '<U8')
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['aa', 'bbb', 'ccccc', ' dddd ', ''])



    def test_iterable_str_to_array_1d_bytes_1(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', 'dddddd', ''], np.dtype('|S3'))
        self.assertEqual(a1.dtype.str, '|S3')
        self.assertEqual(a1.tolist(), [b'aa', b'bbb', b'ccc', b'ddd', b''])
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_bytes_2(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', 'dddddd', ''], bytes)
        self.assertEqual(a1.dtype.str, '|S6')
        self.assertEqual(a1.tolist(), [b'aa', b'bbb', b'ccccc', b'dddddd', b''])
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_bytes_3(self) -> None:
        a1 = iterable_str_to_array_1d(['aa', 'bbb', 'ccccc', 'dddddd', ''], np.dtype('|S1'))
        self.assertEqual(a1.dtype.str, '|S1')
        self.assertEqual(a1.tolist(), [b'a', b'b', b'c', b'd', b''])
        self.assertFalse(a1.flags.writeable)



    def test_iterable_str_to_array_1d_complex_1(self) -> None:
        a1 = iterable_str_to_array_1d(['(3+0j)', '(100+0j)'], complex)
        self.assertEqual(a1.dtype, np.dtype(complex))
        self.assertEqual(a1.tolist(), [(3+0j), (100+0j)])

    def test_iterable_str_to_array_1d_complex_2(self) -> None:
        a1 = iterable_str_to_array_1d(['3+0j', '100+nanj'], complex)
        self.assertEqual(a1.dtype, np.dtype(complex))

    def test_iterable_str_to_array_1d_complex_3(self) -> None:
        a1 = iterable_str_to_array_1d(['-2+1.2j', '1.5+4.2j'], complex)
        self.assertEqual(a1.dtype, np.dtype(complex))
        self.assertEqual(a1.tolist(), [(-2+1.2j), (1.5+4.2j)])

    def test_iterable_str_to_array_1d_complex_4(self) -> None:
        a1 = iterable_str_to_array_1d(['(-0+infj)', '0j'], complex)
        self.assertEqual(a1.dtype, np.dtype(complex))
        self.assertEqual(a1.tolist(), [complex('-0+infj'), (0j)])

    # NOTE: this causes a seg fault
    # def test_iterable_str_to_array_1d_d4(self) -> None:
    #     with self.assertRaises(ValueError):
    #         a1 = iterable_str_to_array_1d(['-2+1.2j', '1.5+-4.2j'], complex)


    def test_iterable_str_to_array_1d_dt64_1(self) -> None:
        a1 = iterable_str_to_array_1d(['2020-01-01', '2020-02-01'], 'datetime64[D]')
        self.assertEqual(a1.dtype, np.dtype('<M8[D]'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)])

    def test_iterable_str_to_array_1d_dt64_2(self) -> None:
        a1 = iterable_str_to_array_1d(['2020-01-01', '2020-02-01'], np.datetime64)
        self.assertEqual(a1.dtype, np.dtype('<M8[D]'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)])

    def test_iterable_str_to_array_1d_dt64_2(self) -> None:
        a1 = iterable_str_to_array_1d(['2020-01-01', '2020-02-01'], np.datetime64)
        self.assertEqual(a1.dtype, np.dtype('<M8[D]'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)])


    #---------------------------------------------------------------------------



    def test_iterable_str_to_array_1d_parse_1(self) -> None:
        a1 = iterable_str_to_array_1d(['20', '30'], None)
        self.assertEqual(a1.dtype, np.dtype(np.int64))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [20, 30])

    def test_iterable_str_to_array_1d_parse_2(self) -> None:
        a1 = iterable_str_to_array_1d(['true', 'true', ''], None)
        self.assertEqual(a1.dtype, np.dtype(bool))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [True, True, False])

    def test_iterable_str_to_array_1d_parse_2(self) -> None:
        a1 = iterable_str_to_array_1d(['1.5  ', '   4.5', 'inf   '], None)
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [1.5, 4.5, np.inf])

    def test_iterable_str_to_array_1d_parse_2(self) -> None:
        a1 = iterable_str_to_array_1d(['b', 'ee', 't'], None)
        self.assertEqual(a1.dtype, np.dtype('<U2'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['b', 'ee', 't'])



    #---------------------------------------------------------------------------
    # def test_test_a(self) -> None:
    #     from arraykit import _test
    #     post = _test(
    #             [['true', 'True', 'TRUE', 'FALSE', 'fAlse', 'tRUE'],
    #             ['g', 't', 'w', 'g', 'true', 'f']]
    #             )
    #     print(post)

    #---------------------------------------------------------------------------

    def test_delimited_to_arrays_a(self) -> None:

        msg = [
            'false,true,true,false',
            'true,f,f,true',
            'True,False,TRUE,FALSE',
        ]

        dtypes0 = [bool, np.dtype(bool), bool]
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=0)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 3)
        self.assertTrue(all(len(e) == 4 for e in post0))

        dtypes1 = [bool, np.dtype(bool), bool, bool]
        post1 = delimited_to_arrays(msg, dtypes=dtypes1, axis=1)
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
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 40)
        self.assertTrue(all(len(e) == 3 for e in post0))


    def test_delimited_to_arrays_c(self) -> None:

        msg = [
            ','.join(['True', '10'] * 20),
            ','.join(['True', '-2000'] * 20),
            ','.join(['False', '82342343'] * 20),
        ]

        dtypes0 = [bool, int] * 20
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 40)
        self.assertTrue(all(len(e) == 3 for e in post0))

        self.assertEqual([a.tolist() for a in post0],
                [[True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343]])

    def test_delimited_to_arrays_d(self) -> None:

        msg = [
            '|'.join(['True', '10'] * 20),
            '|'.join(['True', '-2000'] * 20),
            '|'.join(['False', '82342343'] * 20),
        ]

        dtypes0 = [bool, int] * 20
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1, delimiter='|')
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 40)
        self.assertTrue(all(len(e) == 3 for e in post0))
        self.assertEqual([a.tolist() for a in post0],
                [[True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343], [True, True, False], [10, -2000, 82342343]])

    def test_delimited_to_arrays_e(self) -> None:

        msg = [
            ','.join(['True', '10', 'foo'] * 20),
            ','.join(['True', '-2000', 'bar'] * 20),
            ','.join(['False', '82342343', 'baz'] * 20),
        ]

        dtypes0 = [bool, np.int64, str] * 20
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 60)
        self.assertTrue(all(len(e) == 3 for e in post0))
        self.assertEqual([x.dtype.str for x in post0],
                ['|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3', '|b1', '<i8', '<U3'])

    def test_delimited_to_arrays_f(self) -> None:

        msg = [
            ','.join(['True', '10', 'foo'] * 2),
            ','.join(['True', '-2000', 'bar'] * 2),
            ','.join(['False', '82342343', 'baz'] * 2),
        ]

        dtypes0 = ['<U1'] * 6
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual([a.tolist() for a in post0],
                [['T', 'T', 'F'], ['1', '-', '8'], ['f', 'b', 'b'], ['T', 'T', 'F'], ['1', '-', '8'], ['f', 'b', 'b']])
        self.assertEqual([x.dtype.str for x in post0],
                ['<U1', '<U1', '<U1', '<U1', '<U1', '<U1'])


    def test_delimited_to_arrays_g(self) -> None:

        msg = [
            'false,100,inf,red',
            'true,200,6.5,blue',
            'True,-234,3.2e-10,green',
        ]

        dtypes0 = [bool, int, float, str]
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=1)
        self.assertEqual([a.dtype.kind for a in post0],
                ['b', 'i', 'f', 'U'])

        self.assertEqual([a.tolist() for a in post0],
                    [[False, True, True],
                    [100, 200, -234],
                    [np.inf, 6.5, 3.2e-10],
                    ['red', 'blue', 'green']])

    def test_delimited_to_arrays_h(self) -> None:
        msg = [
            0, 1,
            2, 3,
        ]
        with self.assertRaises(RuntimeError):
            _ = delimited_to_arrays(msg, axis=1)


    def test_delimited_to_arrays_i(self) -> None:
        msg = [
            b'a', b'b',
            b'c', b'd',
        ]
        with self.assertRaises(RuntimeError):
            _ = delimited_to_arrays(msg, axis=1)


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_parse_a(self) -> None:

        msg = [
            'false, 100,  inf,     red',
            'true,  200,  6.5,     blue',
            'True,  -234, 3.2e-10, green',
        ]

        post0 = delimited_to_arrays(msg, dtypes=None, axis=1)
        # import ipdb; ipdb.set_trace()
        self.assertEqual([a.dtype.kind for a in post0],
                ['b', 'i', 'f', 'U'])

        self.assertEqual([a.tolist() for a in post0],
                    [[False, True, True],
                    [100, 200, -234],
                    [np.inf, 6.5, 3.2e-10],
                    ['     red', '     blue', ' green']])

    def test_delimited_to_arrays_parse_b(self) -> None:
        msg = ['0j', '(-0+infj)']
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post], ['c'])




    #---------------------------------------------------------------------------

    @given(st.lists(st.integers(), min_size=1, max_size=10))
    def test_delimited_to_arrays_property_parse_a(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['i', 'i'])
        # NOTE: there are some areas where this fails as we overflow and return zero:
        # 9223372036854775808
        # self.assertEqual(post[0].tolist(), v)

    @given(st.lists(st.booleans(), min_size=1, max_size=10))
    def test_delimited_to_arrays_property_parse_b(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['b', 'b'])
        self.assertEqual(post[0].tolist(), v)

    @given(st.lists(st.floats(), min_size=1, max_size=10))
    def test_delimited_to_arrays_property_parse_c(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['f', 'f'])
        # need to compare without NaNs

    @given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=10))
    def test_delimited_to_arrays_property_parse_d(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['f', 'f'])
        # need to compare without NaNs
        self.assertEqual(post[0].tolist(), v)


    @given(st.lists(st.complex_numbers(), min_size=2, max_size=10))
    def test_delimited_to_arrays_property_parse_e(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['c', 'c'])



if __name__ == '__main__':
    unittest.main()
