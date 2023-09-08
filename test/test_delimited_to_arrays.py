import unittest
import datetime
import csv
import numpy as np

from arraykit import delimited_to_arrays
from arraykit import iterable_str_to_array_1d


class TestUnit(unittest.TestCase):

    def test_iterable_str_to_array_1d_a(self) -> None:
        a1 = iterable_str_to_array_1d(['1', '3', '4'], int)
        self.assertEqual(a1.tolist(), [1, 3, 4])
        self.assertEqual(a1.dtype, np.dtype(int))

        a2 = iterable_str_to_array_1d(['1', '30', '4'], str)
        self.assertEqual(a2.tolist(), ['1', '30', '4'])
        self.assertEqual(a2.dtype, np.dtype('<U2'))

        with self.assertRaises(NotImplementedError):
            a3 = iterable_str_to_array_1d(['1', '3', '4'], object)

    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

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
        with self.assertRaises(TypeError):
            _ = iterable_str_to_array_1d([
                    str(9_223_372_036_854_775_808),
                    '0',
                    str(-9_223_372_036_854_775_809)], np.int64)

        # self.assertEqual(a1.tolist(), [0, 0, 0])
        # self.assertEqual(a1.dtype, np.dtype(np.int64))
        # self.assertFalse(a1.flags.writeable)

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

    def test_iterable_str_to_array_1d_int_9(self) -> None:
        with self.assertRaises(TypeError):
            _ = iterable_str_to_array_1d(['3', '4', 'foo'], int)

    def test_iterable_str_to_array_1d_int_10(self) -> None:
        a1 = iterable_str_to_array_1d(['3', '4', '1']) # no dtype argument
        self.assertEqual(a1.tolist(), [3, 4, 1])

    def test_iterable_str_to_array_1d_int_11(self) -> None:
        a1 = iterable_str_to_array_1d(['3,000', '4,000', '1,000'],
                dtype=int,
                thousandschar=',',
                )
        self.assertEqual(a1.tolist(), [3000, 4000, 1000])

    def test_iterable_str_to_array_1d_int_12(self) -> None:
        a1 = iterable_str_to_array_1d(['3.000', '4.000', '1.000'], dtype=int, thousandschar='.')
        self.assertEqual(a1.tolist(), [3000, 4000, 1000])

    def test_iterable_str_to_array_1d_int_13(self) -> None:
        # TypeError: error parsing integer
        with self.assertRaises(TypeError):
            a1 = iterable_str_to_array_1d(['3.000', '4.000', '1.000'], dtype=int, thousandschar=',')


    #---------------------------------------------------------------------------

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

    def test_iterable_str_to_array_1d_uint_2b(self) -> None:
        with self.assertRaises(TypeError):
            _ = iterable_str_to_array_1d([str(18_446_744_073_709_551_616), '0'], np.uint64)

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

    def test_iterable_str_to_array_1d_uint_5(self) -> None:
        a1 = iterable_str_to_array_1d(['3,000', '4,000', '1,000'],
                dtype=np.uint64,
                thousandschar=',')
        self.assertEqual(a1.tolist(), [3000, 4000, 1000])

    def test_iterable_str_to_array_1d_uint_6(self) -> None:
        a1 = iterable_str_to_array_1d(['3.000', '4.000', '1.000'], dtype=np.uint64, thousandschar='.')
        self.assertEqual(a1.tolist(), [3000, 4000, 1000])

    def test_iterable_str_to_array_1d_uint_7(self) -> None:
        # TypeError: error parsing integer
        with self.assertRaises(TypeError):
            a1 = iterable_str_to_array_1d(['3.000', '4.000', '1.000'], dtype=np.uint64, thousandschar=',')

    #---------------------------------------------------------------------------

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
        self.assertEqual(str(a1[:3].tolist()), '[inf, nan, 1e-200]')
        self.assertTrue((a1[3] - 1.5e34) < 1e-18) # noise!

        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_6(self) -> None:
        if hasattr(np, 'float128'):
            a1 = iterable_str_to_array_1d(['3.2', '3', 'inf', '-inf'], np.float128)
            self.assertEqual(a1.tolist(), [3.2, 3, np.inf, -np.inf])
            self.assertEqual(a1.dtype, np.float128)
            self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_7(self) -> None:
        a1 = iterable_str_to_array_1d(['+inf', '-inf'], float)
        self.assertEqual(str(a1.tolist()), '[inf, -inf]')

    def test_iterable_str_to_array_1d_float_8(self) -> None:
        a1 = iterable_str_to_array_1d(['+nan', '-nan'], float)
        # negative is not ignored
        self.assertEqual(str(a1.tolist()), '[nan, nan]')

    def test_iterable_str_to_array_1d_float_9(self) -> None:
        with self.assertRaises(TypeError):
            a1 = iterable_str_to_array_1d(['+na', '-in'], float)

    def test_iterable_str_to_array_1d_float_10(self) -> None:
        a1 = iterable_str_to_array_1d(['23,1', '1000,2'], dtype=float, decimalchar=',')
        self.assertEqual(a1.tolist(),[23.1, 1000.2])
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)

    def test_iterable_str_to_array_1d_float_11(self) -> None:
        with self.assertRaises(TypeError):
            a1 = iterable_str_to_array_1d(['23.1', '1000.2'], dtype=float, decimalchar=',')
            self.assertEqual(a1.tolist(),[23.1, 1000.2])

    def test_iterable_str_to_array_1d_float_12(self) -> None:
        a1 = iterable_str_to_array_1d(['23.1', '54.5', '1000.2', '23.'], np.float16)
        self.assertEqual(a1.tolist(),[23.09375, 54.5, 1000.0, 23.0])
        self.assertEqual(a1.dtype, np.dtype(np.float16))
        self.assertFalse(a1.flags.writeable)


    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------

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

    def test_iterable_str_to_array_1d_complex_5(self) -> None:
        with self.assertRaises(ValueError):
            a1 = iterable_str_to_array_1d(['-2+1.2j', '1.5+-4.2j'], complex)

    def test_iterable_str_to_array_1d_complex_6(self) -> None:
        # NOTE: malformed complex raise Exception as expected
        with self.assertRaises(ValueError):
            a1 = iterable_str_to_array_1d(['-2+1.2asdfj', '1.5wer4.2j'], complex)

    #---------------------------------------------------------------------------

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

    def test_iterable_str_to_array_1d_dt64_3(self) -> None:
        a1 = iterable_str_to_array_1d(['2020-01-01', '2020-02-01'], np.datetime64)
        self.assertEqual(a1.dtype, np.dtype('<M8[D]'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [datetime.date(2020, 1, 1), datetime.date(2020, 2, 1)])

    def test_iterable_str_to_array_1d_dt64_4(self) -> None:
        with self.assertRaises(ValueError):
            _ = iterable_str_to_array_1d(['202.30', '202.20'], 'datetime64[D]')

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

    def test_iterable_str_to_array_1d_parse_3(self) -> None:
        a1 = iterable_str_to_array_1d(['1.5  ', '   4.5', 'inf   '], None)
        self.assertEqual(a1.dtype, np.dtype(np.float64))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), [1.5, 4.5, np.inf])

    def test_iterable_str_to_array_1d_parse_4(self) -> None:
        a1 = iterable_str_to_array_1d(['b', 'ee', 't'], None)
        self.assertEqual(a1.dtype, np.dtype('<U2'))
        self.assertFalse(a1.flags.writeable)
        self.assertEqual(a1.tolist(), ['b', 'ee', 't'])


    #---------------------------------------------------------------------------
    def test_iterable_str_to_array_1d_raise_a(self) -> None:
        with self.assertRaises(TypeError):
            a1 = iterable_str_to_array_1d([3, 4, 5], None)

    #---------------------------------------------------------------------------


    def test_iterable_str_to_array_1d_empty_a(self) -> None:
        # an empty string is an nan float
        post1 = iterable_str_to_array_1d(['', '', '3.4'], float)
        self.assertEqual(post1.dtype, np.dtype(float))

        post2 = iterable_str_to_array_1d(['', '', ''], None)
        self.assertEqual(post2.tolist(), ['', '', ''])

# +    def test_iterable_str_to_array_1d_empty_b(self) -> None:
# +        # with self.assertRaises(ValueError):
# +            # an empty string is an invalid identifier for
# +        with self.assertRaises(TypeError):
# +            _ = iterable_str_to_array_1d(['', '', '3'], int)
# +
    #---------------------------------------------------------------------------

    def test_delimited_to_arrays_a(self) -> None:

        msg = [
            'false,true,true,false',
            'true,f,f,true',
            'True,False,TRUE,FALSE',
        ]

        dtypes0 = [bool, np.dtype(bool), bool].__getitem__
        post0 = delimited_to_arrays(msg, dtypes=dtypes0, axis=0)
        self.assertTrue(isinstance(post0, list))
        self.assertEqual(len(post0), 3)
        self.assertTrue(all(len(e) == 4 for e in post0))

        dtypes1 = [bool, np.dtype(bool), bool, bool].__getitem__
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

        dtypes0 = ([bool] * 40).__getitem__
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

        dtypes0 = ([bool, int] * 20).__getitem__
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

        dtypes0 = ([bool, int] * 20).__getitem__
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

        dtypes0 = ([bool, np.int64, str] * 20).__getitem__
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

        dtypes0 = (['<U1'] * 6).__getitem__
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

        dtypes0 = ([bool, int, float, str]).__getitem__
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

    def test_delimited_to_arrays_parse_c(self) -> None:

        msg = [
            'false, 10,  inf',
            'true,  20,  6.5',
            'True,  -24, 3.2e-10',
            ]
        dtypes = [None, np.int16, None].__getitem__
        post1 = delimited_to_arrays(msg, dtypes=dtypes, axis=1)
        self.assertEqual([a.dtype.str for a  in post1], ['|b1', '<i2', '<f8'])

    def test_delimited_to_arrays_parse_d(self) -> None:

        msg = [
            'false, 10,  inf',
            'true,  20,  6.5',
            ]
        dtypes = [None].__getitem__
        with self.assertRaises(RuntimeError):
            post1 = delimited_to_arrays(msg, dtypes=dtypes, axis=1)

    def test_delimited_to_arrays_parse_e1(self) -> None:

        msg = [
            'a, "10",  "foo"',
            'b,  "20",  "bar',
            ]
        with self.assertRaises(NotImplementedError):
            post1 = delimited_to_arrays(msg,
                    dtypes=[str, int, 'V'].__getitem__,
                    axis=1,
                    quoting=csv.QUOTE_ALL,
                    skipinitialspace=True,
                    )

    def test_delimited_to_arrays_parse_e2(self) -> None:

        msg = [
            'a, "10",  "foo"',
            'b,  "20",  "bar',
            ]
        post1 = delimited_to_arrays(msg,
                dtypes=[str, int, str].__getitem__,
                axis=1,
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True,
                )
        self.assertEqual([x.tolist() for x in post1],
            [['a', 'b'], [10, 20], ['foo', 'bar']])


    def test_delimited_to_arrays_parse_f(self) -> None:

        msg = [
            'a, 10, foo',
            'b,  20, bar',
            ]
        dtypes = [str, int].__getitem__
        # dtypes fails for argument 2
        with self.assertRaises(RuntimeError):
            _ = delimited_to_arrays(msg, axis=1, dtypes=dtypes)


    def test_delimited_to_arrays_parse_g(self) -> None:
        msg = [
            'a,10,foo',
            'b,20,\0',
            ]
        # if a null character is encountered it used to raise; this seemed unnecessary
        post = delimited_to_arrays(msg, axis=1)
        self.assertEqual( [a.tolist() for a in post],
                [['a', 'b'], [10, 20], ['foo', '']]
                )

    def test_delimited_to_arrays_parse_h(self) -> None:
        msg = [',0', 'False,1']
        post1 = delimited_to_arrays(msg, axis=1)
        # without specifying dtypes we get bool and int
        self.assertEqual([a.tolist() for a in post1], [[False, False], [0, 1]])

        post2 = delimited_to_arrays(msg, axis=1, dtypes=[str, None].__getitem__)
        self.assertEqual([a.tolist() for a in post2], [['', 'False'], [0, 1]])


    def test_delimited_to_arrays_parse_i(self) -> None:
        msg = [
            'a, 10, foo',
            'b,  20,   c',
            ]

        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=False)
        self.assertEqual([a.tolist() for a in post1], [['a', 'b'], [10, 20], [' foo', '   c']])

        post2 = delimited_to_arrays(msg, axis=1, skipinitialspace=True)
        self.assertEqual([a.tolist() for a in post2], [['a', 'b'], [10, 20], ['foo', 'c']])

    def test_delimited_to_arrays_parse_j(self) -> None:
        msg = [
            '2021,2021-04-01,4',
            '2022,2022-05-01,3',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=False)
        self.assertEqual([a.tolist() for a in post1], [[2021, 2022], ['2021-04-01', '2022-05-01'], [4, 3]])


    def test_delimited_to_arrays_parse_k(self) -> None:
        msg = [
            '2021,2021-04,4',
            '2022,2022-05,3',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=False)
        self.assertEqual([a.tolist() for a in post1], [[2021, 2022], ['2021-04', '2022-05'], [4, 3]])


    def test_delimited_to_arrays_parse_l(self) -> None:
        msg = [
            '1,2,3',
            '2-,2-0,-3',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=False)
        self.assertEqual([a.tolist() for a in post1], [['1', '2-'], ['2', '2-0'], [3, -3]])

    def test_delimited_to_arrays_parse_m(self) -> None:
        msg = [
            '  1,   2,3',
            ' 2-, 2-0, -3',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=False)
        self.assertEqual([a.tolist() for a in post1], [['  1', ' 2-'], ['   2', ' 2-0'], [3, -3]])

    def test_delimited_to_arrays_parse_n(self) -> None:
        msg = ['1,2,1', '3,4,5']
        post1 = delimited_to_arrays(msg, axis=1)
        # NOTE: automatic type parsing should always give an int64
        self.assertEqual([str(a.dtype) for a in post1],
                ['int64', 'int64', 'int64'])

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_float_a(self) -> None:
        msg = [
            '1.2, 5.4, 9.2',
            ' 3.5 ,  2.3  ,  6.3   ',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=True)
        self.assertEqual([a.round(1).tolist() for a in post1],
                [[1.2, 3.5], [5.4, 2.3], [9.2, 6.3]]
                )

    def test_delimited_to_arrays_float_b(self) -> None:
        msg = [
            '1.2, inf  , 9.2',
            ' 3.5 ,  2.3  ,  -inf   ',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=True)
        self.assertEqual([a.round(1).tolist() for a in post1],
                [[1.2, 3.5], [np.inf, 2.3], [9.2, -np.inf]]
                )

    def test_delimited_to_arrays_float_c(self) -> None:
        msg = [
            '1.2, nan  , 9.2',
            ' 3.5 ,  2.3  ,  -nan   ',
            ]
        post1 = delimited_to_arrays(msg, axis=1, skipinitialspace=True)
        self.assertEqual(
                [str(a.round(1).tolist()) for a in post1],
                ['[1.2, 3.5]', '[nan, 2.3]', '[9.2, nan]']
                )

    def test_delimited_to_arrays_float_d(self) -> None:
        msg = [
            '1,2;9,2',
            '3,5;2,3',
            ]
        post1 = delimited_to_arrays(msg, axis=1, decimalchar=',', delimiter=';')
        self.assertEqual(
                [str(a.round(1).tolist()) for a in post1],
                ['[1.2, 3.5]', '[9.2, 2.3]']
                )

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_int_a(self) -> None:
        msg = [
            '12, 54, "9,200"',
            ' 35 ,  23  ,  "6,300"   ',
            ]
        dtypes = lambda x: int
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, skipinitialspace=True, dtypes=dtypes, quoting=csv.QUOTE_MINIMAL)
            # this fails until we accept the thousands character

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_quoting_a(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        post1 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_MINIMAL)
        self.assertEqual([a.tolist() for a in post1], [['a', 'b'], [3, -1], [True, False]])

        post2 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL)
        self.assertEqual([a.tolist() for a in post2], [['a', 'b'], [3, -1], [True, False]])

        post3 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_NONE)
        self.assertEqual([a.tolist() for a in post3], [['a', 'b'], [3, -1], [True, False]])

        # this is supported but has no effect
        post4 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_NONNUMERIC)
        self.assertEqual([a.tolist() for a in post4], [['a', 'b'], [3, -1], [True, False]])

    def test_delimited_to_arrays_quoting_b(self) -> None:
        msg  = ['"fo,o",3,True', '"ba,r",-1,False']
        post1 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_MINIMAL)
        self.assertEqual([a.tolist() for a in post1], [['fo,o', 'ba,r'], [3, -1], [True, False]])

        post2 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL)
        self.assertEqual([a.tolist() for a in post2], [['fo,o', 'ba,r'], [3, -1], [True, False]])


    def test_delimited_to_arrays_quoting_c(self) -> None:
        msg  = ['"fo,o",3,True', '"ba,r",-1,False']
        post1 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_NONE)
        # NOTE: with quoting disabled, we observe the comma as a delimiter
        self.assertEqual([a.tolist() for a in post1], [['"fo', '"ba'], ['o"', 'r"'], [3, -1], [True, False]])

    def test_delimited_to_arrays_quoting_d(self) -> None:
        msg  = ['"foo","3","True"', '"bar","-1","False"']
        # with QUOTE_NONE, all remain string types
        post1 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_NONE)
        self.assertEqual([a.tolist() for a in post1], [['"foo"', '"bar"'], ['"3"', '"-1"'], ['"True"', '"False"']])

        # with QUOTE_ALL, quotes are stripped, types are evaluated
        post2 = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL)
        self.assertEqual([a.tolist() for a in post2], [['foo', 'bar'], [3, -1], [True, False]])

        # import ipdb; ipdb.set_trace()

    def test_delimited_to_arrays_quoting_e(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, quoting=20)
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, quoting="foo")


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_delimiter_a(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, delimiter='foo')

    def test_delimited_to_arrays_delimiter_b(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, delimiter=None)


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_escapechar_a(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, escapechar='foo')

    def test_delimited_to_arrays_escapechar_b(self) -> None:
        msg  = ['f/"oo,3,True', 'b/,ar,-1,False']
        post1 = delimited_to_arrays(msg,
                axis=1,
                escapechar='/',
                quoting=csv.QUOTE_NONE,
                doublequote=False,
                )

        self.assertEqual([a.tolist() for a in post1],
                [['f"oo', 'b,ar'], [3, -1], [True, False]])

    def test_delimited_to_arrays_escapechar_c(self) -> None:
        msg  = ['foo,3,True', 'bar,-1,False']
        post1 = delimited_to_arrays(msg,
                axis=1,
                escapechar=None,
                )
        self.assertEqual([a.tolist() for a in post1],
                [['foo', 'bar'], [3, -1], [True, False]])

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_quotechar_a(self) -> None:
        msg  = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL, quotechar='')

        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL, quotechar='foo')

        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_ALL, quotechar=None)

        _ = delimited_to_arrays(msg, axis=1, quoting=csv.QUOTE_NONE, quotechar=None)


    def test_delimited_to_arrays_quotechar_b(self) -> None:
        msg  = ['|foo|,|3|,|True|', '|bar|,|-1|,|False|']
        post1 = delimited_to_arrays(msg, axis=1)
        self.assertEqual([a.dtype.kind for a in post1], ['U', 'U', 'U'])

    def test_delimited_to_arrays_quotechar_c(self) -> None:
        msg  = ['|a|,|3|,|True|', '|b|,|-1|,|False|']
        post1 = delimited_to_arrays(msg, axis=1, quotechar='|')
        self.assertEqual([a.dtype.kind for a in post1], ['U', 'i', 'b'])
        self.assertEqual([a.tolist() for a in post1], [['a', 'b'], [3, -1], [True, False]])

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_doublequote_a(self) -> None:
        msg  = ['"f""oo",3,True', '"b""ar",-1,False']
        post1 = delimited_to_arrays(msg, axis=1, doublequote=True, quoting=csv.QUOTE_ALL)
        self.assertEqual([a.tolist() for a in post1],
            [['f"oo', 'b"ar'], [3, -1], [True, False]])

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_strict_a(self) -> None:
        msg = ['"f"oo",3,True', '"b"ar",-1,False']
        # will fail because , expected after a quite
        with self.assertRaises(RuntimeError):
            _ = delimited_to_arrays(msg, axis=1, strict=True)

        # with stract False we drop only two quotes and keep the rest
        post1 = delimited_to_arrays(msg, axis=1, strict=False)
        self.assertEqual([a.tolist() for a in post1],
            [['foo"', 'bar"'], [3, -1], [True, False]])

    def test_delimited_to_arrays_strict_b(self) -> None:
        msg = ['a,3,True', 'b,-1,False', '', '']

        # empty lines are ignored regardless of strict
        post1 = delimited_to_arrays(msg, axis=1, strict=True)
        self.assertEqual([len(a) for a in post1], [2, 2, 2])

        post2 = delimited_to_arrays(msg, axis=1, strict=False)
        self.assertEqual([len(a) for a in post2], [2, 2, 2])

    def test_delimited_to_arrays_strict_c(self) -> None:
        msg = ['a,3,True', 'b,-1,False', 'c,']
        # strict does not care about different lengths
        post1 = delimited_to_arrays(msg, axis=1, strict=True)
        self.assertEqual([len(a) for a in post1], [3, 3, 2])
        # NOTE: the empty string is being converted to 0... not sure that is correct
        self.assertEqual(post1[1].tolist(), [3, -1, 0])


    def test_delimited_to_arrays_strict_d(self) -> None:
        msg = ['a,3,True', 'b,-1,False,,', 'c,']
        post1 = delimited_to_arrays(msg, axis=1, strict=True)
        self.assertEqual([len(a) for a in post1], [3, 3, 2, 1, 1])


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_axis_a(self) -> None:
        msg = ['a,3,True', 'b,-1,False']
        with self.assertRaises(ValueError):
            _ = delimited_to_arrays(msg, axis=-1)

        with self.assertRaises(ValueError):
            _ = delimited_to_arrays(msg, axis=2)

        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, axis=None)


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_line_select_a(self) -> None:
        msg = ['a,3,True', 'b,-1,False']
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg, line_select=-1)

    def test_delimited_to_arrays_line_select_b(self) -> None:
        msg = ['1,2', 'False,True', 'foo,bar', '3.2,5.2']
        line_select = lambda i: i in (0, 2)
        post1 = delimited_to_arrays(msg, line_select=line_select, axis=0)
        self.assertEqual([x.tolist() for x in post1], [[1, 2], ['foo', 'bar']])

    def test_delimited_to_arrays_line_select_c(self) -> None:
        msg = ['1,2', 'False,True', 'foo,bar', '3.2,5.2']
        line_select = lambda i: i == 1
        post1 = delimited_to_arrays(msg, line_select=line_select, axis=0)
        self.assertEqual([x.tolist() for x in post1], [[False, True]])

    def test_delimited_to_arrays_line_select_d(self) -> None:
        msg  = ['a,3,True,c', 'b,-1,False,d']
        post1 = delimited_to_arrays(msg, axis=1, line_select=lambda i: i in (0, 2))
        self.assertEqual([x.tolist() for x in post1], [['a', 'b'], [True, False]])

    def test_delimited_to_arrays_line_select_e(self) -> None:
        msg  = ['a,3,True,c', 'b,-1,False,d']
        post1 = delimited_to_arrays(msg, axis=1, line_select=lambda i: i == 3)
        self.assertEqual([x.tolist() for x in post1], [['c', 'd']])

    def test_delimited_to_arrays_line_select_f(self) -> None:
        msg  = ['a,3,True,c', 'b,-1,False,d']
        post1 = delimited_to_arrays(msg, axis=1, line_select=lambda i: False)
        self.assertEqual([x.tolist() for x in post1], [])

    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_thousandschar_a(self) -> None:
        msg = [
            '1.000;4',
            '2.000;5.000',
            '4.000;6.000',
        ]
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg,
                    axis=1,
                    dtypes=lambda i: int,
                    delimiter=';',
                    thousandschar='foo',
                    )

    def test_delimited_to_arrays_thousandschar_b(self) -> None:

        msg = [
            '1.000;4',
            '2.000;5.000',
            '4.000;6.000.000',
        ]
        post1 = delimited_to_arrays(msg,
                axis=1,
                dtypes=lambda i: int,
                delimiter=';',
                thousandschar='.',
                )
        self.assertEqual([x.tolist() for x in post1],
            [[1000, 2000, 4000], [4, 5000, 6000000]])


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_decimalchar_a(self) -> None:
        msg = [
            '1.000;4',
            '2.000;5.000',
            '4.000;6.000',
        ]
        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(msg,
                    axis=1,
                    dtypes=lambda i: int,
                    delimiter=';',
                    decimalchar='foo',
                    )

    def test_delimited_to_arrays_decimalchar_b(self) -> None:

        msg = [
            '1.000;4',
            '2.000;5,055',
            '4.000;6.000,155',
        ]
        post1 = delimited_to_arrays(msg,
                axis=1,
                dtypes=(int, float).__getitem__,
                delimiter=';',
                decimalchar=',',
                thousandschar='.',
                )
        self.assertEqual([x.tolist() for x in post1],
            [[1000, 2000, 4000], [4.0, 5.055, 6000.155]])


    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_file_like_a(self) -> None:
        def records():
            msg = [
                '1000;4',
                '2000;5055',
            ]
            yield from msg

        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(records,
                    axis=1,
                    delimiter=';',
                    )

    def test_delimited_to_arrays_file_like_b(self) -> None:

        with self.assertRaises(TypeError):
            _ = delimited_to_arrays(3,
                    axis=1,
                    delimiter=';',
                    dtypes=lambda x: int,
                    )



    #---------------------------------------------------------------------------
    def test_delimited_to_arrays_compare_int_a(self) -> None:
        # genfromtxt might translate an empty field to -1 or 0

        # >>> np.genfromtxt(['1,2', '3,'], dtype=None, delimiter=',')
        # array([[ 1,  2],
        #        [ 3, -1]])
        # >>> np.genfromtxt(['1,', '3,'], dtype=None, delimiter=',')
        # array([[1, 0],
        #        [3, 0]])

        post1 = delimited_to_arrays(('1,2', '3,'), axis=1)
        self.assertEqual([a.tolist() for a in post1], [[1, 3], [2, 0]])

        post3 = delimited_to_arrays(('1,', '3,'), axis=1)
        self.assertEqual([a.tolist() for a in post3], [[1, 3], ['', '']])

    def test_delimited_to_arrays_compare_float_a(self) -> None:
        # genfromtxt might translate an empty field to nan or zero
        # >>> np.genfromtxt(['1.8,2.7', '3.1,'], dtype=None, delimiter=',')
        # array([[1.8, 2.7],
        #     [3.1, nan]])
        # >>> np.genfromtxt(['1.8,', '3.1,'], dtype=None, delimiter=',')
        # array([[1.8, 0. ],
        #     [3.1, 0. ]])

        post1 = delimited_to_arrays(('1.8,2.7', '3.1,'), axis=1)
        self.assertEqual(post1[0].tolist(), [1.8, 3.1])
        self.assertEqual(post1[1][0], 2.7)
        self.assertTrue(np.isnan(post1[1][1]))

        post3 = delimited_to_arrays(('1.8,', '3.1,'), axis=1)
        self.assertEqual([a.tolist() for a in post3], [[1.8, 3.1], ['', '']])



if __name__ == '__main__':
    unittest.main()
