import collections
import datetime
import timeit
import argparse
import io

import numpy as np

from performance.reference.util import mloc as mloc_ref
from performance.reference.util import immutable_filter as immutable_filter_ref
from performance.reference.util import name_filter as name_filter_ref
from performance.reference.util import shape_filter as shape_filter_ref
from performance.reference.util import column_2d_filter as column_2d_filter_ref
from performance.reference.util import column_1d_filter as column_1d_filter_ref
from performance.reference.util import row_1d_filter as row_1d_filter_ref
from performance.reference.util import resolve_dtype as resolve_dtype_ref
from performance.reference.util import resolve_dtype_iter as resolve_dtype_iter_ref
from performance.reference.util import dtype_from_element as dtype_from_element_ref
from performance.reference.util import array_deepcopy as array_deepcopy_ref
from performance.reference.util import isna_element as isna_element_ref

from performance.reference.array_go import ArrayGO as ArrayGOREF

from arraykit import mloc as mloc_ak
from arraykit import immutable_filter as immutable_filter_ak
from arraykit import name_filter as name_filter_ak
from arraykit import shape_filter as shape_filter_ak
from arraykit import column_2d_filter as column_2d_filter_ak
from arraykit import column_1d_filter as column_1d_filter_ak
from arraykit import row_1d_filter as row_1d_filter_ak
from arraykit import resolve_dtype as resolve_dtype_ak
from arraykit import resolve_dtype_iter as resolve_dtype_iter_ak
from arraykit import dtype_from_element as dtype_from_element_ak
from arraykit import array_deepcopy as array_deepcopy_ak
from arraykit import delimited_to_arrays as delimited_to_arrays_ak
from arraykit import isna_element as isna_element_ak

from arraykit import ArrayGO as ArrayGOAK


class Perf:
    FUNCTIONS = ('main',)
    NUMBER = 500_000



class FixtureFileLike:

    COUNT_ROW = 1_000

    def __init__(self):
        records_int = [','.join(str(x) for x in range(1000))] * self.COUNT_ROW
        self.file_like_int = io.StringIO('\n'.join(records_int))

        records_bool = [','.join(str(bool(x % 2)) for x in range(1000))] * self.COUNT_ROW
        self.file_like_bool = io.StringIO('\n'.join(records_bool))

        records_str = [','.join('foobar' for x in range(1000))] * self.COUNT_ROW
        self.file_like_str = io.StringIO('\n'.join(records_str))

        records_float = [','.join('1.2345' for x in range(1000))] * self.COUNT_ROW
        self.file_like_float = io.StringIO('\n'.join(records_float))

#-------------------------------------------------------------------------------
class DelimitedToArraysTypedPandas(Perf, FixtureFileLike):
    NUMBER = 10
    FUNCTIONS = ('bool_uniform', 'int_uniform', 'str_uniform', 'float_uniform')
    # COUNT_ROW = 1_000

    # def pre(self):
    #     records_int = [','.join(str(x) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_int = io.StringIO('\n'.join(records_int))

    #     records_bool = [','.join(str(bool(x % 2)) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_bool = io.StringIO('\n'.join(records_bool))

    #     records_str = [','.join('foobar' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_str = io.StringIO('\n'.join(records_str))

    #     records_float = [','.join('1.2345' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_float = io.StringIO('\n'.join(records_float))

class DelimitedToArraysTypedPandasAK(DelimitedToArraysTypedPandas):
    entry = staticmethod(delimited_to_arrays_ak)

    def pre(self):
        self.dtypes_int = [int] * 1000
        self.dtypes_bool = [bool] * 1000
        self.dtypes_str = [str] * 1000
        self.dtypes_float = [float] * 1000
        self.axis = 1

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, dtypes=self.dtypes_int, axis=self.axis)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, dtypes=self.dtypes_bool, axis=self.axis)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, dtypes=self.dtypes_str, axis=self.axis)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, dtypes=self.dtypes_float, axis=self.axis)


class DelimitedToArraysTypedPandasREF(DelimitedToArraysTypedPandas):
    import pandas
    entry = staticmethod(pandas.read_csv)

    def pre(self):
        self.dtypes_int = {i: int for i in range(1000)}
        self.dtypes_bool = {i: bool for i in range(1000)}
        self.dtypes_str = {i: object for i in range(1000)}
        self.dtypes_float = {i: float for i in range(1000)}

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, dtype=self.dtypes_int)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, dtype=self.dtypes_bool)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, dtype=self.dtypes_str)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, dtype=self.dtypes_float)

#-------------------------------------------------------------------------------

class DelimitedToArraysParsedPandas(Perf, FixtureFileLike):
    NUMBER = 10
    FUNCTIONS = ('bool_uniform', 'int_uniform', 'str_uniform', 'float_uniform')
    COUNT_ROW = 1_000

    # def pre(self):
    #     records_int = [','.join(str(x) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_int = io.StringIO('\n'.join(records_int))

    #     records_bool = [','.join(str(bool(x % 2)) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_bool = io.StringIO('\n'.join(records_bool))

    #     records_str = [','.join('foobar' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_str = io.StringIO('\n'.join(records_str))

    #     records_float = [','.join('1.2345' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_float = io.StringIO('\n'.join(records_float))

class DelimitedToArraysParsedPandasAK(DelimitedToArraysParsedPandas):
    entry = staticmethod(delimited_to_arrays_ak)

    def pre(self):
        self.axis = 1

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, dtypes=None, axis=self.axis)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, dtypes=None, axis=self.axis)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, dtypes=None, axis=self.axis)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, dtypes=None, axis=self.axis)


class DelimitedToArraysParsedPandasREF(DelimitedToArraysParsedPandas):
    import pandas
    entry = staticmethod(pandas.read_csv)


    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float)




#-------------------------------------------------------------------------------
class DelimitedToArraysTypedGenft(Perf, FixtureFileLike):
    NUMBER = 10
    FUNCTIONS = ('bool_uniform', 'int_uniform', 'str_uniform', 'float_uniform')
    COUNT_ROW = 1_000

    # def pre(self):
    #     records_int = [','.join(str(x) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_int = io.StringIO('\n'.join(records_int))

    #     records_bool = [','.join(str(bool(x % 2)) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_bool = io.StringIO('\n'.join(records_bool))

    #     records_str = [','.join('foobar' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_str = io.StringIO('\n'.join(records_str))

    #     records_float = [','.join('1.2345' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_float = io.StringIO('\n'.join(records_float))

class DelimitedToArraysTypedGenftAK(DelimitedToArraysTypedGenft):
    entry = staticmethod(delimited_to_arrays_ak)

    def pre(self):
        self.dtypes_int = [int] * 1000
        self.dtypes_bool = [bool] * 1000
        self.dtypes_str = [str] * 1000
        self.dtypes_float = [float] * 1000
        self.axis = 1

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, dtypes=self.dtypes_int, axis=self.axis)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, dtypes=self.dtypes_bool, axis=self.axis)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, dtypes=self.dtypes_str, axis=self.axis)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, dtypes=self.dtypes_float, axis=self.axis)


class DelimitedToArraysTypedGenftREF(DelimitedToArraysTypedGenft):
    entry = staticmethod(np.genfromtxt)


    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, delimiter=',', dtype=int)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, delimiter=',', dtype=bool)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, delimiter=',', dtype=str)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, delimiter=',', dtype=float)


#-------------------------------------------------------------------------------
class DelimitedToArraysParsedGenft(Perf, FixtureFileLike):
    NUMBER = 10
    FUNCTIONS = ('bool_uniform', 'int_uniform', 'str_uniform', 'float_uniform')
    COUNT_ROW = 1_000

    # def pre(self):
    #     records_int = [','.join(str(x) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_int = io.StringIO('\n'.join(records_int))

    #     records_bool = [','.join(str(bool(x % 2)) for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_bool = io.StringIO('\n'.join(records_bool))

    #     records_str = [','.join('foobar' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_str = io.StringIO('\n'.join(records_str))

    #     records_float = [','.join('1.2345' for x in range(1000))] * self.COUNT_ROW
    #     self.file_like_float = io.StringIO('\n'.join(records_float))

class DelimitedToArraysParsedGenftAK(DelimitedToArraysParsedGenft):
    entry = staticmethod(delimited_to_arrays_ak)

    def pre(self):
        self.axis = 1

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, dtypes=None, axis=self.axis)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, dtypes=None, axis=self.axis)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, dtypes=None, axis=self.axis)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, dtypes=None, axis=self.axis)


class DelimitedToArraysParsedGenftREF(DelimitedToArraysParsedGenft):
    entry = staticmethod(np.genfromtxt)

    def int_uniform(self):
        self.file_like_int.seek(0)
        _ = self.entry(self.file_like_int, delimiter=',', dtype=None)

    def bool_uniform(self):
        self.file_like_bool.seek(0)
        _ = self.entry(self.file_like_bool, delimiter=',', dtype=None)

    def str_uniform(self):
        self.file_like_str.seek(0)
        _ = self.entry(self.file_like_str, delimiter=',', dtype=None)

    def float_uniform(self):
        self.file_like_float.seek(0)
        _ = self.entry(self.file_like_float, delimiter=',', dtype=None)


#-------------------------------------------------------------------------------
class MLoc(Perf):

    def __init__(self):
        self.array = np.arange(100)

    def main(self):
        self.entry(self.array)

class MLocAK(MLoc):
    entry = staticmethod(mloc_ak)

class MLocREF(MLoc):
    entry = staticmethod(mloc_ref)

#-------------------------------------------------------------------------------
class ImmutableFilter(Perf):

    def __init__(self):
        self.array = np.arange(100)

    def main(self):
        a2 = self.entry(self.array)
        a3 = self.entry(a2)

class ImmutableFilterAK(ImmutableFilter):
    entry = staticmethod(immutable_filter_ak)

class ImmutableFilterREF(ImmutableFilter):
    entry = staticmethod(immutable_filter_ref)

#-------------------------------------------------------------------------------
class NameFilter(Perf):

    def __init__(self):
        self.name1 = ('foo', None, ['bar'])
        self.name2 = 'foo'

    def main(self):
        try:
            self.entry(self.name1)
        except TypeError:
            pass
        self.entry(self.name2)

class NameFilterAK(NameFilter):
    entry = staticmethod(name_filter_ak)

class NameFilterREF(NameFilter):
    entry = staticmethod(name_filter_ref)

#-------------------------------------------------------------------------------
class ShapeFilter(Perf):

    def __init__(self):
        self.array1 = np.arange(100)
        self.array2 = self.array1.reshape(20, 5)

    def main(self):
        self.entry(self.array1)
        self.entry(self.array2)

class ShapeFilterAK(ShapeFilter):
    entry = staticmethod(shape_filter_ak)

class ShapeFilterREF(ShapeFilter):
    entry = staticmethod(shape_filter_ref)

#-------------------------------------------------------------------------------
class Column2DFilter(Perf):

    def __init__(self):
        self.array1 = np.arange(100)
        self.array2 = self.array1.reshape(20, 5)

    def main(self):
        self.entry(self.array1)
        self.entry(self.array2)

class Column2DFilterAK(Column2DFilter):
    entry = staticmethod(column_2d_filter_ak)

class Column2DFilterREF(Column2DFilter):
    entry = staticmethod(column_2d_filter_ref)


#-------------------------------------------------------------------------------
class Column1DFilter(Perf):

    def __init__(self):
        self.array1 = np.arange(100)
        self.array2 = self.array1.reshape(100, 1)

    def main(self):
        self.entry(self.array1)
        self.entry(self.array2)

class Column1DFilterAK(Column1DFilter):
    entry = staticmethod(column_1d_filter_ak)

class Column1DFilterREF(Column1DFilter):
    entry = staticmethod(column_1d_filter_ref)

#-------------------------------------------------------------------------------
class Row1DFilter(Perf):

    def __init__(self):
        self.array1 = np.arange(100)
        self.array2 = self.array1.reshape(1, 100)

    def main(self):
        self.entry(self.array1)
        self.entry(self.array2)

class Row1DFilterAK(Row1DFilter):
    entry = staticmethod(row_1d_filter_ak)

class Row1DFilterREF(Row1DFilter):
    entry = staticmethod(row_1d_filter_ref)


#-------------------------------------------------------------------------------
class ResolveDType(Perf):

    def __init__(self):
        self.dtype1 = np.arange(100).dtype
        self.dtype2 = np.array(('a', 'b')).dtype

    def main(self):
        self.entry(self.dtype1, self.dtype2)

class ResolveDTypeAK(ResolveDType):
    entry = staticmethod(resolve_dtype_ak)

class ResolveDTypeREF(ResolveDType):
    entry = staticmethod(resolve_dtype_ref)


#-------------------------------------------------------------------------------
class ResolveDTypeIter(Perf):

    FUNCTIONS = ('iter10', 'iter100000')
    NUMBER = 500

    def __init__(self):
        self.dtypes10 = [np.dtype(int)] * 9 + [np.dtype(float)]
        self.dtypes100000 = (
                [np.dtype(int)] * 50000 +
                [np.dtype(float)] * 49999 +
                [np.dtype(bool)]
                )

    def iter10(self):
        self.entry(self.dtypes10)

    def iter100000(self):
        self.entry(self.dtypes100000)

class ResolveDTypeIterAK(ResolveDTypeIter):
    entry = staticmethod(resolve_dtype_iter_ak)

class ResolveDTypeIterREF(ResolveDTypeIter):
    entry = staticmethod(resolve_dtype_iter_ref)


#-------------------------------------------------------------------------------
class ArrayDeepcopy(Perf):
    FUNCTIONS = ('memo_new', 'memo_shared')
    NUMBER = 500

    def __init__(self):
        self.array1 = np.arange(100_000)
        self.array2 = np.full(100_000, None)
        self.array2[0] = [np.nan] # add a mutable
        self.memo = {}

    def memo_new(self):
        memo = {}
        self.entry(self.array1, memo)
        self.entry(self.array2, memo)

    def memo_shared(self):
        self.entry(self.array1, self.memo)
        self.entry(self.array2, self.memo)

class ArrayDeepcopyAK(ArrayDeepcopy):
    entry = staticmethod(array_deepcopy_ak)

class ArrayDeepcopyREF(ArrayDeepcopy):
    entry = staticmethod(array_deepcopy_ref)


#-------------------------------------------------------------------------------
class ArrayGOPerf(Perf):
    NUMBER = 500

    def __init__(self):
        self.array = np.arange(100).astype(object)

    def main(self):
        ag = self.entry(self.array)
        for i in range(1000):
            ag.append(i)
            if i % 50:
                _ = ag.values

class ArrayGOPerfAK(ArrayGOPerf):
    entry = staticmethod(ArrayGOAK)

class ArrayGOPerfREF(ArrayGOPerf):
    entry = staticmethod(ArrayGOREF)


#-------------------------------------------------------------------------------
class DtypeFromElementPerf(Perf):
    NUMBER = 1000

    def __init__(self):
        NT = collections.namedtuple('NT', tuple('abc'))

        self.values = [
                np.longlong(-1), np.int_(-1), np.intc(-1), np.short(-1), np.byte(-1),
                np.ubyte(1), np.ushort(1), np.uintc(1), np.uint(1), np.ulonglong(1),
                np.half(1.0), np.single(1.0), np.float_(1.0), np.longfloat(1.0),
                np.csingle(1.0j), np.complex_(1.0j), np.clongfloat(1.0j),
                np.bool_(0), np.str_('1'), np.unicode_('1'), np.void(1),
                np.object(), np.datetime64('NaT'), np.timedelta64('NaT'), np.nan,
                12, 12.0, True, None, float('NaN'), object(), (1, 2, 3),
                NT(1, 2, 3), datetime.date(2020, 12, 31), datetime.timedelta(14),
        ]

        # Datetime & Timedelta
        for precision in ['ns', 'us', 'ms', 's', 'm', 'h', 'D', 'M', 'Y']:
            for kind, ctor in (('m', np.timedelta64), ('M', np.datetime64)):
                self.values.append(ctor(12, precision))

        for size in (1, 8, 16, 32, 64, 128, 256, 512):
            self.values.append(bytes(size))
            self.values.append('x' * size)

    def main(self):
        for _ in range(40):
            for val in self.values:
                self.entry(val)

class DtypeFromElementPerfAK(DtypeFromElementPerf):
    entry = staticmethod(dtype_from_element_ak)

class DtypeFromElementPerfREF(DtypeFromElementPerf):
    entry = staticmethod(dtype_from_element_ref)


#-------------------------------------------------------------------------------
class IsNaElementPerf(Perf):
    NUMBER = 1000

    def __init__(self):
        class FloatSubclass(float): pass
        class ComplexSubclass(complex): pass

        self.values = [
                # Na-elements
                np.datetime64('NaT'), np.timedelta64('NaT'), None, float('NaN'), -float('NaN'),

                # Non-float, Non-na elements
                1, 'str', np.datetime64('2020-12-31'), datetime.date(2020, 12, 31), False,
        ]

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

        # Append all the different types of nans across dtypes
        for ctor in float_classes:
            self.values.append(ctor(nan))
            self.values.append(ctor(-nan))

        for ctor in cfloat_classes:
            for complex_nan in complex_nans:
                self.values.append(ctor(complex_nan))

        # Append a wide range of float values, with different precision, across types
        for val in (
                1e-1000, 1e-309, 1e-39, 1e-16, 1e-5, 0.1, 0., 1.0, 1e5, 1e16, 1e39, 1e309, 1e1000,
            ):
            for ctor in float_classes:
                self.values.append(ctor(val))
                self.values.append(ctor(-val))

            for ctor in cfloat_classes:
                self.values.append(ctor(complex(val, val)))
                self.values.append(ctor(complex(-val, val)))
                self.values.append(ctor(complex(val, -val)))
                self.values.append(ctor(complex(-val, -val)))

    def main(self):
        for _ in range(10):
            for val in self.values:
                self.entry(val)

class IsNaElementPerfAK(IsNaElementPerf):
    entry = staticmethod(isna_element_ak)

class IsNaElementPerfREF(IsNaElementPerf):
    entry = staticmethod(isna_element_ref)


#-------------------------------------------------------------------------------

def get_arg_parser():

    p = argparse.ArgumentParser(
        description='ArrayKit performance tool.',
        )
    p.add_argument("--names",
        nargs='+',
        help='Provide one or more performance tests by name.')
    return p

def main():
    options = get_arg_parser().parse_args()
    match = None if not options.names else set(options.names)

    records = [('cls', 'func', 'ak', 'ref', 'ref/ak')]
    for cls_perf in Perf.__subclasses__(): # only get one level
        cls_map = {}
        if match and cls_perf.__name__ not in match:
            continue
        print(cls_perf)
        for cls_runner in cls_perf.__subclasses__():
            if cls_runner.__name__.endswith('AK'):
                cls_map['ak'] = cls_runner
            elif cls_runner.__name__.endswith('REF'):
                cls_map['ref'] = cls_runner
        for func_attr in cls_perf.FUNCTIONS:
            results = {}
            for key, cls_runner in cls_map.items():
                runner = cls_runner()
                if hasattr(runner, 'pre'): #TEMP, for branches
                    raise RuntimeError('convert your pre() method to __init__()')
                f = getattr(runner, func_attr)
                results[key] = timeit.timeit('f()',
                        globals=locals(),
                        number=cls_runner.NUMBER)
            records.append((cls_perf.__name__, func_attr, results['ak'], results['ref'], results['ref'] / results['ak']))

    width = 28
    for record in records:
        print(''.join(
            (r.ljust(width) if isinstance(r, str) else str(round(r, 8)).ljust(width)) for r in record
            ))

if __name__ == '__main__':
    main()
