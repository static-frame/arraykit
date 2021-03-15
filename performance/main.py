import timeit
import argparse

import numpy as np
import pandas as pd

from performance.reference.util import mloc as mloc_ref
from performance.reference.util import immutable_filter as immutable_filter_ref
from performance.reference.util import name_filter as name_filter_ref
from performance.reference.util import shape_filter as shape_filter_ref
from performance.reference.util import column_2d_filter as column_2d_filter_ref
from performance.reference.util import column_1d_filter as column_1d_filter_ref
from performance.reference.util import row_1d_filter as row_1d_filter_ref
from performance.reference.util import resolve_dtype as resolve_dtype_ref
from performance.reference.util import resolve_dtype_iter as resolve_dtype_iter_ref
from performance.reference.util import isin_array as isin_array_ref

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
from arraykit import isin_array as isin_array_ak

from arraykit import ArrayGO as ArrayGOAK


class Perf:
    FUNCTIONS = ('main',)
    NUMBER = 500_000

#-------------------------------------------------------------------------------
class MLoc(Perf):

    def pre(self):
        self.array = np.arange(100)

    def main(self):
        self.entry(self.array)

class MLocAK(MLoc):
    entry = staticmethod(mloc_ak)

class MLocREF(MLoc):
    entry = staticmethod(mloc_ref)

#-------------------------------------------------------------------------------
class ImmutableFilter(Perf):

    def pre(self):
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

    def pre(self):
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

    def pre(self):
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

    def pre(self):
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

    def pre(self):
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

    def pre(self):
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

    def pre(self):
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
    NUMBER = 1000

    def pre(self):
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
class ArrayGOPerf(Perf):
    NUMBER = 1000

    def pre(self):
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

def build_arr(dtype, size, num_nans, num_duplicates):
    if dtype.kind == 'M':
        if dtype == 'datetime64[Y]':
            delta = np.timedelta64(size, 'Y')
        elif dtype == 'datetime64[M]':
            delta = np.timedelta64(size, 'M')
        else:
            delta = np.timedelta64(size, 'D')

        start = np.datetime64('2000-01-01').astype(dtype)
        end = start + delta
        arr = np.arange(start, start + delta).astype(dtype)

        nan_val = np.datetime64('NaT')
    else:
        if dtype.kind == 'm':
            nan_val = np.timedelta64('NaT')
        elif dtype.kind == 'c':
            nan_val = np.complex_(np.nan)
        else:
            nan_val = np.nan

        arr = np.arange(size).astype(dtype)

    if num_nans == 1:
        arr = np.concatenate((arr[:-1], [nan_val]*num_nans))
    elif num_nans > 1:
        arr = np.concatenate((arr, [nan_val]*num_nans))

    if num_duplicates:
        indices = np.arange(size)
        np.random.seed(0)
        np.random.shuffle(indices)

        dups = np.array([arr[i] for i in indices[:num_duplicates]])
        dups[~pd.isnull(dups)].astype(dtype)
        arr = np.concatenate((arr, dups))

    np.random.seed(0)
    np.random.shuffle(arr)
    return arr, (num_nans <= 1 and num_duplicates == 0)

storage = []
def build_subclassses(klass, meth):
    #storage.append(type(f'{klass.__name__}AK', (klass,), dict(entry=staticmethod(globals()[f'{meth}_ak']))))
    #storage.append(type(f'{klass.__name__}REF', (klass,), dict(entry=staticmethod(globals()[f'{meth}_ref']))))
    storage.append(type(f'{klass.__name__}AK', (klass,), dict(entry=staticmethod(isin_array_ak))))
    storage.append(type(f'{klass.__name__}REF', (klass,), dict(entry=staticmethod(isin_array_ref))))

class Obj:
    def __init__(self, val):
        self.val = val
    def __eq__(self, other):
        return self.val == other.val
    def __hash__(self):
        return hash(self.val)

def get_dtypes():
    dtypes = [np.dtype(int), np.dtype(float), np.dtype(np.complex_), np.dtype('O')]
    dtypes.extend((np.dtype(f'datetime64[{f}]') for f in 'DMY'))
    dtypes.extend((np.dtype(f'timedelta64[{f}]') for f in 'DMY'))
    return dtypes

class IsinArrayDtypeUnique1DPerf(Perf):
    NUMBER = 3

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size in (100, 5000, 20000, 100000):
                for num_nans in (0, 1):
                    arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates=0)
                    arr2, arr2_unique = build_arr(dtype, size // 25, num_nans // 25, num_duplicates=0)
                    assert arr1_unique and arr2_unique, 'Expect both arrays to be unique'
                    self.kwargs.append(dict(array=arr1, array_is_unique=True, other=arr2, other_is_unique=True))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {1}, "Expected all arr1's to be 1D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)

class IsinArrayDtypeUnique2DPerf(Perf):
    NUMBER = 3

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size, reshape in [
                    (100, (10, 10)),
                    (5000, (200, 25)),
                    (20000, (200, 100)),
                    (100000, (500, 200)),
                ]:
                for num_nans in (0, 1):
                    arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates=0)
                    arr2, arr2_unique = build_arr(dtype, size // 10, num_nans // 10, num_duplicates=0)
                    assert arr1_unique and arr2_unique, 'Expect both arrays to be unique'
                    self.kwargs.append(dict(array=arr1.reshape(reshape), array_is_unique=True, other=arr2, other_is_unique=True))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {2}, "Expected all arr1's to be 2D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)

class IsinArrayDtypeNonUnique1DPerf(Perf):
    NUMBER = 3

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size in (100, 5000, 20000):
                for num_nans, num_duplicates in ((2 + (size // 2), 0), (size // 2, size // 15), (2 + (size // 8), 0), (size // 8, size // 15)):
                    arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates)
                    arr2, arr2_unique = build_arr(dtype, size // 25, num_nans // 25, num_duplicates)
                    assert not arr1_unique or not arr2_unique, 'Expect at least one of the arrays to contains duplicates'
                    self.kwargs.append(dict(array=arr1, array_is_unique=arr1_unique, other=arr2, other_is_unique=arr2_unique))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {1}, "Expected all arr1's to be 1D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)

class IsinArrayDtypeNonUnique2DPerf(Perf):
    NUMBER = 1

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size, num_nans, num_duplicates, reshape in [
                    (90, 10, 35, (27, 5)),
                    (80, 20, 35, (27, 5)),
                    (4500, 500, 950, (119, 50)),
                    (4000, 1000, 950, (119, 50)),
                    (18000, 2000, 2500, (250, 90)),
                    (16000, 4000, 2500, (250, 90)),
                    (90000, 10000, 15000, (500, 230)),
                    (80000, 20000, 15000, (500, 230)),
                ]:
                arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates)
                arr2, arr2_unique = build_arr(dtype, size // 10, int(num_nans / 10), int(num_duplicates / 10))
                assert not arr1_unique or not arr2_unique, 'Expect at least one of the arrays to contains duplicates'
                self.kwargs.append(dict(array=arr1.reshape(reshape), array_is_unique=arr1_unique, other=arr2, other_is_unique=arr2_unique))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {2}, "Expected all arr1's to be 2D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)

class IsinArrayObject1DPerf(Perf):
    NUMBER = 3

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size in (100, 5000, 20000):
                for num_nans, num_duplicates in ((2 + (size // 2), 0), (size // 2, size // 15), (2 + (size // 8), 0), (size // 8, size // 15)):
                    arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates)
                    arr2, arr2_unique = build_arr(dtype, size // 25, num_nans // 25, num_duplicates)
                    assert not arr1_unique or not arr2_unique, 'Expect at least one of the arrays to contains duplicates'
                    self.kwargs.append(dict(array=arr1, array_is_unique=arr1_unique, other=arr2, other_is_unique=arr2_unique))

        for size in (100, 5000, 20000):
            for num_duplicates in (size // 15, 0):
                tmp_arr1, arr1_unique = build_arr(np.dtype(int), size, 0, num_duplicates)
                tmp_arr2, arr2_unique = build_arr(np.dtype(int), size // 25, 0, num_duplicates)

                arr1 = np.array([Obj(v) for v in tmp_arr1])
                arr2 = np.array([Obj(v) for v in tmp_arr2])

                self.kwargs.append(dict(array=arr1, array_is_unique=arr1_unique, other=arr2, other_is_unique=arr2_unique))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {1}, "Expected all arr1's to be 1D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)

class IsinArrayObject2DPerf(Perf):
    NUMBER = 1

    def pre(self):
        self.kwargs = []
        for dtype in get_dtypes():
            for size, num_nans, num_duplicates, reshape in [
                    (100, 0, 0, (10, 10)),
                    (90, 10, 35, (27, 5)),
                    (80, 20, 35, (27, 5)),
                    (5000, 0, 0, (200, 25)),
                    (4500, 500, 950, (119, 50)),
                    (4000, 1000, 950, (119, 50)),
                    (20000, 0, 0, (200, 100)),
                    (18000, 2000, 2500, (250, 90)),
                    (16000, 4000, 2500, (250, 90)),
                    (100000, 1, 0, (500, 200)),
                    (90000, 10000, 15000, (500, 230)),
                    (80000, 20000, 15000, (500, 230)),
                ]:
                arr1, arr1_unique = build_arr(dtype, size, num_nans, num_duplicates)
                arr2, arr2_unique = build_arr(dtype, size // 10, int(num_nans / 10), int(num_duplicates / 10))
                self.kwargs.append(dict(array=arr1.reshape(reshape).astype(object), array_is_unique=arr1_unique, other=arr2.astype(object), other_is_unique=arr2_unique))

        for size, num_duplicates, reshape in [
                (100, 0, (10, 10)),
                (90, 10, (10, 10)),
                (5000, 0, (200, 25)),
                (4500, 500, (200, 25)),
                (20000, 0, (200, 100)),
                (18000, 2000, (200, 100)),
            ]:
            tmp_arr1, arr1_unique = build_arr(np.dtype(int), size, 0, num_duplicates)
            tmp_arr2, arr2_unique = build_arr(np.dtype(int), size // 10, 0, num_duplicates // 10)

            arr1 = np.array([Obj(v) for v in tmp_arr1]).reshape(reshape)
            arr2 = np.array([Obj(v) for v in tmp_arr2])

            self.kwargs.append(dict(array=arr1, array_is_unique=arr1_unique, other=arr2, other_is_unique=arr2_unique))

    def main(self):
        assert set(x['array'].ndim for x in self.kwargs) == {2}, "Expected all arr1's to be 2D"
        for kwargs in self.kwargs:
            self.entry(**kwargs)


build_subclassses(IsinArrayDtypeUnique1DPerf, 'isin_array')
build_subclassses(IsinArrayDtypeUnique2DPerf, 'isin_array')

build_subclassses(IsinArrayDtypeNonUnique1DPerf, 'isin_array')
build_subclassses(IsinArrayDtypeNonUnique2DPerf, 'isin_array')

build_subclassses(IsinArrayObject1DPerf, 'isin_array')
build_subclassses(IsinArrayObject2DPerf, 'isin_array')


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
        assert cls_map
        for func_attr in cls_perf.FUNCTIONS:
            results = {}
            for key, cls_runner in cls_map.items():
                runner = cls_runner()
                runner.pre()
                f = getattr(runner, func_attr)
                results[key] = timeit.timeit('f()',
                        globals=locals(),
                        number=cls_runner.NUMBER)
            records.append((cls_perf.__name__, func_attr, results['ak'], results['ref'], results['ref'] / results['ak']))

    width = 36
    for record in records:
        print(''.join(
            (r.ljust(width) if isinstance(r, str) else str(round(r, 8)).ljust(width)) for r in record
            ))


if __name__ == '__main__':
    main()
