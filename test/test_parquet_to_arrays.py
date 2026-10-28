import gc
from decimal import Decimal
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from arraykit import parquet_to_arrays

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


@unittest.skipIf(pa is None, 'PyArrow is not installed for fixture generation')
@unittest.skipIf(
    importlib.util.find_spec('arraykit._parquet') is None,
    'Native Parquet backend is not built',
)
class TestUnit(unittest.TestCase):
    @staticmethod
    def to_file(table, **kwargs):
        file = io.BytesIO()
        pq.write_table(table, file, **kwargs)
        file.seek(0)
        return file

    def test_parquet_to_arrays_types(self):
        table = pa.table(
            {
                'int': pa.array([1, 2, 3], type=pa.int32()),
                'uint': pa.array([0, 2**64 - 1, 3], type=pa.uint64()),
                'float': pa.array([1.5, 2.5, 3.5], type=pa.float32()),
                'bool': [True, False, True],
                'string': ['a', 'é', '水'],
                'binary': [b'a\x00', b'b', b'c'],
                'time': pa.array([1, 2, 3], type=pa.timestamp('us')),
            }
        )
        for compression in (None, 'snappy', 'gzip', 'zstd'):
            with self.subTest(compression=compression):
                arrays = parquet_to_arrays(self.to_file(table, compression=compression))
                self.assertIsInstance(arrays, list)
                self.assertEqual(len(arrays), table.num_columns)
                for array, expected in zip(arrays, table.columns):
                    self.assertEqual(array.ndim, 1)
                    self.assertFalse(array.flags.writeable)
                    np.testing.assert_array_equal(array, expected.to_numpy())
                    self.assertEqual(array.dtype, expected.to_numpy().dtype)

    def test_parquet_to_arrays_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'é.parquet'
            pq.write_table(pa.table({'x': [1, 2]}), path)
            for source in (path, str(path), os.fsencode(path)):
                with self.subTest(source=source):
                    np.testing.assert_array_equal(parquet_to_arrays(source)[0], [1, 2])
            with path.open('rb') as file:
                np.testing.assert_array_equal(parquet_to_arrays(file)[0], [1, 2])
                self.assertFalse(file.closed)
                file.seek(0)
                np.testing.assert_array_equal(parquet_to_arrays(file)[0], [1, 2])

    def test_parquet_to_arrays_select(self):
        file = self.to_file(pa.table({'a': [1], 'b': [2], 'c': [3]}))
        arrays = parquet_to_arrays(file, columns_select=['c', 'a'])
        self.assertEqual([a.tolist() for a in arrays], [[3], [1]])
        self.assertEqual(parquet_to_arrays(file, columns_select=[]), [])
        self.assertEqual(
            [a.tolist() for a in parquet_to_arrays(file, columns_select=None)],
            [[1], [2], [3]],
        )
        for selection in (['missing'], ['a', 'a']):
            with self.subTest(selection=selection):
                with self.assertRaises(ValueError):
                    parquet_to_arrays(file, columns_select=selection)
                self.assertFalse(file.closed)
        for selection in ('a', ('a',), [1], [None]):
            with self.subTest(selection=selection):
                with self.assertRaises(TypeError):
                    parquet_to_arrays(file, columns_select=selection)
        with self.assertRaises(TypeError):
            parquet_to_arrays(file, ['a'])

    def test_parquet_to_arrays_nulls_and_row_groups(self):
        table = pa.table(
            {
                'int': [1, None, 3, 4],
                'float': [1.5, None, float('nan'), 4.5],
                'bool': [True, None, False, True],
                'string': ['a', None, '', 'd'],
                'null': pa.nulls(4),
                'time': pa.array([1, None, 3, 4], type=pa.timestamp('ns')),
            }
        )
        arrays = parquet_to_arrays(self.to_file(table, row_group_size=1))
        gc.collect()
        for array, column in zip(arrays, table.columns):
            np.testing.assert_array_equal(array, column.to_numpy())
            self.assertFalse(array.flags.writeable)

    def test_parquet_to_arrays_empty(self):
        table = pa.table({'int': pa.array([], type=pa.int64()), 'string': []})
        arrays = parquet_to_arrays(self.to_file(table))
        self.assertEqual([a.shape for a in arrays], [(0,), (0,)])
        self.assertEqual(arrays[0].dtype, np.dtype(np.int64))
        self.assertEqual(parquet_to_arrays(self.to_file(pa.table({}))), [])

    def test_parquet_to_arrays_nested(self):
        table = pa.table(
            {
                'struct': [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}],
                'list': [[1, 2], [3]],
                'a.b': [5, 6],
            }
        )
        file = self.to_file(table)
        arrays = parquet_to_arrays(file, columns_select=['a.b', 'struct', 'list'])
        self.assertEqual(arrays[0].tolist(), [5, 6])
        self.assertEqual(arrays[1].tolist(), table.column('struct').to_pylist())
        self.assertEqual([a.tolist() for a in arrays[2]], [[1, 2], [3]])
        with self.assertRaises(ValueError):
            parquet_to_arrays(file, columns_select=['struct.a'])

    def test_parquet_to_arrays_dictionary(self):
        table = pa.table({'x': pa.array(['a', 'b', 'a']).dictionary_encode()})
        array = parquet_to_arrays(self.to_file(table))[0]
        self.assertEqual(array.tolist(), ['a', 'b', 'a'])
        self.assertEqual(array.dtype, np.dtype(object))

    def test_parquet_to_arrays_nested_float16(self):
        values = [
            {'x': 1.5}, {'x': None}, None, {'x': -0.0},
            {'x': 2**-24}, {'x': float('inf')}, {'x': float('nan')},
        ]
        table = pa.table({'a': pa.array(values, type=pa.struct([('x', pa.float16())]))})
        array = parquet_to_arrays(self.to_file(table, row_group_size=1))[0]
        self.assertEqual(array.dtype, np.dtype(object))
        self.assertFalse(array.flags.writeable)
        self.assertEqual(array[:3].tolist(), values[:3])
        np.testing.assert_array_equal(
            [value['x'] for value in array[3:]],
            [-0.0, 2**-24, float('inf'), float('nan')],
        )
        self.assertTrue(np.signbit(array[3]['x']))

    def test_parquet_to_arrays_wide_projection(self):
        names = [f'c{i}' for i in range(128)]
        table = pa.table({name: [i] for i, name in enumerate(names)})
        arrays = parquet_to_arrays(self.to_file(table), columns_select=names[::-1])
        self.assertEqual([a[0] for a in arrays], list(range(127, -1, -1)))

    def test_parquet_to_arrays_ambiguous_names(self):
        table = pa.Table.from_arrays([pa.array([1]), pa.array([2])], names=['x', 'x'])
        file = self.to_file(table)
        self.assertEqual([a.tolist() for a in parquet_to_arrays(file)], [[1], [2]])
        with self.assertRaises(ValueError):
            parquet_to_arrays(file, columns_select=['x'])

    def test_parquet_to_arrays_errors(self):
        for source in (io.BytesIO(b'not parquet'), io.StringIO('not parquet'), 42):
            with self.subTest(source=source):
                with self.assertRaises((ValueError, TypeError, OSError)):
                    parquet_to_arrays(source)
        file = self.to_file(pa.table({'a': [1]}))
        file.close()
        with self.assertRaises(ValueError):
            parquet_to_arrays(file)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(OSError):
                parquet_to_arrays(Path(directory) / 'missing.parquet')

    def test_parquet_to_arrays_native_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.parquet'
            pq.write_table(pa.table({'a': [1, 2], 'b': ['x', 'y']}), path)
            code = '''
import importlib.abc
import io
import sys

class NoPyArrow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow' or fullname.startswith('pyarrow.'):
            raise AssertionError('Reader must not import Python PyArrow')
        return None

sys.meta_path.insert(0, NoPyArrow())
from arraykit import parquet_to_arrays
arrays = parquet_to_arrays(sys.argv[1], columns_select=['b', 'a'])
assert [a.tolist() for a in arrays] == [['x', 'y'], [1, 2]]
with open(sys.argv[1], 'rb') as file:
    arrays = parquet_to_arrays(io.BytesIO(file.read()))
assert [a.tolist() for a in arrays] == [[1, 2], ['x', 'y']]
assert not any(name.startswith('pyarrow') for name in sys.modules)
'''
            subprocess.run([sys.executable, '-c', code, str(path)], check=True)

    def test_parquet_to_arrays_file_errors(self):
        class ReadError(io.BytesIO):
            def read(self, size=-1):
                raise RuntimeError('custom read error')

        class SeekError(io.BytesIO):
            def seek(self, offset, whence=0):
                raise RuntimeError('custom seek error')

        data = self.to_file(pa.table({'a': [1]})).getvalue()
        for cls, message in (
            (ReadError, 'custom read error'),
            (SeekError, 'custom seek error'),
        ):
            with self.subTest(cls=cls):
                with self.assertRaisesRegex(RuntimeError, message):
                    parquet_to_arrays(cls(data))
        with self.assertRaises(ValueError):
            parquet_to_arrays('data.parquet\x00ignored')

    def test_parquet_to_arrays_dates(self):
        table = pa.table(
            {
                'date': pa.array([0, None, 100], type=pa.date32()),
                'timestamp_ms': pa.array([0, None, 100], type=pa.timestamp('ms')),
                'timestamp_ns': pa.array([0, None, 100], type=pa.timestamp('ns')),
            }
        )
        arrays = parquet_to_arrays(self.to_file(table, row_group_size=1))
        for array, expected, unit in zip(
            arrays, ([0, 'NaT', 100],) * 3, ('D', 'ms', 'ns')
        ):
            np.testing.assert_array_equal(
                array, np.array(expected, dtype=f'datetime64[{unit}]')
            )

    def test_parquet_to_arrays_nullable_numeric_types(self):
        types = (
            pa.int8(), pa.int16(), pa.int32(), pa.int64(),
            pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
            pa.float16(), pa.float32(), pa.float64(),
        )
        for dtype in types:
            with self.subTest(dtype=dtype):
                table = pa.table({'a': pa.array([1, None, 3], type=dtype)})
                array = parquet_to_arrays(self.to_file(table, row_group_size=1))[0]
                expected_dtype = (
                    np.float64 if pa.types.is_integer(dtype) else dtype.to_pandas_dtype()
                )
                np.testing.assert_array_equal(array, [1, np.nan, 3])
                self.assertEqual(array.dtype, np.dtype(expected_dtype))

    def test_parquet_to_arrays_decimal(self):
        values = [Decimal('1.23'), None, Decimal('-2.50')]
        table = pa.table({'a': pa.array(values, type=pa.decimal128(8, 2))})
        array = parquet_to_arrays(self.to_file(table))[0]
        self.assertEqual(array.tolist(), values)
        self.assertFalse(array.flags.writeable)

    def test_parquet_to_arrays_short_reads(self):
        class ShortRead(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 7))

        data = self.to_file(pa.table({'a': [1, 2]})).getvalue()
        file = ShortRead(data)
        self.assertEqual(parquet_to_arrays(file)[0].tolist(), [1, 2])
        self.assertFalse(file.closed)


class TestOptionalDependency(unittest.TestCase):
    def test_parquet_to_arrays_missing_backend(self):
        with mock.patch.dict('sys.modules', {'arraykit._parquet': None}):
            with self.assertRaisesRegex(ImportError, 'ARRAYKIT_WITH_PARQUET'):
                parquet_to_arrays(io.BytesIO())


if __name__ == '__main__':
    unittest.main()
