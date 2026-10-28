import gc
import io
import os
from pathlib import Path
import tempfile
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


@unittest.skipIf(pa is None, 'PyArrow is not installed')
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
        file = self.to_file(pa.table({'a': [1, 2]}))
        with mock.patch.object(pq, 'read_table', side_effect=AssertionError):
            with mock.patch.object(pq, 'ParquetFile', side_effect=AssertionError):
                np.testing.assert_array_equal(parquet_to_arrays(file)[0], [1, 2])


class TestOptionalDependency(unittest.TestCase):
    def test_parquet_to_arrays_missing_pyarrow(self):
        with mock.patch.dict('sys.modules', {'pyarrow._parquet': None}):
            with self.assertRaises(ImportError):
                parquet_to_arrays(io.BytesIO())


if __name__ == '__main__':
    unittest.main()
