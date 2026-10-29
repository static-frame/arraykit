import atexit
import io
import os
import shutil
import tempfile
import typing as tp

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


COLUMNS = 10
NULL_INTERVAL = 10  # every n-th value is null in nullable fixtures

DIRECTORY = tempfile.mkdtemp(prefix='arraykit-parquet-')
atexit.register(shutil.rmtree, DIRECTORY, ignore_errors=True)


class PayLoad:
    def __init__(self, label: str, size: int, table: pa.Table):
        self.table = table
        self.size = size
        self.path = os.path.join(DIRECTORY, f'{label}-{size}.parquet')
        pq.write_table(table, self.path)
        with open(self.path, 'rb') as f:
            self.data = f.read()
        self.columns = table.column_names
        self.columns_select = self.columns[::2]

    def bytes_io(self) -> io.BytesIO:
        return io.BytesIO(self.data)


# -------------------------------------------------------------------------------
def get_table(get_array: tp.Callable[[int, int], pa.Array], size: int) -> pa.Table:
    return pa.table({f'c{i}': get_array(size, i) for i in range(COLUMNS)})


def get_null_mask(size: int) -> np.ndarray:
    return (np.arange(size) % NULL_INTERVAL) == 0


class FixtureFactory:
    NAME = ''
    SORT = 0
    CACHE = {}  # can be shared for all classes

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        raise NotImplementedError()

    @classmethod
    def get_label_payload(cls, size: int) -> tp.Tuple[str, PayLoad]:
        key = (cls, size)
        if key not in cls.CACHE:
            table = get_table(cls.get_array, size)
            cls.CACHE[key] = PayLoad(cls.NAME, size, table)
        return cls.NAME, cls.CACHE[key]


class FFInt64(FixtureFactory):
    NAME = 'int64'
    SORT = 0

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        return pa.array(np.arange(column, column + size, dtype=np.int64))


class FFInt64Null(FixtureFactory):
    NAME = 'int64 (nulls)'
    SORT = 1

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        values = np.arange(column, column + size, dtype=np.int64)
        return pa.array(values, mask=get_null_mask(size))


class FFFloat64(FixtureFactory):
    NAME = 'float64'
    SORT = 2

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        return pa.array(np.arange(column, column + size, dtype=np.float64) * 0.5)


class FFBool(FixtureFactory):
    NAME = 'bool'
    SORT = 3

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        return pa.array((np.arange(size) + column) % 3 == 0)


class FFDTns(FixtureFactory):
    NAME = 'dt64[ns]'
    SORT = 4

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        values = np.arange(size, dtype=np.int64) * 1_000_000_007 + column
        return pa.array(values.astype('datetime64[ns]'))


class FFString(FixtureFactory):
    NAME = 'string'
    SORT = 5

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        return pa.array([hex(e) * 2 for e in range(column, column + size)])


class FFStringDict(FixtureFactory):
    NAME = 'string (dict)'
    SORT = 6

    @staticmethod
    def get_array(size: int, column: int) -> pa.Array:
        labels = np.array([f'label-{i}' for i in range(100)], dtype=object)
        return pa.array(labels[(np.arange(size) + column) % 100]).dictionary_encode()
