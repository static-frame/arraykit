import os
import sys
import timeit

import arraykit
from arraykit import parquet_to_arrays

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.append(os.getcwd())

from fixtures import (
    PayLoad,
    COLUMNS,
    FFInt64,
    FFInt64Null,
    FFFloat64,
    FFBool,
    FFDTns,
    FFString,
    FFStringDict,
)


class ParquetProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, pl: PayLoad):
        self.pl = pl
        self.path = pl.path
        self.size = pl.size
        self.columns_select = pl.columns_select


def table_to_arrays(table: pa.Table) -> list:
    # ChunkedArray.to_numpy() applies the same null conversions as ArrayKit:
    # nullable integers to float64 with NaN, nullable bool and strings to object.
    return [column.to_numpy() for column in table.columns]


# -------------------------------------------------------------------------------
class AKPath(ParquetProcessor):
    NAME = 'AK: path'
    SORT = 0

    def __call__(self):
        arrays = parquet_to_arrays(self.path)
        assert len(arrays[0]) == self.size


class AKBytesIO(ParquetProcessor):
    NAME = 'AK: BytesIO'
    SORT = 1

    def __call__(self):
        arrays = parquet_to_arrays(self.pl.bytes_io())
        assert len(arrays[0]) == self.size


class PAPath(ParquetProcessor):
    NAME = 'PA: path'
    SORT = 2

    def __call__(self):
        arrays = table_to_arrays(pq.read_table(self.path))
        assert len(arrays[0]) == self.size


class PAPathSingleThread(ParquetProcessor):
    NAME = 'PA: path, 1 thread'
    SORT = 3

    def __call__(self):
        arrays = table_to_arrays(pq.read_table(self.path, use_threads=False))
        assert len(arrays[0]) == self.size


class PABytesIO(ParquetProcessor):
    NAME = 'PA: BytesIO'
    SORT = 4

    def __call__(self):
        arrays = table_to_arrays(pq.read_table(self.pl.bytes_io()))
        assert len(arrays[0]) == self.size


# -------------------------------------------------------------------------------
class AKPathSelect(ParquetProcessor):
    NAME = 'AK: path select'
    SORT = 0

    def __call__(self):
        arrays = parquet_to_arrays(self.path, columns_select=self.columns_select)
        assert len(arrays) == len(self.columns_select)


class AKBytesIOSelect(ParquetProcessor):
    NAME = 'AK: BytesIO select'
    SORT = 1

    def __call__(self):
        arrays = parquet_to_arrays(
            self.pl.bytes_io(), columns_select=self.columns_select
        )
        assert len(arrays) == len(self.columns_select)


class PAPathSelect(ParquetProcessor):
    NAME = 'PA: path select'
    SORT = 2

    def __call__(self):
        table = pq.read_table(self.path, columns=self.columns_select)
        arrays = table_to_arrays(table)
        assert len(arrays) == len(self.columns_select)


class PAPathSingleThreadSelect(ParquetProcessor):
    NAME = 'PA: path select, 1 thread'
    SORT = 3

    def __call__(self):
        table = pq.read_table(
            self.path, columns=self.columns_select, use_threads=False
        )
        arrays = table_to_arrays(table)
        assert len(arrays) == len(self.columns_select)


class PABytesIOSelect(ParquetProcessor):
    NAME = 'PA: BytesIO select'
    SORT = 4

    def __call__(self):
        table = pq.read_table(self.pl.bytes_io(), columns=self.columns_select)
        arrays = table_to_arrays(table)
        assert len(arrays) == len(self.columns_select)


def get_versions() -> str:
    import platform

    return (
        f'OS: {platform.system()} / ArrayKit: {arraykit.__version__} / '
        f'NumPy: {np.__version__} / PyArrow: {pa.__version__} / '
        f'Columns: {COLUMNS}\n'
    )


CLS_FF = (
    FFInt64,
    FFInt64Null,
    FFFloat64,
    FFBool,
    FFDTns,
    FFString,
    FFStringDict,
)
FF_ORDER = [f.NAME for f in sorted(CLS_FF, key=lambda ff: ff.SORT)]

# -------------------------------------------------------------------------------
NUMBER = 5


def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'


def plot_performance(frame, suffix: str = ''):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['size'].unique())
    processor_total = len(frame['cls_processor'].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    cmap = plt.get_cmap('plasma')
    color = cmap(np.arange(processor_total) / processor_total)

    # category is the number of rows
    for cat_count, (cat_label, cat) in enumerate(frame.groupby('size')):
        # fixture is the column type fixture
        fixture_data = {fix_label: fix for fix_label, fix in cat.groupby('fixture')}
        for fixture_count, fixture_label in enumerate(FF_ORDER):
            fixture = fixture_data[fixture_label]
            ax = axes[cat_count][fixture_count]

            # set order by cls_processor, i.e., the type of test being done
            fixture['sort'] = [f.SORT for f in fixture['cls_processor']]
            fixture = fixture.sort_values('sort')

            results = fixture['time'].values.tolist()
            names = [cls.NAME for cls in fixture['cls_processor']]
            names_display = names
            post = ax.bar(names_display, results, color=color)

            title = f'{cat_label:.0e}\n{fixture_label}'

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.8)
            time_max = fixture['time'].max()
            time_min = fixture['time'].min()
            y_ticks = [0, time_min, time_max * 0.5, time_max]
            y_labels = [
                '',
                seconds_to_display(time_min),
                seconds_to_display(time_max * 0.5),
                seconds_to_display(time_max),
            ]
            if time_min > time_max * 0.25:
                # remove the min if it is greater than quarter
                y_ticks.pop(1)
                y_labels.pop(1)

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=4)
            ax.tick_params(
                axis='x',
                bottom=False,
                labelbottom=False,
            )
            ax.tick_params(
                axis='y',
                length=2,
                width=0.5,
                pad=1,
            )
    fig.set_size_inches(10, 3.5)  # width, height
    fig.legend(post, names_display, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(
        0.05, 0.96, f'Parquet to Arrays {suffix.title()}: {NUMBER} Iterations', fontsize=10
    )
    fig.text(0.05, 0.90, get_versions(), fontsize=6)

    fp = f'/tmp/arraykit-parquet-{suffix}.png'
    plt.subplots_adjust(
        left=0.075,
        bottom=0.05,
        right=0.82,
        top=0.80,
        wspace=1.0,  # width
        hspace=0.5,
    )
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


def run_test(processors, suffix):
    records = []
    for size in (10_000, 100_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_payload(size)
            for cls in processors:
                runner = cls(fixture)
                runner()  # warm up file system cache and imports

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                result = timeit.timeit(f'runner()', globals=locals(), number=NUMBER)
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(
        records, columns=('cls_processor', 'number', 'fixture', 'size', 'time')
    )
    print(f)
    plot_performance(f, suffix)


if __name__ == '__main__':
    cls_read = (
        AKPath,
        AKBytesIO,
        PAPath,
        PAPathSingleThread,
        PABytesIO,
    )

    cls_select = (
        AKPathSelect,
        AKBytesIOSelect,
        PAPathSelect,
        PAPathSingleThreadSelect,
        PABytesIOSelect,
    )

    run_test(cls_read, 'read')
    run_test(cls_select, 'select')
