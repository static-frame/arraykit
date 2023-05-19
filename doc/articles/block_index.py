


import os
import sys
import timeit
import typing as tp
from itertools import repeat
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arraykit import BlockIndex
# from arraykit import ErrorInitTypeBlocks

import arraykit as ak

sys.path.append(os.getcwd())

from performance.reference.block_index import from_blocks



#-------------------------------------------------------------------------------

class ArrayProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, arrays: tp.Iterable[np.ndarray]):
        self.arrays = arrays

        bi = BlockIndex()
        for a in self.arrays:
            bi.register(a)
        self.bi = bi

        # tuple index
        _, self.ti = from_blocks(self.arrays)
        assert len(self.bi) == len(self.ti)

        self.selector_int_array = np.arange(0, len(self.bi), 2)
        self.selector_int_list = list(range(0, len(self.bi), 2))
        self.selector_bool_array = (np.arange(len(self.bi)) % 2) == 0
        self.selector_slice = slice(0, len(self.bi), 2)


#-------------------------------------------------------------------------------
class BlockIndexLoad(ArrayProcessor):
    NAME = 'BlockIndex: load'
    SORT = 0

    def __call__(self):
        bi = BlockIndex()
        for a in self.arrays:
            bi.register(a)
        assert bi.shape[0] == ROW_COUNT

class TupleIndexLoad(ArrayProcessor):
    NAME = 'TupleIndex: load'
    SORT = 10

    def __call__(self):
        shape, index = from_blocks(self.arrays)
        assert shape[0] == ROW_COUNT


class BlockIndexCopy(ArrayProcessor):
    NAME = 'BlockIndex: copy'
    SORT = 2

    def __call__(self):
        for _ in range(10):
            _ = self.bi.copy()

class TupleIndexCopy(ArrayProcessor):
    NAME = 'TupleIndex: copy'
    SORT = 12

    def __call__(self):
        for _ in range(10):
            _ = self.ti.copy()


class BlockIndexPickle(ArrayProcessor):
    NAME = 'BlockIndex: pickle'
    SORT = 4

    def __call__(self):
        msg = pickle.dumps(self.bi)
        bi2 = pickle.loads(msg)

class TupleIndexPickle(ArrayProcessor):
    NAME = 'TupleIndex: pickle'
    SORT = 14

    def __call__(self):
        msg = pickle.dumps(self.ti)
        ti2 = pickle.loads(msg)


class BlockIndexLookup(ArrayProcessor):
    NAME = 'BlockIndex: lookup'
    SORT = 1

    def __call__(self):
        bi = self.bi
        for i in range(len(bi)):
            _ = bi[i]

class BlockIndexLookupBlock(ArrayProcessor):
    NAME = 'BlockIndex: lookup block'
    SORT = 1.1

    def __call__(self):
        bi = self.bi
        for i in range(len(bi)):
            _ = bi.get_block(i)

class TupleIndexLookup(ArrayProcessor):
    NAME = 'TupleIndex: lookup'
    SORT = 11

    def __call__(self):
        ti = self.ti
        for i in range(len(ti)):
            _ = ti[i]


class BlockIndexIterIntArray(ArrayProcessor):
    NAME = 'BlockIndex: iter by int array'
    SORT = 5

    def __call__(self):
        _ = list(self.bi.iter_select(self.selector_int_array))

class TupleIndexIterIntArray(ArrayProcessor):
    NAME = 'TupleIndex: iter by int array'
    SORT = 15

    def __call__(self):
        ti = self.ti
        _ = [ti[i] for i in self.selector_int_array]


class BlockIndexIterIntList(ArrayProcessor):
    NAME = 'BlockIndex: iter by int list'
    SORT = 6

    def __call__(self):
        _ = list(self.bi.iter_select(self.selector_int_list))

class TupleIndexIterIntList(ArrayProcessor):
    NAME = 'TupleIndex: iter by int list'
    SORT = 16

    def __call__(self):
        ti = self.ti
        _ = [ti[i] for i in self.selector_int_list]


class BlockIndexIterSlice(ArrayProcessor):
    NAME = 'BlockIndex: iter by slice'
    SORT = 7

    def __call__(self):
        _ = list(self.bi.iter_select(self.selector_slice))

class TupleIndexIterSlice(ArrayProcessor):
    NAME = 'TupleIndex: iter by slice'
    SORT = 17

    def __call__(self):
        ti = self.ti
        _ = list(iter(ti[self.selector_slice]))




class BlockIndexIterBoolArray(ArrayProcessor):
    NAME = 'BlockIndex: iter by bool array'
    SORT = 8

    def __call__(self):
        _ = list(self.bi.iter_select(self.selector_bool_array))

class TupleIndexIterBoolArray(ArrayProcessor):
    NAME = 'TupleIndex: iter by bool array'
    SORT = 18

    def __call__(self):
        ti = self.ti
        _ = [ti[i] for i, b in enumerate(self.selector_bool_array) if b]


#-------------------------------------------------------------------------------
NUMBER = 5

def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f'{seconds * 1e6: .1f} (µs)'
    if seconds < 1e-1:
        return f'{seconds * 1e3: .1f} (ms)'
    return f'{seconds: .1f} (s)'


def plot_performance(frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['size'].unique())
    processor_total = len(frame['cls_processor'].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

    color = cmap(np.arange(processor_total) / processor_total)

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby('size')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.groupby('fixture')):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture['sort'] = [f.SORT for f in fixture['cls_processor']]
            fixture = fixture.sort_values('sort')

            results = fixture['time'].values.tolist()
            x_labels = [f'{i}: {cls.NAME}' for i, cls in
                    zip(range(1, len(results) + 1), fixture['cls_processor'])
                    ]
            x_tick_labels = [str(l + 1) for l in range(len(x_labels))]
            x = np.arange(len(results))
            x_bar = ax.bar(x_labels, results, color=color)

            title = f'{cat_label:.0e}\n{fixture_label}'
            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.5) # larger makes taller tan wide

            time_max = fixture["time"].max()
            time_min = fixture["time"].min()
            y_ticks = [0, time_min, time_max * 0.5, time_max]
            y_labels = [
                "",
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
                axis="y",
                length=2,
                width=0.5,
                pad=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(x_tick_labels, fontsize=4)
            ax.tick_params(
                axis="x",
                length=2,
                width=0.5,
                pad=1,
            )
            # ax.set_yscale('log')

    fig.set_size_inches(9, 3.5) # width, height
    fig.legend(x_bar, x_labels, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(.05, .96, f'BlockIndex Performance: {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    fp = '/tmp/block_index.png'
    plt.subplots_adjust(
            left=0.075,
            bottom=0.05,
            right=0.80,
            top=0.80,
            wspace=0.6, # width
            hspace=0.6,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


#-------------------------------------------------------------------------------

ROW_COUNT = 2

class FixtureFactory:
    NAME = ''

    @staticmethod
    def get_arrays(size: int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def get_label_arrays(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = list(cls.get_arrays(size))
        return cls.NAME, array


class FFColumnar(FixtureFactory):
    NAME = 'columnar'

    @staticmethod
    def get_arrays(size: int) -> tp.Iterator[np.ndarray]:
        while size > 0:
            a = np.arange(ROW_COUNT)
            a.flags.writeable = False
            yield a
            size -= 1

from itertools import cycle
class FFMixed(FixtureFactory):
    NAME = 'mixed'

    @staticmethod
    def get_arrays(size: int) -> tp.Iterator[np.ndarray]:
        widths = cycle((8, 16, 4))
        while size > 0:
            w = next(widths)
            a = np.arange(ROW_COUNT * w).reshape(ROW_COUNT, w)
            a.flags.writeable = False
            yield a
            size -= w

class FFUniform(FixtureFactory):
    NAME = 'uniform'

    @staticmethod
    def get_arrays(size: int) -> tp.Iterator[np.ndarray]:
        a = np.arange(ROW_COUNT * size).reshape(ROW_COUNT, size)
        a.flags.writeable = False
        yield a


def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / ArrayKit: {ak.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    BlockIndexLoad,
    TupleIndexLoad,
    BlockIndexCopy,
    TupleIndexCopy,
    # BlockIndexPickle,
    # TupleIndexPickle,
    BlockIndexLookup,
    TupleIndexLookup,
    # BlockIndexLookupBlock,
    BlockIndexIterIntArray,
    TupleIndexIterIntArray,
    BlockIndexIterIntList,
    TupleIndexIterIntList,
    BlockIndexIterBoolArray,
    TupleIndexIterBoolArray,
    BlockIndexIterSlice,
    TupleIndexIterSlice,
    )

CLS_FF = (
    FFColumnar,
    FFMixed,
    FFUniform,
)


def run_test():
    records = []
    for size in (10_000, 100_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_arrays(size)
            for cls in CLS_PROCESSOR:
                runner = cls(fixture)

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(
                            f'runner()',
                            globals=locals(),
                            number=NUMBER)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(records,
            columns=('cls_processor', 'number', 'fixture', 'size', 'time')
            )
    print(f)
    plot_performance(f)

if __name__ == '__main__':

    run_test()



