


import os
import sys
import timeit
import typing as tp
from itertools import repeat

from arraykit import BlockIndex
from arraykit import ErrorInitBlocks
from arraykit import shape_filter
import arraykit as ak

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())


def from_blocks(
        raw_blocks: tp.Iterable[np.ndarray],
        ):
    index: tp.List[tp.Tuple[int, int]] = [] # columns position to blocks key
    block_count = 0
    row_count = None
    column_count = 0

    for block in raw_blocks:
        if not block.__class__ is np.ndarray:
            raise ErrorInitTypeBlocks(f'found non array block: {block}')
        if block.ndim > 2:
            raise ErrorInitTypeBlocks(f'cannot include array with {block.ndim} dimensions')
        r, c = shape_filter(block)
        if row_count is not None and r != row_count: #type: ignore [unreachable]
            raise ErrorInitTypeBlocks(f'mismatched row count: {r}: {row_count}')
        else:
            row_count = r
        if c == 0:
            continue
        for i in range(c):
            index.append((block_count, i))
        column_count += c
        block_count += 1
    return row_count, column_count

class ArrayProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, array_iter: tp.Iterator[np.ndarray]):
        self.array_iter = array_iter

#-------------------------------------------------------------------------------
class BlockIndexRegister(ArrayProcessor):
    NAME = 'ak.BlockIUndex.register()'
    SORT = 0

    def __call__(self):
        bi = BlockIndex()
        import ipdb; ipdb.set_trace()
        for a in self.array_iter:
            bi.register(a)
        assert bi.shape[0] == ROW_COUNT


class FromBlocks(ArrayProcessor):
    NAME = 'sf.TypeBlocks.from_blocks()'
    SORT = 0

    def __call__(self):
        shape = from_blocks(self.array_iter)
        assert shape[0] == ROW_COUNT



#-------------------------------------------------------------------------------
NUMBER = 1

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
            names = [cls.NAME for cls in fixture['cls_processor']]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f'{cat_label:.0e}\n{FixtureFactory.DENSITY_TO_DISPLAY[density]}\n{FixtureFactory.POSITION_TO_DISPLAY[position]}'

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    seconds_to_display(time_max * .5),
                    seconds_to_display(time_max),
                    ], fontsize=6)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )

    fig.set_size_inches(9, 3.5) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=8)
    # horizontal, vertical
    fig.text(.05, .96, f'first_true_1d() Performance: {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    fp = '/tmp/first_true.png'
    plt.subplots_adjust(
            left=0.075,
            bottom=0.05,
            right=0.80,
            top=0.85,
            wspace=1, # width
            hspace=0.1,
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
        while size:
            a = np.arange(ROW_COUNT)
            a.flags.writeable = False
            yield a
            size -= 1


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
    BlockIndexRegister,
    FromBlocks,
    )

CLS_FF = (
    FFColumnar,
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



