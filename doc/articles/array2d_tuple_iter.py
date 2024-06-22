import os
import sys
import timeit
import typing as tp

from arraykit import array_to_tuple_iter
import arraykit as ak

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

class ArrayProcessor:
    NAME = ''
    SORT = -1

    def __init__(self, array: np.ndarray):
        self.array = array

#-------------------------------------------------------------------------------
class AKArray2DTupleList(ArrayProcessor):
    NAME = 'list(ak.array_to_tuple_iter(a2d))'
    SORT = 0

    def __call__(self):
        _ = list(array_to_tuple_iter(self.array))

class AKArray2DTupleNext(ArrayProcessor):
    NAME = 'next(ak.array_to_tuple_iter(a2d))'
    SORT = 1

    def __call__(self):
        it = array_to_tuple_iter(self.array)
        while True:
            try:
                _ = next(it)
            except StopIteration:
                break

class PyArray2DTupleMapList(ArrayProcessor):
    NAME = 'list(map(tuple, a2d))'
    SORT = 2

    def __call__(self):
        array = self.array
        _ = list(map(tuple, array))

class PyArray2DTupleIterNext(ArrayProcessor):
    NAME = 'tuple(next(iter(a2d)))'
    SORT = 3

    def __call__(self):
        it = iter(self.array)
        while True:
            try:
                _ = tuple(next(it))
            except StopIteration:
                break





#-------------------------------------------------------------------------------
NUMBER = 200

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

    color = cmap(np.arange(processor_total) / max(processor_total, 3))

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby('size')):
        # each fixture is a collection of tests for one display
        fixtures = {fixture_label: fixture for fixture_label, fixture in cat.groupby('fixture')}
        for fixture_count, (fixture_label, fixture) in enumerate(
                (k, fixtures[k]) for k in FixtureFactory.DENSITY_TO_DISPLAY):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture['sort'] = [f.SORT for f in fixture['cls_processor']]
            fixture = fixture.sort_values('sort')

            results = fixture['time'].values.tolist()
            names = [cls.NAME for cls in fixture['cls_processor']]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            # density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f'{cat_label:.0e}\n{FixtureFactory.DENSITY_TO_DISPLAY[fixture_label]}'

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.75) # makes taller than wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    seconds_to_display(time_max * .5),
                    seconds_to_display(time_max),
                    ], fontsize=4)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )

    fig.set_size_inches(8, 4) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(.05, .96, f'array_to_tuple_iter() Performance: {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    fp = '/tmp/array_to_tuple_iter.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.8,
            top=0.85,
            wspace=0.1, # width
            hspace=0.5,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


#-------------------------------------------------------------------------------

class FixtureFactory:
    NAME = ''

    @staticmethod
    def get_array(size: int, width_ratio: int) -> np.ndarray:
        return np.arange(size).reshape(size // width_ratio, width_ratio)

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array

    DENSITY_TO_DISPLAY = {
        'column-2': '2 Column',
        'column-5': '5 Column',
        'column-10': '10 Column',
        'column-20': '20 Column',
    }


class FFC2(FixtureFactory):
    NAME = 'column-2'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, 2)
        return a

class FFC5(FixtureFactory):
    NAME = 'column-5'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, 5)
        return a

class FFC10(FixtureFactory):
    NAME = 'column-10'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, 10)
        return a

class FFC20(FixtureFactory):
    NAME = 'column-20'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, 20)
        return a

def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / ArrayKit: {ak.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    AKArray2DTupleList,
    AKArray2DTupleNext,
    PyArray2DTupleMapList,
    PyArray2DTupleIterNext,
    )


CLS_FF = (
    FFC2,
    FFC5,
    FFC10,
    FFC20,
)


def run_test():
    records = []
    for size in (1_000, 10_000, 100_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
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



