


import os
import sys
import timeit
import typing as tp
from itertools import repeat

from arraykit import first_true_1d

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())



class ArrayProcessor:

    def __init__(self, array: np.ndarray):
        self.array = array

#-------------------------------------------------------------------------------
class AKFirstTrue(ArrayProcessor):
    def __call__(self):
        _ = first_true_1d(self.array, forward=True)

class NPNonZero(ArrayProcessor):
    def __call__(self):
        _ = np.nonzero(self.array)[0][0]

class NPArgMax(ArrayProcessor):
    def __call__(self):
        _ = np.argmax(self.array)


#-------------------------------------------------------------------------------


def plot_performance(frame):
    fixture_total = len(frame['fixture'].unique())
    cat_total = len(frame['category'].unique())
    name_total = len(frame['name'].unique())

    fig, axes = plt.subplots(cat_total, fixture_total)

    # for legend
    name_replace = {
        SFTypeParse.__name__: 'StaticFrame\n(type parsing)',
        SFStr.__name__: 'StaticFrame\n(as string)',
        SFTypeGiven.__name__: 'StaticFrame\n(type given)',

    }

    name_order = {
        SFTypeParse.__name__: 0,
        SFStr.__name__: 0,
        SFTypeGiven.__name__: 0,

        PandasTypeParse.__name__: 1,
        PandasStr.__name__: 1,
        PandasTypeGiven.__name__: 1,

    }

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap('plasma')

    color = cmap(np.arange(name_total) / name_total)

    # categories are read, write
    for cat_count, (cat_label, cat) in enumerate(frame.iter_group_items('category')):
        for fixture_count, (fixture_label, fixture) in enumerate(
                cat.iter_group_items('fixture')):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture = fixture.sort_values('name', key=lambda s:s.iter_element().map_all(name_order))
            results = fixture['time'].values.tolist()
            names = fixture['name'].values.tolist()
            x = np.arange(len(results))
            names_display = [name_replace[l] for l in names]
            post = ax.bar(names_display, results, color=color)

            # ax.set_ylabel()
            cat_io, cat_dtype = cat_label.split(' ')
            title = f'{cat_dtype.title()}\n{fixture_label}'
            ax.set_title(title, fontsize=8)
            ax.set_box_aspect(0.75) # makes taller tan wide
            time_max = fixture['time'].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(['',
                    f'{time_max * 0.5:.3f} (s)',
                    f'{time_max:.3f} (s)',
                    ], fontsize=6)
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    )

    fig.set_size_inches(6, 3.5) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=8)
    # horizontal, vertical
    count = ff.parse(FF_tall_uniform).size
    fig.text(.05, .96, f'Delimited Read Performance: {count:.0e} Elements, {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)
    # get fixtures size reference
    # shape_map = {shape: FIXTURE_SHAPE_MAP[shape] for shape in frame['fixture'].unique()}
    # shape_msg = ' / '.join(f'{v}: {k}' for k, v in shape_map.items())
    # fig.text(.05, .90, shape_msg, fontsize=6)

    fp = '/tmp/dta.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.75,
            top=0.75,
            wspace=-0.2, # width
            hspace=1,
            )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith('linux'):
        os.system(f'eog {fp}&')
    else:
        os.system(f'open {fp}')


#-------------------------------------------------------------------------------

class FixtureFactory:
    name = ''

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.full(size, False, dtype=bool)

    def _get_array_filled(
            size: int,
            start_third: int, # 1 or 2
            density: float, # less than 1
            ) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        count = size * density
        start = int(len(a) * (start_third/3))
        length = len(a) - start
        step = int(length / count)
        fill = np.arange(start, len(a), step)
        a[fill] = True
        return a


    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.name, array


class FFSingleFirstThird(FixtureFactory):
    name = 'single-first_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (1/3))] = True
        return a

class FFSingleSecondThird(FixtureFactory):
    name = 'single-second_third'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size)
        a[int(len(a) * (2/3))] = True
        return a


class FFTenthPostFirstThird(FixtureFactory):
    name = 'tenth_post-first_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=1, density=.1)


class FFTenthPostSecondThird(FixtureFactory):
    name = 'tenth_post-second_third'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=2, density=.1)


def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / StaticFrame: {sf.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    AKFirstTrue,
    NPNonZero,
    NPArgMax,
    )

CLS_FF = (
    FFSingleFirstThird,
    FFSingleSecondThird,
    FFTenthPostFirstThird,
    FFTenthPostSecondThird,
)

NUMBER = 1_000

def run_test():
    records = []
    for size in (100_000, 10_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
            for cls in CLS_PROCESSOR:
                runner = cls(fixture)

                record = [cls.__name__, NUMBER, fixture_label, size]
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
            columns=('name', 'number', 'fixture', 'size', 'time')
            )
    print(f)

    # plot_performance(f)

if __name__ == '__main__':

    run_test()



