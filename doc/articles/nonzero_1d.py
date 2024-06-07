


import os
import sys
import timeit
import typing as tp

from arraykit import nonzero_1d
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
class AKNonZero(ArrayProcessor):
    NAME = 'ak.nonzero_1d()'
    SORT = 0

    def __call__(self):
        _ = nonzero_1d(self.array)

class NPNonZero(ArrayProcessor):
    NAME = 'np.nonzero()'
    SORT = 1

    def __call__(self):
        _ = np.nonzero(self.array)[0]

class NPNonZeroInt64Convert(ArrayProcessor):
    NAME = 'np.nonzero()\n.astype(np.int64)'
    SORT = 3

    def __call__(self):
        _ = np.nonzero(self.array)[0].astype(np.int64)


#-------------------------------------------------------------------------------
NUMBER = 10

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
        for fixture_count, (fixture_label, fixture) in enumerate(fixtures.items()):
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

    fig.set_size_inches(10, 4) # width, height
    fig.legend(post, names_display, loc='center right', fontsize=6)
    # horizontal, vertical
    fig.text(.05, .96, f'nonzero_1d() Performance: {NUMBER} Iterations', fontsize=10)
    fig.text(.05, .90, get_versions(), fontsize=6)

    fp = '/tmp/nonzero.png'
    plt.subplots_adjust(
            left=0.05,
            bottom=0.05,
            right=0.85,
            top=0.85,
            wspace=0.9, # width
            hspace=0.0,
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
    def get_array(size: int, contiguous: bool = True) -> np.ndarray:
        if contiguous:
            return np.full(size, False, dtype=bool)
        # take every other value to force non-contigous data
        return np.full(size * 2, False, dtype=bool)[::2]

    def _get_array_filled(
            size: int,
            start_third: int, #0, 1 or 2
            density: float, # less than 1
            contiguous: bool,
            ) -> np.ndarray:
        a = FixtureFactory.get_array(size, contiguous)
        count = size * density
        start = int(len(a) * (start_third/3))
        length = len(a) - start
        step = max(int(length / count), 1)
        fill = np.arange(start, len(a), step)
        a[fill] = True
        return a

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array

    DENSITY_TO_DISPLAY = {
        'single-c': '1 True C',
        'quarter-c': '25% True C',
        'half-c': '50% True C',
        'full-c': '100% True C',
        'single-nc': '1 True NC',
        'quarter-nc': '25% True NC',
        'half-nc': '50% True NC',
        'full-nc': '100% True NC',
    }

    # POSITION_TO_DISPLAY = {
    #     'first_third': 'Fill 1/3 to End',
    #     'second_third': 'Fill 2/3 to End',
    # }


class FFSingle(FixtureFactory):
    NAME = 'single-c'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, contiguous=True)
        a[len(a) // 2] = True
        return a

class FFSingleNC(FixtureFactory):
    NAME = 'single-nc'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        a = FixtureFactory.get_array(size, contiguous=False)
        a[len(a) // 2] = True
        return a

class FFQuarter(FixtureFactory):
    NAME = 'quarter-c'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=0.25, contiguous=True)

class FFQuarterNC(FixtureFactory):
    NAME = 'quarter-nc'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=0.25, contiguous=False)


class FFHalf(FixtureFactory):
    NAME = 'half-c'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=0.5, contiguous=True)

class FFHalfNC(FixtureFactory):
    NAME = 'half-nc'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=0.5, contiguous=False)



class FFFull(FixtureFactory):
    NAME = 'full-c'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=1, contiguous=True)


class FFFullNC(FixtureFactory):
    NAME = 'full-nc'

    @classmethod
    def get_array(cls, size: int) -> np.ndarray:
        return cls._get_array_filled(size, start_third=0, density=1, contiguous=False)


def get_versions() -> str:
    import platform
    return f'OS: {platform.system()} / ArrayKit: {ak.__version__} / NumPy: {np.__version__}\n'


CLS_PROCESSOR = (
    AKNonZero,
    NPNonZero,
    NPNonZeroInt64Convert,
    )

CLS_FF = (
    FFSingle,
    FFSingleNC,
    FFQuarter,
    FFQuarterNC,
    FFHalf,
    FFHalfNC,
    FFFull,
    FFFullNC,

)


def run_test():
    records = []
    for size in (10_000, 100_000, 1_000_000, 10_000_000):
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



