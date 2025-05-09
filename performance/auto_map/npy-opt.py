import os
import sys
import timeit
from typing import NamedTuple
from itertools import repeat

import arraymap
from arraymap import AutoMap
from arraymap import FrozenAutoMap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from fixtures import (
    PayLoad,
    FFInt64,
    FFInt32,
    FFUInt64,
    FFU8,
    FFU16,
    FFDTD,
    FFDTY,
    FFDTns,
    FFDTs,
)


class MapProcessor:
    NAME = ""
    SORT = -1

    def __init__(self, pl: PayLoad):
        self.array = pl.array
        self.list = pl.list
        self.faml = pl.faml
        self.fama = pl.fama
        self.ama = pl.ama
        self.d = pl.d


# -------------------------------------------------------------------------------
class FAMLInstantiate(MapProcessor):
    NAME = "FAM(L): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.list)
        assert len(fam) == len(self.list)


class AMAInstantiate(MapProcessor):
    NAME = "AM(A): instantiate"
    SORT = 0

    def __call__(self):
        fam = AutoMap(self.array)
        assert len(fam) == len(self.list)


class FAMAInstantiate(MapProcessor):
    NAME = "FAM(A): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.array)
        assert len(fam) == len(self.list)


class FAMAtolistInstantiate(MapProcessor):
    NAME = "FAM(Atolist): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.array.tolist())
        assert len(fam) == len(self.list)


class DictInstantiate(MapProcessor):
    NAME = "Dict: instantiate"
    SORT = 0

    def __call__(self):
        d = dict(zip(self.list, range(len(self.list))))
        assert len(d) == len(self.list)


# -------------------------------------------------------------------------------
class FAMLLookup(MapProcessor):
    NAME = "FAM(L): lookup"
    SORT = 0

    def __call__(self):
        m = self.faml
        for k in self.list:
            _ = m[k]


class FAMALookup(MapProcessor):
    NAME = "FAM(A): lookup"
    SORT = 0

    def __call__(self):
        m = self.fama
        for k in self.list:
            _ = m[k]


class DictLookup(MapProcessor):
    NAME = "Dict: lookup"
    SORT = 0

    def __call__(self):
        m = self.d
        for k in self.list:
            _ = m[k]


# -------------------------------------------------------------------------------
class FAMLLookupScalar(MapProcessor):
    NAME = "FAM(L): lookup scalar"
    SORT = 0

    def __call__(self):
        m = self.faml
        for k in self.array:
            _ = m[k]


class FAMALookupScalar(MapProcessor):
    NAME = "FAM(A): lookup scalar"
    SORT = 0

    def __call__(self):
        m = self.fama
        for k in self.array:
            _ = m[k]


class AMALookupScalar(MapProcessor):
    NAME = "AM(A): lookup scalar"
    SORT = 0

    def __call__(self):
        m = self.ama
        for k in self.array:
            _ = m[k]


class DictLookupScalar(MapProcessor):
    NAME = "Dict: lookup scalar"
    SORT = 0

    def __call__(self):
        m = self.d
        for k in self.array:
            _ = m[k]


# -------------------------------------------------------------------------------
class FAMLNotIn(MapProcessor):
    NAME = "FAM(L): not in"
    SORT = 0

    def __call__(self):
        m = self.faml
        for _ in self.list:
            assert None not in m


class FAMANotIn(MapProcessor):
    NAME = "FAM(A): not in"
    SORT = 0

    def __call__(self):
        m = self.fama
        for _ in self.list:
            assert None not in m


class AMANotIn(MapProcessor):
    NAME = "AM(A): not in"
    SORT = 0

    def __call__(self):
        m = self.ama
        for _ in self.array:
            assert None not in m


class DictNotIn(MapProcessor):
    NAME = "Dict: not in"
    SORT = 0

    def __call__(self):
        m = self.d
        for _ in self.list:
            assert None not in m


# -------------------------------------------------------------------------------
class FAMLKeys(MapProcessor):
    NAME = "FAM(L): keys"
    SORT = 0

    def __call__(self):
        for v in self.faml.keys():
            pass


class FAMAKeys(MapProcessor):
    NAME = "FAM(A): keys"
    SORT = 0

    def __call__(self):
        for v in self.fama.keys():
            pass


class DictKeys(MapProcessor):
    NAME = "Dict: keys"
    SORT = 0

    def __call__(self):
        for v in self.d.keys():
            pass


# -------------------------------------------------------------------------------
class FAMLItems(MapProcessor):
    NAME = "FAM(L): items"
    SORT = 0

    def __call__(self):
        for k, v in self.faml.items():
            pass


class FAMAItems(MapProcessor):
    NAME = "FAM(A): items"
    SORT = 0

    def __call__(self):
        for k, v in self.fama.items():
            pass


class DictItems(MapProcessor):
    NAME = "Dict: items"
    SORT = 0

    def __call__(self):
        for k, v in self.d.items():
            pass


def get_versions() -> str:
    import platform

    return f"OS: {platform.system()} / ArrayMap: {arraymap.__version__} / NumPy: {np.__version__}\n"


CLS_FF = (
    # FFInt32,
    FFInt64,
    # FFUInt32,
    # FFUInt64,
    # FFFloat64,
    # FFU8,
    # FFU16,
    # FFU32,
    # FFU64,
    # FFU128,
    # FFS8,
    # FFS16,
    # FFS32,
    # FFS64,
    # FFS128,
    # FFObject,
    FFDTY,
    FFDTD,
    FFDTs,
    FFDTns,
)
FF_ORDER = [f.NAME for f in sorted(CLS_FF, key=lambda ff: ff.SORT)]

# -------------------------------------------------------------------------------
NUMBER = 2


def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f"{seconds * 1e6: .1f} (µs)"
    if seconds < 1e-1:
        return f"{seconds * 1e3: .1f} (ms)"
    return f"{seconds: .1f} (s)"


def plot_performance(frame, suffix: str = ""):
    fixture_total = len(frame["fixture"].unique())
    cat_total = len(frame["size"].unique())
    processor_total = len(frame["cls_processor"].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')
    cmap = plt.get_cmap("plasma")
    color = cmap(np.arange(processor_total) / processor_total)

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby("size")):

        # fixture is the data type fixture
        fixture_data = {fix_label: fix for fix_label, fix in cat.groupby("fixture")}
        for fixture_count, fixture_label in enumerate(FF_ORDER):
            fixture = fixture_data[fixture_label]
            ax = axes[cat_count][fixture_count]

            # set order by cls_processor, i.e., the type of test being done
            fixture["sort"] = [f.SORT for f in fixture["cls_processor"]]
            fixture = fixture.sort_values("sort")

            results = fixture["time"].values.tolist()
            names = [cls.NAME for cls in fixture["cls_processor"]]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            # density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f"{cat_label:.0e}\n{fixture_label}"

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.8)
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
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                axis="x",
                bottom=False,
                labelbottom=False,
            )
            ax.tick_params(
                axis="y",
                length=2,
                width=0.5,
                pad=1,
            )
    fig.set_size_inches(9, 3)  # width, height
    fig.legend(post, names_display, loc="center right", fontsize=6)
    # horizontal, vertical
    fig.text(0.05, 0.96, f"AutoMap {suffix.title()}: {NUMBER} Iterations", fontsize=10)
    fig.text(0.05, 0.90, get_versions(), fontsize=6)

    fp = f"/tmp/arraymap-{suffix}.png"
    plt.subplots_adjust(
        left=0.075,
        bottom=0.05,
        right=0.85,
        top=0.80,
        wspace=1.0,  # width
        hspace=0.5,
    )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith("linux"):
        os.system(f"eog {fp}&")
    else:
        os.system(f"open {fp}")


def run_test(processors, suffix):
    records = []
    for size in (10_000, 100_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
            for cls in processors:
                runner = cls(fixture)

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(f"runner()", globals=locals(), number=NUMBER)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(
        records, columns=("cls_processor", "number", "fixture", "size", "time")
    )
    print(f)
    plot_performance(f, suffix)


if __name__ == "__main__":

    CLS_PROCESSOR = (
        FAMLInstantiate,
        FAMAInstantiate,
        AMAInstantiate,
        DictInstantiate,
        # FAMLLookup,
        # FAMALookup,
        # DictLookup,
        # FAMLLookupScalar,
        # FAMALookupScalar,
        # DictLookupScalar,
        # FAMLNotIn,
        # FAMANotIn,
        # DictNotIn,
        # FAMLKeys,
        # FAMAKeys,
        # DictKeys,
    )

    cls_instantiate = (
        FAMLInstantiate,
        FAMAInstantiate,
        AMAInstantiate,
        DictInstantiate,
    )

    cls_lookup = (
        FAMLLookupScalar,
        FAMALookupScalar,
        AMALookupScalar,
        DictLookupScalar,
        # FAMLNotIn,
        # FAMANotIn,
        # AMANotIn,
        # DictNotIn,
    )

    run_test(cls_instantiate, "instantiate")
    run_test(cls_lookup, "lookup")
