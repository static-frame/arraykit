


import timeit
import numpy as np

from performance.reference.util import immutable_filter as immutable_filter_ref
from arraykit import immutable_filter as immutable_filter_ak

class Perf:
    pass

class ImmutableFilter(Perf):

    FUNCTIONS = ('main',)

    def pre(self):
        self.array = np.arange(100)

class ImmutableFilterAK(ImmutableFilter):

    def main(self):
        immutable_filter_ak(self.array)

class ImmutableFilterREF(ImmutableFilter):

    def main(self):
        immutable_filter_ref(self.array)



# def performance(
#         module: types.ModuleType,
#         cls: tp.Type[PerfTest]
#         ) -> PerformanceRecord:
#     #row = []
#     row = {}
#     row['name'] = cls.__name__
#     row['iterations'] = cls.NUMBER
#     for f in cls.FUNCTION_NAMES:
#         if hasattr(cls, f):
#             result = timeit.timeit(cls.__name__ + '.' + f + '()',
#                     globals=vars(module),
#                     number=cls.NUMBER)
#             row[f] = result
#         else:
#             row[f] = np.nan
#     return row

def main():
    for cls_perf in Perf.__subclasses__(): # only get one level
        cls_map = {}
        for cls_runner in cls_perf.__subclasses__():
            if cls_runner.__name__.endswith('AK'):
                cls_map['ak'] = cls_runner
            elif cls_runner.__name__.endswith('REF'):
                cls_map['ref'] = cls_runner
        for func_attr in cls_perf.FUNCTIONS:
            for key, cls_runner in cls_map.items():
                runner = cls_runner()
                runner.pre()
                print(runner, func_attr)
                f = getattr(runner, func_attr)
                f()


if __name__ == '__main__':
    main()