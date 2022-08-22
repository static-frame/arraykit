import unittest
import typing as tp
from urllib import request

# import numpy as np

from hypothesis import strategies as st
from hypothesis import given

# from arraykit import delimited_to_arrays
# from arraykit import iterable_str_to_array_1d
from arraykit import delimited_to_arrays


class DelimitedSource(tp.NamedTuple):
    url: str
    delimiter: str
    skip_header: str
    expected_dtypes: tp.Sequence[str]
    expected_row_count: int

SOURCES = {
    'noa-1': DelimitedSource(
        url='https://www.ndbc.noaa.gov/view_text_file.php?filename=46222h2018.txt.gz&dir=data/historical/stdmet/',
        delimiter=' ',
        skip_header=2,
        expected_dtypes=['i','i','i','i','i','i','f','f','f','f','f','i','f','f','f','f','f','f'],
        expected_row_count=16824,
        )
}

def download_and_split(fp: str) -> tp.Iterable[str]:
    with request.urlopen(fp) as response: #pragma: no cover
        file = response.read().decode('utf-8')
        return file.split('\n')


class TestUnit(unittest.TestCase):

    #---------------------------------------------------------------------------

    def test_sources(self):
        for label, ds in SOURCES.items():
            lines = iter(download_and_split(ds.url))
            for _ in range(ds.skip_header):
                next(lines)
            post = delimited_to_arrays(lines,
                    delimiter=ds.delimiter,
                    skipinitialspace=True,
                    axis=1)
            self.assertEqual([a.dtype.kind for a in post], ds.expected_dtypes)
            self.assertTrue(all(len(a) == ds.expected_row_count for a in post))
            # import ipdb; ipdb.set_trace()





if __name__ == '__main__':
    unittest.main()
