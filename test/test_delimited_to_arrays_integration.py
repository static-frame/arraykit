import unittest
import typing as tp
from urllib import request
import hashlib
import os

from arraykit import delimited_to_arrays

# NOTE: this is not implemented as an automated test as it downloads data on the fly

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
    fn = hashlib.sha224(bytes(fp, encoding='utf8')).hexdigest() + '.txt'
    if os.path.exists('/tmp'):
        fp_destination = os.path.join('/tmp', fn)
    else:
        fp_destination = None

    if fp_destination and os.path.exists(fp_destination):
        print(f'using {fp_destination}')
        with open(fp_destination) as f:
            return f.readlines()

    with request.urlopen(fp) as response: #pragma: no cover
        contents = response.read().decode('utf-8')
        if fp_destination:
            print(f'writing {fp_destination}')
            with open(fp_destination, 'w') as f:
                f.write(contents)
        return contents.split('\n')


def process_sources():
    for label, ds in SOURCES.items():
        lines = iter(download_and_split(ds.url))
        for _ in range(ds.skip_header):
            next(lines)
        print(label)
        post = delimited_to_arrays(lines,
                delimiter=ds.delimiter,
                skipinitialspace=True,
                axis=1)
        assert [a.dtype.kind for a in post] == ds.expected_dtypes
        assert all(len(a) == ds.expected_row_count for a in post)


if __name__ == '__main__':
    process_sources()
