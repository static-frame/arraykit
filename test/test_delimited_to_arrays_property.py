import unittest
import numpy as np

from hypothesis import strategies as st
from hypothesis import given

# from arraykit import delimited_to_arrays
# from arraykit import iterable_str_to_array_1d
from arraykit import delimited_to_arrays


class TestUnit(unittest.TestCase):

    #---------------------------------------------------------------------------

    @given(st.lists(st.integers(min_value=-9223372036854775808, max_value=9223372036854775807), min_size=1, max_size=40))
    def test_delimited_to_arrays_parse_a(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['i', 'i'])
        self.assertEqual(post[0].tolist(), v)

    @given(st.lists(st.booleans(), min_size=1, max_size=40))
    def test_delimited_to_arrays_parse_b(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['b', 'b'])
        self.assertEqual(post[0].tolist(), v)

    @given(st.lists(st.floats(), min_size=1, max_size=40))
    def test_delimited_to_arrays_parse_c(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['f', 'f'])
        # need to handle NaNs

    @given(st.lists(st.floats(allow_nan=False), min_size=1, max_size=40))
    def test_delimited_to_arrays_parse_d(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['f', 'f'])
        # no NaNs
        self.assertTrue(np.allclose(post[0], v, equal_nan=True))

    @given(st.lists(st.complex_numbers(), min_size=2, max_size=40))
    def test_delimited_to_arrays_parse_e(self, v) -> None:
        msg = [f'{x},{x}' for x in v]
        post = delimited_to_arrays(msg, dtypes=None, axis=1)
        self.assertEqual([a.dtype.kind for a in post],
                ['c', 'c'])


if __name__ == '__main__':
    unittest.main()
