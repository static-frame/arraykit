import unittest
import numpy as np

from hypothesis import strategies as st
from hypothesis import given
from hypothesis.extra.numpy import arrays

from arraykit import nonzero_1d


class TestUnit(unittest.TestCase):

    #---------------------------------------------------------------------------

    @given(st.lists(st.booleans()))
    def test_nonzero_1d_a(self, v) -> None:
        array = np.array(v, dtype=bool)
        post1 = nonzero_1d(array)
        post2, = np.nonzero(array) # unpack tuple
        self.assertEqual(post1.tolist(), post2.tolist())

    @given(arrays(np.dtype(np.bool_), st.integers(min_value=0, max_value=10_000)))
    def test_nonzero_1d_b(self, array) -> None:
        post1 = nonzero_1d(array)
        post2, = np.nonzero(array) # unpack tuple
        self.assertEqual(post1.tolist(), post2.tolist())




if __name__ == '__main__':
    unittest.main()
