import unittest
import numpy as np

from arraykit import arg_equal_1d

class TestUnit(unittest.TestCase):

    def test_arg_equal_1d_a1(self) -> None:
        a = np.arange(6).reshape(2, 3)
        with self.assertRaises(TypeError):
            arg_equal_1d(a)

        with self.assertRaises(ValueError):
            arg_equal_1d(a, None)
