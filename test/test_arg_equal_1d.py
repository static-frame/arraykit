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


    def test_arg_equal_1d_int_a(self) -> None:
        a = np.array([4, 0, 4, 0, 5, 8, 0])
        self.assertEqual(arg_equal_1d(a, 0).tolist(), [1, 3, 6])
        self.assertEqual(arg_equal_1d(a, 4).tolist(), [0, 2])

    def test_arg_equal_1d_int_b(self) -> None:
        a = np.arange(100_000)
        self.assertEqual(arg_equal_1d(a, a[99_999]).tolist(), [99_999])
        self.assertEqual(arg_equal_1d(a, a[99_999]).tolist(), [99_999])

    def test_arg_equal_1d_int_c(self) -> None:
        a = np.array([4, 0, 4, 0, 5, 8, 0])
        self.assertEqual(arg_equal_1d(a, 20).tolist(), [])

    def test_arg_equal_1d_int_d(self) -> None:
        a = np.array([4, 0, 4, 0, 5, 8, 0])
        self.assertEqual(arg_equal_1d(a, "foo").tolist(), [])

    def test_arg_equal_1d_int_d(self) -> None:
        a = np.array([4, 0, 4, 0, 5, 8, 0])
        self.assertEqual(arg_equal_1d(a, None).tolist(), [])

    def test_arg_equal_1d_int_e(self) -> None:
        # NOTE: this is consistent with numpy
        a = np.array([4, 0, 4, 0, 5, 8, 0])
        self.assertEqual(arg_equal_1d(a, False).tolist(), [1, 3, 6])

    def test_arg_equal_1d_int_e(self) -> None:
        # NOTE: this is consistent with numpy
        a = np.array([4, 0, 4, 0, 5, 8, 1])
        self.assertEqual(arg_equal_1d(a, True).tolist(), [6])
