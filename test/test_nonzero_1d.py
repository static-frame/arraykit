import unittest
import numpy as np

from arraykit import nonzero_1d

class TestUnit(unittest.TestCase):

    def test_nonzero_1d_a(self) -> None:
        self.assertEqual(
            nonzero_1d(np.array([False, True, True, True])).tolist(),
            [1, 2, 3]
        )
        self.assertEqual(
            nonzero_1d(np.array([False, True, False, True])).tolist(),
            [1, 3]
        )
        self.assertEqual(
            nonzero_1d(np.array([False, False, False, False])).tolist(),
            []
        )

    def test_nonzero_1d_b(self) -> None:
        self.assertEqual(
            nonzero_1d(np.array([False, False, False, False, True])).tolist(),
            [4]
        )
        self.assertEqual(
            nonzero_1d(np.array([True, False, False, False, False])).tolist(),
            [0]
        )

    def test_nonzero_1d_c(self) -> None:
        a1 = np.full(100_000, False)
        a1[99_999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [99999])
        a1[999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [999, 99999])

    def test_nonzero_1d_d(self) -> None:
        a1 = np.full(10_000_000, False)
        a1[99_999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [99999])
        a1[999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [999, 99999])
        a1[0] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [0, 999, 99999])


    def test_nonzero_1d_e(self) -> None:
        a1 = np.full(10_000_000, False)
        a1[9_999_999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [9_999_999])
        a1[999] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [999, 9_999_999])
        a1[0] = True
        self.assertEqual(nonzero_1d(a1).tolist(), [0, 999, 9_999_999])