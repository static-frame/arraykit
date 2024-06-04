import unittest
import numpy as np

from arraykit import nonzero_1d

class TestUnit(unittest.TestCase):

    def test_nonzero_1d_a1(self) -> None:
        self.assertEqual(
            nonzero_1d(np.array([], dtype=bool)).tolist(),
            []
        )

    def test_nonzero_1d_a2(self) -> None:
        self.assertEqual(
            nonzero_1d(np.array([False], dtype=bool)).tolist(),
            []
        )

    def test_nonzero_1d_a3(self) -> None:
        self.assertEqual(
            nonzero_1d(np.array([True], dtype=bool)).tolist(),
            [0]
        )

    def test_nonzero_1d_a4(self) -> None:
        with self.assertRaises(ValueError):
            nonzero_1d(np.array([0, 1]))

        with self.assertRaises(ValueError):
            nonzero_1d(np.array(['a', 'bbb']))

    def test_nonzero_1d_a5(self) -> None:
        with self.assertRaises(ValueError):
            nonzero_1d(np.arange(10).reshape(5, 2).astype(bool))


    def test_nonzero_1d_b1(self) -> None:
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

    def test_nonzero_1d_b2(self) -> None:
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

    def test_nonzero_1d_f(self) -> None:
        # non-contiguous
        a1 = np.arange(40).reshape(10, 4) % 3 == 0
        a2 = a1[:, 3]
        self.assertEqual(nonzero_1d(a2).tolist(), [0, 3, 6, 9])

        a3 = a1[:, 1]
        self.assertEqual(nonzero_1d(a3).tolist(), [2, 5, 8])

    def test_nonzero_1d_g(self) -> None:
        a1 = np.arange(20).reshape(4, 5) % 3 == 0
        a2 = a1[:, 4]
        # array([False,  True, False, False])
        self.assertEqual(nonzero_1d(a2).tolist(), [1])
