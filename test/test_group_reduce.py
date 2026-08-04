import unittest

import numpy as np
from arraykit import factorize, group_reduce


class TestUnit(unittest.TestCase):
    # ------------------------------------------------------------------
    # basic behavior

    def test_group_reduce_sum_f64(self) -> None:
        codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        values = np.array([1.0, 10.0, 2.0, 100.0, 20.0, 3.0])
        post = group_reduce(codes, 3, values, 'sum')
        self.assertEqual(post.tolist(), [6.0, 30.0, 100.0])
        self.assertEqual(post.dtype, np.dtype(np.float64))

    def test_group_reduce_all_ops_f64(self) -> None:
        codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        values = np.array([1.0, 10.0, 2.0, 100.0, 20.0, 3.0])
        self.assertEqual(group_reduce(codes, 3, values, 'sum').tolist(), [6.0, 30.0, 100.0])
        self.assertEqual(group_reduce(codes, 3, values, 'prod').tolist(), [6.0, 200.0, 100.0])
        self.assertEqual(group_reduce(codes, 3, values, 'min').tolist(), [1.0, 10.0, 100.0])
        self.assertEqual(group_reduce(codes, 3, values, 'max').tolist(), [3.0, 20.0, 100.0])

    def test_group_reduce_all_ops_i64(self) -> None:
        codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        values = np.array([1, 10, 2, 100, 20, 3], dtype=np.int64)
        for op in ('sum', 'prod', 'min', 'max'):
            post = group_reduce(codes, 3, values, op)
            self.assertEqual(post.dtype, np.dtype(np.int64))
        self.assertEqual(group_reduce(codes, 3, values, 'sum').tolist(), [6, 30, 100])
        self.assertEqual(group_reduce(codes, 3, values, 'prod').tolist(), [6, 200, 100])
        self.assertEqual(group_reduce(codes, 3, values, 'min').tolist(), [1, 10, 100])
        self.assertEqual(group_reduce(codes, 3, values, 'max').tolist(), [3, 20, 100])

    def test_group_reduce_count(self) -> None:
        # count returns int64 group sizes regardless of values dtype
        codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        post = group_reduce(codes, 3, np.array([1.0, 2, 3, 4, 5, 6]), 'count')
        self.assertEqual(post.tolist(), [3, 2, 1])
        self.assertEqual(post.dtype, np.dtype(np.int64))
        # count ignores the values dtype entirely
        post = group_reduce(codes, 3, np.array([1, 2, 3, 4, 5, 6], dtype=np.int64), 'count')
        self.assertEqual(post.tolist(), [3, 2, 1])

    def test_group_reduce_nan_propagates(self) -> None:
        # min/max propagate NaN, matching np.min/np.max (not the nan-skipping variants)
        codes = np.array([0, 1, 0, 2, 1, 0], dtype=np.intp)
        values = np.array([1.0, np.nan, 2.0, 5.0, np.nan, 3.0])
        mx = group_reduce(codes, 3, values, 'max')
        mn = group_reduce(codes, 3, values, 'min')
        self.assertEqual(mx[0], 3.0)
        self.assertTrue(np.isnan(mx[1]))
        self.assertEqual(mx[2], 5.0)
        self.assertEqual(mn[0], 1.0)
        self.assertTrue(np.isnan(mn[1]))
        # sum also propagates NaN
        s = group_reduce(codes, 3, values, 'sum')
        self.assertTrue(np.isnan(s[1]))

    def test_group_reduce_single_group(self) -> None:
        codes = np.array([0, 0, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0])
        self.assertEqual(group_reduce(codes, 1, values, 'sum').tolist(), [6.0])

    def test_group_reduce_empty(self) -> None:
        codes = np.array([], dtype=np.intp)
        values = np.array([], dtype=np.float64)
        self.assertEqual(group_reduce(codes, 0, values, 'sum').tolist(), [])

    def test_group_reduce_outputs_immutable(self) -> None:
        codes = np.array([0, 1, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0])
        for op in ('sum', 'prod', 'min', 'max', 'count'):
            post = group_reduce(codes, 2, values, op)
            self.assertFalse(post.flags.writeable)

    # ------------------------------------------------------------------
    # equivalence to a per-group numpy reduction

    def test_group_reduce_equivalence_f64(self) -> None:
        rng = np.random.RandomState(0)
        for _ in range(20):
            size = int(rng.randint(1, 40))
            n = int(rng.randint(size, size + 500))
            codes = rng.randint(0, size, n).astype(np.intp)
            values = rng.rand(n) * 100
            for op, npf in (('sum', np.sum), ('min', np.min), ('max', np.max)):
                got = group_reduce(codes, size, values, op)
                for g in range(size):
                    mask = codes == g
                    if np.any(mask):  # real usage (factorize) has no empty groups
                        self.assertTrue(np.isclose(got[g], npf(values[mask])), op)

    def test_group_reduce_equivalence_i64(self) -> None:
        rng = np.random.RandomState(1)
        for _ in range(20):
            size = int(rng.randint(1, 40))
            n = int(rng.randint(size, size + 500))
            codes = rng.randint(0, size, n).astype(np.intp)
            values = rng.randint(-1000, 1000, n).astype(np.int64)
            for op, npf in (('sum', np.sum), ('min', np.min), ('max', np.max)):
                got = group_reduce(codes, size, values, op)
                for g in range(size):
                    mask = codes == g
                    if np.any(mask):
                        self.assertEqual(got[g], npf(values[mask]), op)

    def test_group_reduce_with_factorize(self) -> None:
        # the intended pipeline: factorize(sort=True) -> group_reduce
        key = np.array([30, 10, 20, 10, 30, 20, 10])
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        uniques, codes = factorize(key, sort=True)
        self.assertEqual(uniques.tolist(), [10, 20, 30])
        post = group_reduce(codes, len(uniques), values, 'sum')
        # group 10 -> 2+4+7=13; group 20 -> 3+6=9; group 30 -> 1+5=6
        self.assertEqual(post.tolist(), [13.0, 9.0, 6.0])

    # ------------------------------------------------------------------
    # errors

    def test_group_reduce_errors(self) -> None:
        codes = np.array([0, 1, 0], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):  # unknown op
            group_reduce(codes, 2, values, 'median')
        with self.assertRaises(ValueError):  # length mismatch
            group_reduce(codes, 2, np.array([1.0, 2.0]), 'sum')
        with self.assertRaises(ValueError):  # code out of range
            group_reduce(np.array([0, 5], dtype=np.intp), 2, np.array([1.0, 2.0]), 'sum')
        with self.assertRaises(ValueError):  # negative code
            group_reduce(np.array([0, -1], dtype=np.intp), 2, np.array([1.0, 2.0]), 'sum')
        with self.assertRaises(ValueError):  # codes wrong dtype
            group_reduce(np.array([0, 1], dtype=np.int32), 2, np.array([1.0, 2.0]), 'sum')
        with self.assertRaises(ValueError):  # values unsupported dtype
            group_reduce(codes, 2, np.array([1, 2, 3], dtype=np.int32), 'sum')
        with self.assertRaises(ValueError):  # negative size
            group_reduce(codes, -1, values, 'sum')
        with self.assertRaises(ValueError):  # 2d codes
            group_reduce(
                np.array([[0, 1]], dtype=np.intp), 2, np.array([[1.0, 2.0]]), 'sum'
            )
        with self.assertRaises(TypeError):  # not an array
            group_reduce([0, 1, 0], 2, values, 'sum')


if __name__ == '__main__':
    unittest.main()
