import sys
import unittest

import numpy as np
from arraykit import fill_directional


def reference(array, target, *, forward=True, axis=0, limit=0):
    """Pure-Python reference: carry the last (forward) or next (backward) non-target
    value across target runs, capped at `limit` consecutive fills."""
    out = array.copy()

    def fill_lane(vals, tgt):
        last = None
        count = 0
        order = range(len(vals)) if forward else range(len(vals) - 1, -1, -1)
        for i in order:
            if tgt[i]:
                if last is not None and (limit == 0 or count < limit):
                    vals[i] = last
                    count += 1
            else:
                last = vals[i]
                count = 0

    if array.ndim == 1:
        fill_lane(out, target)
    elif axis == 0:
        for c in range(out.shape[1]):
            col = out[:, c].copy()
            fill_lane(col, target[:, c])
            out[:, c] = col
    else:
        for r in range(out.shape[0]):
            row = out[r, :].copy()
            fill_lane(row, target[r, :])
            out[r, :] = row
    return out


class TestUnit(unittest.TestCase):
    # ------------------------------------------------------------------
    # basic 1D behavior

    def test_forward_1d(self) -> None:
        a = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        t = np.isnan(a)
        post = fill_directional(a, t)
        self.assertEqual(post.tolist(), [1.0, 1.0, 1.0, 4.0, 4.0])

    def test_backward_1d(self) -> None:
        a = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        t = np.isnan(a)
        post = fill_directional(a, t, forward=False)
        # trailing target has no source and stays NaN
        self.assertTrue(
            np.array_equal(post, [1.0, 4.0, 4.0, 4.0, np.nan], equal_nan=True)
        )

    def test_leading_target_unchanged(self) -> None:
        a = np.array([np.nan, np.nan, 3.0, np.nan])
        post = fill_directional(a, np.isnan(a))
        self.assertTrue(
            np.array_equal(post, [np.nan, np.nan, 3.0, 3.0], equal_nan=True)
        )

    def test_all_target(self) -> None:
        a = np.array([np.nan, np.nan, np.nan])
        post = fill_directional(a, np.isnan(a))
        self.assertTrue(np.array_equal(post, a, equal_nan=True))

    def test_no_target(self) -> None:
        a = np.arange(5.0)
        post = fill_directional(a, np.zeros(5, dtype=bool))
        self.assertEqual(post.tolist(), a.tolist())

    def test_empty(self) -> None:
        a = np.array([], dtype=float)
        post = fill_directional(a, np.array([], dtype=bool))
        self.assertEqual(len(post), 0)

    # ------------------------------------------------------------------
    # limit

    def test_limit_forward(self) -> None:
        a = np.array([1.0, np.nan, np.nan, np.nan, 5.0])
        post = fill_directional(a, np.isnan(a), limit=1)
        self.assertTrue(
            np.array_equal(post, [1.0, 1.0, np.nan, np.nan, 5.0], equal_nan=True)
        )

    def test_limit_resets_per_run(self) -> None:
        a = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])
        post = fill_directional(a, np.isnan(a), limit=1)
        self.assertTrue(
            np.array_equal(post, [1.0, 1.0, 3.0, 3.0, np.nan, 6.0], equal_nan=True)
        )

    # ------------------------------------------------------------------
    # dtypes

    def test_int(self) -> None:
        a = np.array([5, 0, 0, 8, 0])
        t = np.array([False, True, True, False, True])
        self.assertEqual(fill_directional(a, t).tolist(), [5, 5, 5, 8, 8])

    def test_object(self) -> None:
        a = np.array([None, 'a', None, 'b', None, None], dtype=object)
        t = np.array([v is None for v in a])
        post = fill_directional(a, t)
        self.assertEqual(post.tolist(), [None, 'a', 'a', 'b', 'b', 'b'])

    def test_object_no_refcount_leak(self) -> None:
        marker = object()
        a = np.array([marker, None, None, marker, None], dtype=object)
        t = np.array([False, True, True, False, True])
        base = sys.getrefcount(marker)
        for _ in range(2000):
            post = fill_directional(a, t)
            del post
        self.assertEqual(sys.getrefcount(marker), base)

    def test_datetime(self) -> None:
        a = np.array(['2020-01-01', 'NaT', '2020-01-03'], dtype='datetime64[D]')
        t = np.isnat(a)
        post = fill_directional(a, t)
        self.assertEqual(
            post.tolist(),
            np.array(
                ['2020-01-01', '2020-01-01', '2020-01-03'], dtype='datetime64[D]'
            ).tolist(),
        )

    # ------------------------------------------------------------------
    # 2D

    def test_2d_axis0(self) -> None:
        a = np.array([[1.0, np.nan], [np.nan, 5.0], [3.0, np.nan]])
        post = fill_directional(a, np.isnan(a), axis=0)
        self.assertTrue(
            np.array_equal(post, [[1.0, np.nan], [1.0, 5.0], [3.0, 5.0]], equal_nan=True)
        )

    def test_2d_axis1(self) -> None:
        a = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 6.0]])
        post = fill_directional(a, np.isnan(a), axis=1)
        self.assertTrue(
            np.array_equal(
                post, [[1.0, 1.0, 3.0], [np.nan, np.nan, 6.0]], equal_nan=True
            )
        )

    # ------------------------------------------------------------------
    # immutability

    def test_immutable(self) -> None:
        a = np.array([1.0, np.nan, 3.0])
        post = fill_directional(a, np.isnan(a))
        self.assertFalse(post.flags.writeable)

    # ------------------------------------------------------------------
    # differential vs reference across a matrix of parameters

    def test_matches_reference(self) -> None:
        rng = np.random.default_rng(42)
        arrays_1d = (
            np.round(rng.random(300), 2),
            np.where(rng.random(300) < 0.5, np.nan, rng.random(300)),
            rng.integers(0, 4, 300),
        )
        for a in arrays_1d:
            t = np.isnan(a) if a.dtype.kind == 'f' else (a % 3 == 0)
            for forward in (True, False):
                for limit in (0, 1, 3):
                    post = fill_directional(a, t, forward=forward, limit=limit)
                    exp = reference(a, t, forward=forward, limit=limit)
                    self.assertTrue(
                        np.array_equal(post, exp, equal_nan=(a.dtype.kind == 'f'))
                    )
        for shape in ((8, 5), (5, 8), (1, 6), (6, 1), (10, 10)):
            a = np.where(rng.random(shape) < 0.35, np.nan, rng.random(shape))
            t = np.isnan(a)
            for axis in (0, 1):
                for forward in (True, False):
                    for limit in (0, 2):
                        post = fill_directional(
                            a, t, forward=forward, axis=axis, limit=limit
                        )
                        exp = reference(
                            a, t, forward=forward, axis=axis, limit=limit
                        )
                        self.assertTrue(np.array_equal(post, exp, equal_nan=True))

    # ------------------------------------------------------------------
    # errors

    def test_error_target_not_bool(self) -> None:
        a = np.arange(3.0)
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros(3, dtype=int))

    def test_error_shape_mismatch(self) -> None:
        a = np.arange(3.0)
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros(4, dtype=bool))

    def test_error_ndim_mismatch(self) -> None:
        a = np.arange(6.0).reshape(2, 3)
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros(6, dtype=bool))

    def test_error_bad_axis(self) -> None:
        a = np.arange(6.0).reshape(2, 3)
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros((2, 3), dtype=bool), axis=2)

    def test_error_negative_limit(self) -> None:
        a = np.arange(3.0)
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros(3, dtype=bool), limit=-1)

    def test_error_3d(self) -> None:
        a = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            fill_directional(a, np.zeros((2, 2, 2), dtype=bool))


if __name__ == '__main__':
    unittest.main()
