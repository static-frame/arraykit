import unittest

import numpy as np

from arraykit import group_ordering
from arraykit import factorize


def offsets_from_codes(codes, size):
    # CSR-style offsets: [0, *cumsum(bincount(codes, minlength=size))]
    counts = np.bincount(codes, minlength=size)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)


class TestUnit(unittest.TestCase):
    # ------------------------------------------------------------------
    # basic behavior

    def test_group_ordering_basic_a(self) -> None:
        codes = np.array([0, 0, 0, 1, 1, 2], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(offsets.tolist(), [0, 3, 5, 6])

    def test_group_ordering_basic_b(self) -> None:
        codes = np.array([2, 0, 0, 2, 1, 1, 0, 0, 3, 0], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [1, 2, 6, 7, 9, 4, 5, 0, 3, 8])
        self.assertEqual(offsets.tolist(), [0,  5,  7,  9, 10])

    def test_group_ordering_interleaved(self) -> None:
        codes = np.array([2, 0, 1, 0, 2, 1], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        # group 0 -> positions 1, 3; group 1 -> 2, 5; group 2 -> 0, 4
        self.assertEqual(offsets.tolist(), [0, 2, 4, 6])
        self.assertEqual(perm[offsets[0]:offsets[1]].tolist(), [1, 3])
        self.assertEqual(perm[offsets[1]:offsets[2]].tolist(), [2, 5])
        self.assertEqual(perm[offsets[2]:offsets[3]].tolist(), [0, 4])

    def test_group_ordering_stability(self) -> None:
        # original positions within each group must stay ascending
        codes = np.array([0, 1, 0, 1, 0, 1], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm[offsets[0]:offsets[1]].tolist(), [0, 2, 4])
        self.assertEqual(perm[offsets[1]:offsets[2]].tolist(), [1, 3, 5])


    # ------------------------------------------------------------------
    # parity against numpy oracle

    def test_group_ordering_parity_argsort(self) -> None:
        rng = np.random.default_rng(0)
        for size in (1, 5, 50, 500):
            codes = rng.integers(0, size, size=10_000).astype(np.intp)
            perm, offsets = group_ordering(codes)
            expected = np.argsort(codes, kind='stable').astype(np.intp)
            self.assertEqual(perm.tolist(), expected.tolist())
            self.assertEqual(
                offsets.tolist(), offsets_from_codes(codes, size).tolist()
            )

    def test_group_ordering_parity_inferred_size(self) -> None:
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 100, size=5_000).astype(np.intp)
        perm, offsets = group_ordering(codes)
        size = int(codes.max()) + 1
        self.assertEqual(len(offsets), size + 1)
        expected = np.argsort(codes, kind='stable').astype(np.intp)
        self.assertEqual(perm.tolist(), expected.tolist())

    # ------------------------------------------------------------------
    # dtype / shape

    def test_group_ordering_dtypes(self) -> None:
        codes = np.array([0, 1, 0, 2], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.dtype, np.dtype(np.intp))
        self.assertEqual(offsets.dtype, np.dtype(np.intp))

    def test_group_ordering_offsets_length(self) -> None:
        codes = np.array([0, 1, 2, 3], dtype=np.intp)
        _, offsets = group_ordering(codes, size=10)
        self.assertEqual(len(offsets), 11)
        self.assertEqual(offsets[-1], 4)

    # ------------------------------------------------------------------
    # size keyword

    def test_group_ordering_size_explicit(self) -> None:
        codes = np.array([0, 0, 1, 1], dtype=np.intp)
        perm, offsets = group_ordering(codes, size=2)
        self.assertEqual(perm.tolist(), [0, 1, 2, 3])
        self.assertEqual(offsets.tolist(), [0, 2, 4])

    def test_group_ordering_size_trailing_empty(self) -> None:
        codes = np.array([0, 0, 1], dtype=np.intp)
        _, offsets = group_ordering(codes, size=4)
        # groups 2 and 3 are empty: offsets[g] == offsets[g+1]
        self.assertEqual(offsets.tolist(), [0, 2, 3, 3, 3])

    def test_group_ordering_size_none(self) -> None:
        codes = np.array([0, 1, 1], dtype=np.intp)
        perm, offsets = group_ordering(codes, size=None)
        self.assertEqual(offsets.tolist(), [0, 1, 3])

    def test_group_ordering_size_is_keyword_only(self) -> None:
        codes = np.array([0, 1], dtype=np.intp)
        with self.assertRaises(TypeError):
            group_ordering(codes, 2)

    # ------------------------------------------------------------------
    # edge cases

    def test_group_ordering_empty(self) -> None:
        codes = np.array([], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [])
        self.assertEqual(offsets.tolist(), [0])

    def test_group_ordering_empty_with_size(self) -> None:
        codes = np.array([], dtype=np.intp)
        perm, offsets = group_ordering(codes, size=3)
        self.assertEqual(perm.tolist(), [])
        self.assertEqual(offsets.tolist(), [0, 0, 0, 0])

    def test_group_ordering_single(self) -> None:
        codes = np.array([0], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [0])
        self.assertEqual(offsets.tolist(), [0, 1])

    def test_group_ordering_single_group(self) -> None:
        codes = np.array([0, 0, 0], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [0, 1, 2])
        self.assertEqual(offsets.tolist(), [0, 3])

    def test_group_ordering_all_distinct(self) -> None:
        codes = np.array([3, 2, 1, 0], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertEqual(perm.tolist(), [3, 2, 1, 0])
        self.assertEqual(offsets.tolist(), [0, 1, 2, 3, 4])

    # ------------------------------------------------------------------
    # validation

    def test_group_ordering_not_array(self) -> None:
        with self.assertRaises(TypeError):
            group_ordering([0, 1, 2])

    def test_group_ordering_2d(self) -> None:
        codes = np.array([[0, 1], [1, 0]], dtype=np.intp)
        with self.assertRaises(ValueError):
            group_ordering(codes)

    def test_group_ordering_wrong_dtype(self) -> None:
        # pick an integer width that differs from intp on this platform
        # (intp is 32-bit on some Windows builds, 64-bit elsewhere)
        wrong = np.int32 if np.dtype(np.intp).itemsize != 4 else np.int64
        codes = np.array([0, 1, 2], dtype=wrong)
        with self.assertRaises(ValueError):
            group_ordering(codes)

    def test_group_ordering_wrong_dtype_float(self) -> None:
        codes = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        with self.assertRaises(ValueError):
            group_ordering(codes)

    def test_group_ordering_non_contiguous(self) -> None:
        codes = np.arange(10, dtype=np.intp)[::2]
        self.assertFalse(codes.flags['C_CONTIGUOUS'])
        with self.assertRaises(ValueError):
            group_ordering(codes)

    def test_group_ordering_negative_code_inferred(self) -> None:
        codes = np.array([0, -1, 1], dtype=np.intp)
        with self.assertRaises(ValueError):
            group_ordering(codes)

    def test_group_ordering_out_of_range(self) -> None:
        codes = np.array([0, 1, 5], dtype=np.intp)
        with self.assertRaises(ValueError):
            group_ordering(codes, size=3)

    def test_group_ordering_negative_size(self) -> None:
        codes = np.array([0, 1], dtype=np.intp)
        with self.assertRaises(ValueError):
            group_ordering(codes, size=-1)

    def test_group_ordering_size_out_of_range_zero(self) -> None:
        codes = np.array([0, 1], dtype=np.intp)
        with self.assertRaises(ValueError):
            group_ordering(codes, size=0)

    def test_group_ordering_infer_overflow(self) -> None:
        # a code at the intp max would overflow when inferring size = c + 1
        codes = np.array([np.iinfo(np.intp).max], dtype=np.intp)
        with self.assertRaises(OverflowError):
            group_ordering(codes)

    def test_group_ordering_size_overflow(self) -> None:
        # an explicit size at the intp max would overflow computing size + 1
        codes = np.array([0, 1], dtype=np.intp)
        with self.assertRaises(OverflowError):
            group_ordering(codes, size=np.iinfo(np.intp).max)

    # ------------------------------------------------------------------
    # immutability

    def test_group_ordering_outputs_immutable(self) -> None:
        codes = np.array([0, 1, 0], dtype=np.intp)
        perm, offsets = group_ordering(codes)
        self.assertFalse(perm.flags.writeable)
        self.assertFalse(offsets.flags.writeable)

    # ------------------------------------------------------------------
    # round-trip with factorize

    def test_group_ordering_with_factorize(self) -> None:
        a = np.array(['b', 'a', 'b', 'c', 'a', 'a'])
        uniques, codes = factorize(a)
        perm, offsets = group_ordering(codes, size=len(uniques))
        ordered = a[perm]
        # each group's slice of the reordered array is constant
        for g in range(len(uniques)):
            segment = ordered[offsets[g]:offsets[g + 1]]
            self.assertTrue((segment == segment[0]).all())


if __name__ == '__main__':
    unittest.main()
