import unittest

import numpy as np

from arraykit import transition_slices_from_group


def slices_to_pairs(slices):
    return [(s.start, s.stop, s.step) for s in slices]


class TestUnit(unittest.TestCase):

    def test_transition_slices_from_group_1d_a(self) -> None:
        group = np.array([10, 10, 10, 20, 20, 30])
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertFalse(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 3, None), (3, 5, None), (5, None, None)],
        )

    def test_transition_slices_from_group_1d_object_a(self) -> None:
        group = np.array(['a', 'a', 'b', 'b', 'c'], dtype=object)
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertFalse(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, 4, None), (4, None, None)],
        )

    def test_transition_slices_from_group_2d_a(self) -> None:
        group = np.array(
            [
                [1, 2],
                [1, 2],
                [1, 3],
                [1, 3],
                [2, 4],
            ]
        )
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertTrue(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, 4, None), (4, None, None)],
        )

    def test_transition_slices_from_group_2d_mixed_types_a(self) -> None:
        # object rows are compared by value (rich equality), consistent with the
        # 1d object path: 1, '1', and 2 are all distinct, so no rows merge
        group = np.array([[1], ['1'], [2]], dtype=object)
        self.assertNotEqual(group[0, 0], group[1, 0])
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertTrue(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 1, None), (1, 2, None), (2, None, None)],
        )

    def test_transition_slices_from_group_2d_object_value_equality(self) -> None:
        # equal values across rows merge even when not the same object;
        # 1 == 1.0 by value, so rows 0 and 1 form one group
        group = np.array([[1, 2], [1.0, 2], [1, 3]], dtype=object)
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertTrue(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, None, None)],
        )

    def test_transition_slices_from_group_2d_non_contiguous_a(self) -> None:
        group = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
            ],
            dtype=np.int64,
        ).T
        self.assertFalse(group.flags.c_contiguous)
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertTrue(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, None, None)],
        )

    def test_transition_slices_from_group_empty_a(self) -> None:
        slices, group_to_tuple = transition_slices_from_group(np.array([], dtype=int))
        self.assertFalse(group_to_tuple)
        self.assertEqual(slices_to_pairs(slices), [])

    def test_transition_slices_from_group_invalid_ndim_a(self) -> None:
        with self.assertRaises(NotImplementedError):
            transition_slices_from_group(np.arange(8).reshape(2, 2, 2))

    # ---------------------------------------------------------------------------
    # dtype-specific 1d paths (typed fast scans)

    def test_transition_slices_from_group_1d_dtypes(self) -> None:
        # the integral fast paths dispatch by width; cover each
        for dtype in (
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float32, np.float64,
        ):
            group = np.array([10, 10, 10, 20, 20, 30], dtype=dtype)
            slices, group_to_tuple = transition_slices_from_group(group)
            self.assertFalse(group_to_tuple)
            self.assertEqual(
                slices_to_pairs(slices),
                [(0, 3, None), (3, 5, None), (5, None, None)],
                dtype,
            )

    def test_transition_slices_from_group_1d_bool(self) -> None:
        group = np.array([True, True, False, False, True])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, 4, None), (4, None, None)],
        )

    def test_transition_slices_from_group_1d_float_nan(self) -> None:
        # NaN != NaN, so each adjacent NaN is its own group (matches numpy `!=`)
        group = np.array([1.0, np.nan, np.nan, 2.0])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 1, None), (1, 2, None), (2, 3, None), (3, None, None)],
        )

    def test_transition_slices_from_group_1d_float_signed_zero(self) -> None:
        # +0.0 == -0.0, so no transition (a byte compare would wrongly split)
        group = np.array([0.0, -0.0, 0.0])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(slices_to_pairs(slices), [(0, None, None)])

    def test_transition_slices_from_group_1d_datetime_nat(self) -> None:
        # NaT is unequal to everything, including itself
        d = np.datetime64('2020-01-01')
        nat = np.datetime64('NaT')
        group = np.array([d, nat, nat, d])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 1, None), (1, 2, None), (2, 3, None), (3, None, None)],
        )

    def test_transition_slices_from_group_1d_timedelta_nat(self) -> None:
        t = np.timedelta64(5, 'D')
        nat = np.timedelta64('NaT')
        group = np.array([t, t, nat, nat])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, 3, None), (3, None, None)],
        )

    def test_transition_slices_from_group_1d_str(self) -> None:
        group = np.array(['a', 'a', 'bb', 'b', 'b'])
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, 3, None), (3, None, None)],
        )

    def test_transition_slices_from_group_1d_non_contiguous(self) -> None:
        # strided 1d view exercises the generic fallback (stride != itemsize)
        # underlying buffer is [1,1,2,2,3,3]; the [::2] view is [1,2,3], so a
        # naive contiguous read would give the wrong groups
        group = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)[::2]
        self.assertFalse(group.flags.c_contiguous)
        slices, _ = transition_slices_from_group(group)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 1, None), (1, 2, None), (2, None, None)],
        )

    def test_transition_slices_from_group_2d_float_nan(self) -> None:
        # 2d compares rows bytewise (via void view), so identical-bit NaN rows
        # are equal -- unlike the 1d float path
        group = np.array([[np.nan], [np.nan], [1.0]])
        slices, group_to_tuple = transition_slices_from_group(group)
        self.assertTrue(group_to_tuple)
        self.assertEqual(
            slices_to_pairs(slices),
            [(0, 2, None), (2, None, None)],
        )
