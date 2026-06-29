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
        group = np.array([[1], ['1'], [2]], dtype=object)
        self.assertNotEqual(group[0, 0], group[1, 0])
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
