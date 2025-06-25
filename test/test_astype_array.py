import unittest

import numpy as np

from arraykit import astype_array

class TestUnit(unittest.TestCase):

    def test_astype_array_a1(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a1.flags.writeable = False

        a2 = astype_array(a1, np.int64)
        self.assertEqual(id(a1), id(a2))


    def test_astype_array_a2(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a1.flags.writeable = False

        a2 = astype_array(a1, np.float64)
        self.assertNotEqual(id(a1), id(a2))
        self.assertEqual(a2.dtype, np.dtype(np.float64))


    def test_astype_array_a3(self) -> None:
        a1 = np.array([False, True, False])

        a2 = astype_array(a1, np.int8)
        self.assertEqual(a2.dtype, np.dtype(np.int8))
        self.assertFalse(a2.flags.writeable)

    def test_astype_array_b(self) -> None:
        a1 = np.array(['2021', '2024'], dtype=np.datetime64)

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertFalse(a2.flags.writeable)
        self.assertEqual(list(a2), [np.datetime64('2021'), np.datetime64('2024')])

