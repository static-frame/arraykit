import unittest

import numpy as np

from arraykit import astype_array

class TestUnit(unittest.TestCase):

    def test_astype_array_a1(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a1.flags.writeable = False

        a2 = astype_array(a1, np.int64)
        self.assertEqual(id(a1), id(a2))


