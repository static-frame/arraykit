import unittest

import numpy as np

from arraykit import is_objectable_dt64

class TestUnit(unittest.TestCase):

    def test_is_objectable_dt64_a(self) -> None:
        a1 = np.array(['2022-01-04', '1954-04-12'], dtype=np.datetime64)
        self.assertFalse(is_objectable_dt64(a1))



