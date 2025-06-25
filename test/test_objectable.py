import unittest

import numpy as np

from arraykit import is_objectable_dt64
from arraykit import is_objectable

class TestUnit(unittest.TestCase):

    def test_is_objectable_a1(self) -> None:
        a1 = np.array(['2022-01-04', '1954-04-12'], dtype=np.datetime64)
        self.assertTrue(is_objectable(a1))

    def test_is_objectable_a2(self) -> None:
        a1 = np.array(['10000-01-04', '1954-04-12'], dtype=np.datetime64)
        self.assertFalse(is_objectable(a1))

    def test_is_objectable_b(self) -> None:
        a1 = np.array([10, 20])
        self.assertTrue(is_objectable(a1))

    def test_is_objectable_c(self) -> None:
        a1 = np.array([True, False])
        self.assertTrue(is_objectable(a1))

    def test_is_objectable_d(self) -> None:
        a1 = np.array(['b', 'ccc'])
        self.assertTrue(is_objectable(a1))

    def test_is_objectable_e(self) -> None:
        a1 = np.array(['b', None, False], dtype=object)
        self.assertTrue(is_objectable(a1))


    #---------------------------------------------------------------------------

    def test_is_objectable_dt64_a1(self) -> None:
        a1 = np.array(['2022-01-04', '1954-04-12'], dtype=np.datetime64)
        self.assertTrue(is_objectable_dt64(a1))

    def test_is_objectable_dt64_a2(self) -> None:
        a1 = np.array(['2022-01-04', '', '1954-04-12'], dtype=np.datetime64)
        self.assertTrue(is_objectable_dt64(a1))

    def test_is_objectable_dt64_a3(self) -> None:
        a1 = np.array(['2022-01-04', '1954-04-12', '', ''], dtype=np.datetime64)
        self.assertTrue(is_objectable_dt64(a1))


    def test_is_objectable_dt64_b(self) -> None:
        # years are nevery objectable
        a1 = np.array(['2022', '2023'], dtype=np.datetime64)
        self.assertFalse(is_objectable_dt64(a1))


    def test_is_objectable_dt64_c(self) -> None:
        a1 = np.array(['-120-01-01', '2023-04-05'], dtype='datetime64[m]')
        self.assertFalse(is_objectable_dt64(a1))

    def test_is_objectable_dt64_d(self) -> None:
        a1 = np.array(['2024-01-01', '2023-04-05', '10000-01-01'], dtype='datetime64[s]')
        self.assertFalse(is_objectable_dt64(a1))


    def test_is_objectable_dt64_e(self) -> None:
        a1 = np.array(['2024-01-01', '2023-04-05'], dtype='datetime64[ns]')
        self.assertFalse(is_objectable_dt64(a1))

