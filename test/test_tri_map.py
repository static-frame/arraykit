import unittest

from arraykit import TriMap


class TestUnit(unittest.TestCase):

    def test_tri_map_init_a(self) -> None:
        with self.assertRaises(TypeError):
            tm1 = TriMap()

        tm2 = TriMap(10, 20)
        tm3 = TriMap(10_000, 20_000)
