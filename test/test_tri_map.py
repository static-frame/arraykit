import unittest

from arraykit import TriMap


class TestUnit(unittest.TestCase):

    def test_tri_map_init_a(self) -> None:
        with self.assertRaises(TypeError):
            tm1 = TriMap()

        tm2 = TriMap(10, 20)
        tm3 = TriMap(10_000, 20_000)

    def test_tri_map_repr_a(self) -> None:
        tm = TriMap(10_000, 20_000)
        self.assertEqual(str(tm), '<arraykit.TriMap(len: 0, src_connected: 0, dst_connected: 0, is_many: false)>')


