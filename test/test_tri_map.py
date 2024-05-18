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


    def test_tri_map_register_one_a(self) -> None:
        tm = TriMap(500, 200)
        tm.register_one(3, 100)

        with self.assertRaises(TypeError):
            tm.register_one(3,)

        with self.assertRaises(TypeError):
            tm.register_one(3, 'a')

        with self.assertRaises(TypeError):
            tm.register_one('b', 'a')

        with self.assertRaises(TypeError):
            tm.register_one()

    def test_tri_map_register_one_b(self) -> None:
        tm = TriMap(2000, 2000)
        for i in range(2000):
            tm.register_one(i, i)

        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 2000, src_connected: 2000, dst_connected: 2000, is_many: false)>')

