import unittest
import numpy as np

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

    def test_tri_map_register_one_c(self) -> None:
        tm = TriMap(20, 30)
        self.assertFalse(tm.is_many())

        with self.assertRaises(ValueError):
            tm.register_one(22, 0)
        with self.assertRaises(ValueError):
            tm.register_one(0, 30)

        tm.register_one(19, 29)
        tm.register_one(18, 29)
        self.assertTrue(tm.is_many())


    def test_tri_map_src_no_fill_a(self) -> None:
        tm = TriMap(3, 3)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, -1)
        self.assertTrue(tm.src_no_fill())
        self.assertFalse(tm.dst_no_fill())

    def test_tri_map_register_unmatched_dst_a(self) -> None:
        tm = TriMap(10, 8)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_unmatched_dst()

        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 8, src_connected: 3, dst_connected: 8, is_many: false)>')

    def test_tri_map_register_many_a(self) -> None:
        tm = TriMap(100, 50)
        tm.register_many(3, np.array([2,5,8]))

        with self.assertRaises(TypeError):
            tm.register_many("foo", np.array([2,5,8]))

        with self.assertRaises(TypeError):
            tm.register_many(3, [3, 2])

    def test_tri_map_register_many_b(self) -> None:
        tm = TriMap(100, 50)
        with self.assertRaises(ValueError):
            tm.register_many(3, np.array([2, 5, 8], dtype=float))

    def test_tri_map_register_many_c(self) -> None:
        tm = TriMap(100, 50)
        tm.register_many(3, np.array([2, 5, 8]))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 3, src_connected: 3, dst_connected: 3, is_many: true)>')

    def test_tri_map_register_many_d1(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int32))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_connected: 200, dst_connected: 200, is_many: true)>')

    def test_tri_map_register_many_d2(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int64))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_connected: 200, dst_connected: 200, is_many: true)>')

