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
        tm.register_many(3, np.array([2,5,8], dtype=np.int64))

        with self.assertRaises(TypeError):
            tm.register_many("foo", np.array([2,5,8]))

        with self.assertRaises(TypeError):
            tm.register_many(3, [3, 2])

        with self.assertRaises(ValueError):
            tm.register_many(3, np.array([2,5,8], dtype=np.int32))

    def test_tri_map_register_many_b(self) -> None:
        tm = TriMap(100, 50)
        with self.assertRaises(ValueError):
            tm.register_many(3, np.array([2, 5, 8], dtype=float))

    def test_tri_map_register_many_c(self) -> None:
        tm = TriMap(100, 50)
        tm.register_many(3, np.array([2, 5, 8], dtype=np.int64))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 3, src_connected: 3, dst_connected: 3, is_many: true)>')

    def test_tri_map_register_many_d1(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int64))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_connected: 200, dst_connected: 200, is_many: true)>')

    def test_tri_map_register_many_d2(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int64))
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_connected: 200, dst_connected: 200, is_many: true)>')

    #---------------------------------------------------------------------------

    def test_tri_map_map_src_no_fill_a(self) -> None:
        src = np.array([10, 20, 30, 40], dtype=np.int64)
        dst = np.array([30, 30, 40, 40], dtype=np.int64)

        tm = TriMap(4, 4)
        tm.register_one(0, -1)
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 1], dtype=np.int64))
        tm.register_many(3, np.array([2, 3], dtype=np.int64))

        post_src = tm.map_src_no_fill(src)
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [10, 20, 30, 30, 40, 40])

        post_dst = tm.map_dst_fill(dst, -1, np.dtype(np.int64))
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [-1, -1, 30, 30, 40, 40])

    def test_tri_map_map_src_no_fill_b(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd'])
        dst = np.array(['cc', 'cc', 'dddd', 'dddd'])

        tm = TriMap(4, 4)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_one(3, 3)

        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), ['a', 'bbb', 'cc', 'dddd'])

    def test_tri_map_map_src_no_fill_c(self) -> None:
        src = np.array(['aaaaa', 'bbb', 'cc', 'dddd'])

        tm = TriMap(4, 4)
        tm.register_many(0, np.array([1, 3], dtype=np.int64))
        tm.register_many(1, np.array([0, 2], dtype=np.int64))

        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), ['aaaaa', 'aaaaa', 'bbb', 'bbb'])

    def test_tri_map_map_src_no_fill_c(self) -> None:
        src = np.array([None, 'bbb', 3, False])

        tm = TriMap(4, 4)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_one(3, 3)
        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), [None, 'bbb', 3, False])

    def test_tri_map_map_src_no_fill_d(self) -> None:
        src = np.array([None, 'bbb', 3, False])

        tm = TriMap(4, 4)
        tm.register_many(0, np.array([1, 3], dtype=np.int64))
        tm.register_many(1, np.array([0, 2], dtype=np.int64))

        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), [None, None, 'bbb', 'bbb'])


    #---------------------------------------------------------------------------

    def test_tri_map_map_src_fill_a(self) -> None:
        src = np.array([10, 20, 30, 40], dtype=np.int64)
        dst = np.array([0, 20, 30, 50], dtype=np.int64)

        tm = TriMap(4, 4)
        tm.register_one(0, -1)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_one(3, -1)
        tm.register_unmatched_dst()

        post = tm.map_src_fill(src, -1, np.dtype(np.int64))
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), [10, 20, 30, 40, -1, -1])


    def test_tri_map_map_src_fill_b(self) -> None:
        src = np.array(['aa', 'bbbbb', 'ccc', 'dddd'])

        tm = TriMap(4, 4)
        tm.register_one(0, -1)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_one(3, -1)
        tm.register_unmatched_dst()

        post = tm.map_src_fill(src, 'na', np.dtype(str))
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), ['aa', 'bbbbb', 'ccc', 'dddd', 'na', 'na'])

    def test_tri_map_map_src_fill_c(self) -> None:
        src = np.array(['aa', None, False, 300000000000000000000])

        tm = TriMap(4, 4)
        tm.register_one(0, -1)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_one(3, -1)
        tm.register_unmatched_dst()

        post = tm.map_src_fill(src, 'na', np.dtype(str))
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), ['aa', None, False, 300000000000000000000, 'na', 'na'])

    #---------------------------------------------------------------------------

    def test_tri_map_map_a(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd'])
        dst = np.array(['cc', 'a', 'a', 'a', 'cc'])

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), ['a', 'a', 'a', 'bbb', 'cc', 'cc', 'dddd'])

        post_dst = tm.map_dst_fill(dst, '', np.dtype(str))
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), ['a', 'a', 'a', '', 'cc', 'cc', ''])


    def test_tri_map_map_b(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd', 'a'])
        dst = np.array(['cc', 'dddd', 'a', 'bbb', 'cc'])

        tm = TriMap(len(src), len(dst))
        tm.register_one(0, 2)
        tm.register_one(1, 3)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, 1)
        tm.register_one(4, 2)

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), ['a', 'bbb', 'cc', 'cc', 'dddd', 'a'])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), ['a', 'bbb', 'cc', 'cc', 'dddd', 'a'])

    def test_tri_map_map_c(self) -> None:
        src = np.array([0, 200, 300, 400, 0])
        dst = np.array([300, 400, 0, 200, 300])

        tm = TriMap(len(src), len(dst))
        tm.register_one(0, 2)
        tm.register_one(1, 3)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, 1)
        tm.register_one(4, 2)

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(),  [0, 200, 300, 300, 400, 0])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(),  [0, 200, 300, 300, 400, 0])


    def test_tri_map_map_d(self) -> None:
        src = np.array([0, 200, 300, 300, 0])
        dst = np.array([300, 200, 300, 200, 300, -1, -1])

        tm = TriMap(len(src), len(dst))
        tm.register_many(1, np.array([1, 3], dtype=np.dtype(np.int64)))
        tm.register_many(2, np.array([0, 2, 4], dtype=np.dtype(np.int64)))
        tm.register_many(3, np.array([0, 2, 4], dtype=np.dtype(np.int64)))


        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [200, 200, 300, 300, 300, 300, 300, 300])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [200, 200, 300, 300, 300, 300, 300, 300])



    def test_tri_map_map_e(self) -> None:
        src = np.array([0, 200, 300, 5, 0])
        dst = np.array([-1, 200, 300, 200, 300, -1, -1])

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 3], dtype=np.dtype(np.int64)))
        tm.register_many(2, np.array([2, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_one(4, -1)
        tm.register_unmatched_dst()

        post_src = tm.map_src_fill(src, -20, np.dtype(int))
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [0, 200, 200, 300, 300, 5, 0, -20, -20, -20])

        post_dst = tm.map_dst_fill(dst, -20, np.dtype(int))
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [-20, 200, 200, 300, 300, -20, -20, -1, -1, -1])

