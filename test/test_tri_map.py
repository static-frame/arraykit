import unittest
import numpy as np

from arraykit import TriMap

dt64 = np.datetime64
nat = dt64('nat')


class TestUnit(unittest.TestCase):

    def test_tri_map_init_a(self) -> None:
        with self.assertRaises(TypeError):
            tm1 = TriMap()

        tm2 = TriMap(10, 20)
        tm3 = TriMap(10_000, 20_000)

    def test_tri_map_repr_a(self) -> None:
        tm = TriMap(10_000, 20_000)
        self.assertEqual(str(tm), '<arraykit.TriMap(len: 0, src_fill: -1, dst_fill: -1, is_many: false, is_finalized: false)>')

    def test_tri_map_repr_b(self) -> None:
        tm = TriMap(6, 6)
        tm.register_many(0, np.array([0, 1], dtype=np.int64))
        tm.register_one(1, -1)
        tm.register_one(2, -1)
        tm.register_one(3, -1)
        tm.register_one(4, -1)
        tm.register_one(5, -1)
        tm.finalize()
        self.assertEqual(str(tm), '<arraykit.TriMap(len: 7, src_fill: 0, dst_fill: 5, is_many: true, is_finalized: true)>')


    def test_tri_map_finalize_a(self) -> None:

        tm = TriMap(10, 20)
        tm.finalize()
        with self.assertRaises(RuntimeError):
            tm.finalize()


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
        tm.finalize()
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 2000, src_fill: 0, dst_fill: 0, is_many: false, is_finalized: true)>')

    def test_tri_map_register_one_c(self) -> None:
        tm = TriMap(20, 30)
        with self.assertRaises(RuntimeError):
            self.assertFalse(tm.is_many())

        with self.assertRaises(ValueError):
            tm.register_one(22, 0)
        with self.assertRaises(ValueError):
            tm.register_one(0, 30)

        tm.register_one(19, 29)
        tm.register_one(18, 29)
        tm.finalize()
        self.assertTrue(tm.is_many())


    def test_tri_map_src_no_fill_a(self) -> None:
        tm = TriMap(3, 3)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, -1)
        tm.finalize()
        self.assertTrue(tm.src_no_fill())
        self.assertFalse(tm.dst_no_fill())

    def test_tri_map_register_unmatched_dst_a(self) -> None:
        tm = TriMap(10, 8)
        tm.register_one(0, 0)
        tm.register_one(1, 1)
        tm.register_one(2, 2)
        tm.register_unmatched_dst()

        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 8, src_fill: -1, dst_fill: -1, is_many: false, is_finalized: false)>')

        tm.finalize()
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 8, src_fill: 5, dst_fill: 0, is_many: false, is_finalized: true)>')


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
        tm.finalize()
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 3, src_fill: 0, dst_fill: 0, is_many: true, is_finalized: true)>')

    def test_tri_map_register_many_d1(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int64))
        tm.finalize()
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_fill: 0, dst_fill: 0, is_many: true, is_finalized: true)>')

    def test_tri_map_register_many_d2(self) -> None:
        tm = TriMap(100, 50)
        for i in range(100):
            tm.register_many(i, np.array([3, 20], dtype=np.int64))
        tm.finalize()
        self.assertEqual(repr(tm), '<arraykit.TriMap(len: 200, src_fill: 0, dst_fill: 0, is_many: true, is_finalized: true)>')

    #---------------------------------------------------------------------------

    def test_tri_map_map_src_no_fill_a(self) -> None:
        src = np.array([10, 20, 30, 40], dtype=np.int64)
        dst = np.array([30, 30, 40, 40], dtype=np.int64)

        tm = TriMap(4, 4)
        tm.register_one(0, -1)
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 1], dtype=np.int64))
        tm.register_many(3, np.array([2, 3], dtype=np.int64))
        tm.finalize()

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
        tm.finalize()

        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), ['a', 'bbb', 'cc', 'dddd'])

    def test_tri_map_map_src_no_fill_c(self) -> None:
        src = np.array(['aaaaa', 'bbb', 'cc', 'dddd'])

        tm = TriMap(4, 4)
        tm.register_many(0, np.array([1, 3], dtype=np.int64))
        tm.register_many(1, np.array([0, 2], dtype=np.int64))
        tm.finalize()

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
        tm.finalize()

        post = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post.flags.writeable)
        self.assertEqual(post.tolist(), [None, 'bbb', 3, False])

    def test_tri_map_map_src_no_fill_d(self) -> None:
        src = np.array([None, 'bbb', 3, False])

        tm = TriMap(4, 4)
        tm.register_many(0, np.array([1, 3], dtype=np.int64))
        tm.register_many(1, np.array([0, 2], dtype=np.int64))
        tm.finalize()

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
        tm.finalize()

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
        tm.finalize()

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
        tm.finalize()

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
        tm.finalize()

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
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), ['a', 'bbb', 'cc', 'cc', 'dddd', 'a'])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), ['a', 'bbb', 'cc', 'cc', 'dddd', 'a'])

    def test_tri_map_map_c(self) -> None:
        src = np.array([0, 200, 300, 400, 0], dtype=np.int64)
        dst = np.array([300, 400, 0, 200, 300], dtype=np.int64)

        tm = TriMap(len(src), len(dst))
        tm.register_one(0, 2)
        tm.register_one(1, 3)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, 1)
        tm.register_one(4, 2)
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(),  [0, 200, 300, 300, 400, 0])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(),  [0, 200, 300, 300, 400, 0])


    def test_tri_map_map_d(self) -> None:
        src = np.array([0, 200, 300, 300, 0], dtype=np.int64)
        dst = np.array([300, 200, 300, 200, 300, -1, -1], dtype=np.int64)

        tm = TriMap(len(src), len(dst))
        tm.register_many(1, np.array([1, 3], dtype=np.dtype(np.int64)))
        tm.register_many(2, np.array([0, 2, 4], dtype=np.dtype(np.int64)))
        tm.register_many(3, np.array([0, 2, 4], dtype=np.dtype(np.int64)))
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [200, 200, 300, 300, 300, 300, 300, 300])

        post_dst = tm.map_dst_no_fill(dst)
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [200, 200, 300, 300, 300, 300, 300, 300])



    def test_tri_map_map_e(self) -> None:
        src = np.array([0, 200, 300, 5, 0], dtype=np.int64)
        dst = np.array([-1, 200, 300, 200, 300, -1, -1], dtype=np.int64)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 3], dtype=np.dtype(np.int64)))
        tm.register_many(2, np.array([2, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_one(4, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, -20, np.dtype(np.int64))
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [0, 200, 200, 300, 300, 5, 0, -20, -20, -20])

        post_dst = tm.map_dst_fill(dst, -20, np.dtype(np.int64))
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [-20, 200, 200, 300, 300, -20, -20, -1, -1, -1])


    def test_tri_map_map_object_a(self) -> None:
        src = np.array([0, 200, 300], dtype=np.int64)
        dst = np.array([-1, 400, 200], dtype=np.int64)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_one(1, 2)
        tm.register_one(2, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, None, np.dtype(np.object_))
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [0, 200, 300, None, None])

        post_dst = tm.map_dst_fill(dst, None, np.dtype(np.object_))
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [None, 200, None, -1, 400])


    def test_tri_map_map_object_b(self) -> None:
        src = np.array([0, 20000, 300], dtype=np.int64)
        dst = np.array([-1, 20000, 20000, 20000], dtype=np.int64)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(2, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, None, np.dtype(np.object_))
        del src
        self.assertFalse(post_src.flags.writeable)
        self.assertEqual(post_src.tolist(), [0, 20000, 20000, 20000, 300, None])
        # we reuse the same instance
        self.assertEqual(id(post_src[1]), id(post_src[2]))
        self.assertEqual(id(post_src[1]), id(post_src[3]))

        post_dst = tm.map_dst_fill(dst, None, np.dtype(np.object_))
        del dst
        self.assertFalse(post_dst.flags.writeable)
        self.assertEqual(post_dst.tolist(), [None, 20000, 20000, 20000, None, -1])

    def test_tri_map_map_object_c(self) -> None:
        src = np.array([True, False, True], dtype=np.bool_)
        dst = np.array([False, False, False], dtype=np.bool_)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([0, 1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, -1)
        tm.finalize()

        post_src = tm.map_src_fill(src, None, np.dtype(np.object_))
        self.assertEqual(post_src.tolist(), [True, False, False, False, True])

        post_dst = tm.map_dst_fill(dst, None, np.dtype(np.object_))
        self.assertEqual(post_dst.tolist(), [None, False, False, False, None])


    def test_tri_map_map_bool_a(self) -> None:
        src = np.array([True, False, True], dtype=np.bool_)
        dst = np.array([False, False, False], dtype=np.bool_)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([0, 1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, -1)
        tm.finalize()

        post_src = tm.map_src_fill(src, False, np.dtype(np.bool_))
        self.assertEqual(post_src.tolist(), [True, False, False, False, True])

        post_dst = tm.map_dst_fill(dst, False, np.dtype(np.bool_))
        self.assertEqual(post_dst.tolist(), [False, False, False, False, False])


    def test_tri_map_map_int_a(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.int32)
        dst = np.array([-1, 20, 20, 8], dtype=np.int32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, -10, np.dtype(np.int8))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, -10])
        self.assertEqual(post_src.dtype, np.dtype(np.int32))

        post_dst = tm.map_dst_fill(dst, -10, np.dtype(np.int8))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.int32))
        self.assertEqual(post_dst.tolist(), [-10, 20, 20, 8, 8, -1])

    def test_tri_map_map_int_b(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.int16)
        dst = np.array([-1, 20, 20, 8], dtype=np.int16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, -10, np.dtype(np.int8))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, -10])
        self.assertEqual(post_src.dtype, np.dtype(np.int16))

        post_dst = tm.map_dst_fill(dst, -10, np.dtype(np.int8))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.int16))
        self.assertEqual(post_dst.tolist(), [-10, 20, 20, 8, 8, -1])


    def test_tri_map_map_int_c(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.int8)
        dst = np.array([-1, 20, 20, 8], dtype=np.int8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, -10, np.dtype(np.int8))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, -10])
        self.assertEqual(post_src.dtype, np.dtype(np.int8))

        post_dst = tm.map_dst_fill(dst, -10, np.dtype(np.int8))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.int8))
        self.assertEqual(post_dst.tolist(), [-10, 20, 20, 8, 8, -1])


    def test_tri_map_map_uint_a(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint16)
        dst = np.array([7, 20, 20, 8], dtype=np.uint16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.uint8))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.uint16))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.uint8))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.uint16))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_uint_b(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint8)
        dst = np.array([7, 20, 20, 8], dtype=np.uint8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.uint32))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.uint32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.uint32))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.uint32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_uint_c(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint32)
        dst = np.array([7, 20, 20, 8], dtype=np.uint32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.uint64))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.uint64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.uint64))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.uint64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

        self.assertEqual(str(tm), '<arraykit.TriMap(len: 6, src_fill: 1, dst_fill: 1, is_many: true, is_finalized: true)>')

    def test_tri_map_map_uint_d(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint8)
        dst = np.array([7, 20, 20, 8], dtype=np.uint8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int16))
        del src
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.int16))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int16))
        del dst
        self.assertEqual(post_dst.dtype, np.dtype(np.int16))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_uint_e(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint8)
        dst = np.array([7, 20, 20, 8], dtype=np.uint8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int32))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.int32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int32))
        self.assertEqual(post_dst.dtype, np.dtype(np.int32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_uint_f(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint16)
        dst = np.array([7, 20, 20, 8], dtype=np.uint16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int32))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.int32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int32))
        self.assertEqual(post_dst.dtype, np.dtype(np.int32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_uint_g(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint32)
        dst = np.array([7, 20, 20, 8], dtype=np.uint32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.int64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int64))
        self.assertEqual(post_dst.dtype, np.dtype(np.int64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_uint_h(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint8)
        dst = np.array([7, 20, 20, 8], dtype=np.uint8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.int64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int64))
        self.assertEqual(post_dst.dtype, np.dtype(np.int64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_uint_i(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint64)
        dst = np.array([7, 20, 20, 8], dtype=np.uint64)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int64))
        self.assertEqual(post_dst.dtype, np.dtype(np.float64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])



    def test_tri_map_map_float_a(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.uint8)
        dst = np.array([7, 20, 20, 8], dtype=np.uint8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float64))
        self.assertEqual(post_dst.dtype, np.dtype(np.float64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_float_b(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.int8)
        dst = np.array([7, 20, 20, 8], dtype=np.int8)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float64))
        self.assertEqual(post_dst.dtype, np.dtype(np.float64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_float_c(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float32)
        dst = np.array([7, 20, 20, 8], dtype=np.float32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float64))
        self.assertEqual(post_dst.dtype, np.dtype(np.float64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_d(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float64)
        dst = np.array([7, 20, 20, 8], dtype=np.float64)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float64))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float64))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float64))
        self.assertEqual(post_dst.dtype, np.dtype(np.float64))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_e(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float32)
        dst = np.array([7, 20, 20, 8], dtype=np.float32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int8))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int8))
        self.assertEqual(post_dst.dtype, np.dtype(np.float32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_f(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float32)
        dst = np.array([7, 20, 20, 8], dtype=np.float32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.uint16))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.uint16))
        self.assertEqual(post_dst.dtype, np.dtype(np.float32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_g(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float32)
        dst = np.array([7, 20, 20, 8], dtype=np.float32)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float32))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float32))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float32))
        self.assertEqual(post_dst.dtype, np.dtype(np.float32))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_h(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float16)
        dst = np.array([7, 20, 20, 8], dtype=np.float16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.float16))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float16))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.float16))
        self.assertEqual(post_dst.dtype, np.dtype(np.float16))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])


    def test_tri_map_map_float_i(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float16)
        dst = np.array([7, 20, 20, 8], dtype=np.float16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.int8))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float16))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.int8))
        self.assertEqual(post_dst.dtype, np.dtype(np.float16))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_float_i(self) -> None:
        src = np.array([0, 20, 8, 8], dtype=np.float16)
        dst = np.array([7, 20, 20, 8], dtype=np.float16)

        # full outer
        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_many(1, np.array([1, 2], dtype=np.dtype(np.int64)))
        tm.register_one(2, 3)
        tm.register_one(3, 3)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, 17, np.dtype(np.uint8))
        self.assertEqual(post_src.tolist(), [0, 20, 20, 8, 8, 17])
        self.assertEqual(post_src.dtype, np.dtype(np.float16))

        post_dst = tm.map_dst_fill(dst, 17, np.dtype(np.uint8))
        self.assertEqual(post_dst.dtype, np.dtype(np.float16))
        self.assertEqual(post_dst.tolist(), [17, 20, 20, 8, 8, 7])

    def test_tri_map_map_bytes_a(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd', 'a'], dtype=np.bytes_)
        dst = np.array(['cc', 'dddd', 'a', 'bbb', 'cc'], dtype=np.bytes_)

        tm = TriMap(len(src), len(dst))
        tm.register_one(0, 2)
        tm.register_one(1, 3)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, 1)
        tm.register_one(4, 2)
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        self.assertEqual(post_src.tolist(), [b'a', b'bbb', b'cc', b'cc', b'dddd', b'a'])

        post_dst = tm.map_dst_no_fill(dst)
        self.assertEqual(post_dst.tolist(), [b'a', b'bbb', b'cc', b'cc', b'dddd', b'a'])

    def test_tri_map_map_bytes_a(self) -> None:
        src = np.array([b'a', b'bbb', b'cc'], dtype=np.bytes_)
        dst = np.array([b'cc', b'dddd', b'eee'], dtype=np.bytes_)

        tm = TriMap(len(src), len(dst))
        tm.register_one(0, -1)
        tm.register_one(1, -1)
        tm.register_one(2, 0)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, b'--', np.dtype(np.bytes_))
        post_dst = tm.map_dst_fill(dst, b'--', np.dtype(np.bytes_))
        self.assertEqual(post_src.tolist(), [b'a', b'bbb', b'cc', b'--', b'--'])
        self.assertEqual(post_dst.tolist(), [b'--', b'--', b'cc', b'dddd', b'eee'])
    #---------------------------------------------------------------------------

    def test_tri_map_map_unicode_a(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd'])
        dst = np.array(['cc', 'a', 'a', 'a', 'cc'])

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        self.assertEqual(post_src.tolist(), ['a', 'a', 'a', 'bbb', 'cc', 'cc', 'dddd'])

        post_dst = tm.map_dst_fill(dst, '====', np.array('====').dtype)
        self.assertEqual(post_dst.tolist(), ['a', 'a', 'a', '====', 'cc', 'cc', '===='])

    def test_tri_map_map_unicode_b(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd'])
        dst = np.array(['cc', 'a', 'a', 'a', 'cc'])

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.finalize()

        post_src = tm.map_src_no_fill(src)
        self.assertEqual(post_src.tolist(), ['a', 'a', 'a', 'bbb', 'cc', 'cc', 'dddd'])

        post_dst1 = tm.map_dst_fill(dst, b'====', np.array(b'====').dtype)
        self.assertEqual(post_dst1.tolist(), ['a', 'a', 'a', '====', 'cc', 'cc', '===='])

        post_dst2 = tm.map_dst_fill(dst, b'?', np.array(b'?').dtype)
        self.assertEqual(post_dst2.tolist(), ['a', 'a', 'a', '?', 'cc', 'cc', '?'])

    def test_tri_map_map_unicode_c(self) -> None:
        src = np.array(['a', 'bbb', 'cc', 'dddd'])
        dst = np.array(['cc', 'a', 'a', 'a', 'cc'])

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.finalize()

        post_dst1 = tm.map_dst_fill(dst, 3, np.dtype(object))
        self.assertEqual(post_dst1.tolist(), ['a', 'a', 'a', 3, 'cc', 'cc', 3])

        self.assertEqual(str(tm), '<arraykit.TriMap(len: 7, src_fill: 0, dst_fill: 2, is_many: true, is_finalized: true)>')

    #---------------------------------------------------------------------------

    def test_tri_map_map_dt64_a(self) -> None:
        src = np.array(['2022-01', '1954-03', '1743-09', '1988-12'], dtype=np.datetime64)
        dst = np.array(['1743-09', '2022-01', '2022-01', '2022-01', '1743-09', '2005-11'], dtype=np.datetime64)

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, nat, np.dtype('datetime64'))
        self.assertEqual(post_src.dtype, np.dtype('datetime64[M]'))
        # string to permit NaN comparison
        self.assertEqual([str(dt) for dt in post_src],
        ['2022-01', '2022-01', '2022-01', '1954-03', '1743-09', '1743-09', '1988-12', 'NaT'])

        post_dst = tm.map_dst_fill(dst, nat, np.dtype('datetime64'))
        self.assertEqual(post_dst.dtype, np.dtype('datetime64[M]'))
        self.assertEqual([str(dt) for dt in post_dst],
        ['2022-01', '2022-01', '2022-01', 'NaT', '1743-09', '1743-09', 'NaT', '2005-11'])


    def test_tri_map_map_dt64_b(self) -> None:
        src = np.array(['2022-01', '1954-03', '1743-09', '1988-12'], dtype=np.datetime64)
        dst = np.array(['1743-09', '2022-01', '2022-01', '2022-01', '1743-09', '2005-11'], dtype=np.datetime64)

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, '1999-12', np.dtype('datetime64'))
        self.assertEqual(post_src.dtype, np.dtype('datetime64[M]'))
        self.assertEqual([str(dt) for dt in post_src],
        ['2022-01', '2022-01', '2022-01', '1954-03', '1743-09', '1743-09', '1988-12', '1999-12'])

        post_dst = tm.map_dst_fill(dst, '1999-12', np.dtype('datetime64'))
        self.assertEqual(post_dst.dtype, np.dtype('datetime64[M]'))
        self.assertEqual([str(dt) for dt in post_dst],
        ['2022-01', '2022-01', '2022-01', '1999-12', '1743-09', '1743-09', '1999-12', '2005-11'])

    def test_tri_map_map_dt64_c(self) -> None:
        src = np.array(['2022-01', '1954-03', '1743-09', '1988-12'], dtype=np.datetime64)
        dst = np.array(['1743-09', '2022-01', '2022-01', '2022-01', '1743-09', '2005-11'], dtype=np.datetime64)

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, '1999', np.dtype('datetime64[Y]'))
        self.assertEqual(post_src.dtype, np.dtype('datetime64[M]'))
        # NOTE: the year dtype is "fit" within the year-mo by defaulting to the first month; we might not want to permit this
        self.assertEqual([str(dt) for dt in post_src],
            ['2022-01', '2022-01', '2022-01', '1954-03', '1743-09', '1743-09', '1988-12', '1999-01'])

        post_dst = tm.map_dst_fill(dst, '1999', np.dtype('datetime64[Y]'))
        self.assertEqual(post_dst.dtype, np.dtype('datetime64[M]'))
        self.assertEqual([str(dt) for dt in post_dst],
            ['2022-01', '2022-01', '2022-01', '1999-01', '1743-09', '1743-09', '1999-01', '2005-11'])

    def test_tri_map_map_dt64_d(self) -> None:
        src = np.array(['2022-01', '1954-03', '1743-09', '1988-12'], dtype=np.datetime64)
        dst = np.array(['1743-09', '2022-01', '2022-01', '2022-01', '1743-09', '2005-11'], dtype=np.datetime64)

        tm = TriMap(len(src), len(dst))
        tm.register_many(0, np.array([1, 2, 3], dtype=np.dtype(np.int64)))
        tm.register_one(1, -1)
        tm.register_many(2, np.array([0, 4], dtype=np.dtype(np.int64)))
        tm.register_one(3, -1)
        tm.register_unmatched_dst()
        tm.finalize()

        post_src = tm.map_src_fill(src, '1999-09-09', np.dtype('datetime64[D]'))
        self.assertEqual(post_src.dtype, np.dtype('datetime64[D]'))
        self.assertEqual([str(dt) for dt in post_src],
            ['2022-01-01', '2022-01-01', '2022-01-01', '1954-03-01', '1743-09-01', '1743-09-01', '1988-12-01', '1999-09-09'])

        post_dst = tm.map_dst_fill(dst, '1999-09-09', np.dtype('datetime64[D]'))
        self.assertEqual(post_dst.dtype, np.dtype('datetime64[D]'))
        self.assertEqual([str(dt) for dt in post_dst],
            ['2022-01-01', '2022-01-01', '2022-01-01', '1999-09-09', '1743-09-01', '1743-09-01', '1999-09-09', '2005-11-01'])
