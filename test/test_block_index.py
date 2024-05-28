import unittest
import ctypes
import sys
import pickle

import numpy as np

from arraykit import BlockIndex
from arraykit import ErrorInitTypeBlocks


class TestUnit(unittest.TestCase):

    def test_block_index_init_a(self) -> None:
        bi1 = BlockIndex()
        self.assertEqual(bi1.dtype, np.dtype(float))
        s = bi1.shape
        self.assertEqual(s, (0, 0))
        del bi1
        self.assertEqual(s, (0, 0))
        del s

    def test_block_index_init_b1(self) -> None:
        with self.assertRaises(ValueError):
            _ = BlockIndex(3, 2, 10, 2)

    def test_block_index_init_b1(self) -> None:
        with self.assertRaises(TypeError):
            _ = BlockIndex(3, 2, 10, 2, 'a')

    def test_block_index_init_c1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(2))
        block, row, bir_count, bir_capacity, bi, dt = bi1.__getstate__()

        bi2 = BlockIndex(block, row, bir_count, bir_capacity, bi, np.dtype(np.int64))
        self.assertTrue("dtype('int64')" in bi2.__repr__())
        self.assertEqual(bi2.dtype, np.dtype(np.int64))

    def test_block_index_init_c2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(2))
        block, row, bir_count, bir_capacity, bi, dt = bi1.__getstate__()

        with self.assertRaises(TypeError):
            bi2 = BlockIndex(block, row, bir_count, bir_capacity, bi, 'a')


    def test_block_index_init_d(self) -> None:
        bi1 = BlockIndex()
        self.assertTrue('None' in repr(bi1))

    #---------------------------------------------------------------------------

    def test_block_index_register_a(self) -> None:
        bi1 = BlockIndex()
        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register('foo')

        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register(3.5)

    def test_block_index_register_b(self) -> None:

        bi1 = BlockIndex()
        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register(np.array(0))

        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register(np.arange(12).reshape(2,3,2))


    def test_block_index_register_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.arange(6).reshape(3,2))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (2, 0), (2, 1)])
        self.assertEqual(bi1.shape, (3, 4))
        self.assertEqual(bi1.rows, 3)
        self.assertEqual(bi1.columns, 4)

    def test_block_index_register_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(2))
        bi1.register(np.arange(12).reshape(2,6))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
            )
        self.assertEqual(bi1.shape, (2, 14))
        self.assertEqual(bi1.rows, 2)
        self.assertEqual(bi1.columns, 14)

    def test_block_index_register_e(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register(np.arange(12).reshape(3,4))


    def test_block_index_register_f(self) -> None:
        bi1 = BlockIndex()
        a1 = np.arange(20000).reshape(2, 10_000)
        bi1.register(a1)
        self.assertEqual(bi1.rows, 2)
        self.assertEqual(bi1.columns, 10_000)


    def test_block_index_register_g(self) -> None:
        bi1 = BlockIndex()
        a1 = np.array(()).reshape(4, 0)
        self.assertFalse(bi1.register(a1))
        self.assertEqual(bi1.shape, (4, 0))
        # as not dtype has been registered, we will get default float
        self.assertEqual(bi1.dtype, np.dtype(float))

        a2 = np.arange(8).reshape(4, 2).astype(bool)
        self.assertTrue(bi1.register(a2))
        self.assertEqual(bi1.shape, (4, 2))
        self.assertEqual(bi1.dtype, np.dtype(bool))


    def test_block_index_register_h(self) -> None:
        bi1 = BlockIndex()
        a1 = np.array(()).reshape(0, 4).astype(bool)
        self.assertTrue(bi1.register(a1))
        self.assertEqual(bi1.shape, (0, 4))
        self.assertEqual(bi1.dtype, np.dtype(bool))

        a2 = np.array(()).reshape(0, 0).astype(float)
        self.assertFalse(bi1.register(a2))
        self.assertEqual(bi1.shape, (0, 4))
        # dtype is still bool
        self.assertEqual(bi1.dtype, np.dtype(bool))

        a3 = np.array(()).reshape(0, 3).astype(int)
        self.assertTrue(bi1.register(a3))
        self.assertEqual(bi1.shape, (0, 7))
        self.assertEqual(bi1.dtype, np.dtype(object))


    def test_block_index_register_i(self) -> None:
        bi1 = BlockIndex()
        # NOTE: this value in one context returned an unset exception; I think I have now covered those cases but cannot reproduce the failure; testing the full size is too slow, so reducing here as a placeholder
        size = 2_147_483_649 // 100
        post = bi1.register(np.array(()).reshape(0, size))
        self.assertEqual(bi1.shape, (0, size))


    #---------------------------------------------------------------------------

    def test_block_index_to_bytes_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
            )
        data = bi1.to_bytes()
        bd = ctypes.sizeof(ctypes.c_ssize_t)
        post = [int.from_bytes(
            data[slice(i, i+bd)], sys.byteorder, signed=True) for i in
            range(0, len(data), bd)
            ]
        self.assertEqual(post, [0, 0, 0, 1, 0, 2, 1, 0, 1, 1])


    #---------------------------------------------------------------------------

    def test_block_index_copy_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        s1 = bi1.shape
        bi2 = bi1.copy()
        self.assertEqual(bi1.to_list(), bi2.to_list())
        self.assertEqual(bi1.dtype, bi2.dtype)
        del bi1
        self.assertEqual(bi2.shape, s1)

    def test_block_index_copy_b(self) -> None:
        dt1 = np.dtype(np.float64)
        bi1 = BlockIndex(0, 2, 0, 8, b"", dt1)
        bi2 = bi1.copy()
        dt2 = bi1.dtype
        del dt1
        del bi1
        self.assertTrue('float64' in repr(bi2))
        del bi2
        self.assertEqual(dt2, np.dtype(np.float64))


    #---------------------------------------------------------------------------
    def test_block_index_sizeof_a(self) -> None:
        bi1 = BlockIndex()
        so1 = sys.getsizeof(bi1)
        bi1.register(np.arange(100).reshape(2,50))
        so2 = sys.getsizeof(bi1)
        self.assertTrue(so1 < so2)

    #---------------------------------------------------------------------------

    def test_block_index_len_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(len(bi1), 8)

    def test_block_index_len_b(self) -> None:
        bi1 = BlockIndex()
        self.assertEqual(len(bi1), 0)

    #---------------------------------------------------------------------------


    def test_block_index_getitem_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(bi1[3], (0, 3))
        self.assertEqual(bi1[7], (1, 1))

        with self.assertRaises(IndexError):
            bi1[8]


    def test_block_index_getitem_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))

        # lookup by scalar
        a1 = np.array([3, 7])
        self.assertEqual(bi1[a1[0]], (0, 3))
        self.assertEqual(bi1[a1[1]], (1, 1))

    #---------------------------------------------------------------------------
    def test_block_index_getitem_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        self.assertEqual(bi1.shape, (2, 6))
        self.assertEqual(bi1.columns, 6)

        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(bi1.shape, (2, 8))
        self.assertEqual(bi1.columns, 8)

    def test_block_index_getitem_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))

        with self.assertRaises(TypeError):
            bi1['a']
        with self.assertRaises(TypeError):
            bi1[3:5]

    #---------------------------------------------------------------------------
    def test_block_index_get_state_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))

        block, row, bir_count, bir_capacity, bi, dt = bi1.__getstate__()
        self.assertEqual((block, row, bir_count, bir_capacity), (3, 2, 9, 16))
        self.assertTrue(isinstance(bi, bytes))
        self.assertIs(dt, np.dtype(int))

        bi2 = BlockIndex(block, row, bir_count, bir_capacity, bi, dt)
        self.assertEqual(repr(bi1), repr(bi2))

    #---------------------------------------------------------------------------
    def test_block_index_pickle_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))

        msg = pickle.dumps(bi1)
        bi2 = pickle.loads(msg)

        self.assertEqual(repr(bi1), repr(bi2))
        self.assertEqual(bi1.to_list(), bi2.to_list())

    #---------------------------------------------------------------------------
    def test_block_index_dtype_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        self.assertEqual(bi1.dtype, np.dtype(int))

        bi1.register(np.arange(2).astype(float))
        self.assertEqual(bi1.dtype, np.dtype(float))

        bi1.register(np.arange(2).astype(bool))
        self.assertEqual(bi1.dtype, np.dtype(object))

    def test_block_index_dtype_b(self) -> None:
        bi1 = BlockIndex()
        self.assertEqual(bi1.dtype, None)

        bi1.register(np.arange(2))
        bi1.register(np.arange(2).astype(bool))
        self.assertEqual(bi1.dtype, np.dtype(object))


    #---------------------------------------------------------------------------
    def test_block_index_get_block_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))
        bi1.register(np.arange(2))

        self.assertEqual(bi1.get_block(6), 2)
        self.assertEqual(bi1.get_block(5), 1)
        self.assertEqual(bi1.get_block(1), 1)
        self.assertEqual(bi1.get_block(0), 0)

    def test_block_index_get_column_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))
        bi1.register(np.arange(2))

        self.assertEqual(bi1.get_column(6), 0)
        self.assertEqual(bi1.get_column(5), 4)
        self.assertEqual(bi1.get_column(1), 0)
        self.assertEqual(bi1.get_column(0), 0)


    #---------------------------------------------------------------------------
    def test_block_index_iter_a1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(2))

        biit = iter(bi1)
        self.assertEqual(next(biit), (0, 0))

        self.assertEqual(list(bi1), [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)])
        self.assertEqual(list(reversed(bi1)), [(2, 0), (1, 2), (1, 1), (1, 0), (0, 0)])

    def test_block_index_iter_a2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(4).reshape(2,2))

        with self.assertRaises(TypeError):
            _ = bi1.iter_select(None)

        with self.assertRaises(TypeError):
            _ = bi1.iter_select(np.array(['a', 'b']))

        with self.assertRaises(TypeError):
            _ = bi1.iter_select(np.arange(4).reshape(2,2))

    def test_block_index_iter_b1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(4).reshape(2,2))

        biit1 = bi1.iter_select(np.array([0,3,4]))
        self.assertEqual(list(biit1), [(0, 0), (2, 0), (2, 1)])
        self.assertEqual(list(reversed(biit1)), [(2, 1), (2, 0), (0, 0)])

        biit2 = bi1.iter_select(np.array([0,3,4], dtype=np.uint8))
        self.assertEqual(list(biit2), [(0, 0), (2, 0), (2, 1)])
        self.assertEqual(list(reversed(biit2)), [(2, 1), (2, 0), (0, 0)])

    def test_block_index_iter_b2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(4).reshape(2,2))

        biit1 = bi1.iter_select(list(np.array([0,3,4])))
        self.assertEqual(list(biit1), [(0, 0), (2, 0), (2, 1)])
        self.assertEqual(list(reversed(biit1)), [(2, 1), (2, 0), (0, 0)])

    def test_block_index_iter_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(4).reshape(2,2))

        biit1 = bi1.iter_select([0,3,4])
        self.assertEqual(list(biit1), [(0, 0), (2, 0), (2, 1)])
        self.assertEqual(list(reversed(biit1)), [(2, 1), (2, 0), (0, 0)])

        biit2 = bi1.iter_select([0,3,4])
        self.assertEqual(list(biit2), [(0, 0), (2, 0), (2, 1)])
        self.assertEqual(list(reversed(biit2)), [(2, 1), (2, 0), (0, 0)])


    def test_block_index_iter_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(4).reshape(2,2))

        with self.assertRaises(TypeError):
            _ = list(bi1.iter_select([0,3,'b']))


    #---------------------------------------------------------------------------
    def test_block_index_iter_select_slice_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        self.assertEqual(list(bi1.iter_select((slice(None)))),
            [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
            )

        self.assertEqual(list(bi1.iter_select((slice(4, None)))),
            [(2, 1), (2, 2), (2, 3), (2, 4)]
            )

        self.assertEqual(list(bi1.iter_select((slice(None)))),
            [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
            )

        self.assertEqual(list(bi1.iter_select((slice(1, 8, 2)))),
            [(0, 1), (2, 0), (2, 2), (2, 4)]
            )

    def test_block_index_iter_select_slice_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        self.assertEqual(list(bi1.iter_select((slice(7, 3, -1)))),
            [(2, 4), (2, 3), (2, 2), (2, 1)]
            )

        self.assertEqual(list(bi1.iter_select((slice(None, None, -1)))),
            [(2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (0, 1), (0, 0)]
            )

    def test_block_index_iter_select_slice_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(6).reshape(2,3))

        self.assertEqual(list(bi1.iter_select(slice(1,5))),
            [(0, 1), (1, 0), (2, 0), (2, 1)]
            )

        self.assertEqual(list(reversed(bi1.iter_select(slice(1,5)))),
            [(2, 1), (2, 0), (1, 0), (0, 1)]
            )


    def test_block_index_iter_select_slice_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(2))

        self.assertEqual(list(bi1.iter_select(slice(None))),
            [(0, 0), (0, 1), (0, 2), (1, 0)]
            )
        self.assertEqual(list(bi1.iter_select(slice(20, 24))),
            []
            )
        self.assertEqual(list(bi1.iter_select(slice(0, 100, 10))),
            [(0, 0)]
            )
        self.assertEqual(list(bi1.iter_select(slice(0, 100, 3))),
            [(0, 0), (1, 0)]
            )

    def test_block_index_iter_select_slice_e(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(12).reshape(2,6))

        self.assertEqual(list(bi1.iter_select(slice(11, None, -3))),
            [(1, 5), (1, 2), (0, 5), (0, 2)]
            )
        self.assertEqual(list(bi1.iter_select(slice(11, None, -4))),
            [(1, 5), (1, 1), (0, 3)]
            )


    #---------------------------------------------------------------------------
    def test_block_index_iter_select_boolean_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        sel1 = np.array([x % 2 == 0 for x in range(len(bi1))])
        self.assertEqual(list(bi1.iter_select(sel1)),
                [(0, 0), (1, 0), (2, 1), (2, 3)]
                )

        sel2 = np.full(len(bi1), False)
        sel2[0] = True
        sel2[-1] = True
        self.assertEqual(list(bi1.iter_select(sel2)),
                [(0, 0), (2, 4)]
                )

    def test_block_index_iter_select_boolean_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))

        with self.assertRaises(TypeError):
            bi1.iter_select(np.array([False, True]))

        with self.assertRaises(TypeError):
            bi1.iter_select(np.full(20, True))


    def test_block_index_iter_select_boolean_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        self.assertEqual(list(bi1.iter_select(np.full(len(bi1), False))),
                []
                )
        self.assertEqual(list(bi1.iter_select(np.full(len(bi1), True))),
                [(0, 0), (0, 1), (1, 0)]
                )

    #---------------------------------------------------------------------------

    def test_block_index_iter_select_sequence_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        self.assertEqual(list(bi1.iter_select([0, -1, -2, -8])),
                [(0, 0), (2, 4), (2, 3), (0, 0)]
                )
        self.assertEqual(list(bi1.iter_select(np.array([0, -1, -2, -8]))),
                [(0, 0), (2, 4), (2, 3), (0, 0)]
                )

    def test_block_index_iter_select_sequence_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        with self.assertRaises(IndexError):
            _ = list(bi1.iter_select([-9]))



    def test_block_index_iter_select_sequence_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(4).reshape(2,2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(10).reshape(2,5))

        with self.assertRaises(TypeError):
            _ = list(bi1.iter_select(['b', 'c']))



    def test_block_index_iter_select_sequence_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(10).reshape(2,5))

        sel = [0, 3, 4]
        it1 = iter(bi1.iter_select(sel))
        it2 = iter(it1)
        del sel
        del bi1
        del it1
        self.assertEqual(list(it2), [(0, 0), (0, 3), (0, 4)])


    #---------------------------------------------------------------------------

    def test_block_index_iter_contiguous_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(2))
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(2))

        self.assertEqual(
            list(bi1.iter_contiguous([1,2,6,7])),
            [(0, slice(1, 3, None)), (2, slice(2, 3, None)), (3, slice(0, 1, None))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous([7,6,2,1])),
            [(3, slice(0, 1, None)), (2, slice(2, 3, None)), (0, slice(2, 0, -1))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous([7, 6, 2, 1], ascending=True)),
            [(0, slice(1, 3, None)), (2, slice(2, 3, None)), (3, slice(0, 1, None))]
            )


    def test_block_index_iter_contiguous_b(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(16).reshape(2,8))

        self.assertEqual(
            list(bi1.iter_contiguous([0,1,6,7])),
            [(0, slice(0, 2, None)), (0, slice(6, 8, None))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(slice(None))),
            [(0, slice(0, 8, None))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(slice(1, 6))),
            [(0, slice(1, 6, None))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(slice(0, 8, 3))),
            [(0, slice(0, 1, None)), (0, slice(3, 4, None)), (0, slice(6, 7, None))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(slice(0, 8, 3), reduce=True)),
            [(0, 0), (0, 3), (0, 6)]
            )

    def test_block_index_iter_contiguous_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(16).reshape(2,8))

        with self.assertRaises(TypeError):
            list(bi1.iter_contiguous([0,1,6,7], False))


    def test_block_index_iter_contiguous_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(8).reshape(2,4))
        bi1.register(np.arange(8).reshape(2,4))

        self.assertEqual(
            list(bi1.iter_contiguous(slice(7,1,-1))),
            [(1, slice(3, None, -1)), (0, slice(3, 1, -1))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous(slice(7,1,-1), ascending=True)),
            [(0, slice(2, 4)), (1, slice(0, 4))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous(slice(8,1,-1), ascending=True)),
            [(0, slice(2, 4)), (1, slice(0, 4))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous(slice(8,None,-1), ascending=True)),
            [(0, slice(0, 4)), (1, slice(0, 4))]
            )

    def test_block_index_iter_contiguous_e1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))

        self.assertEqual(
            list(bi1.iter_contiguous([6, 0, 7])),
            [(6, slice(0, 1)), (0, slice(0, 1)), (7, slice(0, 1))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous([6, 0, 7], ascending=True)),
            [(0, slice(0, 1)), (6, slice(0, 1)), (7, slice(0, 1))]
            )

        self.assertEqual(
            list(bi1.iter_contiguous(np.array([6, 0, 7]))),
            [(6, slice(0, 1)), (0, slice(0, 1)), (7, slice(0, 1))]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(np.array([6, 0, 7]), ascending=True)),
            [(0, slice(0, 1)), (6, slice(0, 1)), (7, slice(0, 1))]
            )

    def test_block_index_iter_contiguous_e2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))

        self.assertEqual(
            list(bi1.iter_contiguous([6, 0, 7], reduce=True)),
            [(6, 0), (0, 0), (7, 0)]
            )
        self.assertEqual(
            list(bi1.iter_contiguous([6, 0, 7], ascending=True, reduce=True)),
            [(0, 0), (6, 0), (7, 0)]
            )

        self.assertEqual(
            list(bi1.iter_contiguous(np.array([6, 0, 7]), reduce=True)),
            [(6, 0), (0, 0), (7, 0)]
            )
        self.assertEqual(
            list(bi1.iter_contiguous(np.array([6, 0, 7]), ascending=True, reduce=True)),
            [(0, 0), (6, 0), (7, 0)]
            )


    def test_block_index_iter_contiguous_f1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(4).reshape(2,2))

        key = np.array([2, 3, 5])

        def gen1():
            yield from bi1.iter_select(key)
        post1 = list(gen1())
        self.assertEqual(post1, [(0, 2), (1, 0), (1, 2)])



    def test_block_index_iter_contiguous_f2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(4).reshape(2,2))
        key = np.array([2, 3, 5])

        def gen2():
            yield from bi1.iter_contiguous(key)
        post2 = list(gen2())

        post1 = list(bi1.iter_contiguous(key))
        self.assertEqual(post1, post2)



    def test_block_index_iter_contiguous_g(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(4).reshape(2,2))

        with self.assertRaises(TypeError):
            _ = list(bi1.iter_contiguous('a'))


    def test_block_index_iter_contiguous_h1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))

        sel = np.array([1, 1, 1, 0, 0, 0]).astype(bool)
        post1 = list(bi1.iter_contiguous(sel))
        post2 = list(bi1.iter_contiguous(sel, ascending=True))
        self.assertEqual(post1, post2)
        self.assertEqual(post1, [(0, slice(0, 3, None))])

    def test_block_index_iter_contiguous_h2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))

        sel = np.array([1, 0, 1, 0, 1, 0]).astype(bool)
        post1 = list(bi1.iter_contiguous(sel))
        post2 = list(bi1.iter_contiguous(sel, ascending=True))
        self.assertEqual(post1, post2)
        self.assertEqual(post1,
                [(0, slice(0, 1, None)),
                (0, slice(2, 3, None)),
                (1, slice(1, 2, None))])

        post3 = list(bi1.iter_contiguous(sel, ascending=True, reduce=True))
        self.assertEqual(post3,
                [(0, 0),
                (0, 2),
                (1, 1)])


    def test_block_index_iter_contiguous_i1(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))

        self.assertEqual(list(bi1.iter_select(slice(0, 0))), [])
        self.assertEqual(list(bi1.iter_contiguous(slice(0, 0))), [])

        self.assertEqual(list(bi1.iter_select(slice(30, 60, 2))), [])
        self.assertEqual(list(bi1.iter_contiguous(slice(30, 60, 2))), [])

    def test_block_index_iter_contiguous_i2(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(6).reshape(2,3))

        self.assertEqual(list(bi1.iter_select([])), [])
        self.assertEqual(list(bi1.iter_contiguous([])), [])

        self.assertEqual(list(bi1.iter_select(np.full(len(bi1), False))), [])
        self.assertEqual(list(bi1.iter_contiguous(np.full(len(bi1), False))), [])



    #---------------------------------------------------------------------------

    def test_block_index_iter_block_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(6).reshape(2,3))
        bi1.register(np.arange(2))
        bi1.register(np.arange(6).reshape(2,3))

        slc = slice(None)
        self.assertEqual(list(bi1.iter_block()), [(0, slc), (1, slc), (2, slc)])
        self.assertEqual(list(reversed(bi1.iter_block())), [(2, slc), (1, slc), (0, slc)])


    def test_block_index_iter_block_b(self) -> None:
        bi1 = BlockIndex()
        self.assertEqual(list(bi1.iter_block()), [])


    def test_block_index_iter_block_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))
        bi1.register(np.arange(2))

        slc = slice(None)
        self.assertEqual(list(bi1.iter_block()), [(i, slc) for i in range(8)])

    #---------------------------------------------------------------------------

    def test_block_index_shape_a(self) -> None:
        bi1 = BlockIndex()
        self.assertEqual(bi1.shape, (0, 0))
        self.assertEqual(bi1.rows, -1) # kept to show no assignemt

        bi1.register(np.array(()).reshape(2,0))
        self.assertEqual(bi1.shape, (2, 0))
        self.assertEqual(bi1.rows, 2)

        with self.assertRaises(ErrorInitTypeBlocks):
            bi1.register(np.array(()).reshape(3,0))
