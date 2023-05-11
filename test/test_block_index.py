
import unittest
import ctypes
import sys
import pickle

import numpy as np

from arraykit import BlockIndex
from arraykit import ErrorInitBlocks


class TestUnit(unittest.TestCase):

    def test_block_index_init_a(self) -> None:
        bi1 = BlockIndex()
        # print(bi1)

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
        with self.assertRaises(ErrorInitBlocks):
            bi1.register('foo')

        with self.assertRaises(ErrorInitBlocks):
            bi1.register(3.5)

    def test_block_index_register_b(self) -> None:

        bi1 = BlockIndex()
        with self.assertRaises(ErrorInitBlocks):
            bi1.register(np.array(0))

        with self.assertRaises(ErrorInitBlocks):
            bi1.register(np.arange(12).reshape(2,3,2))


    def test_block_index_register_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.arange(6).reshape(3,2))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (2, 0), (2, 1)])
        self.assertEqual(bi1.shape, (3, 4))

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

    def test_block_index_register_e(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        with self.assertRaises(ErrorInitBlocks):
            bi1.register(np.arange(12).reshape(3,4))


    def test_block_index_register_f(self) -> None:
        bi1 = BlockIndex()
        a1 = np.arange(20000).reshape(2, 10_000) #.reshape(2, 10_000)
        bi1.register(a1)


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

        bi2 = bi1.copy()
        self.assertEqual(bi1.to_list(), bi2.to_list())


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

        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(bi1.shape, (2, 8))


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
