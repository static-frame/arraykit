import unittest
import ctypes
import sys

import numpy as np

from arraykit import BlockIndex


class TestUnit(unittest.TestCase):

    def test_block_index_a(self) -> None:
        bi1 = BlockIndex()
        # print(bi1)


    def test_block_index_append_a(self) -> None:
        bi1 = BlockIndex()
        with self.assertRaises(TypeError):
            bi1.register('foo')

        with self.assertRaises(TypeError):
            bi1.register(3.5)

    def test_block_index_append_b(self) -> None:

        bi1 = BlockIndex()
        with self.assertRaises(TypeError):
            bi1.register(np.array(0))

        with self.assertRaises(TypeError):
            bi1.register(np.arange(12).reshape(2,3,2))


    def test_block_index_append_c(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.array((3, 4, 5)))
        bi1.register(np.arange(6).reshape(3,2))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (2, 0), (2, 1)])

    def test_block_index_append_d(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(2))
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(2))
        bi1.register(np.arange(12).reshape(2,6))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
            )


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


    def test_block_index_copy_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))

        bi2 = bi1.copy()
        self.assertEqual(bi1.to_list(), bi2.to_list())


    def test_block_index_len_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(len(bi1), 8)

    def test_block_index_getitem_a(self) -> None:
        bi1 = BlockIndex()
        bi1.register(np.arange(12).reshape(2,6))
        bi1.register(np.arange(4).reshape(2,2))
        self.assertEqual(bi1[3], (0, 3))
        self.assertEqual(bi1[7], (1, 1))

        with self.assertRaises(IndexError):
            bi1[8]


