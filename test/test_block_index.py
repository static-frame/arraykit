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
            bi1.append('foo')

        with self.assertRaises(TypeError):
            bi1.append(3.5)

    def test_block_index_append_b(self) -> None:

        bi1 = BlockIndex()
        with self.assertRaises(TypeError):
            bi1.append(np.array(0))

        with self.assertRaises(TypeError):
            bi1.append(np.arange(12).reshape(2,3,2))


    def test_block_index_append_c(self) -> None:
        bi1 = BlockIndex()
        bi1.append(np.array((3, 4, 5)))
        bi1.append(np.array((3, 4, 5)))

        bi1.append(np.arange(6).reshape(3,2))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (2, 0), (2, 1)])

    def test_block_index_append_d(self) -> None:
        bi1 = BlockIndex()
        bi1.append(np.arange(2))
        bi1.append(np.arange(12).reshape(2,6))
        bi1.append(np.arange(2))
        bi1.append(np.arange(12).reshape(2,6))
        self.assertEqual(bi1.to_list(),
            [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5)]
            )


    def test_block_index_to_bytes_a(self) -> None:
        bi1 = BlockIndex()
        bi1.append(np.arange(6).reshape(2,3))
        bi1.append(np.arange(4).reshape(2,2))
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
