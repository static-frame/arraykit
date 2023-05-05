import unittest

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


    def test_block_index_append_b(self) -> None:
        bi1 = BlockIndex()
        bi1.append(np.array((3, 4, 5)))

        bi1.append(np.array((3, 4, 5, 2)))

        bi1.append(np.arange(10).reshape(2,5))
