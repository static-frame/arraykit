import unittest

from arraykit import BlockIndex


class TestUnit(unittest.TestCase):

    def test_block_index_a(self) -> None:
        bi1 = BlockIndex()
        print(bi1)