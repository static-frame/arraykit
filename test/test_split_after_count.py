
import unittest

from arraykit import split_after_count

class TestUnit(unittest.TestCase):


    #---------------------------------------------------------------------------
    def test_split_after_count_a(self) -> None:
        post = split_after_count('a,b,c,d,e', delimiter=',', count=2)
        self.assertEqual(post[0], 'a,b')
        self.assertEqual(post[1], 'c,d,e')

    def test_split_after_count_b(self) -> None:
        post = split_after_count('a,b,c,d,e', delimiter=',', count=4)
        self.assertEqual(post[0], 'a,b,c,d')
        self.assertEqual(post[1], 'e')

    def test_split_after_count_c(self) -> None:
        post = split_after_count('a,b,c,d,e', delimiter=',', count=5)
        self.assertEqual(post[0], 'a,b,c,d,e')
        self.assertEqual(post[1], '')

    def test_split_after_count_d(self) -> None:
        post = split_after_count('a', delimiter=',', count=5)
        self.assertEqual(post[0], 'a')
        self.assertEqual(post[1], '')

    def test_split_after_count_f(self) -> None:
        post = split_after_count('a,', delimiter=',', count=1)
        self.assertEqual(post[0], 'a')
        self.assertEqual(post[1], '')

    def test_split_after_count_g(self) -> None:
        post = split_after_count(',', delimiter=',', count=1)
        self.assertEqual(post[0], '')
        self.assertEqual(post[1], '')

    def test_split_after_count_h(self) -> None:
        post = split_after_count('a,b,c,d,e', delimiter='|', count=5)
        self.assertEqual(post[0], 'a,b,c,d,e')
        self.assertEqual(post[1], '')


    #---------------------------------------------------------------------------
    def test_split_after_count_exception_a(self) -> None:
        with self.assertRaises(ValueError):
            post = split_after_count(3, delimiter=',', count=2)

    def test_split_after_count_exception_b(self) -> None:
        with self.assertRaises(ValueError):
            post = split_after_count('a,', delimiter=',', count=0)

    def test_split_after_count_exception_c(self) -> None:
        with self.assertRaises(TypeError):
            post = split_after_count('a,b,c', delimiter='foo', count=2)

    def test_split_after_count_exception_d(self) -> None:
        with self.assertRaises(TypeError):
            post = split_after_count('a,b,c', escapechar=32, count=2)

    def test_split_after_count_exception_e(self) -> None:
        with self.assertRaises(TypeError):
            post = split_after_count('a,b,c', quoting='234', count=2)




if __name__ == '__main__':
    unittest.main()
