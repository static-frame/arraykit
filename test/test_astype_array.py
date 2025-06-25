import unittest

import numpy as np

from arraykit import astype_array

class TestUnit(unittest.TestCase):

    def test_astype_array_a1(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a1.flags.writeable = False

        a2 = astype_array(a1, np.int64)
        self.assertEqual(id(a1), id(a2))


    def test_astype_array_a2(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a1.flags.writeable = False

        a2 = astype_array(a1, np.float64)
        self.assertNotEqual(id(a1), id(a2))
        self.assertEqual(a2.dtype, np.dtype(np.float64))


    def test_astype_array_a3(self) -> None:
        a1 = np.array([False, True, False])

        a2 = astype_array(a1, np.int8)
        self.assertEqual(a2.dtype, np.dtype(np.int8))
        self.assertTrue(a2.flags.writeable)

    def test_astype_array_b1(self) -> None:
        a1 = np.array(['2021', '2024'], dtype=np.datetime64)

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertTrue(a2.flags.writeable)
        self.assertEqual(list(a2), [np.datetime64('2021'), np.datetime64('2024')])


    def test_astype_array_b2(self) -> None:
        a1 = np.array(['2021', '1642'], dtype=np.datetime64)

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertTrue(a2.flags.writeable)
        self.assertEqual(list(a2), [np.datetime64('2021'), np.datetime64('1642')])


    def test_astype_array_b3(self) -> None:
        a1 = np.array(['2021', '2024', '1984', '1642'], dtype=np.datetime64).reshape((2, 2))

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertTrue(a2.flags.writeable)
        self.assertEqual(
                list(list(a) for a in a2),
                [[np.datetime64('2021'), np.datetime64('2024')], [np.datetime64('1984'), np.datetime64('1642')]])

    def test_astype_array_b4(self) -> None:
        a1 = np.array(['2021', '2024', '1532', '1984', '1642', '899'], dtype=np.datetime64).reshape((2, 3))

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertEqual(a2.shape, (2, 3))
        self.assertTrue(a2.flags.writeable)
        self.assertEqual(
                list(list(a) for a in a2),
                [[np.datetime64('2021'), np.datetime64('2024'), np.datetime64('1532')],
                 [np.datetime64('1984'), np.datetime64('1642'), np.datetime64('899')]])

    def test_astype_array_c(self) -> None:
        with self.assertRaises(TypeError):
            _ = astype_array([3, 4, 5], np.int64)


    def test_astype_array_d1(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a2 = astype_array(a1)

        self.assertEqual(a2.dtype, np.dtype(np.float64))
        self.assertEqual(a2.shape, (3,))
        self.assertTrue(a2.flags.writeable)


    def test_astype_array_d2(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a2 = astype_array(a1, None)

        self.assertEqual(a2.dtype, np.dtype(np.float64))
        self.assertEqual(a2.shape, (3,))
        self.assertTrue(a2.flags.writeable)



    def test_astype_array_d3(self) -> None:
        a1 = np.array([10, 20, 30], dtype=np.int64)
        a2 = astype_array(a1, np.int64)

        self.assertEqual(a2.dtype, np.dtype(np.int64))
        self.assertEqual(a2.shape, (3,))
        self.assertTrue(a2.flags.writeable)

        self.assertNotEqual(id(a1), id(a2))

    def test_astype_array_e(self) -> None:
        a1 = np.array(['2021', '2024', '1997', '1984', '2000', '1999'], dtype='datetime64[ns]').reshape((2, 3))

        a2 = astype_array(a1, np.object_)
        self.assertEqual(a2.dtype, np.dtype(np.object_))
        self.assertEqual(a2.shape, (2, 3))
        self.assertTrue(a2.flags.writeable)
        self.assertEqual(
                list(list(a) for a in a2),
                [[np.datetime64('2021-01-01T00:00:00.000000000'),
                  np.datetime64('2024-01-01T00:00:00.000000000'),
                  np.datetime64('1997-01-01T00:00:00.000000000')],
                  [np.datetime64('1984-01-01T00:00:00.000000000'),
                   np.datetime64('2000-01-01T00:00:00.000000000'),
                   np.datetime64('1999-01-01T00:00:00.000000000')]]
                 )

