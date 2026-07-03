import unittest

import numpy as np

from arraykit import factorize


def _eq_elem(x, y):
    # NaN/NaT-aware scalar equality (nan != nan under ==)
    try:
        if x != x and y != y:
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(x == y)
    except (TypeError, ValueError):
        return x is y


def roundtrips(uniques, codes, array):
    reconstructed = uniques[codes]
    if len(array) != len(codes):
        return False
    return all(_eq_elem(reconstructed[i], array[i]) for i in range(len(array)))


class TestUnit(unittest.TestCase):
    # ------------------------------------------------------------------
    # basic behavior

    def test_factorize_int_a(self) -> None:
        a = np.array([10, 10, 10, 20, 20, 30])
        uniques, codes = factorize(a)
        self.assertEqual(uniques.tolist(), [10, 20, 30])  # first-appearance order
        self.assertEqual(codes.tolist(), [0, 0, 0, 1, 1, 2])
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_first_appearance_order(self) -> None:
        a = np.array([30, 10, 10, 20, 30])
        uniques, codes = factorize(a)
        self.assertEqual(uniques.tolist(), [30, 10, 20])
        self.assertEqual(codes.tolist(), [0, 1, 1, 2, 0])

    def test_factorize_codes_dtype_intp(self) -> None:
        _, codes = factorize(np.array([1, 2, 3]))
        self.assertEqual(codes.dtype, np.dtype(np.intp))

    def test_factorize_codes_in_range(self) -> None:
        a = np.arange(50) % 7
        uniques, codes = factorize(a)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < len(uniques)).all())

    def test_factorize_outputs_immutable(self) -> None:
        uniques, codes = factorize(np.array([1, 1, 2]))
        self.assertFalse(uniques.flags.writeable)
        self.assertFalse(codes.flags.writeable)

    # ------------------------------------------------------------------
    # dtype coverage (each hash path)

    def test_factorize_int_widths(self) -> None:
        for dt in (np.int8, np.int16, np.int32, np.int64):
            a = np.array([1, 1, 2, -3, -3, 2], dtype=dt)
            uniques, codes = factorize(a)
            self.assertTrue(roundtrips(uniques, codes, a), dt)
            self.assertEqual(uniques.dtype, np.dtype(dt))

    def test_factorize_uint_widths(self) -> None:
        for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
            a = np.array([1, 1, 2, 200, 200], dtype=dt)
            uniques, codes = factorize(a)
            self.assertTrue(roundtrips(uniques, codes, a), dt)

    def test_factorize_float64(self) -> None:
        a = np.array([1.5, 1.5, 2.5, 0.0, -0.0, np.inf, -np.inf, np.inf])
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))
        # +0.0 and -0.0 are equal -> one group
        self.assertEqual(codes[3], codes[4])
        # +inf and -inf are distinct
        self.assertNotEqual(codes[5], codes[6])
        self.assertEqual(codes[5], codes[7])

    def test_factorize_float32(self) -> None:
        a = np.array([1, 1, 2, np.nan, np.nan], dtype=np.float32)
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_float16(self) -> None:
        a = np.array([1, 1, 2, 2, 3, np.nan, np.nan], dtype=np.float16)
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))
        self.assertEqual(codes[5], codes[6])  # NaN collapse

    def test_factorize_unicode(self) -> None:
        a = np.array(['a', 'a', 'bb', 'b', 'b', 'a'])
        uniques, codes = factorize(a)
        self.assertEqual(uniques.tolist(), ['a', 'bb', 'b'])
        self.assertEqual(codes.tolist(), [0, 0, 1, 2, 2, 0])
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_bytes(self) -> None:
        a = np.array([b'a', b'a', b'bb', b'b'], dtype='S2')
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_datetime64(self) -> None:
        a = np.array(
            ['2020-01-01', '2020-01-01', '2021-01-01', '2020-01-01'],
            dtype='datetime64[D]',
        )
        uniques, codes = factorize(a)
        self.assertEqual(codes.tolist(), [0, 0, 1, 0])
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_timedelta64(self) -> None:
        a = np.array([5, 5, 10, 10, 5], dtype='timedelta64[D]')
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))

    # ------------------------------------------------------------------
    # NaN / NaT semantics

    def test_factorize_float_nan_one_group(self) -> None:
        a = np.array([1.0, np.nan, np.nan, 2.0, np.nan])
        uniques, codes = factorize(a)
        nan_codes = {codes[1], codes[2], codes[4]}
        self.assertEqual(len(nan_codes), 1)  # all NaN share one code
        self.assertEqual(np.isnan(uniques).sum(), 1)  # exactly one NaN unique
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_datetime_nat_one_group(self) -> None:
        nat = np.datetime64('NaT')
        a = np.array([nat, nat, np.datetime64('2020'), nat], dtype='datetime64[Y]')
        _, codes = factorize(a)
        self.assertEqual(codes[0], codes[1])
        self.assertEqual(codes[0], codes[3])
        self.assertNotEqual(codes[0], codes[2])

    def test_factorize_object_nan_matches_float(self) -> None:
        a = np.array(
            ['a', 'a', float('nan'), float('nan'), None, None, 1, 1],
            dtype=object,
        )
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))
        self.assertEqual(codes[2], codes[3])       # float('nan') collapse
        self.assertEqual(codes[4], codes[5])       # None is its own group
        self.assertNotEqual(codes[2], codes[4])    # nan != None

    def test_factorize_object_mixed_types(self) -> None:
        a = np.array([1, '1', 1.0, 2], dtype=object)
        uniques, codes = factorize(a)
        self.assertTrue(roundtrips(uniques, codes, a))
        self.assertNotEqual(codes[0], codes[1])    # 1 (int) != '1' (str)
        self.assertEqual(codes[0], codes[2])       # 1 == 1.0 by value

    # ------------------------------------------------------------------
    # sort=True

    def test_factorize_sort_int(self) -> None:
        a = np.array([30, 10, 10, 20, 30])
        uniques, codes = factorize(a, sort=True)
        self.assertEqual(uniques.tolist(), [10, 20, 30])
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_sort_reconstruction_matches_unsorted(self) -> None:
        a = np.array([3.0, np.nan, 1.0, np.nan, 2.0, 1.0])
        u0, c0 = factorize(a, sort=False)
        u1, c1 = factorize(a, sort=True)
        # both reconstruct the same original array
        self.assertTrue(roundtrips(u0, c0, a))
        self.assertTrue(roundtrips(u1, c1, a))
        # sorted uniques are ascending (NaN sorts last)
        finite = u1[~np.isnan(u1)]
        self.assertEqual(finite.tolist(), sorted(finite.tolist()))

    def test_factorize_sort_strings(self) -> None:
        a = np.array(['c', 'a', 'b', 'a', 'c'])
        uniques, codes = factorize(a, sort=True)
        self.assertEqual(uniques.tolist(), ['a', 'b', 'c'])
        self.assertTrue(roundtrips(uniques, codes, a))

    def test_factorize_sort_is_keyword_only(self) -> None:
        a = np.array([1, 2, 3])
        with self.assertRaises(TypeError):
            factorize(a, True)  # type: ignore

    def test_factorize_sort_unorderable_object_raises(self) -> None:
        a = np.array([1, 'a', 2], dtype=object)
        with self.assertRaises(TypeError):
            factorize(a, sort=True)

    # ------------------------------------------------------------------
    # edge cases

    def test_factorize_empty(self) -> None:
        for a in (np.array([], dtype=np.int64), np.array([], dtype=float),
                  np.array([], dtype='U4'), np.array([], dtype=object)):
            uniques, codes = factorize(a)
            self.assertEqual(len(uniques), 0)
            self.assertEqual(len(codes), 0)
            self.assertEqual(codes.dtype, np.dtype(np.intp))
            self.assertEqual(uniques.dtype, a.dtype)

    def test_factorize_single(self) -> None:
        uniques, codes = factorize(np.array([7]))
        self.assertEqual(uniques.tolist(), [7])
        self.assertEqual(codes.tolist(), [0])

    def test_factorize_all_unique(self) -> None:
        a = np.array([1, 2, 3, 4])
        uniques, codes = factorize(a)
        self.assertEqual(codes.tolist(), [0, 1, 2, 3])
        self.assertEqual(uniques.tolist(), [1, 2, 3, 4])

    def test_factorize_all_same(self) -> None:
        a = np.array([5, 5, 5, 5])
        uniques, codes = factorize(a)
        self.assertEqual(uniques.tolist(), [5])
        self.assertEqual(codes.tolist(), [0, 0, 0, 0])

    def test_factorize_non_contiguous(self) -> None:
        # underlying [1,1,2,2,3,3]; [::2] view is [1,2,3]
        a = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)[::2]
        self.assertFalse(a.flags.c_contiguous)
        uniques, codes = factorize(a)
        self.assertEqual(uniques.tolist(), [1, 2, 3])
        self.assertEqual(codes.tolist(), [0, 1, 2])

    def test_factorize_mutable_input_accepted(self) -> None:
        a = np.array([1, 1, 2])
        self.assertTrue(a.flags.writeable)
        uniques, codes = factorize(a)  # must not require immutable input
        self.assertTrue(roundtrips(uniques, codes, a))

    # ------------------------------------------------------------------
    # input validation

    def test_factorize_requires_array(self) -> None:
        with self.assertRaises(TypeError):
            factorize([1, 2, 3])  # type: ignore

    def test_factorize_requires_1d(self) -> None:
        with self.assertRaises(TypeError):
            factorize(np.arange(4).reshape(2, 2))

    # ------------------------------------------------------------------
    # parity with numpy sort-based unique/inverse

    def test_factorize_parity_np_unique(self) -> None:
        rng = np.random.RandomState(0)
        for _ in range(50):
            a = rng.randint(0, 6, size=rng.randint(0, 40)).astype(np.int64)
            uniques, codes = factorize(a, sort=True)
            u_np, inv_np = np.unique(a, return_inverse=True)
            self.assertEqual(uniques.tolist(), u_np.tolist())
            self.assertEqual(codes.tolist(), inv_np.ravel().tolist())

    def test_factorize_parity_np_unique_strings(self) -> None:
        rng = np.random.RandomState(1)
        pool = np.array(['aa', 'bb', 'cc', 'dd', 'ee'])
        for _ in range(30):
            a = pool[rng.randint(0, 5, size=rng.randint(1, 40))]
            uniques, codes = factorize(a, sort=True)
            u_np, inv_np = np.unique(a, return_inverse=True)
            self.assertEqual(uniques.tolist(), u_np.tolist())
            self.assertEqual(codes.tolist(), inv_np.ravel().tolist())


if __name__ == '__main__':
    unittest.main()
