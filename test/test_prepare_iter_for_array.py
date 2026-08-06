import unittest
from enum import Enum

import numpy as np
from arraykit import prepare_iter_for_array


class Color(Enum):
    R = 1
    G = 2


class _Hinted:
    """An iterable exposing a (possibly inexact) __length_hint__."""

    def __init__(self, items, hint):
        self._items = list(items)
        self._hint = hint

    def __iter__(self):
        return iter(self._items)

    def __length_hint__(self):
        return self._hint


# faithful reference: static_frame.core.util.prepare_iter_for_array, given a precomputed
# copy flag (SF's is_gen_copy_values decision lives in the SF wrapper)
_INEXACT = (float, complex, np.inexact)
_BIG = 1_000_000_000_000_000


def _reference(values, copy):
    if copy:
        vpost = []
    resolved = None
    has_tuple = False
    has_str = has_non = has_inx = has_big = False
    it = iter(values)
    for v in it:
        if copy:
            vpost.append(v)
        vt = v.__class__
        if vt is str or vt is np.str_ or vt is bytes or vt is np.bytes_:
            has_str = True
        elif hasattr(v, '__len__'):
            has_tuple = True
            resolved = object
            break
        elif isinstance(v, Enum):
            resolved = object
            break
        else:
            has_non = True
            if vt in _INEXACT:
                has_inx = True
            elif vt is int and abs(v) > _BIG:
                has_big = True
        if (has_str and has_non) or (has_big and has_inx):
            resolved = object
            break
    if copy:
        vpost.extend(it)
        return resolved, has_tuple, vpost
    return resolved, has_tuple, values


class TestUnit(unittest.TestCase):
    def _check(self, make, copy):
        # make() returns a fresh iterable so the two runs are independent
        r_ak = prepare_iter_for_array(make(), copy)
        r_ref = _reference(make(), copy)
        self.assertIs(r_ak[0], r_ref[0])  # None or the object type, by identity
        self.assertEqual(r_ak[1], r_ref[1])  # has_tuple
        self.assertEqual(list(r_ak[2]), list(r_ref[2]))  # values
        return r_ak

    def test_list_str(self) -> None:
        r = self._check(lambda: ['a', 'b', 'c'], False)
        self.assertIsNone(r[0])

    def test_list_float(self) -> None:
        self._check(lambda: [1.0, 2.0, 3.0], False)

    def test_tuple_int(self) -> None:
        self._check(lambda: (1, 2, 3), False)

    def test_mixed_str_nonstr(self) -> None:
        r = self._check(lambda: [1, 'a', 2.0], False)
        self.assertIs(r[0], object)

    def test_sized_object_has_tuple(self) -> None:
        r = self._check(lambda: [1, (2, 3)], False)
        self.assertIs(r[0], object)
        self.assertTrue(r[1])  # has_tuple

    def test_enum(self) -> None:
        r = self._check(lambda: [Color.R, Color.G], False)
        self.assertIs(r[0], object)
        self.assertFalse(r[1])  # not has_tuple

    def test_bigint_and_inexact(self) -> None:
        r = self._check(lambda: [10**18, 1.5], False)
        self.assertIs(r[0], object)

    def test_bigint_only(self) -> None:
        r = self._check(lambda: [10**18, 2], False)
        self.assertIsNone(r[0])

    def test_numpy_float_scalars(self) -> None:
        r = self._check(lambda: [np.float64(1.5), np.float64(2.5)], False)
        self.assertIsNone(r[0])

    def test_bytes(self) -> None:
        self._check(lambda: [b'x', b'y'], False)

    def test_empty_list(self) -> None:
        r = self._check(list, False)
        self.assertIsNone(r[0])

    def test_generator_copy(self) -> None:
        # a generator is materialized when copy=True; the returned list is the values
        r = self._check(lambda: (str(i) for i in range(4)), True)
        self.assertEqual(list(r[2]), ['0', '1', '2', '3'])
        self.assertIsInstance(r[2], list)

    def test_generator_copy_mixed(self) -> None:
        r = self._check(lambda: (i if i < 2 else str(i) for i in range(4)), True)
        self.assertIs(r[0], object)
        self.assertEqual(list(r[2]), [0, 1, '2', '3'])

    def test_generator_empty_copy(self) -> None:
        self._check(lambda: (x for x in []), True)

    def test_set_copy(self) -> None:
        # a set is materialized (order-independent check of contents); it has __len__,
        # so the list is pre-sized
        r = prepare_iter_for_array({1, 2, 3}, True)
        self.assertIsNone(r[0])
        self.assertEqual(sorted(r[2]), [1, 2, 3])

    def test_dict_copy(self) -> None:
        r = prepare_iter_for_array({'a': 1, 'b': 2}, True)
        self.assertEqual(sorted(r[2]), ['a', 'b'])

    def test_length_hint_exact(self) -> None:
        r = self._check(lambda: _Hinted([1, 2, 3], 3), True)
        self.assertEqual(list(r[2]), [1, 2, 3])

    def test_length_hint_overestimate(self) -> None:
        # a hint larger than the actual length -> trailing slots dropped
        r = prepare_iter_for_array(_Hinted([1, 2, 3], 10), True)
        self.assertEqual(list(r[2]), [1, 2, 3])
        self.assertEqual(len(r[2]), 3)

    def test_length_hint_underestimate(self) -> None:
        # a hint smaller than the actual length -> remaining items appended
        r = prepare_iter_for_array(_Hinted([1, 2, 3, 4, 5], 1), True)
        self.assertEqual(list(r[2]), [1, 2, 3, 4, 5])

    def test_length_hint_inference_preserved(self) -> None:
        # inference still resolves object through the pre-sized path
        r = prepare_iter_for_array(_Hinted([1, (2,)], 5), True)
        self.assertIs(r[0], object)
        self.assertTrue(r[1])
        self.assertEqual(list(r[2]), [1, (2,)])

    def test_no_copy_returns_original(self) -> None:
        src = [1, 2, 3]
        r = prepare_iter_for_array(src, False)
        self.assertIs(r[2], src)  # original object, not a copy

    def test_early_stop_does_not_over_iterate(self) -> None:
        # once object is resolved, inspection stops; a later error-raising element in a
        # non-copy list is never inspected
        r = prepare_iter_for_array([1, (2,), object()], False)
        self.assertIs(r[0], object)
        self.assertTrue(r[1])

    def test_generator_copy_propagates_exception(self) -> None:
        def gen():
            yield 1
            raise ValueError('boom')

        with self.assertRaises(ValueError):
            prepare_iter_for_array(gen(), True)


if __name__ == '__main__':
    unittest.main()
