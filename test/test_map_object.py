import unittest
from enum import Enum

import numpy as np
from arraykit import map_object


class Color(Enum):
    R = 1
    G = 2


# reference: prepare_iter_for_array's inference + build, as static_frame applies it
_INEXACT = (float, complex, np.inexact)
_BIG = 1_000_000_000_000_000


def _reference(arr, func):
    vals = [func(v) for v in arr]
    resolved = None
    has_str = has_non = has_inx = has_big = False
    for v in vals:
        vt = v.__class__
        if vt is str or vt is np.str_ or vt is bytes or vt is np.bytes_:
            has_str = True
        elif hasattr(v, '__len__') or isinstance(v, Enum):
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
    return np.array(vals) if resolved is None else np.array(vals, dtype=object)


class TestUnit(unittest.TestCase):
    def _check(self, arr, func):
        post = map_object(arr, func)
        exp = _reference(arr, func)
        self.assertEqual(post.dtype, exp.dtype, (arr.dtype, exp.dtype))
        self.assertTrue(np.array_equal(post, exp))
        self.assertFalse(post.flags.writeable)
        return post

    def test_map_object_str_from_float(self) -> None:
        post = self._check(np.array([1.5, 2.25, 3.0]), lambda x: str(x))
        self.assertEqual(post.dtype.kind, 'U')

    def test_map_object_str_from_bool(self) -> None:
        post = self._check(np.array([True, False, True]), lambda x: str(x))
        self.assertEqual(post.tolist(), ['True', 'False', 'True'])
        self.assertEqual(post.dtype.kind, 'U')

    def test_map_object_str_from_int(self) -> None:
        self._check(np.array([1, 2, 3], dtype=np.int64), lambda x: str(x))

    def test_map_object_native_float(self) -> None:
        post = self._check(np.array([1.5, 2.5]), lambda x: float(x) * 2)
        self.assertEqual(post.dtype, np.dtype(np.float64))

    def test_map_object_native_int(self) -> None:
        # python-int results auto-detect to the platform default int (int32 on Windows)
        post = self._check(np.array([1, 2, 3]), lambda x: int(x) + 1)
        self.assertEqual(post.dtype, np.dtype(np.int_))

    def test_map_object_tuple_result(self) -> None:
        post = self._check(np.array([1, 2]), lambda x: (int(x), int(x)))
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_list_result(self) -> None:
        post = self._check(np.array([1, 2]), lambda x: [int(x)])
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_mixed_str_nonstr(self) -> None:
        post = self._check(np.array([1, 2, 3]), lambda x: str(x) if x > 1 else int(x))
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_python_float(self) -> None:
        self._check(np.array([1, 2, 3]), lambda x: 1.5)

    def test_map_object_bigint_and_inexact(self) -> None:
        # a large python int mixed with a python float -> object
        post = self._check(np.array([1, 2]), lambda x: 10**18 if x == 1 else 1.5)
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_bigint_only(self) -> None:
        # big ints alone (no inexact) do not force object
        post = self._check(np.array([1, 2]), lambda x: 10**18)
        self.assertNotEqual(post.dtype, np.dtype(object))

    def test_map_object_enum_result(self) -> None:
        post = self._check(np.array([1, 2]), lambda x: Color.R)
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_object_input(self) -> None:
        arr = np.array(['a', 'bb', 'ccc'], dtype=object)
        post = self._check(arr, lambda x: len(x))
        self.assertEqual(post.tolist(), [1, 2, 3])

    def test_map_object_receives_numpy_scalar(self) -> None:
        # elements are boxed as numpy scalars, matching Series/array iteration
        seen = []
        map_object(np.array([1.5, 2.5]), lambda x: seen.append(type(x)) or x)
        self.assertTrue(all(t is np.float64 for t in seen))

    def test_map_object_str_subclass_is_object(self) -> None:
        # a str subclass is not an exact str -> sized object -> object array
        class S(str):
            pass

        post = map_object(np.array([1, 2]), lambda x: S(str(x)))
        self.assertEqual(post.dtype, np.dtype(object))

    def test_map_object_empty(self) -> None:
        post = self._check(np.array([], dtype=np.float64), lambda x: str(x))
        self.assertEqual(len(post), 0)

    def test_map_object_strided_non_contiguous(self) -> None:
        # a strided slice (non-contiguous) must be walked correctly by the running pointer
        base = np.array([1.0, 99.0, 2.0, 99.0, 3.0])
        strided = base[::2]
        self.assertFalse(strided.flags['C_CONTIGUOUS'])
        post = self._check(strided, lambda x: str(x))
        self.assertEqual(post.tolist(), ['1.0', '2.0', '3.0'])

    def test_map_object_strided_object(self) -> None:
        arr = np.array(['a', 'X', 'bb', 'X', 'ccc'], dtype=object)[::2]
        post = self._check(arr, lambda x: len(x))
        self.assertEqual(post.tolist(), [1, 2, 3])

    def test_map_object_propagates_exception(self) -> None:
        def bad(x):
            raise ValueError('boom')

        with self.assertRaises(ValueError):
            map_object(np.array([1, 2]), bad)

    def test_map_object_errors(self) -> None:
        with self.assertRaises(ValueError):  # 2d
            map_object(np.array([[1, 2]]), lambda x: x)
        with self.assertRaises(TypeError):  # not callable
            map_object(np.array([1, 2]), 3)
        with self.assertRaises(TypeError):  # not an array
            map_object([1, 2], lambda x: x)


if __name__ == '__main__':
    unittest.main()
