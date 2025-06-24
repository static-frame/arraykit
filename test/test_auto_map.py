import pickle
import pytest
import numpy as np

from arraykit import AutoMap
from arraykit import FrozenAutoMap
from arraykit import NonUniqueError


# ------------------------------------------------------------------------------


def test_am_extend():
    am1 = AutoMap(("a", "b"))
    am2 = am1 | AutoMap(("c", "d"))
    assert list(am2.keys()) == ["a", "b", "c", "d"]


def test_am_add():
    a = AutoMap()
    for l, key in enumerate(["a", "b", "c", "d"]):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


def test_fam_contains():
    x = []
    fam = FrozenAutoMap(("a", "b", "c"))
    assert (x in fam.values()) == False
    # NOTE: exercise x to force seg fault
    assert len(x) == 0


# ------------------------------------------------------------------------------


def test_fam_constructor_a():
    with pytest.raises(ZeroDivisionError):
        fam = FrozenAutoMap((x / 0 for x in range(3)))


def test_fam_constructor_b():
    fam1 = FrozenAutoMap(range(3))
    fam2 = FrozenAutoMap(fam1)
    assert list(fam2), [0, 1, 2]


# ------------------------------------------------------------------------------


def test_fam_constructor_array_int_a1():
    a1 = np.array((10, 20, 30), dtype=np.int64)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_a2():
    a1 = np.array((10, 20, 30), dtype=np.int32)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_b():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64).reshape(2, 2)
    a1.flags.writeable = False
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_c():
    a1 = np.array((10, 20, 30), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_constructor_array_int_d():
    a1 = np.array((-2, -1, 1, 2), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_constructor_array_int_e():
    # https://github.com/static-frame/arraymap/issues/12
    a1 = np.array((0, 0, 1, 1, 2, 2), dtype=int)
    a2 = a1[[0, 2, 4]]
    a2.flags.writeable = False
    fam1 = FrozenAutoMap(a2)
    assert list(fam1) == [0, 1, 2]

    d1 = {i: int(i) for i in a2}
    fam2 = FrozenAutoMap(d1)
    assert list(fam2) == [0, 1, 2]

    d2 = {0: 0, 3: 1}
    fam3 = FrozenAutoMap(d2)
    assert list(fam3) == [0, 3]


# ------------------------------------------------------------------------------


def test_fam_constructor_array_float_a():
    a1 = np.array((1.2, 8.8, 1.2))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


# ------------------------------------------------------------------------------


def test_fam_constructor_array_dt64_a():
    a1 = np.array(("1970-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam[np.datetime64("2023-05")] == 1
    assert fam[np.datetime64("1970-01")] == 0

    with pytest.raises(KeyError):
        fam[np.datetime64("nat")]

    with pytest.raises(KeyError):
        fam[np.datetime64("1970")]


def test_fam_constructor_array_dt64_b():
    a1 = np.array(("1542", "nat"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[np.datetime64("nat")] == 1
    assert fam[np.datetime64("nat", "D")] == 1
    assert fam[np.datetime64("nat", "ns")] == 1
    assert fam[np.datetime64("1542")] == 0


def test_fam_constructor_array_dt64_c():
    a1 = np.array(("nat", "nat"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    # when we get "generic" dt64 units, we load scalars in a list, and can thus support multiple NaNs
    assert len(fam) == 2


def test_fam_constructor_array_dt64_d():
    a1 = np.array(("2023-05", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


# ------------------------------------------------------------------------------


def test_fam_constructor_array_unicode_a():
    a1 = np.array(("a", "b", "a"))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_unicode_b():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_constructor_array_unicode_c():
    a1 = np.array(("z0Ct", "z0DS", "z0E9"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)


# NOTE
# >>> u = "\x000\x00"
# >>> len(u)
# 3
# >>> a1 = np.array(['', ''], dtype='U4')
# >>> a1[0] = u
# >>> a1
# array(['\x000', ''], dtype='<U4')
# >>> len(a1[0])
# 2


def test_fam_constructor_array_unicode_d1():
    a1 = np.array(["", "\x000"], dtype="U2")
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 2
    assert list(fam) == ["", "\x000"]
    assert "" in fam
    assert "\x000" in fam


def test_fam_constructor_array_unicode_d2():
    a1 = np.array(["", "\x000\x00"], dtype="U3")
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 2
    assert list(fam) == ["", "\x000"]  # we lost the last null
    assert "" in fam
    assert "\x000" in fam


def test_fam_copy_array_unicode_a():
    a1 = np.array(("a", "ccc", "bb"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)
    fam2 = FrozenAutoMap(fam1)
    assert fam2["a"] == 0
    assert fam2["ccc"] == 1
    assert fam2["bb"] == 2


# ------------------------------------------------------------------------------


def test_fam_constructor_array_bytes_a():
    a1 = np.array((b"a", b"b", b"c"))
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_bytes_b():
    a1 = np.array((b"aaa", b"b", b"aaa"))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_bytes_c():
    a1 = np.array((b"aaa", b"b", b"cc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[b"aaa"] == 0
    assert fam[b"b"] == 1
    assert fam[b"cc"] == 2


def test_fam_copy_array_bytes_a():
    a1 = np.array((b"a", b"ccc", b"bb"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)
    fam2 = FrozenAutoMap(fam1)
    assert fam2[b"a"] == 0
    assert fam2[b"ccc"] == 1
    assert fam2[b"bb"] == 2


# ------------------------------------------------------------------------------


def test_fam_array_bytes_get_a():
    a1 = np.array((b"", b"  ", b"   ", b"    "))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get(b"") == 0
    assert fam.get(b" ") == None
    assert fam.get(b"   ") == 2
    assert fam.get(b"    ") == 3


# ------------------------------------------------------------------------------


def test_fam_array_len_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 4


def test_fam_array_len_b():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[10] == 0
    assert fam[20] == 1
    assert fam[30] == 2
    assert fam[40] == 3


# ------------------------------------------------------------------------------


def test_fam_array_int_get_a():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0


def test_fam_array_int_get_b():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0
    assert fam.get(1.1) is None


def test_fam_array_int_get_c1():
    a1 = np.array((1, 5, 10, 20), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3


def test_fam_array_int_get_c2():
    a1 = np.array((1,), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_array_int_get_c3():
    a1 = np.array((19037,), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_array_int_get_d():
    a1 = np.array((1, 5, 10, 20), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3
    assert fam.get(20.1) is None


def test_fam_array_int_get_e():
    a1 = np.array([2147483648], dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(2147483648) == 0
    assert fam.get(a1[0]) == 0


def test_fam_array_int_get_f1():
    ctype = np.int64
    a1 = np.array([np.iinfo(ctype).min, np.iinfo(ctype).max], dtype=ctype)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == [np.iinfo(ctype).min, np.iinfo(ctype).max]


def test_fam_array_int_get_f2():
    ctype = np.int64
    a1 = np.array([np.iinfo(ctype).min, np.iinfo(ctype).max], dtype=ctype)
    a1.flags.writeable = False

    fam = FrozenAutoMap(a1)
    assert fam.get(np.iinfo(ctype).min) == 0
    assert fam.get(np.iinfo(ctype).max) == 1


def test_fam_array_int_get_d():
    a1 = np.array((8, 2, 4, 0, 1), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for ctype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ):
        a2 = a1.astype(ctype)
        for k in a2:
            assert k in fam, f"{type(k)}"
    assert 2.0 in fam
    assert 2.1 not in fam
    assert True in fam
    assert False in fam
    assert 4 in fam


def test_fam_array_int_get_e():
    a1 = np.array((1,), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert a1[0] in fam
    assert 1 in fam


# ------------------------------------------------------------------------------


def test_fam_array_uint_get_a():
    a1 = np.array((1, 100, 300, 4000), dtype=np.uint64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_b():
    a1 = np.arange(0, 100, dtype=np.uint32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_c():
    a1 = np.arange(0, 100, dtype=np.uint16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_d():
    a1 = np.arange(0, 100, dtype=np.uint8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_e():
    a1 = np.array((1,), dtype=np.uint16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_f():
    a1 = np.array((8, 2, 4, 1), dtype=np.uint64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for ctype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ):
        a2 = a1.astype(ctype)
        for k in a2:
            assert k in fam, f"{type(k)}"
        a3 = -a2
        for k in a3:
            assert k not in fam, f"{type(k)}"

    assert True in fam
    assert 4.0 in fam
    assert 4.1 not in fam
    assert 8 in fam
    assert -8 not in fam


# ------------------------------------------------------------------------------


def test_fam_array_float_get_a():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1.5) == 0
    assert fam.get(10.2) == 1
    assert fam.get(a1[1]) == 1
    assert fam.get(8.8) == 2


def test_fam_array_float_get_b():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    # assert fam.get(1.5) == 0
    assert fam.get(a1[0]) == 0
    assert fam.get(a1[1]) == 1
    assert fam.get(a1[2]) == 2


def test_fam_array_float_get_c1():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam.get("f") is None
    assert fam.get(a1[0]) == 0
    assert fam.get(a1[1]) == 1
    assert fam.get(a1[2]) == 2


def test_fam_array_float_get_c2():
    a1 = np.array((0.0,), dtype=np.float16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)


def test_fam_array_float_get_d():
    a1 = np.array((8, 2, 4, 1), dtype=np.float64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for ctype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ):
        a2 = a1.astype(ctype)
        for k in a2:
            assert k in fam, f"{type(k)}"
        a3 = -a2
        for k in a3:
            assert k not in fam, f"{type(k)}"

    assert True in fam
    assert 4.0 in fam
    assert 4.1 not in fam
    assert 8 in fam
    assert -8 not in fam


# ------------------------------------------------------------------------------


def test_fam_array_unicode_get_a():
    a1 = np.array(("bb", "a", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("a") == 1
    assert fam.get("bb") == 0
    assert fam.get("ccc") == 2
    assert fam.get(None) is None
    assert fam.get(3.2) is None
    assert fam.get("cc") is None
    assert fam.get("cccc") is None


def test_fam_array_unicode_get_b():
    a1 = np.array(("", "  ", "   ", "    "))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("") == 0
    assert fam.get(" ") == None
    assert fam.get("   ") == 2
    assert fam.get("    ") == 3


# ------------------------------------------------------------------------------


def test_fam_array_values_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == [10, 20, 30, 40]


def test_fam_array_keys_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    e = next(iter(fam))
    assert isinstance(e, np.int8)


def test_fam_array_items_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [(10, 0), (20, 1), (30, 2), (40, 3)]


def test_fam_array_values_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == ["a", "b", "c", "d"]


def test_fam_array_items_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [("a", 0), ("b", 1), ("c", 2), ("d", 3)]


def test_fam_array_items_c():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)

    fam2 = FrozenAutoMap(fam1)
    assert list(fam2.items()) == [("a", 0), ("b", 1), ("c", 2)]
    assert list(fam1.items()) == [("a", 0), ("b", 1), ("c", 2)]


# ------------------------------------------------------------------------------


def test_am_array_constructor_a():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    am1 = AutoMap(a1)


def test_am_array_constructor_b():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    am1 = AutoMap(a1)
    assert am1[np.datetime64("2023-05")] == 1


def test_am_array_constructor_c():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    am = AutoMap(a1)
    am.update((60, 80))
    am.add(90)
    assert list(am.keys()) == [10, 20, 30, 40, 60, 80, 90]


# ------------------------------------------------------------------------------


def test_fam_array_pickle_a():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)
    fam2 = pickle.loads(pickle.dumps(fam1))
    assert list(fam1.values()) == list(fam2.values())


# ------------------------------------------------------------------------------


def test_fam_array_get_all_a():
    a1 = np.array((1, 100, 300, 4000))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    with pytest.raises(TypeError):
        fam.get_all((3, 3))

    with pytest.raises(TypeError):
        fam.get_all("a")

    with pytest.raises(TypeError):
        fam.get_all(None)


def test_fam_array_get_all_b():
    a1 = np.array((1, 100, 300, 4000))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_all([300, 100])
    assert post1.tolist() == [2, 1]
    x = [y for y in post1]
    del x
    del post1
    post2 = fam.get_all([4000, 4000, 4000])
    assert post2.tolist() == [3, 3, 3]
    x = [y for y in post2]
    del x


def test_fam_array_get_all_c():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    with pytest.raises(KeyError):
        fam.get_all(["bb", "c"])


def test_fam_array_get_all_d1():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_all(np.array(("bb", "a", "ccc", "a", "bb")))
    assert post1.tolist() == [1, 0, 2, 0, 1]
    assert post1.flags.writeable == False


def test_fam_array_get_all_d2():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    with pytest.raises(KeyError):
        fam.get_all(np.array(("bb", "a", "ccc", "aa")))


def test_fam_array_get_all_e():
    a1 = np.array((2,), dtype=np.uint64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam.get_all([2]) == [0]
    assert fam.get_all(a1) == [0]


def test_fam_array_get_all_f1():
    a1 = np.array(("a", "bb", "ccc", "dd"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_all(np.array(["ccc", "dd", "bb", "bb"]))
    assert post.tolist() == [2, 3, 1, 1]


def test_fam_array_get_all_f2():
    a1 = np.array(("a", "bb", "ccc", "dd"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    with pytest.raises(KeyError):
        fam.get_all(np.array(["bb", "c"]))


def test_fam_array_get_all_g1():
    a1 = np.array((b"a", b"bb", b"ccc", b"dd"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_all(np.array([b"ccc", b"dd", b"bb", b"bb"]))
    assert post.tolist() == [2, 3, 1, 1]


def test_fam_array_get_all_g2():
    a1 = np.array((b"a", b"bb", b"ccc", b"dd"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    with pytest.raises(KeyError):
        fam.get_all(np.array([b"dd", b"x"]))


def test_fam_array_get_all_h():
    a1 = np.array((b"a", b""))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_all(np.array([b"", b"", b"a"]))
    assert post.tolist() == [1, 1, 0]


def test_fam_array_get_all_i():
    a1 = np.array((b"foo", b"bar"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    with pytest.raises(KeyError):
        _ = fam.get_all(np.array([b"fo", b"ba"]))

    with pytest.raises(KeyError):
        _ = fam.get_all(np.array([b"", b""]))


def test_fam_array_get_all_j():
    a1 = np.array(("aaaaa", "bb", "ccc", "dd"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    with pytest.raises(KeyError):
        _ = fam.get_all(np.array(["a", "b"]))

    assert fam.get_all(np.array(("bb", "dd", "bb", "dd"))).tolist() == [1, 3, 1, 3]


def test_fam_array_get_all_k1():
    a1 = np.array(("2023-01-05", "1854-05-02"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_all(
        np.array(["1854-05-02", "2023-01-05", "2023-01-05"], np.datetime64)
    )
    assert post.tolist() == [1, 0, 0]


def test_fam_array_get_all_k2():
    a1 = np.array(("2023-01-05", "1854-05-02"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    with pytest.raises(KeyError):
        post = fam.get_all(
            np.array(["1854-05-02", "2023-01-05", "2020-01-05"], np.datetime64)
        )


def test_fam_array_get_all_l():
    a1 = np.array(("2023-01-05", "1854-05-02", "1988-01-01"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    with pytest.raises(KeyError):
        _ = fam.get_all(np.array(["2022-01", "2023-01", "1988-01"], np.datetime64))


def test_fam_array_get_all_m1():
    # NOTE: small than 64bit arrays in FAMs do not get optimal array lookup performance
    a1 = np.array((1, 100, 300), dtype=np.int32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_all(np.array([300, 100], dtype=np.int64))
    assert post1.tolist() == [2, 1]


def test_fam_array_get_all_m2():
    a1 = np.array((1, 100, 300), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_all(np.array([300, 100], dtype=np.int64))
    assert post1.tolist() == [2, 1]


def test_fam_array_get_all_m3():
    a1 = np.array((1, 100, 30), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_all(np.array([30, 100], dtype=np.int64))
    assert post1.tolist() == [2, 1]

    post2 = fam.get_all(np.array([30, 100], dtype=np.int8))
    assert post2.tolist() == [2, 1]


# -------------------------------------------------------------------------------


def test_fam_array_get_any_a1():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_any(["bbb", "ccc", "a", "bbb"])
    assert post1 == [2, 0]

    post2 = fam.get_any(["bbb", "bbb"])
    assert post2 == []


def test_fam_array_get_any_a2():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_any(np.array(("bbb", "a", "ccc", "aa", "bbb")))
    assert post1 == [0, 2]


def test_fam_array_get_any_a3():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    post1 = fam.get_any(np.array(["bbb", "ccc", "a", "bbb"]))
    assert post1 == [2, 0]

    post2 = fam.get_any(np.array(["bbb", "bbb"]))
    assert post2 == []


def test_fam_array_get_any_b():
    a1 = np.array([4294967295], dtype=np.uint32)
    a1.flags.writeable = False
    a1_list = list(a1)
    fam = FrozenAutoMap(a1)
    assert a1[0] in fam
    assert 4294967295 in fam

    post1 = fam.get_any(a1_list)
    assert post1 == list(fam.values())


def test_fam_array_get_any_c1():
    a1 = np.array(("2023-01-05", "1854-05-02"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_any(
        np.array(
            ["1854-05-02", "nat", "1854-05-02", "2023-01-05", "nat"], np.datetime64
        )
    )
    assert post == [1, 1, 0]


def test_fam_array_get_any_c2():
    a1 = np.array(("2023-01-05", "1854-05-02"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_any(
        np.array(["1854-05-02", "2023-01-05", "2020-01-05"], np.datetime64)
    )
    assert post == [1, 0]


def test_fam_array_get_any_d():
    a1 = np.array(("2023-01-05", "1854-05-02", "1988-01-01"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    post = fam.get_any(np.array(["2022-01", "2023-01", "1988-01"], np.datetime64))
    assert post == []


def test_fam_get_dt64_a():
    a1 = np.array(("2023", "1854", "1988"), np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    k1 = np.datetime64("1988-01-01")
    with pytest.raises(KeyError):
        _ = fam[k1]

    k2 = np.datetime64("2023-01-01")
    with pytest.raises(KeyError):
        _ = fam[k2]

def test_fam_get_dt64_b():
    a1 = np.array(("2023", "1854", "1988"), np.datetime64)
    fam = FrozenAutoMap(list(a1))

    k1 = np.datetime64("1988-01-01")
    with pytest.raises(KeyError):
        _ = fam[k1]

    k2 = np.datetime64("2023-01-01")
    with pytest.raises(KeyError):
        _ = fam[k2]


def test_am_get_dt64_a():
    a1 = np.array(("2023", "1854", "1988"), np.datetime64)
    a1.flags.writeable = False
    fam = AutoMap(a1)

    k1 = np.datetime64("1988-01-01")
    with pytest.raises(KeyError):
        _ = fam[k1]

    k2 = np.datetime64("2023-01-01")
    with pytest.raises(KeyError):
        _ = fam[k2]

