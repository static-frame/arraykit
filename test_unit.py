import pytest  # type: ignore
import numpy as np  # type: ignore

import arraykit as ak


def test_array_init_a() -> None:
    with pytest.raises(NotImplementedError):
        ak.ArrayGO(np.array((3, 4, 5)))


def test_array_append_a() -> None:
    ag1 = ak.ArrayGO(('a', 'b', 'c', 'd'))
    assert [*ag1] == ['a', 'b', 'c', 'd']
    assert ag1.values.tolist() == ['a', 'b', 'c', 'd']
    ag1.append('e')
    ag1.extend(('f', 'g'))
    assert [*ag1] == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert ag1.values.tolist() == ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def test_array_append_b() -> None:
    ag1 = ak.ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
    assert [*ag1] == ['a', 'b', 'c', 'd']
    assert ag1.values.tolist() == ['a', 'b', 'c', 'd']
    ag1.append('e')
    ag1.extend(('f', 'g'))
    assert [*ag1] == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert ag1.values.tolist() == ['a', 'b', 'c', 'd', 'e', 'f', 'g']


def test_array_getitem_a() -> None:
    a = np.array(('a', 'b', 'c', 'd'), object)
    a.flags.writeable = False
    ag1 = ak.ArrayGO(a)
    # Ensure no copy for immutable:
    assert ak.mloc(ag1.values) == ak.mloc(a)
    ag1.append('b')
    post = ag1[ag1.values == 'b']
    assert post.tolist() == ['b', 'b']
    assert ag1[[2, 1, 1, 1]].tolist() == ['c', 'b', 'b', 'b']


def test_array_copy_a() -> None:
    ag1 = ak.ArrayGO(np.array(('a', 'b', 'c', 'd'), dtype=object))
    ag1.append('e')
    ag2 = ag1.copy()
    ag1.extend(('f', 'g'))
    assert ag1.values.tolist() == ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    assert ag2.values.tolist() == ['a', 'b', 'c', 'd', 'e']


def test_array_len_a() -> None:
    ag1 = ak.ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
    ag1.append('e')
    assert len(ag1) == 5


def test_resolve_dtype_a() -> None:
    a1 = np.array([1, 2, 3])
    a2 = np.array([False, True, False])
    a3 = np.array(['b', 'c', 'd'])
    a4 = np.array([2.3, 3.2])
    a5 = np.array(['test', 'test again'], dtype='S')
    a6 = np.array([2.3, 5.4], dtype='float32')
    assert ak.resolve_dtype(a1.dtype, a1.dtype) == a1.dtype
    assert ak.resolve_dtype(a1.dtype, a2.dtype) == np.object_
    assert ak.resolve_dtype(a2.dtype, a3.dtype) == np.object_
    assert ak.resolve_dtype(a2.dtype, a4.dtype) == np.object_
    assert ak.resolve_dtype(a3.dtype, a4.dtype) == np.object_
    assert ak.resolve_dtype(a3.dtype, a6.dtype) == np.object_
    assert ak.resolve_dtype(a1.dtype, a4.dtype) == np.float64
    assert ak.resolve_dtype(a1.dtype, a6.dtype) == np.float64
    assert ak.resolve_dtype(a4.dtype, a6.dtype) == np.float64


def test_resolve_dtype_b() -> None:
    a1 = np.array('a').dtype
    a3 = np.array('aaa').dtype
    assert ak.resolve_dtype(a1, a3) == np.dtype(('U', 3))


def test_resolve_dtype_c() -> None:
    a1 = np.array(['2019-01', '2019-02'], dtype=np.datetime64)
    a2 = np.array(['2019-01-01', '2019-02-01'], dtype=np.datetime64)
    a3 = np.array([0, 1], dtype='datetime64[ns]')
    a4 = np.array([0, 1])
    assert str(ak.resolve_dtype(a1.dtype, a2.dtype)) == 'datetime64[D]'
    assert ak.resolve_dtype(a1.dtype, a3.dtype) == np.dtype('<M8[ns]')
    assert ak.resolve_dtype(a1.dtype, a4.dtype) == np.dtype('O')


def test_resolve_dtype_iter_a() -> None:
    a1 = np.array([1, 2, 3])
    a2 = np.array([False, True, False])
    a3 = np.array(['b', 'c', 'd'])
    a4 = np.array([2.3, 3.2])
    a5 = np.array(['test', 'test again'], dtype='S')
    a6 = np.array([2.3, 5.4], dtype='float32')
    assert ak.resolve_dtype_iter((a1.dtype, a1.dtype)) == a1.dtype
    assert ak.resolve_dtype_iter((a2.dtype, a2.dtype)) == a2.dtype
    # Boolean with mixed types:
    assert ak.resolve_dtype_iter((a2.dtype, a2.dtype, a3.dtype)) == np.object_
    assert ak.resolve_dtype_iter((a2.dtype, a2.dtype, a5.dtype)) == np.object_
    assert ak.resolve_dtype_iter((a2.dtype, a2.dtype, a6.dtype)) == np.object_
    # Numeric types go to float64:
    assert ak.resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype)) == np.float64
    # Add in bool or str, goes to object:
    assert ak.resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a2.dtype)) == np.object_
    assert ak.resolve_dtype_iter((a1.dtype, a4.dtype, a6.dtype, a5.dtype)) == np.object_
    # Mixed strings go to the largest:
    assert ak.resolve_dtype_iter((a3.dtype, a5.dtype)) == np.dtype('<U10')
