import pickle
import typing as tp
from functools import partial
import sys
import warnings

import numpy as np
import hypothesis
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import scalar_dtypes
from hypothesis import strategies as st
from hypothesis import given

import pytest

from arraykit import AutoMap
from arraykit import FrozenAutoMap
from arraykit import NonUniqueError

Keys = tp.Set[tp.Union[int, str, float, bool, bytes, tp.Tuple[int, ...]]]

NATIVE_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"
VALID_BYTE_ORDERS = ("=", NATIVE_BYTE_ORDER)


def get_array() -> st.SearchStrategy:
    """
    Labels are suitable for creating non-date Indices (though they might include dates); these labels might force an object array result.
    """

    def proc(a: np.ndarray, contiguous: bool):
        if a.dtype.kind in ("f", "c"):
            a = a[~np.isnan(a)]
        elif a.dtype.kind in ("m", "M"):
            a = a[~np.isnat(a)]

        if a.dtype.byteorder not in VALID_BYTE_ORDERS:
            a = a.astype(a.dtype.newbyteorder(NATIVE_BYTE_ORDER))

        if not contiguous:
            a = np.lib.stride_tricks.as_strided(
                a,
                shape=(len(a) // 2,),
                strides=(a.dtype.itemsize * 2,),
            )

        a.flags.writeable = False
        return a

    def strategy(contiguous: bool):
        return arrays(
            shape=1, unique=True, fill=st.nothing(), dtype=scalar_dtypes()
        ).map(partial(proc, contiguous=contiguous))

    return st.one_of(
        strategy(contiguous=True),
        strategy(contiguous=False),
    )


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am___len__(keys: Keys) -> None:
    assert len(AutoMap(keys)) == len(keys)


@given(keys=get_array())
def test_fam_array___len__(keys: Keys) -> None:
    assert len(FrozenAutoMap(keys)) == len(keys)


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer, others=hypothesis.infer)
def test_am___contains__(keys: Keys, others: Keys) -> None:
    a = AutoMap(keys)
    for key in keys:
        assert key in a
    others -= keys
    for key in others:
        assert key not in a


@given(keys=get_array())
def test_fam_array___contains__(keys: Keys) -> None:
    fam = FrozenAutoMap(keys)
    for key in keys:
        assert key in fam


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer, others=hypothesis.infer)
def test_am___getitem__(keys: Keys, others: Keys) -> None:
    a = AutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index
    others -= keys
    for key in others:
        with pytest.raises(KeyError):
            a[key]


@given(keys=get_array())
def test_fam_array___getitem__(keys: Keys) -> None:
    a = FrozenAutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am___hash__(keys: Keys) -> None:
    assert hash(FrozenAutoMap(keys)) == hash(FrozenAutoMap(keys))


@given(keys=get_array())
def test_fam_array___hash__(keys: Keys) -> None:
    assert hash(FrozenAutoMap(keys)) == hash(FrozenAutoMap(keys))


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am___iter__(keys: Keys) -> None:
    assert [*AutoMap(keys)] == [*keys]


@given(keys=hypothesis.infer)
def test_fam_array___iter__(keys: Keys) -> None:
    assert [*FrozenAutoMap(keys)] == [*keys]


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am___reversed__(keys: Keys) -> None:
    assert [*reversed(AutoMap(keys))] == [*reversed([*keys])]


@given(keys=get_array())
def test_fam_array___reversed__(keys: Keys) -> None:
    assert [*reversed(FrozenAutoMap(keys))] == [*reversed([*keys])]


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am_add(keys: Keys) -> None:
    a = AutoMap()
    for l, key in enumerate(keys):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_am_pickle(keys: Keys) -> None:
    try:
        hypothesis.assume(pickle.loads(pickle.dumps(keys)) == keys)
    except (TypeError, pickle.PicklingError):
        hypothesis.assume(False)
    a = AutoMap(keys)
    assert pickle.loads(pickle.dumps(a)) == a


@given(keys=get_array())
def test_fam_array_pickle(keys: Keys) -> None:
    a = FrozenAutoMap(keys)
    assert list(pickle.loads(pickle.dumps(a))) == list(a)


# -------------------------------------------------------------------------------
@given(keys=hypothesis.infer)
def test_issue_3(keys: Keys) -> None:
    hypothesis.assume(keys)
    key = keys.pop()
    a = AutoMap(keys)
    a |= (key,)
    with pytest.raises(ValueError):
        a |= (key,)


@given(keys=hypothesis.infer)
def test_am_non_unique_exception(keys: Keys):
    hypothesis.assume(keys)
    duplicate = next(iter(keys))

    with pytest.raises(ValueError):
        AutoMap([*keys, duplicate])

    with pytest.raises(NonUniqueError):
        AutoMap([*keys, duplicate])


@given(keys=get_array())
def test_fam_array_non_unique_exception(keys: Keys):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hypothesis.assume(keys.size)
        duplicate = next(iter(keys))

        with pytest.raises(ValueError):
            FrozenAutoMap([*keys, duplicate])

        with pytest.raises(NonUniqueError):
            FrozenAutoMap([*keys, duplicate])


# -------------------------------------------------------------------------------
@given(keys=get_array())
def test_fam_array_get_all(keys: Keys) -> None:
    fam = FrozenAutoMap(keys)
    keys_list = list(keys)

    post1 = fam.get_all(keys_list)
    assert list(post1) == list(fam.values())

    post2 = fam.get_all(keys)
    assert list(post2) == list(fam.values())


@given(keys=get_array())
def test_fam_array_get_any(keys: Keys) -> None:
    fam = FrozenAutoMap(keys)
    keys_list = list(keys)

    post1 = fam.get_any(keys_list)
    assert post1 == list(fam.values())

    post2 = fam.get_any(keys)
    assert post2 == list(fam.values())


# -------------------------------------------------------------------------------


@given(keys=get_array())
def test_am_array_get_all(keys: Keys) -> None:
    fam = AutoMap(keys)
    keys_list = list(keys)

    post1 = fam.get_all(keys_list)
    assert list(post1) == list(fam.values())

    post2 = fam.get_all(keys)
    assert list(post2) == list(fam.values())


@given(keys=get_array())
def test_am_array_get_any(keys: Keys) -> None:
    fam = AutoMap(keys)
    keys_list = list(keys)

    post1 = fam.get_any(keys_list)
    assert post1 == list(fam.values())

    post2 = fam.get_any(keys)
    assert post2 == list(fam.values())


# -------------------------------------------------------------------------------


@given(keys=get_array())
def test_am_get_all(keys: Keys) -> None:
    keys_list = list(keys)
    fam = AutoMap(keys_list)

    post1 = fam.get_all(keys_list)
    assert list(post1) == list(fam.values())

    post2 = fam.get_all(keys_list)
    assert list(post2) == list(fam.values())


@given(keys=get_array())
def test_am_get_any(keys: Keys) -> None:
    keys_list = list(keys)
    fam = AutoMap(keys_list)

    post1 = fam.get_any(keys_list)
    assert post1 == list(fam.values())

    post2 = fam.get_any(keys_list)
    assert post2 == list(fam.values())
