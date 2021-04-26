import typing as tp
from copy import deepcopy

import numpy as np

DTYPE_DATETIME_KIND = 'M'
DTYPE_TIMEDELTA_KIND = 'm'
DTYPE_COMPLEX_KIND = 'c'
DTYPE_FLOAT_KIND = 'f'
DTYPE_OBJECT_KIND = 'O'
DTYPE_BOOL_KIND = 'b'

DTYPE_STR_KINDS = ('U', 'S') # S is np.bytes_
DTYPE_INT_KINDS = ('i', 'u') # signed and unsigned
DTYPE_INEXACT_KINDS = (DTYPE_FLOAT_KIND, DTYPE_COMPLEX_KIND) # kinds that support NaN values
DTYPE_NAT_KINDS = (DTYPE_DATETIME_KIND, DTYPE_TIMEDELTA_KIND)

DTYPE_OBJECT = np.dtype(object)
DTYPE_BOOL = np.dtype(bool)
DTYPE_STR = np.dtype(str)
DTYPE_INT_DEFAULT = np.dtype(np.int64)
DTYPE_FLOAT_DEFAULT = np.dtype(np.float64)
DTYPE_COMPLEX_DEFAULT = np.dtype(np.complex128)

DTYPES_BOOL = (DTYPE_BOOL,)
DTYPES_INEXACT = (DTYPE_FLOAT_DEFAULT, DTYPE_COMPLEX_DEFAULT)


def mloc(array: np.ndarray) -> int:
    '''Return the memory location of an array.
    '''
    return tp.cast(int, array.__array_interface__['data'][0])

def immutable_filter(src_array: np.ndarray) -> np.ndarray:
    '''Pass an immutable array; otherwise, return an immutable copy of the provided array.
    '''
    if src_array.flags.writeable:
        dst_array = src_array.copy()
        dst_array.flags.writeable = False
        return dst_array
    return src_array # keep it as is

def name_filter(name):
    '''
    For name attributes on containers, only permit recursively hashable objects.
    '''
    try:
        hash(name)
    except TypeError:
        raise TypeError('unhashable name attribute', name)
    return name



def shape_filter(array: np.ndarray) -> tp.Tuple[int, int]:
    '''Represent a 1D array as a 2D array with length as rows of a single-column array.

    Return:
        row, column count for a block of ndim 1 or ndim 2.
    '''
    if array.ndim == 1:
        return array.shape[0], 1
    return array.shape #type: ignore

def column_2d_filter(array: np.ndarray) -> np.ndarray:
    '''Reshape a flat ndim 1 array into a 2D array with one columns and rows of length. This is used (a) for getting string representations and (b) for using np.concatenate and np binary operators on 1D arrays.
    '''
    # it is not clear when reshape is a copy or a view
    if array.ndim == 1:
        return np.reshape(array, (array.shape[0], 1))
    return array

def column_1d_filter(array: np.ndarray) -> np.ndarray:
    '''
    Ensure that a column that might be 2D or 1D is returned as a 1D array.
    '''
    if array.ndim == 2:
        # could assert that array.shape[1] == 1, but this will raise if does not fit
        return np.reshape(array, array.shape[0])
    return array

def row_1d_filter(array: np.ndarray) -> np.ndarray:
    '''
    Ensure that a row that might be 2D or 1D is returned as a 1D array.
    '''
    if array.ndim == 2:
        # could assert that array.shape[0] == 1, but this will raise if does not fit
        return np.reshape(array, array.shape[1])
    return array


#-------------------------------------------------------------------------------

def resolve_dtype(dt1: np.dtype, dt2: np.dtype) -> np.dtype:
    '''
    Given two dtypes, return a compatible dtype that can hold both contents without truncation.
    '''
    # NOTE: this is not taking into account endianness; it is not clear if this is important
    # NOTE: np.dtype(object) == np.object_, so we can return np.object_

    # if the same, return that dtype
    if dt1 == dt2:
        return dt1

    # if either is object, we go to object
    if dt1.kind == 'O' or dt2.kind == 'O':
        return DTYPE_OBJECT

    dt1_is_str = dt1.kind in DTYPE_STR_KINDS
    dt2_is_str = dt2.kind in DTYPE_STR_KINDS
    if dt1_is_str and dt2_is_str:
        # if both are string or string-like, we can use result type to get the longest string
        return np.result_type(dt1, dt2)

    dt1_is_dt = dt1.kind == DTYPE_DATETIME_KIND
    dt2_is_dt = dt2.kind == DTYPE_DATETIME_KIND
    if dt1_is_dt and dt2_is_dt:
        # if both are datetime, result type will work
        return np.result_type(dt1, dt2)

    dt1_is_tdelta = dt1.kind == DTYPE_TIMEDELTA_KIND
    dt2_is_tdelta = dt2.kind == DTYPE_TIMEDELTA_KIND
    if dt1_is_tdelta and dt2_is_tdelta:
        # this may or may not work
        # TypeError: Cannot get a common metadata divisor for NumPy datetime metadata [D] and [Y] because they have incompatible nonlinear base time units
        try:
            return np.result_type(dt1, dt2)
        except TypeError:
            return DTYPE_OBJECT

    dt1_is_bool = dt1.type is np.bool_
    dt2_is_bool = dt2.type is np.bool_

    # if any one is a string or a bool, we have to go to object; we handle both cases being the same above; result_type gives a string in mixed cases
    if (dt1_is_str or dt2_is_str
            or dt1_is_bool or dt2_is_bool
            or dt1_is_dt or dt2_is_dt
            or dt1_is_tdelta or dt2_is_tdelta
            ):
        return DTYPE_OBJECT

    # if not a string or an object, can use result type
    return np.result_type(dt1, dt2)

def resolve_dtype_iter(dtypes: tp.Iterable[np.dtype]) -> np.dtype:
    '''Given an iterable of one or more dtypes, do pairwise comparisons to determine compatible overall type. Once we get to object we can stop checking and return object.

    Args:
        dtypes: iterable of one or more dtypes.
    '''
    dtypes = iter(dtypes)
    dt_resolve = next(dtypes)

    for dt in dtypes:
        dt_resolve = resolve_dtype(dt_resolve, dt)
        if dt_resolve == DTYPE_OBJECT:
            return dt_resolve
    return dt_resolve



def array_deepcopy(
        array: np.ndarray,
        memo: tp.Optional[tp.Dict[int, tp.Any]],
        ) -> np.ndarray:
    '''
    Create a deepcopy of an array, handling memo lookup, insertion, and object arrays.
    '''
    ident = id(array)
    if memo is not None and ident in memo:
        return memo[ident]

    if array.dtype == DTYPE_OBJECT:
        post = deepcopy(array, memo)
    else:
        post = array.copy()

    if post.ndim > 0:
        post.flags.writeable = array.flags.writeable

    if memo is not None:
        memo[ident] = post
    return post


def _isin_1d(
        array: np.ndarray,
        other: tp.FrozenSet[tp.Any]
        ) -> np.ndarray:
    '''
    Iterate over an 1D array to build a 1D Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: np.ndarray = np.empty(array.shape, dtype=DTYPE_BOOL)

    for i, element in enumerate(array):
        result[i] = element in other

    result.flags.writeable = False
    return result


def _isin_2d(
        array: np.ndarray,
        other: tp.FrozenSet[tp.Any]
        ) -> np.ndarray:
    '''
    Iterate over an 2D array to build a 2D, immutable, Boolean ndarray representing whether or not the original element is in the set

    Args:
        array: The source array
        other: The set of elements being looked for
    '''
    result: np.ndarray = np.empty(array.shape, dtype=DTYPE_BOOL)

    for (i, j), v in np.ndenumerate(array):
        result[i, j] = v in other

    result.flags.writeable = False
    return result


def isin_array(*,
        array: np.ndarray,
        array_is_unique: bool,
        other: np.ndarray,
        other_is_unique: bool,
        ) -> np.ndarray:
    '''Core isin processing after other has been converted to an array.
    '''
    if array.dtype == DTYPE_OBJECT or other.dtype == DTYPE_OBJECT:
        # both funcs return immutable arrays
        func = _isin_1d if array.ndim == 1 else _isin_2d
        try:
            return func(array, frozenset(other)) # Isolate the frozenset creation to it's own try-except
        except TypeError: # only occur when something is unhashable.
            pass

    assume_unique = array_is_unique and other_is_unique
    func = np.in1d if array.ndim == 1 else np.isin

    result = func(array, other, assume_unique=assume_unique) #type: ignore
    result.flags.writeable = False

    return result


def unique(ar, return_inverse=False):

    ar = np.asanyarray(ar).flatten()

    if return_inverse:
        perm = ar.argsort(kind='quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar

    mask = np.empty(aux.shape, dtype=np.bool_)
    mask[:1] = True
    if aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')

        mask[1:aux_firstnan] = (aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = aux[mask]
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        return ret, inv_idx

    return ret
