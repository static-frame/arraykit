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


def isna_element(value: tp.Any) -> bool:
    '''Return Boolean if value is an NA. This does not yet handle pd.NA
    '''
    try:
        return np.isnan(value) #type: ignore
    except TypeError:
        pass

    if isinstance(value, (np.datetime64, np.timedelta64)):
        return np.isnat(value) #type: ignore

    return value is None


def dtype_from_element(value: tp.Optional[tp.Hashable]) -> np.dtype:
    '''Given an arbitrary hashable to be treated as an element, return the appropriate dtype. This was created to avoid using np.array(value).dtype, which for a Tuple does not return object.
    '''
    if value is np.nan:
        # NOTE: this will not catch all NaN instances, but will catch any default NaNs in function signatures that reference the same NaN object found on the NP root namespace
        return DTYPE_FLOAT_DEFAULT
    if value is None:
        return DTYPE_OBJECT
    if isinstance(value, tuple):
        return DTYPE_OBJECT
    if hasattr(value, 'dtype'):
        return value.dtype #type: ignore
    # NOTE: calling array and getting dtype on np.nan is faster than combining isinstance, isnan calls
    return np.array(value).dtype


NDITER_FLAGS = ('external_loop', 'buffered', 'zerosize_ok')
BUFFERSIZE_NUMERATOR = 16 * 1024 ** 2
# for 8 bytes this would give 2,097,152 bytes

def array_bytes_to_file(
        array: np.ndarray,
        file: tp.BinaryIO,
        ):
    buffersize = max(BUFFERSIZE_NUMERATOR // array.itemsize, 1)
    flags = array.flags
    if flags.f_contiguous and not flags.c_contiguous:
        for chunk in np.nditer(
                array,
                flags=NDITER_FLAGS,
                buffersize=buffersize,
                order='F',
                ):
            file.write(chunk.tobytes('C'))
    else:
        for chunk in np.nditer(
                array,
                flags=NDITER_FLAGS,
                buffersize=buffersize,
                order='C',
                ):
            file.write(chunk.tobytes('C'))

def get_new_indexers_and_screen_ref(
        indexers: np.ndarray,
        positions: np.ndarray,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:

    positions = indexers.argsort()

    # get the sorted indexers
    indexers = indexers[positions]

    mask = np.empty(indexers.shape, dtype=DTYPE_BOOL)
    mask[0] = True
    mask[1:] = indexers[1:] != indexers[:-1]

    new_indexers = np.empty(mask.shape, dtype=DTYPE_INT_DEFAULT)
    new_indexers[positions] = np.cumsum(mask) - 1
    new_indexers.flags.writeable = False

    return new_indexers, indexers[mask]


def get_new_indexers_and_screen_ak(
        indexers: np.ndarray,
        positions: np.ndarray,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
    from arraykit import get_new_indexers_and_screen as ak_routine

    if len(positions) > len(indexers):
        return np.unique(indexers, return_inverse=True)

    return ak_routine(indexers, positions)


def split_after_count(string: str, delimiter: str, count: int):
    *left, right = string.split(delimiter, maxsplit=count)
    return ','.join(left), right

def count_iteration(iterable: tp.Iterable):
    count = 0
    for i in iterable:
        count += 1
    return count

