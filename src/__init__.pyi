import typing as tp

import numpy as np  # type: ignore

_T = tp.TypeVar('_T')

__version__: str

class ErrorInitTypeBlocks:
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None: ...
    def with_traceback(self, tb: Exception) -> Exception: ...
    def __setstate__(self) -> None: ...

class ArrayGO:

    values: np.ndarray
    def __init__(
        self, iterable: tp.Iterable[object], *, own_iterable: bool = ...
    ) -> None: ...
    def __iter__(self) -> tp.Iterator[tp.Any]: ...
    def __getitem__(self, __key: object) -> tp.Any: ...
    def __len__(self) -> int: ...
    def __getnewargs__(self) -> tp.Tuple[np.ndarray]: ...
    def append(self, __value: object) -> None: ...
    def copy(self: _T) -> _T: ...
    def extend(self, __values: tp.Iterable[object]) -> None: ...

class BlockIndex:
    shape: tp.Tuple[int, int]
    dtype: np.dtype
    rows: int
    columns: int

    def __init__() -> None: ...
    def register(self, __value: np.ndarray) -> bool: ...
    def to_list(self,) -> tp.List[int]: ...
    def to_bytes(self,) -> bytes: ...
    def copy(self,) -> 'BlockIndex': ...
    def __len__(self,) -> int: ...
    def __iter__(self,) -> tp.Iterator[tp.Tuple[int, int]]: ...
    def __getitem__(self, __key: int) -> tp.Tuple[int, int]: ...
    def __getstate__(self,) -> tp.Tuple[int, int, int, int, bytes]: ...
    def __setstate__(self, state: tp.Tuple[int, int, int, int, bytes]) -> None: ...
    def get_block(self, __key: int) -> int: ...
    def get_column(self, __key: int) -> int: ...
    def iter_select(self,
            __key: tp.Union[slice, np.ndarray, tp.List[int]],
            ) -> tp.Iterator[tp.Tuple[int, int]]: ...
    def iter_contiguous(self,
            __key: tp.Union[slice, np.ndarray, tp.List[int]],
            ascending: bool = False,
            ) -> tp.Iterator[tp.Tuple[int, int]]: ...


def iterable_str_to_array_1d(
        iterable: tp.Iterable[str],
        *,
        dtype: tp.Optional[tp.Any] = None,
        thousandschar: str = ',',
        decimalchar: str = '.',
        ) -> np.ndarray: ...

def delimited_to_arrays(
        file_like: tp.Iterable[str],
        *,
        axis: int = 0,
        dtypes: tp.Optional[tp.Callable[[int], tp.Any]] = None,
        line_select: tp.Optional[tp.Callable[[int], bool]] = None,
        delimiter: str = ',',
        doublequote: bool = True,
        escapechar: str = '',
        quotechar: str = '"',
        quoting: int = 0,
        skipinitialspace: bool = False,
        strict: bool = False,
        thousandschar: str = ',',
        decimalchar: str = '.',
        ) -> tp.List[np.array]: ...

def split_after_count(
        string: str,
        *,
        delimiter: str = ',',
        count: int = 0,
        doublequote: bool = True,
        escapechar: str = '',
        quotechar: str = '"',
        quoting: int = 0,
        strict: bool = False,
        ) -> tp.Tuple[str, str]: ...

def count_iteration(__iterable: tp.Iterable) -> int: ...

def immutable_filter(__array: np.ndarray) -> np.ndarray: ...
def mloc(__array: np.ndarray) -> int: ...
def name_filter(__name: tp.Hashable) -> tp.Hashable: ...
def shape_filter(__array: np.ndarray) -> np.ndarray: ...
def column_2d_filter(__array: np.ndarray) -> np.ndarray: ...
def column_1d_filter(__array: np.ndarray) -> np.ndarray: ...
def row_1d_filter(__array: np.ndarray) -> np.ndarray: ...
def array_deepcopy(__array: np.ndarray, memo: tp.Dict[int, tp.Any]) -> np.ndarray: ...
def resolve_dtype(__d1: np.dtype, __d2: np.dtype) -> np.dtype: ...
def resolve_dtype_iter(__dtypes: tp.Iterable[np.dtype]) -> np.dtype: ...
def isna_element(__value: tp.Any, include_none: bool = True) -> bool: ...
def dtype_from_element(__value: tp.Optional[tp.Hashable]) -> np.dtype: ...
def get_new_indexers_and_screen(indexers: np.ndarray, positions: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]: ...

def first_true_1d(__array: np.ndarray, *, forward: bool) -> int: ...
def first_true_2d(__array: np.ndarray, *, forward: bool, axis: int) -> np.ndarray: ...
def slice_to_ascending_slice(__slice: slice, __size: int) -> slice: ...
