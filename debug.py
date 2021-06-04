from arraykit import array_to_duplicated_hashable
import numpy as np

class PO:
    def __init__(self, v) -> None:
        self.v = v
    def __repr__(self) -> str:
        return f'PO<{self.v}>'


def handle_value_one_boundary(i, value, is_dup, set_obj, dict_obj):
    seen = set_obj
    assert dict_obj == None

    if value not in seen:
        seen.add(value)
    else:
        is_dup[i] = True


def handle_value_exclude_boundaries(i, value, is_dup, set_obj, dict_obj):
    duplicates = set_obj
    first_unique_locations = dict_obj

    if value not in first_unique_locations:
        first_unique_locations[value] = i
    else:
        is_dup[i] = True

        # Second time seeing a duplicate
        if value not in duplicates:
            is_dup[first_unique_locations[value]] = True

        # always update last
        duplicates.add(value)


def handle_value_include_boundaries(i, value, is_dup, set_obj, dict_obj):
    seen = set_obj
    last_duplicate_locations = dict_obj

    if value not in seen:
        seen.add(value)
    else:
        is_dup[i] = True

        # always update last
        last_duplicate_locations[value] = i


def iterate_1d(array, axis, reverse, is_dup, process_value_func, set_obj, dict_obj):
    if reverse:
        iterator = reversed(array)
    else:
        iterator = array

    size = len(array)

    for i, value in enumerate(iterator):
        if reverse:
            i = size - i - 1

        process_value_func(i, value, is_dup, set_obj, dict_obj)


def iterate_2d(array, axis, reverse, is_dup, process_value_func, set_obj, dict_obj):
    size = array.shape[axis]

    if axis == 0:
        iterator = array
    else:
        iterator = array.T

    if reverse:
        iterator = reversed(iterator)

    for i, value in enumerate(map(tuple, iterator)):
        if reverse:
            i = size - i - 1

        process_value_func(i, value, is_dup, set_obj, dict_obj)


def python_impl(
        array: np.ndarray,
        axis: int = 0,
        exclude_first: bool = False,
        exclude_last: bool = False,
    ) -> np.ndarray:
    '''
    Algorithm for finding duplicates in unsortable arrays for hashables. This will always be an object array.

    Note:
        np.unique fails under the same conditions that sorting fails, so there is no need to try np.unique: must go to set drectly.
    '''
    size = array.shape[axis]

    reverse = not exclude_first and exclude_last

    if array.ndim == 1:
        iterate_func = iterate_1d
    else:
        iterate_func = iterate_2d

    is_dup = np.full(size, False)

    set_obj = set()
    if exclude_first ^ exclude_last:
        dict_obj = None
        process_value_func = handle_value_one_boundary

    elif not exclude_first and not exclude_last:
        dict_obj = dict()
        process_value_func = handle_value_exclude_boundaries

    else:
        dict_obj = dict()
        process_value_func = handle_value_include_boundaries

    iterate_func(array, axis, reverse, is_dup, process_value_func, set_obj, dict_obj)

    if exclude_first and exclude_last:
        is_dup[list(dict_obj.values())] = False

    return is_dup


def dprint(*args, debug):
    '''Debug print'''
    if debug:
        print(*args)


def run_test(array, debug=True):
    def _test(*args):
        dprint(args[1:], debug=debug)

        python_result = python_impl(*args)
        dprint('python:', python_result, debug=debug)

        c_result = array_to_duplicated_hashable(*args);
        dprint('c     :', c_result, debug=debug)
        assert (python_result == c_result).all()

    _test(array, 0, True, False) # include_boundaries
    _test(array, 0, False, False) # one_boundary (normal)
    _test(array, 0, False, True) # one_boundary (reverse)
    _test(array, 0, True, True) # exclude_boundaries

    if len(array.shape) == 2:
        _test(array, 1, True, False)
        _test(array, 1, False, False)
        _test(array, 1, False, True)
        _test(array, 1, True, True)


def test_arr1d(debug=True):
    arr = np.array([1, 2, 2, 1, 3, 2, 6], dtype=object)

    # Test with normally constructed array
    run_test(arr, debug=debug)

    arr2d = np.array([[2, 1, 2],
                      [3, 2, 3],
                      [3, 2, 3],
                      [2, 1, 2],
                      [4, 3, 4],
                      [3, 2, 3],
                      [6, 6, 6]], dtype=object)

    # Test with array slices
    run_test(arr2d[:, 1], debug=debug)
    run_test(arr2d.T[1], debug=debug)


def test_arr2d(debug=True):
    arr2d = np.array([
        [1, 2, 2, 1, 3, 2, 6],
        [2, 3, 3, 2, 4, 3, 6],
        [2, 3, 3, 2, 4, 3, 6],
        [1, 2, 2, 1, 3, 2, 6],
        [3, 4, 4, 3, 5, 4, 6],
        [2, 3, 3, 2, 4, 3, 6],
    ], dtype=object)

    run_test(arr2d, debug=debug)
    run_test(arr2d.T, debug=debug)


def test_misc(debug=True):
    arr = np.array([1, PO(1), 2, 3, 1, PO(1), 2, 3, 2, -1, -233, 'aslkj', 'df', 'df', True, True, None, 1])
    run_test(arr, debug=debug)

    arr = np.arange(20).reshape(4, 5).astype(object)
    run_test(arr, debug=debug)
    run_test(arr.T, debug=debug)


# arr = np.array([
#     [1, 2, 2, 1, 3, 2, 6],
#     [2, 3, 3, 2, 4, 3, 6],
#     [2, 3, 3, 2, 4, 3, 6],
#     [1, 2, 2, 1, 3, 2, 6],
#     [3, 4, 4, 3, 5, 4, 6],
#     [2, 3, 3, 2, 4, 3, 6],
# ], dtype=object)
# array_to_duplicated_hashable(arr, 1)


test_arr1d(debug=False)
test_arr2d(debug=False)
test_misc(debug=False)
print('Done')
