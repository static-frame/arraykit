from arraykit import array_to_duplicated_hashable
import numpy as np

class PO:
    def __init__(self, v) -> None:
        self.v = v
    def __repr__(self) -> str:
        return f'PO<{self.v}>'


def new(
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

    if array.ndim == 1:
        value_source = array
    else:
        if axis == 0:
            value_source = map(tuple, array)
        else:
            value_source = map(tuple, array.T)

    is_dupe = np.full(size, False)

    if exclude_first and not exclude_last:
        # Optimize for route requiring least amount of data structure

        found = set()

        for idx, v in enumerate(value_source):
            if v not in found:
                found.add(v)
            else:
                is_dupe[idx] = True

        return is_dupe

    first_unique_locations = {}
    last_duplicate_locations = {}

    for idx, v in enumerate(value_source):
        if v not in first_unique_locations:
            first_unique_locations[v] = idx
        else:
            is_dupe[idx] = True

            if v not in last_duplicate_locations and not exclude_first:
                is_dupe[first_unique_locations[v]] = True

            # always update last
            last_duplicate_locations[v] = idx

    if exclude_last: # overwrite with False
        is_dupe[list(last_duplicate_locations.values())] = False

    return is_dupe

def test(*args, **kwargs):
    assert (new(*args, **kwargs) == array_to_duplicated_hashable(*args, **kwargs)).all(), (args, kwargs)


arr = np.array([1, PO(1), 2, 3, 1, PO(1), 2, 3, 2, -1, -233, 'aslkj', 'df', 'df', True, True, None, 1])
#array_to_duplicated_hashable(np.arange(5))
#array_to_duplicated_hashable(np.arange(5), 213)
#array_to_duplicated_hashable(np.arange(5), 1)
#array_to_duplicated_hashable(np.arange(5), 1, True)
#array_to_duplicated_hashable(np.arange(5), 1, 123)
#array_to_duplicated_hashable(np.arange(5), 1, True)
test(arr, 0, True, False)
test(arr, 0, False, False)
test(arr, 0, False, True)
test(arr, 0, True, True)
