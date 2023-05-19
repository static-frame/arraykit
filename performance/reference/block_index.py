
from arraykit import shape_filter
from arraykit import resolve_dtype

import typing as tp
import numpy as np

#-------------------------------------------------------------------------------
def from_blocks(
        raw_blocks: tp.Iterable[np.ndarray],
        ):
    '''Simulation of legacy routine within TypeBlocks.
    '''
    index: tp.List[tp.Tuple[int, int]] = [] # columns position to blocks key
    block_count = 0
    row_count = None
    column_count = 0
    dtype = None

    for block in raw_blocks:
        if not block.__class__ is np.ndarray:
            raise ErrorInitTypeBlocks(f'found non array block: {block}')
        if block.ndim > 2:
            raise ErrorInitTypeBlocks(f'cannot include array with {block.ndim} dimensions')

        r, c = shape_filter(block)

        if row_count is not None and r != row_count: #type: ignore [unreachable]
            raise ErrorInitTypeBlocks(f'mismatched row count: {r}: {row_count}')
        else:
            row_count = r
        if c == 0:
            continue

        if dtype is None:
            dtype = block.dtype
        else:
            dtype = resolve_dtype(dtype, block.dtype)

        for i in range(c):
            index.append((block_count, i))
        column_count += c
        block_count += 1
    return (row_count, column_count), index

#-------------------------------------------------------------------------------


def cols_to_slice(indices: tp.Sequence[int]) -> slice:
    '''Translate an iterable of contiguous integers into a slice.
Integers are assumed to be ordered (ascending or descending) and contiguous.
    '''
    start_idx = indices[0]
    # single column as a single slice
    if len(indices) == 1:
        return slice(start_idx, start_idx + 1)

    stop_idx = indices[-1]
    if stop_idx > start_idx: # ascending indices
        return slice(start_idx, stop_idx + 1)

    if stop_idx == 0:
        return slice(start_idx, None, -1)
    # stop is less than start, need to reduce by 1 to cover range
    return slice(start_idx, stop_idx - 1, -1)

def indices_to_contiguous_pairs(indices: tp.Iterable[tp.Tuple[int, int]]
    ) -> tp.Iterator[tp.Tuple[int, slice]]:
    '''Indices are pairs of (block_idx, value); convert these to pairs of (block_idx, slice) when we identify contiguous indices
within a block (these are block slices)
    '''
    # store pairs of block idx, ascending col list
    last: tp.Optional[tp.Tuple[int, int]] = None

    for block_idx, col in indices:
        if not last:
            last = (block_idx, col)
            bundle = [col]
            continue
        if last[0] == block_idx and abs(col - last[1]) == 1:
            # if contiguous, update last, add to bundle
            last = (block_idx, col)
            # do not need to store all col, only the last,
            # however probably easier to just accumulate all
            bundle.append(col)
            continue
        # either new block, or not contiguous on same block
        yield (last[0], cols_to_slice(bundle))
        # start a new bundle
        bundle = [col]
        last = (block_idx, col)

    # last can be None
    if last and bundle:
        yield (last[0], cols_to_slice(bundle))


class IterContiguous:
    def __init__(self, indices):
        self.indices = iter(indices)
        self.last_block = -1
        self.last_column = -1
        self.next_block = -1
        self.next_column = -1

    @staticmethod
    def build_slice(start, end_inclusive):
        # this works, but we reatain slices to force 2D selections; we might explore changing this
        # if start == end_inclusive:
        #     return start

        if start <= end_inclusive:
            return slice(start, end_inclusive + 1, None) # can be 1
        # reverse slice
        if end_inclusive == 0:
            return slice(start, None, -1)
        return slice(start, end_inclusive - 1, -1)

    def getter(self) -> tp.Tuple[int, slice]:
        slice_start = -1
        while True:
            if self.next_block == -2:
                return None # terminate the loop
            if self.next_block != -1:
                # we found a discontinuity on the last iteration, so this is a new start
                self.last_block = self.next_block
                self.last_column = self.next_column
                slice_start = self.last_column
                self.next_block = -1 # clear next statte

            try:
                block, column = next(self.indices)
            except StopIteration:
                self.next_block = -2
                return self.last_block, self.build_slice(slice_start, self.last_column)

            if self.last_block == -1: # only on init
                self.last_block = block
                self.last_column = column
                slice_start = column
                continue

            if self.last_block == block: # in the same block
                if abs(column - self.last_column) == 1:
                    self.last_column = column
                    continue
                else: # not contiguous, need to emit a slice, store next
                    self.next_block = block
                    self.next_column = column

                    return self.last_block, self.build_slice(slice_start, self.last_column)

            # blocks are not equal, must emit a slice
            self.next_block = block
            self.next_column = column

            return self.last_block, self.build_slice(slice_start, self.last_column)


    def iter(self) -> tp.Iterator[tp.Tuple[int, slice]]:
        while True:
            post = self.getter()
            if post is not None:
                yield post
            else:
                break

#-------------------------------------------------------------------------------



if __name__ == '__main__':
    samples = (
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 3), (2, 0), (3, 0), (3, 1), (3, 2)],
        [(0, 0), (2, 1), (3, 5), (10, 1)],
        [(0, 0), (2, 1), (2, 2), (2, 5), (2, 6), (10, 1)],
        [(10, 1)],
        [(10, 1)],
        [(0, 0), (2, 3), (2, 2), (2, 1), (2, 6), (10, 1)],
        [(2, 3), (0, 0), (2, 2), (2, 1), (2, 6), (2, 7)],
    )
    for sample in samples:
        p1 = list(indices_to_contiguous_pairs(sample))
        print(sample)
        print(p1)


        iterc = IterContiguous(sample)
        p2 = list(iterc.iter())
        print(p2)