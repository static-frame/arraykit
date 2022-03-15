import numpy as np
import sys
from arraykit import array_to_duplicated_hashable


def main(setup):
    if setup == 'small_1d':
        ITERATIONS = 5_000
        axes = (0,)
        arrays = [
            np.array([0,0,1,0,None,None,0,1,None], dtype=object),
            np.array([0,0,1,0,'q','q',0,1,'q'], dtype=object),
            np.array(['q','q','q', 'a', 'w', 'w'], dtype=object),
            np.array([0,1,2,2,1,4,5,3,4,5,5,6], dtype=object),
        ]

    elif setup == 'large_1d':
        ITERATIONS = 10
        axes = (0,)

        rs = np.random.RandomState(0)
        arrays = [
            np.arange(100_000).astype(object), # All unique
            np.full(100_000, fill_value='abc').astype(object), # All duplicated
            rs.randint(0, 100, 100_000).astype(object), # Many repeated elements from small subset
            rs.randint(0, 10_000, 100_000).astype(object), # Many repeated elements from medium subset
            rs.randint(0, 75_000, 100_000).astype(object), # Some repeated elements from a large subset
            np.hstack([np.arange(15), np.arange(90_000), np.arange(15), np.arange(9970)]).astype(object), # Custom
        ]

    elif setup == 'small_2d':
        ITERATIONS = 5_000
        axes = (0, 1)
        arrays = [
            np.array([[None, None, None, 32, 17, 17], [2,2,2,False,'q','q'], [2,2,2,False,'q','q'], ], dtype=object),
            np.array([[None, None, None, 32, 17, 17], [2,2,2,False,'q','q'], [2,2,2,False,'q','q'], ], dtype=object),
            np.array([[50, 50, 32, 17, 17], [2,2,1,3,3]], dtype=object),
        ]

    elif setup == 'large_2d':
        ITERATIONS = 10
        axes = (0, 1)
        arrays = [
            np.arange(100_000).reshape(10_000, 10).astype(object),
            np.hstack([np.arange(15), np.arange(90_000), np.arange(15), np.arange(9970)]).reshape(10_000, 10).astype(object),
        ]

    else:
        assert False, "Impossible state!"

    for _ in range(ITERATIONS):
        for arr in arrays:
            for axis in axes:
                array_to_duplicated_hashable(arr, axis, True, False)
                array_to_duplicated_hashable(arr, axis, False, True)
                array_to_duplicated_hashable(arr, axis, False, False)


if __name__ == '__main__':
    try:
        setup = sys.argv[1]
        assert setup in ('small_1d', 'large_1d', 'small_2d', 'large_2d')
    except IndexError:
        print('Expected a setup arg!')
        sys.exit(1)
    except AssertionError:
        print(f'Invalid setup arg: {setup}')
        sys.exit(1)

    main(setup)
