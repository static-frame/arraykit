#!/usr/bin/env python
from functools import partial

import numpy as np  # type: ignore
from arraykit import isin_array

funcTT = partial(isin_array, array_is_unique=True, other_is_unique=True)
funcTF = partial(isin_array, array_is_unique=True, other_is_unique=False)
funcFT = partial(isin_array, array_is_unique=False, other_is_unique=True)
funcFF = partial(isin_array, array_is_unique=False, other_is_unique=False)


arr1 = np.array([1, 5, 2, 3, 4], dtype=int)
arr2 = np.array([1, 4, 7, 9], dtype=int)
post = funcTT(array=arr1, other=arr2)

# e_1d = np.array([1, 0, 0, 0, 1, 0, 1], dtype=bool)
arr1 = np.array([1, 5, 2, 3, 4, 5, 1], dtype=int)
arr2 = np.array([1, 4, 7, 9], dtype=int)
print(arr1)
print(arr2)
post = funcFF(array=arr1, other=arr2)
print(post)
