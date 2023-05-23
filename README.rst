

.. image:: https://img.shields.io/pypi/pyversions/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/pypi/v/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/conda/vn/conda-forge/arraykit.svg
  :target: https://anaconda.org/conda-forge/arraykit

.. image:: https://img.shields.io/github/actions/workflow/status/static-frame/arraykit/ci.yml?branch=master&label=build&logo=Github
  :target: https://github.com/static-frame/arraykit/actions/workflows/ci.yml


arraykit
=============

The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Packages: https://pypi.org/project/arraykit



Dependencies
--------------

ArrayKit requires the following:

- Python >= 3.7
- NumPy >= 1.18.5



What is New in ArrayKit
-------------------------

0.4.6
............

Corrected handling of empty selections in ``BlockIndex.iter_contiguous()``.


0.4.5
............

Corrected handling of ``ascending`` in ``BlockIndex.iter_contiguous()`` with Boolean arrays.


0.4.4
............

Corrected deallocation routines in ``BlockIndex.iter_contiguous()``.


0.4.3
............

Added ``BlockIndex.iter_contiguous()`` with options for ``ascending`` and ``reduce`` sequences.


0.4.2
............

Added ``slice_to_ascending_slice()``.

Updated ``BlockIndex.shape`` to internally cache the shape tuple.

Corrected ``BlockIndex.iter_select()`` handling of negative integers in sequences.


0.4.1
............

Updated ``BlockIndex.register()`` to handle 0-column 2D arrays and return False.

Added ``BlockIndex.rows``, ``BlockIndex.columns`` properties.

Updated unset ``BlockIndex.dtype`` to return a float dtype.


0.4.0
............

Added ``BlockIndex``, a tool to be used by ``TypeBlocks`` for mapping realized column positions to arrays.

Corrected potential issue in ``AK_CPL_resize_buffer`` that could segfault for very large offsets.


0.3.4
............

Simplified implementation of ``NaTType`` identification in ``isna_element()``.


0.3.3
............

Corrected potential memory leak in ``isna_element()``.


0.3.2
............

Optimization to ``delimited_to_arrays()`` character reading per line.


0.3.1
............

Improvements to ``delimited_to_arrays()``, including proper loading of ``float16`` types.

Extended ``deepcopy_array()`` to permit ``memo`` as None.

Corrected all compiler warnings on Windows.


0.3.0
............

Added ``first_true_1d()``, ``first_true_2d()``. Added tools for performance graphing.


0.2.9
............

Corrected segmentation fault resulting from attempting to parse invalid ``datetime64`` strings in ``AK_CPL_to_array_via_cast``.


0.2.8
............

Added ``include_none`` argument to ``isna_element()``; implemented identification of Pandas pd.Timestamp NaT.


0.2.7
............

Updated most-recent NumPy references to 1.23.5.


0.2.6
............

Maintenance release.


0.2.5
............

Optimization to numerical array creation in ``delimited_to_arrays()``.


0.2.4
............

Set NumPy minimum version at 1.18.5.


0.2.3
............

Extended arguments to and functionality in ``split_after_count()`` to support the complete CSV dialect interface.

Now building wheels for 3.11.


0.2.2
............

Refinements to ensure typed-parsed ints are always int64 in ``delimited_to_arrays()``.


0.2.1
............

Implemented ``count_iteration``, ``split_after_count``.


0.2.0
............

Implemented ``delimited_to_arrays``, ``iterable_str_to_array_1d``.


0.1.13
............

Now building Python 3.10 wheels.


0.1.12
............

Added ``get_new_indexers_and_screen``.


0.1.10
............

Updated minimum NumPy to 1.18.5


0.1.9
............

Improvements to performance of ``array_deepcopy``.

Added ``dtype_from_element``.


0.1.8
............

Revised cross compile releases.


0.1.7
............

Added ``dtype_from_element()``.


0.1.6
............

Explicit imports in ``__init__.py`` for better static analysis.


0.1.5
............

Added ``isna_element()``.


0.1.3
............

Redesigned package structure for inclusion of ``py.typed`` and ``__init__.pyi``.

``array_deepcopy`` now accepts kwargs and makes the ``memo`` dict optional.


0.1.2
..........

Maintenance release of the following interfaces:

``immutable_filter``
``mloc``
``shape_filter``
``column_2d_filter``
``column_1d_filter``
``row_1d_filter``
``array_deepcopy``
``resolve_dtype``
``resolve_dtype_iter``