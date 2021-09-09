

.. image:: https://img.shields.io/pypi/pyversions/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/pypi/v/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/conda/vn/conda-forge/arraykit.svg
  :target: https://anaconda.org/conda-forge/arraykit

.. image:: https://img.shields.io/github/workflow/status/InvestmentSystems/arraykit/CI?label=build&logo=Github
  :target: https://github.com/InvestmentSystems/arraykit/actions?query=workflow%3ACI


arraykit
=============

The ArrayKit library provides utilities for creating and transforming NumPy arrays, implementing performance-critical StaticFrame operations as Python C extensions.

Code: https://github.com/InvestmentSystems/arraykit

Packages: https://pypi.org/project/arraykit


Dependencies
--------------

ArrayKit requires the following:

- Python >= 3.6
- NumPy >= 1.17.4



What is New in ArrayKit
-------------------------

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