

.. image:: https://img.shields.io/pypi/pyversions/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/pypi/v/arraykit.svg
  :target: https://pypi.org/project/arraykit

.. image:: https://img.shields.io/conda/vn/conda-forge/arraykit.svg
  :target: https://anaconda.org/conda-forge/arraykit

.. image:: https://img.shields.io/github/workflow/status/InvestmentSystems/arraykit/CI?label=test&logo=Github
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
- NumPy >= 1.16.5


What is New in ArrayKit
-------------------------

0.1.3 dev
............

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