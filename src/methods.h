# ifndef ARRAYKIT_SRC_METHODS_H
# define ARRAYKIT_SRC_METHODS_H
# include "Python.h"

// A fast counter of unsized iterators
PyObject *
count_iteration(PyObject *Py_UNUSED(m), PyObject *iterable);

// Reshape if necessary a row that might be 2D or 1D is returned as a 1D array.
PyObject *
row_1d_filter(PyObject *Py_UNUSED(m), PyObject *a);

// Convert any slice to an ascending slice that covers the same values.
PyObject *
slice_to_ascending_slice(PyObject *Py_UNUSED(m), PyObject *args);

// Reshape if necessary a flat ndim 1 array into a 2D array with one columns and rows of length.
// related example: https://github.com/RhysU/ar/blob/master/ar-python.cpp
PyObject *
column_2d_filter(PyObject *Py_UNUSED(m), PyObject *a);

// Reshape if necessary a column that might be 2D or 1D is returned as a 1D array.
PyObject *
column_1d_filter(PyObject *Py_UNUSED(m), PyObject *a);

// Represent a 1D array as a 2D array with length as rows of a single-column array.
// https://stackoverflow.com/questions/56182259/how-does-one-acces-numpy-multidimensionnal-array-in-c-extensions
PyObject *
shape_filter(PyObject *Py_UNUSED(m), PyObject *a);

PyObject *
name_filter(PyObject *Py_UNUSED(m), PyObject *n);

// Return the integer version of the pointer to underlying data-buffer of array.
PyObject *
mloc(PyObject *Py_UNUSED(m), PyObject *a);

PyObject *
immutable_filter(PyObject *Py_UNUSED(m), PyObject *a);

PyObject *
resolve_dtype(PyObject *Py_UNUSED(m), PyObject *args);

PyObject *
resolve_dtype_iter(PyObject *Py_UNUSED(m), PyObject *arg);

PyObject *
nonzero_1d(PyObject *Py_UNUSED(m), PyObject *a);

PyObject *
first_true_1d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

PyObject *
first_true_2d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

PyObject *
dtype_from_element(PyObject *Py_UNUSED(m), PyObject *arg);

PyObject *
isna_element(PyObject *m, PyObject *args, PyObject *kwargs);

PyObject *
get_new_indexers_and_screen(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

// Specialized array deepcopy that stores immutable arrays in an optional memo dict that can be provided with kwargs.
PyObject *
array_deepcopy(PyObject *m, PyObject *args, PyObject *kwargs);

PyObject *
immutable_filter(PyObject *Py_UNUSED(m), PyObject *a);

# endif /* ARRAYKIT_SRC_METHODS_H */
