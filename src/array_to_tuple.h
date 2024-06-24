# ifndef ARRAYKIT_SRC_ARRAY_TO_TUPLE_H_
# define ARRAYKIT_SRC_ARRAY_TO_TUPLE_H_

# include "Python.h"

// Given a 1D or 2D array, return a 1D object array of tuples.
PyObject *
array_to_tuple_array(PyObject *Py_UNUSED(m), PyObject *a);

// Given a 2D array, return an iterator of row tuples.
PyObject *
array_to_tuple_iter(PyObject *Py_UNUSED(m), PyObject *a);

# endif // ARRAYKIT_SRC_ARRAY_TO_TUPLE_H_
