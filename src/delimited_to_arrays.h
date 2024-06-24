# ifndef ARRAYKIT_SRC_DELIMITED_TO_ARRAYS_H_
# define ARRAYKIT_SRC_DELIMITED_TO_ARRAYS_H_

# include "Python.h"

// NOTE: implement skip_header, skip_footer in client Python, not here.
PyObject *
delimited_to_arrays(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

PyObject *
iterable_str_to_array_1d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

PyObject *
split_after_count(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs);

# endif /* ARRAYKIT_SRC_DELIMITED_TO_ARRAYS_H_ */
