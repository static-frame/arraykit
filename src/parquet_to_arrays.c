# include "Python.h"

# include "parquet_to_arrays.h"

PyObject *
parquet_to_arrays(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyObject *module = PyImport_ImportModule("arraykit._parquet");
    if (module == NULL) {
        if (PyErr_ExceptionMatches(PyExc_ModuleNotFoundError)) {
            PyObject *error = PyErr_GetRaisedException();
            PyObject *name = PyObject_GetAttrString(error, "name");
            int missing_backend = name != NULL && PyUnicode_Check(name)
                    && PyUnicode_CompareWithASCIIString(name, "arraykit._parquet") == 0;
            Py_XDECREF(name);
            PyErr_Clear();
            if (missing_backend) {
                Py_DECREF(error);
                PyErr_SetString(PyExc_ImportError,
                        "Native Parquet support is not installed; rebuild with "
                        "ARRAYKIT_WITH_PARQUET=1 and Arrow/Parquet C++ libraries");
            }
            else {
                PyErr_SetRaisedException(error);
            }
        }
        return NULL;
    }
    PyObject *function = PyObject_GetAttrString(module, "parquet_to_arrays");
    Py_DECREF(module);
    if (function == NULL) return NULL;
    PyObject *result = PyObject_Call(function, args, kwargs);
    Py_DECREF(function);
    return result;
}
