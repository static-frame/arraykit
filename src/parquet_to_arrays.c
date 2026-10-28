# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NO_IMPORT_ARRAY
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include "numpy/arrayobject.h"

# include "parquet_to_arrays.h"

PyObject *
parquet_to_arrays(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwargs)
{
    PyObject *source;
    PyObject *columns_select = Py_None;
    static char *kwlist[] = {"file", "columns_select", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|$O:parquet_to_arrays",
            kwlist, &source, &columns_select)) {
        return NULL;
    }

    PyObject *selection = NULL;
    PyObject *source_normalized = NULL;
    PyObject *module = NULL;
    PyObject *reader = NULL;
    PyObject *paths = NULL;
    PyObject *indices = NULL;
    PyObject *schema = NULL;
    PyObject *names = NULL;
    PyObject *lookup = NULL;
    PyObject *seen = NULL;
    PyObject *table = NULL;
    PyObject *result = NULL;
    PyObject *value = NULL;
    PyObject *column = NULL;
    PyObject *array = NULL;

    if (columns_select != Py_None) {
        if (!PyList_Check(columns_select)) {
            PyErr_SetString(PyExc_TypeError,
                    "columns_select must be a list of strings or None");
            goto error;
        }
        selection = PyList_GetSlice(columns_select, 0, PyList_GET_SIZE(columns_select));
        if (selection == NULL) goto error;
        seen = PySet_New(NULL);
        if (seen == NULL) goto error;
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(selection); ++i) {
            PyObject *name = PyList_GET_ITEM(selection, i);
            if (!PyUnicode_Check(name)) {
                PyErr_SetString(PyExc_TypeError,
                        "columns_select must contain only strings");
                goto error;
            }
            int contains = PySet_Contains(seen, name);
            if (contains < 0) goto error;
            if (contains) {
                PyErr_Format(PyExc_ValueError, "duplicate column selection: %R", name);
                goto error;
            }
            if (PySet_Add(seen, name) < 0) goto error;
        }
    }

    source_normalized = PyOS_FSPath(source);
    if (source_normalized == NULL) {
        if (!PyErr_ExceptionMatches(PyExc_TypeError)) goto error;
        PyErr_Clear();
        source_normalized = Py_NewRef(source);
    }
    if (PyBytes_Check(source_normalized)) {
        value = PyUnicode_DecodeFSDefaultAndSize(
                PyBytes_AS_STRING(source_normalized), PyBytes_GET_SIZE(source_normalized));
        if (value == NULL) goto error;
        Py_SETREF(source_normalized, value);
        value = NULL;
    }
    module = PyImport_ImportModule("pyarrow._parquet");
    if (module == NULL) goto error;
    value = PyObject_GetAttrString(module, "ParquetReader");
    if (value == NULL) goto error;
    reader = PyObject_CallNoArgs(value);
    Py_CLEAR(value);
    if (reader == NULL) goto error;
    value = PyObject_CallMethod(reader, "open", "(O)", source_normalized);
    if (value == NULL) goto error;
    Py_CLEAR(value);

    if (selection != NULL) {
        paths = PyObject_GetAttrString(reader, "column_paths");
        schema = PyObject_GetAttrString(reader, "schema_arrow");
        if (paths == NULL || schema == NULL) goto error;
        names = PyObject_GetAttrString(schema, "names");
        if (names == NULL) goto error;
        indices = PyList_New(0);
        if (indices == NULL) goto error;
        lookup = PyDict_New();
        if (lookup == NULL) goto error;
        for (Py_ssize_t i = 0; i < PyList_Size(names); ++i) {
            PyObject *name = PyList_GetItem(names, i);
            int contains = PyDict_Contains(lookup, name);
            if (contains < 0) goto error;
            value = contains ? Py_NewRef(Py_None) : PyList_New(0);
            if (value == NULL || PyDict_SetItem(lookup, name, value) < 0) goto error;
            Py_CLEAR(value);
        }
        // Parquet indices refer to leaves, not top-level Arrow fields.
        for (Py_ssize_t i = 0; i < PyList_Size(paths); ++i) {
            PyObject *path = PyList_GetItem(paths, i);
            PyObject *root = PyList_GetItem(path, 0);
            if (root == NULL) goto error;
            PyObject *leaves = PyDict_GetItemWithError(lookup, root);
            if (leaves == NULL) {
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_ValueError, "inconsistent Parquet schema");
                }
                goto error;
            }
            if (leaves != Py_None) {
                value = PyLong_FromSsize_t(i);
                if (value == NULL || PyList_Append(leaves, value) < 0) goto error;
                Py_CLEAR(value);
            }
        }
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(selection); ++i) {
            PyObject *name = PyList_GET_ITEM(selection, i);
            PyObject *leaves = PyDict_GetItemWithError(lookup, name);
            if (leaves == NULL && PyErr_Occurred()) goto error;
            if (leaves == NULL || leaves == Py_None) {
                PyErr_Format(PyExc_ValueError,
                        "column name is missing or ambiguous: %R", name);
                goto error;
            }
            Py_ssize_t size = PyList_GET_SIZE(indices);
            if (PyList_SetSlice(indices, size, size, leaves) < 0) goto error;
        }
    }

    // Both reading and NumPy conversion are compiled PyArrow entry points.
    table = PyObject_CallMethod(reader, "read_all", "(O)",
            indices == NULL ? Py_None : indices);
    if (table == NULL) goto error;
    value = PyObject_GetAttrString(table, "num_columns");
    if (value == NULL) goto error;
    Py_ssize_t count = PyLong_AsSsize_t(value);
    Py_CLEAR(value);
    if (count == -1 && PyErr_Occurred()) goto error;
    result = PyList_New(count);
    if (result == NULL) goto error;
    for (Py_ssize_t i = 0; i < count; ++i) {
        column = PyObject_CallMethod(table, "column", "n", i);
        if (column == NULL) goto error;
        array = PyObject_CallMethod(column, "to_numpy", NULL);
        Py_CLEAR(column);
        if (array == NULL) goto error;
        if (!PyArray_Check(array) || PyArray_NDIM((PyArrayObject *)array) != 1) {
            PyErr_SetString(PyExc_TypeError, "Parquet column did not produce a 1D array");
            goto error;
        }
        PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
        PyList_SET_ITEM(result, i, array);
        array = NULL;
    }
    goto done;

error:
    Py_CLEAR(result);
done:
    Py_XDECREF(array);
    Py_XDECREF(column);
    Py_XDECREF(value);
    Py_XDECREF(table);
    Py_XDECREF(names);
    Py_XDECREF(schema);
    Py_XDECREF(lookup);
    Py_XDECREF(seen);
    Py_XDECREF(indices);
    Py_XDECREF(paths);
    Py_XDECREF(reader);
    Py_XDECREF(module);
    Py_XDECREF(selection);
    Py_XDECREF(source_normalized);
    return result;
}
