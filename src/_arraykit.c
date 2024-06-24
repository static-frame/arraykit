# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

# include "array_go.h"
# include "array_to_tuple.h"
# include "block_index.h"
# include "delimited_to_arrays.h"
# include "methods.h"
# include "tri_map.h"

static PyMethodDef arraykit_methods[] =  {
    {"immutable_filter", immutable_filter, METH_O, NULL},
    {"mloc", mloc, METH_O, NULL},
    {"name_filter", name_filter, METH_O, NULL},
    {"shape_filter", shape_filter, METH_O, NULL},
    {"column_2d_filter", column_2d_filter, METH_O, NULL},
    {"column_1d_filter", column_1d_filter, METH_O, NULL},
    {"row_1d_filter", row_1d_filter, METH_O, NULL},
    {"slice_to_ascending_slice", slice_to_ascending_slice, METH_VARARGS, NULL},
    {"array_deepcopy",
            (PyCFunction)array_deepcopy,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"array_to_tuple_array", array_to_tuple_array, METH_O, NULL},
    {"array_to_tuple_iter", array_to_tuple_iter, METH_O, NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"first_true_1d",
            (PyCFunction)first_true_1d,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"first_true_2d",
            (PyCFunction)first_true_2d,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"delimited_to_arrays",
            (PyCFunction)delimited_to_arrays,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"iterable_str_to_array_1d",
            (PyCFunction)iterable_str_to_array_1d,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"split_after_count",
            (PyCFunction)split_after_count,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"count_iteration", count_iteration, METH_O, NULL},
    {"nonzero_1d", nonzero_1d, METH_O, NULL},
    {"isna_element",
            (PyCFunction)isna_element,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"dtype_from_element", dtype_from_element, METH_O, NULL},
    {"get_new_indexers_and_screen",
            (PyCFunction)get_new_indexers_and_screen,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {NULL},
};

static struct PyModuleDef arraykit_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_arraykit",
    .m_doc = NULL,
    .m_size = -1,
    .m_methods = arraykit_methods,
};

PyObject *
PyInit__arraykit(void)
{
    import_array();

    ErrorInitTypeBlocks = PyErr_NewExceptionWithDoc(
            "arraykit.ErrorInitTypeBlocks",
            "RuntimeError error in block initialization.",
            PyExc_RuntimeError,
            NULL);
    if (ErrorInitTypeBlocks == NULL) {
        return NULL;
    }

    PyObject *copy = PyImport_ImportModule("copy");
    if (copy == NULL) {
        return NULL;
    }
    PyObject *deepcopy = PyObject_GetAttrString(copy, "deepcopy");
    Py_DECREF(copy);
    if (deepcopy == NULL) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&arraykit_module);
    if (!m ||
        PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(AK_VERSION)) ||
        PyType_Ready(&BlockIndexType) ||
        PyType_Ready(&BIIterType) ||
        PyType_Ready(&BIIterSeqType) ||
        PyType_Ready(&BIIterSliceType) ||
        PyType_Ready(&BIIterBoolType) ||
        PyType_Ready(&BIIterContiguousType) ||
        PyType_Ready(&BIIterBlockType) ||
        PyType_Ready(&TriMapType) ||
        PyType_Ready(&ArrayGOType) ||
        PyModule_AddObject(m, "BlockIndex", (PyObject *) &BlockIndexType) ||
        PyModule_AddObject(m, "TriMap", (PyObject *) &TriMapType) ||
        PyModule_AddObject(m, "ArrayGO", (PyObject *) &ArrayGOType) ||
        PyModule_AddObject(m, "deepcopy", deepcopy) ||
        PyModule_AddObject(m, "ErrorInitTypeBlocks", ErrorInitTypeBlocks)
    ){
        Py_DECREF(deepcopy);
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

