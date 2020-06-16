# include "Python.h"
# include "structmember.h"

# define PY_ARRAY_UNIQUE_SYMBOL SF_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

// Bug in NumPy < 1.16 (https://github.com/numpy/numpy/pull/12131):
# undef PyDataType_ISBOOL
# define PyDataType_ISBOOL(obj) PyTypeNum_ISBOOL(((PyArray_Descr*)(obj))->type_num)

typedef struct {
    PyObject_VAR_HEAD
    PyArrayObject* array;
    PyListObject* list;
    PyArray_Descr* dtype;
} ArrayGOObject;

static PyTypeObject ArrayGOType;

PyDoc_STRVAR(
    ArrayGO___doc__,
    "\n"
    "A grow only, one-dimensional, object type array, "
    "specifically for usage in IndexHierarchy IndexLevel objects.\n"
    "\n"
    "Args:\n"
    "    own_iterable: flag iterable as ownable by this instance.\n"
);

PyDoc_STRVAR(
    ArrayGO_copy___doc__,
    "Return a new ArrayGO with an immutable array from this ArrayGO\n"
);

PyDoc_STRVAR(ArrayGO_values___doc__, "Return the immutable labels array\n");

PyArrayObject* SFUtil_ImmutableFilter(PyArrayObject* src_array) {
    if (PyArray_FLAGS(src_array) & NPY_ARRAY_WRITEABLE) {
        PyArrayObject* dst_array = (PyArrayObject*) PyArray_NewCopy(src_array, NPY_ANYORDER);
        PyArray_CLEARFLAGS(dst_array, NPY_ARRAY_WRITEABLE);
        return dst_array;
    }

    Py_INCREF(src_array);
    return src_array;
}


PyArray_Descr* SFUtil_ResolveDTypes(PyArray_Descr* d1, PyArray_Descr* d2) {

    if (PyArray_EquivTypes(d1, d2)) {
        Py_INCREF(d1);
        return d1;
    }

    if (
        PyDataType_ISOBJECT(d1)
        || PyDataType_ISOBJECT(d2)
        || PyDataType_ISBOOL(d1)
        || PyDataType_ISBOOL(d2)
        || (PyDataType_ISSTRING(d1) != PyDataType_ISSTRING(d2))
        || (
            /* PyDataType_ISDATETIME matches both NPY_DATETIME *and* NPY_TIMEDELTA,
            So we need the PyArray_EquivTypenums check too: */
            (PyDataType_ISDATETIME(d1) || PyDataType_ISDATETIME(d2))
            && !PyArray_EquivTypenums(d1->type_num, d2->type_num)
        )
    ) {
        return PyArray_DescrFromType(NPY_OBJECT);
    }

    PyArray_Descr* result = PyArray_PromoteTypes(d1, d2);

    if (!result) {
        PyErr_Clear();
        return PyArray_DescrFromType(NPY_OBJECT);
    }

    return result;
}


PyArray_Descr* SFUtil_ResolveDTypesIter(PyObject* dtypes) {

    PyObject* iterator;
    PyArray_Descr* resolved;
    PyArray_Descr* dtype;
    PyArray_Descr* temp;

    iterator = PyObject_GetIter(dtypes);

    if (iterator == NULL) {
        return NULL;
    }

    resolved = NULL;

    while ((dtype = (PyArray_Descr*) PyIter_Next(iterator))) {

        if (!PyArray_DescrCheck(dtype)) {

            PyErr_Format(
                PyExc_TypeError, "argument must be an iterable over %s, not %s",
                ((PyTypeObject*) &PyArrayDescr_Type)->tp_name, Py_TYPE(dtype)->tp_name
            );

            Py_DECREF(iterator);
            Py_DECREF(dtype);
            Py_XDECREF(resolved);

            return NULL;
        }

        if (!resolved) {
            resolved = dtype;
            continue;
        }

        temp = SFUtil_ResolveDTypes(resolved, dtype);

        Py_DECREF(resolved);
        Py_DECREF(dtype);

        resolved = temp;

        if (!resolved || PyDataType_ISOBJECT(resolved)) {
            break;
        }
    }

    Py_DECREF(iterator);

    return resolved;
}


static PyObject* immutable_filter(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (!PyArray_Check(arg)) {
        return PyErr_Format(PyExc_TypeError,
            "immutable_filter() argument must be numpy array, not %s",
            Py_TYPE(arg)->tp_name
        );
    }

    return (PyObject*) SFUtil_ImmutableFilter((PyArrayObject*) arg);
}


static PyObject* mloc(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (!PyArray_Check(arg)) {
        return PyErr_Format(PyExc_TypeError,
            "mloc() argument must be numpy array, not %s", Py_TYPE(arg)->tp_name
        );
    }

    return PyLong_FromVoidPtr(PyArray_DATA((PyArrayObject*) arg));
}


static PyObject* resolve_dtype_iter(PyObject* Py_UNUSED(util), PyObject* arg) {
    return (PyObject*) SFUtil_ResolveDTypesIter(arg);
}


static PyObject* resolve_dtype(PyObject* Py_UNUSED(util), PyObject* args) {

    PyArray_Descr *d1;
    PyArray_Descr *d2;

    if (
        !PyArg_ParseTuple(
            args, "O!O!:resolve_dtype",
            &PyArrayDescr_Type, &d1, &PyArrayDescr_Type, &d2
        )
    ) {
        return NULL;
    }

    return (PyObject*) SFUtil_ResolveDTypes(d1, d2);
}


static PyObject* name_filter(PyObject* Py_UNUSED(util), PyObject* arg) {

    if (PyObject_Hash(arg) == -1) {
        return PyErr_Format(
            PyExc_TypeError, "Unhashable name (type '%s').",
            Py_TYPE(arg)->tp_name
        );
    }

    Py_INCREF(arg);
    return arg;
}

static int update_array_cache(ArrayGOObject* self) {

    PyObject* container;
    PyObject* temp;

    if (self->list) {

        if (self->array) {

            container = PyTuple_Pack(2, (PyObject*) self->array, (PyObject*) self->list);

            if (!container) {
                return -1;
            }

            temp = (PyObject*) self->array;
            self->array = (PyArrayObject*) PyArray_Concatenate(container, 0);
            Py_DECREF(container);
            Py_DECREF(temp);

        } else {

            self->array = (PyArrayObject*) PyArray_FROM_OT((PyObject*) self->list, self->dtype->type_num);
        }

        PyArray_CLEARFLAGS(self->array, NPY_ARRAY_WRITEABLE);

        temp = (PyObject*) self->list;
        self->list = NULL;
        Py_DECREF(temp);
    }

    return 0;
}

/* Methods: */


static int ArrayGO___init__(ArrayGOObject* self, PyObject* args, PyObject* kwargs) {

    PyObject* temp;
    PyObject* iterable;
    int own_iterable;
    int parsed;

    char* argnames[] = {"iterable", "dtype", "own_iterable", NULL};

    temp = (PyObject*) self->dtype;
    self->dtype = NULL;
    Py_XDECREF(temp);

    parsed = PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|$O&p:ArrayGO", argnames,
        &iterable, PyArray_DescrConverter, &self->dtype, &own_iterable
    );

    if (!parsed) {
        return -1;
    }

    if (!self->dtype) {
        self->dtype = PyArray_DescrFromType(NPY_OBJECT);
    }

    if (PyArray_Check(iterable)) {

        temp = (PyObject*) self->array;

        if (own_iterable) {
            PyArray_CLEARFLAGS((PyArrayObject*) iterable, NPY_ARRAY_WRITEABLE);
            Py_INCREF(iterable);
        } else {
            iterable = (PyObject*) SFUtil_ImmutableFilter((PyArrayObject*) iterable);
        }

        if (!PyArray_EquivTypes(PyArray_DESCR((PyArrayObject*) iterable), self->dtype)) {
            PyErr_Format(
                PyExc_TypeError, "bad dtype given to ArrayGO initializer (expected '%S', got '%S')",
                PyArray_DESCR((PyArrayObject*) iterable), self->dtype
            );
            return -1;
        }

        self->array = (PyArrayObject*) iterable;
        Py_XDECREF(temp);

        temp = (PyObject*) self->list;
        self->list = NULL;
        Py_XDECREF(temp);

    } else {

        temp = (PyObject*) self->list;

        if (PyList_Check(iterable) && own_iterable) {
            Py_INCREF(iterable);
        } else {
            iterable = PySequence_List(iterable);
        }

        self->list = (PyListObject*) iterable;
        Py_XDECREF(temp);

        temp = (PyObject*) self->array;
        self->array = NULL;
        Py_XDECREF(temp);
    }

    return 0;
}


static PyObject* ArrayGO_append(ArrayGOObject* self, PyObject* value) {
    if (!self->list) {
        self->list = (PyListObject*) PyList_New(1);
        if (!self->list) {
            return NULL;
        }
        Py_INCREF(value);
        PyList_SET_ITEM(self->list, 0, value);
    } else if (PyList_Append((PyObject*) self->list, value)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject* ArrayGO_extend(ArrayGOObject* self, PyObject* values) {
    if (!self->list) {
        self->list = (PyListObject*) PySequence_List(values);
        if (!self->list) {
            return NULL;
        }

    } else {
        Py_ssize_t len = PyList_Size((PyObject*) self->list);
        if (len < 0 || PyList_SetSlice((PyObject*) self->list, len, len, values)) {
            return NULL;
        }
    }
    Py_RETURN_NONE;
}


static PyObject* ArrayGO_copy(ArrayGOObject* self, PyObject* Py_UNUSED(unused)) {
    ArrayGOObject* copy = PyObject_New(ArrayGOObject, &ArrayGOType);
    copy->array = self->array;
    copy->list = (PyListObject*) PySequence_List((PyObject*) self->list);
    copy->dtype = self->dtype;
    Py_XINCREF(copy->array);
    Py_INCREF(copy->dtype);
    return (PyObject*) copy;
}

static PyObject* ArrayGO___iter__(ArrayGOObject* self){
    return (self->list && update_array_cache(self)) ? NULL : PyObject_GetIter((PyObject*) self->array);
}

static PyObject* ArrayGO___getitem__(ArrayGOObject* self, PyObject* key) {
    return (self->list && update_array_cache(self)) ? NULL : PyObject_GetItem((PyObject*) self->array, key);
}

static Py_ssize_t ArrayGO___len__(ArrayGOObject* self) {
    return (self->array ? PyArray_SIZE(self->array) : 0) + (self->list ? PyList_Size((PyObject*) self->list) : 0);
}

static PyObject* ArrayGO_values___get__(ArrayGOObject* self, void* Py_UNUSED(closure)) {
    return (self->list && update_array_cache(self)) ? NULL : (Py_INCREF(self->array), (PyObject*) self->array);
}

static void ArrayGO___del__(ArrayGOObject* self) {
    Py_XDECREF(self->dtype);
    Py_XDECREF(self->array);
    Py_XDECREF(self->list);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static struct PyGetSetDef ArrayGO_properties[] = {
    {"values", (getter)ArrayGO_values___get__, NULL, ArrayGO_values___doc__, NULL},
    {NULL},
};

static PyMethodDef ArrayGO_methods[] = {
    {"append", (PyCFunction) ArrayGO_append, METH_O, NULL},
    {"extend", (PyCFunction) ArrayGO_extend, METH_O, NULL},
    {"copy", (PyCFunction) ArrayGO_copy, METH_NOARGS, ArrayGO_copy___doc__},
    {NULL},
};

static PyMappingMethods ArrayGO_as_mapping = {
    .mp_length = (lenfunc) ArrayGO___len__,
    .mp_subscript = (binaryfunc) ArrayGO___getitem__,
};

static PyTypeObject ArrayGOType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &ArrayGO_as_mapping,
    .tp_basicsize = sizeof(ArrayGOObject),
    .tp_dealloc = (destructor) ArrayGO___del__,
    .tp_doc = ArrayGO___doc__,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = ArrayGO_properties,
    .tp_init = (initproc) ArrayGO___init__,
    .tp_iter = (getiterfunc) ArrayGO___iter__,
    .tp_methods = ArrayGO_methods,
    .tp_name = "ArrayGO",
    .tp_new = PyType_GenericNew,
};

// Boilerplate:

static PyMethodDef arraykit_methods[] =  {
    {"immutable_filter", immutable_filter, METH_O, NULL},
    {"mloc", mloc, METH_O, NULL},
    {"name_filter", name_filter, METH_O, NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {NULL},
};

static struct PyModuleDef arraykit_module = {
    PyModuleDef_HEAD_INIT, "arraykit", NULL, -1, arraykit_methods,
};

PyObject* PyInit_arraykit(void) {
    import_array();
    PyObject* arraykit = PyModule_Create(&arraykit_module);
    if (!arraykit ||
        PyModule_AddStringConstant(arraykit, "__version__", AK_VERSION) ||
        PyType_Ready(&ArrayGOType) ||
        PyModule_AddObject(arraykit, "ArrayGO", (PyObject*) &ArrayGOType))
    {
        Py_XDECREF(arraykit);
        return NULL;
    }

    return arraykit;
}
