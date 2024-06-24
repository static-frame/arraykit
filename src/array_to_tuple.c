# include "Python.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

# include "array_to_tuple.h"
# include "utilities.h"

// Given a 1D or 2D array, return a 1D object array of tuples.
PyObject *
array_to_tuple_array(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY(a);
    PyArrayObject *input_array = (PyArrayObject *)a;
    int ndim = PyArray_NDIM(input_array);
    if (ndim != 1 && ndim != 2) {
        return PyErr_Format(PyExc_NotImplementedError,
                "Expected 1D or 2D array, not %i.",
                ndim);
    }

    npy_intp num_rows = PyArray_DIM(input_array, 0);
    npy_intp dims[] = {num_rows};
    // NOTE: this initializes values to NULL, not None
    PyObject* output = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    if (output == NULL) {
        return NULL;
    }

    PyObject** output_data = (PyObject**)PyArray_DATA((PyArrayObject*)output);
    PyObject** p = output_data;
    PyObject** p_end = p + num_rows;
    npy_intp i = 0;
    PyObject* tuple;
    PyObject* item;

    if (ndim == 2) {
        npy_intp num_cols = PyArray_DIM(input_array, 1);
        npy_intp j;
        while (p < p_end) {
            tuple = PyTuple_New(num_cols);
            if (tuple == NULL) {
                goto error;
            }
            for (j = 0; j < num_cols; ++j) {
                // cannot assume input_array is contiguous
                item = PyArray_ToScalar(PyArray_GETPTR2(input_array, i, j), input_array);
                if (item == NULL) {
                    Py_DECREF(tuple);
                    goto error;
                }
                PyTuple_SET_ITEM(tuple, j, item); // steals reference to item
            }
            *p++ = tuple; // assign with new ref, no incr needed
            i++;
        }
    }
    else if (PyArray_TYPE(input_array) != NPY_OBJECT) { // ndim == 1, not object
        while (p < p_end) {
            tuple = PyTuple_New(1);
            if (tuple == NULL) {
                goto error;
            }
            // scalar returned in is native PyObject from object arrays
            item = PyArray_ToScalar(PyArray_GETPTR1(input_array, i), input_array);
            if (item == NULL) {
                Py_DECREF(tuple);
                goto error;
            }
            PyTuple_SET_ITEM(tuple, 0, item); // steals reference to item
            *p++ = tuple; // assign with new ref, no incr needed
            i++;
        }
    }
    else { // ndim == 1, object
        while (p < p_end) {
            item = *(PyObject**)PyArray_GETPTR1(input_array, i);
            Py_INCREF(item); // always incref
            if (PyTuple_Check(item)) {
                tuple = item; // do not double pack
            }
            else {
                tuple = PyTuple_New(1);
                if (tuple == NULL) {
                    goto error;
                }
                PyTuple_SET_ITEM(tuple, 0, item); // steals reference to item
            }
            *p++ = tuple; // assign with new ref, no incr needed
            i++;
        }
    }
    PyArray_CLEARFLAGS((PyArrayObject *)output, NPY_ARRAY_WRITEABLE);
    return output;
error:
    p = output_data;
    p_end = p + num_rows;
    while (p < p_end) { // decref all tuples within array
        Py_XDECREF(*p++); // xdec as might be NULL
    }
    Py_DECREF(output);
    return NULL;
}

//------------------------------------------------------------------------------
// ArrayToTupleIterator

static PyTypeObject ATTType;

typedef struct ATTObject {
    PyObject_HEAD
    PyArrayObject* array;
    npy_intp num_rows;
    npy_intp num_cols;
    Py_ssize_t pos; // current index state, mutated in-place
} ATTObject;

static inline PyObject *
ATT_new(PyArrayObject* array,
        npy_intp num_rows,
        npy_intp num_cols) {
    ATTObject* a2dt = PyObject_New(ATTObject, &ATTType);
    if (!a2dt) {
        return NULL;
    }
    Py_INCREF((PyObject*)array);
    a2dt->array = array;
    a2dt->num_rows = num_rows;
    a2dt->num_cols = num_cols; // -1 for 1D array
    a2dt->pos = 0;
    return (PyObject *)a2dt;
}

static inline void
ATT_dealloc(ATTObject *self) {
    Py_DECREF((PyObject*)self->array);
    PyObject_Del((PyObject*)self);
}

static inline PyObject*
ATT_iter(ATTObject *self) {
    Py_INCREF(self);
    return (PyObject*)self;
}

static inline PyObject *
ATT_iternext(ATTObject *self) {
    Py_ssize_t i = self->pos;
    if (i < self->num_rows) {
        npy_intp num_cols = self->num_cols;
        PyArrayObject* array = self->array;
        PyObject* item;
        PyObject* tuple;

        if (num_cols > -1) { // ndim == 2
            tuple = PyTuple_New(num_cols);
            if (tuple == NULL) {
                return NULL;
            }
            for (npy_intp j = 0; j < num_cols; ++j) {
                // cannot assume array is contiguous
                item = PyArray_ToScalar(PyArray_GETPTR2(array, i, j), array);
                if (item == NULL) {
                    Py_DECREF(tuple);
                    return NULL;
                }
                PyTuple_SET_ITEM(tuple, j, item); // steals ref
            }
        }
        else if (PyArray_TYPE(array) != NPY_OBJECT) { // ndim == 1, not object
            tuple = PyTuple_New(1);
            if (tuple == NULL) {
                return NULL;
            }
            item = PyArray_ToScalar(PyArray_GETPTR1(array, i), array);
            if (item == NULL) {
                Py_DECREF(tuple);
                return NULL;
            }
            PyTuple_SET_ITEM(tuple, 0, item); // steals ref
        }
        else { // ndim == 1, object
            item = *(PyObject**)PyArray_GETPTR1(array, i);
            Py_INCREF(item); // always incref
            if (PyTuple_Check(item)) {
                tuple = item; // do not double pack
            }
            else {
                tuple = PyTuple_New(1);
                if (tuple == NULL) {
                    Py_DECREF(item);
                    return NULL;
                }
                PyTuple_SET_ITEM(tuple, 0, item); // steals ref
            }
        }
        self->pos++;
        return tuple;
    }
    return NULL;
}

// static PyObject *
// ATT_reversed(ATTObject *self) {
//     return ATT_new(self->bi, !self->reversed);
// }

static inline PyObject *
ATT_length_hint(ATTObject *self) {
    Py_ssize_t len = Py_MAX(0, self->num_rows - self->pos);
    return PyLong_FromSsize_t(len);
}

static PyMethodDef ATT_methods[] = {
    {"__length_hint__", (PyCFunction)ATT_length_hint, METH_NOARGS, NULL},
    // {"__reversed__", (PyCFunction)ATT_reversed, METH_NOARGS, NULL},
    {NULL},
};

static PyTypeObject ATTType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(ATTObject),
    .tp_dealloc = (destructor) ATT_dealloc,
    .tp_iter = (getiterfunc) ATT_iter,
    .tp_iternext = (iternextfunc) ATT_iternext,
    .tp_methods = ATT_methods,
    .tp_name = "arraykit.ATTIterator",
};

// Given a 2D array, return an iterator of row tuples.
PyObject *
array_to_tuple_iter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY(a);
    PyArrayObject *array = (PyArrayObject *)a;
    int ndim = PyArray_NDIM(array);
    if (ndim != 1 && ndim != 2) {
        return PyErr_Format(PyExc_NotImplementedError,
                "Expected 1D or 2D array, not %i.",
                ndim);
    }
    npy_intp num_rows = PyArray_DIM(array, 0);
    npy_intp num_cols = -1; // indicate 1d
    if (ndim == 2) {
        num_cols = PyArray_DIM(array, 1);
    }
    return ATT_new(array, num_rows, num_cols);
}
