# include "Python.h"
# include "structmember.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h" // Needed for Datetime scalar expansions
# include "numpy/ufuncobject.h"

//------------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------------

// Bug in NumPy < 1.16 (https://github.com/numpy/numpy/pull/12131):
# undef PyDataType_ISBOOL
# define PyDataType_ISBOOL(obj) \
    PyTypeNum_ISBOOL(((PyArray_Descr*)(obj))->type_num)

// Given a PyObject, raise if not an array.
# define AK_CHECK_NUMPY_ARRAY(O)                                              \
    if (!PyArray_Check(O)) {                                                  \
        return PyErr_Format(PyExc_TypeError, "expected numpy array (got %s)", \
                            Py_TYPE(O)->tp_name);                             \
    }

// Given a PyObject, raise if not an array or is not one or two dimensional.
# define AK_CHECK_NUMPY_ARRAY_1D_2D(O) \
    do {\
        AK_CHECK_NUMPY_ARRAY(O)\
        int ndim = PyArray_NDIM((PyArrayObject *)O);\
        if (ndim != 1 && ndim != 2) {\
            return PyErr_Format(PyExc_NotImplementedError,\
                    "expected 1D or 2D array (got %i)",\
                    ndim);\
        }\
    } while (0)

// Placeholder of not implemented pathways / debugging.
# define AK_NOT_IMPLEMENTED(msg)\
    do {\
        PyErr_SetString(PyExc_NotImplementedError, msg);\
        return NULL;\
    } while (0)

// To simplify lines merely checking for `!value`
# define AK_RETURN_NULL_IF_NOT(obj) \
    if (!obj) { \
        return NULL; \
    }

# define _AK_DEBUG_BEGIN() \
    do {                   \
        fprintf(stderr, "XXX %s:%i:%s: ", __FILE__, __LINE__, __FUNCTION__);

# define _AK_DEBUG_END()       \
        fprintf(stderr, "\n"); \
        fflush(stderr);        \
    } while (0)

# define AK_DEBUG_OBJ(obj)              \
    _AK_DEBUG_BEGIN();                  \
        fprintf(stderr, #obj " = ");    \
        PyObject_Print(obj, stderr, 0); \
    _AK_DEBUG_END()

# define AK_DEBUG(msg)          \
    _AK_DEBUG_BEGIN();          \
        fprintf(stderr, #msg);  \
    _AK_DEBUG_END()

# if defined __GNUC__ || defined __clang__
# define AK_LIKELY(X) __builtin_expect(!!(X), 1)
# define AK_UNLIKELY(X) __builtin_expect(!!(X), 0)
# else
# define AK_LIKELY(X) (!!(X))
# define AK_UNLIKELY(X) (!!(X))
# endif

//------------------------------------------------------------------------------
// C-level utility functions
//------------------------------------------------------------------------------

// Takes and returns a PyArrayObject, optionally copying a mutable array and setting it as immutable
PyArrayObject *
AK_ImmutableFilter(PyArrayObject *a)
{
    // https://numpy.org/devdocs/reference/c-api/array.html#array-flags
    if (PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE) {
        if ((a = (PyArrayObject *)PyArray_NewCopy(a, NPY_ANYORDER))) {
            PyArray_CLEARFLAGS(a, NPY_ARRAY_WRITEABLE);
        }
        return a;
    }
    Py_INCREF(a);
    return a;
}

PyArray_Descr*
AK_ResolveDTypes(PyArray_Descr *d1, PyArray_Descr *d2)
{
    if (PyArray_EquivTypes(d1, d2)) {
        Py_INCREF(d1);
        return d1;
    }
    if (PyDataType_ISOBJECT(d1) || PyDataType_ISOBJECT(d2)
        || PyDataType_ISBOOL(d1) || PyDataType_ISBOOL(d2)
        || (PyDataType_ISSTRING(d1) != PyDataType_ISSTRING(d2))
        || ((PyDataType_ISDATETIME(d1) || PyDataType_ISDATETIME(d2))
            // PyDataType_ISDATETIME matches NPY_DATETIME or NPY_TIMEDELTA, so
            // we need to make sure we didn't get one of each:
            && !PyArray_EquivTypenums(d1->type_num, d2->type_num)))
    {
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    PyArray_Descr *d = PyArray_PromoteTypes(d1, d2);
    if (!d) {
        PyErr_Clear();
        return PyArray_DescrFromType(NPY_OBJECT);
    }
    return d;
}

PyArray_Descr*
AK_ResolveDTypeIter(PyObject *dtypes)
{
    PyObject *iterator = PyObject_GetIter(dtypes);
    AK_RETURN_NULL_IF_NOT(iterator);

    PyArray_Descr *resolved = NULL;
    PyArray_Descr *dtype;
    while ((dtype = (PyArray_Descr*) PyIter_Next(iterator))) {
        if (!PyArray_DescrCheck(dtype)) {
            PyErr_Format(
                    PyExc_TypeError, "argument must be an iterable over %s, not %s",
                    ((PyTypeObject *) &PyArrayDescr_Type)->tp_name,
                    Py_TYPE(dtype)->tp_name
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
        Py_SETREF(resolved, AK_ResolveDTypes(resolved, dtype));
        Py_DECREF(dtype);
        if (!resolved || PyDataType_ISOBJECT(resolved)) {
            break;
        }
    }
    Py_DECREF(iterator);
    return resolved;
}

// Numpy implementation: https://github.com/numpy/numpy/blob/a14c41264855e44ebd6187d7541b5b8d59bb32cb/numpy/core/src/multiarray/methods.c#L1557
PyObject*
AK_ArrayDeepCopy(PyArrayObject *array, PyObject *memo)
{
    PyObject *id = PyLong_FromVoidPtr((PyObject*)array);
    if (!id) {
        return NULL;
    }
    PyObject *found = PyDict_GetItemWithError(memo, id);
    if (found) { // found will be NULL if not in dict
        Py_INCREF(found); // got a borrowed ref, increment first
        Py_DECREF(id);
        return found;
    }
    else if (PyErr_Occurred()) {
        goto error;
    }

    // if dtype is object, call deepcopy with memo
    PyObject *array_new;
    PyArray_Descr *dtype = PyArray_DESCR(array); // borrowed ref

    if (PyDataType_ISOBJECT(dtype)) {
        PyObject *copy = PyImport_ImportModule("copy");
        if (!copy) {
            goto error;
        }
        PyObject *deepcopy = PyObject_GetAttrString(copy, "deepcopy");
        Py_DECREF(copy);
        if (!deepcopy) {
            goto error;
        }
        array_new = PyObject_CallFunctionObjArgs(deepcopy, array, memo, NULL);
        Py_DECREF(deepcopy);
        if (!array_new) {
            goto error;
        }
    }
    else {
        Py_INCREF(dtype); // PyArray_FromArray steals a reference
        array_new = PyArray_FromArray(
                array,
                dtype,
                NPY_ARRAY_ENSURECOPY);
        if (!array_new || PyDict_SetItem(memo, id, array_new)) {
            Py_XDECREF(array_new);
            goto error;
        }
    }
    // set immutable
    PyArray_CLEARFLAGS((PyArrayObject *)array_new, NPY_ARRAY_WRITEABLE);
    Py_DECREF(id);
    return array_new;
error:
    Py_DECREF(id);
    return NULL;
}


//------------------------------------------------------------------------------
// AK module public methods
//------------------------------------------------------------------------------

// Return the integer version of the pointer to underlying data-buffer of array.
static PyObject *
mloc(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY(a);
    return PyLong_FromVoidPtr(PyArray_DATA((PyArrayObject *)a));
}

//------------------------------------------------------------------------------
// filter functions

static PyObject *
immutable_filter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY(a);
    return (PyObject *)AK_ImmutableFilter((PyArrayObject *)a);
}


static PyObject *
name_filter(PyObject *Py_UNUSED(m), PyObject *n)
{
    if (AK_UNLIKELY(PyObject_Hash(n) == -1)) {
        return PyErr_Format(PyExc_TypeError, "unhashable name (type '%s')",
                            Py_TYPE(n)->tp_name);
    }
    Py_INCREF(n);
    return n;
}

// Represent a 1D array as a 2D array with length as rows of a single-column array.
// https://stackoverflow.com/questions/56182259/how-does-one-acces-numpy-multidimensionnal-array-in-c-extensions
static PyObject *
shape_filter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY_1D_2D(a);
    PyArrayObject *array = (PyArrayObject *)a;

    npy_intp size0 = PyArray_DIM(array, 0);
    // If 1D array, set size for axis 1 at 1, else use 2D array to get the size of axis 1
    npy_intp size1 = PyArray_NDIM(array) == 1 ? 1 : PyArray_DIM(array, 1);
    return Py_BuildValue("ii", size0, size1);
}

// Reshape if necessary a flat ndim 1 array into a 2D array with one columns and rows of length.
// related example: https://github.com/RhysU/ar/blob/master/ar-python.cpp
static PyObject *
column_2d_filter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY_1D_2D(a);
    PyArrayObject *array = (PyArrayObject *)a;

    if (PyArray_NDIM(array) == 1) {
        // https://numpy.org/doc/stable/reference/c-api/types-and-structures.html#c.PyArray_Dims
        npy_intp dim[2] = {PyArray_DIM(array, 0), 1};
        PyArray_Dims shape = {dim, 2};
        // PyArray_Newshape might return NULL and set PyErr, so no handling to do here
        return PyArray_Newshape(array, &shape, NPY_ANYORDER); // already a PyObject*
    }
    Py_INCREF(a); // returning borrowed ref, must increment
    return a;
}

// Reshape if necessary a column that might be 2D or 1D is returned as a 1D array.
static PyObject *
column_1d_filter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY_1D_2D(a);
    PyArrayObject *array = (PyArrayObject *)a;

    if (PyArray_NDIM(array) == 2) {
        npy_intp dim[1] = {PyArray_DIM(array, 0)};
        PyArray_Dims shape = {dim, 1};
        // NOTE: this will set PyErr if shape is not compatible
        return PyArray_Newshape(array, &shape, NPY_ANYORDER);
    }
    Py_INCREF(a);
    return a;
}

// Reshape if necessary a row that might be 2D or 1D is returned as a 1D array.
static PyObject *
row_1d_filter(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY_1D_2D(a);
    PyArrayObject *array = (PyArrayObject *)a;

    if (PyArray_NDIM(array) == 2) {
        npy_intp dim[1] = {PyArray_DIM(array, 1)};
        PyArray_Dims shape = {dim, 1};
        // NOTE: this will set PyErr if shape is not compatible
        return PyArray_Newshape(array, &shape, NPY_ANYORDER);
    }
    Py_INCREF(a);
    return a;
}

//------------------------------------------------------------------------------
// array utility

// Specialized array deepcopy that stores immutable arrays in memo dict.
static PyObject *
array_deepcopy(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *array, *memo;
    if (!PyArg_UnpackTuple(args, "array_deepcopy", 2, 2, &array, &memo)) {
        return NULL;
    }
    AK_CHECK_NUMPY_ARRAY(array);
    if (!PyDict_CheckExact(memo)) {
        PyErr_Format(PyExc_TypeError, "expected a dict (got %s)",
                Py_TYPE(memo)->tp_name);
        return NULL;
    }
    return AK_ArrayDeepCopy((PyArrayObject*)array, memo);
}

//------------------------------------------------------------------------------
// type resolution

static PyObject *
resolve_dtype(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArray_Descr *d1, *d2;
    AK_RETURN_NULL_IF_NOT(PyArg_ParseTuple(args, "O!O!:resolve_dtype",
                          &PyArrayDescr_Type, &d1, &PyArrayDescr_Type, &d2));
    return (PyObject *)AK_ResolveDTypes(d1, d2);
}

static PyObject *
resolve_dtype_iter(PyObject *Py_UNUSED(m), PyObject *arg)
{
    return (PyObject *)AK_ResolveDTypeIter(arg);
}

//------------------------------------------------------------------------------
// isin

static PyObject *
AK_isin_array_dtype_use_np(PyArrayObject *array, PyArrayObject *other, int assume_unique)
{
    PyObject* result = NULL;

    PyObject* args = PyTuple_Pack(2, (PyObject*)array, (PyObject*)other);
    AK_RETURN_NULL_IF_NOT(args);

    PyObject* kwarg = PyDict_New();
    if (!kwarg) {
        Py_DECREF(args);
        return NULL;
    }

    PyObject* assume_unique_obj = PyLong_FromLong((long)assume_unique);
    if (!assume_unique_obj) {
        goto failure;
    }

    int success = PyDict_SetItemString(kwarg, "assume_unique", assume_unique_obj);
    Py_DECREF(assume_unique_obj);
    if (success == -1) {
        goto failure;
    }

    PyObject* numpy = PyImport_ImportModule("numpy");
    if (!numpy) {
        goto failure;
    }

    PyObject* func = PyObject_GetAttrString(numpy, PyArray_NDIM(array) == 1 ? "in1d": "isin");
    Py_DECREF(numpy);
    if (!func) {
        goto failure;
    }

    result = PyObject_Call(func, args, kwarg);
    Py_DECREF(func);
failure:
    // These will always exist.
    Py_DECREF(args);
    Py_DECREF(kwarg);

    return result;
}

static PyObject *
AK_isin_array_object(PyArrayObject *array, PyArrayObject *other)
{
    /*  Algorithm:

        for loc, element in loc_iter(array):
            result[loc] = element in set(other)
    */

    PyObject *compare_elements = PyFrozenSet_New((PyObject*)other);
    AK_RETURN_NULL_IF_NOT(compare_elements);

    PyArrayObject *arrays[2];
    npy_uint32 arrays_flags[2];
    PyArray_Descr *op_dtypes[2];
    arrays[0] = array;
    arrays[1] = NULL;
    arrays_flags[0] = NPY_ITER_READONLY;
    arrays_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    op_dtypes[0] = PyArray_DescrFromType(NPY_OBJECT);
    op_dtypes[1] = PyArray_DescrFromType(NPY_BOOL);

    // No inner iteration - inner loop is handled by CopyArray code
    // Reference objects are OK.
    int iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;

    // Construct the iterator
    NpyIter *iter = NpyIter_MultiNew(
            2,              // number of arrays
            arrays,
            iter_flags,
            NPY_KEEPORDER,  // Maintain existing order for `array`
            NPY_NO_CASTING, // No casting will be required
            arrays_flags,
            op_dtypes);

    if (!iter) {
        Py_DECREF(compare_elements);
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        Py_DECREF(compare_elements);
        NpyIter_Deallocate(iter);
        return NULL;
    }

    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // If we don't need the GIL, iteration can be multi-threaded!
    NPY_BEGIN_THREADS_DEF;
    if (!NpyIter_IterationNeedsAPI(iter)) {
        // This will likely never happen, since I am pretty sure that object
        // dtypes need the API. However, I don't know enough about the internals
        // of numpy iteration to know that this will *never happen....
        NPY_BEGIN_THREADS;
    }

    do {
        char* src_data = dataptr[0];
        char* dst_data = dataptr[1];
        npy_intp size = sizeptr[0];
        npy_intp src_stride = strideptr[0];
        npy_intp dst_stride = strideptr[1];

        PyObject* obj_ref = NULL;

        while (size--) {
            // Object arrays contains pointers to PyObjects, so we will only temporarily
            // look at the reference here.
            memcpy(&obj_ref, src_data, sizeof(obj_ref));

            // 5. Assign into result whether or not the element exists in the set
            // int found = PySequence_Contains(compare_elements, ((PyObject**)data)[0]);
            npy_bool found = (npy_bool)PySequence_Contains(compare_elements, obj_ref);

            if (found == -1) {
                NpyIter_Deallocate(iter);
                Py_DECREF(compare_elements);
                return NULL;
            }

            *dst_data = found;

            src_data += src_stride;
            dst_data += dst_stride;
        }

        // Increment the iterator to the next inner loop
    } while(iternext(iter));

    NPY_END_THREADS;

    Py_DECREF(compare_elements);

    // If the API was needed, it may have thrown an error
    if (NpyIter_IterationNeedsAPI(iter) && PyErr_Occurred()) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // Get the result from the iterator object array
    PyObject *ret = (PyObject*)NpyIter_GetOperandArray(iter)[1];
    if (!ret) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    Py_INCREF(ret);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    return ret;
}

static PyObject *
isin_array(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    int array_is_unique, other_is_unique;
    PyArrayObject *array, *other;

    static char *kwlist[] = {"array", "array_is_unique", "other", "other_is_unique", NULL};

    AK_RETURN_NULL_IF_NOT(PyArg_ParseTupleAndKeywords(args, kwargs, "O!iO!i:isin_array",
                                     kwlist,
                                     &PyArray_Type, &array, &array_is_unique,
                                     &PyArray_Type, &other, &other_is_unique));

    if (PyArray_NDIM(other) != 1) {
        return PyErr_Format(PyExc_TypeError, "Expected other to be 1-dimensional");
    }

    PyArray_Descr* array_dtype = PyArray_DTYPE(array);
    PyArray_Descr* other_dtype = PyArray_DTYPE(other);

    // Use Python sets to handle object arrays
    if (PyDataType_ISOBJECT(array_dtype) || PyDataType_ISOBJECT(other_dtype)) {
        return AK_isin_array_object(array, other);
    }
    // Use numpy in1d logic for dtype arrays
    return AK_isin_array_dtype_use_np(array, other, array_is_unique && other_is_unique);
}

//------------------------------------------------------------------------------
// ArrayGO
//------------------------------------------------------------------------------

typedef struct {
    PyObject_VAR_HEAD
    PyObject *array;
    PyObject *list;
} ArrayGOObject;

static PyTypeObject ArrayGOType;

PyDoc_STRVAR(
    ArrayGO_doc,
    "\n"
    "A grow only, one-dimensional, object type array, "
    "specifically for usage in IndexHierarchy IndexLevel objects.\n"
    "\n"
    "Args:\n"
    "    own_iterable: flag iterable as ownable by this instance.\n"
);

PyDoc_STRVAR(
    ArrayGO_copy_doc,
    "Return a new ArrayGO with an immutable array from this ArrayGO\n"
);

PyDoc_STRVAR(ArrayGO_values_doc, "Return the immutable labels array\n");

//------------------------------------------------------------------------------
// ArrayGO utility functions

static int
update_array_cache(ArrayGOObject *self)
{
    if (self->list) {
        if (self->array) {
            PyObject *container = PyTuple_Pack(2, self->array, self->list);
            if (!container) {
                return -1;
            }
            Py_SETREF(self->array, PyArray_Concatenate(container, 0));
            Py_DECREF(container);
        }
        else {
            self->array = PyArray_FROM_OT(self->list, NPY_OBJECT);
        }
        PyArray_CLEARFLAGS((PyArrayObject *)self->array, NPY_ARRAY_WRITEABLE);
        Py_CLEAR(self->list);
    }
    return 0;
}

//------------------------------------------------------------------------------
// ArrayGO Methods:

static PyObject *
ArrayGO_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    char* argnames[] = {"iterable", "own_iterable", NULL};
    PyObject *iterable;
    int own_iterable = 0;
    int parsed = PyArg_ParseTupleAndKeywords(
        args, kwargs, "O|$p:ArrayGO", argnames, &iterable, &own_iterable
    );
    AK_RETURN_NULL_IF_NOT(parsed);

    ArrayGOObject *self = (ArrayGOObject *)cls->tp_alloc(cls, 0);
    AK_RETURN_NULL_IF_NOT(self);

    if (PyArray_Check(iterable)) {
        if (!PyDataType_ISOBJECT(PyArray_DESCR((PyArrayObject *)iterable))) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "only object arrays are supported");
            Py_DECREF(self);
            return NULL;
        }
        if (own_iterable) {
            // ArrayGO(np.array(...), own_iterable=True)
            PyArray_CLEARFLAGS((PyArrayObject *)iterable, NPY_ARRAY_WRITEABLE);
            self->array = iterable;
            Py_INCREF(iterable);
            return (PyObject *)self;
        }
        // ArrayGO(np.array(...))
        self->array = (PyObject *)AK_ImmutableFilter((PyArrayObject *)iterable);
        if (!self->array) {
            Py_CLEAR(self);
        }
        return (PyObject *)self;
    }
    if (PyList_CheckExact(iterable) && own_iterable) {
        // ArrayGO([...], own_iterable=True)
        self->list = iterable;
        Py_INCREF(iterable);
        return (PyObject *)self;
    }
    // ArrayGO([...])
    self->list = PySequence_List(iterable);
    if (!self->list) {
        Py_CLEAR(self);
    }
    return (PyObject *)self;
}


static PyObject *
ArrayGO_append(ArrayGOObject *self, PyObject *value)
{
    if (!self->list) {
        self->list = PyList_New(1);
        AK_RETURN_NULL_IF_NOT(self->list);

        Py_INCREF(value);
        PyList_SET_ITEM(self->list, 0, value);
    }
    else if (PyList_Append(self->list, value)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
ArrayGO_extend(ArrayGOObject *self, PyObject *values)
{
    if (!self->list) {
        self->list = PySequence_List(values);
        AK_RETURN_NULL_IF_NOT(self->list);

        Py_RETURN_NONE;
    }
    Py_ssize_t len = PyList_Size(self->list);
    if (len < 0 || PyList_SetSlice(self->list, len, len, values)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
ArrayGO_getnewargs(ArrayGOObject *self, PyObject *Py_UNUSED(unused))
{
    if (self->list && update_array_cache(self)) {
        return NULL;
    }
    return PyTuple_Pack(1, self->array);
}

static PyObject *
ArrayGO_copy(ArrayGOObject *self, PyObject *Py_UNUSED(unused))
{
    ArrayGOObject *copy = PyObject_GC_New(ArrayGOObject, &ArrayGOType);
    copy->array = self->array;
    copy->list = self->list ? PySequence_List(self->list) : NULL;
    Py_XINCREF(copy->array);
    return (PyObject *)copy;
}

static PyObject *
ArrayGO_iter(ArrayGOObject *self)
{
    if (self->list && update_array_cache(self)) {
        return NULL;
    }
    return PyObject_GetIter(self->array);
}

static PyObject *
ArrayGO_mp_subscript(ArrayGOObject *self, PyObject *key)
{
    if (self->list && update_array_cache(self)) {
        return NULL;
    }
    return PyObject_GetItem(self->array, key);
}

static Py_ssize_t
ArrayGO_mp_length(ArrayGOObject *self)
{
    return ((self->array ? PyArray_SIZE((PyArrayObject *)self->array) : 0)
            + (self->list ? PyList_Size(self->list) : 0));
}

static PyObject *
ArrayGO_values_getter(ArrayGOObject *self, void* Py_UNUSED(closure))
{
    if (self->list && update_array_cache(self)) {
        return NULL;
    }
    Py_INCREF(self->array);
    return self->array;
}

static int
ArrayGO_traverse(ArrayGOObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->array);
    Py_VISIT(self->list);
    return 0;
}

static int
ArrayGO_clear(ArrayGOObject *self)
{
    Py_CLEAR(self->array);
    Py_CLEAR(self->list);
    return 0;
}

static void
ArrayGO_dealloc(ArrayGOObject *self)
{
    Py_XDECREF(self->array);
    Py_XDECREF(self->list);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// ArrayGo method bundles

static struct PyGetSetDef ArrayGO_getset[] = {
    {"values", (getter)ArrayGO_values_getter, NULL, ArrayGO_values_doc, NULL},
    {NULL},
};

static PyMethodDef ArrayGO_methods[] = {
    {"append", (PyCFunction)ArrayGO_append, METH_O, NULL},
    {"copy", (PyCFunction)ArrayGO_copy, METH_NOARGS, ArrayGO_copy_doc},
    {"extend", (PyCFunction)ArrayGO_extend, METH_O, NULL},
    {"__getnewargs__", (PyCFunction)ArrayGO_getnewargs, METH_NOARGS, NULL},
    {NULL},
};

static PyMappingMethods ArrayGO_as_mapping = {
    .mp_length = (lenfunc)ArrayGO_mp_length,
    .mp_subscript = (binaryfunc) ArrayGO_mp_subscript,
};

//------------------------------------------------------------------------------
// ArrayGo PyTypeObject
// https://docs.python.org/3/c-api/typeobj.html

static PyTypeObject ArrayGOType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &ArrayGO_as_mapping,
    .tp_basicsize = sizeof(ArrayGOObject),
    .tp_clear = (inquiry)ArrayGO_clear,
    .tp_dealloc = (destructor)ArrayGO_dealloc,
    .tp_doc = ArrayGO_doc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_getset = ArrayGO_getset,
    .tp_iter = (getiterfunc)ArrayGO_iter,
    .tp_methods = ArrayGO_methods,
    .tp_name = "arraykit.ArrayGO",
    .tp_new = ArrayGO_new,
    .tp_traverse = (traverseproc)ArrayGO_traverse,
};

//------------------------------------------------------------------------------
// ArrayKit module definition
//------------------------------------------------------------------------------

static PyMethodDef arraykit_methods[] =  {
    {"immutable_filter", immutable_filter, METH_O, NULL},
    {"mloc", mloc, METH_O, NULL},
    {"name_filter", name_filter, METH_O, NULL},
    {"shape_filter", shape_filter, METH_O, NULL},
    {"column_2d_filter", column_2d_filter, METH_O, NULL},
    {"column_1d_filter", column_1d_filter, METH_O, NULL},
    {"row_1d_filter", row_1d_filter, METH_O, NULL},
    {"array_deepcopy", array_deepcopy, METH_VARARGS, NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"isin_array", (PyCFunction)isin_array, METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL},
};

static struct PyModuleDef arraykit_module = {
    PyModuleDef_HEAD_INIT, "arraykit", NULL, -1, arraykit_methods,
};

PyObject *
PyInit_arraykit(void)
{
    import_array();
    PyObject *m = PyModule_Create(&arraykit_module);
    if (!m ||
        PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(AK_VERSION)) ||
        PyType_Ready(&ArrayGOType) ||
        PyModule_AddObject(m, "ArrayGO", (PyObject *) &ArrayGOType))
    {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}
