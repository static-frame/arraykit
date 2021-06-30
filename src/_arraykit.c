# include "Python.h"
# include "structmember.h"
# include <stdbool.h>

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

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

# define AK_INT_MAX_COERCIBLE_TO_FLOAT 9007199256349108l

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
    if (iterator == NULL) {
        // No need to set exception here. GetIter already sets TypeError
        return NULL;
    }
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
    if (!resolved) {
        // this could happen if this function gets an empty tuple
        PyErr_SetString(PyExc_ValueError, "iterable passed to resolve dtypes is empty");
    }
    return resolved;
}

// Perform a deepcopy on an array, using an optional memo dictionary, and specialized to depend on immutable arrays.
PyObject*
AK_ArrayDeepCopy(PyArrayObject *array, PyObject *memo)
{
    PyObject *id = PyLong_FromVoidPtr((PyObject*)array);
    if (!id) {
        return NULL;
    }

    if (memo) {
        PyObject *found = PyDict_GetItemWithError(memo, id);
        if (found) { // found will be NULL if not in dict
            Py_INCREF(found); // got a borrowed ref, increment first
            Py_DECREF(id);
            return found;
        }
        else if (PyErr_Occurred()) {
            goto error;
        }
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
        // if not a n object dtype, we will force a copy (even if this is an immutable array) so as to not hold on to any references
        Py_INCREF(dtype); // PyArray_FromArray steals a reference
        array_new = PyArray_FromArray(
                array,
                dtype,
                NPY_ARRAY_ENSURECOPY);
        if (!array_new) {
            goto error;
        }
        if (memo && PyDict_SetItem(memo, id, array_new)) {
            Py_DECREF(array_new);
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

    int size0 = PyArray_DIM(array, 0);
    // If 1D array, set size for axis 1 at 1, else use 2D array to get the size of axis 1
    int size1 = PyArray_NDIM(array) == 1 ? 1 : PyArray_DIM(array, 1);
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

static char *array_deepcopy_kwarg_names[] = {
    "array",
    "memo",
    NULL
};

// Specialized array deepcopy that stores immutable arrays in an optional memo dict that can be provided with kwargs.
static PyObject *
array_deepcopy(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyObject *array;
    PyObject *memo = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|O!:array_deepcopy", array_deepcopy_kwarg_names,
            &array,
            &PyDict_Type, &memo)) {
        return NULL;
    }
    AK_CHECK_NUMPY_ARRAY(array);
    return AK_ArrayDeepCopy((PyArrayObject*)array, memo);
}

//------------------------------------------------------------------------------
// type resolution

static PyObject *
resolve_dtype(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArray_Descr *d1, *d2;
    if (!PyArg_ParseTuple(args, "O!O!:resolve_dtype",
                          &PyArrayDescr_Type, &d1, &PyArrayDescr_Type, &d2))
    {
        return NULL;
    }
    return (PyObject *)AK_ResolveDTypes(d1, d2);
}

static PyObject *
resolve_dtype_iter(PyObject *Py_UNUSED(m), PyObject *arg)
{
    return (PyObject *)AK_ResolveDTypeIter(arg);
}

typedef enum IsGenCopyValues {
    ERR,
    IS_GEN,
    NOT_GEN_COPY,
    NOT_GEN_NO_COPY
} IsGenCopyValues;

static IsGenCopyValues
AK_is_gen_copy_values(PyObject *values, PyObject *frozenAutoMap)
{
    if(PyObject_HasAttrString(values, "__len__")) {
        if (PySet_Check(values) || PyDict_Check(values) ||
            PyDictValues_Check(values) || PyDictKeys_Check(values))
        {
            return NOT_GEN_COPY;
        }

        switch (PyObject_IsInstance(values, frozenAutoMap))
        {
        case -1:
            return ERR;

        case 0:
            return NOT_GEN_NO_COPY;

        default: // 1
            return NOT_GEN_COPY;
        }
    }

    return IS_GEN;
}

static PyObject*
is_gen_copy_values(PyObject *m, PyObject *arg)
{
    // This only exists to allow `AK_is_gen_copy_values` to be tested

    PyObject *frozenAutoMap = PyObject_GetAttrString(m, "FrozenAutoMap");
    if (!frozenAutoMap) {
        return NULL;
    }

    PyObject *is_gen = Py_False;
    PyObject *copy_values = Py_False;

    IsGenCopyValues is_gen_ret = AK_is_gen_copy_values(arg, frozenAutoMap);
    Py_DECREF(frozenAutoMap);

    if (is_gen_ret == IS_GEN) {
        is_gen = Py_True;
        copy_values = Py_True;
    }
    else if (is_gen_ret == NOT_GEN_COPY) {
        copy_values = Py_True;
    }
    else if (is_gen_ret == ERR) {
        return NULL;
    }

    return PyTuple_Pack(2, is_gen, copy_values);
}

static PyObject*
prepare_iter_for_array(PyObject *m, PyObject *args, PyObject *kwargs)
{
    PyObject *values;
    int restrict_copy = 0; // False by default

    static char *kwlist[] = {"values", "restrict_copy", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p:prepare_iter_for_array",
                                     kwlist,
                                     &values,
                                     &restrict_copy))
    {
        return NULL;
    }

    PyObject *frozenAutoMap = PyObject_GetAttrString(m, "FrozenAutoMap");
    if (!frozenAutoMap) {
        return NULL;
    }

    bool is_gen = false;
    bool copy_values = false;

    IsGenCopyValues is_gen_ret = AK_is_gen_copy_values(values, frozenAutoMap);
    Py_DECREF(frozenAutoMap);

    if (is_gen_ret == IS_GEN) {
        is_gen = true;
        copy_values = true;
    }
    else if (is_gen_ret == NOT_GEN_COPY) {
        copy_values = true;
    }
    else if (is_gen_ret == ERR) {
        return NULL;
    }

    Py_ssize_t len;

    if (!is_gen) {
        len = PyObject_Size(values);
        if (len == -1) {
            return NULL;
        }
        if (len == 0) {
            return PyTuple_Pack(3, Py_None, Py_False, values);
        }
    }

    PyObject *copied_values = NULL;

    if (restrict_copy) {
        copy_values = false;
    }
    else if (copy_values) {
        copied_values = PyList_New(0);
        if (!copied_values) {
            return NULL;
        }
    }

    len = 0;
    bool is_object = false;
    bool has_tuple = false;
    bool has_str = false;
    bool has_non_str = false;
    bool has_inexact = false;
    bool has_big_int = false;

    PyObject *iterator = PyObject_GetIter(values);
    if (!iterator) {
        goto failure;
    }
    PyObject *item = NULL;

    while ((item = PyIter_Next(iterator))) {
        if (copy_values) {
            if (-1 == PyList_Append(copied_values, item)) {
                goto failure_during_iteration;
            }
            ++len;
        }

        // This API call always succeeds.
        if (PyUnicode_Check(item)) {
            has_str = true;
        }
        else if (PyObject_HasAttrString(item, "__len__")) {
            has_tuple = true;
            is_object = true;
        }
        else {
            has_non_str = true;

            // These two API calls always succeed. 96% sure this catches np types
            if (!has_inexact && (PyFloat_Check(item) || PyComplex_Check(item))) {
                has_inexact = true;
            }
            else if (!has_big_int && PyLong_Check(item)) {
                int overflow;
                npy_int64 item_val = PyLong_AsLongLongAndOverflow(item, &overflow);
                if (-1 == item_val) {
                    if (PyErr_Occurred()) {
                        goto failure_during_iteration;
                    }
                }
                else {
                    has_big_int |= (overflow != 0 || Py_ABS(item_val) > AK_INT_MAX_COERCIBLE_TO_FLOAT);
                }
            }
        }

        is_object |= (has_str && has_non_str) || (has_big_int && has_inexact);

        if (is_object) {
            if (copy_values) {
                if (-1 == PyList_SetSlice(copied_values, len, len, iterator)) {
                    goto failure_during_iteration;
                }
            }
            Py_DECREF(item);
            break;
        }

        Py_DECREF(item);
    }

    Py_DECREF(iterator);

    if (PyErr_Occurred()) {
        goto failure;
    }

    PyObject *resolved = NULL;
    if (is_object) {
        resolved = (PyObject*)PyArray_DescrFromType(NPY_OBJECT);
    } else {
        resolved = Py_None;
    }
    Py_INCREF(resolved);

    PyObject *is_tuple_obj = PyBool_FromLong(has_tuple);
    if (!is_tuple_obj) {
        Py_DECREF(resolved);
        goto failure;
    }

    if (copy_values) {
        PyObject *ret = PyTuple_Pack(3, resolved, is_tuple_obj, copied_values);
        Py_DECREF(resolved);
        Py_DECREF(is_tuple_obj);
        Py_DECREF(copied_values);
        return ret;
    }

    PyObject *ret = PyTuple_Pack(3, resolved, is_tuple_obj, values);
    Py_DECREF(resolved);
    Py_DECREF(is_tuple_obj);
    return ret;

failure_during_iteration:
    Py_DECREF(iterator);
    Py_DECREF(item);

failure:
    Py_XDECREF(copied_values);
    return NULL;
}

//------------------------------------------------------------------------------
// general utility

static PyObject *
dtype_from_element(PyObject *Py_UNUSED(m), PyObject *arg)
{
    // -------------------------------------------------------------------------
    // 1. Handle fast, exact type checks first.

    // None
    if (arg == Py_None) {
        return (PyObject*)PyArray_DescrFromType(NPY_OBJECT);
    }

    // Float
    if (PyFloat_CheckExact(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_DOUBLE);
    }

    // Integers
    if (PyLong_CheckExact(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_LONG);
    }

    // Bool
    if (PyBool_Check(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_BOOL);
    }

    PyObject* dtype = NULL;

    // String
    if (PyUnicode_CheckExact(arg)) {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_UNICODE);
        if (descr == NULL) {
            return NULL;
        }
        dtype = (PyObject*)PyArray_DescrFromObject(arg, descr);
        Py_DECREF(descr);
        return dtype;
    }

    // Bytes
    if (PyBytes_CheckExact(arg)) {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_STRING);
        if (descr == NULL) {
            return NULL;
        }
        dtype = (PyObject*)PyArray_DescrFromObject(arg, descr);
        Py_DECREF(descr);
        return dtype;
    }

    // -------------------------------------------------------------------------
    // 2. Construct dtype (slightly more complicated)

    // Already known
    dtype = PyObject_GetAttrString(arg, "dtype");
    if (dtype) {
        return dtype;
    }
    PyErr_Clear();

    // -------------------------------------------------------------------------
    // 3. Handles everything else.
    return (PyObject*)PyArray_DescrFromType(NPY_OBJECT);
}

static PyObject *
isna_element(PyObject *Py_UNUSED(m), PyObject *arg)
{
    // None
    if (arg == Py_None) {
        Py_RETURN_TRUE;
    }

    // NaN
    if (PyFloat_Check(arg)) {
        return PyBool_FromLong(isnan(PyFloat_AS_DOUBLE(arg)));
    }
    if (PyArray_IsScalar(arg, Half)) {
        return PyBool_FromLong(npy_half_isnan(PyArrayScalar_VAL(arg, Half)));
    }
    if (PyArray_IsScalar(arg, Float32)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float32)));
    }
    if (PyArray_IsScalar(arg, Float64)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float64)));
    }
    # ifdef PyFloat128ArrType_Type
    if (PyArray_IsScalar(arg, Float128)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float128)));
    }
    # endif

    // Complex NaN
    if (PyComplex_Check(arg)) {
        Py_complex val = ((PyComplexObject*)arg)->cval;
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    if (PyArray_IsScalar(arg, Complex64)) {
        npy_cfloat val = PyArrayScalar_VAL(arg, Complex64);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    if (PyArray_IsScalar(arg, Complex128)) {
        npy_cdouble val = PyArrayScalar_VAL(arg, Complex128);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    # ifdef PyComplex256ArrType_Type
    if (PyArray_IsScalar(arg, Complex256)) {
        npy_clongdouble val = PyArrayScalar_VAL(arg, Complex256);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    # endif

    // NaT - Datetime
    if (PyArray_IsScalar(arg, Datetime)) {
        return PyBool_FromLong(PyArrayScalar_VAL(arg, Datetime) == NPY_DATETIME_NAT);
    }

    // NaT - Timedelta
    if (PyArray_IsScalar(arg, Timedelta)) {
        return PyBool_FromLong(PyArrayScalar_VAL(arg, Timedelta) == NPY_DATETIME_NAT);
    }

    Py_RETURN_FALSE;
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
    if (!parsed) {
        return NULL;
    }
    ArrayGOObject *self = (ArrayGOObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }

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
        if (!self->list) {
            return NULL;
        }
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
        if (!self->list) {
            return NULL;
        }
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
    {"array_deepcopy",
            (PyCFunction)array_deepcopy,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"is_gen_copy_values", is_gen_copy_values, METH_O, NULL},
    {"prepare_iter_for_array", (PyCFunction)prepare_iter_for_array, METH_VARARGS | METH_KEYWORDS, NULL},
    {"isna_element", isna_element, METH_O, NULL},
    {"dtype_from_element", dtype_from_element, METH_O, NULL},
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
    PyObject *m = PyModule_Create(&arraykit_module);
    if (!m) {
        return NULL;
    }

    PyObject *automap_module = PyImport_ImportModule("automap");
    if (!automap_module) {
        Py_DECREF(m);
        return NULL;
    }

    PyObject *frozenAutoMapClass = PyObject_GetAttrString(automap_module, "FrozenAutoMap");
    Py_DECREF(automap_module);
    if (!frozenAutoMapClass) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(AK_VERSION)) ||
        PyType_Ready(&ArrayGOType) ||
        PyModule_AddObject(m, "ArrayGO", (PyObject *) &ArrayGOType) ||
        PyModule_AddObject(m, "FrozenAutoMap", frozenAutoMapClass))
    {
        Py_DECREF(frozenAutoMapClass);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
