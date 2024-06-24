#ifndef ARRAYKIT_SRC_UTILITIES_H_
#define ARRAYKIT_SRC_UTILITIES_H_

# include "Python.h"
# include "stdbool.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

static const size_t UCS4_SIZE = sizeof(Py_UCS4);

//------------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------------

// Given a PyObject, raise if not an array.
# define AK_CHECK_NUMPY_ARRAY(O)                 \
    if (!PyArray_Check(O)) {                     \
        return PyErr_Format(PyExc_TypeError,     \
                "Expected NumPy array, not %s.", \
                Py_TYPE(O)->tp_name);            \
    }

// Given a PyObject, raise if not an array or is not one or two dimensional.
# define AK_CHECK_NUMPY_ARRAY_1D_2D(O)                    \
    do {                                                  \
        AK_CHECK_NUMPY_ARRAY(O)                           \
        int ndim = PyArray_NDIM((PyArrayObject *)O);      \
        if (ndim != 1 && ndim != 2) {                     \
            return PyErr_Format(PyExc_NotImplementedError,\
                    "Expected 1D or 2D array, not %i.",   \
                    ndim);                                \
        }                                                 \
    } while (0)

// Given a PyObject, raise if not an array or is not two dimensional.
# define AK_CHECK_NUMPY_ARRAY_2D(O)                       \
    do {                                                  \
        AK_CHECK_NUMPY_ARRAY(O)                           \
        int ndim = PyArray_NDIM((PyArrayObject *)O);      \
        if (ndim != 2) {                                  \
            return PyErr_Format(PyExc_NotImplementedError,\
                    "Expected a 2D array, not %i.",       \
                    ndim);                                \
        }                                                 \
    } while (0)

# if defined __GNUC__ || defined __clang__
# define AK_LIKELY(X) __builtin_expect(!!(X), 1)
# define AK_UNLIKELY(X) __builtin_expect(!!(X), 0)
# else
# define AK_LIKELY(X) (!!(X))
# define AK_UNLIKELY(X) (!!(X))
# endif

// Placeholder of not implemented pathways / debugging.
# define AK_NOT_IMPLEMENTED(msg)                        \
    do {                                                \
        PyErr_SetString(PyExc_NotImplementedError, msg);\
        return NULL;                                    \
    } while (0)

# define _AK_DEBUG_BEGIN() \
    do {                   \
        fprintf(stderr, "--- %s: %i: %s: ", __FILE__, __LINE__, __FUNCTION__);

# define _AK_DEBUG_END()       \
        fprintf(stderr, "\n"); \
        fflush(stderr);        \
    } while (0)

# define AK_DEBUG(msg)          \
    _AK_DEBUG_BEGIN();          \
        fprintf(stderr, #msg);  \
    _AK_DEBUG_END()

# define AK_DEBUG_MSG_OBJ(msg, obj)     \
    _AK_DEBUG_BEGIN();                  \
        fprintf(stderr, #msg " ");      \
        PyObject_Print(obj, stderr, 0); \
    _AK_DEBUG_END()

# define AK_DEBUG_MSG_REFCNT(msg, obj)     \
    _AK_DEBUG_BEGIN();                  \
        fprintf(stderr, #msg " ref count: ");      \
        PyObject_Print(PyLong_FromSsize_t(Py_REFCNT((PyObject*)obj)), stderr, 0); \
    _AK_DEBUG_END()

# define AK_DEBUG_OBJ(obj)              \
    _AK_DEBUG_BEGIN();                  \
        fprintf(stderr, #obj " = ");    \
        PyObject_Print(obj, stderr, 0); \
    _AK_DEBUG_END()

// Takes and returns a PyArrayObject, optionally copying a mutable array and setting it as immutable
static inline PyArrayObject *
AK_immutable_filter(PyArrayObject *a)
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

// Returns NULL on error.
static inline PyArray_Descr *
AK_resolve_dtype(PyArray_Descr *d1, PyArray_Descr *d2)
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

// Returns NULL on error. Returns a new reference.
static inline PyObject *
AK_build_pair_ssize_t(Py_ssize_t a, Py_ssize_t b)
{
    PyObject* t = PyTuple_New(2);
    if (t == NULL) {
        return NULL;
    }
    PyObject* py_a = PyLong_FromSsize_t(a);
    if (py_a == NULL) {
        Py_DECREF(t);
        return NULL;
    }
    PyObject* py_b = PyLong_FromSsize_t(b);
    if (py_b == NULL) {
        Py_DECREF(t);
        Py_DECREF(py_a);
        return NULL;
    }
    // steals refs
    PyTuple_SET_ITEM(t, 0, py_a);
    PyTuple_SET_ITEM(t, 1, py_b);
    return t;
}

// Returns a new ref; returns NULL on error. Any start or stop less than 0 will be set to NULL.
static inline PyObject *
AK_build_slice(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step)
{
    PyObject* py_start = NULL;
    PyObject* py_stop = NULL;
    PyObject* py_step = NULL;

    if (start >= 0) {
        py_start = PyLong_FromSsize_t(start);
        if (py_start == NULL) {return NULL;}
    }
    if (stop >= 0) {
        py_stop = PyLong_FromSsize_t(stop);
        if (py_stop == NULL) {return NULL;}
    }
    // do not set a step if not necessary
    if (step != 0 && step != 1) {
        py_step = PyLong_FromSsize_t(step);
        if (py_step == NULL) {return NULL;}
    }

    // might be NULL, let return
    PyObject* new = PySlice_New(py_start, py_stop, py_step);

    Py_XDECREF(py_start);
    Py_XDECREF(py_stop);
    Py_XDECREF(py_step);

    return new;
}

// Utility function for converting slices; returns NULL on error; returns a new reference.
static inline PyObject *
AK_slice_to_ascending_slice(PyObject* slice, Py_ssize_t size)
{
    Py_ssize_t step_count = -1;
    Py_ssize_t start = 0;
    Py_ssize_t stop = 0;
    Py_ssize_t step = 0;

    if (PySlice_Unpack(slice, &start, &stop, &step)) {
        return NULL;
    }
    if (step > 0) {
        Py_INCREF(slice);
        return slice;
    }
    step_count = PySlice_AdjustIndices(
            size,
            &start,
            &stop,
            step);

    // step will be negative; shift original start value down to find new start
    return AK_build_slice(
            start + (step * (step_count - 1)),
            start + 1,
            -step);
}

// Given a Boolean, contiguous 1D array, return the index positions in an int64 array. Through experimentation it has been verified that doing full-size allocation of memory provides the best performance at all scales. Using NpyIter, or using, bit masks does not improve performance over pointer arithmetic. Prescanning for all empty is very effective. Note that NumPy benefits from first counting the nonzeros, then allocating only enough data for the expexted number of indices.
static inline PyObject *
AK_nonzero_1d(PyArrayObject* array) {
    PyObject* final;
    npy_intp count_max = PyArray_SIZE(array);

    if (count_max == 0) { // return empty array
        npy_intp dims = {count_max};
        final = PyArray_SimpleNew(1, &dims, NPY_INT64);
        PyArray_CLEARFLAGS((PyArrayObject*)final, NPY_ARRAY_WRITEABLE);
        return final;
    }
    lldiv_t size_div = lldiv((long long)count_max, 8); // quot, rem

    Py_ssize_t count = 0;
    // the maximum number of collected integers is equal to or less than count_max
    Py_ssize_t capacity = count_max;
    npy_int64* indices = (npy_int64*)malloc(sizeof(npy_int64) * capacity);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    if (PyArray_IS_C_CONTIGUOUS(array)) {
        npy_bool* p_start = (npy_bool*)PyArray_DATA(array);
        npy_bool* p = p_start;
        npy_bool* p_end = p + count_max;
        npy_bool* p_end_roll = p_end - size_div.rem;

        while (p < p_end_roll) {
            if (*(npy_uint64*)p == 0) {
                p += 8; // no true within this 8 byte roll region
                continue;
            }
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
            if (*p) {indices[count++] = p - p_start;}
            p++;
        }
        while (p < p_end) {
            if (*p) {indices[count++] = p - p_start;}
            p++;
        }
    }
    else {
        npy_intp i = 0; // position within Boolean array
        npy_intp i_end = count_max;
        npy_intp i_end_roll = count_max - size_div.rem;
        while (i < i_end_roll) {
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
        }
        while (i < i_end) {
            if (*(npy_bool*)PyArray_GETPTR1(array, i)) {indices[count++] = i;}
            i++;
        }
    }
    NPY_END_THREADS;

    npy_intp dims = {count};
    final = PyArray_SimpleNewFromData(1, &dims, NPY_INT64, (void*)indices);
    if (!final) {
        free(indices);
        return NULL;
    }
    // This ensures that the array frees the indices array; this has been tested by calling free(indices) and observing segfault
    PyArray_ENABLEFLAGS((PyArrayObject*)final, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS((PyArrayObject*)final, NPY_ARRAY_WRITEABLE);
    return final;
}

#endif  /* ARRAYKIT_SRC_UTILITIES_H_ */
