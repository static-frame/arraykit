#ifndef ARRAYKIT_SRC_UTILITIES_H_
#define ARRAYKIT_SRC_UTILITIES_H_

# include "Python.h"
# include "stdbool.h"

# include "numpy/arrayobject.h"

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
PyArrayObject *
AK_immutable_filter(PyArrayObject *a);

// Returns NULL on error.
PyArray_Descr *
AK_resolve_dtype(PyArray_Descr *d1, PyArray_Descr *d2);

// Returns NULL on error. Returns a new reference.
PyObject *
AK_build_pair_ssize_t(Py_ssize_t a, Py_ssize_t b);

// Returns a new ref; returns NULL on error. Any start or stop less than 0 will be set to NULL.
PyObject *
AK_build_slice(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t step);

// Utility function for converting slices; returns NULL on error; returns a new reference.
PyObject *
AK_slice_to_ascending_slice(PyObject* slice, Py_ssize_t size);

// Given a Boolean, contiguous 1D array, return the index positions in an int64 array. Through experimentation it has been verified that doing full-size allocation of memory provides the best performance at all scales. Using NpyIter, or using, bit masks does not improve performance over pointer arithmetic. Prescanning for all empty is very effective. Note that NumPy benefits from first counting the nonzeros, then allocating only enough data for the expexted number of indices.
PyObject *
AK_nonzero_1d(PyArrayObject* array);

#endif  /* ARRAYKIT_SRC_UTILITIES_H_ */
