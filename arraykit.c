# include "Python.h"
# include "structmember.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

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

// Placeholder of not implemented functions.
# define AK_NOT_IMPLEMENTED\
    do {\
        PyErr_SetNone(PyExc_NotImplementedError);\
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
    return resolved;
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

    int size0 = (int)PyArray_DIM(array, 0);
    // If 1D array, set size for axis 1 at 1, else use 2D array to get the size of axis 1
    int size1 = (int)(PyArray_NDIM(array) == 1 ? 1 : PyArray_DIM(array, 1));
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

//------------------------------------------------------------------------------
// rolling

static int
assign_into_slice_from_slice(PyObject *dst, int dst_start, int dst_stop,
                             PyObject *src, int src_start, int src_stop)
{
    PyObject* shifted_src = PySequence_GetSlice(src, src_start, src_stop);
    if (!shifted_src) {
        return -1;
    }

    int success = PySequence_SetSlice(dst, dst_start, dst_stop, shifted_src);
    Py_DECREF(shifted_src);
    return success;
}

// Naive Re-implementation of C
static PyObject *
_roll_1d_a(PyArrayObject* array, int shift)
{
    /*
        cls           ak           ref          ref/ak
        Roll1dInt     3.32787074   4.06750092   1.22225328
        Roll1dFloat   3.32698173   4.06643037   1.2222581
        Roll1dObject  37.89614459  38.76268129  1.02286609
    */

    // Create an empty array
    PyArray_Descr* dtype = PyArray_DESCR(array);
    Py_INCREF(dtype); // PyArray_Empty steals a reference to dtype

    PyObject* post = PyArray_Empty(
                PyArray_NDIM(array),
                PyArray_DIMS(array),
                dtype,
                0);
    if (!post) {
        return NULL;
    }

    int success;

    // First Assign
    success = assign_into_slice_from_slice(post, 0, shift, (PyObject*)array, -shift, (int)PyArray_SIZE(array));
    if (success == -1) {
        Py_DECREF(post);
        return NULL;
    }

    // Second Assign
    success = assign_into_slice_from_slice(post, shift, (int)PyArray_SIZE(array), (PyObject*)array, 0, -shift);
    if (success == -1) {
        Py_DECREF(post);
        return NULL;
    }

    return post;
}

// Manual iteration using Numpy C api
static PyObject *
_roll_1d_b(PyArrayObject* array, int shift, int size)
{
    /*
        cls           ak          ref         ref/ak
        Roll1dInt     3.94763173  0.13514971  0.03423564
        Roll1dFloat   3.95269516  0.13621643  0.03446166
        Roll1dObject  1.03418866  0.46459488  0.4492361
    */

    // Create an empty array
    PyArray_Descr* dtype = PyArray_DESCR(array);
    Py_INCREF(dtype); // PyArray_Empty steals a reference to dtype

    PyArrayObject* post = (PyArrayObject*)PyArray_Empty(
                PyArray_NDIM(array),
                PyArray_DIMS(array),
                dtype,
                0);
    if (!post) {
        return NULL;
    }

    npy_intp array_stride = PyArray_STRIDE(array, 0);
    npy_intp post_stride = PyArray_STRIDE(post, 0);
    char* array_dataptr = PyArray_BYTES(array);
    char* post_dataptr = PyArray_BYTES(post);

    for (int i = 0; i < size; ++i) {
        int src_i = (i + size - shift) % size;

        PyObject* obj = PyArray_GETITEM(array, array_dataptr + (array_stride * src_i));
        if (!obj) {
            Py_DECREF(post);
            return NULL;
        }

        if (PyArray_SETITEM(post, post_dataptr + (i * post_stride), obj) == -1) {
            Py_DECREF(post);
            return NULL;
        }
    }

    return (PyObject*)post;
}

// Being clever with C for primitives, struggling with Objects
static PyObject *
_roll_1d_c(PyArrayObject *array, int shift)
{
    /*
        cls           ak           ref          ref/ak
        Roll1dInt     2.82467638   4.14947038   1.46900736
        Roll1dFloat   2.89442847   4.13699139   1.42929474
        Roll1dObject  112.6879144  38.81264949  0.34442602
    */
    // Tell the constructor to automatically allocate the output.
    // The data type of the output will match that of the input.
    PyArrayObject *arrays[2];
    npy_uint32 arrays_flags[2];
    arrays[0] = array;
    arrays[1] = NULL;
    arrays_flags[0] = NPY_ITER_READONLY;
    arrays_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    // No inner iteration - inner loop is handled by CopyArray code
    // Reference objects are OK.
    int iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;

    // Construct the iterator
    NpyIter *iter = NpyIter_MultiNew(
            2,              // number of arrays
            arrays,
            iter_flags,
            NPY_KEEPORDER,  // Maintain existing order for `array`
            NPY_NO_CASTING, // Both arrays will have the same dtype so casting isn't needed or allowed
            arrays_flags,
            NULL);          // We don't have to specify dtypes since it will use array's

    /* Per the documentation for NPY_ITER_REFS_OK:

        Indicates that arrays with reference types (object arrays or structured arrays
        containing an object type) may be accepted and used in the iterator. If this flag
        is enabled, the caller must be sure to check whether NpyIter_IterationNeedsAPI(iter)
        is true, in which case it may not release the GIL during iteration.

        However, `NpyIter_IterationNeedsAPI` is not documented at all. So.......
    */

    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;

    if (!PyDataType_ISOBJECT(PyArray_DESCR(array))) {
        do {
            char* src_data = dataptr[0];
            char* dst_data = dataptr[1];
            npy_intp size = *sizeptr;

            npy_intp offset = ((size - shift) % size) * itemsize;
            npy_intp first_chunk = (size * itemsize) - offset;

            memcpy(dst_data, src_data + offset, first_chunk);
            memcpy(dst_data + first_chunk, src_data, offset);
        } while (iternext(iter));
    }
    else {
        // Object arrays contain pointers to arrays.
        do {
            char* src_data = dataptr[0];
            char* dst_data = dataptr[1];
            npy_intp size = *sizeptr;

            PyObject* src_ref = NULL;
            PyObject* dst_ref = NULL;

            for (int i = 0; i < size; ++i) {
                npy_intp offset = ((i + size - shift) % size) * itemsize;

                // Update our temp PyObject* 's
                memcpy(&src_ref, src_data + offset, sizeof(src_ref));
                memcpy(&dst_ref, dst_data, sizeof(dst_ref));

                // Copy the reference
                memcpy(dst_data, &src_ref, sizeof(src_ref));

                // Claim the reference
                Py_XINCREF(src_ref);

                // Release the reference in dst
                Py_XDECREF(dst_ref);

                dst_data += itemsize;
            }
        } while (iternext(iter));
    }

    // Get the result from the iterator object array
    PyArrayObject *ret = NpyIter_GetOperandArray(iter)[1];
    if (!ret) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    Py_INCREF(ret);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject*)ret;
}

// Being clever with C for primitives, and figuring out Objects
static PyObject *
_roll_1d_d(PyArrayObject *array, int shift)
{
    /*
        Roll1d20kInt     2.91365521  4.25724612  1.46113586
        Roll1d20kFloat   3.21448036  4.40039245  1.36892809
        Roll1d20kObject  6.7969062   8.32454664  1.22475526
        Roll1d1kInt      0.33637808  1.32518703  3.93957601
        Roll1d1kFloat    0.32248451  1.24809331  3.87024272
        Roll1d1kObject   1.46907919  2.9891046   2.03467901
    */
    // Tell the constructor to automatically allocate the output.
    // The data type of the output will match that of the input.
    PyArrayObject *arrays[2];
    npy_uint32 arrays_flags[2];
    arrays[0] = array;
    arrays[1] = NULL;
    arrays_flags[0] = NPY_ITER_READONLY;
    arrays_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    // No inner iteration - inner loop is handled by CopyArray code
    // Reference objects are OK.
    int iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;

    // Construct the iterator
    NpyIter *iter = NpyIter_MultiNew(
            2,              // number of arrays
            arrays,
            iter_flags,
            NPY_KEEPORDER,  // Maintain existing order for `array`
            NPY_NO_CASTING, // Both arrays will have the same dtype so casting isn't needed or allowed
            arrays_flags,
            NULL);          // We don't have to specify dtypes since it will use array's

    /* Per the documentation for NPY_ITER_REFS_OK:

        Indicates that arrays with reference types (object arrays or structured arrays
        containing an object type) may be accepted and used in the iterator. If this flag
        is enabled, the caller must be sure to check whether NpyIter_IterationNeedsAPI(iter)
        is true, in which case it may not release the GIL during iteration.

        However, `NpyIter_IterationNeedsAPI` is not documented at all. So.......
    */

    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;

    do {
        char* src_data = dataptr[0];
        char* dst_data = dataptr[1];
        npy_intp size = *sizeptr;

        npy_intp offset = ((size - shift) % size) * itemsize;
        npy_intp first_chunk = (size * itemsize) - offset;

        memcpy(dst_data, src_data + offset, first_chunk);
        memcpy(dst_data + first_chunk, src_data, offset);

        // Increment ref counts of objects.
        if (PyDataType_ISOBJECT(PyArray_DESCR(array))) {
            dst_data = dataptr[1];
            while (size--) {
                Py_INCREF(*(PyObject**)dst_data);
                dst_data += itemsize;
            }
        }
    } while (iternext(iter));

    // Get the result from the iterator object array
    PyArrayObject *ret = NpyIter_GetOperandArray(iter)[1];
    if (!ret) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    Py_INCREF(ret);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject*)ret;
}

static PyObject *
roll_1d(PyObject *Py_UNUSED(m), PyObject *args)
{
    /* Algorithm.

        size = len(array)
        if size <= 1:
            return array.copy()

        shift = shift % size
        if shift == 0:
            return array.copy()

        post = np.empty(size, dtype=array.dtype)
        post[0:shift] = array[-shift:]
        post[shift:] = array[0:-shift]
        return post
    */
    PyArrayObject *array;
    int shift;

    if (!PyArg_ParseTuple(args, "O!i:roll_1d", &PyArray_Type, &array, &shift))
    {
        return NULL;
    }

    // Must be signed in order for modulo to work properly for negative shift values
    int size = (int)PyArray_SIZE(array);

    uint8_t is_empty = (size == 0);

    if (!is_empty) {
        shift = shift % size;
    }

    if (is_empty || (shift == 0)) {
        PyObject* copy = PyArray_Copy(array);
        if (!copy) {
            return NULL;
        }
        return copy;
    }

    // Silence UnuSEd fUnCTioN warnings.
    if (0) {
        return _roll_1d_a(array, shift);       // Basically the same
        return _roll_1d_b(array, shift, size); // Way slower
        return _roll_1d_c(array, shift);       // Faster for primitives, same for objects
    }
    return _roll_1d_d(array, shift);         // Faster for primitives & objects!
}

// -----------------------------------------------------------------------------

static PyObject *
_roll_2d_a(PyArrayObject *array, npy_uint shift, int axis)
{
    /*
    if axis == 0: # roll rows
        post[0:shift, :] = array[-shift:, :]
        post[shift:, :] = array[0:-shift, :]
        return post

    # roll columns
    post[:, 0:shift] = array[:, -shift:]
    post[:, shift:] = array[:, 0:-shift]
    */
    // Tell the constructor to automatically allocate the output.
    // The data type of the output will match that of the input.
    PyArrayObject *arrays[2];
    npy_uint32 arrays_flags[2];
    arrays[0] = array;
    arrays[1] = NULL;
    arrays_flags[0] = NPY_ITER_READONLY;
    arrays_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;

    // No inner iteration - inner loop is handled by CopyArray code
    // Reference objects are OK.
    int iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK;

    // Construct the iterator
    NpyIter *iter = NpyIter_MultiNew(
            2,              // number of arrays
            arrays,
            iter_flags,
            NPY_KEEPORDER,
            NPY_NO_CASTING, // Both arrays will have the same dtype so casting isn't needed or allowed
            arrays_flags,
            NULL);          // We don't have to specify dtypes since it will use array's

    /* Per the documentation for NPY_ITER_REFS_OK:

        Indicates that arrays with reference types (object arrays or structured arrays
        containing an object type) may be accepted and used in the iterator. If this flag
        is enabled, the caller must be sure to check whether NpyIter_IterationNeedsAPI(iter)
        is true, in which case it may not release the GIL during iteration.

        However, `NpyIter_IterationNeedsAPI` is not documented at all. So.......
    */

    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    if (!iternext) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;

    npy_uint NUM_ROWS = (npy_uint)PyArray_DIM(array, 0);
    npy_uint rowsize  = (npy_uint)PyArray_DIM(array, 1);
    npy_uint bytes_in_row = rowsize * itemsize;

    do {
        char *src_data = dataptr[0];
        char *dst_data = dataptr[1];
        npy_intp size = *sizeptr;
        npy_uint total_bytes = size * itemsize;

        if (axis == 0) {
            /*
            Shift by rows! This is the easy case.

            Imagine we have this array:
            [0 1 2]
            [3 4 5]
            [6 7 8]

            In memory, this is stored contiguously as: [0 1 2 3 4 5 6 7 8]
            Placing parentheses, we can visualize where the rows are like so:
                [(0 1 2) (3 4 5) (6 7 8)]

            Given this, all we are concerned about is two contiguous blocks of memory.

            For example, if shift = -1, we can copy from row[1] -> END to the front

            source = [(0 1 2) (3 4 5) (6 7 8)]
                               | | |   | | |
                         -----------------
                       | | |   | | |
                       V V V   V V V
            buffer = [(3 4 5) (6 7 8) (X X X)]

            Now, we fill in the missing tail bytes with row[0] from the src buffer

            source = [(0 1 2) (3 4 5) (6 7 8)]
                       | | |
                         -----------------
                                       | | |
                                       V V V
            buffer = [(3 4 5) (6 7 8) (0 1 2)]

            Now, our internal memory represents the result of a row shift.
            We can see this if we represent the final buffer as a 2D grid:

            [3 4 5]
            [6 7 8]
            [0 1 2]
            */

            // Easiest case! Merely shift the rows
            npy_intp offset = (NUM_ROWS - shift) * bytes_in_row;
            npy_intp chunksize = total_bytes - offset;

            memcpy(dst_data, src_data + offset, chunksize);
            memcpy(dst_data + chunksize, src_data, offset);
        }
        else {
            /*
            Shift by columns! This is the more difficult case.

            Let's use a slightly different array
            [0 1 2 3 4]
            [5 6 7 8 9]
            [A B C D E]

            If we shift by 2, our goal array will be:
            [3 4 0 1 2]
            [8 9 5 6 7]
            [D E A B C]

            Alternatively, we want our contiguous memory to go from:

            source = [(0 1 2 3 4) (5 6 7 8 9) (A B C D E)]
            buffer = [(3 4 0 1 2) (8 9 5 6 7) (D E A B C)]

            In order to do this as efficiently as possible, we first fill the result buffer with the source shifted.

            source = [(0 1 2 3 4) (5 6 7 8 9) (A B C D E)]
                        \ \ \ \ \   \ \ \ \ \   \ \ \
                         \ \ \  ----  \ \ \ ----  \ \ \
                          \ \ \    \ \ \ \ \   \ \ \ \ \
            buffer = [(X X 0 1 2) (3 4 5 6 7) (8 9 A B C)]

            Now, all that's left is to fix the incorrect values

            buffer = [(X X 0 1 2) (3 4 5 6 7) (8 9 A B C)]
                       ^ ^         ^ ^         ^ ^

            We can fill these by copying the values from each row

            source = [(0 1 2 3 4) (5 6 7 8 9) (A B C D E)]
                             | |         | |         | |
                        -------     -------     -------
                       | |         | |         | |
                       V V         V V         V V
            buffer = [(3 4 0 1 2) (8 9 5 6 7) (D E A B C)]

            Now, our internal memory represents the result of a row shift.
            We can see this if we represent the final buffer as a 2D grid:

            [3 4 0 1 2]
            [8 9 5 6 7]
            [D E A B C]
            */
            if (shift > rowsize / 2) {
                /* SHIFT LEFT

                This branch is optimized for cases where the offset is greater than half of the columns.

                For this, instead of shifting right and being forced to fill in a large section for each row,
                we shift left and only have to fill in small section

                Example: Shift by 4

                Inefficient
                [0 1 2 3 4]   [0 1 2 3 4]
                 \               | | | |
                  ------        -------
                         \     | | | |
                         V     V V V V
                [X X X X 0]   [1 2 3 4 0]

                Efficient
                [0 1 2 3 4]   [0 1 2 3 4]
                  / / / /      |
                  | | | |       -------
                  | | | |              |
                 / / / /               V
                [1 2 3 4 X]   [1 2 3 4 0]
                */
                npy_intp offset = (rowsize - shift) * itemsize;
                npy_intp num_bytes = total_bytes - offset;
                memcpy(dst_data, src_data + offset, num_bytes);

                num_bytes = offset; // This is how much we need to copy for each column.

                // Update the shifted portion of each row.
                for (size_t i = 0; i < NUM_ROWS; ++i) {
                    npy_intp row_offset = i * bytes_in_row;

                    // We need to fill in the rightmost values of this row since we shifted by an offset
                    npy_intp dst_offset = row_offset + bytes_in_row - num_bytes;
                    npy_intp src_offset = row_offset;

                    memcpy(dst_data + dst_offset, src_data + src_offset, num_bytes);
                }
            }
            else {
                // SHIFT RIGHT
                npy_intp offset = shift * itemsize;
                npy_intp num_bytes = total_bytes - offset;
                memcpy(dst_data+offset, src_data, num_bytes);

                num_bytes = offset; // This is how much we need to copy for each column.

                // Update the shifted portion of each row.
                for (size_t i = 0; i < NUM_ROWS; ++i) {
                    npy_intp row_offset = i * bytes_in_row;

                    // We need to fill in the leftmost values of this row since we shifted by an offset
                    npy_intp dst_offset = row_offset;
                    npy_intp src_offset = row_offset + ((rowsize - shift) * itemsize);

                    memcpy(dst_data + dst_offset, src_data + src_offset, num_bytes);
                }
            }
        }
    } while (iternext(iter));

    // Get the result from the iterator object array
    PyArrayObject *ret = NpyIter_GetOperandArray(iter)[1];
    if (!ret) {
        NpyIter_Deallocate(iter);
        return NULL;
    }
    Py_INCREF(ret);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        Py_DECREF(ret);
        return NULL;
    }

    return (PyObject*)ret;
}

static PyObject *
roll_2d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    /* Algorithm.

        size = array.shape[axis]

        if shift != 0:
            shift = shift % size

        if size <= 1 or shift == 0:
            return array.copy()

        if shift < 0:
            shift = size + shift

        if axis == 0:
            post[0:shift, :] = array[-shift:, :]
            post[shift:, :] = array[0:-shift, :]
            return post

        post[:, 0:shift] = array[:, -shift:]
        post[:, shift:] = array[:, 0:-shift]
        return post
    */
    PyArrayObject *array;
    int shift;
    int axis; // npy_intp

    static char *kwlist[] = {"array", "shift", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!ii:roll_1d",
                                     kwlist,
                                     &PyArray_Type, &array,
                                     &shift, &axis))
    {
        return NULL;
    }

    if (axis != 0 && axis != 1) {
        PyErr_SetString(PyExc_ValueError, "Axis must be 0 or 1");
        return NULL;
    }

    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2D");
        return NULL;
    }

    // Must be signed in order for modulo to work properly for negative shift values
    int size = (int)PyArray_DIM(array, axis);

    uint8_t is_empty = (size == 0);

    if (!is_empty) {
        shift = shift % size;
        if (shift < 0) {
            shift = size + shift;
        }
    }

    if (is_empty || (shift == 0)) {
        PyObject* copy = PyArray_Copy(array);
        if (!copy) {
            return NULL;
        }
        return copy;
    }

    return _roll_2d_a(array, (npy_uint)shift, axis);
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
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"roll_1d", roll_1d, METH_VARARGS, NULL},
    {"roll_2d", (PyCFunction)roll_2d, METH_VARARGS | METH_KEYWORDS, NULL},
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
