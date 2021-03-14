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

// Placeholder of not implemented functions.
# define AK_NOT_IMPLEMENTED\
    do {\
        PyErr_SetNone(PyExc_NotImplementedError);\
        return NULL;\
    } while (0)

// To simplify lines merely checking for `!value`
# define AK_CHECK_NOT(obj) \
    if (!obj) { \
        return NULL; \
    }

// To simplify lines going to a label failure on `!value`
# define AK_GOTO_ON_NOT(obj, label) \
    if (!obj) { \
        goto label; \
    }


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
    AK_CHECK_NOT(iterator)

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
// type resolution

static PyObject *
resolve_dtype(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArray_Descr *d1, *d2;
    AK_CHECK_NOT(PyArg_ParseTuple(args, "O!O!:resolve_dtype",
                          &PyArrayDescr_Type, &d1, &PyArrayDescr_Type, &d2))
    return (PyObject *)AK_ResolveDTypes(d1, d2);
}

static PyObject *
resolve_dtype_iter(PyObject *Py_UNUSED(m), PyObject *arg)
{
    return (PyObject *)AK_ResolveDTypeIter(arg);
}

//------------------------------------------------------------------------------
// utils

static int
is_nan(PyObject *a)
{
    double v = PyFloat_AsDouble(a);

    // Need to disambiguate, since v could be -1 and no failure happened
    if (v == -1 && PyErr_Occurred()) {
        return -1;
    }

    return isnan(v);
}

static int
is_nanj(PyObject *a)
{
    return isnan(((PyComplexObject*)a)->cval.real);
}

static int
is_nat(PyObject *a)
{
    // NaT - Datetime
    if (PyArray_IsScalar(a, Datetime)) { // Cannot fail
        return PyArrayScalar_VAL(a, Datetime) == NPY_DATETIME_NAT;
    }

    // NaT - Timedelta
    if (PyArray_IsScalar(a, Timedelta)) { // Cannot fail
        return PyArrayScalar_VAL(a, Timedelta) == NPY_DATETIME_NAT;
    }

    Py_UNREACHABLE();
}

//------------------------------------------------------------------------------
// isin

# define AK_PPRINT(obj) \
    printf(""#obj""); printf(": "); PyObject_Print(obj, stdout, 0); printf("\n"); fflush(stdout);

static PyArrayObject *
AK_concat_arrays(PyArrayObject *arr1, PyArrayObject *arr2)
{
    PyObject *container = PyTuple_Pack(2, arr1, arr2);
    AK_CHECK_NOT(container)

    PyArrayObject *array = (PyArrayObject*)PyArray_Concatenate(container, 0);
    Py_DECREF(container);
    return array;
}

static PyArrayObject*
AK_compare_two_slices_from_array(PyArrayObject *arr, Py_ssize_t l1, Py_ssize_t l2, Py_ssize_t r1, Py_ssize_t r2, int EQ)
{
    PyObject* left_slice = NULL;
    PyObject* right_slice = NULL;
    PyObject* comparison = NULL;

    left_slice = PySequence_GetSlice((PyObject*)arr, l1, l2);
    AK_GOTO_ON_NOT(left_slice, failure)

    right_slice = PySequence_GetSlice((PyObject*)arr, r1, r2);
    AK_GOTO_ON_NOT(right_slice, failure)

    comparison = PyObject_RichCompare(left_slice, right_slice, EQ);
    AK_GOTO_ON_NOT(comparison, failure)

    Py_DECREF(left_slice);
    Py_DECREF(right_slice);

    return (PyArrayObject*)comparison;

failure:
    Py_XDECREF(left_slice);
    Py_XDECREF(right_slice);
    return NULL;
}

static int
AK_build_unique_arr_mask(PyArrayObject *sar, npy_bool* mask)
{
    /* Algorithm (assumes `sar` is sorted & mask is initialized to [1, 0, ...] & len(mask) == len(sar)

        // cfmM = [Complex, Float, Datetime, Timedelta]
        if sar.dtype.kind in "cfmM" and np.isnan(sar[-1]):
            if sar.dtype.kind == "c":  # for complex all NaNs are considered equivalent
                aux_firstnan = np.searchsorted(np.isnan(sar), True, side='left')
            else:
                aux_firstnan = np.searchsorted(sar, sar[-1], side='left')

            mask[1:aux_firstnan] = (sar[1:aux_firstnan] != sar[:aux_firstnan - 1])
            mask[aux_firstnan] = True
            mask[aux_firstnan + 1:] = False
        else:
            mask[1:] = sar[1:] != sar[:-1]
    */
    // 0. Deallocate on failure
    PyArrayObject* comparison = NULL;
    PyObject* last_element = NULL;

    // 1. Determine if last element contains NaNs/NaTs
    size_t size = (size_t)PyArray_SIZE(sar);
    PyArray_Descr* dtype = PyArray_DESCR(sar);

    int is_float = PyDataType_ISFLOAT(dtype);
    int is_complex = PyDataType_ISCOMPLEX(dtype);
    int is_dt = PyDataType_ISDATETIME(dtype);

    int contains_nan = 0;

    if (is_float | is_complex | is_dt) {
        last_element = PyObject_GetItem((PyObject*)sar, PyLong_FromLong(-1));
        AK_GOTO_ON_NOT(last_element, failure)
        if (is_float) {
            contains_nan = is_nan(last_element);
        }
        else if (is_complex) {
            contains_nan = is_nanj(last_element);
        }
        else {
            // This will always be false as long as numpy < 1.18. NaT sort to the front
            contains_nan = is_nat(last_element);
        }
    }

    // 2. Populate mask
    if (contains_nan) {
        // 3. Discover the location of the first NaN element
        size_t firstnan = 0;
        if (is_complex) {
            // TODO: I don't understand the necessity of this branch.
            // aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        }

        // This gives back an array of 1-element since `last_element` is a single element
        PyObject* firstnan_obj = PyArray_SearchSorted(sar, last_element, NPY_SEARCHLEFT, NULL);
        AK_GOTO_ON_NOT(firstnan_obj, failure)

        firstnan = *(size_t*)PyArray_DATA((PyArrayObject*)firstnan_obj);
        Py_DECREF(firstnan_obj);

        // 4. Build mask in such a way to only include 1 NaN value
        comparison = AK_compare_two_slices_from_array(sar, 1, firstnan, 0, firstnan - 1, Py_NE);
        AK_GOTO_ON_NOT(comparison, failure)
        npy_bool* comparison_arr = (npy_bool*)PyArray_DATA(comparison);

        for (size_t i = 1; i < firstnan; ++i) {
            mask[i] = comparison_arr[i-1];
        }
        mask[firstnan] = 1;
        for (size_t i = firstnan + 1; i < size; ++i) {
            mask[i] = 0;
        }
    }
    else {
        // 3. Build mask through a simple [1:] != [:-1] slice comparison
        comparison = AK_compare_two_slices_from_array(sar, 1, size, 0, size - 1, Py_NE);
        AK_GOTO_ON_NOT(comparison, failure)
        npy_bool* comparison_arr = (npy_bool*)PyArray_DATA(comparison);

        for (size_t i = 1; i < (size_t)size; ++i) {
            mask[i] = comparison_arr[i-1];
        }
    }

    Py_DECREF(comparison);
    Py_XDECREF(last_element); // Only populated when sar contains NaNs/NaTs

    return 1;

failure:
    Py_XDECREF(comparison);
    Py_XDECREF(last_element);
    return 0;
}

static PyArrayObject*
AK_get_unique_arr(PyArrayObject *original_arr)
{
    /* Algorithm

        sar = copy(original_arr)
        sar.sort()

        mask = np.empty(sar.shape, dtype=np.bool_)
        mask[0] = True

        build_mask(...)

        return sar[mask]
    */

    // 1. Initialize
    PyObject* mask = NULL; // Deallocate on failure

    size_t size = PyArray_SIZE(original_arr);
    PyArray_Descr* dtype = PyArray_DESCR(original_arr);

    npy_bool mask_arr[size];

    // 2. Get a copy of the original arr since sorting is in-place
    PyArrayObject* sar = (PyArrayObject*)PyArray_FromArray(
            original_arr,
            dtype,
            NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSURECOPY);
    AK_CHECK_NOT(sar)
    if (PyArray_Sort(sar, 0, NPY_QUICKSORT) == -1) { // In-place
        goto failure;
    }

    // 3. Build mask
    memset(mask_arr, 0, sizeof(mask_arr));
    mask_arr[0] = 1;
    AK_GOTO_ON_NOT(AK_build_unique_arr_mask(sar, mask_arr), failure)

    mask = PyArray_NewFromDescr(
                &PyArray_Type,                         // class (subtype)
                PyArray_DescrFromType(NPY_BOOL),       // dtype (descr)
                PyArray_NDIM(sar),                     // ndim (nd)
                PyArray_DIMS(sar),                     // dims
                NULL,                                  // strides
                mask_arr,                              // data
                NPY_ARRAY_DEFAULT | NPY_ARRAY_OWNDATA, // flags
                NULL);                                 // sublclass (obj)
    AK_GOTO_ON_NOT(mask, failure)

    // 4. Filter sar
    PyArrayObject *filtered_arr = (PyArrayObject*)PyObject_GetItem((PyObject*)sar, (PyObject*)mask);
    AK_GOTO_ON_NOT(filtered_arr, failure)

    Py_DECREF(sar);
    Py_DECREF(mask);
    return filtered_arr;

failure:
    Py_DECREF(sar); // Cannot be NULL
    Py_XDECREF(mask);
    return NULL;
}

static PyObject*
AK_get_unique_arr_w_inverse(PyArrayObject *original_arr)
{
    /* Algorithm

        ordered_idx = original_arr.argsort(kind='quicksort')
        sar = original_arr[ordered_idx]

        mask = np.empty(sar.shape, dtype=np.bool_)
        mask[0] = True

        AK_build_unique_arr_mask(sar, mask)

        ret = sar[mask]
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[ordered_idx] = imask
        return ret, inv_idx
    */

    // 1. Initialize
    PyObject *ordered_idx = NULL;
    PyArrayObject *sar = NULL;
    PyArrayObject* mask = NULL;
    PyObject *filtered_arr = NULL;
    PyObject* cumsum = NULL;
    PyObject* imask = NULL;
    PyObject* inv_idx = NULL;

    size_t size = PyArray_SIZE(original_arr);

    npy_bool mask_arr[size];

    // 2. Get sorted indices & sort array
    ordered_idx = PyArray_ArgSort(original_arr, 0, NPY_QUICKSORT);
    AK_GOTO_ON_NOT(ordered_idx, failure)

    sar = (PyArrayObject*)PyObject_GetItem((PyObject*)original_arr, ordered_idx);
    AK_GOTO_ON_NOT(sar, failure)

    // 3. Build mask
    memset(mask_arr, 0, sizeof(mask_arr));
    mask_arr[0] = 1;
    AK_GOTO_ON_NOT(AK_build_unique_arr_mask(sar, mask_arr), failure)

    mask = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,                         // subtype
                PyArray_DescrFromType(NPY_BOOL),       // dtype
                PyArray_NDIM(sar),                     // nd
                PyArray_DIMS(sar),                     // dims
                NULL,                                  // strides
                mask_arr,                              // data
                NPY_ARRAY_DEFAULT | NPY_ARRAY_OWNDATA, // flags
                NULL);                                 // sublclass (obj)
    AK_GOTO_ON_NOT(mask, failure)

    // 4. Filter arr
    filtered_arr = PyObject_GetItem((PyObject*)sar, (PyObject*)mask);
    AK_GOTO_ON_NOT(filtered_arr, failure)

    // 5. Determine the inverse index
    cumsum = PyArray_CumSum(
            mask,    // array
            0,       // axis
            NPY_INT, // dtype
            NULL);   // out-array
    AK_GOTO_ON_NOT(cumsum, failure)

    imask = PyNumber_Subtract(cumsum, PyLong_FromLong(1));
    AK_GOTO_ON_NOT(imask, failure)

    inv_idx = PyArray_Empty(
            PyArray_NDIM(mask),             // nd
            PyArray_DIMS(mask),             // dims
            PyArray_DescrFromType(NPY_INT), // dtype
            0);                             // is_f_order

    if (PyObject_SetItem(inv_idx, ordered_idx, imask)) {
        goto failure;
    }

    // 6. Pack it up in a tuple and return
    PyObject* ret = PyTuple_Pack(2, filtered_arr, inv_idx);
    AK_GOTO_ON_NOT(ret, failure)

    Py_DECREF(ordered_idx);
    Py_DECREF(sar);
    Py_DECREF(mask);
    Py_DECREF(filtered_arr);
    Py_DECREF(cumsum);
    Py_DECREF(imask);
    Py_DECREF(inv_idx);
    return ret;

failure:
    Py_XDECREF(ordered_idx);
    Py_XDECREF(sar);
    Py_XDECREF(mask);
    Py_XDECREF(filtered_arr);
    Py_XDECREF(cumsum);
    Py_XDECREF(imask);
    Py_XDECREF(inv_idx);
    return NULL;
}

static PyObject *
AK_isin_array_dtype(PyArrayObject *array, PyArrayObject *other, int assume_unique)
{
    /* Algorithm:

        array = np.ravel(array)

        if not assume_unique:
            array, rev_idx = np.unique(array, return_inverse=True)
            other = np.unique(other)

        concatenated = np.concatenate((array, other))

        ordered_idx = concatenated.argsort(kind='mergesort')
        sorted_arr = concatenated[ordered_idx]

        flag = np.concatenate(((sorted_arr[1:] == sorted_arr[:-1]), [False]))

        ret = np.empty(concatenated.shape, dtype=bool)
        ret[ordered_idx] = flag

        if assume_unique:
            return ret[:len(array)]
        else:
            return ret[rev_idx]
    */
    // 0. Deallocate on failure
    PyArrayObject* flattened_array = NULL;
    PyObject *reverse_idx = NULL;
    PyArrayObject* concatenated = NULL;
    PyArrayObject *ordered_idx = NULL;
    PyArrayObject* sorted_arr = NULL;
    PyArrayObject* comparison = NULL;
    PyArrayObject* ret = NULL;

    // 1. Capture original array shape for return value
    int array_ndim = PyArray_NDIM(array);
    npy_intp* array_dims = PyArray_DIMS(array);
    size_t array_size = PyArray_SIZE(array);

    // 2. Ravel the array as we want to operate on 1D arrays only. (other is guaranteed to be 1D)
    flattened_array = (PyArrayObject*)PyArray_Flatten(array, NPY_CORDER);
    AK_GOTO_ON_NOT(flattened_array, failure)
    Py_INCREF(flattened_array);

    if (!assume_unique) {
        PyObject* arr_and_rev_idx = AK_get_unique_arr_w_inverse(flattened_array);
        PyArrayObject *raveled_array_unique = (PyArrayObject*)PyTuple_GET_ITEM(arr_and_rev_idx, 0);
        AK_GOTO_ON_NOT(raveled_array_unique, failure)
        Py_INCREF(raveled_array_unique);

        reverse_idx = PyTuple_GET_ITEM(arr_and_rev_idx, 1);
        AK_GOTO_ON_NOT(reverse_idx, failure)
        Py_INCREF(reverse_idx);

        PyArrayObject *other_unique = AK_get_unique_arr(other);
        AK_GOTO_ON_NOT(other_unique, failure)
        Py_INCREF(other_unique);

        // 3. Concatenate
        concatenated = AK_concat_arrays(raveled_array_unique, other_unique);
        Py_DECREF(arr_and_rev_idx);
        Py_DECREF(raveled_array_unique);
        Py_DECREF(other_unique);
    }
    else {
        // 3. Concatenate
        concatenated = AK_concat_arrays(flattened_array, other);
    }
    AK_GOTO_ON_NOT(concatenated, failure)

    size_t concatenated_size = PyArray_SIZE(concatenated);

    // 4: Sort
    ordered_idx = (PyArrayObject*)PyArray_ArgSort(concatenated, 0, NPY_MERGESORT);
    AK_GOTO_ON_NOT(ordered_idx, failure)
    npy_intp* ordered_idx_arr = (npy_intp*)PyArray_DATA(ordered_idx);

    // 5. Find duplicates
    sorted_arr = (PyArrayObject*)PyObject_GetItem((PyObject*)concatenated, (PyObject*)ordered_idx);
    AK_GOTO_ON_NOT(sorted_arr, failure)

    comparison = AK_compare_two_slices_from_array(sorted_arr, 1, concatenated_size, 0, concatenated_size - 1, Py_EQ);
    AK_GOTO_ON_NOT(comparison, failure)
    npy_bool* comparison_arr = (npy_bool*)PyArray_DATA(comparison);

    if (!assume_unique) {
        // 6: Construct empty array
        PyObject* tmp = PyArray_Empty(
                PyArray_NDIM(concatenated),      // nd
                PyArray_DIMS(concatenated),      // dims
                PyArray_DescrFromType(NPY_BOOL), // dtype
                0);                              // is_f_order

        Py_INCREF(tmp);

        npy_intp dims[1] = {1};

        PyArrayObject *false = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,                         // class (subtype)
                PyArray_DescrFromType(NPY_BOOL),       // dtype (descr)
                1,                                     // ndim (nd)
                dims,                                  // dims
                NULL,                                  // strides
                "\0",                                  // data
                NPY_ARRAY_DEFAULT | NPY_ARRAY_OWNDATA, // flags
                NULL);                                 // sublclass (obj)
        Py_INCREF(false);

        PyArrayObject* xyz = AK_concat_arrays(comparison, false);
        Py_INCREF(xyz);

        // TODO: Comparison is missing a trailing False value...
        if (PyObject_SetItem(tmp, (PyObject*)ordered_idx, (PyObject*)xyz)) {
            goto failure;
        }

        Py_INCREF(ordered_idx);
        Py_INCREF(xyz);
        Py_INCREF(tmp);
        Py_INCREF(reverse_idx);

        ret = (PyArrayObject*)PyObject_GetItem(tmp, reverse_idx);

        if (array_ndim == 2) {
            PyObject* shape = PyTuple_Pack(2, PyLong_FromLong(array_dims[0]), PyLong_FromLong(array_dims[1]));
            ret = (PyArrayObject*)PyArray_Reshape(ret, shape);
        }

        Py_DECREF(tmp);
        Py_DECREF(reverse_idx);
    }
    else {
        // 6: Construct empty array
        ret = (PyArrayObject*)PyArray_Empty(
                array_ndim,                      // nd
                array_dims,                      // dims
                PyArray_DescrFromType(NPY_BOOL), // dtype
                0);                              // is_f_order

        AK_GOTO_ON_NOT(ret, failure)

        size_t stride = 0;
        if (array_ndim == 2) {
            stride = (size_t)array_dims[1];
        }

        // 7: Assign into duplicates array
        for (size_t i = 0; i < (size_t)PyArray_SIZE(ordered_idx); ++i) {
            size_t idx_0 = (size_t)ordered_idx_arr[i];
            if (idx_0 >= array_size) { continue; }

            // We are guaranteed that flag_ar[i] is always a valid index
            if (array_ndim == 1) {
                *(npy_bool *) PyArray_GETPTR1(ret, idx_0) = comparison_arr[i];
            }
            else {
                size_t idx_1 = idx_0 / stride;
                idx_0 = idx_0 - (stride * idx_1);

                *(npy_bool *) PyArray_GETPTR2(ret, idx_1, idx_0) = comparison_arr[i];
            }
        }
    }

    // 8. Cleanup & Return!
    Py_DECREF(flattened_array);
    Py_DECREF(concatenated);
    Py_DECREF(ordered_idx);
    Py_DECREF(sorted_arr);
    Py_DECREF(comparison);

    return (PyObject*)ret;

failure:
    Py_XDECREF(flattened_array);
    Py_XDECREF(reverse_idx);
    Py_XDECREF(concatenated);
    Py_XDECREF(ordered_idx);
    Py_XDECREF(sorted_arr);
    Py_XDECREF(comparison);
    return NULL;
}

/*
static PyObject *
AK_isin_array_dtype_use_np(PyArrayObject *array, PyArrayObject *other, int assume_unique)
{
    PyObject* numpy = NULL;
    PyObject* func = NULL;
    PyObject* args = NULL;
    PyObject* kwarg = NULL;

    numpy = PyImport_ImportModule("numpy");
    AK_GOTO_ON_NOT(numpy, failure)

    if (PyArray_NDIM(array) == 1) {
        func = PyObject_GetAttrString(numpy, "in1d");
    }
    else {
        func = PyObject_GetAttrString(numpy, "isin");
    }
    AK_GOTO_ON_NOT(func, failure)

    args = PyTuple_Pack(2, (PyObject*)array, (PyObject*)other);
    AK_GOTO_ON_NOT(args, failure)

    kwarg = PyDict_New();
    AK_GOTO_ON_NOT(kwarg, failure);
    if (PyDict_SetItemString(kwarg, "assume_unique", PyLong_FromLong((long)assume_unique)) == -1) {
        goto failure;
    }

    PyObject* result = PyObject_Call(func, args, kwarg);
    AK_GOTO_ON_NOT(result, failure)

    Py_DECREF(numpy);
    Py_DECREF(func);
    Py_DECREF(args);
    Py_DECREF(kwarg);

    return result;

failure:
    Py_XDECREF(numpy);
    Py_XDECREF(func);
    Py_XDECREF(args);
    Py_XDECREF(kwarg);
    return NULL;
}
*/

static PyObject *
AK_isin_array_object(PyArrayObject *array, PyArrayObject *other)
{
    /*  Algorithm:

        for loc, element in loc_iter(array):
            result[loc] = element in set(other)
    */

    // 0. Deallocate on failure
    PyObject* compare_elements = NULL;
    PyArrayObject* result = NULL;
    NpyIter *iter = NULL;

    // 1. Capture original array shape for return value
    int array_ndim = PyArray_NDIM(array);
    npy_intp* array_dims = PyArray_DIMS(array);

    compare_elements = PyFrozenSet_New((PyObject*)other);
    AK_CHECK_NOT(compare_elements)

    // 2: Construct empty array
    result = (PyArrayObject*)PyArray_Empty(
            array_ndim,                      // nd
            array_dims,                      // dims
            PyArray_DescrFromType(NPY_BOOL), // dtype
            0);                              // is_f_order
    AK_GOTO_ON_NOT(result, failure)

    // 3. Set up iteration
    // https://numpy.org/doc/stable/reference/c-api/iterator.html?highlight=npyiter_multinew#simple-iteration-example
    iter = NpyIter_New(array,
                       NPY_ITER_READONLY | NPY_ITER_REFS_OK | NPY_ITER_EXTERNAL_LOOP,
                       NPY_KEEPORDER,
                       NPY_NO_CASTING,
                       NULL);
    AK_GOTO_ON_NOT(iter, failure)

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
    AK_GOTO_ON_NOT(iternext, failure)

    char** dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *sizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // 4. Iterate!
    int i = 0;
    do {
        int j = 0;
        char* data = *dataptr;
        npy_intp size = *sizeptr;
        npy_intp stride = *strideptr;

        while (size--) {
            PyObject* obj;
            memcpy(&obj, data, sizeof(obj));
            AK_GOTO_ON_NOT(obj, failure)
            Py_INCREF(obj);

            // 5. Assign into result whether or not the element exists in the set
            int found = PySequence_Contains(compare_elements, obj);
            Py_DECREF(obj);

            if (found == -1) {
                goto failure;
            }

            if (array_ndim == 1){
                *(npy_bool *) PyArray_GETPTR1(result, j) = (npy_bool)found;
            }
            else {
                *(npy_bool *) PyArray_GETPTR2(result, i, j) = (npy_bool)found;
            }

            data += stride;
            ++j;
        }

        ++i;
        // Increment the iterator to the next inner loop
    } while(iternext(iter));

    Py_DECREF(compare_elements);
    NpyIter_Deallocate(iter);

    return (PyObject*)result;

failure:
    Py_DECREF(compare_elements);
    Py_XDECREF(result);
    if (iter != NULL) {
        NpyIter_Deallocate(iter);
    }
    return NULL;
}

static PyObject *
isin_array(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    int array_is_unique, other_is_unique;
    PyArrayObject *array, *other;

    static char *kwlist[] = {"array", "array_is_unique", "other", "other_is_unique", NULL};

    AK_CHECK_NOT(PyArg_ParseTupleAndKeywords(args, kwargs, "O!iO!i:isin_array",
                                     kwlist,
                                     &PyArray_Type, &array, &array_is_unique,
                                     &PyArray_Type, &other, &other_is_unique))

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
    return AK_isin_array_dtype(array, other, array_is_unique && other_is_unique);
    //return AK_isin_array_dtype_use_np(array, other, array_is_unique && other_is_unique);
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
    AK_CHECK_NOT(parsed)

    ArrayGOObject *self = (ArrayGOObject *)cls->tp_alloc(cls, 0);
    AK_CHECK_NOT(self)

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
        AK_CHECK_NOT(self->list)

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
        AK_CHECK_NOT(self->list)

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
