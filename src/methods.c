# include "Python.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

# include "methods.h"
# include "utilities.h"

PyObject *
count_iteration(PyObject *Py_UNUSED(m), PyObject *iterable)
{
    PyObject *iter = PyObject_GetIter(iterable);
    if (iter == NULL) return NULL;

    int count = 0;
    PyObject *v;

    while ((v = PyIter_Next(iter))) {
        count++;
        Py_DECREF(v);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) {
        return NULL;
    }
    PyObject* result = PyLong_FromLong(count);
    if (result == NULL) return NULL;
    return result;
}

PyObject *
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

PyObject *
slice_to_ascending_slice(PyObject *Py_UNUSED(m), PyObject *args) {

    PyObject* slice;
    PyObject* size;
    if (!PyArg_ParseTuple(args,
            "O!O!:slice_to_ascending_slice",
            &PySlice_Type, &slice,
            &PyLong_Type, &size)) {
        return NULL;
    }
    // will delegate NULL on eroror
    return AK_slice_to_ascending_slice(slice, PyLong_AsSsize_t(size));
}

PyObject *
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

PyObject *
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

PyObject *
shape_filter(PyObject *Py_UNUSED(m), PyObject *a) {
    AK_CHECK_NUMPY_ARRAY_1D_2D(a);
    PyArrayObject *array = (PyArrayObject *)a;
    npy_intp rows = PyArray_DIM(array, 0);
    // If 1D array, set size for axis 1 at 1, else use 2D array to get the size of axis 1
    npy_intp cols = PyArray_NDIM(array) == 1 ? 1 : PyArray_DIM(array, 1);
    return AK_build_pair_ssize_t(rows, cols);
}

PyObject *
name_filter(PyObject *Py_UNUSED(m), PyObject *n) {
    if (AK_UNLIKELY(PyObject_Hash(n) == -1)) {
        return PyErr_Format(PyExc_TypeError,
                "unhashable name (type '%s')",
                Py_TYPE(n)->tp_name);
    }
    Py_INCREF(n);
    return n;
}

PyObject *
mloc(PyObject *Py_UNUSED(m), PyObject *a)
{
    AK_CHECK_NUMPY_ARRAY(a);
    return PyLong_FromVoidPtr(PyArray_DATA((PyArrayObject *)a));
}

PyObject *
immutable_filter(PyObject *Py_UNUSED(m), PyObject *a) {
    AK_CHECK_NUMPY_ARRAY(a);
    return (PyObject *)AK_immutable_filter((PyArrayObject *)a);
}

PyObject *
resolve_dtype(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyArray_Descr *d1, *d2;
    if (!PyArg_ParseTuple(args,
            "O!O!:resolve_dtype",
            &PyArrayDescr_Type, &d1,
            &PyArrayDescr_Type, &d2)) {
        return NULL;
    }
    return (PyObject *)AK_resolve_dtype(d1, d2);
}

PyObject *
resolve_dtype_iter(PyObject *Py_UNUSED(m), PyObject *arg) {
    PyObject *iterator = PyObject_GetIter(arg);
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
        Py_SETREF(resolved, AK_resolve_dtype(resolved, dtype));
        Py_DECREF(dtype);
        if (!resolved || PyDataType_ISOBJECT(resolved)) {
            break;
        }
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (!resolved) {
        // this could happen if this function gets an empty tuple
        PyErr_SetString(PyExc_ValueError, "iterable passed to resolve dtypes is empty");
    }
    return (PyObject *)resolved;
}

PyObject *
nonzero_1d(PyObject *Py_UNUSED(m), PyObject *a) {
    AK_CHECK_NUMPY_ARRAY(a);
    PyArrayObject* array = (PyArrayObject*)a;
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    return AK_nonzero_1d(array);
}

static char *first_true_1d_kwarg_names[] = {
    "array",
    "forward",
    NULL
};

PyObject *
first_true_1d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *array = NULL;
    int forward = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O!|$p:first_true_1d",
            first_true_1d_kwarg_names,
            &PyArray_Type, &array,
            &forward
            )) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Array must be 1-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(array)) {
        PyErr_SetString(PyExc_ValueError, "Array must be contiguous");
        return NULL;
    }

    npy_intp lookahead = sizeof(npy_uint64);
    npy_intp size = PyArray_SIZE(array);
    lldiv_t size_div = lldiv((long long)size, lookahead); // quot, rem

    npy_bool *array_buffer = (npy_bool*)PyArray_DATA(array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    Py_ssize_t position = -1;
    npy_bool *p;
    npy_bool *p_end;
    npy_bool *p_end_roll;

    if (forward) {
        p = array_buffer;
        p_end = p + size;
        p_end_roll = p_end - size_div.rem;

        while (p < p_end_roll) {
            if (*(npy_uint64*)p != 0) {
                break; // found a true within lookahead
            }
            p += lookahead;
        }
        while (p < p_end) {
            if (*p) break;
            p++;
        }
    }
    else {
        p = array_buffer + size - 1;
        p_end = array_buffer - 1;
        p_end_roll = p_end + size_div.rem;

        while (p > p_end_roll) {
            if (*(npy_uint64*)(p - lookahead + 1) != 0) {
                break; // found a true within lookahead
            }
            p -= lookahead;
        }
        while (p > p_end) {
            if (*p) break;
            p--;
        }
    }
    if (p != p_end) { // else, return -1
        position = p - array_buffer;
    }
    NPY_END_THREADS;

    PyObject* post = PyLong_FromSsize_t(position);
    return post;
}

static char *first_true_2d_kwarg_names[] = {
    "array",
    "forward",
    "axis",
    NULL
};

PyObject *
first_true_2d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyArrayObject *array = NULL;
    int forward = 1;
    int axis = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O!|$pi:first_true_2d",
            first_true_2d_kwarg_names,
            &PyArray_Type,
            &array,
            &forward,
            &axis
            )) {
        return NULL;
    }
    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2-dimensional");
        return NULL;
    }
    if (PyArray_TYPE(array) != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type bool");
        return NULL;
    }
    if (axis < 0 || axis > 1) {
        PyErr_SetString(PyExc_ValueError, "Axis must be 0 or 1");
        return NULL;
    }

    // NOTE: we copy the entire array into contiguous memory when necessary.
    // axis = 0 returns the pos per col
    // axis = 1 returns the pos per row (as contiguous bytes)
    // if c contiguous:
    //      axis == 0: transpose, copy to C
    //      axis == 1: keep
    // if f contiguous:
    //      axis == 0: transpose, keep
    //      axis == 1: copy to C
    // else
    //     axis == 0: transpose, copy to C
    //     axis == 1: copy to C

    bool transpose = !axis; // if 1, false
    bool corder = true;
    if ((PyArray_IS_C_CONTIGUOUS(array) && axis == 1) ||
        (PyArray_IS_F_CONTIGUOUS(array) && axis == 0)) {
        corder = false;
    }
    // create pointer to "indicator" array; if newly allocated, it will need to be decrefed before function termination
    PyArrayObject *array_ind = NULL;
    bool decref_array_ind = false;

    if (transpose && !corder) {
        array_ind = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (!transpose && corder) {
        array_ind = (PyArrayObject *)PyArray_NewCopy(array, NPY_CORDER);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else if (transpose && corder) {
        PyArrayObject *tmp = (PyArrayObject *)PyArray_Transpose(array, NULL);
        if (tmp == NULL) return NULL;

        array_ind = (PyArrayObject *)PyArray_NewCopy(tmp, NPY_CORDER);
        Py_DECREF((PyObject*)tmp);
        if (array_ind == NULL) return NULL;
        decref_array_ind = true;
    }
    else {
        array_ind = array; // can use array, no decref needed
    }

    npy_intp lookahead = sizeof(npy_uint64);

    // buffer of indicators
    npy_bool *buffer_ind = (npy_bool*)PyArray_DATA(array_ind);

    npy_intp count_row = PyArray_DIM(array_ind, 0);
    npy_intp count_col = PyArray_DIM(array_ind, 1);

    lldiv_t div_col = lldiv((long long)count_col, lookahead); // quot, rem

    npy_intp dims_post = {count_row};
    PyArrayObject *array_pos = (PyArrayObject*)PyArray_EMPTY(
            1,         // ndim
            &dims_post,// shape
            NPY_INT64, // dtype
            0          // fortran
            );
    if (array_pos == NULL) {
        return NULL;
    }
    npy_int64 *buffer_pos = (npy_int64*)PyArray_DATA(array_pos);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    npy_intp position;
    npy_bool *p;
    npy_bool *p_start;
    npy_bool *p_end;

    // iterate one row at a time; short-circult when found
    // for axis 1 rows are rows; for axis 0, rows are (post transpose) columns
    for (npy_intp r = 0; r < count_row; r++) {
        position = -1; // update for each row

        if (forward) {
            // get start of each row
            p_start = buffer_ind + (count_col * r);
            p = p_start;
            p_end = p + count_col; // end of each row

            // scan each row from the front and terminate when True
            // remove from the end the remainder
            while (p < p_end - div_col.rem) {
                if (*(npy_uint64*)p != 0) {
                    break; // found a true
                }
                p += lookahead;
            }
            while (p < p_end) {
                if (*p) {break;}
                p++;
            }
            if (p != p_end) {
                position = p - p_start;
            }
        }
        else { // reverse
            // start at the next row, then subtract one for last elem in previous row
            p_start = buffer_ind + (count_col * (r + 1)) - 1;
            p = p_start;
            // end is 1 less than start of each row
            p_end = buffer_ind + (count_col * r) - 1;

            while (p > p_end + div_col.rem) {
                // must go to start of lookahead
                if (*(npy_uint64*)(p - lookahead + 1) != 0) {
                    break; // found a true
                }
                p -= lookahead;
            }
            while (p > p_end) {
                if (*p) {break;}
                p--;
            }
            if (p != p_end) {
                position = p - (p_end + 1);
            }
        }
        *buffer_pos++ = position;
    }

    NPY_END_THREADS;

    if (decref_array_ind) {
        Py_DECREF(array_ind); // created in this function
    }
    return (PyObject *)array_pos;
}

PyObject *
dtype_from_element(PyObject *Py_UNUSED(m), PyObject *arg)
{
    // -------------------------------------------------------------------------
    // 1. Handle fast, exact type checks first.
    if (arg == Py_None) {
        return (PyObject*)PyArray_DescrFromType(NPY_OBJECT);
    }
    if (PyFloat_CheckExact(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_FLOAT64);
    }
    if (PyLong_CheckExact(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_INT64);
    }
    if (PyBool_Check(arg)) {
        return (PyObject*)PyArray_DescrFromType(NPY_BOOL);
    }

    PyObject* dtype = NULL;
    // String
    if (PyUnicode_CheckExact(arg)) {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_UNICODE);
        if (descr == NULL) return NULL;
        dtype = (PyObject*)PyArray_DescrFromObject(arg, descr);
        Py_DECREF(descr);
        return dtype;
    }
    // Bytes
    if (PyBytes_CheckExact(arg)) {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_STRING);
        if (descr == NULL) return NULL;
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

static char *isna_element_kwarg_names[] = {
    "element",
    "include_none",
    NULL
};

PyObject *
isna_element(PyObject *m, PyObject *args, PyObject *kwargs)
{
    PyObject *element;
    int include_none = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|p:isna_element", isna_element_kwarg_names,
            &element,
            &include_none)) {
        return NULL;
    }

    // None
    if (include_none && element == Py_None) {
        Py_RETURN_TRUE;
    }

    // NaN
    if (PyFloat_Check(element)) {
        return PyBool_FromLong(isnan(PyFloat_AS_DOUBLE(element)));
    }
    if (PyArray_IsScalar(element, Half)) {
        return PyBool_FromLong(npy_half_isnan(PyArrayScalar_VAL(element, Half)));
    }
    if (PyArray_IsScalar(element, Float32)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(element, Float32)));
    }
    if (PyArray_IsScalar(element, Float64)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(element, Float64)));
    }
    # ifdef PyFloat128ArrType_Type
    if (PyArray_IsScalar(element, Float128)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(element, Float128)));
    }
    # endif

    // Complex NaN
    if (PyComplex_Check(element)) {
        Py_complex val = ((PyComplexObject*)element)->cval;
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    if (PyArray_IsScalar(element, Complex64)) {
        npy_cfloat val = PyArrayScalar_VAL(element, Complex64);
        return PyBool_FromLong(isnan(npy_crealf(val)) || isnan(npy_cimagf(val)));
    }
    if (PyArray_IsScalar(element, Complex128)) {
        npy_cdouble val = PyArrayScalar_VAL(element, Complex128);
        return PyBool_FromLong(isnan(npy_creal(val)) || isnan(npy_cimag(val)));
    }
    # ifdef PyComplex256ArrType_Type
    if (PyArray_IsScalar(element, Complex256)) {
        npy_clongdouble val = PyArrayScalar_VAL(element, Complex256);
        return PyBool_FromLong(isnan(npy_creall(val)) || isnan(npy_cimagl(val)));
    }
    # endif

    // NaT - Datetime
    if (PyArray_IsScalar(element, Datetime)) {
        return PyBool_FromLong(PyArrayScalar_VAL(element, Datetime) == NPY_DATETIME_NAT);
    }
    // NaT - Timedelta
    if (PyArray_IsScalar(element, Timedelta)) {
        return PyBool_FromLong(PyArrayScalar_VAL(element, Timedelta) == NPY_DATETIME_NAT);
    }
    // Try to identify Pandas Timestamp NATs
    if (PyObject_HasAttrString(element, "to_numpy")) {
        // strcmp returns 0 on match
        return PyBool_FromLong(strcmp(element->ob_type->tp_name, "NaTType") == 0);
        // the long way
        // PyObject *to_numpy = PyObject_GetAttrString(element, "to_numpy");
        // if (to_numpy == NULL) {
        //     return NULL;
        // }
        // if (!PyCallable_Check(to_numpy)) {
        //     Py_DECREF(to_numpy);
        //     Py_RETURN_FALSE;
        // }
        // PyObject* scalar = PyObject_CallFunction(to_numpy, NULL);
        // Py_DECREF(to_numpy);
        // if (scalar == NULL) {
        //     return NULL;
        // }
        // if (!PyArray_IsScalar(scalar, Datetime)) {
        //     Py_DECREF(scalar);
        //     Py_RETURN_FALSE;
        // }
        // PyObject* pb = PyBool_FromLong(PyArrayScalar_VAL(scalar, Datetime) == NPY_DATETIME_NAT);
        // Py_DECREF(scalar);
        // return pb;
    }
    Py_RETURN_FALSE;
}

PyObject *
get_new_indexers_and_screen(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    /*
    Used to determine the new indexers and index screen in an index hierarchy selection.

    Example:

        Context:
            We are an index hierarchy, constructing a new index hierarchy from a
            selection of ourself. We need to build this up for each depth. For
            example:

            index_at_depth: ["a", "b", "c", "d"]
            indexer_at_depth: [1, 0, 0, 2, 3, 0, 3, 3, 2]
            (i.e. our index_hierarchy has these labels at depth = ["b", "a", "a", "c", "d", "a", "d", "d", "c"])

            Imagine we are choosing this selection:
                index_hierarchy.iloc[1:4]
                At our depth, this would result in these labels: ["a", "a", "c", "d"]

            We need to output:
                index_screen: [0, 2, 3]
                    - New index is created by: index_at_depth[[0, 2, 3]] (i.e. ["a", "c", "d"])
                new_indexer:  [0, 0, 1, 2]
                    - When applied to our new_index, results in ["a", "a", "c", "d"]

        Function:
            input:
                indexers:  [0, 0, 2, 3] (i.e. indexer_at_depth[1:4])
                positions: [0, 1, 2, 3] (i.e. which ilocs from index that ``indexers`` maps to)

            algorithm:
                Loop through ``indexers``. Since we know that ``indexers`` only contains
                integers from 0 -> ``num_unique`` - 1, we can use a new indexers called
                ``element_locations`` to keep track of which elements have been found, and when.
                (Use ``num_unique`` as the flag for which elements have not been
                found since it's not possible for one of our inputs to equal that)

                Using the above example, this would look like:

                    element_locations =
                    [4, 4, 4, 4] (starting)
                    [0, 4, 4, 4] (first loop)  indexers[0] = 0, so mark it as the 0th element found
                    [0, 4, 4, 4] (second loop) indexers[1] = 0, already marked, move on
                    [0, 4, 1, 4] (third loop)  indexers[2] = 2, so mark it as the 1th element found
                    [0, 4, 1, 2] (fourth loop) indexers[3] = 3, so mark it as the 2th element found

                Now, if during this loop, we discover every single element, it means
                we can exit early, and just return back the original inputs, since
                those arrays contain all the information the caller needs! This is the
                core optimization of this function.
                Example:
                    indexers  = [0, 3, 1, 2, 3, 1, 0, 0]
                    positions = [0, 1, 2, 3]

                    There is no remapping needed! Simple re-use everything!

                Now, if we don't find all the elements, then we need to construct
                ``new_indexers`` and ``index_screen``.

                We can construct ``new_indexers`` during the loop, by using the
                information we have placed into ``element_locations``.

                Using the above example, this would look like:
                    [x, x, x, x] (starting)
                    [0, x, x, x] (first loop)  element_locations[indexers[0]] = 0
                    [0, 0, x, x] (second loop) element_locations[indexers[1]] = 0
                    [0, 0, 1, x] (third loop)  element_locations[indexers[2]] = 1
                    [0, 0, 1, 2] (fourth loop) element_locations[indexers[3]] = 2

                Finally, all that's left is to construct ``index_screen``, which
                is essentially a way to condense and remap ``element_locations``.
                See ``AK_get_index_screen`` for more details.

            output:
                index_screen: [0, 2, 3]
                new_indexer:  [0, 0, 1, 2]

    Equivalent Python code:

        num_unique = len(positions)
        element_locations = np.full(num_unique, num_unique, dtype=np.int64)
        order_found = np.full(num_unique, num_unique, dtype=np.int64)
        new_indexers = np.empty(len(indexers), dtype=np.int64)

        num_found = 0

        for i, element in enumerate(indexers):
            if element_locations[element] == num_unique:
                element_locations[element] = num_found
                order_found[num_found] = element
                num_found += 1

            if num_found == num_unique:
                return positions, indexers

            new_indexers[i] = element_locations[element]

        return order_found[:num_found], new_indexers
    */
    PyArrayObject *indexers;
    PyArrayObject *positions;

    static char *kwlist[] = {"indexers", "positions", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
            kwargs,
            "O!O!:get_new_indexers_and_screen", kwlist,
            &PyArray_Type, &indexers,
            &PyArray_Type, &positions
        ))
    {
        return NULL;
    }

    if (PyArray_NDIM(indexers) != 1) {
        PyErr_SetString(PyExc_ValueError, "indexers must be 1-dimensional");
        return NULL;
    }

    if (PyArray_NDIM(positions) != 1) {
        PyErr_SetString(PyExc_ValueError, "positions must be 1-dimensional");
        return NULL;
    }

    if (PyArray_TYPE(indexers) != NPY_INT64) {
        PyErr_SetString(PyExc_ValueError, "Array must be of type np.int64");
        return NULL;
    }

    npy_intp num_unique = PyArray_SIZE(positions);

    if (num_unique > PyArray_SIZE(indexers)) {
        // This algorithm is only optimal if the number of unique elements is
        // less than the number of elements in the indexers.
        // Otherwise, the most optimal code is ``np.unique(indexers, return_index=True)``
        // and we don't want to re-implement that in C.
        PyErr_SetString(
                PyExc_ValueError,
                "Number of unique elements must be less than or equal to the length of ``indexers``"
                );
        return NULL;
    }

    npy_intp dims = {num_unique};
    PyArrayObject *element_locations = (PyArrayObject*)PyArray_EMPTY(
            1,         // ndim
            &dims,     // shape
            NPY_INT64, // dtype
            0          // fortran
            );
    if (element_locations == NULL) {
        return NULL;
    }

    PyArrayObject *order_found = (PyArrayObject*)PyArray_EMPTY(
            1,         // ndim
            &dims,     // shape
            NPY_INT64, // dtype
            0          // fortran
            );
    if (order_found == NULL) {
        Py_DECREF(element_locations);
        return NULL;
    }

    PyObject *num_unique_pyint = PyLong_FromLong((long)num_unique);
    if (num_unique_pyint == NULL) {
        goto fail;
    }

    // We use ``num_unique`` here to signal that we haven't found the element yet
    // This works, because each element must be 0 < num_unique.
    int fill_success = PyArray_FillWithScalar(element_locations, num_unique_pyint);
    if (fill_success != 0) {
        Py_DECREF(num_unique_pyint);
        goto fail;
    }

    fill_success = PyArray_FillWithScalar(order_found, num_unique_pyint);
    Py_DECREF(num_unique_pyint);
    if (fill_success != 0) {
        goto fail;
    }

    PyArrayObject *new_indexers = (PyArrayObject*)PyArray_EMPTY(
            1,                      // ndim
            PyArray_DIMS(indexers), // shape
            NPY_INT64,              // dtype
            0                       // fortran
            );
    if (new_indexers == NULL) {
        goto fail;
    }

    // We know that our incoming dtypes are all int64! This is a safe cast.
    // Plus, it's easier (and less error prone) to work with native C-arrays
    // over using numpy's iteration APIs.
    npy_int64 *element_location_values = (npy_int64*)PyArray_DATA(element_locations);
    npy_int64 *order_found_values = (npy_int64*)PyArray_DATA(order_found);
    npy_int64 *new_indexers_values = (npy_int64*)PyArray_DATA(new_indexers);

    // Now, implement the core algorithm by looping over the ``indexers``.
    // We need to use numpy's iteration API, as the ``indexers`` could be
    // C-contiguous, F-contiguous, both, or neither.
    // See https://numpy.org/doc/stable/reference/c-api/iterator.html#simple-iteration-example
    NpyIter *indexer_iter = NpyIter_New(
            indexers,                                   // array
            NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP, // iter flags
            NPY_KEEPORDER,                              // order
            NPY_NO_CASTING,                             // casting
            NULL                                        // dtype
            );
    if (indexer_iter == NULL) {
        Py_DECREF(new_indexers);
        goto fail;
    }

    // The iternext function gets stored in a local variable so it can be called repeatedly in an efficient manner.
    NpyIter_IterNextFunc *indexer_iternext = NpyIter_GetIterNext(indexer_iter, NULL);
    if (indexer_iternext == NULL) {
        NpyIter_Deallocate(indexer_iter);
        Py_DECREF(new_indexers);
        goto fail;
    }

    // All of these will be updated by the iterator
    char **dataptr = NpyIter_GetDataPtrArray(indexer_iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(indexer_iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(indexer_iter);

    // No gil is required from here on!
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    size_t i = 0;
    Py_ssize_t num_found = 0;
    do {
        // Get the inner loop data/stride/inner_size values
        char* data = *dataptr;
        npy_intp stride = *strideptr;
        npy_intp inner_size = *innersizeptr;
        npy_int64 element;

        while (inner_size--) {
            element = *((npy_int64 *)data);

            if (element_location_values[element] == num_unique) {
                element_location_values[element] = num_found;
                order_found_values[num_found] = element;
                ++num_found;

                if (num_found == num_unique) {
                    // This insight is core to the performance of the algorithm.
                    // If we have found every possible indexer, we can simply return
                    // back the inputs! Essentially, we can observe on <= single pass
                    // that we have the opportunity for re-use
                    goto finish_early;
                }
            }

            new_indexers_values[i] = element_location_values[element];

            data += stride;
            ++i;
        }

    // Increment the iterator to the next inner loop
    } while(indexer_iternext(indexer_iter));

    NPY_END_THREADS;

    NpyIter_Deallocate(indexer_iter);
    Py_DECREF(element_locations);

    // new_positions = order_found[:num_unique]
    PyObject *new_positions = PySequence_GetSlice(
            (PyObject*)order_found,
            0,
            num_found);
    Py_DECREF(order_found);
    if (new_positions == NULL) {
        return NULL;
    }

    // return new_positions, new_indexers
    PyObject *result = PyTuple_Pack(2, new_positions, new_indexers);
    Py_DECREF(new_indexers);
    Py_DECREF(new_positions);
    return result;

    finish_early:
        NPY_END_THREADS;

        NpyIter_Deallocate(indexer_iter);
        Py_DECREF(element_locations);
        Py_DECREF(order_found);
        Py_DECREF(new_indexers);
        return PyTuple_Pack(2, positions, indexers);

    fail:
        Py_DECREF(element_locations);
        Py_DECREF(order_found);
        return NULL;
}

static char *array_deepcopy_kwarg_names[] = {
    "array",
    "memo",
    NULL
};

PyObject *
array_deepcopy(PyObject *m, PyObject *args, PyObject *kwargs)
{
    PyObject *array;
    PyObject *memo = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|O:array_deepcopy", array_deepcopy_kwarg_names,
            &array,
            &memo)) {
        return NULL;
    }
    if ((memo == NULL) || (memo == Py_None)) {
        memo = NULL;
    }
    else {
        if (!PyDict_Check(memo)) {
            PyErr_SetString(PyExc_TypeError, "memo must be a dict or None");
            return NULL;
        }
    }
    AK_CHECK_NUMPY_ARRAY(array);

    // Perform a deepcopy on an array, using an optional memo dictionary, and specialized to depend on immutable arrays. This depends on the module object to get the deepcopy method. The `memo` object can be None.
    PyObject *id = PyLong_FromVoidPtr(array);
    if (!id) return NULL;

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
    PyArray_Descr *dtype = PyArray_DESCR((PyArrayObject*)array); // borrowed ref

    if (PyDataType_ISOBJECT(dtype)) {
        // we store the deepcopy function on this module for faster lookup here
        PyObject *deepcopy = PyObject_GetAttrString(m, "deepcopy");
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
                (PyArrayObject*)array,
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
