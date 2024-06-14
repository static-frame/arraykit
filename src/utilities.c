# include "Python.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

# include "utilities.h"

PyArrayObject *
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

PyArray_Descr *
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

PyObject *
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

PyObject *
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

PyObject *
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

PyObject *
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
