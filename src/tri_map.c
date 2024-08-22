# include "Python.h"
# include "stdbool.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"

# include "tri_map.h"
# include "utilities.h"

static inline NPY_DATETIMEUNIT
AK_dt_unit_from_array(PyArrayObject* a) {
    // This is based on get_datetime_metadata_from_dtype in the NumPy source, but that function is private. This does not check that the dtype is of the appropriate type.
    PyArray_Descr* dt = PyArray_DESCR(a); // borrowed ref
    PyArray_DatetimeMetaData* dma = &(((PyArray_DatetimeDTypeMetaData *)PyDataType_C_METADATA(dt))->meta);
    // PyArray_DatetimeMetaData* dma = &(((PyArray_DatetimeDTypeMetaData *)PyArray_DESCR(a)->c_metadata)->meta);
    return dma->base;
}

typedef struct TriMapOne {
    Py_ssize_t from; // signed
    Py_ssize_t to;
} TriMapOne;

typedef struct TriMapManyTo {
    Py_ssize_t start;
    Py_ssize_t stop;
} TriMapManyTo;

typedef struct TriMapManyFrom {
    npy_intp src;
    PyArrayObject* dst;
} TriMapManyFrom;

typedef struct TriMapObject {
    PyObject_HEAD
    Py_ssize_t src_len;
    Py_ssize_t dst_len;
    Py_ssize_t len;
    bool is_many;
    bool finalized;

    PyObject* src_match; // array object
    npy_bool* src_match_data; // contiguous C array
    PyObject* dst_match; // array object
    npy_bool* dst_match_data; // contiguous C array

    PyObject* final_src_fill; // array object
    PyObject* final_dst_fill; // array object

    // register one
    TriMapOne* src_one;
    Py_ssize_t src_one_count;
    Py_ssize_t src_one_capacity;

    TriMapOne* dst_one;
    Py_ssize_t dst_one_count;
    Py_ssize_t dst_one_capacity;

    // register_many
    TriMapManyTo* many_to; // two integers for contiguous assignment region
    TriMapManyFrom* many_from; // int and array, for src and dst (together)
    Py_ssize_t many_count;
    Py_ssize_t many_capacity;

} TriMapObject;

PyObject *
TriMap_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs) {
    TriMapObject *self = (TriMapObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }
    return (PyObject *)self;
}

PyDoc_STRVAR(
    TriMap_doc,
    "\n"
    "A utilty for three-way join mappings."
);

// Returns 0 on success, -1 on error.
int
TriMap_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t src_len;
    Py_ssize_t dst_len;
    if (!PyArg_ParseTuple(args,
            "nn:__init__",
            &src_len,
            &dst_len)) {
        return -1;
    }
    TriMapObject* tm = (TriMapObject*)self;
    // handle all C types
    tm->src_len = src_len;
    tm->dst_len = dst_len;
    tm->is_many = false;
    tm->finalized = false;
    tm->len = 0;

    // we create arrays, and also pre-extract pointers to array data for fast insertion; we keep the array for optimal summing routines
    npy_intp dims_src_len[] = {src_len};
    tm->src_match = PyArray_ZEROS(1, dims_src_len, NPY_BOOL, 0);
    if (tm->src_match == NULL) {
        return -1;
    }
    tm->src_match_data = (npy_bool*)PyArray_DATA((PyArrayObject*)tm->src_match);

    npy_intp dims_dst_len[] = {dst_len};
    tm->dst_match = PyArray_ZEROS(1, dims_dst_len, NPY_BOOL, 0);
    if (tm->dst_match == NULL) {
        return -1;
    }
    tm->dst_match_data = (npy_bool*)PyArray_DATA((PyArrayObject*)tm->dst_match);

    // register one
    tm->src_one_count = 0;
    tm->src_one_capacity = 16;
    tm->src_one = (TriMapOne*)PyMem_Malloc(
            sizeof(TriMapOne) * tm->src_one_capacity);
    if (tm->src_one == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }
    tm->dst_one_count = 0;
    tm->dst_one_capacity = 16;
    tm->dst_one = (TriMapOne*)PyMem_Malloc(
            sizeof(TriMapOne) * tm->dst_one_capacity);
    if (tm->dst_one == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }
    // register many
    tm->many_count = 0;
    tm->many_capacity = 16;
    tm->many_to = (TriMapManyTo*)PyMem_Malloc(
            sizeof(TriMapManyTo) * tm->many_capacity);
    if (tm->many_to == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }
    tm->many_from = (TriMapManyFrom*)PyMem_Malloc(
            sizeof(TriMapManyFrom) * tm->many_capacity);
    if (tm->many_from == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }

    return 0;
}

void
TriMap_dealloc(TriMapObject *self) {
    // NOTE: we use XDECREF incase init fails before these objects get allocated
    Py_XDECREF(self->src_match);
    Py_XDECREF(self->dst_match);
    Py_XDECREF(self->final_src_fill);
    Py_XDECREF(self->final_dst_fill);

    if (self->src_one != NULL) {
        PyMem_Free(self->src_one);
    }
    if (self->dst_one != NULL) {
        PyMem_Free(self->dst_one);
    }
    if (self->many_to != NULL) {
        PyMem_Free(self->many_to);
    }
    if (self->many_from != NULL) {
        // decref all arrays before freeing
        for (Py_ssize_t i = 0; i < self->many_count; i++) {
            // NOTE: using dot to get to pointer?
            Py_DECREF((PyObject*)self->many_from[i].dst);
        }
        PyMem_Free(self->many_from);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *
TriMap_repr(TriMapObject *self) {
    const char *is_many = self->is_many ? "true" : "false";
    const char *is_finalized = self->finalized ? "true" : "false";

    npy_intp src_fill;
    npy_intp dst_fill;
    if (self->finalized) {
        src_fill = PyArray_SIZE((PyArrayObject*)self->final_src_fill);
        dst_fill = PyArray_SIZE((PyArrayObject*)self->final_dst_fill);
    }
    else {
        src_fill = -1;
        dst_fill = -1;
    }

    return PyUnicode_FromFormat("<%s(len: %i, src_fill: %i, dst_fill: %i, is_many: %s, is_finalized: %s)>",
            Py_TYPE(self)->tp_name,
            self->len,
            src_fill,
            dst_fill,
            is_many,
            is_finalized);
}

// Provide the integer positions connecting the `src` to the `dst`. If there is no match to `src` or `dst`, the unmatched position can be provided with -1. From each side, a connection is documented to the current `len`. Each time this is called `len` is incremented, indicating the inrease in position in the `final`. Return NULL on error.

// Inner function for calling from C; returns 0 on success, -1 on error. Exceptions will be set on error.
static inline int
AK_TM_register_one(TriMapObject* tm, Py_ssize_t src_from, Py_ssize_t dst_from) {
    bool src_matched = src_from >= 0;
    bool dst_matched = dst_from >= 0;
    if (src_from >= tm->src_len || dst_from >= tm->dst_len) {
        PyErr_SetString(PyExc_ValueError, "Out of bounds locator");
        return -1;
    }
    if (src_matched) {
        if (AK_UNLIKELY(tm->src_one_count == tm->src_one_capacity)) {
            tm->src_one_capacity <<= 1; // get 2x the capacity
            tm->src_one = PyMem_Realloc(tm->src_one,
                    sizeof(TriMapOne) * tm->src_one_capacity);
            if (tm->src_one == NULL) {
                PyErr_SetNone(PyExc_MemoryError);
                return -1;
            }
        }
        tm->src_one[tm->src_one_count] = (TriMapOne){src_from, tm->len};
        tm->src_one_count += 1;
    }
    if (dst_matched) {
        if (AK_UNLIKELY(tm->dst_one_count == tm->dst_one_capacity)) {
            tm->dst_one_capacity <<= 1; // get 2x the capacity
            tm->dst_one = PyMem_Realloc(tm->dst_one,
                    sizeof(TriMapOne) * tm->dst_one_capacity);
            if (tm->dst_one == NULL) {
                PyErr_SetNone(PyExc_MemoryError);
                return -1;
            }
        }
        tm->dst_one[tm->dst_one_count] = (TriMapOne){dst_from, tm->len};
        tm->dst_one_count += 1;
    }
    if (src_matched && dst_matched) {
        if (!tm->is_many) {
            // if we have seen this connection before, we have a many
            if (tm->src_match_data[src_from] || tm->dst_match_data[dst_from]) {
                tm->is_many = true;
            }
        }
        tm->src_match_data[src_from] = NPY_TRUE;
        tm->dst_match_data[dst_from] = NPY_TRUE;
    }
    tm->len += 1;
    return 0;
}

// Public function for calling from Python.
PyObject *
TriMap_register_one(TriMapObject *self, PyObject *args) {
    Py_ssize_t src_from;
    Py_ssize_t dst_from;
    if (!PyArg_ParseTuple(args,
            "nn:register_one",
            &src_from,
            &dst_from)) {
        return NULL;
    }
    if (self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot register post finalization");
        return NULL;
    }
    if (AK_TM_register_one(self, src_from, dst_from)) {
        return NULL;
    }
    Py_RETURN_NONE;
}

PyObject *
TriMap_register_unmatched_dst(TriMapObject *self) {
    if (self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot register post finalization");
        return NULL;
    }
    PyArrayObject* dst_match_array = (PyArrayObject *)self->dst_match;
    PyObject* sum_scalar = PyArray_Sum(
            dst_match_array,
            0,
            NPY_INT64, // this converts before sum; not sure this is necessary
            NULL);
    if (sum_scalar == NULL) {
        return NULL;
    }
    // for a 1D array PyArray_SUM returns a scalar
    npy_int64 sum = PyArrayScalar_VAL(sum_scalar, Int64);
    Py_DECREF(sum_scalar);

    if (sum < self->dst_len) {
        PyArrayObject* dst_unmatched = (PyArrayObject *)PyObject_CallMethod(
                self->dst_match, // PyObject
                "__invert__",
                NULL);
        if (dst_unmatched == NULL) {
            return NULL;
        }
        // derive indices for unmatched locations, call each with register_one
        PyArrayObject* indices = (PyArrayObject*)AK_nonzero_1d(dst_unmatched);
        if (indices == NULL) {
            Py_DECREF((PyObject*)dst_unmatched);
            return NULL;
        }
        // borrow ref to array in 1-element tuple
        npy_int64 *index_data = (npy_int64 *)PyArray_DATA(indices);
        npy_intp index_len = PyArray_SIZE(indices);

        for (npy_intp i = 0; i < index_len; i++) {
            if (AK_TM_register_one(self, -1, index_data[i])) {
                Py_DECREF((PyObject*)dst_unmatched);
                Py_DECREF((PyObject*)indices);
                return NULL;
            }
        }
        Py_DECREF((PyObject*)dst_unmatched);
        Py_DECREF((PyObject*)indices);
    }
    Py_RETURN_NONE;
}

// Given an integer (for the src) and an array of integers (for the dst), store mappings from src to final and dst to final.
PyObject *
TriMap_register_many(TriMapObject *self, PyObject *args) {
    Py_ssize_t src_from;
    PyArrayObject* dst_from;
    if (!PyArg_ParseTuple(args,
            "nO!:register_many",
            &src_from,
            &PyArray_Type, &dst_from)) {
        return NULL;
    }
    if (self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot register post finalization");
        return NULL;
    }
    int dst_from_type = PyArray_TYPE(dst_from);
    if (dst_from_type != NPY_INT64) {
        PyErr_SetString(PyExc_ValueError, "`dst_from` must be a 64 bit integer array");
        return NULL;
    }
    npy_intp increment = PyArray_SIZE(dst_from);

    if (AK_UNLIKELY(self->many_count == self->many_capacity)) {
        self->many_capacity <<= 1; // get 2x the capacity
        self->many_to = PyMem_Realloc(self->many_to,
                sizeof(TriMapManyTo) * self->many_capacity);
        if (self->many_to == NULL) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }
        self->many_from = PyMem_Realloc(self->many_from,
                sizeof(TriMapManyFrom) * self->many_capacity);
        if (self->many_from == NULL) {
            PyErr_SetNone(PyExc_MemoryError);
            return NULL;
        }
    }
    // define contiguous region in final to map to
    self->many_to[self->many_count] = (TriMapManyTo){self->len, self->len + increment};

    Py_INCREF((PyObject*)dst_from); // decrefs on dealloc
    self->many_from[self->many_count] = (TriMapManyFrom){src_from, dst_from};
    self->many_count += 1;

    self->src_match_data[src_from] = NPY_TRUE;
    // iterate over dst_from and set values to True; cannot assume that dst_from is contiguous; dst_match_data is contiguous
    for (Py_ssize_t i = 0; i < increment; i++){
        npy_int64 pos = *(npy_int64*)PyArray_GETPTR1(dst_from, i); // always int64
        self->dst_match_data[pos] = NPY_TRUE;
    }
    self->len += increment;
    self->is_many = true;
    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------
// Determine, for src and dst, which indices will need fill values, and store those indices as an integer array in final_src_fill, final_dst_fill
PyObject *
TriMap_finalize(TriMapObject *self, PyObject *Py_UNUSED(unused)) {
    TriMapObject* tm = (TriMapObject*)self;

    if (self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot call finalize twice");
        return NULL;
    }
    // predefine all PyObjects to use goto error
    PyObject* final_src_match = NULL;
    PyObject* final_dst_match = NULL;
    PyObject* final_src_unmatched = NULL;
    PyObject* final_dst_unmatched = NULL;

    npy_intp dims[] = {tm->len};

    // initialize all to False
    final_src_match = PyArray_ZEROS(1, dims, NPY_BOOL, 0);
    if (final_src_match == NULL) {
        goto error;
    }
    final_dst_match = PyArray_ZEROS(1, dims, NPY_BOOL, 0);
    if (final_dst_match == NULL) {
        goto error;
    }

    npy_bool* final_src_match_data = (npy_bool*)PyArray_DATA(
            (PyArrayObject*)final_src_match);
    npy_bool* final_dst_match_data = (npy_bool*)PyArray_DATA(
            (PyArrayObject*)final_dst_match);

    TriMapOne* o;
    TriMapOne* o_end;
    o = tm->src_one;
    o_end = o + tm->src_one_count;
    for (; o < o_end; o++) {
        final_src_match_data[o->to] = NPY_TRUE;
    }
    o = tm->dst_one;
    o_end = o + tm->dst_one_count;
    for (; o < o_end; o++) {
        final_dst_match_data[o->to] = NPY_TRUE;
    }
    // many assign from src and dst into the same final positions
    npy_bool* s;
    npy_bool* d;
    npy_bool* end;
    TriMapManyTo* m = tm->many_to;
    TriMapManyTo* m_end = m + tm->many_count;

    for (; m < m_end; m++) {
        d = final_dst_match_data + m->start;
        s = final_src_match_data + m->start;
        end = final_src_match_data + m->stop;
        while (s < end) {
            *s++ = NPY_TRUE;
            *d++ = NPY_TRUE;
        }
    }
    // NOTE: could sum first to see if nonzero call is necessary; would skip invert and nonzero calls
    final_src_unmatched = PyObject_CallMethod(
            final_src_match, // PyObject
            "__invert__",
            NULL);
    if (final_src_unmatched == NULL) {
        goto error;
    }

    final_dst_unmatched = PyObject_CallMethod(
            final_dst_match, // PyObject
            "__invert__",
            NULL);
    if (final_dst_unmatched == NULL) {
        goto error;
    }
    tm->final_src_fill = AK_nonzero_1d((PyArrayObject*)final_src_unmatched);
    if (tm->final_src_fill == NULL) {
        goto error;
    }
    tm->final_dst_fill = AK_nonzero_1d((PyArrayObject*)final_dst_unmatched);
    if (tm->final_dst_fill == NULL) {
        goto error;
    }
    Py_DECREF(final_src_match);
    Py_DECREF(final_dst_match);
    Py_DECREF(final_src_unmatched);
    Py_DECREF(final_dst_unmatched);

    tm->finalized = true;
    Py_RETURN_NONE;
error: // all PyObject initialized to NULL, no more than 1 ref
    Py_XDECREF(final_src_match);
    Py_XDECREF(final_dst_match);
    Py_XDECREF(final_src_unmatched);
    Py_XDECREF(final_dst_unmatched);
    return NULL;
}

PyObject *
TriMap_is_many(TriMapObject *self, PyObject *Py_UNUSED(unused)) {
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    if (self->is_many) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// Return True if the `src` will not need a fill. This is only correct of `src` is binding to a left join or an inner join.
PyObject *
TriMap_src_no_fill(TriMapObject *self, PyObject *Py_UNUSED(unused)) {
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    if (PyArray_SIZE((PyArrayObject*)self->final_src_fill) == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// Return True if the `dst` will not need a fill. This is only correct of `dst` is binding to a left join or an inner join.
PyObject *
TriMap_dst_no_fill(TriMapObject *self, PyObject *Py_UNUSED(unused)) {
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    if (PyArray_SIZE((PyArrayObject*)self->final_dst_fill) == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

# define AK_TM_TRANSFER_SCALAR(npy_type_to, npy_type_from) do {        \
    npy_type_to* array_to_data = (npy_type_to*)PyArray_DATA(array_to); \
    TriMapOne* o = one_pairs;                                          \
    TriMapOne* o_end = o + one_count;                                  \
    for (; o < o_end; o++) {                                           \
        array_to_data[o->to] = (npy_type_to)                           \
                *(npy_type_from*)PyArray_GETPTR1(                      \
                array_from, o->from);                                  \
    }                                                                  \
    npy_type_to* t;                                                    \
    npy_type_to* t_end;                                                \
    npy_type_to f;                                                     \
    npy_int64 f_pos;                                                   \
    npy_intp dst_pos;                                                  \
    PyArrayObject* dst;                                                \
    for (Py_ssize_t i = 0; i < tm->many_count; i++) {                  \
        t = array_to_data + tm->many_to[i].start;                      \
        t_end = array_to_data + tm->many_to[i].stop;                   \
        if (from_src) {                                                \
            f = (npy_type_to)*(npy_type_from*)PyArray_GETPTR1(         \
                    array_from, tm->many_from[i].src);                 \
            while (t < t_end) {                                        \
                *t++ = f;                                              \
            }                                                          \
        }                                                              \
        else {                                                         \
            dst_pos = 0;                                               \
            dst = tm->many_from[i].dst;                                \
            while (t < t_end) {                                        \
                f_pos = *(npy_int64*)PyArray_GETPTR1(dst, dst_pos);    \
                *t++ = (npy_type_to)                                   \
                        *(npy_type_from*)PyArray_GETPTR1(              \
                        array_from, f_pos);                            \
                dst_pos++;                                             \
            }                                                          \
        }                                                              \
    }                                                                  \
} while (0)                                                            \

// Based on `tm` state, transfer from src or from dst (depending on `from_src`) to a `array_to`, a newly created contiguous array that is compatible with the values in `array_from`. Returns -1 on error. This only needs to match to / from type combinations that are possible from `resolve_dtype`, i.e., bool never goes to integer.
static inline int
AK_TM_transfer_scalar(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_from,
        PyArrayObject* array_to) {
    Py_ssize_t one_count = from_src ? tm->src_one_count : tm->dst_one_count;
    TriMapOne* one_pairs = from_src ? tm->src_one : tm->dst_one;

    switch(PyArray_TYPE(array_to)){
        case NPY_BOOL:
            AK_TM_TRANSFER_SCALAR(npy_bool, npy_bool);
            return 0;
        case NPY_INT64:
            switch (PyArray_TYPE(array_from)) {
                case NPY_INT64:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_int64);
                    return 0;
                case NPY_INT32:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_int32);
                    return 0;
                case NPY_INT16:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_int16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_int8);
                    return 0;
                case NPY_UINT32:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_uint32);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_int64, npy_uint8);
                    return 0;
            }
            break;
        case NPY_INT32:
            switch (PyArray_TYPE(array_from)) {
                case NPY_INT32:
                    AK_TM_TRANSFER_SCALAR(npy_int32, npy_int32);
                    return 0;
                case NPY_INT16:
                    AK_TM_TRANSFER_SCALAR(npy_int32, npy_int16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_int32, npy_int8);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_int32, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_int32, npy_uint8);
                    return 0;
            }
            break;
        case NPY_INT16:
            switch (PyArray_TYPE(array_from)) {
                case NPY_INT16:
                    AK_TM_TRANSFER_SCALAR(npy_int16, npy_int16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_int16, npy_int8);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_int16, npy_uint8);
                    return 0;
            }
            break;
        case NPY_INT8:
            AK_TM_TRANSFER_SCALAR(npy_int8, npy_int8);
            return 0;
        case NPY_UINT64:
            switch (PyArray_TYPE(array_from)) {
                case NPY_UINT64:
                    AK_TM_TRANSFER_SCALAR(npy_uint64, npy_uint64);
                    return 0;
                case NPY_UINT32:
                    AK_TM_TRANSFER_SCALAR(npy_uint64, npy_uint32);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_uint64, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_uint64, npy_uint8);
                    return 0;
            }
            break;
        case NPY_UINT32:
            switch (PyArray_TYPE(array_from)) {
                case NPY_UINT32:
                    AK_TM_TRANSFER_SCALAR(npy_uint32, npy_uint32);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_uint32, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_uint32, npy_uint8);
                    return 0;
            }
            break;
        case NPY_UINT16:
            switch (PyArray_TYPE(array_from)) {
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_uint16, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_uint16, npy_uint8);
                    return 0;
            }
            break;
        case NPY_UINT8:
            AK_TM_TRANSFER_SCALAR(npy_uint8, npy_uint8);
            return 0;
        case NPY_FLOAT64:
            switch (PyArray_TYPE(array_from)) {
                case NPY_FLOAT64:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_float64);
                    return 0;
                case NPY_FLOAT32:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_float32);
                    return 0;
                case NPY_FLOAT16:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_float16);
                    return 0;
                case NPY_INT64:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_int64);
                    return 0;
                case NPY_INT32:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_int32);
                    return 0;
                case NPY_INT16:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_int16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_int8);
                    return 0;
                case NPY_UINT64:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_uint64);
                    return 0;
                case NPY_UINT32:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_uint32);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_float64, npy_uint8);
                    return 0;
            }
            break;
        case NPY_FLOAT32:
            switch (PyArray_TYPE(array_from)) {
                case NPY_FLOAT32:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_float32);
                    return 0;
                case NPY_FLOAT16:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_float16);
                    return 0;
                case NPY_INT16:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_int16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_int8);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_float32, npy_uint8);
                    return 0;
            }
            break;
        case NPY_FLOAT16:
            switch (PyArray_TYPE(array_from)) {
                case NPY_FLOAT16:
                    AK_TM_TRANSFER_SCALAR(npy_float16, npy_float16);
                    return 0;
                case NPY_INT8:
                    AK_TM_TRANSFER_SCALAR(npy_float16, npy_int8);
                    return 0;
                case NPY_UINT16:
                    AK_TM_TRANSFER_SCALAR(npy_float16, npy_uint16);
                    return 0;
                case NPY_UINT8:
                    AK_TM_TRANSFER_SCALAR(npy_float16, npy_uint8);
                    return 0;
            }
            break;
        case NPY_DATETIME: {
            AK_TM_TRANSFER_SCALAR(npy_int64, npy_int64);
            return 0;
        }
    }
    PyErr_SetString(PyExc_TypeError, "No handling for types");
    return -1;
}
#undef AK_TM_TRANSFER_SCALAR

// Returns -1 on error. Specialized transfer from any type of an array to an object array.
static inline int
AK_TM_transfer_object(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_from,
        PyArrayObject* array_to
        ) {
    Py_ssize_t one_count = from_src ? tm->src_one_count : tm->dst_one_count;
    TriMapOne* one_pairs = from_src ? tm->src_one : tm->dst_one;

    // NOTE: could use PyArray_Scalar instead of PyArray_GETITEM if we wanted to store scalars instead of Python objects; however, that is pretty uncommon for object arrays to store PyArray_Scalars
    bool f_is_obj = PyArray_TYPE(array_from) == NPY_OBJECT;

    // the passed in object array is contiguous and have NULL (not None) in each position
    PyObject** array_to_data = (PyObject**)PyArray_DATA(array_to);
    PyObject* pyo;
    void* f;
    TriMapOne* o = one_pairs;
    TriMapOne* o_end = o + one_count;
    for (; o < o_end; o++) {
        f = PyArray_GETPTR1(array_from, o->from);
        if (f_is_obj) {
            pyo = *(PyObject**)f;
            Py_INCREF(pyo);
        }
        else { // will convert any value to an object
            pyo = PyArray_GETITEM(array_from, f);
        }
        array_to_data[o->to] = pyo;
    }
    PyObject** t;
    PyObject** t_end;
    npy_intp dst_pos;
    npy_int64 f_pos;
    PyArrayObject* dst;
    for (Py_ssize_t i = 0; i < tm->many_count; i++) {
        t = array_to_data + tm->many_to[i].start;
        t_end = array_to_data + tm->many_to[i].stop;

        if (from_src) {
            f = PyArray_GETPTR1(array_from, tm->many_from[i].src);
            if (f_is_obj) {
                pyo = *(PyObject**)f;
                Py_INCREF(pyo); // pre add new ref so equal to PyArray_GETITEM
            }
            else {
                pyo = PyArray_GETITEM(array_from, f); // given a new ref
            }
            while (t < t_end) {
                Py_INCREF(pyo); // one more than we need
                *t++ = pyo;
            }
            Py_DECREF(pyo); // remove the extra ref
        }
        else { // from_dst, dst is an array
            dst_pos = 0;
            dst = tm->many_from[i].dst;
            while (t < t_end) {
                f_pos = *(npy_int64*)PyArray_GETPTR1(dst, dst_pos);
                f = PyArray_GETPTR1(array_from, f_pos);
                if (f_is_obj) {
                    pyo = *(PyObject**)f;
                    Py_INCREF(pyo);
                }
                else {
                    pyo = PyArray_GETITEM(array_from, f);
                }
                *t++ = pyo;
                dst_pos++;
            }
        }
    }
    return 0;
}

// Returns -1 on error. Specialized transfer from any type of an array to an object array. For usage with merge, Will only transfer if the destination is not NULL.
static inline int
AK_TM_transfer_object_if_null(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_from,
        PyArrayObject* array_to
        ) {
    Py_ssize_t one_count = from_src ? tm->src_one_count : tm->dst_one_count;
    TriMapOne* one_pairs = from_src ? tm->src_one : tm->dst_one;

    // NOTE: could use PyArray_Scalar instead of PyArray_GETITEM if we wanted to store scalars instead of Python objects; however, that is pretty uncommon for object arrays to store PyArray_Scalars
    bool f_is_obj = PyArray_TYPE(array_from) == NPY_OBJECT;

    // the passed in object array is contiguous and have NULL (not None) in each position
    PyObject** array_to_data = (PyObject**)PyArray_DATA(array_to);
    PyObject* pyo;
    void* f;
    TriMapOne* o = one_pairs;
    TriMapOne* o_end = o + one_count;
    for (; o < o_end; o++) {
        if (array_to_data[o->to] == NULL) {
            f = PyArray_GETPTR1(array_from, o->from);
            if (f_is_obj) {
                pyo = *(PyObject**)f;
                Py_INCREF(pyo);
            }
            else { // will convert any value to an object
                pyo = PyArray_GETITEM(array_from, f);
            }
            array_to_data[o->to] = pyo;
        }
    }
    PyObject** t;
    PyObject** t_end;
    npy_intp dst_pos;
    npy_int64 f_pos;
    PyArrayObject* dst;
    for (Py_ssize_t i = 0; i < tm->many_count; i++) {
        t = array_to_data + tm->many_to[i].start;
        t_end = array_to_data + tm->many_to[i].stop;

        if (from_src) {
            while (t < t_end) {
                if (*t == NULL) {
                    f = PyArray_GETPTR1(array_from, tm->many_from[i].src);
                    if (f_is_obj) {
                        pyo = *(PyObject**)f;
                        Py_INCREF(pyo);
                    }
                    else {
                        pyo = PyArray_GETITEM(array_from, f); // given a new ref
                    }
                    *t++ = pyo;
                }
                else {
                    t++;
                }
            }
        }
        else { // from_dst, dst is an array
            dst_pos = 0;
            dst = tm->many_from[i].dst;
            while (t < t_end) {
                if (*t == NULL) {
                    f_pos = *(npy_int64*)PyArray_GETPTR1(dst, dst_pos);
                    f = PyArray_GETPTR1(array_from, f_pos);
                    if (f_is_obj) {
                        pyo = *(PyObject**)f;
                        Py_INCREF(pyo);
                    }
                    else {
                        pyo = PyArray_GETITEM(array_from, f);
                    }
                    *t++ = pyo;
                    dst_pos++;
                }
                else {
                    t++;
                    dst_pos++;
                }
            }
        }
    }
    return 0;
}

// Returns -1 on error.
static inline int
AK_TM_fill_object(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_to,
        PyObject* fill_value) {

    PyArrayObject* final_fill = (PyArrayObject*)(from_src
            ? tm->final_src_fill : tm->final_dst_fill);
    PyObject** array_to_data = (PyObject**)PyArray_DATA(array_to);
    npy_int64* p = (npy_int64*)PyArray_DATA(final_fill);
    npy_int64* p_end = p + PyArray_SIZE(final_fill);
    PyObject** target;
    while (p < p_end) {
        target = array_to_data + *p++;
        Py_INCREF(fill_value);
        *target = fill_value;
    }
    return 0;
}

#define AK_TM_TRANSFER_FLEXIBLE(c_type, from_src, array_from, array_to) do {\
    Py_ssize_t one_count = from_src ? tm->src_one_count : tm->dst_one_count;\
    TriMapOne* one_pairs = from_src ? tm->src_one : tm->dst_one;           \
    npy_intp t_element_size = PyArray_ITEMSIZE(array_to);                  \
    npy_intp t_element_cp = t_element_size / sizeof(c_type);               \
    npy_intp f_element_size = PyArray_ITEMSIZE(array_from);                \
    c_type* array_to_data = (c_type*)PyArray_DATA(array_to);               \
    c_type* f;                                                             \
    c_type* t;                                                             \
    c_type* t_end;                                                         \
    npy_intp dst_pos;                                                      \
    npy_int64 f_pos;                                                       \
    PyArrayObject* dst;                                                    \
    TriMapOne* o = one_pairs;                                              \
    TriMapOne* o_end = o + one_count;                                      \
    for (; o < o_end; o++) {                                               \
        f = (c_type*)PyArray_GETPTR1(array_from, o->from);                 \
        t = array_to_data + t_element_cp * o->to;                          \
        memcpy(t, f, f_element_size);                                      \
    }                                                                      \
    for (Py_ssize_t i = 0; i < tm->many_count; i++) {                      \
        t = array_to_data + t_element_cp * tm->many_to[i].start;           \
        t_end = array_to_data + t_element_cp * tm->many_to[i].stop;        \
        if (from_src) {                                                    \
            f = (c_type*)PyArray_GETPTR1(array_from, tm->many_from[i].src);\
            for (; t < t_end; t += t_element_cp) {                         \
                memcpy(t, f, f_element_size);                              \
            }                                                              \
        }                                                                  \
        else {                                                             \
            dst_pos = 0;                                                   \
            dst = tm->many_from[i].dst;                                    \
            for (; t < t_end; t += t_element_cp) {                         \
                f_pos = *(npy_int64*)PyArray_GETPTR1(dst, dst_pos);        \
                f = (c_type*)PyArray_GETPTR1(array_from, f_pos);           \
                memcpy(t, f, f_element_size);                              \
                dst_pos++;                                                 \
            }                                                              \
        }                                                                  \
    }                                                                      \
} while (0)                                                                \

// Returns -1 on error.
static inline int
AK_TM_fill_unicode(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_to,
        PyObject* fill_value) {
    PyArrayObject* final_fill = (PyArrayObject*)(from_src
            ? tm->final_src_fill : tm->final_dst_fill);

    Py_UCS4* array_to_data = (Py_UCS4*)PyArray_DATA(array_to);
    // code points per element
    npy_intp cp = PyArray_ITEMSIZE(array_to) / UCS4_SIZE;

    bool decref_fill_value = false;
    if (PyBytes_Check(fill_value)) {
        fill_value = PyUnicode_FromEncodedObject(fill_value, "utf-8", NULL);
        if (fill_value == NULL) {
            return -1;
        }
        decref_fill_value = true;
    }
    else if (!PyUnicode_Check(fill_value)) {
        return -1;
    }
    Py_ssize_t fill_cp = PyUnicode_GET_LENGTH(fill_value) * UCS4_SIZE; // code points
    // p is the index position to fill
    npy_int64* p = (npy_int64*)PyArray_DATA(final_fill);
    npy_int64* p_end = p + PyArray_SIZE(final_fill);
    Py_UCS4* target;
    while (p < p_end) {
        target = array_to_data + (*p * cp);
        // disabling copying a null
        if (PyUnicode_AsUCS4(fill_value, target, fill_cp, 0) == NULL) {
            return -1;
        }
        p++;
    }
    if (decref_fill_value) {
        Py_DECREF(fill_value);
    }
    return 0;
}

// Returns -1 on error.
static inline int
AK_TM_fill_string(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_to,
        PyObject* fill_value) {
    PyArrayObject* final_fill = (PyArrayObject*)(from_src
            ? tm->final_src_fill : tm->final_dst_fill);

    char* array_to_data = (char*)PyArray_DATA(array_to);
    npy_intp cp = PyArray_ITEMSIZE(array_to);
    if (!PyBytes_Check(fill_value)) {
        return -1;
    }
    Py_ssize_t fill_cp = PyBytes_GET_SIZE(fill_value);
    const char* fill_data = PyBytes_AS_STRING(fill_value);
    // p is the index position to fill
    npy_int64* p = (npy_int64*)PyArray_DATA(final_fill);
    npy_int64* p_end = p + PyArray_SIZE(final_fill);
    char* target;
    while (p < p_end) {
        target = array_to_data + (*p++ * cp);
        memcpy(target, fill_data, fill_cp);
    }
    return 0;
}

// Returns NULL on error.
static inline PyObject *
AK_TM_map_no_fill(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_from) {
    if (!(PyArray_NDIM(array_from) == 1)) {
        PyErr_SetString(PyExc_TypeError, "Array must be 1D");
        return NULL;
    }
    npy_intp dims[] = {tm->len};
    PyArrayObject* array_to;
    bool dtype_is_obj = PyArray_TYPE(array_from) == NPY_OBJECT;
    bool dtype_is_unicode = PyArray_TYPE(array_from) == NPY_UNICODE;
    bool dtype_is_string = PyArray_TYPE(array_from) == NPY_STRING;

    // create to array
    if (dtype_is_obj) { // initializes values to NULL
        array_to = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_OBJECT);
    }
    else {
        PyArray_Descr* dtype = PyArray_DESCR(array_from); // borowed ref
        Py_INCREF(dtype);
        array_to = (PyArrayObject*)PyArray_Empty(1, dims, dtype, 0); // steals dtype ref
    }
    if (array_to == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }
    // transfer values
    if (dtype_is_obj) {
        if (AK_TM_transfer_object(tm, from_src, array_from, array_to)) {
            Py_DECREF((PyObject*)array_to);
            return NULL;
        }
    }
    else if (dtype_is_unicode) {
        AK_TM_TRANSFER_FLEXIBLE(Py_UCS4, from_src, array_from, array_to);
    }
    else if (dtype_is_string) {
        AK_TM_TRANSFER_FLEXIBLE(char, from_src, array_from, array_to);
    }
    else {
        if (AK_TM_transfer_scalar(tm, from_src, array_from, array_to)) {
            Py_DECREF((PyObject*)array_to);
            return NULL;
        }
    }
    PyArray_CLEARFLAGS(array_to, NPY_ARRAY_WRITEABLE);
    return (PyObject*)array_to;
}

PyObject *
TriMap_map_src_no_fill(TriMapObject *self, PyObject *arg) {
    if (!PyArray_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Must provide an array");
        return NULL;
    }
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    PyArrayObject* array_from = (PyArrayObject*)arg;
    bool from_src = true;
    return AK_TM_map_no_fill(self, from_src, array_from);
}

PyObject *
TriMap_map_dst_no_fill(TriMapObject *self, PyObject *arg) {
    if (!PyArray_Check(arg)) {
        PyErr_SetString(PyExc_TypeError, "Must provide an array");
        return NULL;
    }
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    PyArrayObject* array_from = (PyArrayObject*)arg;
    bool from_src = false;
    return AK_TM_map_no_fill(self, from_src, array_from);
}

static inline PyObject *
TriMap_map_merge(TriMapObject *tm, PyObject *args)
{
    // both are "from_" arrays
    PyArrayObject* array_src;
    PyArrayObject* array_dst;

    if (!PyArg_ParseTuple(args,
            "O!O!:map_merge",
            &PyArray_Type, &array_src,
            &PyArray_Type, &array_dst
            )) {
        return NULL;
    }
    if (!tm->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    if (!(PyArray_NDIM(array_src) == 1)) {
        PyErr_SetString(PyExc_TypeError, "Array src must be 1D");
        return NULL;
    }
    if (!(PyArray_NDIM(array_dst) == 1)) {
        PyErr_SetString(PyExc_TypeError, "Array dst must be 1D");
        return NULL;
    }
    // passing a borrowed refs; returns a new ref
    PyArray_Descr* dtype = AK_resolve_dtype(
            PyArray_DESCR(array_src),
            PyArray_DESCR(array_dst));
    bool dtype_is_obj = dtype->type_num == NPY_OBJECT;
    bool dtype_is_unicode = dtype->type_num == NPY_UNICODE;
    bool dtype_is_string = dtype->type_num == NPY_STRING;

    npy_intp dims[] = {tm->len};

    // create to array_to
    PyArrayObject* array_to;
    if (dtype_is_obj) {
        Py_DECREF(dtype); // not needed
        // will initialize to NULL, not None
        array_to = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_OBJECT);
    }
    else if (dtype_is_unicode || dtype_is_string) {
        array_to = (PyArrayObject*)PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    }
    else {
        array_to = (PyArrayObject*)PyArray_Empty(1, dims, dtype, 0); // steals dtype ref
    }
    if (array_to == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return NULL;
    }

    // if we have fill values in src, we need to transfer from dst
    bool transfer_from_dst = PyArray_SIZE((PyArrayObject*)tm->final_src_fill) != 0;

    if (dtype_is_obj) {
        if (AK_TM_transfer_object(tm, true, array_src, array_to)) {
            Py_DECREF((PyObject*)array_to);
            return NULL;
        }
        if (transfer_from_dst) {
            if (AK_TM_transfer_object_if_null(tm, false, array_dst, array_to)) {
                Py_DECREF((PyObject*)array_to);
                return NULL;
            }
        }
    }
    else if (dtype_is_unicode) {
        AK_TM_TRANSFER_FLEXIBLE(Py_UCS4, true, array_src, array_to);
        if (transfer_from_dst) {
            AK_TM_TRANSFER_FLEXIBLE(Py_UCS4, false, array_dst, array_to);
        }
    }
    else if (dtype_is_string) {
        AK_TM_TRANSFER_FLEXIBLE(char, true, array_src, array_to);
        if (transfer_from_dst) {
            AK_TM_TRANSFER_FLEXIBLE(char, false, array_dst, array_to);
        }
    }
    else {
        if (AK_TM_transfer_scalar(tm, true, array_src, array_to)) {
            Py_DECREF((PyObject*)array_to);
            return NULL;
        }
        if (transfer_from_dst) {
            if (AK_TM_transfer_scalar(tm, false, array_dst, array_to)) {
                Py_DECREF((PyObject*)array_to);
                return NULL;
            }
        }
    }
    return (PyObject*)array_to;
}

// Returns NULL on error.
static inline PyObject *
AK_TM_map_fill(TriMapObject* tm,
        bool from_src,
        PyArrayObject* array_from,
        PyObject* fill_value,
        PyArray_Descr* fill_value_dtype) {
    if (!(PyArray_NDIM(array_from) == 1)) {
        PyErr_SetString(PyExc_TypeError, "Array must be 1D");
        return NULL;
    }
    // passing a borrowed ref; returns a new ref
    PyArray_Descr* dtype = AK_resolve_dtype(PyArray_DESCR(array_from), fill_value_dtype);
    bool dtype_is_obj = dtype->type_num == NPY_OBJECT;
    bool dtype_is_unicode = dtype->type_num == NPY_UNICODE;
    bool dtype_is_string = dtype->type_num == NPY_STRING;

    npy_intp dims[] = {tm->len};
    PyArrayObject* array_to;

    if (dtype_is_obj) {
        Py_DECREF(dtype); // not needed
        // will initialize to NULL, not None
        array_to = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_OBJECT);
        Py_INCREF(array_from); // normalize refs when casting
    }
    else if (dtype_is_unicode || dtype_is_string) {
        array_to = (PyArrayObject*)PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
        Py_INCREF(array_from); // normalize refs when casting
    }
    else {
        array_to = (PyArrayObject*)PyArray_Empty(1, dims, dtype, 0); // steals dtype ref
        if (PyArray_TYPE(array_from) == NPY_DATETIME &&
                PyArray_TYPE(array_to) == NPY_DATETIME &&
                AK_dt_unit_from_array(array_from) != AK_dt_unit_from_array(array_to)
                ) {
            // if trying to cast into a dt64 array, need to pre-convert; array_from is originally borrowed; calling cast sets it to a new ref
            dtype = PyArray_DESCR(array_to); // borrowed ref
            Py_INCREF(dtype);
            array_from = (PyArrayObject*)PyArray_CastToType(array_from, dtype, 0);
        }
        else {
            Py_INCREF(array_from); // normalize refs when casting
        }
    }
    if (array_to == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        Py_DECREF((PyObject*)array_from);
        return NULL;
    }
    // array_from, array_to inc refed and dec refed on error
    if (dtype_is_obj) {
        if (AK_TM_transfer_object(tm, from_src, array_from, array_to)) {
            goto error;
        }
        if (AK_TM_fill_object(tm, from_src, array_to, fill_value)) {
            goto error;
        }
    }
    else if (dtype_is_unicode) {
        AK_TM_TRANSFER_FLEXIBLE(Py_UCS4, from_src, array_from, array_to);
        if (AK_TM_fill_unicode(tm, from_src, array_to, fill_value)) {
            goto error;
        }
    }
    else if (dtype_is_string) {
        AK_TM_TRANSFER_FLEXIBLE(char, from_src, array_from, array_to);
        if (AK_TM_fill_string(tm, from_src, array_to, fill_value)) {
            goto error;
        }
    }
    else {
        // Most simple is to fill with scalar, then overwrite values as needed; for object and flexible dtypes this is not efficient; for object dtypes, this obligates us to decref the filled value when assigning
        if (PyArray_FillWithScalar(array_to, fill_value)) { // -1 on error
            goto error;
        }
        if (AK_TM_transfer_scalar(tm, from_src, array_from, array_to)) {
            goto error;
        }
    }
    Py_DECREF((PyObject*)array_from); // ref inc for this function
    PyArray_CLEARFLAGS(array_to, NPY_ARRAY_WRITEABLE);
    return (PyObject*)array_to;
error:
    Py_DECREF((PyObject*)array_to);
    Py_DECREF((PyObject*)array_from);
    return NULL;
}
#undef AK_TM_TRANSFER_FLEXIBLE

PyObject *
TriMap_map_src_fill(TriMapObject *self, PyObject *args) {
    PyArrayObject* array_from;
    PyObject* fill_value;
    PyArray_Descr* fill_value_dtype;
    if (!PyArg_ParseTuple(args,
            "O!OO!:map_src_fill",
            &PyArray_Type, &array_from,
            &fill_value,
            &PyArrayDescr_Type, &fill_value_dtype
            )) {
        return NULL;
    }
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    bool from_src = true;
    return AK_TM_map_fill(self, from_src, array_from, fill_value, fill_value_dtype);
}

PyObject *
TriMap_map_dst_fill(TriMapObject *self, PyObject *args) {
    PyArrayObject* array_from;
    PyObject* fill_value;
    PyArray_Descr* fill_value_dtype;
    if (!PyArg_ParseTuple(args,
            "O!OO!:map_dst_fill",
            &PyArray_Type, &array_from,
            &fill_value,
            &PyArrayDescr_Type, &fill_value_dtype
            )) {
        return NULL;
    }
    if (!self->finalized) {
        PyErr_SetString(PyExc_RuntimeError, "Finalization is required");
        return NULL;
    }
    bool from_src = false;
    return AK_TM_map_fill(self, from_src, array_from, fill_value, fill_value_dtype);
}



static PyMethodDef TriMap_methods[] = {
    {"register_one", (PyCFunction)TriMap_register_one, METH_VARARGS, NULL},
    {"register_unmatched_dst", (PyCFunction)TriMap_register_unmatched_dst, METH_NOARGS, NULL},
    {"register_many", (PyCFunction)TriMap_register_many, METH_VARARGS, NULL},
    {"finalize", (PyCFunction)TriMap_finalize, METH_NOARGS, NULL},
    {"is_many", (PyCFunction)TriMap_is_many, METH_NOARGS, NULL},
    {"src_no_fill", (PyCFunction)TriMap_src_no_fill, METH_NOARGS, NULL},
    {"dst_no_fill", (PyCFunction)TriMap_dst_no_fill, METH_NOARGS, NULL},
    {"map_src_no_fill", (PyCFunction)TriMap_map_src_no_fill, METH_O, NULL},
    {"map_dst_no_fill", (PyCFunction)TriMap_map_dst_no_fill, METH_O, NULL},
    {"map_src_fill", (PyCFunction)TriMap_map_src_fill, METH_VARARGS, NULL},
    {"map_dst_fill", (PyCFunction)TriMap_map_dst_fill, METH_VARARGS, NULL},
    {"map_merge", (PyCFunction)TriMap_map_merge, METH_VARARGS, NULL},
    {NULL},
};

PyTypeObject TriMapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(TriMapObject), // this does not get size of struct
    .tp_dealloc = (destructor)TriMap_dealloc,
    .tp_doc = TriMap_doc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = TriMap_methods,
    .tp_name = "arraykit.TriMap",
    .tp_new = TriMap_new,
    .tp_init = TriMap_init,
    .tp_repr = (reprfunc)TriMap_repr,
};
