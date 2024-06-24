# include "Python.h"
# include "structmember.h"

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"

# include "block_index.h"
# include "utilities.h"

PyObject * ErrorInitTypeBlocks;

// Returns NULL on error. Returns a new reference. Note that a reference is stolen from the PyObject argument.
static inline PyObject *
AK_build_pair_ssize_t_pyo(Py_ssize_t a, PyObject* py_b)
{
    if (py_b == NULL) { // construction failed
        return NULL;
    }
    PyObject* t = PyTuple_New(2);
    if (t == NULL) {
        return NULL;
    }
    PyObject* py_a = PyLong_FromSsize_t(a);
    if (py_a == NULL) {
        Py_DECREF(t);
        return NULL;
    }
    // steals refs
    PyTuple_SET_ITEM(t, 0, py_a);
    PyTuple_SET_ITEM(t, 1, py_b);
    return t;
}

// Given inclusive start, end indices, returns a new reference to a slice. Returns NULL on error. If `reduce` is True, single width slices return an integer.
static inline PyObject *
AK_build_slice_inclusive(Py_ssize_t start, Py_ssize_t end, bool reduce)
{
    if (reduce && start == end) {
        return PyLong_FromSsize_t(start); // new ref
    }
    // assert(start >= 0);
    if (start <= end) {
        return AK_build_slice(start, end + 1, 1);
    }
    // end of 0 goes to -1, gets converted to None
    return AK_build_slice(start, end - 1, -1);
}

// NOTE: we use platform size types here, which are appropriate for the values, but might pose issues if trying to pass pickles between 32 and 64 bit machines.
typedef struct BlockIndexRecord {
    Py_ssize_t block; // signed
    Py_ssize_t column;
} BlockIndexRecord;

typedef struct BlockIndexObject {
    PyObject_HEAD
    Py_ssize_t block_count;
    Py_ssize_t row_count;
    Py_ssize_t bir_count;
    Py_ssize_t bir_capacity;
    BlockIndexRecord* bir;
    PyArray_Descr* dtype;
    bool shape_recache;
    PyObject* shape;
} BlockIndexObject;

// Returns a new reference to tuple. Returns NULL on error. Python already wraps negative numbers up to negative length when used in the sequence slot
static inline PyObject *
AK_BI_item(BlockIndexObject* self, Py_ssize_t i) {
    if (!((size_t)i < (size_t)self->bir_count)) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }
    BlockIndexRecord* biri = &self->bir[i];
    return AK_build_pair_ssize_t(biri->block, biri->column); // may be NULL
}

//------------------------------------------------------------------------------
// BI Iterator

typedef struct BIIterObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    bool reversed;
    Py_ssize_t pos; // current index state, mutated in-place
} BIIterObject;

static inline PyObject *
BIIter_new(BlockIndexObject *bi, bool reversed) {
    BIIterObject *bii = PyObject_New(BIIterObject, &BIIterType);
    if (!bii) {
        return NULL;
    }
    Py_INCREF((PyObject*)bi);
    bii->bi = bi;
    bii->reversed = reversed;
    bii->pos = 0;
    return (PyObject *)bii;
}

void
BIIter_dealloc(BIIterObject *self) {
    Py_DECREF((PyObject*)self->bi);
    PyObject_Del((PyObject*)self);
}

PyObject *
BIIter_iter(BIIterObject *self) {
    Py_INCREF(self);
    return (PyObject*)self;
}

PyObject *
BIIter_iternext(BIIterObject *self) {
    Py_ssize_t i;
    if (self->reversed) {
        i = self->bi->bir_count - ++self->pos;
        if (i < 0) {
            return NULL;
        }
    }
    else {
        i = self->pos++;
    }
    if (self->bi->bir_count <= i) {
        return NULL;
    }
    return AK_BI_item(self->bi, i); // return new ref
}

PyObject *
BIIter_reversed(BIIterObject *self) {
    return BIIter_new(self->bi, !self->reversed);
}

PyObject *
BIIter_length_hint(BIIterObject *self) {
    // this works for reversed as we use self->pos to subtract from length
    Py_ssize_t len = Py_MAX(0, self->bi->bir_count - self->pos);
    return PyLong_FromSsize_t(len);
}

static PyMethodDef BIIter_methods[] = {
    {"__length_hint__", (PyCFunction)BIIter_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)BIIter_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterObject),
    .tp_dealloc = (destructor) BIIter_dealloc,
    .tp_iter = (getiterfunc) BIIter_iter,
    .tp_iternext = (iternextfunc) BIIter_iternext,
    .tp_methods = BIIter_methods,
    .tp_name = "arraykit.BlockIndexIterator",
};

//------------------------------------------------------------------------------
// BI Iterator sequence selection

typedef enum BIIterSelectorKind {
    BIIS_SEQ, // BIIterSeqType
    BIIS_SLICE,
    BIIS_BOOLEAN, // BIIterBoolType
    BIIS_UNKNOWN
} BIIterSelectorKind;

// Forward-def
static inline PyObject *
BIIterSelector_new(BlockIndexObject *bi,
        PyObject* selector,
        bool reversed,
        BIIterSelectorKind kind,
        bool ascending
        );

typedef struct BIIterSeqObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    bool reversed;
    PyObject* selector;
    Py_ssize_t pos; // current pos in sequence, mutated in-place
    Py_ssize_t len;
    bool is_array;
} BIIterSeqObject;

void
BIIterSeq_dealloc(BIIterSeqObject *self) {
    Py_DECREF((PyObject*)self->bi);
    Py_DECREF(self->selector);
    PyObject_Del((PyObject*)self);
}

PyObject *
BIIterSeq_iter(BIIterSeqObject *self) {
    Py_INCREF(self);
    return (PyObject*)self;
}

// Returns -1 on end of sequence; return -1 with exception set on
static inline Py_ssize_t
BIIterSeq_iternext_index(BIIterSeqObject *self)
{
    Py_ssize_t i;
    if (self->reversed) {
        i = self->len - ++self->pos;
        if (i < 0) {
            return -1;
        }
    }
    else {
        i = self->pos++;
    }
    if (self->len <= i) {
        return -1;
    }
    // use i to get index from selector
    Py_ssize_t t = 0;
    if (self->is_array) {
        PyArrayObject *a = (PyArrayObject *)self->selector;
        switch (PyArray_TYPE(a)) { // type of passed in array
            case NPY_INT64:
                t = (Py_ssize_t)*(npy_int64*)PyArray_GETPTR1(a, i);
                break;
            case NPY_INT32:
                t = *(npy_int32*)PyArray_GETPTR1(a, i);
                break;
            case NPY_INT16:
                t = *(npy_int16*)PyArray_GETPTR1(a, i);
                break;
            case NPY_INT8:
                t = *(npy_int8*)PyArray_GETPTR1(a, i);
                break;
            case NPY_UINT64:
                t = (Py_ssize_t)*(npy_uint64*)PyArray_GETPTR1(a, i);
                break;
            case NPY_UINT32:
                t = *(npy_uint32*)PyArray_GETPTR1(a, i);
                break;
            case NPY_UINT16:
                t = *(npy_uint16*)PyArray_GETPTR1(a, i);
                break;
            case NPY_UINT8:
                t = *(npy_uint8*)PyArray_GETPTR1(a, i);
                break;
        }
    }
    else { // is a list
        PyObject* o = PyList_GET_ITEM(self->selector, i); // borrow
        if (PyNumber_Check(o)) { // handles scalars
            t = PyNumber_AsSsize_t(o, NULL);
        }
        else {
            PyErr_SetString(PyExc_TypeError, "element type not suitable for indexing");
            return -1;
        }
    }
    if (t < 0) {
        t = self->bi->bir_count + t;
    }
    // we have to ensure valid range here to set an index error and distinguish from end of iteration
    if (!((size_t)t < (size_t)self->bi->bir_count)) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }
    return t;
}

PyObject *
BIIterSeq_iternext(BIIterSeqObject *self)
{
    Py_ssize_t i = BIIterSeq_iternext_index(self);
    if (i == -1) {
        return NULL; // an error is set
    }
    return AK_BI_item(self->bi, i); // return new ref
}

PyObject *
BIIterSeq_reversed(BIIterSeqObject *self)
{
    return BIIterSelector_new(self->bi, self->selector, !self->reversed, BIIS_SEQ, false);
}

PyObject *
BIIterSeq_length_hint(BIIterSeqObject *self)
{
    // this works for reversed as we use self-> index to subtract from length
    Py_ssize_t len = Py_MAX(0, self->len - self->pos);
    return PyLong_FromSsize_t(len);
}

static PyMethodDef BIiterSeq_methods[] = {
    {"__length_hint__", (PyCFunction)BIIterSeq_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)BIIterSeq_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterSeqType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterSeqObject),
    .tp_dealloc = (destructor) BIIterSeq_dealloc,
    .tp_iter = (getiterfunc) BIIterSeq_iter,
    .tp_iternext = (iternextfunc) BIIterSeq_iternext,
    .tp_methods = BIiterSeq_methods,
    .tp_name = "arraykit.BlockIndexIteratorSequence",
};

//------------------------------------------------------------------------------
// BI Iterator slice selection

typedef struct BIIterSliceObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    bool reversed;
    PyObject* selector; // slice
    Py_ssize_t count; // count of , mutated in-place
    // these are the normalized values truncated to the span of the bir_count; len is the realized length after extraction; step is always set to 1 if missing; len is 0 if no realized values
    Py_ssize_t pos;
    Py_ssize_t step;
    Py_ssize_t len;
} BIIterSliceObject;

void
BIIterSlice_dealloc(BIIterSliceObject *self) {
    // AK_DEBUG_MSG_REFCNT("start dealloc", self);
    Py_DECREF((PyObject*)self->bi);
    Py_DECREF(self->selector);
    PyObject_Del((PyObject*)self);
}

PyObject *
BIIterSlice_iter(BIIterSliceObject *self) {
    Py_INCREF(self);
    return (PyObject*)self;
}

// NOTE: this does not use `reversed`, as pos, step, and count are set in BIIterSelector_new
static inline Py_ssize_t
BIIterSlice_iternext_index(BIIterSliceObject *self)
{
    if (self->len == 0 || self->count >= self->len) {
        return -1;
    }
    Py_ssize_t i = self->pos;
    self->pos += self->step;
    self->count++; // by counting index we we do not need to compare to stop
    // i will never be out of range
    return i;
}

PyObject *
BIIterSlice_iternext(BIIterSliceObject *self) {
    Py_ssize_t i = BIIterSlice_iternext_index(self);
    if (i == -1) {
        return NULL;
    }
    return AK_BI_item(self->bi, i); // return new ref
}

PyObject *
BIIterSlice_reversed(BIIterSliceObject *self)
{
    return BIIterSelector_new(self->bi, self->selector, !self->reversed, BIIS_SLICE, false);
}

PyObject *
BIIterSlice_length_hint(BIIterSliceObject *self)
{
    // this works for reversed as we use self-> index to subtract from length
    Py_ssize_t len = Py_MAX(0, self->len - self->count);
    return PyLong_FromSsize_t(len);
}

static PyMethodDef BIiterSlice_methods[] = {
    {"__length_hint__", (PyCFunction)BIIterSlice_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)BIIterSlice_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterSliceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterSliceObject),
    .tp_dealloc = (destructor) BIIterSlice_dealloc,
    .tp_iter = (getiterfunc) BIIterSlice_iter,
    .tp_iternext = (iternextfunc) BIIterSlice_iternext,
    .tp_methods = BIiterSlice_methods,
    .tp_name = "arraykit.BlockIndexIteratorSlice",
};

//------------------------------------------------------------------------------
// BI Iterator Boolean array selection

typedef struct BIIterBooleanObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    bool reversed;
    PyObject* selector;
    Py_ssize_t pos; // current index, mutated in-place
    Py_ssize_t len;
} BIIterBooleanObject;

void
BIIterBoolean_dealloc(BIIterBooleanObject *self) {
    Py_DECREF((PyObject*)self->bi);
    Py_DECREF(self->selector);
    PyObject_Del((PyObject*)self);
}

PyObject *
BIIterBoolean_iter(BIIterBooleanObject *self)
{
    Py_INCREF(self);
    return (PyObject*)self;
}

static inline Py_ssize_t
BIIterBoolean_iternext_index(BIIterBooleanObject *self)
{
    npy_bool v = 0;
    Py_ssize_t i = -1;
    PyArrayObject* a = (PyArrayObject*) self->selector;

    if (!self->reversed) {
        while (self->pos < self->len) {
            v = *(npy_bool*)PyArray_GETPTR1(a, self->pos);
            if (v) {
                i = self->pos;
                self->pos++;
                break;
            }
            self->pos++;
        }
    }
    else { // reversed
        while (self->pos >= 0) {
            v = *(npy_bool*)PyArray_GETPTR1(a, self->pos);
            if (v) {
                i = self->pos;
                self->pos--;
                break;
            }
            self->pos--;
        }
    }
    if (i != -1) {
        return i;
    }
    return -1; // no True remain
}

PyObject *
BIIterBoolean_iternext(BIIterBooleanObject *self) {
    Py_ssize_t i = BIIterBoolean_iternext_index(self);
    if (i == -1) {
        return NULL;
    }
    return AK_BI_item(self->bi, i); // return new ref
}

PyObject *
BIIterBoolean_reversed(BIIterBooleanObject *self)
{
    return BIIterSelector_new(self->bi, self->selector, !self->reversed, BIIS_BOOLEAN, 0);
}

// NOTE: no length hint given as we would have to traverse whole array and count True... not sure it is worht it.
static PyMethodDef BIiterBoolean_methods[] = {
    {"__reversed__", (PyCFunction)BIIterBoolean_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterBoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterBooleanObject),
    .tp_dealloc = (destructor) BIIterBoolean_dealloc,
    .tp_iter = (getiterfunc) BIIterBoolean_iter,
    .tp_iternext = (iternextfunc) BIIterBoolean_iternext,
    .tp_methods = BIiterBoolean_methods,
    .tp_name = "arraykit.BlockIndexIteratorBoolean",
};

//------------------------------------------------------------------------------
// BI Iterator Contigous

typedef struct BIIterContiguousObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    PyObject* iter; // own reference to core iterator
    bool reversed;
    Py_ssize_t last_block;
    Py_ssize_t last_column;
    Py_ssize_t next_block;
    Py_ssize_t next_column;
    bool reduce; // optionally reduce slices to integers
} BIIterContiguousObject;

// Create a new contiguous slice iterator. Return NULL on error. Steals a reference to PyObject* iter.
static inline PyObject *
BIIterContiguous_new(BlockIndexObject *bi,
        bool reversed,
        PyObject* iter,
        bool reduce)
{
    BIIterContiguousObject *bii = PyObject_New(BIIterContiguousObject, &BIIterContiguousType);
    if (!bii) {
        return NULL;
    }
    Py_INCREF((PyObject*)bi);
    bii->bi = bi;

    bii->iter = iter; // steals ref
    bii->reversed = reversed;

    bii->last_block = -1;
    bii->last_column = -1;
    bii->next_block = -1;
    bii->next_column = -1;
    bii->reduce = reduce;

    return (PyObject *)bii;
}

void
BIIterContiguous_dealloc(BIIterContiguousObject *self)
{
    Py_DECREF((PyObject*)self->bi);
    Py_DECREF(self->iter);
    PyObject_Del((PyObject*)self);
}

// Simply incref this object and return it.
PyObject *
BIIterContiguous_iter(BIIterContiguousObject *self)
{
    Py_INCREF(self);
    return (PyObject*)self;
}

// Returns a new reference.
PyObject *
BIIterContiguous_reversed(BIIterContiguousObject *self)
{
    bool reversed = !self->reversed;

    PyObject* selector = NULL;
    PyTypeObject* type = Py_TYPE(self->iter);
    if (type == &BIIterSeqType) {
        selector = ((BIIterSeqObject*)self->iter)->selector;
    }
    else if (type == &BIIterSliceType) {
        selector = ((BIIterSliceObject*)self->iter)->selector;
    }
    else if (type == &BIIterBoolType) {
        selector = ((BIIterBooleanObject*)self->iter)->selector;
    }
    if (selector == NULL) {
        return NULL;
    }

    PyObject* iter = BIIterSelector_new(self->bi,
            selector,
            reversed,
            BIIS_UNKNOWN, // let type be determined by selector
            0);
    if (iter == NULL) {
        return NULL;
    }
    PyObject* biiter = BIIterContiguous_new(self->bi,
            reversed,
            iter, // steals ref
            self->reduce);
    return biiter;
}

PyObject *
BIIterContiguous_iternext(BIIterContiguousObject *self)
{
    Py_ssize_t i = -1;
    PyObject* iter = self->iter;
    PyTypeObject* type = Py_TYPE(iter);

    Py_ssize_t slice_start = -1;
    Py_ssize_t block;
    Py_ssize_t column;

    while (1) {
        if (self->next_block == -2) {
            break; // terminate
        }
        if (self->next_block != -1) {
            // discontinuity found on last iteration, set new start
            self->last_block = self->next_block;
            self->last_column = slice_start = self->next_column;
            self->next_block = self->next_column = -1; // clear next state
        }
        if (type == &BIIterSeqType) {
            i = BIIterSeq_iternext_index((BIIterSeqObject*)iter);
        }
        else if (type == &BIIterSliceType) {
            i = BIIterSlice_iternext_index((BIIterSliceObject*)iter);
        }
        else if (type == &BIIterBoolType) {
            i = BIIterBoolean_iternext_index((BIIterBooleanObject*)iter);
        }
        if (i == -1) { // end of iteration or error
            if (PyErr_Occurred()) {
                break;
            }
            // no more pairs, return previous slice_start, flag for end on next call
            self->next_block = -2;
            if (self->last_block == -1) { // iter produced no values, terminate
                break;
            }
            return AK_build_pair_ssize_t_pyo( // steals ref
                    self->last_block,
                    AK_build_slice_inclusive(slice_start,
                            self->last_column,
                            self->reduce));
        }
        // i is gauranteed to be within the range of self->bit_count at this point; the only source of arbitrary indices is in BIIterSeq_iternext_index, and that function validates the range
        BlockIndexRecord* biri = &self->bi->bir[i];
        block = biri->block;
        column = biri->column;

        // inititialization
        if (self->last_block == -1) {
            self->last_block = block;
            self->last_column = column;
            slice_start = column;
            continue;
        }
        if (self->last_block == block && llabs(column - self->last_column) == 1) {
            // contiguious region found, can be postive or negative
            self->last_column = column;
            continue;
        }
        self->next_block = block;
        self->next_column = column;
        return AK_build_pair_ssize_t_pyo( // steals ref
                self->last_block,
                AK_build_slice_inclusive(slice_start,
                        self->last_column,
                        self->reduce));
    }
    return NULL;
}

// not implementing __length_hint__
static PyMethodDef BIIterContiguous_methods[] = {
    {"__reversed__", (PyCFunction)BIIterContiguous_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterContiguousType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterContiguousObject),
    .tp_dealloc = (destructor) BIIterContiguous_dealloc,
    .tp_iter = (getiterfunc) BIIterContiguous_iter,
    .tp_iternext = (iternextfunc) BIIterContiguous_iternext,
    .tp_methods = BIIterContiguous_methods,
    .tp_name = "arraykit.BlockIndexContiguousIterator",
};

//------------------------------------------------------------------------------
// BI Iterator Block Slice

typedef struct BIIterBlockObject {
    PyObject_HEAD
    BlockIndexObject *bi;
    bool reversed;
    Py_ssize_t pos; // current index state, mutated in-place
    PyObject* null_slice;
} BIIterBlockObject;

static inline PyObject *
BIIterBlock_new(BlockIndexObject *bi, bool reversed) {
    BIIterBlockObject *bii = PyObject_New(BIIterBlockObject, &BIIterBlockType);
    if (!bii) {
        return NULL;
    }
    Py_INCREF((PyObject*)bi);
    bii->bi = bi;
    bii->reversed = reversed;
    bii->pos = 0;

    // create a new ref of the null slice
    PyObject* ns = AK_build_slice(-1, -1, 1); // get all null; new ref
    if (ns == NULL) {
        return NULL;
    }
    bii->null_slice = ns;
    return (PyObject *)bii;
}

void
BIIterBlock_dealloc(BIIterBlockObject *self) {
    Py_DECREF((PyObject*)self->bi);
    Py_DECREF(self->null_slice);
    PyObject_Del((PyObject*)self);
}

PyObject *
BIIterBlock_iter(BIIterBlockObject *self) {
    Py_INCREF(self);
    return (PyObject*)self;
}

PyObject *
BIIterBlock_iternext(BIIterBlockObject *self) {
    Py_ssize_t i;
    if (self->reversed) {
        i = self->bi->block_count - ++self->pos;
        if (i < 0) {
            return NULL;
        }
    }
    else {
        i = self->pos++;
    }
    if (self->bi->block_count <= i) {
        return NULL;
    }
    // AK_build_pair_ssize_t_pyo steals the reference to the object; so incref here
    Py_INCREF(self->null_slice);
    PyObject* t = AK_build_pair_ssize_t_pyo(i, self->null_slice); // return new ref
    if (t == NULL) {
        // if tuple creation failed need to undo incref
        Py_DECREF(self->null_slice);
    }
    return t;
}

PyObject *
BIIterBlock_reversed(BIIterBlockObject *self) {
    return BIIterBlock_new(self->bi, !self->reversed);
}

PyObject *
BIIterBlock_length_hint(BIIterBlockObject *self) {
    // this works for reversed as we use self->pos to subtract from length
    Py_ssize_t len = Py_MAX(0, self->bi->block_count - self->pos);
    return PyLong_FromSsize_t(len);
}

static PyMethodDef BIIterBlock_methods[] = {
    {"__length_hint__", (PyCFunction)BIIterBlock_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)BIIterBlock_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BIIterBlockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(BIIterBlockObject),
    .tp_dealloc = (destructor) BIIterBlock_dealloc,
    .tp_iter = (getiterfunc) BIIterBlock_iter,
    .tp_iternext = (iternextfunc) BIIterBlock_iternext,
    .tp_methods = BIIterBlock_methods,
    .tp_name = "arraykit.BlockIndexBlockIterator",
};

//------------------------------------------------------------------------------

// NOTE: this constructor returns one of three different PyObject types. We do this to consolidate error reporting and type checks.
// The ascending argument is applied before consideration of a reverse iterator
PyObject *
BIIterSelector_new(BlockIndexObject *bi,
        PyObject* selector,
        bool reversed,
        BIIterSelectorKind kind,
        bool ascending) {

    bool is_array = false;
    bool incref_selector = true; // incref borrowed selector; but if a new ref is made, do not

    Py_ssize_t len = -1;
    Py_ssize_t pos = 0;
    Py_ssize_t stop = 0;
    Py_ssize_t step = 0;

    if (PyArray_Check(selector)) {
        if (kind == BIIS_SLICE) {
            PyErr_SetString(PyExc_TypeError, "Arrays cannot be used as selectors for slice iterators");
            return NULL;
        }
        is_array = true;
        PyArrayObject *a = (PyArrayObject *)selector;
        if (PyArray_NDIM(a) != 1) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be 1-dimensional");
            return NULL;
        }
        len = PyArray_SIZE(a);

        char k = PyArray_DESCR(a)->kind;
        if (kind == BIIS_UNKNOWN) {
            if (k == 'i' || k == 'u') {
                kind = BIIS_SEQ;
            }
            else if (k == 'b') {
                kind = BIIS_BOOLEAN;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "Arrays kind not supported");
                return NULL;
            }
        }
        else if (kind == BIIS_SEQ && k != 'i' && k != 'u') {
            PyErr_SetString(PyExc_TypeError, "Arrays must be integer kind");
            return NULL;
        }
        else if (kind == BIIS_BOOLEAN && k != 'b') {
            PyErr_SetString(PyExc_TypeError, "Arrays must be Boolean kind");
            return NULL;
        }

        if (kind == BIIS_BOOLEAN) {
            if (len != bi->bir_count) {
                PyErr_SetString(PyExc_TypeError, "Boolean arrays must match BlockIndex size");
                return NULL;
            }
        }
        else if (ascending) { // not Boolean
            // NOTE: we can overwrite selector here as we have a borrowed refernce; sorting gives us a new reference, so we do not need to incref below
            selector = PyArray_NewCopy(a, NPY_CORDER);
            // sort in-place; can use a non-stable sort
            if (PyArray_Sort((PyArrayObject*)selector, 0, NPY_QUICKSORT)) {
                return NULL; // returns -1 on error
            }; // new ref
            incref_selector = false;
        }
    }
    else if (PySlice_Check(selector)) {
        if (kind == BIIS_UNKNOWN) {
            kind = BIIS_SLICE;
        }
        else if (kind != BIIS_SLICE) {
            PyErr_SetString(PyExc_TypeError, "Slices cannot be used as selectors for this type of iterator");
            return NULL;
        }

        if (ascending) {
            // NOTE: we are abandoning the borrowed reference
            selector = AK_slice_to_ascending_slice(selector, bi->bir_count); // new ref
            incref_selector = false;
        }
        if (PySlice_Unpack(selector, &pos, &stop, &step)) {
            return NULL;
        }
        len = PySlice_AdjustIndices(bi->bir_count, &pos, &stop, step);

        if (reversed) {
            pos += (step * (len - 1));
            step *= -1;
        }
    }
    else if (PyList_CheckExact(selector)) {
        if (kind == BIIS_UNKNOWN) {
            kind = BIIS_SEQ;
        }
        else if (kind != BIIS_SEQ) {
            PyErr_SetString(PyExc_TypeError, "Lists cannot be used as for non-sequence iterators");
            return NULL;
        }
        len = PyList_GET_SIZE(selector);

        if (ascending) {
            // abandoning borrowed ref
            selector = PyObject_CallMethod(selector, "copy", NULL); // new ref
            if (selector == NULL) {
                return NULL;
            }
            PyObject* post = PyObject_CallMethod(selector, "sort", NULL); // new ref
            if (post == NULL) {
                return NULL;
            }
            Py_DECREF(post); // just a None
            incref_selector = false;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Input type not supported");
        return NULL;
    }

    PyObject *bii = NULL;
    switch (kind) {
        case BIIS_SEQ: {
            BIIterSeqObject* it = PyObject_New(BIIterSeqObject, &BIIterSeqType);
            if (it == NULL) {goto error;}
            it->bi = bi;
            it->selector = selector;
            it->reversed = reversed;
            it->len = len;
            it->pos = 0;
            it->is_array = is_array;
            bii = (PyObject*)it;
            break;
        }
        case BIIS_SLICE: {
            BIIterSliceObject* it = PyObject_New(BIIterSliceObject, &BIIterSliceType);
            if (it == NULL) {goto error;}
            it->bi = bi;
            it->selector = selector;
            it->reversed = reversed;
            it->len = len;
            it->pos = pos;
            it->step = step;
            it->count = 0;
            bii = (PyObject*)it;
            break;
        }
        case BIIS_BOOLEAN: {
            BIIterBooleanObject* it = PyObject_New(BIIterBooleanObject, &BIIterBoolType);
            if (it == NULL) {goto error;}
            it->bi = bi;
            it->selector = selector;
            it->reversed = reversed;
            it->len = len;
            it->pos = reversed ? len - 1 : 0;
            bii = (PyObject*)it;
            break;
        }
        case BIIS_UNKNOWN:
            goto error; // should not get here!
    }
    Py_INCREF((PyObject*)bi);

    if (incref_selector) {
        Py_INCREF(selector);
    }
    return bii;
error: // nothing shold be increfed when we get here
    return NULL;
}

//------------------------------------------------------------------------------
// block index new, init, memory

// Returns 0 on succes, -1 on error.
static inline int
AK_BI_BIR_new(BlockIndexObject* bi) {
    BlockIndexRecord* bir = (BlockIndexRecord*)PyMem_Malloc(
            sizeof(BlockIndexRecord) * bi->bir_capacity);
    if (bir == NULL) {
        PyErr_SetNone(PyExc_MemoryError);
        return -1;
    }
    bi->bir = bir;
    return 0;
}

// Returns 0 on success, -1 on error
static inline int
AK_BI_BIR_resize(BlockIndexObject* bi, Py_ssize_t increment) {
    Py_ssize_t target = bi->bir_count + increment;
    Py_ssize_t capacity = bi->bir_capacity;
    if (AK_UNLIKELY(target >= capacity)) {
        while (capacity < target) {
            capacity <<= 1; // get 2x the capacity
        }
        bi->bir = PyMem_Realloc(bi->bir,
                sizeof(BlockIndexRecord) * capacity);
        if (bi->bir == NULL) {
            PyErr_SetNone(PyExc_MemoryError);
            return -1;
        }
        bi->bir_capacity = capacity;
    }
    return 0;
}

PyDoc_STRVAR(
    BlockIndex_doc,
    "\n"
    "A grow only, reference lookup of realized columns to block, block columns."
);

PyObject *
BlockIndex_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs) {
    BlockIndexObject *self = (BlockIndexObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }
    return (PyObject *)self;
}

// Returns 0 on success, -1 on error.
int
BlockIndex_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    BlockIndexObject* bi = (BlockIndexObject*)self;

    Py_ssize_t block_count = 0;
    Py_ssize_t row_count = -1; // mark as unset
    Py_ssize_t bir_count = 0;
    Py_ssize_t bir_capacity = 8;
    PyObject* bir_bytes = NULL;
    PyObject* dtype = NULL;

    if (!PyArg_ParseTuple(args,
            "|nnnnO!O:__init__",
            &block_count,
            &row_count,
            &bir_count,
            &bir_capacity,
            &PyBytes_Type, &bir_bytes,
            &dtype)) {
        return -1;
    }
    if (bir_count > bir_capacity) {
        PyErr_SetString(PyExc_ValueError, "record count exceeds capacity");
        return -1;
    }
    // handle all Py_ssize_t
    bi->block_count = block_count;
    bi->row_count = row_count;
    bi->bir_count = bir_count;
    bi->bir_capacity = bir_capacity;

    bi->shape_recache = true; // always init to true
    bi->shape = NULL;

    // Load the bi->bir struct array, if defined
    bi->bir = NULL;
    // always set bi to capacity defined at this point
    if (AK_BI_BIR_new(bi)) {
        return -1;
    }
    if (bir_bytes != NULL) {
        // already know bir is a bytes object
        char* data = PyBytes_AS_STRING(bir_bytes);
        memcpy(bi->bir, data, bi->bir_count * sizeof(BlockIndexRecord));
        // bir_bytes is a borrowed ref
    }

    bi->dtype = NULL;
    if (dtype != NULL && dtype != Py_None) {
        if (PyArray_DescrCheck(dtype)) {
            Py_INCREF(dtype);
            bi->dtype = (PyArray_Descr*)dtype;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "dtype argument must be a dtype");
            return -1;
        }
    }
    return 0;
}

void
BlockIndex_dealloc(BlockIndexObject *self) {
    if (self->bir != NULL) {
        PyMem_Free(self->bir);
    }
    // both dtype and shape might not be set
    Py_XDECREF((PyObject*)self->dtype);
    Py_XDECREF(self->shape);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

//------------------------------------------------------------------------------

// Returns NULL on error, True if the block should be reatained, False if the block has zero columns and should not be retained. This checks and raises on non-array inputs, dimensions other than 1 or 2, and mis-aligned columns.
PyObject *
BlockIndex_register(BlockIndexObject *self, PyObject *value) {
    if (!PyArray_Check(value)) {
        PyErr_Format(ErrorInitTypeBlocks, "Found non-array block: %R", value);
        return NULL;
    }
    PyArrayObject *a = (PyArrayObject *)value;
    int ndim = PyArray_NDIM(a);

    if (ndim < 1 || ndim > 2) {
        PyErr_Format(ErrorInitTypeBlocks, "Array block has invalid dimensions: %i", ndim);
        return NULL;
    }
    Py_ssize_t increment = ndim == 1 ? 1 : PyArray_DIM(a, 1);

    // assign alignment on first observation; otherwise force alignemnt. We do this regardless of if the array has no columns.
    Py_ssize_t alignment = PyArray_DIM(a, 0);
    if (self->row_count == -1) {
        self->row_count = alignment;
        self->shape_recache = true; // setting rows, must recache shape
    }
    else if (self->row_count != alignment) {
        PyErr_Format(ErrorInitTypeBlocks,
                "Array block has unaligned row count: found %i, expected %i",
                alignment,
                self->row_count);
        return NULL;
    }
    // if we are not adding columns, we are not adding types, so we are not changing the  dtype or shape
    if (increment == 0) {
        Py_RETURN_FALSE;
    }

    PyArray_Descr* dt = PyArray_DESCR(a); // borrowed ref
    self->shape_recache = true; // adjusting columns, must recache shape

    if (self->dtype == NULL) { // if not already set
        Py_INCREF((PyObject*)dt);
        self->dtype = dt;
    }
    else if (!PyDataType_ISOBJECT(self->dtype)) { // if object cannot resolve further
        PyArray_Descr* dtr = AK_resolve_dtype(self->dtype, dt); // new ref
        if (dtr == NULL) {
            return NULL;
        }
        Py_DECREF((PyObject*)self->dtype);
        self->dtype = dtr;
    }

    // create space for increment new records
    if (AK_BI_BIR_resize(self, increment)) {
        return NULL;
    };

    // pull out references
    BlockIndexRecord* bir = self->bir;
    Py_ssize_t bc = self->block_count;
    Py_ssize_t birc = self->bir_count;
    for (Py_ssize_t i = 0; i < increment; i++) {
        bir[birc] = (BlockIndexRecord){bc, i};
        birc++;
    }
    self->bir_count = birc;
    self->block_count++;
    Py_RETURN_TRUE;
}

//------------------------------------------------------------------------------
// exporters

PyObject *
BlockIndex_to_list(BlockIndexObject *self, PyObject *Py_UNUSED(unused)) {
    PyObject* list = PyList_New(self->bir_count);
    if (list == NULL) {
        return NULL;
    }
    BlockIndexRecord* bir = self->bir;

    for (Py_ssize_t i = 0; i < self->bir_count; i++) {
        PyObject* item = AK_build_pair_ssize_t(bir[i].block, bir[i].column);
        if (item == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        // set_item steals reference
        PyList_SET_ITEM(list, i, item);
    }
    return list;
}

// Returns NULL on error
static inline PyObject *
AK_BI_to_bytes(BlockIndexObject *self) {
    Py_ssize_t size = self->bir_count * sizeof(BlockIndexRecord);
    // bytes might be null on error
    PyObject* bytes = PyBytes_FromStringAndSize((const char*)self->bir, size);
    return bytes;
}

// Returns NULL on error
PyObject *
BlockIndex_to_bytes(BlockIndexObject *self, PyObject *Py_UNUSED(unused)) {
    return AK_BI_to_bytes(self);
}

//------------------------------------------------------------------------------
// pickle support

// Returns NULL on error, PyObject* otherwise.
PyObject *
BlockIndex_getstate(BlockIndexObject *self) {
    PyObject* bi = AK_BI_to_bytes(self);
    if (bi == NULL) {
        return NULL;
    }
    PyObject* dt = self->dtype == NULL ? Py_None : (PyObject*) self->dtype;
    // state might be NULL on failure; assume exception set
    PyObject* state = Py_BuildValue("nnnnNO", // use N to steal ref of bytes
            self->block_count,
            self->row_count,
            self->bir_count,
            self->bir_capacity,
            bi,  // stolen new ref
            dt); // increfs passed object
    return state;
}

// State returned here is a tuple of keys, suitable for usage as an `args` argument.
PyObject *
BlockIndex_setstate(BlockIndexObject *self, PyObject *state)
{
    if (!PyTuple_CheckExact(state) || !PyTuple_GET_SIZE(state)) {
        PyErr_SetString(PyExc_ValueError, "Unexpected pickled object.");
        return NULL;
    }
    BlockIndex_init((PyObject*)self, state, NULL);
    Py_RETURN_NONE;
}

//------------------------------------------------------------------------------
// getters

// In a shape tuple, rows will never be negative.
PyObject *
BlockIndex_shape_getter(BlockIndexObject *self, void* Py_UNUSED(closure))
{
    if (self->shape == NULL || self->shape_recache) {
        Py_XDECREF(self->shape); // get rid of old if it exists
        self->shape = AK_build_pair_ssize_t(
                self->row_count < 0 ? 0 : self->row_count,
                self->bir_count);
    }
    // shape is not null and shape_recache is false
    Py_INCREF(self->shape); // for caller
    self->shape_recache = false;
    return self->shape;
}

// Unset rows will be -1.
PyObject *
BlockIndex_rows_getter(BlockIndexObject *self, void* Py_UNUSED(closure)){
    return PyLong_FromSsize_t(self->row_count);
}

PyObject *
BlockIndex_columns_getter(BlockIndexObject *self, void* Py_UNUSED(closure)){
    return PyLong_FromSsize_t(self->bir_count);
}

// Return the resolved dtype for all registered blocks. If no block have been registered, this will return a float dtype.
PyObject *
BlockIndex_dtype_getter(BlockIndexObject *self, void* Py_UNUSED(closure)){
    if (self->dtype != NULL) {
        Py_INCREF(self->dtype);
        return (PyObject*)self->dtype;
    }
    // NOTE: could use NPY_DEFAULT_TYPE here; SF defines this explicitly as float64
    return (PyObject*)PyArray_DescrFromType(NPY_FLOAT64);
}

static struct PyGetSetDef BlockIndex_getset[] = {
    {"shape", (getter)BlockIndex_shape_getter, NULL, NULL, NULL},
    {"rows", (getter)BlockIndex_rows_getter, NULL, NULL, NULL},
    {"columns", (getter)BlockIndex_columns_getter, NULL, NULL, NULL},
    {"dtype", (getter)BlockIndex_dtype_getter, NULL, NULL, NULL},
    {NULL},
};

//------------------------------------------------------------------------------
// general methods

PyObject *
BlockIndex_repr(BlockIndexObject *self) {
    PyObject* dt = self->dtype == NULL ? Py_None : (PyObject*) self->dtype;
    return PyUnicode_FromFormat("<%s(blocks: %i, rows: %i, columns: %i, dtype: %R)>",
            Py_TYPE(self)->tp_name,
            self->block_count,
            self->row_count,
            self->bir_count,
            dt);
}

PyObject *
BlockIndex_copy(BlockIndexObject *self, PyObject *Py_UNUSED(unused))
{
    PyTypeObject* cls = Py_TYPE(self); // borrowed ref
    BlockIndexObject *bi = (BlockIndexObject *)cls->tp_alloc(cls, 0);
    if (bi == NULL) {
        return NULL;
    }
    bi->block_count = self->block_count;
    bi->row_count = self->row_count;
    bi->bir_count = self->bir_count;
    bi->bir_capacity = self->bir_capacity;

    bi->shape_recache = true; // could copy, but do not want to copy a pending cache state
    bi->shape = NULL;

    bi->bir = NULL;
    AK_BI_BIR_new(bi); // do initial alloc to self->bir_capacity
    memcpy(bi->bir,
            self->bir,
            self->bir_count * sizeof(BlockIndexRecord));

    bi->dtype = NULL;
    if (self->dtype != NULL) {
        bi->dtype = self->dtype;
        Py_INCREF((PyObject*)bi->dtype);
    }
    return (PyObject *)bi;
}

Py_ssize_t
BlockIndex_length(BlockIndexObject *self){
    return self->bir_count;
}

PyObject *
BlockIndex_sizeof(BlockIndexObject *self) {
    return PyLong_FromSsize_t(
        Py_TYPE(self)->tp_basicsize
        + (self->bir_capacity) * sizeof(BlockIndexRecord)
    );
}

// Given an index, return just the block index.
PyObject *
BlockIndex_get_block(BlockIndexObject *self, PyObject *key){
    if (PyNumber_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, NULL);
        if (!((size_t)i < (size_t)self->bir_count)) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
        return PyLong_FromSsize_t(self->bir[i].block); // maybe NULL, exception will be set
    }
    PyErr_SetString(PyExc_TypeError, "An integer is required.");
    return NULL;
}

// Given an index, return just the column index.
PyObject *
BlockIndex_get_column(BlockIndexObject *self, PyObject *key){
    if (PyNumber_Check(key)) {
        Py_ssize_t i = PyNumber_AsSsize_t(key, NULL);
        if (!((size_t)i < (size_t)self->bir_count)) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
        return PyLong_FromSsize_t(self->bir[i].column); // maybe NULL, exception will be set
    }
    PyErr_SetString(PyExc_TypeError, "An integer is required.");
    return NULL;
}

//------------------------------------------------------------------------------
// iterators

PyObject *
BlockIndex_iter(BlockIndexObject* self) {
    return BIIter_new(self, false);
}

PyObject *
BlockIndex_reversed(BlockIndexObject* self) {
    return BIIter_new(self, true);
}

// Given key, return an iterator of a selection.
PyObject *
BlockIndex_iter_select(BlockIndexObject *self, PyObject *selector){
    return BIIterSelector_new(self, selector, false, BIIS_UNKNOWN, false);
}

static char *iter_contiguous_kargs_names[] = {
    "selector",
    "ascending",
    "reduce",
    NULL
};

// Given key, return an iterator of a selection.
PyObject *
BlockIndex_iter_contiguous(BlockIndexObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject* selector;
    int ascending = 0; // must be int for parsing to "p"
    int reduce = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|$pp:iter_contiguous",
            iter_contiguous_kargs_names,
            &selector,
            &ascending,
            &reduce
            )) {
        return NULL;
    }
    PyObject* iter = BIIterSelector_new(self, selector, false, BIIS_UNKNOWN, ascending);
    if (iter == NULL) {
        return NULL; // exception set
    }
    PyObject* biiter = BIIterContiguous_new(self, false, iter, reduce); // might be NULL, steals iter ref
    return biiter;
}

// Given key, return an iterator of a selection.
PyObject *
BlockIndex_iter_block(BlockIndexObject *self){
    return BIIterBlock_new(self, false);
}

//------------------------------------------------------------------------------
// slot / method def

static PySequenceMethods BlockIndex_as_sequece = {
    .sq_length = (lenfunc)BlockIndex_length,
    .sq_item = (ssizeargfunc)AK_BI_item,
};

static PyMethodDef BlockIndex_methods[] = {
    // {"__getitem__", (PyCFunction)BlockIndex_subscript, METH_O, NULL},
    {"register", (PyCFunction)BlockIndex_register, METH_O, NULL},
    {"__getstate__", (PyCFunction) BlockIndex_getstate, METH_NOARGS, NULL},
    {"__setstate__", (PyCFunction) BlockIndex_setstate, METH_O, NULL},
    {"__sizeof__", (PyCFunction) BlockIndex_sizeof, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) BlockIndex_reversed, METH_NOARGS, NULL},
    {"to_list", (PyCFunction)BlockIndex_to_list, METH_NOARGS, NULL},
    {"to_bytes", (PyCFunction)BlockIndex_to_bytes, METH_NOARGS, NULL},
    {"copy", (PyCFunction)BlockIndex_copy, METH_NOARGS, NULL},
    {"get_block", (PyCFunction) BlockIndex_get_block, METH_O, NULL},
    {"get_column", (PyCFunction) BlockIndex_get_column, METH_O, NULL},
    {"iter_select", (PyCFunction) BlockIndex_iter_select, METH_O, NULL},
    {"iter_contiguous",
            (PyCFunction) BlockIndex_iter_contiguous,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"iter_block", (PyCFunction) BlockIndex_iter_block, METH_NOARGS, NULL},
    // {"__getnewargs__", (PyCFunction)BlockIndex_getnewargs, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject BlockIndexType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    // .tp_as_mapping = &BlockIndex_as_mapping,
    .tp_as_sequence = &BlockIndex_as_sequece,
    .tp_basicsize = sizeof(BlockIndexObject),
    .tp_dealloc = (destructor)BlockIndex_dealloc,
    .tp_doc = BlockIndex_doc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = BlockIndex_getset,
    .tp_iter = (getiterfunc)BlockIndex_iter,
    .tp_methods = BlockIndex_methods,
    .tp_name = "arraykit.BlockIndex",
    .tp_new = BlockIndex_new,
    .tp_init = BlockIndex_init,
    .tp_repr = (reprfunc) BlockIndex_repr,
    // .tp_traverse = (traverseproc)BlockIndex_traverse,
};
