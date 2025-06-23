// For background on the hashtable design first implemented in AutoMap, see the following:
// https://github.com/brandtbucher/automap/blob/b787199d38d6bfa1b55484e5ea1e89b31cc1fa72/automap.c#L12
# include <math.h>
# include "Python.h"
# include "stdbool.h"

# define PY_SSIZE_T_CLEAN

# define NO_IMPORT_ARRAY
# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"
# include "auto_map.h"
# include "utilities.h"

//------------------------------------------------------------------------------
// Common

PyObject *NonUniqueError;

// The main storage "table" is an array of TableElement
typedef struct TableElement{
    Py_ssize_t keys_pos;
    Py_hash_t hash;
} TableElement;

// Table configuration; experimentation shows that these values work well:
# define LOAD 0.9
# define SCAN 16


// Partial, two-argument version of PyUnicode_FromKindAndData for consistent templating with bytes version.
static inline PyObject*
PyUnicode_FromUCS4AndData(const void *buffer, Py_ssize_t size) {
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, buffer, size);
}

typedef enum KeysArrayType{
    KAT_LIST = 0, // must be falsy

    KAT_INT8, // order matters as ranges of size are used in selection
    KAT_INT16,
    KAT_INT32,
    KAT_INT64,

    KAT_UINT8,
    KAT_UINT16,
    KAT_UINT32,
    KAT_UINT64,

    KAT_FLOAT16,
    KAT_FLOAT32,
    KAT_FLOAT64,

    KAT_UNICODE,
    KAT_STRING,

    KAT_DTY,
    KAT_DTM,
    KAT_DTW,
    KAT_DTD,

    KAT_DTh,
    KAT_DTm,
    KAT_DTs,
    KAT_DTms,
    KAT_DTus,
    KAT_DTns,
    KAT_DTps,
    KAT_DTfs,
    KAT_DTas,
} KeysArrayType;


KeysArrayType
at_to_kat(int array_t, PyArrayObject* a) {
    switch (array_t) {
        case NPY_INT64:
            return KAT_INT64;
        case NPY_INT32:
            return KAT_INT32;
        case NPY_INT16:
            return KAT_INT16;
        case NPY_INT8:
            return KAT_INT8;

        case NPY_UINT64:
            return KAT_UINT64;
        case NPY_UINT32:
            return KAT_UINT32;
        case NPY_UINT16:
            return KAT_UINT16;
        case NPY_UINT8:
            return KAT_UINT8;

        case NPY_FLOAT64:
            return KAT_FLOAT64;
        case NPY_FLOAT32:
            return KAT_FLOAT32;
        case NPY_FLOAT16:
            return KAT_FLOAT16;

        case NPY_UNICODE:
            return KAT_UNICODE;
        case NPY_STRING:
            return KAT_STRING;

        case NPY_DATETIME: {
            NPY_DATETIMEUNIT dtu = AK_dt_unit_from_array(a);
            switch (dtu) {
                case NPY_FR_Y:
                    return KAT_DTY;
                case NPY_FR_M:
                    return KAT_DTM;
                case NPY_FR_W:
                    return KAT_DTW;
                case NPY_FR_D:
                    return KAT_DTD;
                case NPY_FR_h:
                    return KAT_DTh;
                case NPY_FR_m:
                    return KAT_DTm;
                case NPY_FR_s:
                    return KAT_DTs;
                case NPY_FR_ms:
                    return KAT_DTms;
                case NPY_FR_us:
                    return KAT_DTus;
                case NPY_FR_ns:
                    return KAT_DTns;
                case NPY_FR_ps:
                    return KAT_DTps;
                case NPY_FR_fs:
                    return KAT_DTfs;
                case NPY_FR_as:
                    return KAT_DTas;
                case NPY_FR_ERROR:
                case NPY_FR_GENERIC:
                    return KAT_LIST; // fall back to list
            }
        }
        default:
            return KAT_LIST;
    }
}

// To determine when we can use direct array lookups, this function return 1 if we match, 0 if we do not match. Given a keys array type and the kind of lookup key, return true only for the largest KAT types.
int
kat_is_kind(KeysArrayType kat, char kind) {
    switch (kat) {
        case KAT_INT64:
        // case KAT_INT32:
        // case KAT_INT16:
        // case KAT_INT8:
            return kind == 'i';

        case KAT_UINT64:
        // case KAT_UINT32:
        // case KAT_UINT16:
        // case KAT_UINT8:
            return kind == 'u';

        case KAT_FLOAT64:
        // case KAT_FLOAT32:
        // case KAT_FLOAT16:
            return kind == 'f';

        case KAT_UNICODE:
            return kind == 'U';
        case KAT_STRING:
            return kind == 'S';

        case KAT_DTY:
        case KAT_DTM:
        case KAT_DTW:
        case KAT_DTD:
        case KAT_DTh:
        case KAT_DTm:
        case KAT_DTs:
        case KAT_DTms:
        case KAT_DTus:
        case KAT_DTns:
        case KAT_DTps:
        case KAT_DTfs:
        case KAT_DTas:
            return kind == 'M';

        default:
            return 0;
    }
}

// Given a KAT, determine if it matches a NumPy dt64 unit.
bool
kat_is_datetime_unit(KeysArrayType kat, NPY_DATETIMEUNIT unit) {
    switch (kat) {
        case KAT_DTY:
            if (unit == NPY_FR_Y ) {return true;}
            break;
        case KAT_DTM:
            if (unit == NPY_FR_M ) {return true;}
            break;
        case KAT_DTW:
            if (unit == NPY_FR_W ) {return true;}
            break;
        case KAT_DTD:
            if (unit == NPY_FR_D ) {return true;}
            break;
        case KAT_DTh:
            if (unit == NPY_FR_h ) {return true;}
            break;
        case KAT_DTm:
            if (unit == NPY_FR_m ) {return true;}
            break;
        case KAT_DTs:
            if (unit == NPY_FR_s ) {return true;}
            break;
        case KAT_DTms:
            if (unit == NPY_FR_ms) {return true;}
            break;
        case KAT_DTus:
            if (unit == NPY_FR_us) {return true;}
            break;
        case KAT_DTns:
            if (unit == NPY_FR_ns) {return true;}
            break;
        case KAT_DTps:
            if (unit == NPY_FR_ps) {return true;}
            break;
        case KAT_DTfs:
            if (unit == NPY_FR_fs) {return true;}
            break;
        case KAT_DTas:
            if (unit == NPY_FR_as) {return true;}
            break;
        default: // non dt64 KATs
            return false;
    }
    return false;
}

typedef struct FAMObject{
    PyObject_HEAD
    Py_ssize_t table_size;
    TableElement *table;    // an array of TableElement structs
    PyObject *keys;
    KeysArrayType keys_array_type;
    Py_ssize_t keys_size;
    Py_UCS4* key_buffer;
} FAMObject;

typedef enum ViewKind{
    ITEMS,
    KEYS,
    VALUES,
} ViewKind;

// Return the end pointer, or the pointer to the location after the last valid character. The end pointer minus the start pointer is the number of characters. For an empty string, all characters are NULL, and the start pointer and end pointer should be equal. NOTE: would like to use strchr(str, '\0') instead of this routine, but some buffers might not have a null terminator and stread by full to the the dt_size.
static inline Py_UCS4*
ucs4_get_end_p(Py_UCS4* p_start, Py_ssize_t dt_size) {
    for (Py_UCS4* p = p_start + dt_size - 1; p >= p_start; p--) {
        if (*p != '\0') {
            return p + 1; // 1 after first non-null
        }
    }
    return p_start;
}

static inline char*
char_get_end_p(char* p_start, Py_ssize_t dt_size) {
    for (char* p = p_start + dt_size - 1; p >= p_start; p--) {
        if (*p != '\0') {
            return p + 1; // 1 after first non-null
        }
    }
    return p_start;
}

// This masks the input with INT64_MAX, which removes the MSB; we then cast to an int64; the range is now between 0 and INT64_MAX. We then use the MSB of the original value; if set, we negate the number, producing negative values for the upper half of the uint64 range. Note that we only need to check for hash -1 in this branch.
static inline Py_hash_t
uint_to_hash(npy_uint64 v) {
    Py_hash_t hash = (Py_hash_t)(v & INT64_MAX);
    if (v >> 63) {
        hash = -hash;
    }
    if (hash == -1) { // might happen due to overflow on 32 bit systems
        return -2;
    }
    return hash;
}

static inline Py_hash_t
int_to_hash(npy_int64 v) {
    Py_hash_t hash = (Py_hash_t)v;
    if (hash == -1) {
        return -2;
    }
    return hash;
}

// This is a adapted from https://github.com/python/cpython/blob/ba65a065cf07a7a9f53be61057a090f7311a5ad7/Python/pyhash.c#L92
#define HASH_MODULUS (((size_t)1 << 61) - 1)
#define HASH_BITS 61
static inline Py_hash_t
double_to_hash(double v)
{
    int e, sign;
    double m;
    Py_uhash_t x, y;

    if (isinf(v)) {
        return v > 0 ? 314159 : -314159;
    }
    if (isnan(v)) {
        return 0;
    }
    m = frexp(v, &e);
    sign = 1;
    if (m < 0) {
        sign = -1;
        m = -m;
    }
    x = 0;
    while (m) {
        x = ((x << 28) & HASH_MODULUS) | x >> (HASH_BITS - 28);
        m *= 268435456.0;  /* 2**28 */
        e -= 28;
        y = (Py_uhash_t)m;  /* pull out integer part */
        m -= y;
        x += y;
        if (x >= HASH_MODULUS)
            x -= HASH_MODULUS;
    }
    e = e >= 0 ? e % HASH_BITS : HASH_BITS-1-((-1-e) % HASH_BITS);
    x = ((x << e) & HASH_MODULUS) | x >> (HASH_BITS - e);
    x = x * sign;
    if (x == (Py_uhash_t)-1)
        x = (Py_uhash_t)-2;
    return (Py_hash_t)x;
}

// The `str` arg is a pointer to a C-array of Py_UCS4; we will only read `len` characters from this. This is a "djb2" hash algorithm.
static inline Py_hash_t
unicode_to_hash(Py_UCS4 *str, Py_ssize_t len) {
    Py_UCS4* p = str;
    Py_UCS4* p_end = str + len;
    Py_hash_t hash = 5381;
    while (p < p_end) {
        hash = ((hash << 5) + hash) + *p++;
    }
    if (hash == -1) {
        return -2;
    }
    return hash;
}

static inline Py_hash_t
string_to_hash(char *str, Py_ssize_t len) {
    char* p = str;
    char* p_end = str + len;
    Py_hash_t hash = 5381;
    while (p < p_end) {
        hash = ((hash << 5) + hash) + *p++;
    }
    if (hash == -1) {
        return -2;
    }
    return hash;
}

//------------------------------------------------------------------------------
// FrozenAutoMapIterator functions

typedef struct FAMIObject {
    PyObject_HEAD
    FAMObject *fam;
    PyArrayObject* keys_array;
    ViewKind kind;
    bool reversed;
    Py_ssize_t index; // current index state, mutated in-place
} FAMIObject;

static void
fami_dealloc(FAMIObject *self)
{
    Py_DECREF(self->fam);
    PyObject_Del((PyObject *)self);
}

static FAMIObject *
fami_iter(FAMIObject *self)
{
    Py_INCREF(self);
    return self;
}

// For a FAMI, Return appropriate PyObject for items, keys, and values. For consistency with NumPy array iteration, arrays use PyArray_ToScalar instead of PyArray_GETITEM.
static PyObject *
fami_iternext(FAMIObject *self)
{
    Py_ssize_t index;
    if (self->reversed) {
        index = self->fam->keys_size - ++self->index;
        if (index < 0) {
            return NULL;
        }
    }
    else {
        index = self->index++;
    }
    if (self->fam->keys_size <= index) {
        return NULL;
    }
    switch (self->kind) {
        case ITEMS: {
            if (self->fam->keys_array_type) {
                return Py_BuildValue(
                    "NN",
                    PyArray_ToScalar(PyArray_GETPTR1(self->keys_array, index), self->keys_array),
                    PyLong_FromSsize_t(index)
                );
            }
            else {
                PyObject* t = PyTuple_New(2);
                if (!t) { return NULL; }
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13+
                PyObject* k = PyList_GetItemRef(self->fam->keys, index);
#else
                PyObject* k = PyList_GET_ITEM(self->fam->keys, index);
                Py_XINCREF(k);
#endif
                if (!k) { return NULL; }
                PyTuple_SET_ITEM(t, 0, k);
                PyTuple_SET_ITEM(t, 1, PyLong_FromSsize_t(index));
                return t;
            }
        }
        case KEYS: {
            if (self->fam->keys_array_type) {
                return PyArray_ToScalar(PyArray_GETPTR1(self->keys_array, index), self->keys_array);
            }
            else {
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13+
                PyObject* yield = PyList_GetItemRef(self->fam->keys, index);
#else
                PyObject* yield = PyList_GET_ITEM(self->fam->keys, index);
                Py_XINCREF(yield);
#endif
                if (!yield) { return NULL; }
                return yield;
            }
        }
        case VALUES: {
            return PyLong_FromSsize_t(index);
        }
    }
    Py_UNREACHABLE();
}

static PyObject *
fami_length_hint(FAMIObject *self)
{
    Py_ssize_t len = Py_MAX(0, self->fam->keys_size - self->index);
    return PyLong_FromSsize_t(len);
}

static PyObject *fami_new(FAMObject *, ViewKind, bool);

static PyObject *
fami_reversed(FAMIObject *self)
{
    return fami_new(self->fam, self->kind, !self->reversed);
}

static PyMethodDef fami_methods[] = {
    {"__length_hint__", (PyCFunction)fami_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction)fami_reversed, METH_NOARGS, NULL},
    {NULL},
};

PyTypeObject FAMIType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_basicsize = sizeof(FAMIObject),
    .tp_dealloc = (destructor) fami_dealloc,
    .tp_iter = (getiterfunc) fami_iter,
    .tp_iternext = (iternextfunc) fami_iternext,
    .tp_methods = fami_methods,
    .tp_name = "arraykit.FrozenAutoMapIterator",
};

static PyObject *
fami_new(FAMObject *fam, ViewKind kind, bool reversed)
{
    FAMIObject *fami = PyObject_New(FAMIObject, &FAMIType);
    if (!fami) {
        return NULL;
    }
    Py_INCREF(fam);
    fami->fam = fam;
    if (fam->keys_array_type) {
        fami->keys_array = (PyArrayObject *)fam->keys;
    }
    else {
        fami->keys_array = NULL;
    }
    fami->kind = kind;
    fami->reversed = reversed;
    fami->index = 0;
    return (PyObject *)fami;
}

//------------------------------------------------------------------------------
// FrozenAutoMapView functions

// A FAMVObject contains a reference to the FAM from which it was derived
typedef struct FAMVObject{
    PyObject_HEAD
    FAMObject *fam;
    ViewKind kind;
} FAMVObject;

# define FAMV_SET_OP(name, op)                            \
static PyObject *                                         \
name(PyObject *left, PyObject *right)                     \
{                                                         \
    left = PySet_New(left);                               \
    if (!left) {                                          \
        return NULL;                                      \
    }                                                     \
    right = PySet_New(right);                             \
    if (!right) {                                         \
        Py_DECREF(left);                                  \
        return NULL;                                      \
    }                                                     \
    PyObject *result = PyNumber_InPlace##op(left, right); \
    Py_DECREF(left);                                      \
    Py_DECREF(right);                                     \
    return result;                                        \
}

FAMV_SET_OP(famv_and, And)
FAMV_SET_OP(famv_or, Or)
FAMV_SET_OP(famv_subtract, Subtract)
FAMV_SET_OP(famv_xor, Xor)

# undef FAMV_SET_OP

static PyNumberMethods famv_as_number = {
    .nb_and = (binaryfunc) famv_and,
    .nb_or = (binaryfunc) famv_or,
    .nb_subtract = (binaryfunc) famv_subtract,
    .nb_xor = (binaryfunc) famv_xor,
};

static int fam_contains(FAMObject *, PyObject *);
static PyObject *famv_fami_new(FAMVObject *);

static int
famv_contains(FAMVObject *self, PyObject *other)
{
    if (self->kind == KEYS) {
        return fam_contains(self->fam, other);
    }
    PyObject *iterator = famv_fami_new(self);
    if (!iterator) {
        return -1;
    }
    int result = PySequence_Contains(iterator, other);
    Py_DECREF(iterator);
    return result;
}

static PySequenceMethods famv_as_sequence = {
    .sq_contains = (objobjproc) famv_contains,
};

static void
famv_dealloc(FAMVObject *self)
{
    Py_DECREF(self->fam);
    PyObject_Del((PyObject *)self);
}

static PyObject *
famv_fami_new(FAMVObject *self)
{
    return fami_new(self->fam, self->kind, false);
}

static PyObject *
famv_length_hint(FAMVObject *self)
{
    return PyLong_FromSsize_t(self->fam->keys_size);
}

static PyObject *
famv_reversed(FAMVObject *self)
{
    return fami_new(self->fam, self->kind, true);
}

static PyObject *
famv_isdisjoint(FAMVObject *self, PyObject *other)
{
    PyObject *intersection = famv_and((PyObject *)self, other);
    if (!intersection) {
        return NULL;
    }
    Py_ssize_t result = PySet_GET_SIZE(intersection);
    Py_DECREF(intersection);
    return PyBool_FromLong(result);
}

static PyObject *
famv_richcompare(FAMVObject *self, PyObject *other, int op)
{
    PyObject *left = PySet_New((PyObject *)self);
    if (!left) {
        return NULL;
    }
    PyObject *right = PySet_New(other);
    if (!right) {
        Py_DECREF(left);
        return NULL;
    }
    PyObject *result = PyObject_RichCompare(left, right, op);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}

static PyMethodDef famv_methods[] = {
    {"__length_hint__", (PyCFunction) famv_length_hint, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) famv_reversed, METH_NOARGS, NULL},
    {"isdisjoint", (PyCFunction) famv_isdisjoint, METH_O, NULL},
    {NULL},
};

PyTypeObject FAMVType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &famv_as_number,
    .tp_as_sequence = &famv_as_sequence,
    .tp_basicsize = sizeof(FAMVObject),
    .tp_dealloc = (destructor) famv_dealloc,
    .tp_iter = (getiterfunc) famv_fami_new,
    .tp_methods = famv_methods,
    .tp_name = "arraykit.FrozenAutoMapView",
    .tp_richcompare = (richcmpfunc) famv_richcompare,
};

static PyObject *
famv_new(FAMObject *fam, ViewKind kind)
{
    FAMVObject *famv = (FAMVObject *)PyObject_New(FAMVObject, &FAMVType);
    if (!famv) {
        return NULL;
    }
    famv->kind = kind;
    famv->fam = fam;
    Py_INCREF(fam);
    return (PyObject *)famv;
}

//------------------------------------------------------------------------------
// FrozenAutoMap functions

// Given a key and a computed hash, return the table_pos if that hash and key are found, or if not, the first table position that has not been assigned. Return -1 on error.
static Py_ssize_t
lookup_hash_obj(FAMObject *self, PyObject *key, Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyObject *guess = NULL;
    PyObject *keys = self->keys;
    int result = -1;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) { // Miss. Found a position that can be used for insertion.
                return table_pos;
            }
            if (h != hash) { // Collision.
                table_pos++;
                continue;
            }
            guess = PyList_GET_ITEM(keys, table[table_pos].keys_pos);
            if (guess == key) { // Hit. Object ID comparison
                return table_pos;
            }

            // if key is a dt64, only do PyObject_RichCompareBool if units match
            if (PyArray_IsScalar(key, Datetime) && PyArray_IsScalar(guess, Datetime)) {
                if (AK_dt_unit_from_scalar((PyDatetimeScalarObject *)key)
                    != AK_dt_unit_from_scalar((PyDatetimeScalarObject *)guess)) {
                    table_pos++;
                    continue;
                }
            }

            result = PyObject_RichCompareBool(guess, key, Py_EQ);
            if (result < 0) { // Error.
                return -1;
            }
            if (result) { // Hit.
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Used for both integer and datetime types; for this reason kat is passed in separately.
static Py_ssize_t
lookup_hash_int(FAMObject *self, npy_int64 key, Py_hash_t hash, KeysArrayType kat)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask; // taking the modulo

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_int64 k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) { // Miss. Position that can be used for insertion.
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (kat) {
                case KAT_INT64:
                    k = *(npy_int64*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT32:
                    k = *(npy_int32*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT16:
                    k = *(npy_int16*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_INT8:
                    k = *(npy_int8*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// NOTE: kat is passed in separately to match the interface of lookup_hash_int.
static Py_ssize_t
lookup_hash_uint(FAMObject *self, npy_uint64 key, Py_hash_t hash, KeysArrayType kat)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_uint64 k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (kat) {
                case KAT_UINT64:
                    k = *(npy_uint64*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT32:
                    k = *(npy_uint32*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT16:
                    k = *(npy_uint16*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_UINT8:
                    k = *(npy_uint8*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// NOTE: kat is passed in separately to match the interface of lookup_hash_int
static Py_ssize_t
lookup_hash_double(FAMObject *self, npy_double key, Py_hash_t hash, KeysArrayType kat)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    npy_double k = 0;
    Py_hash_t h = 0;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            switch (kat) {
                case KAT_FLOAT64:
                    k = *(npy_double*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_FLOAT32:
                    k = *(npy_float*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
                    break;
                case KAT_FLOAT16:
                    k = npy_half_to_double(*(npy_half*)PyArray_GETPTR1(a, table[table_pos].keys_pos));
                    break;
                default:
                    return -1;
            }
            if (key == k) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Compare a passed Py_UCS4 array to stored keys. This does not use any dynamic memory. Returns -1 on error.
static Py_ssize_t
lookup_hash_unicode(
        FAMObject *self,
        Py_UCS4* key,
        Py_ssize_t key_size,
        Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_ITEMSIZE(a) / UCS4_SIZE;
    Py_ssize_t cmp_bytes = Py_MIN(key_size, dt_size) * UCS4_SIZE;

    Py_hash_t h = 0;
    Py_UCS4* p_start = NULL;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            p_start = (Py_UCS4*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
            // memcmp returns 0 on match
            if (!memcmp(p_start, key, cmp_bytes)) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


// Compare a passed char array to stored keys. This does not use any dynamic memory. Returns -1 on error.
static Py_ssize_t
lookup_hash_string(
        FAMObject *self,
        char* key,
        Py_ssize_t key_size,
        Py_hash_t hash)
{
    TableElement *table = self->table;
    Py_ssize_t mask = self->table_size - 1;
    Py_hash_t mixin = Py_ABS(hash);
    Py_ssize_t table_pos = hash & mask;

    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_ITEMSIZE(a);
    Py_ssize_t cmp_bytes = Py_MIN(key_size, dt_size);

    Py_hash_t h = 0;
    char* p_start = NULL;

    while (1) {
        for (Py_ssize_t i = 0; i < SCAN; i++) {
            h = table[table_pos].hash;
            if (h == -1) {
                return table_pos;
            }
            if (h != hash) {
                table_pos++;
                continue;
            }
            p_start = (char*)PyArray_GETPTR1(a, table[table_pos].keys_pos);
            // memcmp returns 0 on match
            if (!memcmp(p_start, key, cmp_bytes)) {
                return table_pos;
            }
            table_pos++;
        }
        table_pos = (5 * (table_pos - SCAN) + (mixin >>= 1) + 1) & mask;
    }
}


static Py_ssize_t
lookup_int(FAMObject *self, PyObject* key) {
    npy_int64 v = 0;
    // NOTE: we handle PyArray Scalar Byte, Short, UByte, UShort with PyNumber_Check, below, saving four branches here
    if (PyArray_IsScalar(key, LongLong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, LongLong);
    }
    else if (PyArray_IsScalar(key, Long)) {
        v = (npy_int64)PyArrayScalar_VAL(key, Long);
    }
    else if (PyLong_Check(key)) {
        v = PyLong_AsLongLong(key);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Double)) {
        double dv = PyArrayScalar_VAL(key, Double);
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyFloat_Check(key)) {
        double dv = PyFloat_AsDouble(key);
        if (dv == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        v = (npy_int64)dv; // truncate to integer
        if (v != dv) {
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, ULongLong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, ULongLong);
    }
    else if (PyArray_IsScalar(key, ULong)) {
        v = (npy_int64)PyArrayScalar_VAL(key, ULong);
    }
    else if (PyArray_IsScalar(key, Int)) {
        v = (npy_int64)PyArrayScalar_VAL(key, Int);
    }
    else if (PyArray_IsScalar(key, UInt)) {
        v = (npy_int64)PyArrayScalar_VAL(key, UInt);
    }
    else if (PyArray_IsScalar(key, Float)) {
        double dv = (double)PyArrayScalar_VAL(key, Float);
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyArray_IsScalar(key, Half)) {
        double dv = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        if (floor(dv) != dv) {
            return -1;
        }
        v = (npy_int64)dv;
    }
    else if (PyBool_Check(key)) {
        v = PyObject_IsTrue(key);
    }
    else if (PyNumber_Check(key)) {
        // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
        v = (npy_int64)PyNumber_AsSsize_t(key, PyExc_OverflowError);
        if (v == -1 && PyErr_Occurred()) {
            return -1;
        }
    }
    else {
        return -1;
    }
    Py_hash_t hash = int_to_hash(v);
    return lookup_hash_int(self, v, hash, self->keys_array_type);
}


// In current usage as an AM, np.datetime64 will match to any numpy Scalar that is at or greater than the resolution of the values stored here. No matches are made to other numerics or Python datetime objects. For AM to be consistent with FAM, we will do the same for now.
static Py_ssize_t
lookup_datetime(FAMObject *self, PyObject* key) {
    npy_int64 v = 0; // int64
    if (PyArray_IsScalar(key, Datetime)) {
        v = (npy_int64)PyArrayScalar_VAL(key, Datetime);
        // if we observe a NAT, we skip unit checks

        if (v != NPY_DATETIME_NAT) {
            NPY_DATETIMEUNIT key_unit = AK_dt_unit_from_scalar(
                    (PyDatetimeScalarObject *)key);
            if (!kat_is_datetime_unit(self->keys_array_type, key_unit)) {
                return -1;
            }
        }
    }
    else {
        return -1;
    }
    Py_hash_t hash = int_to_hash(v);
    return lookup_hash_int(self, v, hash, KAT_INT64);
}


static Py_ssize_t
lookup_uint(FAMObject *self, PyObject* key) {
    npy_uint64 v = 0;

    // NOTE: we handle PyArray Scalar Byte, Short, UByte, UShort with PyNumber_Check, below, saving four branches here
    if (PyArray_IsScalar(key, ULongLong)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, ULongLong);
    }
    else if (PyArray_IsScalar(key, ULong)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, ULong);
    }
    else if (PyArray_IsScalar(key, LongLong)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, LongLong);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyArray_IsScalar(key, Long)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, Long);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyLong_Check(key)) {
        v = PyLong_AsUnsignedLongLong(key);
        if (v == (unsigned long long)-1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Double)) {
        double dv = PyArrayScalar_VAL(key, Double);
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyFloat_Check(key)) {
        double dv = PyFloat_AsDouble(key);
        if (dv == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        if (dv < 0) {
            return -1;
        }
        v = (npy_uint64)dv; // truncate to integer
        if (v != dv) {
            return -1;
        }
    }
    else if (PyArray_IsScalar(key, Int)) {
        npy_int64 si = (npy_int64)PyArrayScalar_VAL(key, Int);
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else if (PyArray_IsScalar(key, UInt)) {
        v = (npy_uint64)PyArrayScalar_VAL(key, UInt);
    }
    else if (PyArray_IsScalar(key, Float)) {
        double dv = (double)PyArrayScalar_VAL(key, Float);
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyArray_IsScalar(key, Half)) {
        double dv = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        if (dv < 0 || floor(dv) != dv) {
            return -1;
        }
        v = (npy_uint64)dv;
    }
    else if (PyBool_Check(key)) {
        v = PyObject_IsTrue(key);
    }
    else if (PyNumber_Check(key)) {
        // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
        npy_int64 si = PyNumber_AsSsize_t(key, PyExc_OverflowError);
        if (si == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return -1;
        }
        if (si < 0) {
            return -1;
        }
        v = (npy_uint64)si;
    }
    else {
        return -1;
    }
    return lookup_hash_uint(self, v, uint_to_hash(v), self->keys_array_type);
}


static Py_ssize_t
lookup_double(FAMObject *self, PyObject* key) {
        double v = 0;
        if (PyArray_IsScalar(key, Double)) {
            v = PyArrayScalar_VAL(key, Double);
        }
        else if (PyFloat_Check(key)) {
            v = PyFloat_AsDouble(key);
            if (v == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
        }
        else if (PyLong_Check(key)) {
            v = (double)PyLong_AsLongLong(key);
            if (v == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
        }
        // NOTE: we handle PyArray Scalar Byte, Short with PyNumber_Check, below, saving four branches here
        else if (PyArray_IsScalar(key, LongLong)) {
            v = (double)PyArrayScalar_VAL(key, LongLong);
        }
        else if (PyArray_IsScalar(key, Long)) {
            v = (double)PyArrayScalar_VAL(key, Long);
        }
        else if (PyArray_IsScalar(key, Int)) {
            v = (double)PyArrayScalar_VAL(key, Int);
        }
        else if (PyArray_IsScalar(key, ULongLong)) {
            v = (double)PyArrayScalar_VAL(key, ULongLong);
        }
        else if (PyArray_IsScalar(key, ULong)) {
            v = (double)PyArrayScalar_VAL(key, ULong);
        }
        else if (PyArray_IsScalar(key, UInt)) {
            v = (double)PyArrayScalar_VAL(key, UInt);
        }
        else if (PyArray_IsScalar(key, Float)) {
            v = (double)PyArrayScalar_VAL(key, Float);
        }
        else if (PyArray_IsScalar(key, Half)) {
            v = npy_half_to_double(PyArrayScalar_VAL(key, Half));
        }
        else if (PyBool_Check(key)) {
            v = PyObject_IsTrue(key);
        }
        else if (PyNumber_Check(key)) {
            // NOTE: this returns a Py_ssize_t, which might be 32 bit. This can be used for PyArray_Scalars <= ssize_t.
            npy_int64 si = PyNumber_AsSsize_t(key, PyExc_OverflowError);
            if (si == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                return -1;
            }
            v = (double)si;
        }
        else {
            return -1;
        }
        return lookup_hash_double(self, v, double_to_hash(v), self->keys_array_type);
}


static Py_ssize_t
lookup_unicode(FAMObject *self, PyObject* key) {
    // NOTE: while we can identify and use PyArray_IsScalar(key, Unicode), this did not improve performance and fails on Windows.
    if (!PyUnicode_Check(key)) {
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_ITEMSIZE(a) / UCS4_SIZE;
    // if the key_size is greater than the dtype size of the array, we know there cannot be a match
    Py_ssize_t k_size = PyUnicode_GetLength(key);
    if (k_size > dt_size) {
        return -1;
    }
    // The buffer will have dt_size + 1 storage. We copy a NULL character so do not have to clear the buffer, but instead can reuse it and still discover the lookup
    if (!PyUnicode_AsUCS4(key, self->key_buffer, dt_size+1, 1)) {
        return -1; // exception will be set
    }
    Py_hash_t hash = unicode_to_hash(self->key_buffer, k_size);
    return lookup_hash_unicode(self, self->key_buffer, k_size, hash);
}


static Py_ssize_t
lookup_string(FAMObject *self, PyObject* key) {
    if (!PyBytes_Check(key)) {
        return -1;
    }
    PyArrayObject *a = (PyArrayObject *)self->keys;
    Py_ssize_t dt_size = PyArray_ITEMSIZE(a);
    Py_ssize_t k_size = PyBytes_GET_SIZE(key);
    if (k_size > dt_size) {
        return -1;
    }
    char* k = PyBytes_AS_STRING(key);
    Py_hash_t hash = string_to_hash(k, k_size);
    return lookup_hash_string(self, k, k_size, hash);
}


// Given a key as a PyObject, return the Py_ssize_t keys_pos value stored in the TableElement. Return -1 on key not found (without setting an exception) and -1 on error (with setting an exception).
static Py_ssize_t
lookup(FAMObject *self, PyObject *key) {
    Py_ssize_t table_pos = -1;

    switch (self->keys_array_type) {
        case KAT_INT64:
        case KAT_INT32:
        case KAT_INT16:
        case KAT_INT8:
            table_pos = lookup_int(self, key);
            break;
        case KAT_UINT64:
        case KAT_UINT32:
        case KAT_UINT16:
        case KAT_UINT8:
            table_pos = lookup_uint(self, key);
            break;
        case KAT_FLOAT64:
        case KAT_FLOAT32:
        case KAT_FLOAT16:
            table_pos = lookup_double(self, key);
            break;
        case KAT_UNICODE:
            table_pos = lookup_unicode(self, key);
            break;
        case KAT_STRING:
            table_pos = lookup_string(self, key);
            break;
        case KAT_DTY:
        case KAT_DTM:
        case KAT_DTW:
        case KAT_DTD:
        case KAT_DTh:
        case KAT_DTm:
        case KAT_DTs:
        case KAT_DTms:
        case KAT_DTus:
        case KAT_DTns:
        case KAT_DTps:
        case KAT_DTfs:
        case KAT_DTas:
            table_pos = lookup_datetime(self, key);
            break;
        case KAT_LIST: {
            Py_hash_t hash = PyObject_Hash(key);
            if (hash == -1) {
                return -1;
            }
            table_pos = lookup_hash_obj(self, key, hash);
            break;
        }
    }
    // A -1 hash is an unused storage location
    if ((table_pos < 0) || (self->table[table_pos].hash == -1)) {
        return -1;
    }
    return self->table[table_pos].keys_pos;
}

// Insert a key_pos, hash pair into the table. Assumes table already has appropriate size. When inserting a new item, `hash` is -1, forcing a fresh hash to be computed here. Return 0 on success, -1 on error.
static int
insert_obj(
        FAMObject *self,
        PyObject *key,  // NOTE: a borrowed reference
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = PyObject_Hash(key);
        if (hash == -1) {
            return -1;
        }
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_obj(self, key, hash);

    if (table_pos < 0) {
        return -1;
    }
    // We expect, on insertion, to get back a table_pos that points to an unassigned hash value (-1); if we get anything else, we have found a match to an already-existing key, and thus raise a NonUniqueError error.
    if (self->table[table_pos].hash != -1) {
        PyErr_SetObject(NonUniqueError, key);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_int(
        FAMObject *self,
        npy_int64 key,
        Py_ssize_t keys_pos,
        Py_hash_t hash,
        KeysArrayType kat)
{
    if (hash == -1) {
        hash = int_to_hash(key);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_int(self, key, hash, kat);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyObject* er = PyLong_FromLongLong(key); // for error reporting
        if (er == NULL) {
            return -1;
        }
        PyErr_SetObject(NonUniqueError, er);
        Py_DECREF(er);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash; // key is the hash
    return 0;
}


static int
insert_uint(
        FAMObject *self,
        npy_uint64 key,
        Py_ssize_t keys_pos,
        Py_hash_t hash,
        KeysArrayType kat)
{
    if (hash == -1) {
        hash = uint_to_hash(key);
    }
    Py_ssize_t table_pos = lookup_hash_uint(self, key, hash, kat);

    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyObject* er = PyLong_FromUnsignedLongLong(key);
        if (er == NULL) {
            return -1;
        }
        PyErr_SetObject(NonUniqueError, er);
        Py_DECREF(er);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_double(
        FAMObject *self,
        npy_double key,
        Py_ssize_t keys_pos,
        Py_hash_t hash,
        KeysArrayType kat)
{
    if (hash == -1) {
        hash = double_to_hash(key);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_double(self, key, hash, kat);

    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyObject* er = PyFloat_FromDouble(key);
        if (er == NULL) {
            return -1;
        }
        PyErr_SetObject(NonUniqueError, er);
        Py_DECREF(er);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_unicode(
        FAMObject *self,
        Py_UCS4* key,
        Py_ssize_t key_size,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = unicode_to_hash(key, key_size);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_unicode(self, key, key_size, hash);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyObject* er = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, key, key_size);
        if (er == NULL) {
            return -1;
        }
        PyErr_SetObject(NonUniqueError, er);
        Py_DECREF(er);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


static int
insert_string(
        FAMObject *self,
        char* key,
        Py_ssize_t key_size,
        Py_ssize_t keys_pos,
        Py_hash_t hash)
{
    if (hash == -1) {
        hash = string_to_hash(key, key_size);
    }
    // table position is not dependent on keys_pos
    Py_ssize_t table_pos = lookup_hash_string(self, key, key_size, hash);
    if (table_pos < 0) {
        return -1;
    }
    if (self->table[table_pos].hash != -1) {
        PyObject* er = PyBytes_FromStringAndSize(key, key_size);
        if (er == NULL) {
            return -1;
        }
        PyErr_SetObject(NonUniqueError, er);
        Py_DECREF(er);
        return -1;
    }
    self->table[table_pos].keys_pos = keys_pos;
    self->table[table_pos].hash = hash;
    return 0;
}


//------------------------------------------------------------------------------

// Called in fam_new(), extend(), append(), with the size of observed keys. This table is updated only when append or extending. Only if there is an old table will keys be accessed Returns 0 on success, -1 on failure.
static int
grow_table(FAMObject *self, Py_ssize_t keys_size)
{
    Py_ssize_t keys_load = keys_size / LOAD;
    Py_ssize_t size_old = self->table_size;
    if (keys_load < size_old) {
        return 0;
    }

    // get the next power of 2 greater than current keys_load
    Py_ssize_t size_new = 1;
    while (size_new <= keys_load) {
        size_new <<= 1;
    }
    // size_new > keys_load; we know that keys_load >= size_old, so size_new must be > size_old
    TableElement *table_old = self->table;
    TableElement *table_new = PyMem_New(TableElement, size_new + SCAN - 1);
    if (!table_new) {
        return -1;
    }

    // initialize all hash and keys_pos values to -1
    Py_ssize_t table_pos;
    for (table_pos = 0; table_pos < size_new + SCAN - 1; table_pos++) {
        table_new[table_pos].hash = -1;
        table_new[table_pos].keys_pos = -1;
    }
    self->table = table_new;
    self->table_size = size_new;

    // if we have an old table, move them into the new table
    if (size_old) {
        if (self->keys_array_type) {
            PyErr_SetString(PyExc_NotImplementedError, "Cannot grow table for array keys");
            goto restore;
        }
        Py_ssize_t i;
        Py_hash_t h;
        for (table_pos = 0; table_pos < size_old + SCAN - 1; table_pos++) {
            i = table_old[table_pos].keys_pos;
            h = table_old[table_pos].hash;
            if ((h != -1) && insert_obj(self, PyList_GET_ITEM(self->keys, i), i, h))
            {
                goto restore;
            }
        }
    }
    PyMem_Del(table_old);
    return 0;
restore:
    PyMem_Del(self->table);
    self->table = table_old;
    self->table_size = size_old;
    return -1;
}


// Given a new, possibly un-initialized FAMObject, copy attrs from self to new. Return 0 on success, -1 on error.
int
copy_to_new(PyTypeObject *cls, FAMObject *self, FAMObject *new)
{
    if (self->keys_array_type) {
        new->keys = self->keys;
        Py_INCREF(new->keys);
    }
    else {
        new->keys = PySequence_List(self->keys);
        if (!new->keys) {
            return -1;
        }
    }
    new->table_size = self->table_size;
    new->keys_array_type = self->keys_array_type;
    new->keys_size = self->keys_size;

    new->key_buffer = NULL;
    if (new->keys_array_type == KAT_UNICODE) {
        PyArrayObject *a = (PyArrayObject *)new->keys;
        Py_ssize_t dt_size = PyArray_ITEMSIZE(a) / UCS4_SIZE;
        new->key_buffer = (Py_UCS4*)PyMem_Malloc((dt_size+1) * UCS4_SIZE);
    }

    Py_ssize_t table_size_alloc = new->table_size + SCAN - 1;
    new->table = PyMem_New(TableElement, table_size_alloc);
    if (!new->table) {
        // Py_DECREF(new->keys); // assume this will get cleaned up
        return -1;
    }
    memcpy(new->table, self->table, table_size_alloc * sizeof(TableElement));
    return 0;
}


static PyObject *
fam_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs);


// Create a copy of self. Used in `fam_or()`. Returns a new FAMObject on success, NULL on error.
static FAMObject *
copy(PyTypeObject *cls, FAMObject *self)
{
    if (!PyType_IsSubtype(cls, &AMType) && !PyObject_TypeCheck(self, &AMType)) {
        Py_INCREF(self);
        return self;
    }
    // fam_new to allocate and full struct attrs
    FAMObject *new = (FAMObject*)fam_new(cls, NULL, NULL);
    if (!new) {
        return NULL;
    }
    if (copy_to_new(cls, self, new)) {
        Py_DECREF(new); // assume this will decref any partially set attrs of new
    }
    return new;
}



// Returns -1 on error, 0 on success.
static int
extend(FAMObject *self, PyObject *keys)
{
    if (self->keys_array_type) {
        PyErr_SetString(PyExc_NotImplementedError, "Not supported for array keys");
        return -1;
    }
    // this should fail for self->keys types that are not a list
    keys = PySequence_Fast(keys, "expected an iterable of keys");
    if (!keys) {
        return -1;
    }
    Py_ssize_t size_extend = PySequence_Fast_GET_SIZE(keys);
    self->keys_size += size_extend;

    if (grow_table(self, self->keys_size)) {
        Py_DECREF(keys);
        return -1;
    }

    PyObject **keys_fi = PySequence_Fast_ITEMS(keys);

    for (Py_ssize_t index = 0; index < size_extend; index++) {
        // get the new keys_size after each append
        if (insert_obj(self, keys_fi[index], PyList_GET_SIZE(self->keys), -1) ||
            PyList_Append(self->keys, keys_fi[index]))
        {
            Py_DECREF(keys);
            return -1;
        }
    }
    Py_DECREF(keys);
    return 0;
}


// Returns -1 on error, 0 on success.
static int
append(FAMObject *self, PyObject *key)
{
    if (self->keys_array_type) {
        PyErr_SetString(PyExc_NotImplementedError, "Not supported for array keys");
        return -1;
    }
    self->keys_size++;

    if (grow_table(self, self->keys_size)) {
        return -1;
    }
    // keys_size is already incremented; provide last index
    if (insert_obj(self, key, self->keys_size - 1, -1) ||
        PyList_Append(self->keys, key))
    {
        return -1;
    }
    return 0;
}


static Py_ssize_t
fam_length(FAMObject *self)
{
    return self->keys_size;
}


// Given a key for a FAM, return the Python integer associated with that key. Utility function used in both fam_subscript() and fam_get()
static PyObject *
get(FAMObject *self, PyObject *key, PyObject *missing) {
    Py_ssize_t keys_pos = lookup(self, key);
    if (keys_pos < 0) {
        if (PyErr_Occurred()) {
            return NULL;
        }
        if (missing) {
            Py_INCREF(missing);
            return missing;
        }
        PyErr_SetObject(PyExc_KeyError, key);
        return NULL;
    }
    return PyLong_FromSsize_t(keys_pos);
}


// Give an array of the same kind as KAT, lookup and load all keys_pos. Depends on self, key_size, key_array, table_pos, i, k, b
# define GET_ALL_SCALARS(npy_type_src, npy_type_dst, kat, lookup_func, hash_func, to_obj_func, post_deref) \
{                                                                      \
    npy_type_dst v;                                                    \
    for (; i < key_size; i++) {                                        \
        v = post_deref(*(npy_type_src*)PyArray_GETPTR1(key_array, i)); \
        table_pos = lookup_func(self, v, hash_func(v), kat);           \
        if (table_pos < 0 || (self->table[table_pos].hash == -1)) {    \
            Py_DECREF(array);                                          \
            if (PyErr_Occurred()) {                                    \
                return NULL;                                           \
            }                                                          \
            k = to_obj_func(v);                                        \
            if (k == NULL) {                                           \
                return NULL;                                           \
            }                                                          \
            PyErr_SetObject(PyExc_KeyError, k);                        \
            Py_DECREF(k);                                              \
            return NULL;                                               \
        }                                                              \
        b[i] = (npy_int64)self->table[table_pos].keys_pos;             \
    }                                                                  \
}                                                                      \

# define GET_ALL_DT64(npy_type_src, npy_type_dst, kat, lookup_func, hash_func) \
{                                                                      \
    npy_type_dst v;                                                    \
    for (; i < key_size; i++) {                                        \
        v = *(npy_type_src*)PyArray_GETPTR1(key_array, i);             \
        table_pos = lookup_func(self, v, hash_func(v), kat);           \
        if (table_pos < 0 || (self->table[table_pos].hash == -1)) {    \
            Py_DECREF(array);                                          \
            if (PyErr_Occurred()) {                                    \
                return NULL;                                           \
            }                                                          \
            k = PyArray_ToScalar(&v, key_array);                       \
            if (k == NULL) {                                           \
                return NULL;                                           \
            }                                                          \
            PyErr_SetObject(PyExc_KeyError, k);                        \
            Py_DECREF(k);                                              \
            return NULL;                                               \
        }                                                              \
        b[i] = (npy_int64)self->table[table_pos].keys_pos;             \
    }                                                                  \
}                                                                      \

# define GET_ALL_FLEXIBLE(char_type, get_end_func, lookup_func, hash_func, to_obj_func) \
{                                                                             \
    char_type* v;                                                             \
    Py_ssize_t dt_size = PyArray_ITEMSIZE(key_array) / sizeof(char_type);\
    Py_ssize_t k_size;                                                        \
    for (; i < key_size; i++) {                                               \
        v = (char_type*)PyArray_GETPTR1(key_array, i);                        \
        k_size = get_end_func(v, dt_size) - v;                                \
        table_pos = lookup_func(self, v, k_size, hash_func(v, k_size));       \
        if (table_pos < 0 || (self->table[table_pos].hash == -1)) {           \
            Py_DECREF(array);                                                 \
            if (PyErr_Occurred()) {                                           \
                return NULL;                                                  \
            }                                                                 \
            k = to_obj_func(v, k_size);                                       \
            if (k == NULL) {                                                  \
                return NULL;                                                  \
            }                                                                 \
            PyErr_SetObject(PyExc_KeyError, k);                               \
            Py_DECREF(k);                                                     \
            return NULL;                                                      \
        }                                                                     \
        b[i] = (npy_int64)self->table[table_pos].keys_pos;                    \
    }                                                                         \
}                                                                             \

// Given a list or array of keys, return an array of the lookup-up integer values. If any unmatched keys are found, a KeyError will raise. An immutable array is always returned.
static PyObject *
fam_get_all(FAMObject *self, PyObject *key) {
    Py_ssize_t key_size = 0;
    Py_ssize_t keys_pos = -1;
    PyObject* k = NULL;
    PyObject *array = NULL;
    Py_ssize_t i = 0;

    int key_is_list;
    if (PyList_CheckExact(key)) {
        key_is_list = 1;
        key_size = PyList_GET_SIZE(key);
    }
    else if (PyArray_Check(key)) {
        key_is_list = 0;
        key_size = PyArray_SIZE((PyArrayObject *)key);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Must provide a list or array.");
        return NULL;
    }

    // construct array to be returned; this is a little expensive if we do not yet know if we can use it
    npy_intp dims[] = {key_size};
    array = PyArray_EMPTY(1, dims, NPY_INT64, 0);
    if (array == NULL) {
        return NULL;
    }
    npy_int64* b = (npy_int64*)PyArray_DATA((PyArrayObject*)array);

    if (key_is_list) {
        for (; i < key_size; i++) {
            k = PyList_GET_ITEM(key, i); // borrow
            keys_pos = lookup(self, k);
            if (keys_pos < 0) {
                Py_DECREF(array);
                if (PyErr_Occurred()) {
                    return NULL;
                }
                PyErr_SetObject(PyExc_KeyError, k);
                return NULL;
            }
            b[i] = (npy_int64)keys_pos;
        }
    }
    else { // key is an array
        PyArrayObject* key_array = (PyArrayObject *)key;
        // if key is an np array of the same kind as this FAMs keys, we can do optimized lookups; otherwise, we have to go through scalar to do full branching and coercion into lookup
        int key_array_t = PyArray_TYPE(key_array);

        // NOTE: we only match numeric kinds of the KAT is 64 bit; we could support, for each key_array_t, a switch for every KAT, but the size of that code is huge and the performance benefit is not massive
        if (kat_is_kind(self->keys_array_type, PyArray_DESCR(key_array)->kind)) {
            Py_ssize_t table_pos;
            switch (key_array_t) { // type of passed in array
                case NPY_INT64:
                    GET_ALL_SCALARS(npy_int64, npy_int64, KAT_INT64, lookup_hash_int, int_to_hash, PyLong_FromLongLong,);
                    break;
                case NPY_INT32:
                    GET_ALL_SCALARS(npy_int32, npy_int64, KAT_INT32, lookup_hash_int, int_to_hash, PyLong_FromLongLong,);
                    break;
                case NPY_INT16:
                    GET_ALL_SCALARS(npy_int16, npy_int64, KAT_INT16, lookup_hash_int, int_to_hash, PyLong_FromLongLong,);
                    break;
                case NPY_INT8:
                    GET_ALL_SCALARS(npy_int8, npy_int64, KAT_INT8, lookup_hash_int, int_to_hash, PyLong_FromLongLong,);
                    break;
                case NPY_UINT64:
                    GET_ALL_SCALARS(npy_uint64, npy_uint64, KAT_UINT64, lookup_hash_uint, uint_to_hash, PyLong_FromUnsignedLongLong,);
                    break;
                case NPY_UINT32:
                    GET_ALL_SCALARS(npy_uint32, npy_uint64, KAT_UINT32, lookup_hash_uint, uint_to_hash, PyLong_FromUnsignedLongLong,);
                    break;
                case NPY_UINT16:
                    GET_ALL_SCALARS(npy_uint16, npy_uint64, KAT_UINT16, lookup_hash_uint, uint_to_hash, PyLong_FromUnsignedLongLong,);
                    break;
                case NPY_UINT8:
                    GET_ALL_SCALARS(npy_uint8, npy_uint64, KAT_UINT8, lookup_hash_uint, uint_to_hash, PyLong_FromUnsignedLongLong,);
                    break;
                case NPY_FLOAT64:
                    GET_ALL_SCALARS(npy_double, npy_double, KAT_FLOAT64, lookup_hash_double, double_to_hash, PyFloat_FromDouble,);
                    break;
                case NPY_FLOAT32:
                    GET_ALL_SCALARS(npy_float, npy_double, KAT_FLOAT32, lookup_hash_double, double_to_hash, PyFloat_FromDouble,);
                    break;
                case NPY_FLOAT16:
                    GET_ALL_SCALARS(npy_half, npy_double, KAT_FLOAT16, lookup_hash_double, double_to_hash, PyFloat_FromDouble, npy_half_to_double);
                    break;
                case NPY_UNICODE:
                    GET_ALL_FLEXIBLE(Py_UCS4, ucs4_get_end_p, lookup_hash_unicode, unicode_to_hash, PyUnicode_FromUCS4AndData);
                    break;
                case NPY_STRING:
                    GET_ALL_FLEXIBLE(char, char_get_end_p, lookup_hash_string, string_to_hash, PyBytes_FromStringAndSize);
                    break;
                case NPY_DATETIME: {
                    NPY_DATETIMEUNIT key_unit = AK_dt_unit_from_array(key_array);
                    if (!kat_is_datetime_unit(self->keys_array_type, key_unit)) {
                        PyErr_SetString(PyExc_KeyError, "datetime64 units do not match");
                        Py_DECREF(array);
                        return NULL;
                    }
                    GET_ALL_DT64(npy_int64, npy_int64, KAT_INT64, lookup_hash_int, int_to_hash);
                    break;
                }
            }
        }
        else {
            for (; i < key_size; i++) {
                k = PyArray_ToScalar(PyArray_GETPTR1(key_array, i), key_array);
                if (k == NULL) {
                    Py_DECREF(array);
                    return NULL;
                }
                keys_pos = lookup(self, k);
                if (keys_pos < 0) {
                    Py_DECREF(array);
                    if (PyErr_Occurred()) {
                        Py_DECREF(k);
                        return NULL;
                    }
                    PyErr_SetObject(PyExc_KeyError, k);
                    Py_DECREF(k);
                    return NULL;
                }
                Py_DECREF(k);
                b[i] = (npy_int64)keys_pos;
            }
        }
    }

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;

}


# undef GET_ALL_SCALARS
# undef GET_ALL_FLEXIBLE


static inline int
append_ssize_t(
    PyObject* list,
    Py_ssize_t value)
{
    PyObject* v = PyLong_FromSsize_t(value);
    if (v == NULL) {
        return -1;
    }
    int err = PyList_Append(list, v);
    Py_DECREF(v);
    return err;
}

// Give an array of the same kind as KAT, lookup and load any keys_pos. Depends on self, key_size, key_array, table_pos, i, k, values
# define GET_ANY_SCALARS(npy_type_src, npy_type_dst, kat, lookup_func, hash_func, post_deref) \
{                                                                          \
    npy_type_dst v;                                                        \
    for (; i < key_size; i++) {                                            \
        v = post_deref(*(npy_type_src*)PyArray_GETPTR1(key_array, i));     \
        table_pos = lookup_func(self, v, hash_func(v), kat);               \
        if (table_pos < 0 || (self->table[table_pos].hash == -1)) {        \
            if (PyErr_Occurred()) {                                        \
                Py_DECREF(values);                                         \
                return NULL;                                               \
            }                                                              \
            continue;                                                      \
        }                                                                  \
        keys_pos = self->table[table_pos].keys_pos;                        \
        if (append_ssize_t(values, keys_pos)) { \
            Py_DECREF(values);                                             \
            return NULL;                                                   \
        }                                                                  \
    }                                                                      \
}                                                                          \

# define GET_ANY_FLEXIBLE(char_type, get_end_func, lookup_func, hash_func)    \
{                                                                             \
    char_type* v;                                                             \
    Py_ssize_t dt_size = PyArray_ITEMSIZE(key_array) / sizeof(char_type);\
    Py_ssize_t k_size;                                                        \
    for (; i < key_size; i++) {                                               \
        v = (char_type*)PyArray_GETPTR1(key_array, i);                        \
        k_size = get_end_func(v, dt_size) - v;                                \
        table_pos = lookup_func(self, v, k_size, hash_func(v, k_size));       \
        if (table_pos < 0 || (self->table[table_pos].hash == -1)) {           \
            if (PyErr_Occurred()) {                                           \
                Py_DECREF(values);                                            \
                return NULL;                                                  \
            }                                                                 \
            continue;                                                         \
        }                                                                     \
        keys_pos = self->table[table_pos].keys_pos;                           \
        if (append_ssize_t(values, keys_pos)) {    \
            Py_DECREF(values);                                                \
            return NULL;                                                      \
        }                                                                     \
    }                                                                         \
}                                                                             \

// Given a list or array of keys, return a list of the lookup-up integer values. If any unmatched keys are found, they are ignored. A list is always returned.
static PyObject *
fam_get_any(FAMObject *self, PyObject *key) {
    Py_ssize_t key_size = 0;
    Py_ssize_t keys_pos = -1;
    Py_ssize_t i = 0;
    PyObject* k = NULL;
    PyObject* values = NULL;

    int key_is_list;
    if (PyList_CheckExact(key)) {
        key_is_list = 1;
        key_size = PyList_GET_SIZE(key);
    }
    else if (PyArray_Check(key)) {
        key_is_list = 0;
        key_size = PyArray_SIZE((PyArrayObject *)key);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Must provide a list or array.");
        return NULL;
    }

    values = PyList_New(0);
    if (!values) {
        return NULL;
    }

    if (key_is_list) {
        for (; i < key_size; i++) {
            k = PyList_GET_ITEM(key, i); // borrow
            keys_pos = lookup(self, k);
            if (keys_pos < 0) {
                if (PyErr_Occurred()) { // only exit if exception set
                    Py_DECREF(values);
                    return NULL;
                }
                continue;
            }
            if (append_ssize_t(values, keys_pos)) {
                Py_DECREF(values);
                return NULL;
            }
        }
    }
    else {
        PyArrayObject* key_array = (PyArrayObject *)key;
        // if key is an np array of the same kind as this FAMs keys, we can do optimized lookups; otherwise, we have to go through scalar to do full branching and coercion into lookup
        int key_array_t = PyArray_TYPE(key_array);

        if (kat_is_kind(self->keys_array_type, PyArray_DESCR(key_array)->kind)) {
            Py_ssize_t table_pos;
            switch (key_array_t) {
                case NPY_INT64:
                    GET_ANY_SCALARS(npy_int64, npy_int64, KAT_INT64, lookup_hash_int, int_to_hash,);
                    break;
                case NPY_INT32:
                    GET_ANY_SCALARS(npy_int32, npy_int64, KAT_INT32, lookup_hash_int, int_to_hash,);
                    break;
                case NPY_INT16:
                    GET_ANY_SCALARS(npy_int16, npy_int64, KAT_INT16, lookup_hash_int, int_to_hash,);
                    break;
                case NPY_INT8:
                    GET_ANY_SCALARS(npy_int8, npy_int64, KAT_INT8, lookup_hash_int, int_to_hash,);
                    break;
                case NPY_UINT64:
                    GET_ANY_SCALARS(npy_uint64, npy_uint64, KAT_UINT64, lookup_hash_uint, uint_to_hash,);
                    break;
                case NPY_UINT32:
                    GET_ANY_SCALARS(npy_uint32, npy_uint64, KAT_UINT32, lookup_hash_uint, uint_to_hash,);
                    break;
                case NPY_UINT16:
                    GET_ANY_SCALARS(npy_uint16, npy_uint64, KAT_UINT16, lookup_hash_uint, uint_to_hash,);
                    break;
                case NPY_UINT8:
                    GET_ANY_SCALARS(npy_uint8, npy_uint64, KAT_UINT8, lookup_hash_uint, uint_to_hash,);
                    break;
                case NPY_FLOAT64:
                    GET_ANY_SCALARS(npy_double, npy_double, KAT_FLOAT64, lookup_hash_double, double_to_hash,);
                    break;
                case NPY_FLOAT32:
                    GET_ANY_SCALARS(npy_float, npy_double, KAT_FLOAT32, lookup_hash_double, double_to_hash,);
                    break;
                case NPY_FLOAT16:
                    GET_ANY_SCALARS(npy_half, npy_double, KAT_FLOAT16, lookup_hash_double, double_to_hash, npy_half_to_double);
                    break;
                case NPY_UNICODE:
                    GET_ANY_FLEXIBLE(Py_UCS4, ucs4_get_end_p, lookup_hash_unicode, unicode_to_hash);
                    break;
                case NPY_STRING:
                    GET_ANY_FLEXIBLE(char, char_get_end_p, lookup_hash_string, string_to_hash);
                    break;
                case NPY_DATETIME: {
                    NPY_DATETIMEUNIT key_unit = AK_dt_unit_from_array(key_array);
                    if (!kat_is_datetime_unit(self->keys_array_type, key_unit)) {
                        return values;
                    }
                    GET_ANY_SCALARS(npy_int64, npy_int64, KAT_INT64, lookup_hash_int, int_to_hash,);
                    break;
                }
            }
        }
        else {
            for (; i < key_size; i++) {
                k = PyArray_ToScalar(PyArray_GETPTR1(key_array, i), key_array);
                if (k == NULL) {
                    Py_DECREF(values);
                    return NULL;
                }
                keys_pos = lookup(self, k);
                Py_DECREF(k);
                if (keys_pos < 0) {
                    if (PyErr_Occurred()) { // only exit if exception set
                        Py_DECREF(values);
                        return NULL;
                    }
                    continue; // do not raise
                }
                if (append_ssize_t(values, keys_pos)) {
                    Py_DECREF(values);
                    return NULL;
                }
            }
        }
    }
    return values; // might be empty
}


# undef GET_ANY_SCALARS
# undef GET_ANY_FLEXIBLE


static PyObject *
fam_subscript(FAMObject *self, PyObject *key)
{
    return get(self, key, NULL);
}


static PyMappingMethods fam_as_mapping = {
    .mp_length = (lenfunc) fam_length,
    .mp_subscript = (binaryfunc) fam_subscript,
};


static PyObject *
fam_or(PyObject *left, PyObject *right)
{
    if (!PyObject_TypeCheck(left, &FAMType) ||
        !PyObject_TypeCheck(right, &FAMType)
    ) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    FAMObject *updated = copy(Py_TYPE(left), (FAMObject *)left);
    if (!updated) {
        return NULL;
    }
    if (extend(updated, ((FAMObject *)right)->keys)) {
        Py_DECREF(updated);
        return NULL;
    }
    return (PyObject *)updated;
}


static PyNumberMethods fam_as_number = {
    .nb_or = (binaryfunc) fam_or,
};


static int
fam_contains(FAMObject *self, PyObject *key)
{
    if (lookup(self, key) < 0) {
        if (PyErr_Occurred()) {
            return -1;
        }
        return 0;
    }
    return 1;
}


static PySequenceMethods fam_as_sequence = {
    .sq_contains = (objobjproc) fam_contains,
};


static void
fam_dealloc(FAMObject *self)
{
    if (self->table) {
        PyMem_Free(self->table);
    }
    if (self->key_buffer) {
        PyMem_Free(self->key_buffer);
    }
    if (self->keys) {
        Py_DECREF(self->keys);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}


// Return a hash integer for an entire FAM by combining all stored hashes
static Py_hash_t
fam_hash(FAMObject *self)
{
    Py_hash_t hash = 0;
    for (Py_ssize_t i = 0; i < self->table_size; i++) {
        hash = hash * 3 + self->table[i].hash;
    }
    if (hash == -1) { // most not return -1
        return 0;
    }
    return hash;
}


static PyObject *
fam_iter(FAMObject *self)
{
    return fami_new(self, KEYS, false);
}


static PyObject *
fam_getnewargs(FAMObject *self)
{
    return PyTuple_Pack(1, self->keys);
}


static PyObject *
fam_reversed(FAMObject *self)
{
    return fami_new(self, KEYS, true);
}


static PyObject *
fam_sizeof(FAMObject *self)
{
    PyObject *listsizeof = PyObject_CallMethod(self->keys, "__sizeof__", NULL);
    if (!listsizeof) {
        return NULL;
    }
    Py_ssize_t listbytes = PyLong_AsSsize_t(listsizeof);
    Py_DECREF(listsizeof);
    if (listbytes == -1 && PyErr_Occurred()) {
        return NULL;
    }
    return PyLong_FromSsize_t(
        Py_TYPE(self)->tp_basicsize
        + listbytes
        + (self->table_size + SCAN - 1) * sizeof(TableElement)
    );
}


static PyObject *
fam_get(FAMObject *self, PyObject *args)
{
    PyObject *key, *missing = Py_None;
    if (!PyArg_UnpackTuple(args, Py_TYPE(self)->tp_name, 1, 2, &key, &missing))
    {
        return NULL;
    }
    return get(self, key, missing);
}


static PyObject *
fam_items(FAMObject *self)
{
    return famv_new(self, ITEMS);
}


static PyObject *
fam_keys(FAMObject *self)
{
    return famv_new(self, KEYS);
}


static PyObject *
fam_values(FAMObject *self)
{
    return famv_new(self, VALUES);
}


static PyObject *
fam_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    // NOTE: The original fam_new used to be able to provide a same reference back if a fam was in the args; this is tricky now that we have fam_init
    FAMObject *self = (FAMObject *)cls->tp_alloc(cls, 0);
    if (!self) {
        return NULL;
    }
    self->table = NULL;
    self->keys = NULL;
    self->key_buffer = NULL;
    self->keys_size = 0;
    return (PyObject*)self;
}


// This macro can be used with integer and floating point NumPy types, given an `npy_type` and a specialized `insert_func`. Uses context of `fam_init` to get `fam`, `contiguous`, `a`, `keys_size`, and `i`. An optional `post_deref` function can be supplied to transform extracted values before calling the appropriate insert function.
# define INSERT_SCALARS(npy_type, insert_func, kat, post_deref)   \
{                                                                 \
    if (contiguous) {                                             \
        npy_type* b = (npy_type*)PyArray_DATA(a);                 \
        npy_type* b_end = b + keys_size;                          \
        while (b < b_end) {                                       \
            if (insert_func(fam, post_deref(*b), i, -1, kat)) {   \
                goto error;                                       \
            }                                                     \
            b++;                                                  \
            i++;                                                  \
        }                                                         \
    }                                                             \
    else {                                                        \
        for (; i < keys_size; i++) {                              \
            if (insert_func(fam,                                  \
                    post_deref(*(npy_type*)PyArray_GETPTR1(a, i)),\
                    i,                                            \
                    -1,                                           \
                    kat)) {                                       \
                goto error;                                       \
            }                                                     \
        }                                                         \
    }                                                             \
}                                                                 \

// This macro is for inserting flexible-sized types, Unicode (Py_UCS4) or strings (char). Uses context of `fam_init`.
# define INSERT_FLEXIBLE(char_type, insert_func, get_end_func)     \
{                                                                  \
    char_type* p = NULL;                                           \
    if (contiguous) {                                              \
        char_type *b = (char_type*)PyArray_DATA(a);                \
        char_type *b_end = b + keys_size * dt_size;                \
        while (b < b_end) {                                        \
            p = get_end_func(b, dt_size);                          \
            if (insert_func(fam, b, p-b, i, -1)) {                 \
                goto error;                                        \
            }                                                      \
            b += dt_size;                                          \
            i++;                                                   \
        }                                                          \
    }                                                              \
    else {                                                         \
        for (; i < keys_size; i++) {                               \
            char_type* v = (char_type*)PyArray_GETPTR1(a, i);      \
            p = get_end_func(v, dt_size);                          \
            if (insert_func(fam, v, p-v, i, -1)) {                 \
                goto error;                                        \
            }                                                      \
        }                                                          \
    }                                                              \
}                                                                  \

// Initialize an allocated FAMObject. Returns 0 on success, -1 on error.
int
fam_init(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyTypeObject* cls = Py_TYPE(self); // borrowed ref
    const char *name = cls->tp_name;
    FAMObject* fam = (FAMObject*)self;

    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "%s takes no keyword arguments", name);
        return -1;
    }

    KeysArrayType keys_array_type = KAT_LIST; // default, will override if necessary

    PyObject *keys = NULL;
    Py_ssize_t keys_size = 0;

    if (!PyArg_UnpackTuple(args, name, 0, 1, &keys)) {
        return -1;
    }

    if (!keys) {
        keys = PyList_New(0);
        if (!keys) {
            return -1;
        }
    }
    else if (PyObject_TypeCheck(keys, &FAMType)) {
        // Use `keys` as old, `self` as new, and fill from old to new. This returns the same error codes as this function.
        return copy_to_new(cls, (FAMObject*)keys, fam);
    }
    else if (PyArray_Check(keys)) {
        PyArrayObject *a = (PyArrayObject *)keys;
        if (PyArray_NDIM(a) != 1) {
            PyErr_SetString(PyExc_TypeError, "Arrays must be 1-dimensional");
            return -1;
        }

        int array_t = PyArray_TYPE(a);
        keys_size = PyArray_SIZE(a);

        if (cls != &AMType &&
                (PyTypeNum_ISINTEGER(array_t) // signed and unsigned
                || PyTypeNum_ISFLOAT(array_t)
                || PyTypeNum_ISFLEXIBLE(array_t)
                || array_t == NPY_DATETIME ))
            {
            if ((PyArray_FLAGS(a) & NPY_ARRAY_WRITEABLE)) {
                PyErr_Format(PyExc_TypeError, "Arrays must be immutable when given to a %s", name);
                return -1;
            }
            // NOTE: this might return 0 (list) given a dt64 array without a unit
            keys_array_type = at_to_kat(array_t, a);
        }

        if (keys_array_type) { // we have a usable array
            Py_INCREF(keys);
        }
        else {
            // DEBUG_MSG_OBJ("got KAT", PyLong_FromLongLong(keys_array_type));
            // if an AutoMap or an array that we do not handle, create a list
            if (array_t == NPY_DATETIME || array_t == NPY_TIMEDELTA){
                keys = PySequence_List(keys); // force scalars
            }
            else {
                keys = PyArray_ToList(a); // converts to objs
            }
            if (!keys) {
                return -1;
            }
        }
    }
    else { // assume an arbitrary iterable
        keys = PySequence_List(keys);
        if (!keys) {
            return -1;
        }
        keys_size = PyList_GET_SIZE(keys);
    }

    fam->keys = keys;
    fam->keys_array_type = keys_array_type;
    fam->keys_size = keys_size;
    fam->key_buffer = NULL;

    // NOTE: on itialization, grow_table() does not use keys
    if (grow_table(fam, keys_size)) {
        return -1;
    }
    Py_ssize_t i = 0;
    if (keys_array_type) {
        PyArrayObject *a = (PyArrayObject *)fam->keys;
        int contiguous = PyArray_IS_C_CONTIGUOUS(a);
        switch (keys_array_type) {
            case KAT_INT64:
                INSERT_SCALARS(npy_int64, insert_int, keys_array_type,);
                break;
            case KAT_INT32:
                INSERT_SCALARS(npy_int32, insert_int, keys_array_type,);
                break;
            case KAT_INT16:
                INSERT_SCALARS(npy_int16, insert_int, keys_array_type,);
                break;
            case KAT_INT8:
                INSERT_SCALARS(npy_int8, insert_int, keys_array_type,);
                break;
            case KAT_UINT64:
                INSERT_SCALARS(npy_uint64, insert_uint, keys_array_type,);
                break;
            case KAT_UINT32:
                INSERT_SCALARS(npy_uint32, insert_uint, keys_array_type,);
                break;
            case KAT_UINT16:
                INSERT_SCALARS(npy_uint16, insert_uint, keys_array_type,);
                break;
            case KAT_UINT8:
                INSERT_SCALARS(npy_uint8, insert_uint, keys_array_type,);
                break;
            case KAT_FLOAT64:
                INSERT_SCALARS(npy_double, insert_double, keys_array_type,);
                break;
            case KAT_FLOAT32:
                INSERT_SCALARS(npy_float, insert_double, keys_array_type,);
                break;
            case KAT_FLOAT16:
                INSERT_SCALARS(npy_half, insert_double, keys_array_type, npy_half_to_double);
                break;
            case KAT_UNICODE: {
                // Over allocate buffer by 1 so there is room for null at end. This buffer is only used in lookup();
                Py_ssize_t dt_size = PyArray_ITEMSIZE(a) / UCS4_SIZE;
                fam->key_buffer = (Py_UCS4*)PyMem_Malloc((dt_size+1) * UCS4_SIZE);
                INSERT_FLEXIBLE(Py_UCS4, insert_unicode, ucs4_get_end_p);
                break;
            }
            case KAT_STRING: {
                Py_ssize_t dt_size = PyArray_ITEMSIZE(a);
                INSERT_FLEXIBLE(char, insert_string, char_get_end_p);
                break;
            }
            case KAT_DTY:
            case KAT_DTM:
            case KAT_DTW:
            case KAT_DTD:
            case KAT_DTh:
            case KAT_DTm:
            case KAT_DTs:
            case KAT_DTms:
            case KAT_DTus:
            case KAT_DTns:
            case KAT_DTps:
            case KAT_DTfs:
            case KAT_DTas:
                INSERT_SCALARS(npy_int64, insert_int, KAT_INT64,);
                break;
            default:
                return -1;
        }
    }
    else {
        for (; i < keys_size; i++) {
            if (insert_obj(fam, PyList_GET_ITEM(keys, i), i, -1)) {
                goto error;
            }
        }
    }
    return 0;
error:
    // assume all dynamic memory assigned to struct attrs that will be cleaned
    return -1;
}


# undef INSERT_SCALARS
# undef INSERT_FLEXIBLE


static PyObject *
fam_repr(FAMObject *self)
{
    return PyUnicode_FromFormat("%s(%R)", Py_TYPE(self)->tp_name, self->keys);
}


static PyObject *
fam_richcompare(FAMObject *self, PyObject *other, int op)
{
    if (!PyObject_TypeCheck(other, &FAMType)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    return PyObject_RichCompare(self->keys, ((FAMObject *)other)->keys, op);
}


static PyObject*
fam_getstate(FAMObject *self)
{
    PyObject* state = PyTuple_Pack(1, self->keys);
    return state;
}


// State returned here is a tuple of keys, suitable for usage as an `args` argument.
static PyObject*
fam_setstate(FAMObject *self, PyObject *state)
{
    if (!PyTuple_CheckExact(state) || !PyTuple_GET_SIZE(state)) {
        PyErr_SetString(PyExc_ValueError, "Unexpected pickled object.");
        return NULL;
    }
    PyObject *keys = PyTuple_GetItem(state, 0);
    if (PyArray_Check(keys)) {
        // if we an array, make it immutable
        PyArray_CLEARFLAGS((PyArrayObject*)keys, NPY_ARRAY_WRITEABLE);
    }
    fam_init((PyObject*)self, state, NULL);
    Py_RETURN_NONE;
}


static PyMethodDef fam_methods[] = {
    {"__getnewargs__", (PyCFunction) fam_getnewargs, METH_NOARGS, NULL},
    {"__reversed__", (PyCFunction) fam_reversed, METH_NOARGS, NULL},
    {"__sizeof__", (PyCFunction) fam_sizeof, METH_NOARGS, NULL},
    {"__getstate__", (PyCFunction) fam_getstate, METH_NOARGS, NULL},
    {"__setstate__", (PyCFunction) fam_setstate, METH_O, NULL},
    {"get", (PyCFunction) fam_get, METH_VARARGS, NULL},
    {"items", (PyCFunction) fam_items, METH_NOARGS, NULL},
    {"keys", (PyCFunction) fam_keys, METH_NOARGS, NULL},
    {"values", (PyCFunction) fam_values, METH_NOARGS, NULL},
    {"get_all", (PyCFunction) fam_get_all, METH_O, NULL},
    {"get_any", (PyCFunction) fam_get_any, METH_O, NULL},
    {NULL},
};


PyTypeObject FAMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_mapping = &fam_as_mapping,
    .tp_as_number = &fam_as_number,
    .tp_as_sequence = &fam_as_sequence,
    .tp_basicsize = sizeof(FAMObject),
    .tp_dealloc = (destructor) fam_dealloc,
    .tp_doc = "An immutable auto-incremented integer-valued mapping.",
    .tp_hash = (hashfunc) fam_hash,
    .tp_iter = (getiterfunc) fam_iter,
    .tp_methods = fam_methods,
    .tp_name = "arraykit.FrozenAutoMap",
    .tp_new = fam_new,
    .tp_init = fam_init,
    .tp_repr = (reprfunc) fam_repr,
    .tp_richcompare = (richcmpfunc) fam_richcompare,
};


//------------------------------------------------------------------------------
// AutoMap subclass

static PyObject *
am_inplace_or(FAMObject *self, PyObject *other)
{
    if (PyObject_TypeCheck(other, &FAMType)) {
        other = ((FAMObject *)other)->keys;
    }
    if (extend(self, other)) {
        return NULL;
    }
    Py_INCREF(self);
    return (PyObject *)self;
}


static PyNumberMethods am_as_number = {
    .nb_inplace_or = (binaryfunc) am_inplace_or,
};


static PyObject *
am_add(FAMObject *self, PyObject *other)
{
    if (append(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
am_update(FAMObject *self, PyObject *other)
{
    if (PyObject_TypeCheck(other, &FAMType)) {
        other = ((FAMObject *)other)->keys;
    }
    if (extend(self, other)) {
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyMethodDef am_methods[] = {
    {"add", (PyCFunction) am_add, METH_O, NULL},
    {"update", (PyCFunction) am_update, METH_O, NULL},
    {NULL},
};

PyTypeObject AMType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_as_number = &am_as_number,
    .tp_base = &FAMType,
    .tp_doc = "A grow-only autoincremented integer-valued mapping.",
    .tp_methods = am_methods,
    .tp_name = "arraykit.AutoMap",
    .tp_richcompare = (richcmpfunc) fam_richcompare,
};

