# include "Python.h"
# include "structmember.h"
# include "limits.h"
# include "stdbool.h"

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

// Placeholder of not implemented pathways / debugging.
# define AK_NOT_IMPLEMENTED(msg)\
    do {\
        PyErr_Format(PyExc_NotImplementedError,\
                msg);\
        return NULL;\
    } while (0)


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
//------------------------------------------------------------------------------
// CodePointLine: Type, New, Destrctor

typedef struct {
    Py_ssize_t buffer_count; // accumulated number of code points
    Py_ssize_t buffer_capacity; // max number of code points
    Py_UCS4 *buffer;

    Py_ssize_t offsets_count; // accumulated number of elements
    Py_ssize_t offsets_capacity; // max number of elements
    Py_ssize_t *offsets;

    Py_UCS4 *pos_current;
    Py_UCS4 *pos_end;
    Py_ssize_t index_current;

} AK_CodePointLine;

AK_CodePointLine* AK_CPL_New()
{
    AK_CodePointLine *cpl = (AK_CodePointLine*)PyMem_Malloc(sizeof(AK_CodePointLine));
    // TODO: handle error
    cpl->buffer_count = 0;
    cpl->buffer_capacity = 1000;
    cpl->buffer = (Py_UCS4*)PyMem_Malloc(sizeof(Py_UCS4) * cpl->buffer_capacity);
    // TODO: handle error
    cpl->pos_current = cpl->buffer;
    cpl->pos_end = cpl->buffer + cpl->buffer_capacity;

    cpl->offsets_count = 0;
    cpl->offsets_capacity = 500;
    cpl->offsets = (Py_ssize_t*)PyMem_Malloc(
            sizeof(Py_ssize_t) * cpl->offsets_capacity);
    // TODO: handle error
    cpl->index_current = 0;

    return cpl;
}

void AK_CPL_Free(AK_CodePointLine* cpl)
{
    PyMem_Free(cpl->buffer);
    PyMem_Free(cpl->offsets);
    PyMem_Free(cpl);
}

//------------------------------------------------------------------------------
// CodePointLine: Mutation

int AK_CPL_Append(AK_CodePointLine* cpl, PyObject* element)
{
    Py_ssize_t element_length = PyUnicode_GET_LENGTH(element);

    if ((cpl->buffer_count + element_length) >= cpl->buffer_capacity) {
        // realloc
        cpl->buffer_capacity *= 2; // needs to be max of this or element_length
        cpl->buffer = PyMem_Realloc(cpl->buffer,
                sizeof(Py_UCS4) * cpl->buffer_capacity);
        // TODO: handle error
        cpl->pos_end = cpl->buffer + cpl->buffer_capacity;
        cpl->pos_current = cpl->buffer + cpl->buffer_count;
    }
    if (cpl->offsets_count == cpl->offsets_capacity) {
        // realloc
        cpl->offsets_capacity *= 2;
        cpl->offsets = PyMem_Realloc(cpl->offsets,
                sizeof(Py_ssize_t) * cpl->offsets_capacity);
        // TODO: handle error
    }
    // use PyUnicode_CheckExact

    if(!PyUnicode_AsUCS4(element,
            cpl->pos_current,
            cpl->pos_end - cpl->pos_current,
            0)) { // last zero means do not copy null
        return -1; // need to handle error
    }
    // read offset_count, then increment
    cpl->offsets[cpl->offsets_count++] = element_length;

    cpl->buffer_count += element_length;
    cpl->pos_current += element_length; // add to pointer
    return 0;
}

//------------------------------------------------------------------------------
// CodePointLine: Constructors

AK_CodePointLine* AK_CPL_FromIterable(PyObject* iterable)
{
    PyObject *iter = PyObject_GetIter(iterable);
    // TODO: error handle

    AK_CodePointLine *cpl = AK_CPL_New();
    // TODO: handle error

    PyObject *element;
    while ((element = PyIter_Next(iter))) {
        AK_CPL_Append(cpl, element);
        // TODO: handle error
        Py_DECREF(element);
    }
    // TODO: handle error

    Py_DECREF(iter);
    return cpl;
}

//------------------------------------------------------------------------------
// CodePointLine: Navigation

void AK_CPL_CurrentReset(AK_CodePointLine* cpl)
{
    cpl->pos_current = cpl->buffer;
    cpl->index_current = 0;
}

static inline void AK_CPL_CurrentAdvance(AK_CodePointLine* cpl)
{
    // use index_current, then increment
    cpl->pos_current += cpl->offsets[cpl->index_current++];
}

// static inline void AK_CPL_CurrentRetreat(AK_CodePointLine* cpl)
// {
//     if (cpl->index_current > 0) {
//         // decrement index_current, then use
//         // remove the offset at this new position
//         cpl->pos_current -= cpl->offsets[--cpl->index_current];
//     }
// }

//------------------------------------------------------------------------------
// CodePointLine: Element exporters

// Return a null-terminated char array found at the curent position; this will need to be freed
char* AK_CPL_ToNewChars(AK_CodePointLine* cpl)
{
    int points = cpl->offsets[cpl->index_current];
    char* post = (char*)PyMem_Malloc(sizeof(char) * (points + 1));
    // TODO: mask values before insertion
    Py_UCS4 *p = cpl->pos_current;
    for (int i=0; i<points; ++i) {
        post[i] = *p++; // need to masks to size of char
    }
    post[points] = '\0';
    return post;
}


//------------------------------------------------------------------------------
// CodePointLine: Code Point Parsers

static char* TRUE_LOWER = "true";
static char* TRUE_UPPER = "TRUE";
static char* FALSE_LOWER = "false";
static char* FALSE_UPPER = "FALSE";

#define isspace_ascii(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define isdigit_ascii(c) (((unsigned)(c) - '0') < 10u)

#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3


// Extended from pandas/_libs/src/parser/tokenizer.c
static inline int64_t UCS4_to_int64(Py_UCS4 *p_item, Py_UCS4 *end, int *error)
{
    char tsep = '\0'; // thousands seperator; if null processing is skipped
    int64_t int_min = LONG_LONG_MIN;
    int64_t int_max = LONG_LONG_MAX;
    Py_UCS4 *p = p_item;
    int isneg = 0;
    int64_t number = 0;
    int d;
    // Skip leading spaces.
    while (isspace_ascii(*p)) {
        ++p;
        if (p >= end) {return number;}
    }
    // Handle sign.
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        ++p;
    }
    if (p >= end) {return number;}

    // Check that there is a first digit.
    if (!isdigit_ascii(*p)) {
        *error = ERROR_NO_DIGITS;
        return 0;
    }
    if (isneg) {
        // If number is greater than pre_min, at least one more digit can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        int64_t pre_min = int_min / 10;
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((number > pre_min) ||
                    ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                    number = number * 10 - (d - '0');
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((number > pre_min) ||
                    ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                    number = number * 10 - (d - '0');
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        }
    } else {
        // If number is less than pre_max, at least one more digit can be processed without overflowing.
        int64_t pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                    continue;
                } else if (!isdigit_ascii(d)) {
                    break;
                }
                if ((number < pre_max) ||
                    ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                    number = number * 10 + (d - '0');
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        } else {
            while (isdigit_ascii(d)) {
                if ((number < pre_max) ||
                    ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                    number = number * 10 + (d - '0');
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        }
    }
    // Skip trailing spaces.
    // while (isspace_ascii(*p)) {
    //     ++p;
    //     if (p >= end) {return number;}
    // }
    // // Did we use up all the characters?
    // if (*p) {
    //     *error = ERROR_INVALID_CHARS;
    //     return 0;
    // }
    *error = 0;
    return number;
}


// This will take any case of "TRUE" as True, while marking everything else as False; this is the same approach taken with genfromtxt when the dtype is given as bool. This will not fail for invalid true or false strings.
// NP's Boolean conversion in genfromtxt: https://github.com/numpy/numpy/blob/0721406ede8b983b8689d8b70556499fc2aea28a/numpy/lib/_iotools.py#L386
static inline int AK_CPL_IsTrue(AK_CodePointLine* cpl) {
    // must have at least 1 characters
    if (cpl->offsets[cpl->index_current] < 4) {
        return 0;
    }
    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + 4; // we must have at least 4 characters
    int i = 0;
    char c;

    for (;p < end; ++p) {
        c = *p;
        if (c == TRUE_LOWER[i] || c == TRUE_UPPER[i]) {
            ++i;
        }
        else {
            return 0;
        }
    }
    return 1; //matched all characters
}

// Parse full field and return 1 or 0 only for valid Boolean strings. Return 1 if True, 0 if False, -1 anything else
static inline int AK_CPL_ParseBoolean(AK_CodePointLine* cpl) {
    // must have only 4 or 5 characters
    Py_ssize_t size = cpl->offsets[cpl->index_current];
    if (size < 4 || size > 5) {
        return -1;
    }

    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + size; // size is either 4 or 5

    int score = 0;
    int i = 0;
    char c;

    for (;p < end; ++p) {
        c = *p;
        if (score >= 0 && (c == TRUE_LOWER[i] || c == TRUE_UPPER[i])) {
            ++score;
        }
        else if (score <= 0 && (c == FALSE_LOWER[i] || c == FALSE_UPPER[i])) {
            --score;
        }
        else {
            return -1;
        }
        ++i;
    }
    // we might not need to look a the score value
    if (score == 4) {
       return 1;
    }
    else if (score == -5) {
        return 0;
    }
    return -1;
}

static inline int64_t AK_CPL_ParseLong(AK_CodePointLine* cpl)
{
    // char* c = AK_CPL_ToNewChars(cpl);
    // long v = PyOS_strtol(c, NULL, 10);
    // PyMem_Free(c);
    // return v;

    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + cpl->offsets[cpl->index_current]; // size is either 4 or 5
    int error;
    int64_t v = UCS4_to_int64(p, end, &error);
    return v;

}

//------------------------------------------------------------------------------
// CodePointLine: Exporters

static inline PyObject* AK_CPL_ToArrayBoolean(AK_CodePointLine* cpl)
{
    npy_intp dims[] = {cpl->offsets_count};
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
    // TODO: check error

    // assuming this is contiguous
    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    // TODO: check error

    npy_bool *array_buffer = (npy_bool*)PyArray_DATA((PyArrayObject*)array);

    AK_CPL_CurrentReset(cpl);

    for (int i=0; i < cpl->offsets_count; ++i) {
        // this is forgiving in that invalid strings remain false
        if (AK_CPL_IsTrue(cpl)) {
            array_buffer[i] = 1;
        }
        AK_CPL_CurrentAdvance(cpl);
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}


static inline PyObject* AK_CPL_ToArrayLong(AK_CodePointLine* cpl)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_LONGLONG);
    // TODO: check error

    // assuming this is contiguous
    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    // TODO: check error

    npy_longlong *array_buffer = (npy_longlong*)PyArray_DATA((PyArrayObject*)array);
    npy_longlong *end = array_buffer + count;

    AK_CPL_CurrentReset(cpl);
    while (array_buffer < end) {
        *array_buffer++ = AK_CPL_ParseLong(cpl);
        AK_CPL_CurrentAdvance(cpl);
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}



// Returns a new reference.
PyObject* AK_CPL_ToUnicode(AK_CodePointLine* cpl)
{
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
            cpl->buffer,
            cpl->buffer_count);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// CodePointGrid Type, New, Destrctor

typedef struct {
    Py_ssize_t lines_count; // accumulated number of code points
    Py_ssize_t lines_capacity; // max number of code points
    AK_CodePointLine **lines; // array of pointers
} AK_CodePointGrid;

AK_CodePointGrid* AK_CPG_New()
{
    AK_CodePointGrid *cpg = (AK_CodePointGrid*)PyMem_Malloc(sizeof(AK_CodePointGrid));
    cpg->lines_count = 0;
    cpg->lines_capacity = 100;
    cpg->lines = (AK_CodePointLine**)PyMem_Malloc(
            sizeof(AK_CodePointLine*) * cpg->lines_capacity);
    // NOTE: initialize lines to NULL?
    return cpg;
}

void AK_CPG_Free(AK_CodePointGrid* cpg)
{
    for (int i=0; i < cpg->lines_count; ++i) {
        AK_CPL_Free(cpg->lines[i]);
    }
    PyMem_Free(cpg->lines);
    PyMem_Free(cpg);
}
//------------------------------------------------------------------------------
// CodePointGrid: Mutation

static inline int AK_CPG_AppendAtLine(
        AK_CodePointGrid* cpg,
        int line,
        PyObject* element)
{
    if (line >= cpg->lines_capacity) {
        cpg->lines_capacity *= 2;
        // NOTE: as we sure we are only copying pointers?
        cpg->lines = PyMem_Realloc(cpg->lines,
                sizeof(AK_CodePointLine*) * cpg->lines_capacity);
        // TODO: handle error, initialize lines to NULL
    }
    // for now we assume sequential acesss, so should only check if equal
    if (line >= cpg->lines_count) {
        // initialize a CPL in this position
        cpg->lines[line] = AK_CPL_New();
        ++cpg->lines_count;
    }
    AK_CPL_Append(cpg->lines[line], element);
    // handle failure
    return 1;
}

//------------------------------------------------------------------------------
// CodePointGrid: Constructors

// Given an iterable, load a CPG. If axis is 0, interpret the first level of as the primary level (rows become columns); if axis is 1, align values by position per row (rows are partitioned into columns).
AK_CodePointGrid* AK_CPG_FromIterable(
        PyObject* iterable,
        int axis)
{
    AK_CodePointGrid* cpg = AK_CPG_New();
    // expect an iterable of iterables
    PyObject *outer_iter = PyObject_GetIter(iterable);
    // TODO: handle error
    PyObject *outer;
    PyObject *inner_iter;
    PyObject *inner;

    int inner_count;
    int outer_count = 0;

    int *count_src = axis == 0 ? &outer_count : &inner_count;

    while ((outer = PyIter_Next(outer_iter))) {
        inner_iter = PyObject_GetIter(outer);
        // TODO: handle error
        inner_count = 0;

        while ((inner = PyIter_Next(inner_iter))) {
            AK_CPG_AppendAtLine(cpg, *count_src, inner);
            // TODO: handle error
            ++inner_count;
            Py_DECREF(inner);
        }
        ++outer_count;
        Py_DECREF(outer);
        Py_DECREF(inner_iter);
    }
    Py_DECREF(outer_iter);
    return cpg;
}

//------------------------------------------------------------------------------
// CPL: Exporters

PyObject* AK_CPG_ToUnicodeList(AK_CodePointGrid* cpg)
{
    PyObject* list = PyList_New(0);
    // handle error
    for (int i = 0; i < cpg->lines_count; ++i) {
        if (PyList_Append(list, AK_CPL_ToUnicode(cpg->lines[i]))) {
           // handle error
        }
    }
    return list;
}

PyObject* AK_CPG_ToArrayList(AK_CodePointGrid* cpg, PyObject* dtypes)
{
    PyObject* list = PyList_New(cpg->lines_count);
    // handle error
    for (int i = 0; i < cpg->lines_count; ++i) {

        PyObject* dtype_specifier = PyList_GetItem(dtypes, i);
        if (!dtype_specifier) {
            Py_DECREF(list);
            return NULL;
        }

        PyArray_Descr* dtype;
        if (PyObject_TypeCheck(dtype_specifier, &PyArrayDescr_Type)) {
            dtype = (PyArray_Descr* )dtype_specifier;
        }
        else { // converter2 set NULL for None
            PyArray_DescrConverter2(dtype_specifier, &dtype);
        }

        PyObject* array;

        if (!dtype) {
            AK_NOT_IMPLEMENTED("no handling for undefined dtype yet");
        }
        else if (PyDataType_ISBOOL(dtype)) {
            array = AK_CPL_ToArrayBoolean(cpg->lines[i]);
        }
        else if (PyDataType_ISINTEGER(dtype)) {
            array = AK_CPL_ToArrayLong(cpg->lines[i]);
        }
        else if (PyDataType_ISCOMPLEX(dtype)) {
            AK_NOT_IMPLEMENTED("no handling for complex dtype yet");
        }
        else {
            AK_NOT_IMPLEMENTED("no handling for other dtypes yet");
        }

        if (PyList_SetItem(list, i, array)) { // steals reference
           // handle error
        }
    }
    return list;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// CSV from CPython

// typedef struct {
//     PyObject *error_obj;   /* CSV exception */
//     PyObject *dialects;   /* Dialect registry */
//     PyTypeObject *dialect_type;
//     PyTypeObject *reader_type;
//     PyTypeObject *writer_type;
//     long field_limit;   /* max parsed field size */
// } _csvstate;

typedef enum {
    START_RECORD, START_FIELD, ESCAPED_CHAR, IN_FIELD,
    IN_QUOTED_FIELD, ESCAPE_IN_QUOTED_FIELD, QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL,AFTER_ESCAPED_CRNL
} ParserState;

typedef enum {
    QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE
} QuoteStyle;

typedef struct {
    QuoteStyle style;
    const char *name;
} StyleDesc;

static const StyleDesc quote_styles[] = {
    { QUOTE_MINIMAL,    "QUOTE_MINIMAL" },
    { QUOTE_ALL,        "QUOTE_ALL" },
    { QUOTE_NONNUMERIC, "QUOTE_NONNUMERIC" },
    { QUOTE_NONE,       "QUOTE_NONE" },
    { 0 }
};

typedef struct {
    PyObject_HEAD

    char doublequote;           /* is " represented by ""? */
    char skipinitialspace;      /* ignore spaces following delimiter? */
    char strict;                /* raise exception on bad CSV */
    int quoting;                /* style of quoting to write */
    Py_UCS4 delimiter;          /* field separator */
    Py_UCS4 quotechar;          /* quote character */
    Py_UCS4 escapechar;         /* escape character */
    PyObject *lineterminator;   /* string to write between records */

} DialectObj;

typedef struct {
    PyObject_HEAD

    PyObject *input_iter;   /* iterate over this for input lines */

    DialectObj *dialect;    /* parsing dialect */

    PyObject *fields;           /* field list for current record */
    ParserState state;          /* current CSV parse state */
    Py_UCS4 *field;             /* temporary buffer */
    Py_ssize_t field_size;      /* size of allocated buffer */
    Py_ssize_t field_len;       /* length of current field */
    int numeric_field;          /* treat field as numeric */
    unsigned long line_num;     /* Source-file line number */
} ReaderObj;

// dialect

static PyObject *
get_nullchar_as_None(Py_UCS4 c)
{
    if (c == '\0') {
        Py_RETURN_NONE;
    }
    else
        return PyUnicode_FromOrdinal(c);
}

static PyObject *
Dialect_get_lineterminator(DialectObj *self, void *Py_UNUSED(ignored))
{
    Py_XINCREF(self->lineterminator);
    return self->lineterminator;
}

static PyObject *
Dialect_get_delimiter(DialectObj *self, void *Py_UNUSED(ignored))
{
    return get_nullchar_as_None(self->delimiter);
}

static PyObject *
Dialect_get_escapechar(DialectObj *self, void *Py_UNUSED(ignored))
{
    return get_nullchar_as_None(self->escapechar);
}

static PyObject *
Dialect_get_quotechar(DialectObj *self, void *Py_UNUSED(ignored))
{
    return get_nullchar_as_None(self->quotechar);
}

static PyObject *
Dialect_get_quoting(DialectObj *self, void *Py_UNUSED(ignored))
{
    return PyLong_FromLong(self->quoting);
}



static int
_set_bool(const char *name, char *target, PyObject *src, bool dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        int b = PyObject_IsTrue(src);
        if (b < 0)
            return -1;
        *target = (char)b;
    }
    return 0;
}

static int
_set_int(const char *name, int *target, PyObject *src, int dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        int value;
        if (!PyLong_CheckExact(src)) {
            PyErr_Format(PyExc_TypeError,
                         "\"%s\" must be an integer", name);
            return -1;
        }
        value = _PyLong_AsInt(src);
        if (value == -1 && PyErr_Occurred()) {
            return -1;
        }
        *target = value;
    }
    return 0;
}

static int
_set_char(const char *name, Py_UCS4 *target, PyObject *src, Py_UCS4 dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        *target = '\0';
        if (src != Py_None) {
            Py_ssize_t len;
            if (!PyUnicode_Check(src)) {
                PyErr_Format(PyExc_TypeError,
                    "\"%s\" must be string, not %.200s", name,
                    Py_TYPE(src)->tp_name);
                return -1;
            }
            len = PyUnicode_GetLength(src);
            if (len > 1) {
                PyErr_Format(PyExc_TypeError,
                    "\"%s\" must be a 1-character string",
                    name);
                return -1;
            }
            /* PyUnicode_READY() is called in PyUnicode_GetLength() */
            if (len > 0)
                *target = PyUnicode_READ_CHAR(src, 0);
        }
    }
    return 0;
}

static int
_set_str(const char *name, PyObject **target, PyObject *src, const char *dflt)
{
    if (src == NULL)
        *target = PyUnicode_DecodeASCII(dflt, strlen(dflt), NULL);
    else {
        if (src == Py_None)
            *target = NULL;
        else if (!PyUnicode_Check(src)) {
            PyErr_Format(PyExc_TypeError,
                         "\"%s\" must be a string", name);
            return -1;
        }
        else {
            if (PyUnicode_READY(src) == -1)
                return -1;
            Py_INCREF(src);
            Py_XSETREF(*target, src);
        }
    }
    return 0;
}

static int
dialect_check_quoting(int quoting)
{
    const StyleDesc *qs;

    for (qs = quote_styles; qs->name; qs++) {
        if ((int)qs->style == quoting)
            return 0;
    }
    PyErr_Format(PyExc_TypeError, "bad \"quoting\" value");
    return -1;
}

#define D_OFF(x) offsetof(DialectObj, x)

static struct PyMemberDef Dialect_memberlist[] = {
    { "skipinitialspace",   T_BOOL, D_OFF(skipinitialspace), READONLY },
    { "doublequote",        T_BOOL, D_OFF(doublequote), READONLY },
    { "strict",             T_BOOL, D_OFF(strict), READONLY },
    { NULL }
};

static PyGetSetDef Dialect_getsetlist[] = {
    { "delimiter",          (getter)Dialect_get_delimiter},
    { "escapechar",             (getter)Dialect_get_escapechar},
    { "lineterminator",         (getter)Dialect_get_lineterminator},
    { "quotechar",              (getter)Dialect_get_quotechar},
    { "quoting",                (getter)Dialect_get_quoting},
    {NULL},
};

static void
Dialect_dealloc(DialectObj *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    Py_CLEAR(self->lineterminator);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

static void
Dialect_finalize(DialectObj *self)
{
    Py_CLEAR(self->lineterminator);
}

static char *dialect_kws[] = {
    "dialect",
    "delimiter",
    "doublequote",
    "escapechar",
    "lineterminator",
    "quotechar",
    "quoting",
    "skipinitialspace",
    "strict",
    NULL
};


// static _csvstate *
// _csv_state_from_type(PyTypeObject *type, const char *name)
// {
//     PyObject *module = _PyType_GetModuleByDef(type, &_csvmodule);
//     if (module == NULL) {
//         return NULL;
//     }
//     _csvstate *module_state = PyModule_GetState(module);
//     if (module_state == NULL) {
//         PyErr_Format(PyExc_SystemError,
//                      "%s: No _csv module state found", name);
//         return NULL;
//     }
//     return module_state;
// }

static PyObject *
dialect_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    DialectObj *self;
    PyObject *ret = NULL;

    PyObject *dialect = NULL; // TODO: remove this argument
    PyObject *delimiter = NULL;
    PyObject *doublequote = NULL;
    PyObject *escapechar = NULL;
    PyObject *lineterminator = NULL;
    PyObject *quotechar = NULL;
    PyObject *quoting = NULL;
    PyObject *skipinitialspace = NULL;
    PyObject *strict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "|OOOOOOOOO", dialect_kws,
            &dialect,
            &delimiter,
            &doublequote,
            &escapechar,
            &lineterminator,
            &quotechar,
            &quoting,
            &skipinitialspace,
            &strict))
        return NULL;

    // _csvstate *module_state = _csv_state_from_type(type, "dialect_new");
    // if (module_state == NULL) {
    //     return NULL;
    // }

    if (dialect != NULL) {
        AK_NOT_IMPLEMENTED("dialect must be None");
    }

    // if (dialect != NULL) {
    //     // if (PyUnicode_Check(dialect)) {
    //     //     dialect = get_dialect_from_registry(dialect, module_state);
    //     //     if (dialect == NULL)
    //     //         return NULL;
    //     // }
    //     // else
    //     Py_INCREF(dialect);
    //     /* Can we reuse this instance? */
    //     if (PyObject_TypeCheck(dialect, module_state->dialect_type) &&
    //         delimiter == NULL &&
    //         doublequote == NULL &&
    //         escapechar == NULL &&
    //         lineterminator == NULL &&
    //         quotechar == NULL &&
    //         quoting == NULL &&
    //         skipinitialspace == NULL &&
    //         strict == NULL)
    //         return dialect;
    // }

    self = (DialectObj *)type->tp_alloc(type, 0);
    if (self == NULL) {
        Py_CLEAR(dialect);
        return NULL;
    }
    self->lineterminator = NULL;

    Py_XINCREF(delimiter);
    Py_XINCREF(doublequote);
    Py_XINCREF(escapechar);
    Py_XINCREF(lineterminator);
    Py_XINCREF(quotechar);
    Py_XINCREF(quoting);
    Py_XINCREF(skipinitialspace);
    Py_XINCREF(strict);

//     if (dialect != NULL) {
// #define DIALECT_GETATTR(v, n)
//         if (v == NULL)
//             v = PyObject_GetAttrString(dialect, n)
//         DIALECT_GETATTR(delimiter, "delimiter");
//         DIALECT_GETATTR(doublequote, "doublequote");
//         DIALECT_GETATTR(escapechar, "escapechar");
//         DIALECT_GETATTR(lineterminator, "lineterminator");
//         DIALECT_GETATTR(quotechar, "quotechar");
//         DIALECT_GETATTR(quoting, "quoting");
//         DIALECT_GETATTR(skipinitialspace, "skipinitialspace");
//         DIALECT_GETATTR(strict, "strict");
//         PyErr_Clear();
//     }

    /* check types and convert to C values */
#define DIASET(meth, name, target, src, default) \
    if (meth(name, target, src, default)) \
        goto err
    DIASET(_set_char, "delimiter", &self->delimiter, delimiter, ',');
    DIASET(_set_bool, "doublequote", &self->doublequote, doublequote, true);
    DIASET(_set_char, "escapechar", &self->escapechar, escapechar, 0);
    DIASET(_set_str, "lineterminator", &self->lineterminator, lineterminator, "\r\n");
    DIASET(_set_char, "quotechar", &self->quotechar, quotechar, '"');
    DIASET(_set_int, "quoting", &self->quoting, quoting, QUOTE_MINIMAL);
    DIASET(_set_bool, "skipinitialspace", &self->skipinitialspace, skipinitialspace, false);
    DIASET(_set_bool, "strict", &self->strict, strict, false);

    /* validate options */
    if (dialect_check_quoting(self->quoting))
        goto err;
    if (self->delimiter == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "\"delimiter\" must be a 1-character string");
        goto err;
    }
    if (quotechar == Py_None && quoting == NULL)
        self->quoting = QUOTE_NONE;
    if (self->quoting != QUOTE_NONE && self->quotechar == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "quotechar must be set if quoting enabled");
        goto err;
    }
    if (self->lineterminator == 0) {
        PyErr_SetString(PyExc_TypeError, "lineterminator must be set");
        goto err;
    }

    ret = (PyObject *)self;
    Py_INCREF(self);
err:
    Py_CLEAR(self);
    Py_CLEAR(dialect);
    Py_CLEAR(delimiter);
    Py_CLEAR(doublequote);
    Py_CLEAR(escapechar);
    Py_CLEAR(lineterminator);
    Py_CLEAR(quotechar);
    Py_CLEAR(quoting);
    Py_CLEAR(skipinitialspace);
    Py_CLEAR(strict);
    return ret;
}

/* Since dialect is now a heap type, it inherits pickling method for
 * protocol 0 and 1 from object, therefore it needs to be overriden */

PyDoc_STRVAR(dialect_reduce_doc, "raises an exception to avoid pickling");

static PyObject *
Dialect_reduce(PyObject *self, PyObject *args) {
    PyErr_Format(PyExc_TypeError,
        "cannot pickle '%.100s' instances", _PyType_Name(Py_TYPE(self)));
    return NULL;
}

static struct PyMethodDef dialect_methods[] = {
    {"__reduce__", Dialect_reduce, METH_VARARGS, dialect_reduce_doc},
    {"__reduce_ex__", Dialect_reduce, METH_VARARGS, dialect_reduce_doc},
    {NULL, NULL}
};

// PyDoc_STRVAR(Dialect_Type_doc,
// "CSV dialect\n"
// "\n"
// "The Dialect type records CSV parsing and generation options.\n");

static PyType_Slot Dialect_Type_slots[] = {
    // {Py_tp_doc, (char*)Dialect_Type_doc},
    {Py_tp_members, Dialect_memberlist},
    {Py_tp_getset, Dialect_getsetlist},
    {Py_tp_new, dialect_new},
    {Py_tp_methods, dialect_methods},
    {Py_tp_finalize, Dialect_finalize},
    {Py_tp_dealloc, Dialect_dealloc},
    {0, NULL}
};

PyType_Spec Dialect_Type_spec = {
    .name = "Dialect",
    .basicsize = sizeof(DialectObj),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = Dialect_Type_slots,
};


/*
 * Return an instance of the dialect type, given a Python instance or kwarg
 * description of the dialect
 */
// static PyObject *
// _call_dialect(_csvstate *module_state, PyObject *dialect_inst, PyObject *kwargs)
// {
//     PyObject *type = (PyObject *)module_state->dialect_type;
//     if (dialect_inst) {
//         return _PyObject_FastCallDict(type, &dialect_inst, 1, kwargs);
//     }
//     else {
//         return _PyObject_FastCallDict(type, NULL, 0, kwargs);
//     }
// }

// reader

static int
parse_save_field(ReaderObj *self)
{
    PyObject *field;

    field = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
                                      (void *) self->field, self->field_len);
    if (field == NULL)
        return -1;
    self->field_len = 0;
    if (self->numeric_field) {
        PyObject *tmp;

        self->numeric_field = 0;
        tmp = PyNumber_Float(field);
        Py_DECREF(field);
        if (tmp == NULL)
            return -1;
        field = tmp;
    }
    if (PyList_Append(self->fields, field) < 0) {
        Py_DECREF(field);
        return -1;
    }
    Py_DECREF(field);
    return 0;
}

static int
parse_grow_buff(ReaderObj *self)
{
    assert((size_t)self->field_size <= PY_SSIZE_T_MAX / sizeof(Py_UCS4));

    Py_ssize_t field_size_new = self->field_size ? 2 * self->field_size : 4096;
    Py_UCS4 *field_new = self->field;
    PyMem_Resize(field_new, Py_UCS4, field_size_new);
    if (field_new == NULL) {
        PyErr_NoMemory();
        return 0;
    }
    self->field = field_new;
    self->field_size = field_size_new;
    return 1;
}

static int
parse_add_char(ReaderObj *self, Py_UCS4 c)
{
    // if (self->field_len >= module_state->field_limit) {
    //     PyErr_Format(PyExc_RuntimeError,
    //                  "field larger than field limit (%ld)",
    //                  module_state->field_limit);
    //     return -1;
    // }
    if (self->field_len == self->field_size && !parse_grow_buff(self))
        return -1;
    self->field[self->field_len++] = c;
    return 0;
}

static int
parse_process_char(ReaderObj *self, Py_UCS4 c)
{
    DialectObj *dialect = self->dialect;

    switch (self->state) {
    case START_RECORD: /* start of record */
        if (c == '\0')
            /* empty line - return [] */
            break;
        else if (c == '\n' || c == '\r') {
            self->state = EAT_CRNL;
            break;
        }
        /* normal character - handle as START_FIELD */
        self->state = START_FIELD;
        /* fallthru */
    case START_FIELD: /* expecting field */
        if (c == '\n' || c == '\r' || c == '\0') {
            /* save empty field - return [fields] */
            if (parse_save_field(self) < 0)
                return -1;
            self->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->quotechar &&
                 dialect->quoting != QUOTE_NONE) { /* start quoted field */
            self->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->escapechar) { /* possible escaped character */
            self->state = ESCAPED_CHAR;
        }
        else if (c == ' ' && dialect->skipinitialspace)
            /* ignore space at start of field */
            ;
        else if (c == dialect->delimiter) { /* save empty field */
            if (parse_save_field(self) < 0)
                return -1;
        }
        else { /* begin new unquoted field */
            if (dialect->quoting == QUOTE_NONNUMERIC)
                self->numeric_field = 1;
            if (parse_add_char(self, c) < 0)
                return -1;
            self->state = IN_FIELD;
        }
        break;

    case ESCAPED_CHAR:
        if (c == '\n' || c=='\r') {
            if (parse_add_char(self, c) < 0)
                return -1;
            self->state = AFTER_ESCAPED_CRNL;
            break;
        }
        if (c == '\0')
            c = '\n';
        if (parse_add_char(self, c) < 0)
            return -1;
        self->state = IN_FIELD;
        break;

    case AFTER_ESCAPED_CRNL:
        if (c == '\0')
            break;
        /*fallthru*/

    case IN_FIELD: /* in unquoted field */
        if (c == '\n' || c == '\r' || c == '\0') {
            /* end of line - return [fields] */
            if (parse_save_field(self) < 0)
                return -1;
            self->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->escapechar) { /* possible escaped character */
            self->state = ESCAPED_CHAR;
        }
        else if (c == dialect->delimiter) { /* save field - wait for new field */
            if (parse_save_field(self) < 0)
                return -1;
            self->state = START_FIELD;
        }
        else { /* normal character - save in field */
            if (parse_add_char(self, c) < 0)
                return -1;
        }
        break;

    case IN_QUOTED_FIELD: /* in quoted field */
        if (c == '\0')
            ;
        else if (c == dialect->escapechar) { /* Possible escape character */
            self->state = ESCAPE_IN_QUOTED_FIELD;
        }
        else if (c == dialect->quotechar &&
                 dialect->quoting != QUOTE_NONE) {
            if (dialect->doublequote) { /* doublequote; " represented by "" */
                self->state = QUOTE_IN_QUOTED_FIELD;
            }
            else { /* end of quote part of field */
                self->state = IN_FIELD;
            }
        }
        else { /* normal character - save in field */
            if (parse_add_char(self, c) < 0)
                return -1;
        }
        break;

    case ESCAPE_IN_QUOTED_FIELD:
        if (c == '\0')
            c = '\n';
        if (parse_add_char(self, c) < 0)
            return -1;
        self->state = IN_QUOTED_FIELD;
        break;

    case QUOTE_IN_QUOTED_FIELD:
        /* doublequote - seen a quote in a quoted field */
        if (dialect->quoting != QUOTE_NONE &&
            c == dialect->quotechar) {
            /* save "" as " */
            if (parse_add_char(self, c) < 0)
                return -1;
            self->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->delimiter) { /* save field - wait for new field */
            if (parse_save_field(self) < 0)
                return -1;
            self->state = START_FIELD;
        }
        else if (c == '\n' || c == '\r' || c == '\0') {
            /* end of line - return [fields] */
            if (parse_save_field(self) < 0)
                return -1;
            self->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (!dialect->strict) {
            if (parse_add_char(self, c) < 0)
                return -1;
            self->state = IN_FIELD;
        }
        else { /* illegal */
            PyErr_Format(PyExc_RuntimeError, "'%c' expected after '%c'",
                            dialect->delimiter,
                            dialect->quotechar);
            return -1;
        }
        break;

    case EAT_CRNL:
        if (c == '\n' || c == '\r')
            ;
        else if (c == '\0')
            self->state = START_RECORD;
        else {
            PyErr_Format(PyExc_RuntimeError,
                         "new-line character seen in unquoted field - do you need to open the file in universal-newline mode?");
            return -1;
        }
        break;

    }
    return 0;
}

static int
parse_reset(ReaderObj *self)
{
    Py_XSETREF(self->fields, PyList_New(0));
    if (self->fields == NULL)
        return -1;
    self->field_len = 0;
    self->state = START_RECORD;
    self->numeric_field = 0;
    return 0;
}

static PyObject *
Reader_iternext(ReaderObj *self)
{
    PyObject *fields = NULL;
    Py_UCS4 c;
    Py_ssize_t pos, linelen;
    unsigned int kind;
    const void *data;
    PyObject *lineobj;

    // _csvstate *module_state = _csv_state_from_type(Py_TYPE(self),
    //                                                "Reader.__next__");
    // if (module_state == NULL) {
    //     return NULL;
    // }

    if (parse_reset(self) < 0)
        return NULL;
    do {
        lineobj = PyIter_Next(self->input_iter);
        if (lineobj == NULL) {
            /* End of input OR exception */
            if (!PyErr_Occurred() && (self->field_len != 0 ||
                                      self->state == IN_QUOTED_FIELD)) {
                if (self->dialect->strict)
                    PyErr_SetString(PyExc_RuntimeError,
                                    "unexpected end of data");
                else if (parse_save_field(self) >= 0)
                    break;
            }
            return NULL;
        }
        if (!PyUnicode_Check(lineobj)) {
            PyErr_Format(PyExc_RuntimeError,
                         "iterator should return strings, "
                         "not %.200s "
                         "(the file should be opened in text mode)",
                         Py_TYPE(lineobj)->tp_name
                );
            Py_DECREF(lineobj);
            return NULL;
        }
        if (PyUnicode_READY(lineobj) == -1) {
            Py_DECREF(lineobj);
            return NULL;
        }
        ++self->line_num;
        kind = PyUnicode_KIND(lineobj);
        data = PyUnicode_DATA(lineobj);
        pos = 0;
        linelen = PyUnicode_GET_LENGTH(lineobj);
        while (linelen--) {
            c = PyUnicode_READ(kind, data, pos);
            if (c == '\0') {
                Py_DECREF(lineobj);
                PyErr_Format(PyExc_RuntimeError,
                             "line contains NUL");
                goto err;
            }
            if (parse_process_char(self, c) < 0) {
                Py_DECREF(lineobj);
                goto err;
            }
            pos++;
        }
        Py_DECREF(lineobj);
        if (parse_process_char(self, 0) < 0)
            goto err;
    } while (self->state != START_RECORD);

    fields = self->fields;
    self->fields = NULL;
err:
    return fields;
}

static void
Reader_dealloc(ReaderObj *self)
{
    PyTypeObject *tp = Py_TYPE(self);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(self->dialect);
    Py_CLEAR(self->input_iter);
    Py_CLEAR(self->fields);
    if (self->field != NULL) {
        PyMem_Free(self->field);
        self->field = NULL;
    }
    PyObject_GC_Del(self);
    Py_DECREF(tp);
}

static void
Reader_finalize(ReaderObj *self)
{
    Py_CLEAR(self->dialect);
    Py_CLEAR(self->input_iter);
    Py_CLEAR(self->fields);
    if (self->field != NULL) {
        PyMem_Free(self->field);
        self->field = NULL;
    }
}

static int
Reader_traverse(ReaderObj *self, visitproc visit, void *arg)
{
    Py_VISIT(self->dialect);
    Py_VISIT(self->input_iter);
    Py_VISIT(self->fields);
    return 0;
}

static int
Reader_clear(ReaderObj *self)
{
    Py_CLEAR(self->dialect);
    Py_CLEAR(self->input_iter);
    Py_CLEAR(self->fields);
    return 0;
}

// PyDoc_STRVAR(Reader_Type_doc,
// "CSV reader\n"
// "\n"
// "Reader objects are responsible for reading and parsing tabular data\n"
// "in CSV format.\n"
// );

static struct PyMethodDef Reader_methods[] = {
    { NULL, NULL }
};
#define R_OFF(x) offsetof(ReaderObj, x)

static struct PyMemberDef Reader_memberlist[] = {
    { "dialect", T_OBJECT, R_OFF(dialect), READONLY },
    { "line_num", T_ULONG, R_OFF(line_num), READONLY },
    { NULL }
};

static PyType_Slot Reader_Type_slots[] = {
    // {Py_tp_doc, (char*)Reader_Type_doc},
    {Py_tp_traverse, Reader_traverse},
    {Py_tp_clear, Reader_clear},
    {Py_tp_iter, PyObject_SelfIter},
    {Py_tp_iternext, Reader_iternext},
    {Py_tp_methods, Reader_methods},
    {Py_tp_members, Reader_memberlist},
    {Py_tp_finalize, Reader_finalize},
    {Py_tp_dealloc, Reader_dealloc},
    {0, NULL}
};

PyType_Spec Reader_Type_spec = {
    .name = "Reader",
    .basicsize = sizeof(ReaderObj),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .slots = Reader_Type_slots
};


static PyObject *
csv_reader(PyObject *iterable, PyObject *keyword_args)
{
    // PyObject * iterator, * dialect = NULL;
    // _csvstate *module_state = get_csv_state(module);

    // ReaderObj * self = PyObject_GC_New(
    //     ReaderObj,
    //     module_state->reader_type);

    // These were previously stored on the module object
    PyObject *type_reader = (PyObject *)PyType_FromSpec(&Reader_Type_spec);
    PyObject *type_dialect = (PyObject *)PyType_FromSpec(&Dialect_Type_spec);

    ReaderObj *self = PyObject_GC_New(ReaderObj, (PyTypeObject *)type_reader); // NOT SURE!

    if (!self)
        return NULL;

    self->dialect = NULL;
    self->fields = NULL;
    self->input_iter = NULL;
    self->field = NULL;
    self->field_size = 0;
    self->line_num = 0;

    if (parse_reset(self) < 0) {
        Py_DECREF(self);
        return NULL;
    }

    // only need to take iterator
    // if (!PyArg_UnpackTuple(args, "", 1, 2, &iterator, &dialect)) {
    //     Py_DECREF(self);
    //     return NULL;
    // }
    self->input_iter = PyObject_GetIter(iterable);
    if (self->input_iter == NULL) {
        Py_DECREF(self);
        return NULL;
    }

    // self->dialect = (DialectObj *)_call_dialect(module_state, dialect,
    //                                             keyword_args);
    self->dialect = (DialectObj *)_PyObject_FastCallDict(type_dialect, NULL, 0, keyword_args);

    if (self->dialect == NULL) {
        Py_DECREF(self);
        return NULL;
    }

    PyObject_GC_Track(self);
    return (PyObject *)self;
}


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// AK_SequenceStrToArray1DAuto(PyObject* sequence)
// Determine the type dynamically
// Ideas: keep the same sequence and mutate it in-place with Python objects when necessary
//      track observations through iteration to determine type to give to PyArray_FromAny

// Only identify things that are not true strings and need conversion: bools, int, float, complex
// genfromtxt behavior:
// if one complex and the rest are floats: complex
// if any non-numeric strings (other than nan) found with other numerics: str
// if all bool str (case insensitive): bool
// if all integer (no decimal: is this local dependent?): int
// if intger with nan: float
// nans interpreted as: empty cell, any case of "nan"; not "na"
// None is never converted to object; None and True go to string (even with dtype object)
//      'None' and 'True' given in array construction is the same, same for astype(object)


// Complex Numbers
// Python `complex()` will take any string that looks a like a float; if a "+"" is present the second component must have a "j", otherwise a "complex() arg is a malformed string" is raised. Outer parenthesis are optional, but must be balanced if present
// NP genfromtxt will interpret complex numbers with dtype None; Parenthesis are optional. With dtype complex, complex notations will be interpreted. Balanced parenthesis are not required; unbalanced parenthesis, as well as missing "j", do not raise and instead result in NaNs.
// PyObject*
// AK_SequenceStrToArray1DComplex(PyObject* sequence)
// {
//     PyObject *sequence_fast = PySequence_Fast(sequence, "could not create sequence");
//     Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence_fast);
//     if (!size) {
//         return NULL;
//     }

//     npy_intp dims[] = {size};
//     PyObject* array = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);

//     PyObject *element;
//     for (Py_ssize_t pos = 0; pos < size; ++pos)
//     {
//         element = PySequence_Fast_GET_ITEM(sequence_fast, pos); // borrowed ref
//         if (!element) {
//             return NULL;
//         }
//         PyObject *v = PyObject_CallFunctionObjArgs((PyObject *)&PyComplex_Type, element, NULL);
//         if (!v) {
//             return NULL;
//         }
//         PyArray_SETITEM((PyArrayObject *)array,
//                 PyArray_GETPTR1((PyArrayObject *)array, pos),
//                 v);
//     }
//     // NOTE: will be made immutable in caller
//     Py_DECREF(sequence_fast);
//     return array;
// }

// static inline PyObject*
// AK_SequenceStrToArray1DBoolean(PyObject* sequence)
// {
//     PyObject *sequence_fast = PySequence_Fast(sequence, "could not create sequence");
//     Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence_fast);
//     if (!size) {
//         return NULL;
//     }

//     npy_intp dims[] = {size};
//     PyArray_Descr *dtype = PyArray_DescrFromType(NPY_BOOL);
//     PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
//     PyObject *lower = PyUnicode_FromString("lower");
//     if (!lower) {
//         return NULL;
//     }

//     PyObject *element;
//     for (Py_ssize_t pos = 0; pos < size; ++pos) {
//         element = PySequence_Fast_GET_ITEM(sequence_fast, pos); // borrowed ref
//         if (!element) {
//             return NULL;
//         }
//         PyObject* element_lower = PyObject_CallMethodObjArgs(element, lower, NULL);
//         if (!element_lower) {
//             return NULL;
//         }
//         if (PyUnicode_CompareWithASCIIString(element_lower, "true") == 0) {
//             *(npy_bool *) PyArray_GETPTR1((PyArrayObject *)array, pos) = 1;
//         }
//     }
//     Py_DECREF(sequence_fast);
//     Py_DECREF(lower);
//     // NOTE: will be made immutable in caller
//     return array;
// }


// Convert an sequence of strings to a 1D array.
static inline PyObject*
AK_IterableStrToArray1D(
        PyObject *sequence,
        PyObject *dtype_specifier)
{
        // Convert specifier into a dtype if necessary
        PyArray_Descr* dtype;
        if (PyObject_TypeCheck(dtype_specifier, &PyArrayDescr_Type)) {
            dtype = (PyArray_Descr* )dtype_specifier;
        }
        else { // converter2 set NULL for None
            PyArray_DescrConverter2(dtype_specifier, &dtype);
        }

        AK_CodePointLine* cpl = AK_CPL_FromIterable(sequence);
        PyObject* array;

        if (!dtype) {
            AK_NOT_IMPLEMENTED("no handling for undefined dtype yet");
        }
        else if (PyDataType_ISBOOL(dtype)) {
            array = AK_CPL_ToArrayBoolean(cpl);
            // TODO: handle error
        }
        else if (PyDataType_ISINTEGER(dtype)) {
            array = AK_CPL_ToArrayLong(cpl);
            // TODO: handle error
        }
        else {
            AK_NOT_IMPLEMENTED("no handling for undefined dtype yet");
        }

        AK_CPL_Free(cpl);
        return array;
}


//------------------------------------------------------------------------------
// AK module public methods
//------------------------------------------------------------------------------



static PyObject *
delimited_to_arrays(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *file_like, *dtypes, *axis;
    // load axis as long (l), avoid creation below
    if (!PyArg_ParseTuple(args, "OOO:delimited_to_arrays",
            &file_like,
            &dtypes,
            &axis)) // TODO: enforce this is an int?
    {
        return NULL;
    }
    // NOTE: consider taking shape_estimate?

    // Parse text
    // For now, we import and use the CSV module directly
    PyObject *module_csv = PyImport_ImportModule("csv");
    if (!module_csv) {
        return NULL;
    }
    PyObject *reader = PyObject_GetAttrString(module_csv, "reader");
    Py_DECREF(module_csv);
    if (!reader) {
        return NULL;
    }
    // TODO: pass in full parameters for parsing
    PyObject *reader_instance = PyObject_CallFunctionObjArgs(reader, file_like, NULL);
    Py_DECREF(reader);
    if (!reader_instance) {
        return NULL;
    }

    AK_CodePointGrid* cpg = AK_CPG_FromIterable(reader_instance, PyLong_AsLong(axis));
    // TODO: handle error
    Py_DECREF(reader_instance);

    PyObject* arrays = AK_CPG_ToArrayList(cpg, dtypes);
    AK_CPG_Free(cpg);
    return arrays; // could be NULL
}

// If dtype_specifier is given, always try to return that dtype
// If dtype of object is given, leave strings
// If dtype of None is given, discover
static PyObject *
iterable_str_to_array_1d(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *iterable, *dtype_specifier;

    if (!PyArg_ParseTuple(args, "OO:iterable_str_to_array_1d",
            &iterable,
            &dtype_specifier))
    {
        return NULL;
    }
    PyObject* array = AK_IterableStrToArray1D(iterable, dtype_specifier);
    return array;
}

static PyObject *
_test(PyObject *Py_UNUSED(m), PyObject *value)
{
    AK_CodePointGrid* cpg = AK_CPG_FromIterable(value, 1);
    PyObject* post = AK_CPG_ToUnicodeList(cpg);
    AK_CPG_Free(cpg);
    return post;
}


//------------------------------------------------------------------------------
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
    {"delimited_to_arrays", delimited_to_arrays, METH_VARARGS, NULL},
    {"iterable_str_to_array_1d", iterable_str_to_array_1d, METH_VARARGS, NULL},
    {"_test", _test, METH_O, NULL},
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
