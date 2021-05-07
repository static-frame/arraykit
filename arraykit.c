# include "Python.h"
# include "structmember.h"
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

// Py_UNREACHABLE() isn't available in Python 3.6:
# ifndef Py_UNREACHABLE
# define Py_UNREACHABLE() Py_FatalError("https://xkcd.com/2200")
# endif

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
        PyErr_SetString(PyExc_NotImplementedError, msg);\
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


// Given a dtype_specifier, which might be a dtype or None, assign a fresh dtype object (or NULL) to dtype_returned. Returns 1 on success. This will never set dtype_returned to None.
static inline int
AK_DTypeFromSpecifier(PyObject *dtype_specifier, PyArray_Descr **dtype_returned)
{
    PyArray_Descr* dtype;
    if (PyObject_TypeCheck(dtype_specifier, &PyArrayDescr_Type)) {
        dtype = (PyArray_Descr* )dtype_specifier;
    }
    else { // converter2 set NULL for None
        PyArray_DescrConverter2(dtype_specifier, &dtype);
    }
    // make a copy as we will give ownership to array and might mutate
    if (dtype) {
        dtype = PyArray_DescrNew(dtype);
        // this can fail
    }
    *dtype_returned = dtype;
    return 1;
}








//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// TypePArser: Type, New, Destrctor

#define AK_is_digit(c) (((unsigned)(c) - '0') < 10u)
#define AK_is_space(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define AK_is_sign(c) (((c) == '+') || ((c) == '-'))
#define AK_is_paren_open(c) ((c) == '(')
#define AK_is_paren_close(c) ((c) == ')')
#define AK_is_decimal(c) ((c) == '.')

#define AK_is_a(c) (((c) == 'a') || ((c) == 'A'))
#define AK_is_e(c) (((c) == 'e') || ((c) == 'E'))
#define AK_is_f(c) (((c) == 'f') || ((c) == 'F'))
#define AK_is_i(c) (((c) == 'i') || ((c) == 'I'))
#define AK_is_j(c) (((c) == 'j') || ((c) == 'J'))
#define AK_is_l(c) (((c) == 'l') || ((c) == 'L'))
#define AK_is_n(c) (((c) == 'n') || ((c) == 'N'))
#define AK_is_r(c) (((c) == 'r') || ((c) == 'R'))
#define AK_is_s(c) (((c) == 's') || ((c) == 'S'))
#define AK_is_t(c) (((c) == 't') || ((c) == 'T'))
#define AK_is_u(c) (((c) == 'u') || ((c) == 'U'))

//------------------------------------------------------------------------------
typedef enum {
    TPS_UNKNOWN,
    TPS_BOOL,
    TPS_INT,
    TPS_FLOAT,
    TPS_COMPLEX,
    TPS_STRING,
    TPS_EMPTY
} AK_TypeParserState;

AK_TypeParserState
AK_TPS_resolve(AK_TypeParserState previous, AK_TypeParserState new) {
    // unlikely case
    if (new == TPS_UNKNOWN) {return TPS_STRING;}

    // propagate new if previous is unknown or empty
    if ((previous == TPS_UNKNOWN) || (previous == TPS_EMPTY)) {return new;}

    // if either are string, go to string
    if (previous == TPS_STRING || new == TPS_STRING) {return TPS_STRING;}

    // handle both new, previous bool directly
    if (previous == TPS_BOOL) {
        if (new == TPS_EMPTY || new == TPS_BOOL) {return TPS_BOOL;}
        else {return TPS_STRING;} // bool found with anything except empty is string
    }
    if (new == TPS_BOOL) {
        if (previous == TPS_EMPTY) {return TPS_BOOL;}
        else {return TPS_STRING;} // bool found with anything except empty is string
    }
    // numerical promotion
    if (previous == TPS_INT) {
        if (new == TPS_EMPTY || new == TPS_INT) {return TPS_INT;}
        if (new == TPS_FLOAT) {return TPS_FLOAT;}
        if (new == TPS_COMPLEX) {return TPS_COMPLEX;}
    }
    if (previous == TPS_FLOAT) {
        if (new == TPS_EMPTY || new == TPS_INT || new == TPS_FLOAT) {return TPS_FLOAT;}
        if (new == TPS_COMPLEX) {return TPS_COMPLEX;}
    }
    // previous == TPS_COMPLEX, new is TPS_EMPTY, TPS_INT, TPS_FLOAT, or TPS_COMPLEX
    return TPS_COMPLEX;
}

PyArray_Descr*
AK_TPS_to_dtype(AK_TypeParserState state) {
    PyArray_Descr *dtype = NULL;
    PyArray_Descr *dtype_final;

    switch(state) {
        case TPS_UNKNOWN:
            // AK_DEBUG("TPS_to_dtype: UNKNOWN");
            dtype = PyArray_DescrFromType(NPY_UNICODE);
            break;
        case TPS_EMPTY:
            dtype = PyArray_DescrFromType(NPY_FLOAT64);
            break;
        case TPS_STRING:
            // AK_DEBUG("TPS_to_dtype: STRING");
            dtype = PyArray_DescrFromType(NPY_UNICODE);
            break;
        case TPS_BOOL:
            dtype = PyArray_DescrFromType(NPY_BOOL);
            break;
        case TPS_INT:
            dtype = PyArray_DescrFromType(NPY_INT64);
            break;
        case TPS_FLOAT:
            dtype = PyArray_DescrFromType(NPY_FLOAT64);
            break;
        case TPS_COMPLEX:
            dtype = PyArray_DescrFromType(NPY_COMPLEX128);
            break;
    }
    // get a fresh instance as we might mutate
    dtype_final = PyArray_DescrNew(dtype);
    Py_DECREF(dtype);
    return dtype_final;

}

//------------------------------------------------------------------------------
typedef struct AK_TypeParser {

    bool contiguous_leading_space;
    bool previous_numeric;
    bool contiguous_numeric;

    // counting will always stop before 8 or less
    npy_int8 count_bool;
    npy_int8 count_sign;
    npy_int8 count_e;
    npy_int8 count_j;
    npy_int8 count_decimal;
    npy_int8 count_nan;
    npy_int8 count_inf;
    npy_int8 count_paren_open;
    npy_int8 count_paren_close;

    // bound by number of chars in field
    Py_ssize_t last_sign_pos; // signed
    Py_ssize_t count_leading_space;
    Py_ssize_t count_digit;
    Py_ssize_t count_not_space;

    AK_TypeParserState parsed_field;
    AK_TypeParserState parsed_line;

} AK_TypeParser;

// Initialize all state.
void AK_TP_reset_field(AK_TypeParser* tp)
{
    tp->previous_numeric = false;
    tp->contiguous_numeric = false;
    tp->contiguous_leading_space = false;

    tp->count_bool = 0;
    tp->count_sign = 0;
    tp->count_e = 0;
    tp->count_j = 0;
    tp->count_decimal = 0;
    tp->count_nan = 0;
    tp->count_inf = 0;
    tp->count_paren_open = 0;
    tp->count_paren_close = 0;

    tp->last_sign_pos = -1;
    tp->count_leading_space = 0;
    tp->count_digit = 0;
    tp->count_not_space = 0;

    tp->parsed_field = TPS_UNKNOWN;
    // NOTE: do not reset parsed_line
}

AK_TypeParser*
AK_TP_New()
{
    AK_TypeParser *tp = (AK_TypeParser*)PyMem_Malloc(sizeof(AK_TypeParser));
    // TODO: handle error
    AK_TP_reset_field(tp);
    tp->parsed_line = TPS_UNKNOWN;
    return tp;
}

void
AK_TP_Free(AK_TypeParser* tp)
{
    PyMem_Free(tp);
}

//------------------------------------------------------------------------------
// TypePArser: char, field processors

bool
AK_TP_process_char(AK_TypeParser* tp,
        char c,
        Py_ssize_t pos)
{
    if (tp->parsed_field != TPS_UNKNOWN) {
        // if parsed_field is set to anything other than TPS_UNKNOWN, we should not be calling process_char anymore
        Py_UNREACHABLE();
    }

    // evaluate space ..........................................................
    bool space = false;
    if (AK_is_space(c)) {
        if (pos == 0) {
            tp->contiguous_leading_space = true;
        }
        if (tp->contiguous_leading_space) {
            ++tp->count_leading_space;
            return true;
        }
        space = true;
    }
    else if (AK_is_paren_open(c)) {
        ++tp->count_paren_open;
        ++tp->count_leading_space;
        space = true;
        // open paren permitted only in first non-space position
        if ((pos > 0 && !tp->contiguous_leading_space) || tp->count_paren_open > 1) {
            tp->parsed_field = TPS_STRING;
            return false;
        }
    }
    else if (AK_is_paren_close(c)) {
        ++tp->count_paren_close;
        space = true;
        // NOTE: might evaluate if previous is contiguous numeric
        if (tp->count_paren_close > 1) { // CHECK!
            tp->parsed_field = TPS_STRING;
            return false;
        }
    }
    else {
        ++tp->count_not_space;
    }
    // no longer in contiguous leading space
    tp->contiguous_leading_space = false;
    Py_ssize_t pos_field = pos - tp->count_leading_space;

    // evaluate numeric, non-positional ........................................
    bool numeric = false;
    bool digit = false;

    if (space) {}
    else if (AK_is_digit(c)) {
        ++tp->count_digit;
        digit = true;
        numeric = true;
    }
    else if (AK_is_decimal(c)) {
        ++tp->count_decimal;
        if (tp->count_decimal > 2) { // complex can have 2
            tp->parsed_field = TPS_STRING;
            return false;
        }
        numeric = true;
    }
    else if (AK_is_sign(c)) {
        ++tp->count_sign;
        if (tp->count_sign > 4) { // complex can have 4
            tp->parsed_field = TPS_STRING;
            return false;
        }
        tp->last_sign_pos = pos_field;
        numeric = true;
    }
    else if (AK_is_e(c)) {
        ++tp->count_e;
        if ((pos_field == 0) || (tp->count_e > 2)) {
            // can never lead field; true or false have one E; complex can have 2
            tp->parsed_field = TPS_STRING;
            return false;
        }
        numeric = true;
    }
    else if (AK_is_j(c)) {
        ++tp->count_j;
        if ((pos_field == 0) || (tp->count_j > 1)) {
            // can never lead field; complex can have 1
            tp->parsed_field = TPS_STRING;
            return false;
        }
        numeric = true;
    }

    // evaluate contiguous numeric .............................................
    if (numeric) {
        if (pos_field == 0) {
            tp->contiguous_numeric = true;
            tp->previous_numeric = true;
        }
        if (!tp->previous_numeric) {
            tp->contiguous_numeric = false;
        }
        tp->previous_numeric = true; // this char is numeric for next eval
    }
    else { // not numeric
        // only mark as not contiguous_numeric if non space as might be trailing space
        if (tp->contiguous_numeric && !space) {
            tp->contiguous_numeric = false;
        }
        tp->previous_numeric = false;
    }

    // evaluate character positions ............................................
    if (space || digit) {
        return true;
    }
    if (tp->last_sign_pos >= 0) { // initialized to -1
        pos_field -= tp->last_sign_pos + 1;
    }

    switch (pos_field) {
        case 0:
            if      (AK_is_t(c)) {++tp->count_bool;}
            else if (AK_is_f(c)) {--tp->count_bool;}
            else if (AK_is_n(c)) {++tp->count_nan;}
            else if (AK_is_i(c)) {++tp->count_inf;}
            break;
        case 1:
            if      (AK_is_r(c)) {++tp->count_bool;} // true
            else if (AK_is_a(c)) {
                --tp->count_bool; // false
                ++tp->count_nan;
                }
            else if (AK_is_n(c)) {++tp->count_inf;}
            break;
        case 2:
            if      (AK_is_u(c)) {++tp->count_bool;}
            else if (AK_is_l(c)) {--tp->count_bool;}
            else if (AK_is_n(c)) {++tp->count_nan;}
            else if (AK_is_f(c)) {++tp->count_inf;}
            break;
        case 3:
            if      (AK_is_e(c)) {++tp->count_bool;} // true
            else if (AK_is_s(c)) {--tp->count_bool;} // false
            break;
        case 4:
            if      (AK_is_e(c)) {--tp->count_bool;} // false
            break;
    }
    // continue processing
    return true;
}



AK_TypeParserState
AK_TP_resolve_field(AK_TypeParser* tp,
        Py_ssize_t count)
{
    if (count == 0) {return TPS_EMPTY;}

    // if parsed_field was set in AK_TP_process_char
    if (tp->parsed_field != TPS_UNKNOWN) {return tp->parsed_field;}

    if (tp->count_bool == 4 && tp->count_not_space == 4) {
        return TPS_BOOL;
    }
    if (tp->count_bool == -5 && tp->count_not_space == 5) {
        return TPS_BOOL;
    }

    if (tp->contiguous_numeric) {
        if (tp->count_digit == 0) {return TPS_STRING;}

        // int
        if (tp->count_j == 0 &&
                tp->count_e == 0 &&
                tp->count_decimal == 0 &&
                tp->count_paren_close == 0 &&
                tp->count_paren_open == 0 &&
                tp->count_nan == 0 &&
                tp->count_inf == 0) {
            return TPS_INT;
        }

        // float
        if (tp->count_j == 0 &&
                tp->count_sign <= 2 &&
                tp->count_paren_close == 0 &&
                tp->count_paren_open == 0 &&
                (tp->count_decimal == 1 || tp->count_e == 1)
                ) {
            if (tp->count_sign == 2 && tp->count_e == 0) {
                return TPS_STRING;
            }
            return TPS_FLOAT;
        }
        // complex with j
        if ((tp->count_j == 1) &&
                ((tp->count_paren_close == 0 && tp->count_paren_open == 0) ||
                (tp->count_paren_close == 1 && tp->count_paren_open == 1))
                ) {
            if (tp->count_sign > 2 + tp->count_e) {
                return TPS_STRING;
            }
            return TPS_COMPLEX;
        }
        // complex with parens (no j)
        if (tp->count_j == 0 &&
                tp->count_paren_close == 1 &&
                tp->count_paren_open == 1
                ) {
            if (tp->count_sign > 2 && tp->count_e > 1) {
                return TPS_STRING;
            }
            return TPS_COMPLEX;
        }
    }
    // non contiguous numeric cases
    else if (tp->count_j == 0) {
        if (tp->count_nan == 3 && tp->count_sign + tp->count_nan == tp->count_not_space) {
            return TPS_FLOAT;
        }
        if (tp->count_inf == 3 && tp->count_sign + tp->count_inf == tp->count_not_space) {
            return TPS_FLOAT;
        }
    }
    else if (tp->count_j == 1) {
        if ((tp->count_nan == 3 || tp->count_nan == 6) && tp->count_sign + tp->count_nan + 1 == tp->count_not_space) {
            return TPS_COMPLEX;
        }
        if ((tp->count_inf == 3 || tp->count_inf == 6) && tp->count_sign + tp->count_inf + 1 == tp->count_not_space) {
            return TPS_COMPLEX;
        }
    }
    return TPS_STRING; // default
}


// After field is complete, call process_field to evaluate and set parsed_line
void
AK_TP_process_field(AK_TypeParser* tp,
        Py_ssize_t count)
{
    // resolve with previous parsed_line (or unkown if just initialized)
    tp->parsed_line = AK_TPS_resolve(tp->parsed_line, AK_TP_resolve_field(tp, count));
    AK_TP_reset_field(tp);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// CodePointLine: Type, New, Destrctor

// A AK_CodePointLine stores a contiguous buffer of Py_UCS4 without null terminators between fields. Separately, we store an array of integers, where each integer is the size of each field. The total number of fields is given by offset_count.
typedef struct AK_CodePointLine{
    // NOTE: should these be unsigned int types, like Py_uintptr_t?
    Py_ssize_t buffer_count; // accumulated number of code points
    Py_ssize_t buffer_capacity; // max number of code points
    Py_UCS4 *buffer;

    Py_ssize_t offsets_count; // accumulated number of elements
    Py_ssize_t offsets_capacity; // max number of elements
    Py_ssize_t *offsets;

    Py_ssize_t offset_max; // observe max offset found across all

    Py_UCS4 *pos_current;
    Py_ssize_t index_current;

    char *field;
    AK_TypeParser *type_parser;

} AK_CodePointLine;

AK_CodePointLine* AK_CPL_New(bool type_parse)
{
    AK_CodePointLine *cpl = (AK_CodePointLine*)PyMem_Malloc(sizeof(AK_CodePointLine));
    // TODO: handle error
    cpl->buffer_count = 0;
    cpl->buffer_capacity = 1024;
    cpl->buffer = (Py_UCS4*)PyMem_Malloc(sizeof(Py_UCS4) * cpl->buffer_capacity);
    // TODO: handle error

    cpl->offsets_count = 0;
    cpl->offsets_capacity = 512;
    cpl->offsets = (Py_ssize_t*)PyMem_Malloc(
            sizeof(Py_ssize_t) * cpl->offsets_capacity);
    // TODO: handle error

    cpl->pos_current = cpl->buffer;
    cpl->index_current = 0;
    cpl->offset_max = 0;

    // optional, dynamic values
    cpl->field = NULL;

    if (type_parse) {
        // AK_DEBUG("AK_CodePointLin calling AK_TP_New()");
        cpl->type_parser = AK_TP_New();
    }
    else {
        cpl->type_parser = NULL;
    }
    return cpl;
}

void AK_CPL_Free(AK_CodePointLine* cpl)
{
    PyMem_Free(cpl->buffer);
    PyMem_Free(cpl->offsets);
    if (cpl->field) {
        PyMem_Free(cpl->field);
    }
    if (cpl->type_parser) {
        PyMem_Free(cpl->type_parser);
    }
    PyMem_Free(cpl);
}

//------------------------------------------------------------------------------
// CodePointLine: Mutation

// Resize in place if necessary; noop if not. Return 1 on success.
static inline int
AK_CPL_resize(AK_CodePointLine* cpl, Py_ssize_t count)
{
    if ((cpl->buffer_count + count) >= cpl->buffer_capacity) {
        // realloc
        cpl->buffer_capacity *= 2; // needs to be max of this or element_length
        cpl->buffer = PyMem_Realloc(cpl->buffer,
                sizeof(Py_UCS4) * cpl->buffer_capacity);
        // TODO: handle error
        cpl->pos_current = cpl->buffer + cpl->buffer_count;
    }
    // increment by at most one, so only need to check if equal
    if (cpl->offsets_count == cpl->offsets_capacity) {
        // realloc
        cpl->offsets_capacity *= 2;
        cpl->offsets = PyMem_Realloc(cpl->offsets,
                sizeof(Py_ssize_t) * cpl->offsets_capacity);
        // TODO: handle error
    }
    return 1;
}

// Given a PyUnicode PyObject, load the string content into the CPL
int
AK_CPL_AppendObject(AK_CodePointLine* cpl, PyObject* element)
{
    Py_ssize_t element_length = PyUnicode_GET_LENGTH(element);
    if (!AK_CPL_resize(cpl, element_length)) {
        return -1;
    }
    // use PyUnicode_CheckExact
    if(!PyUnicode_AsUCS4(element,
            cpl->pos_current,
            cpl->buffer + cpl->buffer_capacity - cpl->pos_current,
            0)) { // last zero means do not copy null
        return -1; // need to handle error
    }
    // TODO: if we have a type_parser, need to process chars, then field
    if (cpl->type_parser) {
        Py_UCS4* p = cpl->pos_current;
        Py_UCS4 *end = p + element_length;
        Py_ssize_t pos = 0;
        bool run = true;
        for (; p < end; ++p) {
            run = AK_TP_process_char(cpl->type_parser, (char)*p, pos);
            if (!run) {break;}
            ++pos;
        }
        AK_TP_process_field(cpl->type_parser, element_length);
    }

    // read offset_count, then increment
    cpl->offsets[cpl->offsets_count++] = element_length;
    cpl->buffer_count += element_length;
    cpl->pos_current += element_length; // add to pointer
    if (element_length > cpl->offset_max) cpl->offset_max = element_length;
    return 1;
}

// Add a single point to a line. This does not update offsets. This is valled when updating a character.
int
AK_CPL_AppendPoint(AK_CodePointLine* cpl,
        Py_UCS4 p,
        Py_ssize_t pos)
{
    if (!AK_CPL_resize(cpl, 1)) {
        return -1;
    }
    if (cpl->type_parser) {
        AK_TP_process_char(cpl->type_parser, (char)p, pos);
    }
    *cpl->pos_current++ = p;
    ++cpl->buffer_count;
    return 1;
}

// Append to offsets. This does not update buffer lines. This is called when closing a field.
int
AK_CPL_AppendOffset(AK_CodePointLine* cpl, Py_ssize_t offset)
{
    if (!AK_CPL_resize(cpl, 1)) {
        return -1;
    }
    if (cpl->type_parser) {
        AK_TP_process_field(cpl->type_parser, offset);
    }
    cpl->offsets[cpl->offsets_count++] = offset;

    if (offset > cpl->offset_max) {cpl->offset_max = offset;}
    return 1;
}

//------------------------------------------------------------------------------
// CodePointLine: Constructors

// Given an iterable of unicode objects, load them into a AK_CodePointLine. Used for testing.
AK_CodePointLine*
AK_CPL_FromIterable(PyObject* iterable, bool type_parse)
{
    PyObject *iter = PyObject_GetIter(iterable);
    // TODO: error handle

    AK_CodePointLine *cpl = AK_CPL_New(type_parse);
    // TODO: handle error

    PyObject *element;
    while ((element = PyIter_Next(iter))) {
        AK_CPL_AppendObject(cpl, element);
        // TODO: handle error
        Py_DECREF(element);
    }
    // TODO: handle error

    Py_DECREF(iter);
    return cpl;
}

//------------------------------------------------------------------------------
// CodePointLine: Navigation

void
AK_CPL_CurrentReset(AK_CodePointLine* cpl)
{
    cpl->pos_current = cpl->buffer;
    cpl->index_current = 0;
}

static inline void
AK_CPL_CurrentAdvance(AK_CodePointLine* cpl)
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
// CodePointLine: Code Point Parsers

static char* TRUE_LOWER = "true";
static char* TRUE_UPPER = "TRUE";

#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3


// Convert a Py_UCS4 array to a signed integer. Extended from pandas/_libs/src/parser/tokenizer.c
static inline npy_int64
AK_UCS4_to_int64(Py_UCS4 *p_item, Py_UCS4 *end, int *error)
{
    char tsep = '\0'; // thousands seperator; if null processing is skipped
    npy_int64 int_min = NPY_MIN_INT64;
    npy_int64 int_max = NPY_MAX_INT64;
    int isneg = 0;
    npy_int64 number = 0;
    int d;

    Py_UCS4 *p = p_item;

    while (AK_is_space(*p)) {
        ++p;
        if (p >= end) {return number;}
    }
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        ++p;
    }
    if (p >= end) {return number;}

    // Check that there is a first digit.
    if (!AK_is_digit(*p)) {
        *error = ERROR_NO_DIGITS;
        return 0;
    }
    if (isneg) {
        // If number is greater than pre_min, at least one more digit can be processed without overflowing.
        int dig_pre_min = -(int_min % 10);
        npy_int64 pre_min = int_min / 10;
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                    continue;
                } else if (!AK_is_digit(d)) {
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
            while (AK_is_digit(d)) {
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
        npy_int64 pre_max = int_max / 10;
        int dig_pre_max = int_max % 10;
        d = *p;
        if (tsep != '\0') {
            while (1) {
                if (d == tsep) {
                    ++p;
                    if (p >= end) {return number;}
                    d = *p;
                    continue;
                } else if (!AK_is_digit(d)) {
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
            while (AK_is_digit(d)) {
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
    *error = 0;
    return number;
}

// Convert a Py_UCS4 array to an unsigned integer. Extended from pandas/_libs/src/parser/tokenizer.c
static inline npy_uint64
AK_UCS4_to_uint64(Py_UCS4 *p_item, Py_UCS4 *end, int *error) {

    char tsep = '\0'; // thousands seperator; if null processing is skipped

    npy_uint64 pre_max = NPY_MAX_UINT64 / 10;
    int dig_pre_max = NPY_MAX_UINT64 % 10;
    npy_uint64 number = 0;
    int d;

    Py_UCS4 *p = p_item;
    while (AK_is_space(*p)) {
        ++p;
        if (p >= end) {return number;}
    }
    if (*p == '-') {
        *error = ERROR_INVALID_CHARS;
        return 0;
    } else if (*p == '+') {
        p++;
        if (p >= end) {return number;}
    }

    // Check that there is a first digit.
    if (!AK_is_digit(*p)) {
        *error = ERROR_NO_DIGITS;
        return 0;
    }
    // If number is less than pre_max, at least one more digit can be processed without overflowing.
    d = *p;
    if (tsep != '\0') {
        while (1) {
            if (d == tsep) {
                ++p;
                if (p >= end) {return number;}
                d = *p;
                continue;
            } else if (!AK_is_digit(d)) {
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
        while (AK_is_digit(d)) {
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
    *error = 0;
    return number;
}



// Set the internal field to the current field.
static inline char*
AK_CPL_current_to_field(AK_CodePointLine* cpl)
{
    // NOTE: we assume this is only called after offset_max is complete
    if (!cpl->field) {
        cpl->field = (char*)PyMem_Malloc(sizeof(char) * (cpl->offset_max + 1));
    }
    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + cpl->offsets[cpl->index_current];
    char *t = cpl->field;

    while (p < end) {
        if (AK_is_space(*p)) {
            ++p;
            continue;
        }
        *t++ = (char)*p++;
    }
    *t = '\0';
    // AK_DEBUG_OBJ(PyUnicode_FromString(cpl->field));
    return cpl->field;
}

// This will take any case of "TRUE" as True, while marking everything else as False; this is the same approach taken with genfromtxt when the dtype is given as bool. This will not fail for invalid true or false strings.
// NP's Boolean conversion in genfromtxt: https://github.com/numpy/numpy/blob/0721406ede8b983b8689d8b70556499fc2aea28a/numpy/lib/_iotools.py#L386
static inline bool
AK_CPL_current_to_bool(AK_CodePointLine* cpl) {
    // must have at least 4 characters
    if (cpl->offsets[cpl->index_current] < 4) {
        return 0;
    }
    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + 4; // we must have at least 4 characters
    int i = 0;
    char c;

    while (AK_is_space(*p)) p++;

    for (;p < end; ++p) {
        c = *p;
        if (c == TRUE_LOWER[i] || c == TRUE_UPPER[i]) {
            ++i;
        }
        else {
            return false;
        }
    }
    return true; //matched all characters
}

// NOTE: using PyOS_strtol was an alternative, but needed to be passed a null-terminated char, which would require copying the data out of the CPL. This approach reads directly from the CPL without copying.
static inline npy_int64
AK_CPL_current_to_int64(AK_CodePointLine* cpl)
{
    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + cpl->offsets[cpl->index_current]; // size is either 4 or 5
    int error = 0;
    npy_int64 v = AK_UCS4_to_int64(p, end, &error);
    if (error > 0) {
        return 0;
    }
    return v;
}

static inline npy_uint64
AK_CPL_current_to_uint64(AK_CodePointLine* cpl)
{
    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + cpl->offsets[cpl->index_current]; // size is either 4 or 5
    int error = 0;
    npy_uint64 v = AK_UCS4_to_uint64(p, end, &error);
    if (error > 0) {
        return 0;
    }
    return v;
}

static inline npy_float64
AK_CPL_current_to_float64(AK_CodePointLine* cpl)
{
    return PyOS_string_to_double(AK_CPL_current_to_field(cpl), NULL, NULL);
}


//------------------------------------------------------------------------------
// CodePointLine: Exporters

static inline PyObject*
AK_CPL_ToArrayBoolean(AK_CodePointLine* cpl)
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
        if (AK_CPL_current_to_bool(cpl)) {
            array_buffer[i] = 1;
        }
        AK_CPL_CurrentAdvance(cpl);
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}


// Given a type of signed integer, return the corresponding array.
static inline PyObject*
AK_CPL_ToArrayFloat(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (!array) {
        return NULL;
    }

    if (dtype->elsize == 16) {
        # ifdef PyFloat128ArrType_Type
        npy_float128 *array_buffer = (npy_float128*)PyArray_DATA((PyArrayObject*)array);
        npy_float128 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            // NOTE: cannot cast to npy_float128 here
            *array_buffer++ = AK_CPL_current_to_float64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
        # endif
    }
    else if (dtype->elsize == 8) {
        npy_float64 *array_buffer = (npy_float64*)PyArray_DATA((PyArrayObject*)array);
        npy_float64 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_float64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_float32 *array_buffer = (npy_float32*)PyArray_DATA((PyArrayObject*)array);
        npy_float32 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_float32)AK_CPL_current_to_float64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_float16 *array_buffer = (npy_float16*)PyArray_DATA((PyArrayObject*)array);
        npy_float16 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_float16)AK_CPL_current_to_float64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "cannot create array from integer itemsize");
        Py_DECREF(array);
        return NULL;
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}



// Given a type of signed integer, return the corresponding array.
static inline PyObject*
AK_CPL_ToArrayInt(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (!array) {
        return NULL;
    }
    if (dtype->elsize == 8) {
        npy_int64 *array_buffer = (npy_int64*)PyArray_DATA((PyArrayObject*)array);
        npy_int64 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_int64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_int32 *array_buffer = (npy_int32*)PyArray_DATA((PyArrayObject*)array);
        npy_int32 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_int32)AK_CPL_current_to_int64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_int16 *array_buffer = (npy_int16*)PyArray_DATA((PyArrayObject*)array);
        npy_int16 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_int16)AK_CPL_current_to_int64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 1) {
        npy_int8 *array_buffer = (npy_int8*)PyArray_DATA((PyArrayObject*)array);
        npy_int8 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_int8)AK_CPL_current_to_int64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "cannot create array from integer itemsize");
        Py_DECREF(array);
        return NULL;
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}


// Given a type of signed integer, return the corresponding array.
static inline PyObject*
AK_CPL_ToArrayUInt(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (!array) {
        return NULL;
    }
    if (dtype->elsize == 8) {
        npy_uint64 *array_buffer = (npy_uint64*)PyArray_DATA((PyArrayObject*)array);
        npy_uint64 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_uint64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_uint32 *array_buffer = (npy_uint32*)PyArray_DATA((PyArrayObject*)array);
        npy_uint32 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint32)AK_CPL_current_to_uint64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_uint16 *array_buffer = (npy_uint16*)PyArray_DATA((PyArrayObject*)array);
        npy_uint16 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint16)AK_CPL_current_to_uint64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 1) {
        npy_uint8 *array_buffer = (npy_uint8*)PyArray_DATA((PyArrayObject*)array);
        npy_uint8 *end = array_buffer + count;
        AK_CPL_CurrentReset(cpl);
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint8)AK_CPL_current_to_uint64(cpl);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "cannot create array from unsigned integer itemsize");
        Py_DECREF(array);
        return NULL;
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}


static inline PyObject*
AK_CPL_ToArrayUnicode(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    Py_ssize_t field_points;
    bool capped_points;

    if (dtype->elsize == 0) {
        field_points = cpl->offset_max;
        dtype->elsize = field_points * sizeof(Py_UCS4);
        capped_points = false;
    }
    else {
        // assume that elsize is already given in units of 4
        assert(dtype->elsize % sizeof(Py_UCS4) == 0);
        field_points = dtype->elsize / sizeof(Py_UCS4);
        capped_points = true;
    }

    // assuming this is contiguous
    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (!array) {
        return NULL;
    }

    Py_UCS4 *array_buffer = (Py_UCS4*)PyArray_DATA((PyArrayObject*)array);
    Py_UCS4 *end = array_buffer + count * field_points;

    AK_CPL_CurrentReset(cpl);

    if (capped_points) {
        // NOTE: is it worth branching for this special case?
        Py_ssize_t copy_bytes;
        while (array_buffer < end) {
            if (cpl->offsets[cpl->index_current] >= field_points) {
                copy_bytes = field_points * sizeof(Py_UCS4);
            } else {
                copy_bytes = cpl->offsets[cpl->index_current] * sizeof(Py_UCS4);
            }
            memcpy(array_buffer,
                    cpl->pos_current,
                    copy_bytes);
            array_buffer += field_points;
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else { // faster we always know the offset will fit
        while (array_buffer < end) {
            memcpy(array_buffer,
                    cpl->pos_current,
                    cpl->offsets[cpl->index_current] * sizeof(Py_UCS4));
            array_buffer += field_points;
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

static inline PyObject*
AK_CPL_ToArrayBytes(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    Py_ssize_t field_points;
    bool capped_points;

    if (dtype->elsize == 0) {
        field_points = cpl->offset_max;
        dtype->elsize = field_points;
        capped_points = false;
    }
    else {
        field_points = dtype->elsize;
        capped_points = true;
    }

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (!array) {
        return NULL;
    }

    char *array_buffer = (char*)PyArray_DATA((PyArrayObject*)array);
    char *end = array_buffer + count * field_points;
    Py_ssize_t copy_points;

    AK_CPL_CurrentReset(cpl);

    while (array_buffer < end) {
        if (!capped_points || cpl->offsets[cpl->index_current] < field_points) {
            // if not capped, or capped and offset is less than field points, use offset
            copy_points = cpl->offsets[cpl->index_current];
        }
        else {
            // if capped and offset is greater than feild points, use field points
            copy_points = field_points;
        }

        Py_UCS4 *p = cpl->pos_current;
        Py_UCS4 *p_end = p + copy_points;
        char *field_end = array_buffer + field_points;

        while (p < p_end) {
            *array_buffer++ = (char)*p++; // truncate
        }
        array_buffer = field_end;
        AK_CPL_CurrentAdvance(cpl);
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

static inline PyObject*
AK_CPL_ToArrayDatetime(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    PyArray_Descr *dtype_temp = PyArray_DescrFromType(NPY_STRING);
    if (!dtype_temp) {
        return NULL;
    }

    PyArray_Descr *dtype_pre = PyArray_DescrNew(dtype_temp);
    if (!dtype_pre) {
        return NULL;
    }
    Py_DECREF(dtype_temp);

    PyObject* array_temp = AK_CPL_ToArrayBytes(cpl, dtype_pre); // will steal dtype_pre
    PyObject *array = PyArray_CastToType((PyArrayObject*)array_temp, dtype, 0);
    Py_DECREF(array_temp);

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

static inline PyObject*
AK_CPL_ToArrayComplex(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    PyArray_Descr *dtype_temp = PyArray_DescrFromType(NPY_STRING);
    if (!dtype_temp) {
        return NULL;
    }

    PyArray_Descr *dtype_pre = PyArray_DescrNew(dtype_temp);
    if (!dtype_pre) {
        return NULL;
    }
    Py_DECREF(dtype_temp);

    PyObject* array_temp = AK_CPL_ToArrayBytes(cpl, dtype_pre); // will steal dtype_pre
    PyObject *array = PyArray_CastToType((PyArrayObject*)array_temp, dtype, 0);
    Py_DECREF(array_temp);

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}


// Generic handler for converting a CPL to an array. The dtype given here must already be a fresh instance as it might be mutated. Might return NULL if array creation fails; an exception should be set.
static inline PyObject*
AK_CPL_ToArray(AK_CodePointLine* cpl, PyArray_Descr* dtype) {
    if (!dtype) {
        // get from CPL
        if (cpl->type_parser) {
            // will return a fresh instance
            // AK_DEBUG("CPL calling AK_TPS_to_dtype");
            dtype = AK_TPS_to_dtype(cpl->type_parser->parsed_line);
        }
        else {
            AK_NOT_IMPLEMENTED("no handling for x dtype yet");
        }
    }

    if (PyDataType_ISBOOL(dtype)) {
        return AK_CPL_ToArrayBoolean(cpl);
    }
    else if (PyDataType_ISSTRING(dtype) && dtype->kind == 'U') {
        return AK_CPL_ToArrayUnicode(cpl, dtype);
    }
    else if (PyDataType_ISSTRING(dtype) && dtype->kind == 'S') {
        return AK_CPL_ToArrayBytes(cpl, dtype);
    }
    else if (PyDataType_ISUNSIGNED(dtype)) { // must come before integer check
        return AK_CPL_ToArrayUInt(cpl, dtype);
    }
    else if (PyDataType_ISINTEGER(dtype)) {
        return AK_CPL_ToArrayInt(cpl, dtype);
    }
    else if (PyDataType_ISFLOAT(dtype)) {
        return AK_CPL_ToArrayFloat(cpl, dtype);
    }
    else if (PyDataType_ISDATETIME(dtype)) {
        return AK_CPL_ToArrayDatetime(cpl, dtype);
    }
    else if (PyDataType_ISCOMPLEX(dtype)) {
        return AK_CPL_ToArrayComplex(cpl, dtype);
    }
    else {
        AK_NOT_IMPLEMENTED("no handling for other dtypes yet");
    }
}


// Return a contiguous string of data stored in the buffer, without delimiters. Returns a new reference.
static inline PyObject*
AK_CPL_ToUnicode(AK_CodePointLine* cpl)
{
    return PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND,
            cpl->buffer,
            cpl->buffer_count);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// CodePointGrid Type, New, Destrctor

typedef struct AK_CodePointGrid {
    Py_ssize_t lines_count; // accumulated number of lines
    Py_ssize_t lines_capacity; // max number of lines
    AK_CodePointLine **lines; // array of pointers
    PyObject *dtypes; // Sequence of bools
} AK_CodePointGrid;

AK_CodePointGrid*
AK_CPG_New(PyObject *dtypes)
{
    AK_CodePointGrid *cpg = (AK_CodePointGrid*)PyMem_Malloc(sizeof(AK_CodePointGrid));
    cpg->lines_count = 0;
    cpg->lines_capacity = 100;
    cpg->lines = (AK_CodePointLine**)PyMem_Malloc(
            sizeof(AK_CodePointLine*) * cpg->lines_capacity);
    // NOTE: initialize lines to NULL?
    cpg->dtypes = dtypes;
    return cpg;
}

void
AK_CPG_Free(AK_CodePointGrid* cpg)
{
    for (int i=0; i < cpg->lines_count; ++i) {
        AK_CPL_Free(cpg->lines[i]);
    }
    PyMem_Free(cpg->lines);
    Py_XDECREF(cpg->dtypes);
    PyMem_Free(cpg);
}
//------------------------------------------------------------------------------
// CodePointGrid: Mutation

static inline int
AK_CPG_resize(AK_CodePointGrid* cpg, Py_ssize_t line)
{
    if (line >= cpg->lines_capacity) {
        assert(line == cpg->lines_capacity);

        cpg->lines_capacity *= 2;
        // NOTE: as we sure we are only copying pointers?
        cpg->lines = PyMem_Realloc(cpg->lines,
                sizeof(AK_CodePointLine*) * cpg->lines_capacity);
        // TODO: handle error, initialize lines to NULL
    }
    // for now we assume sequential growth, so should only check if equal
    if (line >= cpg->lines_count) {
        assert(line == cpg->lines_count);
        // determine if we need to parse types
        bool type_parse = false;
        if (!cpg->dtypes) {
            type_parse = true;
        }
        else {
            PyObject* dtype_specifier = PyList_GetItem(cpg->dtypes, line);
            if (!dtype_specifier || dtype_specifier == Py_None) {
                type_parse = true;
            }
        }
        // Always initialize a CPL in the new position
        cpg->lines[line] = AK_CPL_New(type_parse);
        ++cpg->lines_count;
    }

    return 1;
}

static inline int
AK_CPG_AppendObjectAtLine(
        AK_CodePointGrid* cpg,
        Py_ssize_t line,
        PyObject* element)
{
    AK_CPG_resize(cpg, line);
    // handle failure
    AK_CPL_AppendObject(cpg->lines[line], element);
    // handle failure
    return 1;
}

static inline int
AK_CPG_AppendPointAtLine(
        AK_CodePointGrid* cpg,
        Py_ssize_t line,
        Py_ssize_t field_len,
        Py_UCS4 p
        )
{
    AK_CPG_resize(cpg, line);
    // handle failure
    AK_CPL_AppendPoint(cpg->lines[line], p, field_len);
    // handle failure
    return 1;
}

static inline int
AK_CPG_AppendOffsetAtLine(
        AK_CodePointGrid* cpg,
        Py_ssize_t line,
        Py_ssize_t offset)
{
    AK_CPG_resize(cpg, line);
    // handle failure
    AK_CPL_AppendOffset(cpg->lines[line], offset);
    // handle failure
    return 1;
}


//------------------------------------------------------------------------------
// CodePointGrid: Constructors

// Given an iterable, load a CPG. If axis is 0, interpret the first level of as the primary level (rows become columns); if axis is 1, align values by position per row (rows are partitioned into columns).
// AK_CodePointGrid* AK_CPG_FromIterable(
//         PyObject* iterable,
//         int axis)
// {
//     AK_CodePointGrid* cpg = AK_CPG_New();
//     // expect an iterable of iterables
//     PyObject *outer_iter = PyObject_GetIter(iterable);
//     // TODO: handle error
//     PyObject *outer;
//     PyObject *inner_iter;
//     PyObject *inner;

//     int inner_count;
//     int outer_count = 0;

//     int *count_src = axis == 0 ? &outer_count : &inner_count;

//     while ((outer = PyIter_Next(outer_iter))) {
//         inner_iter = PyObject_GetIter(outer);
//         // TODO: handle error
//         inner_count = 0;

//         while ((inner = PyIter_Next(inner_iter))) {
//             AK_CPG_AppendObjectAtLine(cpg, *count_src, inner);
//             // TODO: handle error
//             ++inner_count;
//             Py_DECREF(inner);
//         }
//         ++outer_count;
//         Py_DECREF(outer);
//         Py_DECREF(inner_iter);
//     }
//     Py_DECREF(outer_iter);
//     return cpg;
// }

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

PyObject* AK_CPG_ToArrayList(AK_CodePointGrid* cpg)
{
    PyObject* list = PyList_New(cpg->lines_count);
    // handle error

    PyObject* dtypes = cpg->dtypes;

    for (int i = 0; i < cpg->lines_count; ++i) {

        // If dtypes is not NULL, fetch the dtype_specifier and use it to set dtype; else, pass the dtype as NULL to CPL.
        PyArray_Descr* dtype = NULL;
        if (dtypes) {
            PyObject* dtype_specifier = PyList_GetItem(dtypes, i);
            // Set dtype; this value can be NULL or a dtype (never Py_None)
            AK_DTypeFromSpecifier(dtype_specifier, &dtype);
            // TODO: handle error
        }

        // This function will observe if dtype is NULL and read dtype from the CPL's type_parser if necessary
        PyObject* array = AK_CPL_ToArray(cpg->lines[i], dtype);

        if (PyList_SetItem(list, i, array)) { // steals reference
           // handle error
        }
    }
    return list;
}


//------------------------------------------------------------------------------
// AK_Dialect, based on _csv.c from CPython

typedef enum {
    QUOTE_MINIMAL,
    QUOTE_ALL,
    QUOTE_NONNUMERIC,
    QUOTE_NONE
} AK_DialectQuoteStyle;

typedef struct AK_DialectStyleDesc{
    AK_DialectQuoteStyle style;
    const char *name;
} AK_DialectStyleDesc;

static const AK_DialectStyleDesc quote_styles[] = {
    { QUOTE_MINIMAL,    "QUOTE_MINIMAL" },
    { QUOTE_ALL,        "QUOTE_ALL" },
    { QUOTE_NONNUMERIC, "QUOTE_NONNUMERIC" },
    { QUOTE_NONE,       "QUOTE_NONE" },
    { 0 }
};

static int
AK_Dialect_set_bool(const char *name, char *target, PyObject *src, bool dflt)
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
AK_Dialect_set_int(const char *name, int *target, PyObject *src, int dflt)
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
AK_Dialect_set_char(const char *name, Py_UCS4 *target, PyObject *src, Py_UCS4 dflt)
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
AK_Dialect_set_str(const char *name, PyObject **target, PyObject *src, const char *dflt)
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
AK_Dialect_check_quoting(int quoting)
{
    const AK_DialectStyleDesc *qs;
    for (qs = quote_styles; qs->name; qs++) {
        if ((int)qs->style == quoting)
            return 0;
    }
    PyErr_Format(PyExc_TypeError, "bad \"quoting\" value");
    return -1;
}


typedef struct AK_Dialect{
    char doublequote;           /* is " represented by ""? */
    char skipinitialspace;      /* ignore spaces following delimiter? */
    char strict;                /* raise exception on bad CSV */
    int quoting;                /* style of quoting to write */
    Py_UCS4 delimiter;          /* field separator */
    Py_UCS4 quotechar;          /* quote character */
    Py_UCS4 escapechar;         /* escape character */
    PyObject *lineterminator;   /* string to write between records */
} AK_Dialect;


// check types and convert to C values
#define AK_CALL_WITH_GOTO(meth, name, target, src, default) \
    if (meth(name, target, src, default)) \
        goto err

static AK_Dialect*
AK_DialectNew(PyObject *delimiter,
        PyObject *doublequote,
        PyObject *escapechar,
        PyObject *lineterminator,
        PyObject *quotechar,
        PyObject *quoting,
        PyObject *skipinitialspace,
        PyObject *strict
        )
{
    AK_Dialect *dialect = (AK_Dialect *) PyMem_Malloc(sizeof(AK_Dialect));
    if (dialect == NULL) {
        return NULL;
    }
    // TODO: are these increfs necessary?
    Py_XINCREF(delimiter);
    Py_XINCREF(doublequote);
    Py_XINCREF(escapechar);
    Py_XINCREF(lineterminator);
    Py_XINCREF(quotechar);
    Py_XINCREF(quoting);
    Py_XINCREF(skipinitialspace);
    Py_XINCREF(strict);

    AK_CALL_WITH_GOTO(AK_Dialect_set_char, "delimiter", &dialect->delimiter, delimiter, ',');
    AK_CALL_WITH_GOTO(AK_Dialect_set_bool, "doublequote", &dialect->doublequote, doublequote, true);
    AK_CALL_WITH_GOTO(AK_Dialect_set_char, "escapechar", &dialect->escapechar, escapechar, 0);
    AK_CALL_WITH_GOTO(AK_Dialect_set_str,
            "lineterminator",
            &dialect->lineterminator,
            lineterminator,
            "\r\n");
    AK_CALL_WITH_GOTO(AK_Dialect_set_char, "quotechar", &dialect->quotechar, quotechar, '"');
    AK_CALL_WITH_GOTO(AK_Dialect_set_int, "quoting", &dialect->quoting, quoting, QUOTE_MINIMAL);
    AK_CALL_WITH_GOTO(AK_Dialect_set_bool,
            "skipinitialspace",
            &dialect->skipinitialspace,
            skipinitialspace,
            false);
    AK_CALL_WITH_GOTO(AK_Dialect_set_bool, "strict", &dialect->strict, strict, false);

    /* validate options */
    if (AK_Dialect_check_quoting(dialect->quoting))
        goto err;
    if (dialect->delimiter == 0) {
        PyErr_SetString(PyExc_TypeError,
                "\"delimiter\" must be a 1-character string");
        goto err;
    }
    if (quotechar == Py_None && quoting == NULL)
        dialect->quoting = QUOTE_NONE;
    if (dialect->quoting != QUOTE_NONE && dialect->quotechar == 0) {
        PyErr_SetString(PyExc_TypeError,
               "quotechar must be set if quoting enabled");
        goto err;
    }
    if (dialect->lineterminator == 0) {
        PyErr_SetString(PyExc_TypeError, "lineterminator must be set");
        goto err;
    }
    return dialect;
err:
    Py_CLEAR(dialect);

    Py_CLEAR(delimiter);
    Py_CLEAR(doublequote);
    Py_CLEAR(escapechar);
    Py_CLEAR(lineterminator);
    Py_CLEAR(quotechar);
    Py_CLEAR(quoting);
    Py_CLEAR(skipinitialspace);
    Py_CLEAR(strict);
    return NULL;
}

void
AK_Dialect_Free(AK_Dialect* dialect)
{
    PyMem_Free(dialect);
}

//------------------------------------------------------------------------------
// AK_DelimitedReader, based on _csv.c from CPython

typedef enum {
    START_RECORD,
    START_FIELD,
    ESCAPED_CHAR,
    IN_FIELD,
    IN_QUOTED_FIELD,
    ESCAPE_IN_QUOTED_FIELD,
    QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL,
    AFTER_ESCAPED_CRNL
} AK_DR_ParserState;

typedef struct AK_DelimitedReader{
    PyObject *input_iter;   // iterate over this for input lines
    AK_Dialect *dialect;
    AK_DR_ParserState state;          // current CSV parse state
    Py_ssize_t field_len;
    Py_ssize_t line_number;
    Py_ssize_t field_number;
    int axis;
    // NOTE: this may need to go, or could be used as an indicator in type evaluation
    int numeric_field; // treat field as numeric

} AK_DelimitedReader;

static inline int
AK_DR_close_field(AK_DelimitedReader *dr, AK_CodePointGrid *cpg)
{
    AK_CPG_AppendOffsetAtLine(cpg,
            dr->axis == 0 ? dr->line_number : dr->field_number,
            dr->field_len);
    dr->field_len = 0; // clear to close
    ++dr->field_number; // increment after adding each offset, reset in AK_DR_line_reset
    return 0;
}

static inline int
AK_DR_add_char(AK_DelimitedReader *dr, AK_CodePointGrid *cpg, Py_UCS4 c)
{
    AK_CPG_AppendPointAtLine(cpg,
            dr->axis == 0 ? dr->line_number : dr->field_number,
            dr->field_len,
            c);
    ++dr->field_len; // reset in AK_DR_close_field
    return 0;
}

// Process each char and update AK_DelimitedReader state. When appropriate, call AK_DR_add_char to accumulate field characters, AK_DR_close_field to end a field
static int
AK_DR_process_char(AK_DelimitedReader *dr, AK_CodePointGrid *cpg, Py_UCS4 c)
{
    AK_Dialect *dialect = dr->dialect;

    switch (dr->state) {
    case START_RECORD: /* start of record */
        if (c == '\0') /* empty line - return [] */
            break;
        else if (c == '\n' || c == '\r') {
            dr->state = EAT_CRNL;
            break;
        }
        /* normal character - handle as START_FIELD */
        dr->state = START_FIELD;
        /* fallthru */
    case START_FIELD: /* expecting field */
        if (c == '\n' || c == '\r' || c == '\0') {
            /* save empty field - return [fields] */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->quotechar &&
                 dialect->quoting != QUOTE_NONE) { /* start quoted field */
            dr->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->escapechar) { /* possible escaped character */
            dr->state = ESCAPED_CHAR;
        }
        else if (c == ' ' && dialect->skipinitialspace)
            /* ignore space at start of field */
            ;
        else if (c == dialect->delimiter) { /* save empty field */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
        }
        else { /* begin new unquoted field */
            if (dialect->quoting == QUOTE_NONNUMERIC)
                dr->numeric_field = 1;
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
            dr->state = IN_FIELD;
        }
        break;

    case ESCAPED_CHAR:
        if (c == '\n' || c=='\r') {
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
            dr->state = AFTER_ESCAPED_CRNL;
            break;
        }
        if (c == '\0')
            c = '\n';
        if (AK_DR_add_char(dr, cpg, c) < 0)
            return -1;
        dr->state = IN_FIELD;
        break;

    case AFTER_ESCAPED_CRNL:
        if (c == '\0')
            break;
        /*fallthru*/

    case IN_FIELD: /* in unquoted field */
        if (c == '\n' || c == '\r' || c == '\0') {
            /* end of line - return [fields] */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->escapechar) { /* possible escaped character */
            dr->state = ESCAPED_CHAR;
        }
        else if (c == dialect->delimiter) { /* save field - wait for new field */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
            dr->state = START_FIELD;
        }
        else { /* normal character - save in field */
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
        }
        break;

    case IN_QUOTED_FIELD: /* in quoted field */
        if (c == '\0')
            ;
        else if (c == dialect->escapechar) { /* Possible escape character */
            dr->state = ESCAPE_IN_QUOTED_FIELD;
        }
        else if (c == dialect->quotechar &&
                 dialect->quoting != QUOTE_NONE) {
            if (dialect->doublequote) { /* doublequote; " represented by "" */
                dr->state = QUOTE_IN_QUOTED_FIELD;
            }
            else { /* end of quote part of field */
                dr->state = IN_FIELD;
            }
        }
        else { /* normal character - save in field */
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
        }
        break;

    case ESCAPE_IN_QUOTED_FIELD:
        if (c == '\0')
            c = '\n';
        if (AK_DR_add_char(dr, cpg, c) < 0)
            return -1;
        dr->state = IN_QUOTED_FIELD;
        break;

    case QUOTE_IN_QUOTED_FIELD:
        /* doublequote - seen a quote in a quoted field */
        if (dialect->quoting != QUOTE_NONE && c == dialect->quotechar) {
            /* save "" as " */
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
            dr->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->delimiter) { /* save field - wait for new field */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
            dr->state = START_FIELD;
        }
        else if (c == '\n' || c == '\r' || c == '\0') {
            /* end of line - return [fields] */
            if (AK_DR_close_field(dr, cpg) < 0)
                return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (!dialect->strict) {
            if (AK_DR_add_char(dr, cpg, c) < 0)
                return -1;
            dr->state = IN_FIELD;
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
            dr->state = START_RECORD;
        else {
            PyErr_Format(PyExc_RuntimeError,
                    "new-line character seen in unquoted field - do you need to open the file in universal-newline mode?");
            return -1;
        }
        break;

    }
    return 0;
}

// Called once at the start of processing each line in AK_DR_ProcessLine.
static int
AK_DR_line_reset(AK_DelimitedReader *dr)
{
    dr->field_len = 0;
    dr->state = START_RECORD;
    dr->numeric_field = 0;
    dr->field_number = 0;
    return 0;
}

// Using AK_DelimitedReader's state, process one line (via next(input_iter)); call AK_DR_process_char on each char in that line, loading individual fields into AK_CodePointGrid
static int
AK_DR_ProcessLine(AK_DelimitedReader *dr, AK_CodePointGrid *cpg)
{
    Py_UCS4 c;
    Py_ssize_t pos, linelen;
    unsigned int kind;
    const void *data;
    PyObject *lineobj;

    if (AK_DR_line_reset(dr) < 0)
        return 0;
    do {
        // get one line to parse
        lineobj = PyIter_Next(dr->input_iter);
        if (lineobj == NULL) {
            // End of input OR exception
            if (!PyErr_Occurred() && (dr->field_len != 0 ||
                    dr->state == IN_QUOTED_FIELD)) {
                if (dr->dialect->strict)
                    PyErr_SetString(PyExc_RuntimeError,
                            "unexpected end of data");
                else if (AK_DR_close_field(dr, cpg) >= 0)
                    break;
            }
            return 0;
        }
        if (!PyUnicode_Check(lineobj)) {
            PyErr_Format(PyExc_RuntimeError,
                    "iterator should return strings, "
                    "not %.200s "
                    "(the file should be opened in text mode)",
                    Py_TYPE(lineobj)->tp_name
                    );
            Py_DECREF(lineobj);
            return -1; // check that client takes only > 0
        }
        if (PyUnicode_READY(lineobj) == -1) {
            Py_DECREF(lineobj);
            return -1;
        }

        ++dr->line_number; // initialized to -1
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
                goto exit;
            }
            if (AK_DR_process_char(dr, cpg, c) < 0) {
                Py_DECREF(lineobj);
                goto exit;
            }
            pos++;
        }
        Py_DECREF(lineobj);
        if (AK_DR_process_char(dr, cpg, 0) < 0)
            goto exit;
    } while (dr->state != START_RECORD);

exit:
    return 1;
}

static AK_DelimitedReader*
AK_DR_New(PyObject *iterable,
        int axis,
        PyObject *delimiter,
        PyObject *doublequote,
        PyObject *escapechar,
        PyObject *lineterminator,
        PyObject *quotechar,
        PyObject *quoting,
        PyObject *skipinitialspace,
        PyObject *strict
        )
{
    AK_DelimitedReader *dr = (AK_DelimitedReader*)PyMem_Malloc(sizeof(AK_DelimitedReader));
    if (!dr)
        return NULL;

    dr->input_iter = NULL;
    dr->axis = axis;
    dr->line_number = -1;

    if (AK_DR_line_reset(dr) < 0) {
        Py_DECREF(dr);
        return NULL;
    }

    dr->input_iter = PyObject_GetIter(iterable);
    if (dr->input_iter == NULL) {
        Py_DECREF(dr);
        return NULL;
    }

    dr->dialect = AK_DialectNew(
            delimiter,
            doublequote,
            escapechar,
            lineterminator,
            quotechar,
            quoting,
            skipinitialspace,
            strict);

    if (dr->dialect == NULL) {
        Py_DECREF(dr);
        return NULL;
    }
    return dr;
}

static void
AK_DR_Free(AK_DelimitedReader *dr)
{
    AK_Dialect_Free(dr->dialect);
    dr->dialect = NULL;
    Py_CLEAR(dr->input_iter);
    PyMem_Free(dr);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Convert an sequence of strings to a 1D array.
static inline PyObject*
AK_IterableStrToArray1D(
        PyObject *sequence,
        PyObject *dtype_specifier)
{

        PyArray_Descr* dtype = NULL;
        // will set NULL for None, and propagate NULLs
        AK_DTypeFromSpecifier(dtype_specifier, &dtype);

        // dtype only NULL from here
        bool type_parse = dtype == NULL;

        AK_CodePointLine* cpl = AK_CPL_FromIterable(sequence, type_parse);
        PyObject* array = AK_CPL_ToArray(cpl, dtype);
        AK_CPL_Free(cpl);

        return array;
}


// Numpy implementation: https://github.com/numpy/numpy/blob/a14c41264855e44ebd6187d7541b5b8d59bb32cb/numpy/core/src/multiarray/methods.c#L1557
PyObject*
AK_ArrayDeepCopy(PyArrayObject *array, PyObject *memo)
{
    PyObject *id = PyLong_FromVoidPtr((PyObject*)array);
    if (!id) {
        return NULL;
    }
    PyObject *found = PyDict_GetItemWithError(memo, id);
    if (found) { // found will be NULL if not in dict
        Py_INCREF(found); // got a borrowed ref, increment first
        Py_DECREF(id);
        return found;
    }
    else if (PyErr_Occurred()) {
        goto error;
    }

    // if dtype is object, call deepcopy with memo
    PyObject *array_new;
    PyArray_Descr *dtype = PyArray_DESCR(array); // borrowed ref

    if (PyDataType_ISOBJECT(dtype)) {
        PyObject *copy = PyImport_ImportModule("copy");
        if (!copy) {
            goto error;
        }
        PyObject *deepcopy = PyObject_GetAttrString(copy, "deepcopy");
        Py_DECREF(copy);
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
        Py_INCREF(dtype); // PyArray_FromArray steals a reference
        array_new = PyArray_FromArray(
                array,
                dtype,
                NPY_ARRAY_ENSURECOPY);
        if (!array_new || PyDict_SetItem(memo, id, array_new)) {
            Py_XDECREF(array_new);
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


//------------------------------------------------------------------------------
// AK module public methods
//------------------------------------------------------------------------------

static char *dtoa_kwarg_names[] = {
    "file_like",
    "dtypes",
    "axis",
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

static PyObject*
delimited_to_arrays(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyObject *file_like;
    int axis = 0;

    PyObject *dtypes = NULL;

    PyObject *delimiter = NULL;
    PyObject *doublequote = NULL;
    PyObject *escapechar = NULL;
    PyObject *lineterminator = NULL;
    PyObject *quotechar = NULL;
    PyObject *quoting = NULL;
    PyObject *skipinitialspace = NULL;
    PyObject *strict = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|$OiOOOOOOOO:delimited_to_array", dtoa_kwarg_names,
            &file_like,
            // kwarg only
            &dtypes,
            &axis,
            &delimiter,
            &doublequote,
            &escapechar,
            &lineterminator,
            &quotechar,
            &quoting,
            &skipinitialspace,
            &strict))
        return NULL;

    AK_DelimitedReader *dr = AK_DR_New(file_like,
            axis,
            delimiter,
            doublequote,
            escapechar,
            lineterminator,
            quotechar,
            quoting,
            skipinitialspace,
            strict);

    if (dtypes == Py_None) {
        // Py_DECREF(dtypes); not sure if this is needed
        dtypes = NULL;
    }
    Py_XINCREF(dtypes);
    AK_CodePointGrid* cpg = AK_CPG_New(dtypes);

    // Consume all lines from dr and load into cpg
    while (AK_DR_ProcessLine(dr, cpg)); // check for -1
    AK_DR_Free(dr);

    PyObject* arrays = AK_CPG_ToArrayList(cpg);
    AK_CPG_Free(cpg); // will free reference to dtypes

    return arrays; // could be NULL
}


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
    return AK_IterableStrToArray1D(iterable, dtype_specifier);
}

// static PyObject *
// _test(PyObject *Py_UNUSED(m), PyObject *value)
// {
//     AK_CodePointGrid* cpg = AK_CPG_FromIterable(value, 1);
//     PyObject* post = AK_CPG_ToUnicodeList(cpg);
//     AK_CPG_Free(cpg);
//     return post;
// }


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
// array utility

// Specialized array deepcopy that stores immutable arrays in memo dict.
static PyObject *
array_deepcopy(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *array, *memo;
    if (!PyArg_UnpackTuple(args, "array_deepcopy", 2, 2, &array, &memo)) {
        return NULL;
    }
    AK_CHECK_NUMPY_ARRAY(array);
    if (!PyDict_CheckExact(memo)) {
        PyErr_Format(PyExc_TypeError, "expected a dict (got %s)",
                Py_TYPE(memo)->tp_name);
        return NULL;
    }
    return AK_ArrayDeepCopy((PyArrayObject*)array, memo);
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
    {"array_deepcopy", array_deepcopy, METH_VARARGS, NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"delimited_to_arrays",
            (PyCFunction)delimited_to_arrays,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"iterable_str_to_array_1d", iterable_str_to_array_1d, METH_VARARGS, NULL},
    // {"_test", _test, METH_O, NULL},
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
