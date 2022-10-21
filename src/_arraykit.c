# include "Python.h"
# include "structmember.h"
# include "stdbool.h"
# include "limits.h"
// # include "stdlib.h"

# define PY_ARRAY_UNIQUE_SYMBOL AK_ARRAY_API
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

# include "numpy/arrayobject.h"
# include "numpy/arrayscalars.h"
# include "numpy/halffloat.h"

//------------------------------------------------------------------------------
// Macros

//------------------------------------------------------------------------------
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
        fprintf(stderr, "--- %s: %i: %s: ", __FILE__, __LINE__, __FUNCTION__);

# define _AK_DEBUG_END()       \
        fprintf(stderr, "\n"); \
        fflush(stderr);        \
    } while (0)

# define AK_DEBUG_MSG_OBJ(msg, obj)     \
    _AK_DEBUG_BEGIN();                  \
        fprintf(stderr, #msg " ");      \
        PyObject_Print(obj, stderr, 0); \
    _AK_DEBUG_END()

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
        Py_SETREF(resolved, AK_ResolveDTypes(resolved, dtype));
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
    return resolved;
}

// Perform a deepcopy on an array, using an optional memo dictionary, and specialized to depend on immutable arrays. This depends on the module object to get the deepcopy method.
PyObject*
AK_ArrayDeepCopy(PyObject* m, PyArrayObject *array, PyObject *memo)
{
    PyObject *id = PyLong_FromVoidPtr((PyObject*)array);
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
    PyArray_Descr *dtype = PyArray_DESCR(array); // borrowed ref

    if (PyDataType_ISOBJECT(dtype)) {
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
                array,
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

// Given a dtype_specifier, which might be a dtype, NULL, or None, assign a fresh dtype object (or NULL) to dtype_returned. Returns 0 on success, -1 on failure. This will not interpret a None dtype_specified as a float dtype. This will never set dtype_returned to None (only NULL). Returns a new reference.
static inline int
AK_DTypeFromSpecifier(PyObject *dtype_specifier, PyArray_Descr **dtype_returned)
{
    PyArray_Descr* dtype;
    if (dtype_specifier == NULL) {
        dtype = NULL; // propagate, cannot call into oncverter
    }
    else if (PyObject_TypeCheck(dtype_specifier, &PyArrayDescr_Type)) {
        dtype = (PyArray_Descr* )dtype_specifier;
    }
    else { // converter2 sets NULL for None
        PyArray_DescrConverter2(dtype_specifier, &dtype);
    }
    // if not NULL, make a copy as we will give ownership to array and might mutate
    if (dtype) {
        dtype = PyArray_DescrNew(dtype);
        if (dtype == NULL) return -1;
    }
    *dtype_returned = dtype;
    return 0;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

#define AK_is_digit(c) (((unsigned)(c) - '0') < 10u)
#define AK_is_space(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
#define AK_is_quote(c) (((c) == '"') || ((c) == '\''))
#define AK_is_sign(c) (((c) == '+') || ((c) == '-'))
#define AK_is_paren_open(c) ((c) == '(')
#define AK_is_paren_close(c) ((c) == ')')

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
// Utility setters of C types from possibly NULL PyObject*; all return -1 on error.

static int
AK_set_bool(const char *name,
        bool *target,
        PyObject *src,
        bool dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        int b = PyObject_IsTrue(src);
        if (b < 0) return -1;
        *target = (char)b;
    }
    return 0;
}

static int
AK_set_int(const char *name,
        int *target,
        PyObject *src,
        int dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        if (!PyLong_CheckExact(src)) {
            PyErr_Format(PyExc_TypeError, "\"%s\" must be an integer", name);
            return -1;
        }
        long value = PyLong_AsLong(src);
        if (value == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (value < INT_MIN || value > INT_MAX) {
            PyErr_Format(PyExc_TypeError, "\"%s\" overflowed integer", name);
            return -1;
        }
        *target = (int)value;
    }
    return 0;
}

// Set a character from `src` on `target`; if src is NULL use default. Returns -1 on error, else 0.
static int
AK_set_char(const char *name,
        Py_UCS4 *target,
        PyObject *src,
        Py_UCS4 dflt)
{
    if (src == NULL)
        *target = dflt;
    else {
        *target = '\0';
        if (src != Py_None) {
            Py_ssize_t len;
            if (!PyUnicode_Check(src)) {
                PyErr_Format(PyExc_TypeError,
                        "\"%s\" must be string, not %.200s",
                        name,
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
            if (len > 0)
                *target = PyUnicode_READ_CHAR(src, 0);
        }
    }
    return 0;
}

//------------------------------------------------------------------------------
// TypeParser: Type, New, Destructor

// This defines the observed type of each field, or an entire line of fields. Note that TPS_STRING serves as the last resort, the type that needs no conversion (or cannot be converted).
typedef enum AK_TypeParserState {
    TPS_UNKNOWN, // for initial state
    TPS_BOOL,
    TPS_INT,
    TPS_FLOAT,
    TPS_COMPLEX, // 4
    TPS_STRING,
    TPS_EMPTY // empty fields
} AK_TypeParserState;

// Given previous and new parser states, return a next parser state. Does not error.
AK_TypeParserState
AK_TPS_Resolve(AK_TypeParserState previous, AK_TypeParserState new) {
    // unlikely case
    if (new == TPS_UNKNOWN) return TPS_STRING;

    // propagate new if previous is unknown or empty
    if ((previous == TPS_UNKNOWN) || (previous == TPS_EMPTY)) return new;

    // if either are string, go to string
    if (previous == TPS_STRING || new == TPS_STRING) return TPS_STRING;

    // handle both new, previous bool directly
    if (previous == TPS_BOOL) {
        if (new == TPS_EMPTY || new == TPS_BOOL) return TPS_BOOL;
        else {return TPS_STRING;} // bool found with anything except empty is string
    }
    if (new == TPS_BOOL) {
        if (previous == TPS_EMPTY) return TPS_BOOL;
        else return TPS_STRING; // bool found with anything except empty is string
    }
    // numerical promotion
    if (previous == TPS_INT) {
        if (new == TPS_EMPTY || new == TPS_INT) return TPS_INT;
        if (new == TPS_FLOAT) return TPS_FLOAT;
        if (new == TPS_COMPLEX) return TPS_COMPLEX;
    }
    if (previous == TPS_FLOAT) {
        if (new == TPS_EMPTY || new == TPS_INT || new == TPS_FLOAT) return TPS_FLOAT;
        if (new == TPS_COMPLEX) return TPS_COMPLEX;
    }
    // previous == TPS_COMPLEX, new is TPS_EMPTY, TPS_INT, TPS_FLOAT, or TPS_COMPLEX
    return TPS_COMPLEX;
}

// Given a TypeParser state, return a dtype. Returns NULL on error.
PyArray_Descr*
AK_TPS_ToDtype(AK_TypeParserState state) {
    PyArray_Descr *dtype = NULL;

    switch (state) {
        case TPS_UNKNOWN:
            dtype = PyArray_DescrNewFromType(NPY_UNICODE);
            break;
        case TPS_EMPTY: // all empty defaults to string
            dtype = PyArray_DescrNewFromType(NPY_UNICODE);
            break;
        case TPS_STRING:
            dtype = PyArray_DescrNewFromType(NPY_UNICODE);
            break;
        case TPS_BOOL:
            dtype = PyArray_DescrNewFromType(NPY_BOOL);
            break;
        case TPS_INT:
            dtype = PyArray_DescrNewFromType(NPY_INT64);
            break;
        case TPS_FLOAT:
            dtype = PyArray_DescrNewFromType(NPY_FLOAT64);
            break;
        case TPS_COMPLEX:
            dtype = PyArray_DescrNewFromType(NPY_COMPLEX128);
            break;
    }
    if (dtype == NULL) return NULL; // assume error is set by PyArray_DescrFromType
    return dtype;
}

//------------------------------------------------------------------------------
// An AK_TypeParser accumulates the state in parsing a single code point line. It holds both "active" state in the progress of parsing each field as well as finalized state in parsed_line
typedef struct AK_TypeParser {
    bool previous_numeric;
    bool contiguous_numeric;
    bool contiguous_leading_space;
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

    AK_TypeParserState parsed_field; // state of current field
    AK_TypeParserState parsed_line; // state of current resolved line type

    Py_UCS4 tsep;
    Py_UCS4 decc;

} AK_TypeParser;

// Initialize all state. This returns no error. This is called once per field for each field in a code point line: this is why parsed_field is reset, but parsed_line is not.
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
AK_TP_New(Py_UCS4 tsep, Py_UCS4 decc)
{
    AK_TypeParser *tp = (AK_TypeParser*)PyMem_Malloc(sizeof(AK_TypeParser));
    if (tp == NULL) return (AK_TypeParser*)PyErr_NoMemory();
    AK_TP_reset_field(tp);
    tp->parsed_line = TPS_UNKNOWN;
    tp->tsep = tsep; // take tsep into context for auto eval?
    tp->decc = decc;
    return tp;
}

void
AK_TP_Free(AK_TypeParser* tp)
{
    PyMem_Free(tp);
}

//------------------------------------------------------------------------------

// Given a type parse, process a single character and update the type parser state in `parsed_field`. Return true when processing should continue, false when no further processing is necessary. `pos` is the raw position within the current field.
bool
AK_TP_ProcessChar(AK_TypeParser* tp,
        Py_UCS4 c,
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
    else if (AK_is_quote(c)) {
        // any quote, leading or otherwise, defines a string
        tp->parsed_field = TPS_STRING;
        return false;
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
        if (tp->count_paren_close > 1) {
            tp->parsed_field = TPS_STRING;
            return false;
        }
    }
    else {
        ++tp->count_not_space;
    }
    // no longer in contiguous leading space
    tp->contiguous_leading_space = false;

    // pos_field defines the position within the field less leading space
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
    else if (c == tp->decc) { // is decimal
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

    // given relative positions `pos_field`, map to known character sequences
    switch (pos_field) {
        case 0:
            if      (AK_is_t(c)) {++tp->count_bool;}
            else if (AK_is_f(c)) {--tp->count_bool;}
            else if (AK_is_n(c)) {++tp->count_nan;}
            else if (AK_is_i(c)) {++tp->count_inf;}
            else if (!numeric) { // if not decimal, sign, e, j
                tp->parsed_field = TPS_STRING;
                return false;
            }
            break;
        case 1:
            if      (AK_is_r(c)) {++tp->count_bool;} // true
            else if (AK_is_a(c)) {
                --tp->count_bool; // false
                ++tp->count_nan;
                }
            else if (AK_is_n(c)) {++tp->count_inf;}
            else if (!numeric) {
                tp->parsed_field = TPS_STRING;
                return false;
            }
            break;
        case 2:
            if      (AK_is_u(c)) {++tp->count_bool;}
            else if (AK_is_l(c)) {--tp->count_bool;}
            else if (AK_is_n(c)) {++tp->count_nan;}
            else if (AK_is_f(c)) {++tp->count_inf;}
            else if (!numeric) {
                tp->parsed_field = TPS_STRING;
                return false;
            }
            break;
        case 3:
            if      (AK_is_e(c)) {++tp->count_bool;} // true
            else if (AK_is_s(c)) {--tp->count_bool;} // false
            else if (!numeric) {
                tp->parsed_field = TPS_STRING;
                return false;
            }
            break;
        case 4:
            if      (AK_is_e(c)) {--tp->count_bool;} // false
            else if (!numeric) {
                tp->parsed_field = TPS_STRING;
                return false;
            }
            break;
        default: // character positions > 4
            if (!numeric) {
                tp->parsed_field = TPS_STRING;
                return false;
            }
    }
    return true; // continue processing
}

// This private function is used by AK_TP_ResolveLineResetField to evaluate the state of the AK_TypeParser and determine the resolved AK_TypeParserState.
AK_TypeParserState
AK_TP_resolve_field(AK_TypeParser* tp,
        Py_ssize_t count)
{
    if (count == 0) return TPS_EMPTY;

    // if parsed_field is known, return it
    if (tp->parsed_field != TPS_UNKNOWN) return tp->parsed_field;

    if (tp->count_bool == 4 && tp->count_not_space == 4) {
        return TPS_BOOL;
    }
    if (tp->count_bool == -5 && tp->count_not_space == 5) {
        return TPS_BOOL;
    }
    if (tp->contiguous_numeric) {
        if (tp->count_digit == 0) return TPS_STRING;
        // int
        if (tp->count_j == 0 &&
                tp->count_sign <= 1 &&
                tp->last_sign_pos <= 0 &&
                tp->count_decimal == 0 &&
                tp->count_e == 0 &&
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
        Py_ssize_t count_numeric = (tp->count_sign +
                tp->count_decimal +
                tp->count_e +
                tp->count_j +
                tp->count_digit);

        // one inf and one nan
        if (tp->count_nan == 3 && tp->count_inf == 3 &&
                tp->count_sign + 7 == tp->count_not_space) {
            return TPS_COMPLEX;
        }
        // one nan one number
        if (tp->count_nan == 3 &&
                tp->count_nan + count_numeric == tp->count_not_space) {
            return TPS_COMPLEX;
        }
        // two nans
        if (tp->count_nan == 6 &&
                tp->count_sign + tp->count_nan + 1 == tp->count_not_space) {
            return TPS_COMPLEX;
        }
        // one inf one number
        if (tp->count_inf == 3 &&
                tp->count_inf + count_numeric == tp->count_not_space) {
            return TPS_COMPLEX;
        }
        // two infs
        if (tp->count_inf == 6 &&
                tp->count_sign + tp->count_inf + 1 == tp->count_not_space) {
            return TPS_COMPLEX;
        }
    }
    return TPS_STRING; // default
}

// After field is complete, call AK_TP_ResolveLineResetField to evaluate and set the current parsed_line. This will be called after loading each character in the field. All TypeParse field attributes are reset after this is called. Returns true if the line still needs to be evaluated.
bool
AK_TP_ResolveLineResetField(AK_TypeParser* tp,
        Py_ssize_t count)
{
    if (tp->parsed_line != TPS_STRING) {
        // resolve with previous parsed_line (or unkown if just initialized)
        tp->parsed_line = AK_TPS_Resolve(tp->parsed_line, AK_TP_resolve_field(tp, count));
    }
    AK_TP_reset_field(tp);
    // if string, return false to stop further line processing
    return tp->parsed_line != TPS_STRING;
}

//------------------------------------------------------------------------------
// UCS4 array processors

static char* TRUE_LOWER = "true";
static char* TRUE_UPPER = "TRUE";

#define ERROR_NO_DIGITS 1
#define ERROR_OVERFLOW 2
#define ERROR_INVALID_CHARS 3

// Convert a Py_UCS4 array to a signed integer. Extended from pandas/_libs/src/parser/tokenizer.c. Sets `error` to values greater than 0 on error; never sets error on success.
static inline npy_int64
AK_UCS4_to_int64(Py_UCS4 *p_item, Py_UCS4 *end, int *error, char tsep)
{
    npy_int64 int_min = NPY_MIN_INT64;
    npy_int64 int_max = NPY_MAX_INT64;
    int isneg = 0;
    npy_int64 number = 0;
    int d;

    Py_UCS4 *p = p_item;

    while (AK_is_space(*p)) {
        ++p;
        if (p >= end) return number;
    }
    if (*p == '-') {
        isneg = 1;
        ++p;
    } else if (*p == '+') {
        ++p;
    }
    if (p >= end) return number;

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
                    if (p >= end) return number;
                    d = *p;
                    continue;
                } else if (!AK_is_digit(d)) {
                    break;
                }
                if ((number > pre_min) ||
                    ((number == pre_min) && (d - '0' <= dig_pre_min))) {
                    number = number * 10 - (d - '0');
                    ++p;
                    if (p >= end) return number;
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
                    if (p >= end) return number;
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
                    if (p >= end) return number;
                    d = *p;
                    continue;
                } else if (!AK_is_digit(d)) {
                    break;
                }
                if ((number < pre_max) ||
                    ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                    number = number * 10 + (d - '0');
                    ++p;
                    if (p >= end) return number;
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
                    if (p >= end) return number;
                    d = *p;
                } else {
                    *error = ERROR_OVERFLOW;
                    return 0;
                }
            }
        }
    }
    while (p < end) {
        if (!AK_is_space(*p)) {
            *error = ERROR_INVALID_CHARS;
            return 0;
        }
        p++;
    }
    return number;
}

// Convert a Py_UCS4 array to an unsigned integer. Extended from pandas/_libs/src/parser/tokenizer.c. Sets error to > 0 on error; never sets error on success.
static inline npy_uint64
AK_UCS4_to_uint64(Py_UCS4 *p_item, Py_UCS4 *end, int *error, char tsep)
{
    npy_uint64 pre_max = NPY_MAX_UINT64 / 10;
    npy_uint64 number = 0;
    int dig_pre_max = NPY_MAX_UINT64 % 10;
    int d;

    Py_UCS4 *p = p_item;
    while (AK_is_space(*p)) {
        ++p;
        if (p >= end) return number;
    }
    if (*p == '-') {
        *error = ERROR_INVALID_CHARS;
        return 0;
    } else if (*p == '+') {
        p++;
        if (p >= end) return number;
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
                if (p >= end) return number;
                d = *p;
                continue;
            } else if (!AK_is_digit(d)) {
                break;
            }
            if ((number < pre_max) ||
                ((number == pre_max) && (d - '0' <= dig_pre_max))) {
                number = number * 10 + (d - '0');
                ++p;
                if (p >= end) return number;
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
                if (p >= end) return number;
                d = *p;
            } else {
                *error = ERROR_OVERFLOW;
                return 0;
            }
        }
    }
    while (p < end) {
        if (!AK_is_space(*p)) {
            *error = ERROR_INVALID_CHARS;
            return 0;
        }
        p++;
    }
    return number;
}

// Based on precise_xstrtod from pandas/_libs/src/parser/tokenizer.c.
static inline npy_float64
AK_UCS4_to_float64(Py_UCS4 *p_item, Py_UCS4 *end, int *error, char tsep, char decc)
{
    // Cache powers of 10 in memory.
    static npy_float64 e[] = {
        1.,    1e1,   1e2,   1e3,   1e4,   1e5,   1e6,   1e7,   1e8,   1e9,
        1e10,  1e11,  1e12,  1e13,  1e14,  1e15,  1e16,  1e17,  1e18,  1e19,
        1e20,  1e21,  1e22,  1e23,  1e24,  1e25,  1e26,  1e27,  1e28,  1e29,
        1e30,  1e31,  1e32,  1e33,  1e34,  1e35,  1e36,  1e37,  1e38,  1e39,
        1e40,  1e41,  1e42,  1e43,  1e44,  1e45,  1e46,  1e47,  1e48,  1e49,
        1e50,  1e51,  1e52,  1e53,  1e54,  1e55,  1e56,  1e57,  1e58,  1e59,
        1e60,  1e61,  1e62,  1e63,  1e64,  1e65,  1e66,  1e67,  1e68,  1e69,
        1e70,  1e71,  1e72,  1e73,  1e74,  1e75,  1e76,  1e77,  1e78,  1e79,
        1e80,  1e81,  1e82,  1e83,  1e84,  1e85,  1e86,  1e87,  1e88,  1e89,
        1e90,  1e91,  1e92,  1e93,  1e94,  1e95,  1e96,  1e97,  1e98,  1e99,
        1e100, 1e101, 1e102, 1e103, 1e104, 1e105, 1e106, 1e107, 1e108, 1e109,
        1e110, 1e111, 1e112, 1e113, 1e114, 1e115, 1e116, 1e117, 1e118, 1e119,
        1e120, 1e121, 1e122, 1e123, 1e124, 1e125, 1e126, 1e127, 1e128, 1e129,
        1e130, 1e131, 1e132, 1e133, 1e134, 1e135, 1e136, 1e137, 1e138, 1e139,
        1e140, 1e141, 1e142, 1e143, 1e144, 1e145, 1e146, 1e147, 1e148, 1e149,
        1e150, 1e151, 1e152, 1e153, 1e154, 1e155, 1e156, 1e157, 1e158, 1e159,
        1e160, 1e161, 1e162, 1e163, 1e164, 1e165, 1e166, 1e167, 1e168, 1e169,
        1e170, 1e171, 1e172, 1e173, 1e174, 1e175, 1e176, 1e177, 1e178, 1e179,
        1e180, 1e181, 1e182, 1e183, 1e184, 1e185, 1e186, 1e187, 1e188, 1e189,
        1e190, 1e191, 1e192, 1e193, 1e194, 1e195, 1e196, 1e197, 1e198, 1e199,
        1e200, 1e201, 1e202, 1e203, 1e204, 1e205, 1e206, 1e207, 1e208, 1e209,
        1e210, 1e211, 1e212, 1e213, 1e214, 1e215, 1e216, 1e217, 1e218, 1e219,
        1e220, 1e221, 1e222, 1e223, 1e224, 1e225, 1e226, 1e227, 1e228, 1e229,
        1e230, 1e231, 1e232, 1e233, 1e234, 1e235, 1e236, 1e237, 1e238, 1e239,
        1e240, 1e241, 1e242, 1e243, 1e244, 1e245, 1e246, 1e247, 1e248, 1e249,
        1e250, 1e251, 1e252, 1e253, 1e254, 1e255, 1e256, 1e257, 1e258, 1e259,
        1e260, 1e261, 1e262, 1e263, 1e264, 1e265, 1e266, 1e267, 1e268, 1e269,
        1e270, 1e271, 1e272, 1e273, 1e274, 1e275, 1e276, 1e277, 1e278, 1e279,
        1e280, 1e281, 1e282, 1e283, 1e284, 1e285, 1e286, 1e287, 1e288, 1e289,
        1e290, 1e291, 1e292, 1e293, 1e294, 1e295, 1e296, 1e297, 1e298, 1e299,
        1e300, 1e301, 1e302, 1e303, 1e304, 1e305, 1e306, 1e307, 1e308};

    npy_float64 number = 0.0;
    int exponent = 0;
    bool negative_base = false;
    bool negative_e = false;

    int num_digits = 0;
    int max_digits = 17;
    int n = 0;

    Py_UCS4 *p = p_item;
    while (AK_is_space(*p)) {
        ++p;
        if (p >= end) return number;
    }
    switch (*p) {
        case '-':
            negative_base = true;
            // fall through
        case '+':
            p++;
            if (p >= end) return number; // nothing to do with sign
    }
    // check for inf, nan
    if (AK_is_i(*p)) {
        p++;
        if (p >= end) goto error;
        if (AK_is_n(*p)) {
            p++;
            if (p >= end) goto error;
            if (AK_is_f(*p)) {
                p++;
                while (p < end) {
                    if (!AK_is_space(*p++)) goto error;
                }
                if (negative_base) return -NPY_INFINITY;
                return NPY_INFINITY;
            }
        }
        goto error; // matched i but nothing else
    }
    else if (AK_is_n(*p)) {
        p++;
        if (p >= end) goto error;
        if (AK_is_a(*p)) {
            p++;
            if (p >= end) goto error;
            if (AK_is_n(*p)) {
                p++;
                while (p < end) {
                    if (!AK_is_space(*p++)) goto error;
                }
                return NPY_NAN;
            }
        }
        goto error; // matched n but nothing else
    }
    while (AK_is_digit(*p)) {
        if (num_digits < max_digits) {
            number = number * 10. + (*p - '0');
            num_digits++;
        } else {
            ++exponent;
        }
        p++;
        if (p >= end) goto exit;
        if (tsep != '\0' && *p == (Py_UCS4)tsep) {
            ++p;
            if (p >= end) goto exit;
        }
    }

    if (*p == (Py_UCS4)decc) {
        p++;
        if (p >= end) goto exit;

        while (num_digits < max_digits && AK_is_digit(*p)) {
            number = number * 10. + (*p - '0');
            num_digits++;
            exponent--;
            p++;
            if (p >= end) goto exit;
        }
        if (num_digits >= max_digits)  // Consume extra decimal digits.
            while (AK_is_digit(*p)) {
                ++p;
                if (p >= end) goto exit;
            }
    }
    if (num_digits == 0) {
        *error = ERANGE;
        return 0.0;
    }
    if (AK_is_e(*p)) {
        ++p;
        if (p >= end) goto exit;

        switch (*p) {
            case '-':
                negative_e = true;
                // fall through
            case '+':
                p++;
                if (p >= end) goto exit; // sign is not used
        }
        num_digits = 0; // reset
        while (num_digits < max_digits && AK_is_digit(*p)) {
            n = n * 10 + (*p - '0');
            num_digits++;
            p++;
            if (p >= end) goto exit;
        }
    }
    // if we have anything but space trailing, error
    while (p < end) {
        if (!AK_is_space(*p++)) goto error;
    }
exit:
    if (negative_base) number = -number;
    // n will be zero if no E found
    if (negative_e) {
        exponent -= n;
    }
    else {
        exponent += n;
    }
    // AK_DEBUG_MSG_OBJ("at exit", PyLong_FromLong(exponent));
    // done with p at this point
    if (exponent > 308) {
        *error = ERANGE;
        return HUGE_VAL;
    } else if (exponent > 0) {
        number *= e[exponent];
    } else if (exponent < -308) {  // Subnormal
        if (exponent < -616) {  // Prevent invalid array access.
            number = 0.;
        } else {
            number /= e[-308 - exponent];
            number /= e[308];
        }
    } else {
        number /= e[-exponent];
    }

    if (number == HUGE_VAL || number == -HUGE_VAL) *error = ERANGE;
    return number;
error:
    *error = 1;
    return number;
}

//------------------------------------------------------------------------------
// CodePointLine

// An AK_CodePointLine stores a contiguous buffer of Py_UCS4 without null terminators between fields. Separately, we store an array of integers, where each integer is the size of each field. The total number of fields is given by offset_count.
typedef struct AK_CodePointLine{
    // NOTE: should these be unsigned int types, like Py_uintptr_t?
    Py_ssize_t buffer_count; // accumulated number of code points
    Py_ssize_t buffer_capacity; // max number of code points
    Py_UCS4 *buffer;

    Py_ssize_t offsets_count; // accumulated number of elements, never reset
    Py_ssize_t offsets_capacity; // max number of elements
    Py_ssize_t *offsets;
    Py_ssize_t offset_max; // observe max offset found across all

    // these can be reset
    Py_UCS4 *buffer_current_ptr;
    Py_ssize_t offsets_current_index;

    AK_TypeParser *type_parser;
    bool type_parser_field_active;
    bool type_parser_line_active;

} AK_CodePointLine;

// on error return NULL
AK_CodePointLine*
AK_CPL_New(bool type_parse, Py_UCS4 tsep, Py_UCS4 decc)
{
    AK_CodePointLine *cpl = (AK_CodePointLine*)PyMem_Malloc(sizeof(AK_CodePointLine));
    if (cpl == NULL) return (AK_CodePointLine*)PyErr_NoMemory();

    cpl->buffer_count = 0;
    cpl->buffer_capacity =  16384; // 2048;
    cpl->buffer = (Py_UCS4*)PyMem_Malloc(sizeof(Py_UCS4) * cpl->buffer_capacity);
    if (cpl->buffer == NULL) {
        PyMem_Free(cpl);
        return (AK_CodePointLine*)PyErr_NoMemory();
    }
    cpl->offsets_count = 0;
    cpl->offsets_capacity = 2048; // 16384; // 2048;
    cpl->offsets = (Py_ssize_t*)PyMem_Malloc(sizeof(Py_ssize_t) * cpl->offsets_capacity);
    if (cpl->offsets == NULL) {
        PyMem_Free(cpl->buffer);
        PyMem_Free(cpl);
        return (AK_CodePointLine*)PyErr_NoMemory();
    }
    cpl->buffer_current_ptr = cpl->buffer;
    cpl->offsets_current_index = 0; // position in offsets
    cpl->offset_max = 0;

    // optional, dynamic values
    // cpl->field = NULL;
    if (type_parse) {
        cpl->type_parser = AK_TP_New(tsep, decc);
        if (cpl->type_parser == NULL) {
            PyMem_Free(cpl->offsets);
            PyMem_Free(cpl->buffer);
            PyMem_Free(cpl);
            return NULL; // exception already set
        }
        cpl->type_parser_field_active = true;
        cpl->type_parser_line_active = true;
    }
    else {
        cpl->type_parser = NULL;
        cpl->type_parser_field_active = false;
        cpl->type_parser_line_active = false;
    }
    return cpl;
}

void
AK_CPL_Free(AK_CodePointLine* cpl)
{
    PyMem_Free(cpl->buffer);
    PyMem_Free(cpl->offsets);
    if (cpl->type_parser) { // can exclude the check
        PyMem_Free(cpl->type_parser);
    }
    PyMem_Free(cpl);
}

//------------------------------------------------------------------------------
// CodePointLine: Mutation

static inline int
AK_CPL_resize_buffer(AK_CodePointLine* cpl, Py_ssize_t count)
{
    if (AK_UNLIKELY((cpl->buffer_count + count) >= cpl->buffer_capacity)) {
        // realloc
        cpl->buffer_capacity *= 2; // needs to be max of this or element_length
        cpl->buffer = PyMem_Realloc(cpl->buffer,
                sizeof(Py_UCS4) * cpl->buffer_capacity);
        if (cpl->buffer == NULL) return -1;

        cpl->buffer_current_ptr = cpl->buffer + cpl->buffer_count;
    }
    return 0;
}

static inline int
AK_CPL_resize_offsets(AK_CodePointLine* cpl)
{
    // increment by at most one, so only need to check if equal
    if (AK_UNLIKELY(cpl->offsets_count == cpl->offsets_capacity)) {
        // realloc
        cpl->offsets_capacity *= 2;
        cpl->offsets = PyMem_Realloc(cpl->offsets,
                sizeof(Py_ssize_t) * cpl->offsets_capacity);
        if (cpl->offsets == NULL) return -1;
    }
    return 0;
}

// Given a PyUnicode PyObject representing a complete field, load the string content into the CPL. Used for iterable_str_to_array_1d. Returns 0 on success, -1 on error.
static inline int
AK_CPL_AppendField(AK_CodePointLine* cpl, PyObject* field)
{
    if (!PyUnicode_Check(field)) { // NOTE: this permits subclasses, consider
        PyErr_SetString(PyExc_TypeError, "elements must be strings");
        return -1;
    }
    Py_ssize_t element_length = PyUnicode_GET_LENGTH(field);

    // if we cannot fit field length, resize
    if (AK_CPL_resize_buffer(cpl, element_length)) return -1;

    // we write teh field direclty into the CPL buffer
    if(PyUnicode_AsUCS4(field,
            cpl->buffer_current_ptr,
            cpl->buffer + cpl->buffer_capacity - cpl->buffer_current_ptr,
            0) == NULL) { // last zero means do not copy null
        return -1;
    }
    // if type parsing has been enabled, we must process each char
    if (cpl->type_parser && cpl->type_parser_line_active) {
        Py_UCS4* p = cpl->buffer_current_ptr;
        Py_UCS4 *end = p + element_length;
        Py_ssize_t pos = 0;
        for (; p < end; ++p) {
            cpl->type_parser_field_active = AK_TP_ProcessChar(
                    cpl->type_parser,
                    *p,
                    pos);
            if (!cpl->type_parser_field_active) break;
            ++pos;
        }
        cpl->type_parser_line_active = AK_TP_ResolveLineResetField(cpl->type_parser, element_length);
        cpl->type_parser_field_active = true; // turn back on for next field
    }

    // read offset_count, then increment
    if (AK_CPL_resize_offsets(cpl)) return -1;
    cpl->offsets[cpl->offsets_count++] = element_length;
    cpl->buffer_count += element_length;
    cpl->buffer_current_ptr += element_length; // add to pointer

    if (element_length > cpl->offset_max) {
        cpl->offset_max = element_length;
    }
    return 0;
}

// Add a single point (or chacter) to a line. This does not update offsets. This is valid when updating a character. Returns 0 on success, -1 on error.
static inline int
AK_CPL_AppendPoint(AK_CodePointLine* cpl,
        Py_UCS4 p,
        Py_ssize_t pos)
{
    // based on buffer_count, resize if we cannot fit one more character
    if (AK_CPL_resize_buffer(cpl, 1)) return -1;

    // type_parser might not be active if we already know the dtype
    if (cpl->type_parser
            && cpl->type_parser_line_active
            && cpl->type_parser_field_active) {
        cpl->type_parser_field_active = AK_TP_ProcessChar(
                cpl->type_parser,
                p,
                pos);
    }
    *cpl->buffer_current_ptr++ = p;
    ++cpl->buffer_count;
    return 0;
}

// Append to offsets. This does not update buffer lines. This is called when closing a field. Return -1 on failure, 0 on success.
static inline int
AK_CPL_AppendOffset(AK_CodePointLine* cpl, Py_ssize_t offset)
{
    // this will update cpl->offsets if necessary
    if (AK_CPL_resize_offsets(cpl)) return -1;

    if (cpl->type_parser && cpl->type_parser_line_active) {
        // when we resolve the line, we might determine that no further line processing is necessary
        cpl->type_parser_line_active = AK_TP_ResolveLineResetField(
                cpl->type_parser,
                offset);
        // NOTE: always turn on for next field; we choose not to check type_parser_line_active
        cpl->type_parser_field_active = true;
    }
    // increment offset_count after assignment so we can grow if needed next time
    cpl->offsets[cpl->offsets_count++] = offset;
    if (offset > cpl->offset_max) {cpl->offset_max = offset;}
    return 0;
}

//------------------------------------------------------------------------------
// CodePointLine: Constructors

// Given an iterable of unicode objects, load them into a AK_CodePointLine. Used for iterable_str_to_array_1d. Return NULL on errror.
AK_CodePointLine*
AK_CPL_FromIterable(PyObject* iterable, bool type_parse, Py_UCS4 tsep, Py_UCS4 decc)
{
    PyObject *iter = PyObject_GetIter(iterable);
    if (iter == NULL) return NULL;

    AK_CodePointLine *cpl = AK_CPL_New(type_parse, tsep, decc);
    if (cpl == NULL) {
        Py_DECREF(iter);
        return NULL;
    }

    PyObject *field;
    while ((field = PyIter_Next(iter))) {
        if (AK_CPL_AppendField(cpl, field)) {
            Py_DECREF(field);
            Py_DECREF(iter);
            return NULL;
        }
        Py_DECREF(field);
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) {
        return NULL;
    }
    return cpl;
}

//------------------------------------------------------------------------------
// CodePointLine: Navigation

// Cannot error.
void
AK_CPL_CurrentReset(AK_CodePointLine* cpl)
{
    cpl->buffer_current_ptr = cpl->buffer;
    cpl->offsets_current_index = 0;
}

// Advance the current position by the current offset. Cannot error.
static inline void
AK_CPL_CurrentAdvance(AK_CodePointLine* cpl)
{
    // use offsets_current_index, then increment
    cpl->buffer_current_ptr += cpl->offsets[cpl->offsets_current_index++];
}

//------------------------------------------------------------------------------
// This will take any case of "TRUE" as True, while marking everything else as False; this is the same approach taken with genfromtxt when the dtype is given as bool. This will not fail for invalid true or false strings.
static inline bool
AK_CPL_current_to_bool(AK_CodePointLine* cpl) {
    // must have at least 4 characters
    if (cpl->offsets[cpl->offsets_current_index] < 4) {
        return false;
    }
    Py_UCS4 *p = cpl->buffer_current_ptr;
    Py_UCS4 *end = p + 4; // we must have at least 4 characters for True
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
AK_CPL_current_to_int64(AK_CodePointLine* cpl, int *error, char tsep)
{
    Py_UCS4 *p = cpl->buffer_current_ptr;
    Py_UCS4 *end = p + cpl->offsets[cpl->offsets_current_index]; // size is either 4 or 5
    return AK_UCS4_to_int64(p, end, error, tsep);
}

// Provide start and end buffer positions to provide a range of bytes to read and transform into an integer. Returns 0 on error; does not set exception.
static inline npy_uint64
AK_CPL_current_to_uint64(AK_CodePointLine* cpl, int *error, char tsep)
{
    Py_UCS4 *p = cpl->buffer_current_ptr;
    Py_UCS4 *end = p + cpl->offsets[cpl->offsets_current_index];
    return AK_UCS4_to_uint64(p, end, error, tsep);
}

static inline npy_float64
AK_CPL_current_to_float64(AK_CodePointLine* cpl, int *error, char tsep, char decc)
{
    // interpret an empty field as NaN
    if (cpl->offsets[cpl->offsets_current_index] == 0) {
        return NPY_NAN;
    }
    Py_UCS4 *p = cpl->buffer_current_ptr;
    Py_UCS4 *end = p + cpl->offsets[cpl->offsets_current_index];
    return AK_UCS4_to_float64(p, end, error, tsep, decc);
}


// A wrapper to PyOS_string_to_double. Might set an exception on error.
// static inline npy_float64
// AK_CPL_current_to_float64(AK_CodePointLine* cpl)
// {
//     // interpret an empty field as NaN
//     if (cpl->offsets[cpl->offsets_current_index] == 0) {
//         return NPY_NAN;
//     }
//     char* field = AK_CPL_current_to_field(cpl);
//     // NOTE: field can be NULL on memory failure!
//     // NOTE: this is shown to be much faster than atof in stdlib.h
//     return PyOS_string_to_double(field, NULL, NULL);
// }


//------------------------------------------------------------------------------
// CodePointLine: Exporters

static inline PyObject*
AK_CPL_to_array_bool(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    npy_intp dims[] = {cpl->offsets_count};

    // initialize all values to False
    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }

    npy_bool *array_buffer = (npy_bool*)PyArray_DATA((PyArrayObject*)array);

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);
    for (int i=0; i < cpl->offsets_count; ++i) {
        // this is forgiving in that invalid strings remain false
        if (AK_CPL_current_to_bool(cpl)) {
            array_buffer[i] = 1;
        }
        AK_CPL_CurrentAdvance(cpl);
    }
    NPY_END_THREADS;

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// Given a type of signed integer, return the corresponding array.
static inline PyObject*
AK_CPL_to_array_float(AK_CodePointLine* cpl, PyArray_Descr* dtype, char tsep, char decc)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }
    // initialize error code to 0; only update on error.
    int error = 0;
    bool matched_elsize = true;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);

    if (dtype->elsize == 16) {
        # ifdef PyFloat128ArrType_Type
        npy_float128 *array_buffer = (npy_float128*)PyArray_DATA((PyArrayObject*)array);
        npy_float128 *end = array_buffer + count;
        while (array_buffer < end) {
            // NOTE: cannot cast to npy_float128 here
            *array_buffer++ = AK_CPL_current_to_float64(cpl, &error, tsep, decc);
            AK_CPL_CurrentAdvance(cpl);
        }
        # endif
    }
    else if (dtype->elsize == 8) {
        npy_float64 *array_buffer = (npy_float64*)PyArray_DATA((PyArrayObject*)array);
        npy_float64 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_float64(cpl, &error, tsep, decc);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_float32 *array_buffer = (npy_float32*)PyArray_DATA((PyArrayObject*)array);
        npy_float32 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_float32)AK_CPL_current_to_float64(cpl, &error, tsep, decc);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_float16 *array_buffer = (npy_float16*)PyArray_DATA((PyArrayObject*)array);
        npy_float16 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_float16)AK_CPL_current_to_float64(cpl, &error, tsep, decc);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        matched_elsize = false;
    }

    NPY_END_THREADS;

    if (!matched_elsize) {
        PyErr_SetString(PyExc_TypeError, "cannot create array from itemsize");
        Py_DECREF(array);
        return NULL;
    }
    if (error) {
        PyErr_SetString(PyExc_TypeError, "error parsing float");
        Py_DECREF(array);
        return NULL;
     }

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// Given a type of signed integer, return the corresponding array.
static inline PyObject*
AK_CPL_to_array_int(AK_CodePointLine* cpl, PyArray_Descr* dtype, char tsep)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }
    // initialize error code to 0; only update on error.
    int error = 0;
    bool matched_elsize = true;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);
    if (dtype->elsize == 8) {
        npy_int64 *array_buffer = (npy_int64*)PyArray_DATA((PyArrayObject*)array);
        npy_int64 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_int64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_int32 *array_buffer = (npy_int32*)PyArray_DATA((PyArrayObject*)array);
        npy_int32 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_int32)AK_CPL_current_to_int64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_int16 *array_buffer = (npy_int16*)PyArray_DATA((PyArrayObject*)array);
        npy_int16 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_int16)AK_CPL_current_to_int64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 1) {
        npy_int8 *array_buffer = (npy_int8*)PyArray_DATA((PyArrayObject*)array);
        npy_int8 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_int8)AK_CPL_current_to_int64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        matched_elsize = false;
    }
    NPY_END_THREADS;

    if (!matched_elsize) {
        PyErr_SetString(PyExc_TypeError, "cannot create array from integer itemsize");
        Py_DECREF(array);
        return NULL;
    }
    if (error) {
        PyErr_SetString(PyExc_TypeError, "error parsing integer");
        Py_DECREF(array);
        return NULL;
     }

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// Given a type of signed integer, return the corresponding array. Return NULL on error.
static inline PyObject*
AK_CPL_to_array_uint(AK_CodePointLine* cpl, PyArray_Descr* dtype, char tsep)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    PyObject *array = PyArray_Zeros(1, dims, dtype, 0); // steals dtype ref
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }
    // initialize error code to 0; only update on error.
    int error = 0;
    bool matched_elsize = true;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);
    if (dtype->elsize == 8) {
        npy_uint64 *array_buffer = (npy_uint64*)PyArray_DATA((PyArrayObject*)array);
        npy_uint64 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = AK_CPL_current_to_uint64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 4) {
        npy_uint32 *array_buffer = (npy_uint32*)PyArray_DATA((PyArrayObject*)array);
        npy_uint32 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint32)AK_CPL_current_to_uint64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 2) {
        npy_uint16 *array_buffer = (npy_uint16*)PyArray_DATA((PyArrayObject*)array);
        npy_uint16 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint16)AK_CPL_current_to_uint64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else if (dtype->elsize == 1) {
        npy_uint8 *array_buffer = (npy_uint8*)PyArray_DATA((PyArrayObject*)array);
        npy_uint8 *end = array_buffer + count;
        while (array_buffer < end) {
            *array_buffer++ = (npy_uint8)AK_CPL_current_to_uint64(cpl, &error, tsep);
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else {
        matched_elsize = false;
    }
    NPY_END_THREADS;

    if (!matched_elsize) {
        PyErr_SetString(PyExc_TypeError, "cannot create array from unsigned integer itemsize");
        Py_DECREF(array);
        return NULL;
    }
    if (error != 0) {
        PyErr_SetString(PyExc_TypeError, "error parsing unisigned integer");
        Py_DECREF(array);
        return NULL;
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

static inline PyObject*
AK_CPL_to_array_unicode(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    Py_ssize_t count = cpl->offsets_count;
    npy_intp dims[] = {count};

    Py_ssize_t field_points;
    bool capped_points;

    // mutate the passed dtype as it is new and will be stolen in array construction
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
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }

    Py_UCS4 *array_buffer = (Py_UCS4*)PyArray_DATA((PyArrayObject*)array);
    Py_UCS4 *end = array_buffer + count * field_points;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);
    if (capped_points) {
        // NOTE: is it worth branching for this special case?
        Py_ssize_t copy_bytes;
        while (array_buffer < end) {
            if (cpl->offsets[cpl->offsets_current_index] >= field_points) {
                copy_bytes = field_points * sizeof(Py_UCS4);
            } else {
                copy_bytes = cpl->offsets[cpl->offsets_current_index] * sizeof(Py_UCS4);
            }
            memcpy(array_buffer,
                    cpl->buffer_current_ptr,
                    copy_bytes);
            array_buffer += field_points;
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    else { // faster we always know the offset will fit
        while (array_buffer < end) {
            memcpy(array_buffer,
                    cpl->buffer_current_ptr,
                    cpl->offsets[cpl->offsets_current_index] * sizeof(Py_UCS4));
            array_buffer += field_points;
            AK_CPL_CurrentAdvance(cpl);
        }
    }
    NPY_END_THREADS;

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// Return NULL on error
static inline PyObject*
AK_CPL_to_array_bytes(AK_CodePointLine* cpl, PyArray_Descr* dtype)
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
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }

    char *array_buffer = (char*)PyArray_DATA((PyArrayObject*)array);
    char *end = array_buffer + count * field_points;

    Py_ssize_t copy_points;

    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS;

    AK_CPL_CurrentReset(cpl);
    while (array_buffer < end) {
        if (!capped_points || cpl->offsets[cpl->offsets_current_index] < field_points) {
            // if not capped, or capped and offset is less than field points, use offset
            copy_points = cpl->offsets[cpl->offsets_current_index];
        }
        else {
            // if capped and offset is greater than feild points, use field points
            copy_points = field_points;
        }

        // NOTE: not using memcopy as we need to cast to char to fit each point
        Py_UCS4 *p = cpl->buffer_current_ptr;
        Py_UCS4 *p_end = p + copy_points;
        char *field_end = array_buffer + field_points;

        while (p < p_end) {
            *array_buffer++ = (char)*p++; // truncate
        }
        array_buffer = field_end;
        AK_CPL_CurrentAdvance(cpl);
    }
    NPY_END_THREADS;

    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// If we cannot direclty convert bytes to values, create a bytes array and then use PyArray_CastToType to use numpy to interpet it as a new a array.
static inline PyObject*
AK_CPL_to_array_via_cast(AK_CodePointLine* cpl, PyArray_Descr* dtype)
{
    PyArray_Descr *dtype_bytes = PyArray_DescrNewFromType(NPY_STRING);
    if (dtype_bytes == NULL) {
        Py_DECREF(dtype);
        return NULL;
    }
    PyObject* array_bytes = AK_CPL_to_array_bytes(cpl, dtype_bytes);
    if (array_bytes == NULL) {
        Py_DECREF(dtype);
        // dtype_bytes stolen even if array creation failed
        return NULL;
    }
    PyObject *array = PyArray_CastToType((PyArrayObject*)array_bytes, dtype, 0);
    Py_DECREF(array_bytes);
    if (array == NULL) {
        // expected array to steal dtype reference
        return NULL;
    }
    PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
    return array;
}

// Generic handler for converting a CPL to an array. The dtype given here must already be a fresh instance as it might be mutated. If passed dtype is NULL, must get dtype from type_parser-> parsed_line Might return NULL if array creation fails; an exception should be set. Will return NULL on error.
static inline PyObject*
AK_CPL_ToArray(AK_CodePointLine* cpl, PyArray_Descr* dtype, char tsep, char decc) {
    if (!dtype) {
        // If we have a type_parser on the CPL, we can use that to get the dtype
        if (cpl->type_parser) {
            // will return a fresh instance
            dtype = AK_TPS_ToDtype(cpl->type_parser->parsed_line);
            if (dtype == NULL) return NULL;
        }
        else {
            AK_NOT_IMPLEMENTED("dtype not passed to AK_CPL_ToArray, and CodePointLine has no type_parser");
        }
    }

    if (PyDataType_ISBOOL(dtype)) {
        return AK_CPL_to_array_bool(cpl, dtype);
    }
    else if (PyDataType_ISFLOAT(dtype)) {
        return AK_CPL_to_array_float(cpl, dtype, tsep, decc);
    }
    else if (PyDataType_ISSTRING(dtype) && dtype->kind == 'U') {
        return AK_CPL_to_array_unicode(cpl, dtype);
    }
    else if (PyDataType_ISSTRING(dtype) && dtype->kind == 'S') {
        return AK_CPL_to_array_bytes(cpl, dtype);
    }
    else if (PyDataType_ISUNSIGNED(dtype)) { // must come before integer check
        return AK_CPL_to_array_uint(cpl, dtype, tsep);
    }
    else if (PyDataType_ISINTEGER(dtype)) {
        return AK_CPL_to_array_int(cpl, dtype, tsep);
    }
    else if (PyDataType_ISDATETIME(dtype)) {
        return AK_CPL_to_array_via_cast(cpl, dtype);
    }
    else if (PyDataType_ISCOMPLEX(dtype)) {
        return AK_CPL_to_array_via_cast(cpl, dtype); // no tsep, decc as using NumPy cast
    }

    PyErr_Format(PyExc_NotImplementedError, "No handling for %R", dtype);
    // caller will decref the passed dtype on error
    return NULL;
}

//------------------------------------------------------------------------------
// utility function used by CPG and DR
static inline int
AK_line_select_keep(
        PyObject *line_select,
        bool axis_target,
        int lookup_number)
{
    if (axis_target && (line_select != NULL)) {
        PyObject* number = PyLong_FromLong(lookup_number);
        if (number == NULL) return -1;

        PyObject* keep = PyObject_CallFunctionObjArgs(
                line_select,
                number,
                NULL
                );
        Py_DECREF(number);
        if (keep == NULL) {
            PyErr_Format(PyExc_RuntimeError,
                    "line_select callable failed for input: %d",
                    lookup_number
                    );
            return -1;
        }

        int t = PyObject_IsTrue(keep); // 1 if truthy
        Py_DECREF(keep);
        if (t < 0) {
            return -1; // error
        }
        return t; // 0 or 1
    }
    return 1;
}

//------------------------------------------------------------------------------
// CodePointGrid Type, New, Destructor

typedef struct AK_CodePointGrid {
    Py_ssize_t lines_count;    // accumulated number of lines
    Py_ssize_t lines_capacity; // max number of lines
    AK_CodePointLine **lines;  // array of pointers
    PyObject *dtypes;          // a callable that returns None or a dtype initializer
    Py_UCS4 tsep;
    Py_UCS4 decc;
} AK_CodePointGrid;

// Create a new Code Point Grid; returns NULL on error. Missing `dtypes` has been normalized as NULL.
AK_CodePointGrid*
AK_CPG_New(PyObject *dtypes, Py_UCS4 tsep, Py_UCS4 decc)
{
    // normalize dtypes to NULL or callable
    if ((dtypes == NULL) || (dtypes == Py_None)) {
        dtypes = NULL;
    }
    else if (!PyCallable_Check(dtypes)) {
        PyErr_SetString(PyExc_TypeError, "dtypes must be a callable or None");
        return NULL;
    }

    AK_CodePointGrid *cpg = (AK_CodePointGrid*)PyMem_Malloc(sizeof(AK_CodePointGrid));
    if (cpg == NULL) return (AK_CodePointGrid*)PyErr_NoMemory();

    cpg->tsep = tsep;
    cpg->decc = decc;
    cpg->lines_count = 0;
    cpg->lines_capacity = 1024;
    cpg->lines = (AK_CodePointLine**)PyMem_Malloc(
            sizeof(AK_CodePointLine*) * cpg->lines_capacity);
    if (cpg->lines == NULL) return (AK_CodePointGrid*)PyErr_NoMemory();

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
    PyMem_Free(cpg);
}
//------------------------------------------------------------------------------
// CodePointGrid: Mutation

// Determine if a new CPL needs to be created and add it if needed. Return 0 on succes, -1 on failure.
static inline int
AK_CPG_resize(AK_CodePointGrid* cpg, Py_ssize_t line)
{
    Py_ssize_t lines_count = cpg->lines_count;
    if (line < lines_count) return 0; // most common scenario

    if (AK_UNLIKELY(line >= cpg->lines_capacity)) {
        cpg->lines_capacity *= 2;
        // NOTE: we assume this only copies the pointers, not the data in the CPLs
        cpg->lines = PyMem_Realloc(cpg->lines,
                sizeof(AK_CodePointLine*) * cpg->lines_capacity);
        if (cpg->lines == NULL) return -1;
    }
    // Create the new CPL; first check if we need to set type_parse by calling into the dtypes function. For now we assume sequential growth, so should only check if equal
    if (AK_UNLIKELY(line >= lines_count)) {
        // determine if we need to parse types
        bool type_parse = false;
        if (cpg->dtypes == NULL) {
            type_parse = true;
        }
        else {
            PyObject* line_count = PyLong_FromLong(line);
            if (line_count == NULL) return -1;

            PyObject* dtype_specifier = PyObject_CallFunctionObjArgs(
                    cpg->dtypes,
                    line_count,
                    NULL
                    );
            Py_DECREF(line_count);

            if (dtype_specifier == NULL) {
                // NOTE: not sure how to get the exception from the failed call...
                PyErr_Format(PyExc_RuntimeError,
                        "dtypes callable failed for input: %d",
                        line
                        );
                return -1;
            }
            if (dtype_specifier == Py_None) {
                type_parse = true;
            }
            Py_DECREF(dtype_specifier);
        }
        // Always initialize a CPL in the new position
        AK_CodePointLine *cpl = AK_CPL_New(type_parse, cpg->tsep, cpg->decc);
        if (cpl == NULL) return -1; // memory error set

        cpg->lines[line] = cpl;
        ++cpg->lines_count;
    }
    return 0;
}

// Append a point on the line; called for each character in a field. Return 0 on success, -1 on failure.
static inline int
AK_CPG_AppendPointAtLine(
        AK_CodePointGrid* cpg,
        Py_ssize_t line, // number
        Py_ssize_t field_len,
        Py_UCS4 p
        )
{
    if (AK_CPG_resize(cpg, line)) return -1;
    if (AK_CPL_AppendPoint(cpg->lines[line], p, field_len)) return -1;
    return 0;
}

// Append an offset in a line. Returns 0 on success, -1 on failure.
static inline int
AK_CPG_AppendOffsetAtLine(
        AK_CodePointGrid* cpg,
        Py_ssize_t line,
        Py_ssize_t offset)
{
    // only need to call this if AK_CPG_AppendPointAtLine has not yet been called
    if (AK_CPG_resize(cpg, line)) return -1;
    if (AK_CPL_AppendOffset(cpg->lines[line], offset)) return -1;
    return 0;
}

// Given a fully-loaded CodePointGrid, process each CodePointLine into an array and return a new list of those arrays. Returns NULL on failure.
PyObject* AK_CPG_ToArrayList(AK_CodePointGrid* cpg,
        int axis,
        PyObject* line_select,
        char tsep,
        char decc)
{
    bool ls_inactive = line_select == NULL;
    PyObject *list;

    if (ls_inactive) {
        // if we know how many lines we will need, can pre-allocate
        list = PyList_New(cpg->lines_count);
    }
    else {
        list = PyList_New(0);
    }
    if (list == NULL) return NULL;

    PyObject* dtypes = cpg->dtypes;

    // Iterate over lines in the code point grid
    for (int i = 0; i < cpg->lines_count; ++i) {
        // if axis is axis 1, apply keep
        switch (AK_line_select_keep(line_select, 1 == axis, i)) {
            case -1:
                Py_DECREF(list);
                return NULL;
            case 0:
                continue;
        }
        // If dtypes is not NULL, fetch the dtype_specifier and use it to set dtype; else, pass the dtype as NULL to CPL.
        PyArray_Descr* dtype = NULL;

        if (dtypes != NULL) {
            // NOTE: we call this with i regardless of if we skipped a line
            PyObject* line_count = PyLong_FromLong(i);
            if (line_count == NULL) {
                Py_DECREF(list);
                return NULL;
            }
            PyObject* dtype_specifier = PyObject_CallFunctionObjArgs(
                    dtypes,
                    line_count,
                    NULL
                    );
            Py_DECREF(line_count);
            if (dtype_specifier == NULL) {
                Py_DECREF(list);
                // NOTE: not sure how to get the exception from the failed call...
                PyErr_Format(PyExc_RuntimeError,
                        "dtypes callable failed for input: %d",
                        i
                        );
                return NULL;
            }
            if (dtype_specifier != Py_None) {
                // Set dtype; this value can be NULL or a dtype (never Py_None); if dtype_specifier is Py_None, keep dtype set as NULL (above); this will be a new reference that if used will be stolen in array construction.
                if (AK_DTypeFromSpecifier(dtype_specifier, &dtype)) {
                    Py_DECREF(dtype_specifier);
                    Py_DECREF(list);
                    return NULL;
                }
            }
            Py_DECREF(dtype_specifier);
        }
        // This function will observe if dtype is NULL and read dtype from the CPL's type_parser if necessary
        // NOTE: this creating might be multi-threadable for dtypes that permit C-only buffer transfers
        PyObject* array = AK_CPL_ToArray(cpg->lines[i], dtype, tsep, decc);
        // AK_DEBUG_MSG_OBJ("post AK_CPL_ToArray", dtype);
        if (array == NULL) {
            // if array creation has been aborted due to a bad character, we will already have decrefed the array, which seems to also decref dtype
            Py_DECREF(list);
            return NULL;
        }

        if (ls_inactive) {
            PyList_SET_ITEM(list, i, array); // steals reference
        }
        else {
            if (PyList_Append(list, array)) {
                Py_DECREF(array);
                Py_DECREF(list);
                return NULL;
            }
            Py_DECREF(array); // decref as list owns
        }
    }
    return list;
}

//------------------------------------------------------------------------------
// AK_Dialect, based on _csv.c from CPython

typedef enum AK_DialectQuoteStyle {
    QUOTE_MINIMAL,
    QUOTE_ALL,
    QUOTE_NONNUMERIC, // NOTE: kept to match csv module, but has no effect here
    QUOTE_NONE
} AK_DialectQuoteStyle;

typedef struct AK_DialectStyleDesc{
    AK_DialectQuoteStyle style;
    const char *name;
} AK_DialectStyleDesc;

static const AK_DialectStyleDesc AK_Dialect_quote_styles[] = {
    { QUOTE_MINIMAL,    "QUOTE_MINIMAL" },
    { QUOTE_ALL,        "QUOTE_ALL" },
    { QUOTE_NONNUMERIC, "QUOTE_NONNUMERIC" },
    { QUOTE_NONE,       "QUOTE_NONE" },
    { 0 }
};

static int
AK_Dialect_check_quoting(int quoting)
{
    const AK_DialectStyleDesc *qs;
    for (qs = AK_Dialect_quote_styles; qs->name; qs++) {
        if ((int)qs->style == quoting) // can we compare to long and avoid
            return 0;
    }
    PyErr_Format(PyExc_TypeError, "bad \"quoting\" value");
    return -1;
}

typedef struct AK_Dialect{
    bool doublequote;           // is " represented by ""?
    bool skipinitialspace;      // ignore spaces following delimiter?
    bool strict;                // raise exception on bad CSV
    int quoting;                // style of quoting to write
    Py_UCS4 delimiter;          // field separator
    Py_UCS4 quotechar;          // quote character
    Py_UCS4 escapechar;         // escape character
} AK_Dialect;

// check types and convert to C values
#define AK_Dialect_CALL_SETTER(meth, name, target, src, default) \
    do {\
        if (meth(name, target, src, default)) \
            goto error; \
    } while (0) \

static AK_Dialect*
AK_Dialect_New(PyObject *delimiter,
        PyObject *doublequote,
        PyObject *escapechar,
        PyObject *quotechar,
        PyObject *quoting,
        PyObject *skipinitialspace,
        PyObject *strict
        )
{
    AK_Dialect *dialect = (AK_Dialect *) PyMem_Malloc(sizeof(AK_Dialect));
    if (dialect == NULL) return (AK_Dialect *)PyErr_NoMemory();

    Py_XINCREF(delimiter);
    Py_XINCREF(doublequote);
    Py_XINCREF(escapechar);
    Py_XINCREF(quotechar);
    Py_XINCREF(quoting);
    Py_XINCREF(skipinitialspace);
    Py_XINCREF(strict);

    // all goto error on error from function used in macro setting
    AK_Dialect_CALL_SETTER(AK_set_char,
            "delimiter",
            &dialect->delimiter,
            delimiter,
            ',');
    AK_Dialect_CALL_SETTER(AK_set_bool,
            "doublequote",
            &dialect->doublequote,
            doublequote,
            true);
    AK_Dialect_CALL_SETTER(AK_set_char,
            "escapechar",
            &dialect->escapechar,
            escapechar,
            0);
    AK_Dialect_CALL_SETTER(AK_set_char,
            "quotechar",
            &dialect->quotechar,
            quotechar,
            '"');
    AK_Dialect_CALL_SETTER(AK_set_int,
            "quoting",
            &dialect->quoting,
            quoting,
            QUOTE_MINIMAL); // we set QUOTE_MINIMAL by default
    AK_Dialect_CALL_SETTER(AK_set_bool,
            "skipinitialspace",
            &dialect->skipinitialspace,
            skipinitialspace,
            false);
    AK_Dialect_CALL_SETTER(AK_set_bool,
            "strict",
            &dialect->strict,
            strict,
            false);

    if (AK_Dialect_check_quoting(dialect->quoting))
        goto error;

    if (dialect->delimiter == 0) {
        PyErr_SetString(PyExc_TypeError,
                "\"delimiter\" must be a 1-character string");
        goto error;
    }
    if (quotechar == Py_None && quoting == NULL)
        dialect->quoting = QUOTE_NONE;

    if (dialect->quoting != QUOTE_NONE && dialect->quotechar == 0) {
        PyErr_SetString(PyExc_TypeError,
               "quotechar must be set if quoting enabled");
        goto error;
    }
    // done with all pyobjects
    Py_CLEAR(delimiter);
    Py_CLEAR(doublequote);
    Py_CLEAR(escapechar);
    Py_CLEAR(quotechar);
    Py_CLEAR(quoting);
    Py_CLEAR(skipinitialspace);
    Py_CLEAR(strict);
    return dialect;
error:
    Py_CLEAR(delimiter);
    Py_CLEAR(doublequote);
    Py_CLEAR(escapechar);
    Py_CLEAR(quotechar);
    Py_CLEAR(quoting);
    Py_CLEAR(skipinitialspace);
    Py_CLEAR(strict);
    // We may have gone to error after allocating dialect but found an error in a parameter
    PyMem_Free(dialect);
    return NULL;
}

void
AK_Dialect_Free(AK_Dialect* dialect)
{
    PyMem_Free(dialect); // might alrady be NULL
}

//------------------------------------------------------------------------------
// AK_DelimitedReader, based on _csv.c from CPython

typedef enum AK_DelimitedReaderState {
    START_RECORD,
    START_FIELD,
    ESCAPED_CHAR,
    IN_FIELD,
    IN_QUOTED_FIELD,
    ESCAPE_IN_QUOTED_FIELD,
    QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL,
    AFTER_ESCAPED_CRNL
} AK_DelimitedReaderState;

typedef struct AK_DelimitedReader{
    PyObject *input_iter;
    PyObject *line_select;
    AK_Dialect *dialect;
    AK_DelimitedReaderState state;
    Py_ssize_t field_len;
    Py_ssize_t record_number; // total records loaded
    Py_ssize_t record_iter_number; // records iterated (counting exclusion)
    Py_ssize_t field_number; // field in current record, reset for each record
    int axis;
    Py_ssize_t *axis_pos; // points to either record_number or field_number
} AK_DelimitedReader;

// Called once at the close of each field in a line. Returns 0 on success, -1 on failure
static inline int
AK_DR_close_field(AK_DelimitedReader *dr, AK_CodePointGrid *cpg)
{
    if (AK_CPG_AppendOffsetAtLine(cpg,
            *(dr->axis_pos),
            dr->field_len)) return -1;
    dr->field_len = 0; // clear to close
    // AK_DEBUG_MSG_OBJ("closing field", PyLong_FromLong(dr->field_number));
    ++dr->field_number; // increment after adding each offset, reset in AK_DR_line_reset
    return 0;
}

// Called once to add each character, appending that character to the CPL. Return 0 on success, -1 on failure.
static inline int
AK_DR_add_char(AK_DelimitedReader *dr, AK_CodePointGrid *cpg, Py_UCS4 c)
{
    // NOTE: ideally we could use line_select here; however, we would need to cache the lookup in another container as this is called once per char and line_select is a Python function; further, we would need to increment the field_number separately from another counter, which is done in AK_DR_close_field

    if (AK_CPG_AppendPointAtLine(cpg,
            *(dr->axis_pos),
            dr->field_len,
            c)) return -1;
    ++dr->field_len; // reset in AK_DR_close_field
    return 0;
}

// Process each char and update AK_DelimitedReader state. When appropriate, call AK_DR_add_char to accumulate field characters, AK_DR_close_field to end a field. Return -1 on failure, 0 on success.
static int
AK_DR_process_char(AK_DelimitedReader *dr, AK_CodePointGrid *cpg, Py_UCS4 c)
{
    AK_Dialect *dialect = dr->dialect;

    switch (dr->state) {
    case START_RECORD: // start of record
        if (c == '\0') // empty line
            break;
        else if (c == '\n' || c == '\r') {
            dr->state = EAT_CRNL;
            break;
        }
        dr->state = START_FIELD; // normal character
        // fallthru
    case START_FIELD: // expecting field
        if (c == '\n' || c == '\r' || c == '\0') { // save empty field
            if (AK_DR_close_field(dr, cpg)) return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->quotechar && dialect->quoting != QUOTE_NONE) {
            // start quoted field
            dr->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->escapechar) { // possible escaped character
            dr->state = ESCAPED_CHAR;
        }
        else if (c == ' ' && dialect->skipinitialspace);
        else if (c == dialect->delimiter) { // save empty field
            if (AK_DR_close_field(dr, cpg)) return -1;
        }
        else { // begin new unquoted field
            if (AK_DR_add_char(dr, cpg, c)) return -1;
            dr->state = IN_FIELD;
        }
        break;
    case ESCAPED_CHAR:
        if (c == '\n' || c=='\r') {
            if (AK_DR_add_char(dr, cpg, c)) return -1;
            dr->state = AFTER_ESCAPED_CRNL;
            break;
        }
        if (c == '\0')
            c = '\n';
        if (AK_DR_add_char(dr, cpg, c)) return -1;
        dr->state = IN_FIELD;
        break;
    case AFTER_ESCAPED_CRNL:
        if (c == '\0') break;
        // fallthru
    case IN_FIELD: // in unquoted field
        if (c == '\n' || c == '\r' || c == '\0') { // end of line
            if (AK_DR_close_field(dr, cpg)) return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (c == dialect->escapechar) { // possible escaped character
            dr->state = ESCAPED_CHAR;
        }
        else if (c == dialect->delimiter) { // save field - wait for new field
            if (AK_DR_close_field(dr, cpg)) return -1;
            dr->state = START_FIELD;
        }
        else { // normal character - save in field
            if (AK_DR_add_char(dr, cpg, c)) return -1;
        }
        break;
    case IN_QUOTED_FIELD: // in quoted field
        if (c == '\0');
        else if (c == dialect->escapechar) {
            dr->state = ESCAPE_IN_QUOTED_FIELD;
        }
        else if (c == dialect->quotechar && dialect->quoting != QUOTE_NONE) {
            dr->state = (dialect->doublequote ? QUOTE_IN_QUOTED_FIELD : IN_FIELD);
        }
        else { // normal character - save in field
            if (AK_DR_add_char(dr, cpg, c)) return -1;
        }
        break;
    case ESCAPE_IN_QUOTED_FIELD:
        if (c == '\0') {
            c = '\n';
        }
        if (AK_DR_add_char(dr, cpg, c)) return -1;
        dr->state = IN_QUOTED_FIELD;
        break;
    case QUOTE_IN_QUOTED_FIELD:
        // doublequote - seen a quote in a quoted field
        if (dialect->quoting != QUOTE_NONE && c == dialect->quotechar) {
            // save "" as "
            if (AK_DR_add_char(dr, cpg, c)) return -1;
            dr->state = IN_QUOTED_FIELD;
        }
        else if (c == dialect->delimiter) { // save field - wait for new field
            if (AK_DR_close_field(dr, cpg)) return -1;
            dr->state = START_FIELD;
        }
        else if (c == '\n' || c == '\r' || c == '\0') { // end of line
            if (AK_DR_close_field(dr, cpg)) return -1;
            dr->state = (c == '\0' ? START_RECORD : EAT_CRNL);
        }
        else if (!dialect->strict) {
            if (AK_DR_add_char(dr, cpg, c)) return -1;
            dr->state = IN_FIELD;
        }
        else { // illegal
            PyErr_Format(PyExc_RuntimeError, "'%c' expected after '%c'",
                    dialect->delimiter,
                    dialect->quotechar);
            return -1;
        }
        break;
    case EAT_CRNL:
        if (c == '\n' || c == '\r');
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

// Called once at the start of processing each line in AK_DR_ProcessRecord. This cannot error.
static void
AK_DR_line_reset(AK_DelimitedReader *dr)
{
    dr->field_len = 0;
    dr->state = START_RECORD;
    dr->field_number = 0;
}

// Using AK_DelimitedReader's state, process one record (via next(input_iter)); call AK_DR_process_char on each char in that line, loading individual fields into AK_CodePointGrid. Returns 1 when there are more lines to process, 0 when there are no lines to process, and -1 for error.
static int
AK_DR_ProcessRecord(AK_DelimitedReader *dr,
        AK_CodePointGrid *cpg,
        PyObject *line_select
        )
{
    Py_UCS4 c;
    Py_ssize_t pos, linelen;
    unsigned int kind;
    const void *data;
    PyObject *record;

    AK_DR_line_reset(dr);
    do {
        // get a string, representing one record, to parse
        record = PyIter_Next(dr->input_iter);
        if (record == NULL) {
            if (PyErr_Occurred()) return -1;
            // if parser is in an unexptected state
            if ((dr->field_len != 0) || (dr->state == IN_QUOTED_FIELD)) {
                if (dr->dialect->strict) {
                    PyErr_SetString(PyExc_RuntimeError, "unexpected end of data");
                    return -1;
                }
                // try to close the field, propagate error
                if (AK_DR_close_field(dr, cpg)) return -1;
            }
            return 0; // end of input, not an error
        }
        ++dr->record_iter_number;

        if (!PyUnicode_Check(record)) {
            PyErr_Format(PyExc_RuntimeError,
                    "iterator should return strings, not %.200s "
                    "(the file should be opened in text mode)",
                    Py_TYPE(record)->tp_name
                    );
            Py_DECREF(record);
            return -1;
        }
        if (PyUnicode_READY(record) == -1) {
            Py_DECREF(record);
            return -1;
        }

        switch (AK_line_select_keep(line_select,
                0 == dr->axis,
                dr->record_iter_number)) {
            case -1 :
                Py_DECREF(record);
                return -1;
            case 0:
                Py_DECREF(record);
                return 1; // skip, process more records
        }
        // NOTE: record_number should reflect the processed line count, and exlude any skipped lines. The value is initialized to -1 such the first line is number 0
        ++dr->record_number;
        // AK_DEBUG_MSG_OBJ("processing line", PyLong_FromLong(dr->record_number));

        kind = PyUnicode_KIND(record);
        data = PyUnicode_DATA(record);
        pos = 0;
        linelen = PyUnicode_GET_LENGTH(record);
        while (linelen--) {
            c = PyUnicode_READ(kind, data, pos);
            if (c == '\0') {
                Py_DECREF(record);
                PyErr_Format(PyExc_RuntimeError, "line contains NUL");
                return -1;
            }
            if (AK_DR_process_char(dr, cpg, c)) {
                Py_DECREF(record);
                return -1;
            }
            pos++;
        }
        Py_DECREF(record);
        // force signaling we are at the end of a line
        if (AK_DR_process_char(dr, cpg, '\0')) return -1;

    } while (dr->state != START_RECORD);
    return 1; // more lines to process
}

static void
AK_DR_Free(AK_DelimitedReader *dr)
{
    if (dr->dialect) {
        AK_Dialect_Free(dr->dialect);
    }
    Py_XDECREF(dr->input_iter); // might already be NULL
    PyMem_Free(dr);
}

// The arguments to this constructor are validated before this function is valled. Returns NULL on error.
static AK_DelimitedReader*
AK_DR_New(PyObject *iterable,
        int axis,
        PyObject *delimiter,
        PyObject *doublequote,
        PyObject *escapechar,
        PyObject *quotechar,
        PyObject *quoting,
        PyObject *skipinitialspace,
        PyObject *strict
        )
{
    AK_DelimitedReader *dr = (AK_DelimitedReader*)PyMem_Malloc(sizeof(AK_DelimitedReader));
    if (dr == NULL) return (AK_DelimitedReader*)PyErr_NoMemory();

    dr->axis = axis;

    // we configure axis_pos to be a pointer that points to either record_number or field_number to avoid doing this per char/ field.
    if (axis == 0) {
        dr->axis_pos = &(dr->record_number);
    }
    else {
        dr->axis_pos = &(dr->field_number);
    }

    dr->record_number = -1;
    dr->record_iter_number = -1;
    dr->dialect = NULL; // init in case input_iter fails to init

    dr->input_iter = PyObject_GetIter(iterable); // new ref, decref in free
    if (dr->input_iter == NULL) {
        AK_DR_Free(dr);
        return NULL;
    }

    dr->dialect = AK_Dialect_New(
            delimiter,
            doublequote,
            escapechar,
            quotechar,
            quoting,
            skipinitialspace,
            strict);
    if (dr->dialect == NULL) {
        AK_DR_Free(dr);
        return NULL;
    }
    return dr;
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// Convert an sequence of strings to a 1D array.
static inline PyObject*
AK_IterableStrToArray1D(
    PyObject *sequence,
    PyObject *dtype_specifier,
    Py_UCS4 tsep,
    Py_UCS4 decc)
{
    PyArray_Descr* dtype = NULL;
    // will set dtype_specifier to NULL for None, and propagate NULLs
    if (AK_DTypeFromSpecifier(dtype_specifier, &dtype)) return NULL;

    // dtype only NULL from here
    bool type_parse = dtype == NULL;

    AK_CodePointLine* cpl = AK_CPL_FromIterable(sequence, type_parse, tsep, decc);
    if (cpl == NULL) return NULL;

    PyObject* array = AK_CPL_ToArray(cpl, dtype, tsep, decc);
    AK_CPL_Free(cpl);
    return array; // might be NULL
}

//------------------------------------------------------------------------------
// AK module public methods
//------------------------------------------------------------------------------

static char *delimited_to_ararys_kwarg_names[] = {
    "file_like",
    "axis",
    "dtypes",
    "line_select",
    "delimiter",
    "doublequote",
    "escapechar",
    "quotechar",
    "quoting",
    "skipinitialspace",
    "strict",
    "thousandschar",
    "decimalchar",
    NULL
};

// NOTE: implement skip_header, skip_footer in client Python, not here.
static PyObject*
delimited_to_arrays(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyObject *file_like;
    int axis = 0;
    PyObject *dtypes = NULL;
    PyObject *line_select = NULL;
    PyObject *delimiter = NULL;
    PyObject *doublequote = NULL;
    PyObject *escapechar = NULL;
    PyObject *quotechar = NULL;
    PyObject *quoting = NULL;
    PyObject *skipinitialspace = NULL;
    PyObject *strict = NULL;
    PyObject *thousandschar = NULL;
    PyObject *decimalchar = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|$iOOOOOOOOOOO:delimited_to_arrays",
            delimited_to_ararys_kwarg_names,
            &file_like,
            // kwarg only
            &axis,
            &dtypes,
            &line_select,
            &delimiter,
            &doublequote,
            &escapechar,
            &quotechar,
            &quoting,
            &skipinitialspace,
            &strict,
            &thousandschar,
            &decimalchar))
        return NULL;

    // normalize line_select to NULL or callable
    if ((line_select == NULL) || (line_select == Py_None)) {
        line_select = NULL;
    }
    else if (!PyCallable_Check(line_select)) {
        PyErr_SetString(PyExc_TypeError, "line_select must be a callable or None");
        return NULL;
    }

    if ((axis < 0) || (axis > 1)) {
        PyErr_SetString(PyExc_ValueError, "axis must be 0 or 1");
        return NULL;
    }
    AK_DelimitedReader *dr = AK_DR_New(file_like,
            axis,
            delimiter,
            doublequote,
            escapechar,
            quotechar,
            quoting,
            skipinitialspace,
            strict);
    if (dr == NULL) { // can happen due to validation of dialect parameters
        return NULL;
    }

    Py_UCS4 tsep;
    if (AK_set_char(
            "thousandschar",
            &tsep,
            thousandschar,
            '\0')) {
        AK_DR_Free(dr);
        return NULL; // default is off (skips evaluation)
    }
    Py_UCS4 decc;
    if (AK_set_char(
            "decimalchar",
            &decc,
            decimalchar,
            '.')) {
        AK_DR_Free(dr);
        return NULL;
    }

    // dtypes inc / dec ref bound within CPG life
    AK_CodePointGrid* cpg = AK_CPG_New(dtypes, tsep, decc);
    if (cpg == NULL) { // error will be set
        AK_DR_Free(dr);
        return NULL;
    }
    // Consume all lines from dr and load into cpg
    int status;
    while (true) {
        status = AK_DR_ProcessRecord(dr, cpg, line_select);
        if (status == 1) {
            continue; // more lines to process
        }
        else if (status == 0) {
            break;
        }
        else if (status == -1) {
            AK_DR_Free(dr);
            AK_CPG_Free(cpg);
            return NULL;
        }
        // NOTE: could use PyErr_CheckSignals() at some number of dr->record_number
    }
    AK_DR_Free(dr);

    PyObject* arrays = AK_CPG_ToArrayList(cpg, axis, line_select, tsep, decc);
    // NOTE: do not need to check if arrays is NULL as we will return NULL anyway
    AK_CPG_Free(cpg); // will free reference to dtypes
    return arrays; // could be NULL
}

static char *iterable_str_to_array_1d_kwarg_names[] = {
    "iterable",
    "dtype",
    "thousandschar",
    "decimalchar",
    NULL
};

static PyObject *
iterable_str_to_array_1d(PyObject *Py_UNUSED(m), PyObject *args, PyObject *kwargs)
{
    PyObject *iterable = NULL;
    PyObject *dtype_specifier = NULL;
    PyObject *thousandschar = NULL;
    PyObject *decimalchar = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|OOO:iterable_str_to_array_1d",
            iterable_str_to_array_1d_kwarg_names,
            &iterable,
            // kwarg only
            &dtype_specifier,
            &thousandschar,
            &decimalchar))
        return NULL;

    Py_UCS4 tsep;
    if (AK_set_char(
            "thousandschar",
            &tsep,
            thousandschar,
            '\0')) return NULL;

    Py_UCS4 decc;
    if (AK_set_char(
            "decimalchar",
            &decc,
            decimalchar,
            '.')) return NULL;

    return AK_IterableStrToArray1D(iterable, dtype_specifier, tsep, decc);
}


static PyObject *
split_after_count(PyObject *Py_UNUSED(m), PyObject *args)
{
    PyObject *string = NULL;
    PyObject *delimiter = NULL;
    int count = 0;

    if (!PyArg_ParseTuple(args,
            "OOi:split_after_count",
            &string,
            &delimiter,
            &count)) {
        return NULL;
    }

    if (!PyUnicode_Check(string)) {
        PyErr_Format(PyExc_RuntimeError,
                "a string is required, not %.200s",
                Py_TYPE(string)->tp_name
                );
        return NULL;
    }

    if (count <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                "count must be greater than zero, not %i",
                count
                );
        return NULL;
    }

    Py_UCS4 delim_char;
    if (AK_set_char(
            "delimiter",
            &delim_char,
            delimiter,
            '\0')) return NULL;

    unsigned int kind = PyUnicode_KIND(string);
    const void *data = PyUnicode_DATA(string);
    Py_ssize_t pos = 0;
    Py_ssize_t delim_count = 0;
    Py_ssize_t linelen = PyUnicode_GET_LENGTH(string);
    Py_UCS4 c;

    while (pos < linelen) {
        c = PyUnicode_READ(kind, data, pos);
        if (c == delim_char) {
            delim_count++;
            if (delim_count == count) {
                break; // to not include delim at transition
                // do not increment pos so as to exclude in left
            }
        }
        pos++;
    }

    PyObject* left = PyUnicode_Substring(string, 0, pos);
    PyObject* right = PyUnicode_Substring(string, pos+1, linelen);
    PyObject *result = PyTuple_Pack(2, left, right);
    Py_DECREF(left);
    Py_DECREF(right);
    return result;
}



static PyObject *
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

static char *array_deepcopy_kwarg_names[] = {
    "array",
    "memo",
    NULL
};

// Specialized array deepcopy that stores immutable arrays in an optional memo dict that can be provided with kwargs.
static PyObject *
array_deepcopy(PyObject *m, PyObject *args, PyObject *kwargs)
{
    PyObject *array;
    PyObject *memo = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs,
            "O|O!:array_deepcopy", array_deepcopy_kwarg_names,
            &array,
            &PyDict_Type, &memo)) {
        return NULL;
    }
    AK_CHECK_NUMPY_ARRAY(array);
    return AK_ArrayDeepCopy(m, (PyArrayObject*)array, memo);
}



// Wites array bytes to an open, writeable file. Possibly return number of bytes written. This is similar to what tofile() does but tofile() cannot be used on a _ZipWriteFile when writing into a zip (raises io.UnsupportedOperation: fileno)
static PyObject *
array_bytes_to_file(PyObject *Py_UNUSED(m), PyObject *args)
{

    PyObject *array;
    PyObject *file;

    if (!PyArg_ParseTuple(args, "OO:array_bytes_to_file",
            &array, &file)) // how to validate file type?
    {
        return NULL;
    }
    AK_CHECK_NUMPY_ARRAY(array);

    PyObject *write_func = PyObject_GetAttrString(file, "write");
    if (!write_func) {
        goto error;
    }
    PyObject *mv;
    PyObject *ret;
    size_t elsize = PyArray_DESCR((PyArrayObject*)array)->elsize;

    // this is what PyArray_ToFile to does
    if (PyArray_ISCONTIGUOUS((PyArrayObject*)array)) {
        npy_intp size = PyArray_SIZE((PyArrayObject*)array);
        // might use PyMemoryView_GetContiguous
        mv = PyMemoryView_FromMemory(PyArray_DATA((PyArrayObject*)array), size * elsize, 0);
        ret = PyObject_CallFunctionObjArgs(write_func, mv, NULL);
        Py_DECREF(mv);
        Py_DECREF(ret);    }
    else {
        PyArrayIterObject *it = (PyArrayIterObject *) PyArray_IterNew(array);
        if (it == NULL) {
            return NULL;
        }
        while (it->index < it->size) {
            mv = PyMemoryView_FromMemory(it->dataptr, elsize, 0);
            ret = PyObject_CallFunctionObjArgs(write_func, mv, NULL);

            PyArray_ITER_NEXT(it);
            Py_DECREF(mv);
            Py_DECREF(ret);
        }
        Py_DECREF(it);
    }
    Py_DECREF(write_func);

    // dummy return
    PyObject *post = PyLong_FromLong(3);
    if (!post) {
        return NULL;
    }
    return post;

error:
    return NULL;

}

    // can create memory view object and pass this to the write method
    // PyObject *PyMemoryView_FromMemory(char *mem, Py_ssize_t size, int flags)
    // PyObject *PyMemoryView_GetContiguous(PyObject *obj, int buffertype, char order)

// from PyArray_ToString: create an empty bytes object and write to it

        // fwrite((const void *)it->dataptr,
        //             (size_t) PyArray_DESCR(self)->elsize,
        //             1, fp)


        // ret = PyBytes_FromStringAndSize(NULL, (Py_ssize_t) numbytes);
        // if (ret == NULL) {
        //     Py_DECREF(it);
        //     return NULL;
        // }
        // dptr = PyBytes_AS_STRING(ret);
        // i = it->size;
        // elsize = PyArray_DESCR(self)->elsize;
        // while (i--) {
        //     memcpy(dptr, it->dataptr, elsize);
        //     dptr += elsize;
        //     PyArray_ITER_NEXT(it);
        // }
        // Py_DECREF(it);



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
// general utility

static PyObject *
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

static PyObject *
isna_element(PyObject *Py_UNUSED(m), PyObject *arg)
{
    // None
    if (arg == Py_None) {
        Py_RETURN_TRUE;
    }

    // NaN
    if (PyFloat_Check(arg)) {
        return PyBool_FromLong(isnan(PyFloat_AS_DOUBLE(arg)));
    }
    if (PyArray_IsScalar(arg, Half)) {
        return PyBool_FromLong(npy_half_isnan(PyArrayScalar_VAL(arg, Half)));
    }
    if (PyArray_IsScalar(arg, Float32)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float32)));
    }
    if (PyArray_IsScalar(arg, Float64)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float64)));
    }
    # ifdef PyFloat128ArrType_Type
    if (PyArray_IsScalar(arg, Float128)) {
        return PyBool_FromLong(isnan(PyArrayScalar_VAL(arg, Float128)));
    }
    # endif

    // Complex NaN
    if (PyComplex_Check(arg)) {
        Py_complex val = ((PyComplexObject*)arg)->cval;
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    if (PyArray_IsScalar(arg, Complex64)) {
        npy_cfloat val = PyArrayScalar_VAL(arg, Complex64);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    if (PyArray_IsScalar(arg, Complex128)) {
        npy_cdouble val = PyArrayScalar_VAL(arg, Complex128);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    # ifdef PyComplex256ArrType_Type
    if (PyArray_IsScalar(arg, Complex256)) {
        npy_clongdouble val = PyArrayScalar_VAL(arg, Complex256);
        return PyBool_FromLong(isnan(val.real) || isnan(val.imag));
    }
    # endif

    // NaT - Datetime
    if (PyArray_IsScalar(arg, Datetime)) {
        return PyBool_FromLong(PyArrayScalar_VAL(arg, Datetime) == NPY_DATETIME_NAT);
    }

    // NaT - Timedelta
    if (PyArray_IsScalar(arg, Timedelta)) {
        return PyBool_FromLong(PyArrayScalar_VAL(arg, Timedelta) == NPY_DATETIME_NAT);
    }

    Py_RETURN_FALSE;
}

//------------------------------------------------------------------------------

static PyObject *
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

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!:get_new_indexers_and_screen", kwlist,
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

    PyObject *num_unique_pyint = PyLong_FromLong(num_unique);
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
    npy_int64 num_found = 0;
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
    PyObject *new_positions = PySequence_GetSlice((PyObject*)order_found, 0, num_found);
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
    {"array_deepcopy",
            (PyCFunction)array_deepcopy,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"array_bytes_to_file", array_bytes_to_file, METH_VARARGS, NULL},
    {"resolve_dtype", resolve_dtype, METH_VARARGS, NULL},
    {"resolve_dtype_iter", resolve_dtype_iter, METH_O, NULL},
    {"delimited_to_arrays",
            (PyCFunction)delimited_to_arrays,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"iterable_str_to_array_1d",
            (PyCFunction)iterable_str_to_array_1d,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {"split_after_count", split_after_count, METH_VARARGS, NULL},
    {"count_iteration", count_iteration, METH_O, NULL},
    {"isna_element", isna_element, METH_O, NULL},
    {"dtype_from_element", dtype_from_element, METH_O, NULL},
    {"get_new_indexers_and_screen",
            (PyCFunction)get_new_indexers_and_screen,
            METH_VARARGS | METH_KEYWORDS,
            NULL},
    {NULL},
};

static struct PyModuleDef arraykit_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_arraykit",
    .m_doc = NULL,
    .m_size = -1,
    .m_methods = arraykit_methods,
};

PyObject *
PyInit__arraykit(void)
{
    import_array();
    PyObject *m = PyModule_Create(&arraykit_module);

    PyObject *copy = PyImport_ImportModule("copy");
    if (!copy) {
        Py_XDECREF(m);
        return NULL;
    }
    PyObject *deepcopy = PyObject_GetAttrString(copy, "deepcopy");
    Py_DECREF(copy);
    if (!deepcopy) {
        Py_XDECREF(m);
        return NULL;
    }

    if (!m ||
        PyModule_AddStringConstant(m, "__version__", Py_STRINGIFY(AK_VERSION)) ||
        PyType_Ready(&ArrayGOType) ||
        PyModule_AddObject(m, "ArrayGO", (PyObject *) &ArrayGOType) ||
        PyModule_AddObject(m, "deepcopy", deepcopy))
    {
        Py_DECREF(deepcopy);
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

