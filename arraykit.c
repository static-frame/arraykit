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
// CodePointLine Type, New, Destrctor

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
// CPL Mutation

int AK_CPL_Append(AK_CodePointLine* cpl, PyObject* element)
{
    Py_ssize_t element_length = PyUnicode_GET_LENGTH(element);

    if ((cpl->buffer_count + element_length) >= cpl->buffer_capacity) {
        // realloc
        cpl->buffer_capacity *= 2;
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
    cpl->offsets[cpl->offsets_count] = element_length;
    ++(cpl->offsets_count);
    ++(cpl->index_current);

    cpl->buffer_count += element_length;
    cpl->pos_current += element_length; // add to pointer
    return 0;
}

//------------------------------------------------------------------------------
// CPL Constructors

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

    Py_DECREF(iter);
    return cpl;
}

//------------------------------------------------------------------------------
// CPL Navigation

void AK_CPL_CurrentReset(AK_CodePointLine* cpl)
{
    cpl->pos_current = cpl->buffer;
    cpl->index_current = 0;
}

static inline void AK_CPL_CurrentAdvance(AK_CodePointLine* cpl)
{
    cpl->pos_current += cpl->offsets[cpl->index_current];
    ++(cpl->index_current);
}

static inline void AK_CPL_CurrentRetreat(AK_CodePointLine* cpl)
{
    if (cpl->index_current > 0) {
        // can remove one
        --(cpl->index_current);
        // remove the offset at this new position
        cpl->pos_current -= cpl->offsets[cpl->index_current];
    }
}

//------------------------------------------------------------------------------
// CPL: Code Point Parsers

// This will take any case of "TRUE" as True, while marking everything else as False; this is the same approach taken with genfromtxt when the dtype is given as bool. This will not fail for invalid true or false strings.
// NP's Boolean conversion in genfromtxt: https://github.com/numpy/numpy/blob/0721406ede8b983b8689d8b70556499fc2aea28a/numpy/lib/_iotools.py#L386
static inline int AK_CPL_IsTrue(AK_CodePointLine* cpl) {
    // must have at least 1 characters
    if (cpl->offsets[cpl->index_current] < 4) {
        return 0;
    }
    static char* t_lower = "true";
    static char* t_upper = "TRUE";

    Py_UCS4 *p = cpl->pos_current;
    Py_UCS4 *end = p + 4; // we must have at least 4 characters
    int i = 0;
    char c;

    for (;p < end; ++p) {
        c = *p;
        if (c == t_lower[i] || c == t_upper[i]) {
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

    static char* t_lower = "true";
    static char* t_upper = "TRUE";
    static char* f_lower = "false";
    static char* f_upper = "FALSE";
    int score = 0;
    int i = 0;
    char c;

    for (;p < end; ++p) {
        c = *p;
        if (score >= 0 && (c == t_lower[i] || c == t_upper[i])) {
            ++score;
        }
        else if (score <= 0 && (c == f_lower[i] || c == f_upper[i])) {
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

//------------------------------------------------------------------------------
// CPL: Exporters

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
        // if (AK_CPL_ParseBoolean(cpl) == 1) {
        //     array_buffer[i] = 1;
        // }
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
        ++(cpg->lines_count);
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
    PyObject* list = PyList_New(0);
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
        else if (PyDataType_ISCOMPLEX(dtype)) {
            AK_NOT_IMPLEMENTED("no handling for complex dtype yet");
        }
        else {
            AK_NOT_IMPLEMENTED("no handling for other dtypes yet");
        }

        if (PyList_Append(list, array)) {
           // handle error
        }
        Py_DECREF(array);
    }
    return list;
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

// // Convert an sequence of strings to a 1D array.
// static inline PyObject*
// AK_IterableStrToArray1D(
//         PyObject *sequence,
//         PyObject *dtype_specifier)
// {
//         // Convert specifier into a dtype if necessary
//         PyArray_Descr* dtype;
//         if (PyObject_TypeCheck(dtype_specifier, &PyArrayDescr_Type))
//         {
//             dtype = (PyArray_Descr* )dtype_specifier;
//         }
//         else {
//             // Use converter2 here so that None returns NULL (as opposed to default dtype float); NULL will leave strings unchanged
//             PyArray_DescrConverter2(dtype_specifier, &dtype);
//         }

//         PyObject* array;
//         if (!dtype) {
//             AK_NOT_IMPLEMENTED("no handling for undefined dtype yet");
//         }
//         else if (PyDataType_ISBOOL(dtype)) {
//             array = AK_SequenceStrToArray1DBoolean(sequence);
//         }
//         else if (PyDataType_ISCOMPLEX(dtype)) {
//             array = AK_SequenceStrToArray1DComplex(sequence);
//         }
//         else {
//             // NOTE: Some types can be converted directly from strings. int, float, dt64, variously sized strings: will convert from strings correctly. bool dtypes: will treat string as truthy/falsy
//             Py_XINCREF(dtype); // will steal
//             array = PyArray_FromAny(sequence,
//                     dtype, // can be NULL, object: strings will remain
//                     1,
//                     1,
//                     NPY_ARRAY_FORCECAST, // not sure this is best
//                     NULL);
//         }
//         if (!array) {
//             return NULL;
//         }
//         PyArray_CLEARFLAGS((PyArrayObject *)array, NPY_ARRAY_WRITEABLE);
//         return array;
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
    if (!arrays) {
        AK_CPG_Free(cpg);
        return NULL;
    }

    AK_CPG_Free(cpg);
    return arrays;
}




// static PyObject *
// delimited_to_arrays(PyObject *Py_UNUSED(m), PyObject *args)
// {
//     PyObject *file_like, *dtypes, *axis;
//     if (!PyArg_ParseTuple(args, "OOO:delimited_to_arrays",
//             &file_like,
//             &dtypes,
//             &axis)) // TODO: enforce this is an int?
//     {
//         return NULL;
//     }
//     // NOTE: consider taking shape_estimate as a tuple to pre-size lists?

//     // Parse text
//     // For now, we import and use the CSV module directly; in the future, can vend code from here and implement support for an axis argument to collect data columnwise rather than rowwise
//     PyObject *module_csv = PyImport_ImportModule("csv");
//     if (!module_csv) {
//         return NULL;
//     }
//     PyObject *reader = PyObject_GetAttrString(module_csv, "reader");
//     Py_DECREF(module_csv);
//     if (!reader) {
//         return NULL;
//     }
//     // TODO: pass in full parameters for parsing
//     PyObject *reader_instance = PyObject_CallFunctionObjArgs(reader, file_like, NULL);
//     Py_DECREF(reader);
//     if (!reader_instance) {
//         return NULL;
//     }

//     // Get the an iterator for the appropriate axis
//     PyObject *axis_sequence_fast; // for generic assignment
//     Py_ssize_t count_columns = -1;

//     PyObject *axis0_sequence_iter = PyObject_GetIter(reader_instance);
//     if (!axis0_sequence_iter) {
//         Py_DECREF(reader_instance);
//         return NULL;
//     }

//     if (PyLong_AsLong(axis) == 1) { // this can fail
//         PyObject* axis1_sequences = PyList_New(0);
//         if (!axis1_sequences) {
//             Py_DECREF(axis0_sequence_iter);
//             return NULL;
//         }
//         // get first size
//         Py_ssize_t count_row = 0;

//         PyObject *row;
//         PyObject* column;
//         while ((row = PyIter_Next(axis0_sequence_iter))) {
//             // get count of columns from first row
//             if (count_row == 0) {
//                 count_columns = PyList_Size(row);
//                 // use Py_ssize_t?
//                 for (int i=0; i < count_columns; ++i) {
//                     column = PyList_New(0);
//                     if (PyList_Append(axis1_sequences, column)) // does not steal
//                     {
//                         Py_DECREF(row);
//                         Py_DECREF(axis1_sequences);
//                         return NULL;
//                     }
//                     Py_DECREF(column);
//                 }
//             }
//             // walk through row and append to columns
//             for (int i=0; i < count_columns; ++i) {
//                 PyObject* element = PyList_GetItem(row, i);
//                 column = PyList_GetItem(axis1_sequences, i); // borrowed?
//                 if (PyList_Append(column, element))
//                     {
//                         Py_DECREF(axis1_sequences);
//                         return NULL;
//                     }
//                 Py_DECREF(element);
//             }
//             ++count_row;
//         }
//         axis_sequence_fast = PySequence_Fast(axis1_sequences,
//                 "failed to create sequence");
//         // axis_sequence_iter = PyObject_GetIter(axis1_sequences);
//         Py_DECREF(axis1_sequences);
//     } else {
//         // axis_sequence_iter = axis0_sequence_iter;
//         axis_sequence_fast = PySequence_Fast(axis0_sequence_iter,
//                 "failed to create sequence");
//         count_columns = PySequence_Fast_GET_SIZE(axis_sequence_fast);
//     }
//     Py_DECREF(axis0_sequence_iter);
//     Py_DECREF(reader_instance);

    // List to be return constructer arrays
//     PyObject* arrays = PyList_New(0);
//     if (!arrays) {
//         Py_DECREF(axis_sequence_fast);
//         return NULL;
//     }

//     PyObject *line;
//     for (Py_ssize_t pos = 0; pos < count_columns; ++pos) {
//         line = PySequence_Fast_GET_ITEM(axis_sequence_fast, pos); // borrowed ref
//         if (!line) {
//             return NULL;
//         }
//         PyObject* dtype_specifier = PyList_GetItem(dtypes, pos);
//         if (!dtype_specifier) {
//             Py_DECREF(axis_sequence_fast);
//             Py_DECREF(arrays);
//             return NULL;
//         }

//         Py_INCREF(line); // got a borrowed ref
//         PyObject* array = AK_IterableStrToArray1D(line, dtype_specifier);
//         if (!array) {
//             return NULL;
//         }

//         if (PyList_Append(arrays, array)) {
//             Py_DECREF(axis_sequence_fast);
//             Py_DECREF(array);
//             Py_DECREF(arrays);
//             return NULL;
//         }
//         Py_DECREF(array);
//     }
//     // check if PyErr occurred
//     Py_DECREF(axis_sequence_fast);
//     return arrays;
// }

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
