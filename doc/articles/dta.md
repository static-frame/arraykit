
# Understanding `delimited_to_arrays`: A (Mostly) Faster-than-Pandas CSV Reader

The `delimited_to_arrays` function takes an iterable of strings (which might be an iterator of lines from a file) and extracts fields from that string based on a delimiter. Fields are accumulated into "lines" of contiguous Unicode byte data; lines can accumulate fields by row (axis 0) or by column (axis 1). As lines accumulate fields, types are optionally evaluated based on a heuristic, content- and count-based evaluation. After a single pass of reading the file, all lines are loaded and (optionally per line) types have been evaluated.

After loading, each line is converted to an array. For each line, an empty array of appropriate size can be created. For most types (bool, ints, float, Unicode, bytes), byte data is directly translated into C types that are assigned into the array's buffer. For a few types (datetime64, complex), byte arrays are created, and then a NumPy cast is used to deliver the desired type; this is a stopgap approach until implementation of direct value converters.

The CPython interface to `delimited_to_arrays` is given below:

```C
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
```

The implementation limits features to those essential to the C implementation. The `file_like` parameter is an iterable of strings; this removes concern for file opening and encoding configuration (if necessary). The `axis` parameter is provided with an integer. Both the `dtypes` the `line_select` parameters are Python functions that, when called with an index, return the per-line parameter. For `dytpes`, a dtype initializer can be provided per line to set the type, or `None` to use type evaluation. For `line_select`, a Boolean is returned to select inclusion of the line on the specified axis. (The use of functions for these parameters permits callers to consolidate heterogenous argument types into a single interface.)

As the delimited file parser is built on the Python CSV reader, all relevant parameters from that interface are retained here, including full support for diverse quoting and escaping configurations. A few features can more easily be implemented external to the C implementation: optionally skipping leading or trailing file lines, for example, can be done by preparing the generator of strings.

A few features are closely tied to the expected usage in StaticFrame. Having `dtypes` and `line_select` given as functions permits these values to be specified as sequences or mappings associated with index or columns labels. The `axis` argument (when set to 0) is useful for realizing arrays for columns, where each row is a depth level of a single type. Finally, NumPy Unicode and datetime64 types can be directly created.

Performance of `delimited_to_arrays` on Linux compared to Pandas for loading uniform-typed data (without type evaluation) is around 1.5 times faster than Pandas at scales of 100,000 rows and 500 columns. This ratio appears robust at larger scales; at smaller scales `delimited_to_arrays` approaches 2 times faster. Type evaluation incurs overhead to reduce the outperformance to approximately 1.1 times faster.

An alternative interface,`iterable_str_to_array_1d`, provides a way to load and convert a single line to a single array. This is mostly useful for testing.


## CPython C Orientation

C is not an object-oriented language. We can, however, create collections of state (like instance attributes) with `struct` types, and then create collections of functions (like instance methods) that take "instances" of that `struct` and perform in-place or other operations. The convention used here names `struct`s with full upper camel case names (e.g., `AK_CodePointLine`) while functions that take those `struct`s use abbreviations (e.g., `AK_CPL_New`).

Dynamic allocation happens with "New" functions (e.g., `AK_CPL_New`) using `PyMem_Malloc` while memory freeing happens with "Free" functions (e.g. `AK_CPL_Free`) using `PyMem_Free`. Public functions are named with upper camel case (e.g., `AK_CPL_ToArray`) while private functions (called only within the public functions of the same type group) are named with lower snake case (e.g., `AK_CPL_to_array_bool`).

In C we often mutate arguments in place, and a lot of our structs are designed to have mutable state. We do this because in C we often use returned values for error signaling, and thus pass in pointers for values to be set. We also do this for memory efficiency.

In C, we use `goto`. This is a common pattern in CPython where, on error, we often need to tear-down a bunch of memory and can get reuse of those routines through a direct jump.

In CPython, you have to manage reference counts. Failing to do so properly leads either to segmentation faults (over freeing) or memory leaks (under freeing). A few good resources are listed here:

* https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html
* https://docs.python.org/3/c-api/intro.html#objects-types-and-reference-counts

These routines make extensive use of pointers to arrays of characters (or Unicode code points). Recall that the handle to a C array is a pointer to its head; addition to that pointer "moves" it to point to that many positions forward in the array. Adding one to an array pointer returns a pointer to the next item. Nothing will stop you from advancing beyond the end of the array, so pointer arithmetic must often check that the end of the array has not been exceeded.

In C, pointer arithmetic and pointer dereferencing can happen with concise expressions. There are a couple of common moves.

We often want to dereference a pointer (to get a charcter) and then increment the pointer to be ready to dereference the next character. An example is the following while loop, where we advance through all characters that are spaces.

```C
while (AK_is_space(*p)) p++;
```

A variation of this move is done in assignment, where we assign to a dereferenced pointer and then increment the same pointer to be ready to assign to the next character.

```C
*array_buffer++ = (npy_int32)AK_CPL_current_to_int64(cpl, &error);
```

We can do the same on the right hand side of the assignment, such that we dereference a source and assign it to a dereferenced destination, then increment the source pointer and increment the destination pointer.

```C
*array_buffer++ = (char)*p++;
```


## Main Components

This section summarizes the main `struct`s needed to implement `delimited_to_arrays` as well as their functions. The ordering of this section follows the ordering of the code.


### AK_TypeParserState (AK_TPS)

An Enum that defines the type interpretation of a field or line of fields. Can be unknown (on initialization) or empty (on encountering empty values). Available functions permit "resolving" the type of two states. Unlike with normal dtype resolution, the dtype of "no-return" is string, not object.

* `AK_TPS_Resolve`
* `AK_TPS_ToDtype`


### AK_TypeParser (AK_TP)

A struct used to store the state of the type evaluation of a field, and ultimately of a complete line. One instance is created per line if type evaluation is enabled.

```C
typedef struct AK_TypeParser {
    bool previous_numeric;
    bool contiguous_numeric;
    bool contiguous_leading_space;

    npy_int8 count_bool;
    npy_int8 count_sign;
    npy_int8 count_e;
    npy_int8 count_j;
    npy_int8 count_decimal;
    npy_int8 count_nan;
    npy_int8 count_inf;
    npy_int8 count_paren_open;
    npy_int8 count_paren_close;

    Py_ssize_t last_sign_pos;
    Py_ssize_t count_leading_space;
    Py_ssize_t count_digit;
    Py_ssize_t count_not_space;

    AK_TypeParserState parsed_field;
    AK_TypeParserState parsed_line;

} AK_TypeParser;
```

For each field, counts of numerous indicators are measured as well as the contiguity of various characteristics. This is done by calling `AK_TP_ProcessChar` once for each character per line. After each field is completed, `AK_TP_ResolveLineResetField` is called to determine the evaluated field type (with `AK_TPS_Resolve`) as well as the incrementally evaluated line type.

* `AK_TP_New`
* `AK_TP_Free`
* `AK_TP_reset_field`
* `AK_TP_ProcessChar`
* `AK_TP_resolve_field`
* `AK_TP_ResolveLineResetField`


### Py_UCS4 Converter Functions

These function are designed to convert a region of contiguous byte data (given a pointer to the start and end) to a C type.

These functions are different than common library functions in that they take start/end pointers, not a NULL-terminated string. Common library functions (like `atof`, `strtod` or, from the CPython API, `PyOS_string_to_double`) require a null-terminated string.

All of these functions are based on Pandas tokenizer.c implementations, though adapted to work with a buffer range.

* `AK_UCS4_to_int64`
* `AK_UCS4_to_uint64`
* `AK_UCS4_to_float64`

There exist many ways of converting floats to strings. Perhaps the most accurate is `PyOS_string_to_double`, which offers round trip float representations for all values without noise. This function, however, requires a NULL-terminated string, and at well over a thousand lines, seemed far too complicated to replicate. While I explored an implementation that used this function, it required copying characters out of the contiguous buffer to a pre-allocated null-terminated string, which appeared to materially degrade performance. The `AK_UCS4_to_float64` implementation used here matches Pandas default configuration, though Pandas does permit optionally enabling `PyOS_string_to_double`. The following StackOverflow discussion provides additional context.

* https://stackoverflow.com/questions/44697714/on-the-float-precision-argument-to-pandas-read-csv/73465048#73465048



### AK_CodePointLine (CPL)

A representation of a "line", or a linear collection of fields which might be a column (when axis is 1) or a row (when axis is 0).

```C
typedef struct AK_CodePointLine{
    Py_ssize_t buffer_count;
    Py_ssize_t buffer_capacity;
    Py_UCS4 *buffer;

    Py_ssize_t offsets_count;
    Py_ssize_t offsets_capacity;
    Py_ssize_t *offsets;
    Py_ssize_t offset_max;

    Py_UCS4 *buffer_current_ptr;
    Py_ssize_t offsets_current_index;

    AK_TypeParser *type_parser;
    bool type_parser_field_active;
    bool type_parser_line_active;

} AK_CodePointLine;
```

The core representation consists of two parts, A dynamically growable contiguous array of `Py_UCS4` (`buffer`) (with no null terminators), and a dynamically growable array of offsets, providing the number of `Py_UCS4` in each field and (implicitly, or explicitly with `offset_count`) the number of fields.

The CPL also tracks the `offset_max`, the largest offset observed, while processing each field. This is needed when determining the element size of Unicode or bytes dtypes.

In addition, a CPL on creation optionally composes an `AK_TypeParser` instance. This permits the CPL to optionally call `AK_TP_ProcessChar` for each character as accumulating characters.

In general operation, a CPL has to phases: a loading phase, and conversion phase.

In the loading phase, `AK_CPL_AppendPoint` is called for each point, which in turn calls `AK_TP_ProcessChar` for each character if `type_parser` is active. At the end of each field, `AK_CPL_AppendOffset` is called, which calls `AK_TP_ResolveLineResetField` if `type_parser` is active. This permits loading a CPL in one pass, optionally evaluating type at the same time.

* `AK_CPL_New`
* `AK_CPL_Free`
* `AK_CPL_resize_buffer`
* `AK_CPL_resize_offsets`
* `AK_CPL_AppendPoint`
* `AK_CPL_AppendOffset`

In the conversion phase, a CPL exporter is used to convert the line to an array of a specific type. When reading from the stored bytes, utility functions advance and reset the current position in the buffer, as well as the current index in the offsets.

* `AK_CPL_CurrentReset`
* `AK_CPL_CurrentAdvance`

When converting bytes to C types, given the current CPL positions, utility functions are implemented to return types. The `AK_CPL_current_to_int64`, `AK_CPL_current_to_uint64`, and `AK_CPL_current_to_float64` functions prepare pointers to call the lower level `AK_UCS4_to_int64`, `AK_UCS4_to_uint64`, and `AK_UCS4_to_float64` functions, respectively.

* `AK_CPL_current_to_bool`
* `AK_CPL_current_to_int64`
* `AK_CPL_current_to_uint64`
* `AK_CPL_current_to_float64`

The outermost interface to convert a CPL to an array is `AK_CPL_ToArray`; this function branches to type-specific private functions based on a dtype, provided either directly (via a `dtypes` function passed to `delimited_to_arrays`) or via a `type_parser` on the CPL. These methods use various techniques to convert CPL field bytes to C values that can then be directly written to a pre-sized array buffer, offering excellent performance.

* `AK_CPL_to_array_bool`
* `AK_CPL_to_array_float`
* `AK_CPL_to_array_int`
* `AK_CPL_to_array_uint`
* `AK_CPL_to_array_unicode`
* `AK_CPL_to_array_bytes`
* `AK_CPL_to_array_via_cast`
* `AK_CPL_ToArray`

Finally, an alternative loader, `AK_CPL_FromIterable`, is made available to create a CPL from an iterable of string objects. Each string is loaded with `AK_CPL_AppendField`. These functions are called to implement the public `iterable_str_to_array_1d` interface.

* `AK_CPL_FromIterable`
* `AK_CPL_AppendField`



### AK_CodePointGrid (CPG)

The CodePointGrid is a dynamic container of `AK_CodePointLine`s. It largely serves as the public interface to CPLs, automatically creating new CPLs when needed.

```C
typedef struct AK_CodePointGrid {
    Py_ssize_t lines_count;
    Py_ssize_t lines_capacity;
    AK_CodePointLine **lines;
    PyObject *dtypes;
} AK_CodePointGrid;
```

Given a line number (which, depending on the `axis` argument, is either the `AK_DelimitedReader` `record_number` or `field_number`), `AK_CPG_AppendPointAtLine` will add a character to the appropriate CPL, allocating a new CPL if it does not exist. At the conclusion of each field, `AK_CPG_AppendOffsetAtLine` is called, it in-turn calling `AK_CPL_AppendOffset` on the appropriate CPL.

* `AK_CPG_New`
* `AK_CPG_Free`
* `AK_CPG_resize`
* `AK_CPG_AppendPointAtLine`
* `AK_CPG_AppendOffsetAtLine`

After loading, `AK_CPG_ToArrayList` can be used to call `AK_CPL_ToArray` for each CPL, accumulating all resulting arrays in a list.

* `AK_CPG_ToArrayList`


### AK_Dialect, AK_DelimitedReaderState, AK_DelimitedReader

These components are all extensions of the original CPython CSV reader. As much as possible, the code was simplified and renamed to match the style and usage in ArrayKit.

The `AK_Dialect` struct is a utility container, bundling parsing configuration settings. An instance is composed within the `AK_DelimitedReader`.

```C
typedef struct AK_Dialect{
    char doublequote;
    char skipinitialspace;
    char strict;
    int quoting;
    Py_UCS4 delimiter;
    Py_UCS4 quotechar;
    Py_UCS4 escapechar;
} AK_Dialect;
```

The `AK_DelimitedReader` `struct` holds the incremental state of the delimited parser, as well as the source iterable of strings.

```C
typedef struct AK_DelimitedReader{
    PyObject *input_iter;
    PyObject *line_select;
    AK_Dialect *dialect;
    AK_DelimitedReaderState state;
    Py_ssize_t field_len;
    Py_ssize_t record_number;
    Py_ssize_t record_iter_number;
    Py_ssize_t field_number;
    int axis;
    Py_ssize_t *axis_pos;
} AK_DelimitedReader;
```

The core functionality of the parser is implemented in `AK_DR_process_char`. This function evaluates state one character at at ime, and calls `AK_DR_add_char` with valid characters and `AK_DR_close_field` at the end of fields. As the `AK_DelimitedReader` does not compose a CPG, a CPG is ultimately passed into each of these functions.

* `AK_DR_New`
* `AK_DR_Free`
* `AK_DR_close_field`
* `AK_DR_add_char`
* `AK_DR_process_char`
* `AK_DR_line_reset`

The `AK_DR_ProcessRecord` is the main entry point. Each time the function is called, a CPG is provided, a record from `input_iter` is retrieved, and characters are processed and loaded in the appropriate CPLs.

* `AK_DR_ProcessRecord`


## The `delimited_to_arrays` Function

Given an iterable of "record" strings and appropriately typed parameters, this function creates an `AK_DelimitedReader` and an `AK_CodePointGrid` (CPG). Then, `AK_DR_ProcessRecord`, given the CPG, is called once for each line in the iterable of strings.

Within each record, one character at time is read and passed to `AK_DR_process_char` along with the CPG. `AK_DR_process_char`, as the core parser of delimited state, is able to call `AK_DR_add_char` for each char within a field and then `AK_DR_close_field` once each field is complete. Those functions, which are also given the CPG, call `AK_CPG_AppendPointAtLine` and `AK_CPG_AppendOffsetAtLine` respectively, growing the CPG, and creating and extending new CPLs as necessary.

After all records are processed, the CPG is fully loaded. `AK_CPG_ToArrayList` can then be called to realize each CPL as an array.

```python
>>> import arraykit as ak
>>> ak.delimited_to_arrays(('a|true|1.2', 'b|false|5.4'), delimiter='|', axis=0)
[array(['a', 'true', '1.2'], dtype='<U4'), array(['b', 'false', '5.4'], dtype='<U5')]
>>>
>>> ak.delimited_to_arrays(('a|true|1.2', 'b|false|5.4'), delimiter='|', axis=1)
[array(['a', 'b'], dtype='<U1'), array([ True, False]), array([1.2, 5.4])]
>>>
>>> ak.delimited_to_arrays(('a|true|1.2', 'b|false|5.4'), delimiter='|', axis=1, line_select=lambda i: i != 1)
[array(['a', 'b'], dtype='<U1'), array([1.2, 5.4])]
```


## The `iterable_str_to_array_1d` Function

Given an iterable of strings, call `AK_IterableStrToArray1D`, which in-turn creates a single CPL with `AK_CPL_FromIterable`. The CPL is then used to create an array with `AK_CPL_ToArray`. As `dtype` can be given `None` to force type evaluation, this interface provides a convenient way to test type evaluation on a single CPL.

```python
>>> import arraykit as ak
>>> ak.iterable_str_to_array_1d(('true', 'False'), None)
array([ True, False])
>>>
>>> ak.iterable_str_to_array_1d(('true', 'False'), str)
array(['true', 'False'], dtype='<U5')
```


## Questions & Future Work

* Better performance is available from creating datetime64 and complex values directly from bytes, I just have not figured out how to do that.
* What initial sizes and growth strategies for CPL, CPG are best? Is it worth collecting a record count hint when available (i.e., when the length of the string iterable is known)?
* Nearly all array loading loops are wrapped in `NPY_BEGIN_THREADS`, `NPY_END_THREADS` macros, as no `PyObject`s are involved; is it possible then to multi-thread CPL array creation?
* Rather than using native "locale" to determine the meaning of decimal and comma, they are brought in as parameters (e.g., `decimalchar` and `thousandschar`). While `decimalchar` will be used in type evaluation, `thousandschar` will not (`int` will only be evaluated if there are no thousands delimeters).
* The current implementation of `line_select`, when used on columns, still loads de-selected lines into CPLs; the selection is only used to skip conversion of CPL data to arrays. This means that if a single column is selected, all columns will still be loaded in CPLs. This is suboptimal but avoids compromising performance in the case where a `line_select` is not used. Alternative approaches, in their simplest form, would call the Python `line_select` function once per character, dramatically degrading performance. With a bit greater complexity, a dynamic array could by built to store results of `line_select`, to be used for subsequent lookups.
