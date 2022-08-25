
# Understanding `delimited_to_arrays`, A Faster-than-Pandas CSV Reader

The `delimited_to_arrays` function takes an iterable of strings (which might be an iterator of lines from a file) and extracts fields from that string based on a delimiter. Fields are accumulated into "lines" of contiguous Unicode byte data; lines can accumulate fields by row (axis 0) or by column (axis 1). As lines accumulate fields, types are optionally evaluated based on a heuristic, content- and count-based evaluation. After a single pass of reading the file, all lines are loaded, and optionally per line, types have been evaluated.

After loading, each line is converted to an array. For each line, an empty array of appropriate size can be created. For most types (bool, ints, float, Unicode, bytes), byte data be direclty translated into C types that can be directly assigned into the array's buffer. For a few types (datetime64, complex), we are forced to create byte arrays and then use a NumPy cast to delivery the desired type.

The implementation limits features to those essential to the C implementation. The `file_like` parameter is an iterable of strings; this removes concern for file opening and encoding configuration (if necessary). The `axis` parameter is provided with an integer. Both the `dtypes` parameter and the `line_select` parameter are Python functions that, when called with an index, return the per-line parameter. For `dytpes`, a dtype initializer can be provided per line to set the type, or `None` to use type evaluation. For `line_select`, a Boolean is returned to select inclusion of the line on the specified axis. (The use of functions for these parameters permits callers to consolidate hetergenous argument types into a single interface.) As the delimited file parser is built on the Python CSV reader, all relevant parameters from that interface are retained here, including full support for diverse quoting and escaping configurations. A few features can more easily be implemented external to the C implementation: optionally skipping leading or trailing file lines can be done by preparing the generator of strings.

Performance of `delimited_to_arrays` on Linux compared to Pandas for loading uniform-typed data (without type evaluation) is around 1.5 times faster than Pandas at scales of 100,000 rows and 500 columns. This ratio appears robust at larger scales; at smaller scales `delimited_to_arrays` approaches 2 times faster. Type evaluation incurs overhead to reduce the outperformance to approximately 1.1 times faster.

An alternative interface,`iterable_str_to_array_1d`, provides a way to load and convert a single line to single array. This is mostly useful for testing.


## CPython C Orientation

C is not an object-oriented language. We can, however, create collections of state (like instance attributes) with `struct` types, and then create collections of functions (like instance methods) that take "instances" of that `struct` and perform in-place or other operations. The convention used here names `struct`s with full names (e.g., `AK_CodePointLine`) while functions that take those `struct`s use abbreviations (e.g., `AK_CPL_New`). Dynamic allocation happens with "New" functions (e.g., `AK_CPL_New`) using `PyMem_Malloc` while memory freeing happens with "Free" functions (e.g. `AK_CPL_Free`) using `PyMem_Free`. Public functions are given in upper camel case (e.g., `AK_CPL_ToArray`) while private functions (called only within the public functions of the same type group) are given in lower snake case (e.g., `AK_CPL_to_array_bool`).

In C we often mutate arguments in place, and a lot of our structs are designed to have mutable state. We do this because in C we often use returned values for error signaling, and thus pass in pointers for values to be set. We also do this for memory efficiency.

In C, we use `goto`. This is a common pattern in CPython where, on error, we often need to tear-down a bunch of memory and can get reuse of those routines through a direct jump.

These routines make extensive use of pointers to arrays of characters (or Unicode code points). Recall that the handle to a C array is a pointer to its head; addition to that pointer "moves" it to point to that many positions forward in the array. Adding one to an array pointer returns a pointer to the next item. Nothing will stop you from advancing beyond the end of the array, so pointer arithmetic must often check that the end of the array has not been exceeded.

In C, pointer arithmetic and pointer dereferencing can happen with concise expressions. There are a couple of common moves.

We often want to dereference a pointer (to get a charcter) and then increment the pointer to be read to read the next character. An example is the following while loop, where we advance through all characters that are spaces.

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

### AK_TypeParserState (AK_TPS)

Enum that defines the interpretation of a field or line of types. Can be unknown (on init) or empty (on encountering empty values).

Can be used to get a dtype instance. Can be used, with another AK_TypeParserState, to "resolve" the dtype. Unlike with normal dtype resolution, the dtype of "no-return" is string.

* `AK_TPS_Resolve`
* `AK_TPS_ToDtype`


### AK_TypeParser (AK_TP)
A struct used to store the state of the interpration of a field, and ultimately a line. One instance is created per line.

For each field, counts of numerous indicators are measured as well as the contiguity of various characteristics. This is done by calling AK_TP_ProcessChar once for each character. By the end of each field, we define a AK_TypeParserState for `parsed_field`, and then "resolve" `parsed_field` with a AK_TypeParserState for the `parsed_line` attribute.

At the end of each field, we reset all attributes except `parsed_field`.

* `AK_TP_New`
* `AK_TP_Free`
* `AK_TP_reset_field`
* `AK_TP_ProcessChar`
* `AK_TP_resolve_field`
* `AK_TP_ResolveLineResetField`

### Py_UCS4 Converter functions:

These function are designed to convert a region of a contiguous byte data (given a pointer to the start and end) to a C type.

These functions are different than common library functions in that they they take start/end pointers, nout a NULL-terminated string. Common library functions (like `atof` or `PyOS_string_to_double`) require a null-terminated string.

* `AK_UCS4_to_int64`
* `AK_UCS4_to_uint64`
* `AK_UCS4_to_float64`


### AK_CodePointLine (CPL)

A representation of a "line", or a group of fields which might be a column (when axis is 1) or a row (when axis is 0).

The core representation consists of two parts, A dynamically growable contiguous array of Py_UCS4 (`buffer`) (with no null terminators), and a dynamically growable array of offsets, providing the number of Py_UCS4 in each field and (implicity, or explicitly with `offset_count`) the number of fields.

The CPL also tracks the `offset_max`, the largest offset observed, while processing each field. This is needed when determining the element size of unicode dtypes, or when creating a re-usable Py_UCS4 buffer (for usage of PyOS_string_to_double which requires a NULL-terminated string, loaded with AK_CPL_current_to_field).

In addition, a CPL on creation optionally composes an AK_TypeParser instance. This permits the CPL to optionally call AK_TP_ProcessChar for each char when accumlating chars.

The CPL tracks the pointer to the current position in the buffer, as well as the current index in the offsets.

In general operation, a CPL has to phases: a loading phase, and conversion phase.

In the loading phase, `AK_CPL_AppendPoint` is called for each point, which in turn calls `AK_TP_ProcessChar` for each char if `type_parser` is active. At the end of each field, `AK_CPL_AppendOffset` is called, which calls `AK_TP_ResolveLineResetField` if `type_parser` is active. This permits loading a CPL in one pass, optional evaluating type at the same time.

* `AK_CPL_New`
* `AK_CPL_Free`
* `AK_CPL_resize`
* `AK_CPL_AppendField`
* `AK_CPL_AppendPoint`
* `AK_CPL_AppendOffset`
* `AK_CPL_FromIterable`


* `AK_CPL_CurrentReset`
* `AK_CPL_CurrentAdvance`
* `AK_CPL_current_to_bool`
* `AK_CPL_current_to_int64`
* `AK_CPL_current_to_float64`


In the conversion phase, a CPL exporter is used to convert the line to an array of a specific type. These methods use various techniques to convert CPL field bytes to C values that can then be directly written to a pre-sized array buffer, offering excellent performance.


* `AK_CPL_to_array_bool`
* `AK_CPL_to_array_float`
* `AK_CPL_to_array_int`
* `AK_CPL_to_array_uint`
* `AK_CPL_to_array_unicode`
* `AK_CPL_to_array_bytes`
* `AK_CPL_to_array_via_cast`
* `AK_CPL_ToArray`


### AK_CodePointGrid (CPG)

The CodePointGrid is dynamic container of CodePointLines. It largely serves as the public interface to CPLs.

```C
typedef struct AK_CodePointGrid {
    Py_ssize_t lines_count;    // accumulated number of lines
    Py_ssize_t lines_capacity; // max number of lines
    AK_CodePointLine **lines;  // array of pointers
    PyObject *dtypes;          // a callable that returns None or a dtype initializer
} AK_CodePointGrid;
```


* `AK_CPG_New`
* `AK_CPG_Free`
* `AK_CPG_resize`
* `AK_CPG_AppendPointAtLine`
* `AK_CPG_AppendOffsetAtLine`
* `AK_CPG_ToArrayList`

### AK_Dialect, AK_DelimitedReaderState, AK_DelimitedReader

These components are all extensions of the original CPython csv reader.

Of these, `AK_DelimitedReader` is the primary interface. The associated struct holds the iterable of strings and maitains state regarding the progress of the parsing.


* `AK_DR_New`
* `AK_DR_Free`
* `AK_DR_close_field`
* `AK_DR_add_char`
* `AK_DR_process_char`
* `AK_DR_line_reset`
* `AK_DR_ProcessLine`

### The `delimited_to_arrays` Entry Point

Given appropriately typed parameters, this function creates an `AK_DelimitedReader` and an `AK_CodePointGrid` (CPG). Then, `AK_DR_ProcessLine`, given the CPG, is called once for each line in the iterable of strings. Within each line, one character at time is read and passed to `AK_DR_process_char` along with the CPG. `AK_DR_process_char` is the core parser of delimited state, and is able to call `AK_DR_add_char` for each char within a field and then `AK_DR_close_field` once each field is complete. Those functions, which are also given the CPG, call `AK_CPG_AppendPointAtLine` and `AK_CPG_AppendOffsetAtLine` respectively, growing the CPG, and creating and extending new CPLs, as necessary.

After all records are processed, the CPG is full loaded. `AK_CPG_ToArrayList` can then be called to realize each CPL as an array.