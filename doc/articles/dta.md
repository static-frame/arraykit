
# Understanding `delimited_to_arrays`, A Faster-than-Pandas CSV Reader

The `delimited_to_arrays` function takes an iterable of strings (which might be an iterator of lines from a file) and extracts fields from that string based on a delimiter. Fields are accumulated into "lines" of contiguous Unicode byte data; lines can accumulate fields by row (axis 0) or by column (axis 1). As lines accumulate fields, types are optionally evaluated based on a heuristic, content- and count-based evaluation. After a single pass of reading the file, all lines are loaded, and optionally per line, types have been evaluated.

After loading, each line is converted to an array. For each line, an empty array of appropriate size can be created. For most types (bool, ints, float, Unicode, bytes), byte data be direclty translated into C types that can be directly assigned into the array's buffer. For a few types (datetime64, complex), we are forced to create byte arrays and then use a NumPy cast to delivery the desired type.

The implementation limits features to those essential to the C implementation. The `file_like` parameter is an iterable of strings; this removes concern for file opening and encoding configuration (if necessary). The `axis` parameter is provided with an integer. Both the `dtypes` parameter and the `line_select` parameter are Python functions that, when called with an index, return the per-line parameter. For `dytpes`, a dtype initializer can be provided per line to set the type, or `None` to use type evaluation. For `line_select`, a Boolean is returned to select inclusion of the line on the specified axis. (The use of functions for these parameters permits callers to consolidate hetergenous argument types into a single interface.) As the delimited file parser is built on the Python CSV reader, all relevant parameters from that interface are retained here, including full support for diverse quoting and escaping configurations. A few features can more easily be implemented external to the C implementation: optionally skipping leading or trailing file lines can be done by preparing the generator of strings.

Performance of `delimited_to_arrays` on linux compared to Pandas for loading uniform-typed data (without type evaluation) is around 1.5 times faster than Pandas at scales of 100,000 rows and 500 columns. This ratio appears robust at larger scales; at smaller scales `delimited_to_arrays` approaches 2 times faster. Type evaluation incurs overhead to reduce the outperformance to approximately 1.25 times faster.

An alternative interface,`iterable_str_to_array_1d`, provides a way to load and convert a single line to single array. This is mostly useful for testing.


## CPython C Orientation

C is not an object-oriented language. We can, however, create collections of state (like instance attributes) with `struct` types, and then create collections of functions (like instance methods) that take "instances" of that `struct` and perform in-place or other operations. The convention used here names `struct`s with full names (e.g., `AK_CodePointLine`) while functions that take those `struct`s use abbreviations (e.g., `AK_CPL_New`). Dynamic allocation happens with "New" functions (e.g., `AK_CPL_New`) using `PyMem_Malloc` while memory freeing happens with "Free" functions (e.g. `AK_CPL_Free`) using `PyMem_Free`.

In C we often mutate arguments in place, and a lot of our structs are designed to have mutable state. We do this because in C we often use returned values for error signaling, and thus pass in pointers for values to be set. We also do this for memory efficiency.

In C, we use `goto`. This is a common pattern in CPython where, on error, we often need to tear-down a bunch of memory and can get reuse of those routines through a direct jump.

These routines make extensive use of pointers to arrays of characters (or Unicode code points). Recall that the handle to a C array is a pointer to its head; addition to that pointer "moves" it to point to that many positions forward in the array. Adding one to an array pointer returns a pointer to the next item. Nothing will stop you from advancing beyond the end of the array, so pointer arithmetic must often check that the end of the array has not been exceeded.

In C, pointer arithmetic and pointer dereferencing can happen with concise expressions. There are a couple of common moves.

We often want to dereference a pointer (to get a charcter) and then increment the pointer to be read to read the next character:

```C
char *p;       // pointer to the sart of an array of char
char c = *p++; // de-reference p to assign to c, then increment p to the next point
```

A variation of this move is done in assignment, where we assign to a dereferenced pointer and then increment it to be ready to assign to the next character.

```C
*array_buffer++ = (npy_int32)AK_CPL_current_to_int64(cpl, &error);
```

We can do the same on the right hand side of the assignment, such that we dereference a source and assign it to a dereferenced destination, then increment the source pointer and increment the destination pointer.

```C
*array_buffer++ = (char)*p++; // truncate
```


## Main Components

### AK_TypeParserState (AK_TPS)

Enum that defines the interpretation of a field or line of types. Can be unknown (on init) or empty (on encountering empty values).

Can be used to get a dtype instance. Can be used, with another AK_TypeParserState, to "resolve" the dtype. Unlike with normal dtype resolution, the dtype of "no-return" is string.

### AK_TypeParser (AK_TP)
A struct used to store the state of the interpration of a field, and ultimately a line. One instance is created per line.

For each field, counts of numerous indicators are measured as well as the contiguity of various characteristics. This is done by calling AK_TP_ProcessChar once for each character. By the end of each field, we define a AK_TypeParserState for `parsed_field`, and then "resolve" `parsed_field` with a AK_TypeParserState for the `parsed_line` attribute.

At the end of each field, we reset all attributes except `parsed_field`.


### Py_UCS4 Converter functions:

Given a pointer to an array of Py_UCS4, return a C-type. Do this for signed/unsigned. Would like to do this for floats but hard to find an isolated implementation.

These functions are unique in that they are designed to work with start/end pointer; many similar functions (like `atof` or `PyOS_string_to_double`) require a null-terminated string.


### AK_CodePointLine (CPL)

A representation of a "line", or a group of fields which might be a column (when axis is 1) or a row (when axis is 0).

The core representation consists of two parts, A dynamically growable contiguous array of Py_UCS4 (`buffer`) (with no null terminators), and a dynamically growable array of offsets, providing the number of Py_UCS4 in each field and (implicity, or explicitly with `offset_count`) the number of fields.

The CPL also tracks the `offset_max`, the largest offset observed, while processing each field. This is needed when determining the element size of unicode dtypes, or when creating a re-usable Py_UCS4 buffer (for usage of PyOS_string_to_double which requires a NULL-terminated string, loaded with AK_CPL_current_to_field).

In addition, a CPL on creation optionally composes an AK_TypeParser instance. This permits the CPL to optionally call AK_TP_ProcessChar for each char when accumlating chars.

The CPL tracks the pointer to the current position in the buffer, as well as the current index in the offsets.

In general operation, a CPL has to phases: a loading phase, and conversion phase.

In the loading phase, `AK_CPL_AppendPoint` is called for each point, which in turn calls `AK_TP_ProcessChar` for each char if `type_parser` is active. At the end of each field, `AK_CPL_AppendOffset` is called, which calls `AK_TP_ResolveLineResetField` if `type_parser` is active. This permits loading a CPL in one pass, optional evaluating type at the same time.

In the conversion phase, a CPL exporter is used to convert the line to an array of a specific type (i.e., `AK_CPL_ToArrayBoolean`, `AK_CPL_ToArrayFloat`, `AK_CPL_ToArrayInt`, `AK_CPL_ToArrayUInt`, `AK_CPL_ToArrayUnicode`, `AK_CPL_ToArrayBytes`, `AK_CPL_ToArrayViaCast`). These methods use various techniques to convert CPL field bytes to C values that can then be directly written to a pre-sized array buffer, offering excellent performance.

