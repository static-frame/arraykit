
# Understanding `delimited_to_arrays`, A Faster-than-Pandas CSV Reader

## Orientation

C is not an object-oriented language. We can, however, create collections of state (like instance attributes) with `struct` types, and then create collections of functions (like instance methods) that take "instances" of that `struct` and perform in-place or other operations. Here, structs have full names (i.e., `AK_CodePointLine`) while methods use appreviations (i.e., `AK_CPL_New`).

We are going to mutate a lot of arguments in place, and a lot of our structs have mutable state. We do this because in C we often use returned values for error signaling, and thus pass in pointers for values to be set. We also do this for efficiency (to avoid memory allocation).

We are going to do a lot with pointers to character (or code point) sequences. A few common moves are listed below:

Some common moves:
```C
Py_UCS4 *p;    // pointer to the sart of an array of Py_UCS4
char c = *p++; // de-reference p to get value, then increment to the next point

// which is the same as
char c = *p;
p++
```

We will use `goto`! This is a common pattern in CPython where, on error, we often need to tear-down a bunch of memory and can get reuse of those routines through a direct jump.


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

