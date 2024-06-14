# ifndef ARRAYKIT_SRC_BLOCK_INDEX_H_
# define ARRAYKIT_SRC_BLOCK_INDEX_H_

# include "Python.h"

PyTypeObject BlockIndexType;
PyObject * ErrorInitTypeBlocks;
PyTypeObject BIIterType;
PyTypeObject BIIterSeqType;
PyTypeObject BIIterSliceType;
PyTypeObject BIIterBoolType;
PyTypeObject BIIterContiguousType;
PyTypeObject BIIterBlockType;

# endif /* ARRAYKIT_SRC_BLOCK_INDEX_H_ */
