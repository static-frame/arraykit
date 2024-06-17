# ifndef ARRAYKIT_SRC_BLOCK_INDEX_H_
# define ARRAYKIT_SRC_BLOCK_INDEX_H_

# include "Python.h"

extern PyTypeObject BlockIndexType;
extern PyObject * ErrorInitTypeBlocks;
extern PyTypeObject BIIterType;
extern PyTypeObject BIIterSeqType;
extern PyTypeObject BIIterSliceType;
extern PyTypeObject BIIterBoolType;
extern PyTypeObject BIIterContiguousType;
extern PyTypeObject BIIterBlockType;

# endif /* ARRAYKIT_SRC_BLOCK_INDEX_H_ */
