#ifndef NODEDATA_H
#define NODEDATA_H

#include "common/types.h"
#include "GPUetc/common/GNValue.h"

namespace voltdb{

//#define TREE_EVAL_ 1
#define POST_EXP_ 1
//#define FUNC_CALL_ 1
#define COALESCE_ 1
//#define PHYSICAL_HASH_ 1
//#define METHOD_1_ 1
//#define SHARED_ 1
#define DECOMPOSED1_ 1
//#define DECOMPOSED2_ 1
//#define ASYNC_ 1

typedef struct {
	ValueType data_type;
} GColumnInfo;

}

#endif
