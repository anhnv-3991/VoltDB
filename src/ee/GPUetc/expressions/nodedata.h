
#ifndef NODEDATA_H
#define NODEDATA_H

#include "common/types.h"

namespace voltdb{

typedef struct _EXPRESSIONNODE{
    ExpressionType et;
    int startPos;
    int endPos;
    
}EXPRESSIONNODE;

typedef struct _TreeNode {
	ExpressionType type;	//type of
	int column_idx;		//Index of column in tuple, -1 if not tuple value
	int tuple_idx;			//0: left, outer, 1: right, inner
	GNValue value;		// Value of const, = NULL if not const
} GTreeNode;


//#define TREE_EVAL_ 1
#define POST_EXP_ 1
#define FUNC_CALL_ 1

}

#endif
