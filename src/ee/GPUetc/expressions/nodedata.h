
#ifndef NODEDATA_H
#define NODEDATA_H

#include "common/types.h"

namespace voltdb{

typedef struct _EXPRESSIONNODE{
    ExpressionType et;
    int startPos;
    int endPos;
    
}EXPRESSIONNODE;

}

#endif
