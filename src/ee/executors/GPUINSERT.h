#ifndef GPUINSERT_H_
#define GPUINSERT_H_

#include <iostream>

#include <cuda.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/expressions/treeexpression.h"

using namespace voltdb;
class GPUINSERT {
public:
	GPUINSERT();
	GPUINSERT(int rows, int cols);
	GPUINSERT(GNValue *input);
	GPUINSERT(GNValue *input, int rows, int cols); 	//for debugging
	void tableCopy(GNValue *input, int numOfCells);
	GNValue *getTableAddress();
	void debug();
private:
	GNValue *table_;
	int rows_, cols_;

	void GNValueDebug(GNValue &column_data);
};

#endif
