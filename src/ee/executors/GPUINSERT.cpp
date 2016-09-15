#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "GPUINSERT.h"
#include "scan_common.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>

using namespace voltdb;
GPUINSERT::GPUINSERT()
{
	table_ = NULL;
	rows_ = cols_ = 0;

	checkCudaErrors(cudaMalloc(&table_, MAX_BUFFER_SIZE * sizeof(GNValue)));
}

GPUINSERT::GPUINSERT(int rows, int cols)
{
	table_ = NULL;
	rows_ = rows;
	cols_ = cols;

	checkCudaErrors(cudaMalloc(&table_, MAX_BUFFER_SIZE * sizeof(GNValue)));
}

GPUINSERT::GPUINSERT(GNValue *input)
{
	rows_ = 0;
	cols_ = 0;
	table_ = input;
}

GPUINSERT::GPUINSERT(GNValue *input, int rows, int cols)
{
	table_ = input;
	rows_ = rows;
	cols_ = cols;

	checkCudaErrors(cudaMalloc(&table_, MAX_BUFFER_SIZE * sizeof(GNValue)));
}

GNValue *GPUINSERT::getTableAddress()
{
	return table_;
}

void GPUINSERT::tableCopy(GNValue *input, int numOfCells)
{
	checkCudaErrors(cudaMemcpy(table_, input, numOfCells * sizeof(GNValue), cudaMemcpyHostToDevice));
}

void GPUINSERT::debug()
{
	GNValue *tmp_table = (GNValue *)malloc(sizeof(GNValue) * rows_ * cols_);

	checkCudaErrors(cudaMemcpy(tmp_table, table_, rows_ * cols_ * sizeof(GNValue), cudaMemcpyDeviceToHost));
	for (int i = 0; i < rows_; i++) {
		for (int j = 0; j < cols_; j++) {
			tmp_table[i * cols_ + j].debug();
		}
		std::cout << "End of gpu debug" << std::endl;
	}
}

void GPUINSERT::GNValueDebug(GNValue &column_data)
{
	NValue value;
	long double gtmp = column_data.getMdata();
	char tmp[16];
	memcpy(tmp, &gtmp, sizeof(long double));
	value.setMdataFromGPU(tmp);
	value.setSourceInlinedFromGPU(column_data.getSourceInlined());
	value.setValueTypeFromGPU(column_data.getValueType());

	std::cout << value.debug();
}
