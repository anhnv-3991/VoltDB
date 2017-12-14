#include <cuda.h>
#include "GPUetc/common/GNValue.h"
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"


namespace voltdb {
extern "C" {
void blockAllocator(GBlock *block)
{
	//checkCudaErrors(cudaMalloc(block->gdata, sizeof(GNValue) * MAX_BLOCK_SIZE));
	cudaMalloc(&(block->gdata), sizeof(GNValue) * MAX_BLOCK_SIZE);
	block->block_size = MAX_BLOCK_SIZE;
	block->rows = 0;
}

void blockRemover(GBlock *block)
{
	//checkCudaErrors(cudaFree(block->gdata));
	cudaFree(block->gdata);
}

void gtupleCopy(char *source, char *destination, int size)
{
	//checkCudaErrors(cudaMemcpy(destination, source, size * sizeof(GNValue)));
	cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice);
}
}
}
