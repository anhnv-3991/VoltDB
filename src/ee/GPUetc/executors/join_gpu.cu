#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/nodedata.h"
#include "gcommon/gpu_common.h"
#include "join_gpu.h"


/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/


namespace voltdb {
extern "C" {
__global__ void prefixSumFilter(GTable outer, GTable inner, ulong *count_psum,
									GTree pre_join_pred, GTree join_pred, GTree where_pred,
									int64_t *stack, ValueType *gtype)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	int outer_rows = outer.block_list->rows, outer_cols = outer.block_list->columns;
	int inner_rows = inner.block_list->rows, inner_cols = inner.block_list->columns;
	int64_t *outer_table = outer.block_list->gdata, *inner_table = inner.block_list->gdata;
	GNValue res = GNValue::getTrue();

	count_psum[index] = 0;

	int matched = 0;

	for (int outer_idx = index; outer_idx < outer_rows; outer_idx += offset) {
		for (int inner_idx = 0; inner_idx < inner_rows; inner_idx++) {
			res = (pre_join_pred.size > 0) ? EvaluateItrNonFunc(pre_join_pred,
																	outer_table + outer_idx * outer_cols,
																	inner_table + inner_idx * inner_cols,
																	outer.schema, inner.schema,
																	stack, gtype, offset) : res;
			res = (join_pred.size > 0 && res.isTrue()) ? EvaluateItrNonFunc(join_pred,
																				outer_table + outer_idx * outer_cols,
																				inner_table + inner_idx * inner_cols,
																				outer.schema, inner.schema,
																				stack, gtype, offset) : res;
			res = (where_pred.size > 0 && res.isTrue()) ? EvaluateItrNonFunc(where_pred,
																				outer_table + outer_idx * outer_cols,
																				inner_table + inner_idx * inner_cols,
																				outer.schema, inner.schema,
																				stack, gtype, offset) : res;
			matched += (res.isTrue()) ? 1 : 0;
		}
	}

	count_psum[index] = matched;

	if (index == 0)
		count_psum[outer_rows] = 0;
}

__global__ void join(GTable outer, GTable inner, RESULT *jresult_dev, ulong *count_psum,
						GTree pre_join_pred, GTree join_pred, GTree where_pred,
						int64_t *stack, ValueType *gtype)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	int outer_rows = outer.block_list->rows, outer_cols = outer.block_list->columns;
	int inner_rows = inner.block_list->rows, inner_cols = inner.block_list->columns;
	int64_t *outer_table = outer.block_list->gdata, *inner_table = inner.block_list->gdata;

	GNValue res = GNValue::getTrue();
	ulong write_location = count_psum[index];

	for (int outer_idx = index; outer_idx < outer_rows; outer_idx += offset) {
		for (int inner_idx = 0; inner_idx < inner_rows; inner_idx++) {
			res = (pre_join_pred.size > 0) ? EvaluateItrNonFunc(pre_join_pred,
															outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
															outer.schema, inner.schema,
															stack, gtype, offset) : res;
			res = (join_pred.size > 0 && res.isTrue()) ? EvaluateItrNonFunc(join_pred,
																		outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																		outer.schema, inner.schema,
																		stack, gtype, offset) : res;
			res = (where_pred.size > 0 && res.isTrue()) ? EvaluateItrNonFunc(where_pred,
																			outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																			outer.schema, inner.schema,
																			stack, gtype, offset) : res;

			jresult_dev[write_location].lkey = (res.isTrue()) ? outer_idx : - 1;
			jresult_dev[write_location].rkey = (res.isTrue()) ? inner_idx : -1;
			write_location++;
		}
	}
}

void prefixSumFilterWrapper(GTable outer, GTable inner, ulong *count_psum,
								GTree pre_join_pred, GTree join_pred, GTree where_pred)
{
	int block_x, grid_x;

	block_x = (outer.block_list->rows < BLOCK_SIZE_X) ? outer.block_list->rows : BLOCK_SIZE_X;
	grid_x = (outer.block_list->rows - 1)/block_x + 1;

	dim3 block(block_x, 1, 1);
	dim3 grid(grid_x, 1, 1);

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	prefixSumFilter<<<grid, block>>>(outer, inner, count_psum,
										pre_join_pred, join_pred, where_pred,
										val_stack, type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}


void expFilterWrapper(GTable outer, GTable inner,
						RESULT *jresult_dev, ulong *count_psum,
						GTree pre_join_pred, GTree join_pred, GTree where_pred)
{
	int block_x, grid_x;

	block_x = (outer.block_list->rows < BLOCK_SIZE_X) ? outer.block_list->rows : BLOCK_SIZE_X;
	grid_x = (outer.block_list->rows - 1)/block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	dim3 block(block_x, 1, 1);
	dim3 grid(grid_x, 1, 1);

	join<<<grid, block>>>(outer, inner, jresult_dev, count_psum,
							pre_join_pred, join_pred, where_pred, val_stack, type_stack);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

void prefixSumWrapper(ulong *input, uint ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);

	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

}
}

