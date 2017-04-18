#include "gcommon/gpu_common.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/nodedata.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>
#include <thrust/system/cuda/execution_policy.h>

namespace voltdb {

extern "C" {
__global__ void MarkNonZeros(ulong *input, int size, ulong *mark)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i <= size; i += blockDim.x * gridDim.x) {
		mark[i] = (i < size && input[i] != 0) ? 1 : 0;
	}
}

void MarkNonZerosWrapper(ulong *input, int size, ulong *output)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	MarkNonZeros<<<grid_size, block_size>>>(input, size, output);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void MarkNonZerosAsyncWrapper(ulong *input, int size, ulong *output, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	MarkNonZeros<<<grid_size, block_size, 0, stream>>>(input, size, output);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void RemoveZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		if (input[i] != 0) {
			output[output_location[i]] = input[i];
			out_bound[output_location[i]] = in_bound[i];
		}
		__syncthreads();
	}
}

void RemoveZerosWrapper(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);


	RemoveZeros<<<grid_size, block_size>>>(input, in_bound, output, out_bound, output_location, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void RemoveZerosAsyncWrapper(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);


	RemoveZeros<<<grid_size, block_size, 0, stream>>>(input, in_bound, output, out_bound, output_location, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void MarkTmpLocation(ulong *tmp_location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location[input[i]] = (i != 0) ? 1 : 0;
	}
}

void MarkTmpLocationWrapper(ulong *tmp_location, ulong *input, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1) / block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	MarkTmpLocation<<<grid_size, block_size>>>(tmp_location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void MarkTmpLocationAsyncWrapper(ulong *tmp_location, ulong *input, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1) / block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	MarkTmpLocation<<<grid_size, block_size, 0, stream>>>(tmp_location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void MarkLocation1(ulong *location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index + 1; i < size - 1; i += blockDim.x * gridDim.x) {
		if (input[i] != input[i + 1])
		location[input[i]] = i;
	}

}

__global__ void MarkLocation2(ulong *location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index + 1; i < size - 1; i += blockDim.x * gridDim.x) {
		if (input[i] != input[i - 1])
			location[input[i]] -= (i - 1);
	}
}

void MarkLocationWrapper(ulong *location, ulong *input, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	MarkLocation1<<<grid_size, block_size>>>(location, input, size);
	MarkLocation2<<<grid_size, block_size>>>(location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void MarkLocationAsyncWrapper(ulong *location, ulong *input, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	MarkLocation1<<<grid_size, block_size, 0, stream>>>(location, input, size);
	MarkLocation2<<<grid_size, block_size, 0, stream>>>(location, input, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void ComputeOffset(ulong *input1, ulong *input2, ulong *out, int size)
{
	ulong index = threadIdx.x + blockIdx.x * blockDim.x;

	for (ulong i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i] = i - input1[input2[i]];
	}
}

void ComputeOffsetWrapper(ulong *input1, ulong *input2, ulong *out, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ComputeOffset<<<grid_size, block_size>>>(input1, input2, out, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void ComputeOffsetAsyncWrapper(ulong *input1, ulong *input2, ulong *out, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ComputeOffset<<<grid_size, block_size, 0, stream>>>(input1, input2, out, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

void ExclusiveScanWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

void ExclusiveScanAsyncWrapper(ulong *input, int ele_num, ulong *sum, cudaStream_t stream)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(thrust::system::cuda::par.on(stream), dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaStreamSynchronize(stream));

	*sum = *(dev_ptr + ele_num - 1);
}

void InclusiveScanWrapper(ulong *input, int ele_num)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::inclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());
}

void InclusiveScanAsyncWrapper(ulong *input, int ele_num, cudaStream_t stream)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::inclusive_scan(thrust::system::cuda::par.on(stream), dev_ptr, dev_ptr + ele_num, dev_ptr);
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

unsigned long timeDiff(struct timeval start, struct timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

__global__ void RemoveEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < in_size) {
		ulong write_loc = out_location[index];
		ulong read_loc = in_location[index];
		ulong num = in_location[index + 1] - in_location[index];
		int lkey, rkey;
		ulong i = 0;

		while (i < num) {
			lkey = in_bound[read_loc + i].lkey;
			rkey = in_bound[read_loc + i].rkey;

			if (lkey != -1 && rkey != -1) {
				out_bound[write_loc].lkey = lkey;
				out_bound[write_loc].rkey = rkey;
				write_loc++;
			}
			i++;
		}
	}
}

void RemoveEmptyResultWrapper(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	RemoveEmptyResult<<<grid_size, block_size>>>(out_bound, in_bound, in_location, out_location, in_size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void RemoveEmptyResultAsyncWrapper(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	RemoveEmptyResult<<<grid_size, block_size, 0, stream>>>(out_bound, in_bound, in_location, out_location, in_size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void RemoveEmptyResult2(RESULT *out, RESULT *in, ulong *location, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tmp_location;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location = location[i];
		if (in[i].lkey != -1 && in[i].rkey != -1) {
			out[tmp_location].lkey = in[i].lkey;
			out[tmp_location].rkey = in[i].rkey;
		}
	}
}

void RemoveEmptyResultWrapper2(RESULT *out, RESULT *in, ulong *location, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	RemoveEmptyResult2<<<grid_size, block_size>>>(out, in, location, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void RemoveEmptyResultAsyncWrapper2(RESULT *out, RESULT *in, ulong *location, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	RemoveEmptyResult2<<<grid_size, block_size, 0, stream>>>(out, in, location, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void ExpressionFilter2(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
									,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
									,int64_t *val_stack, ValueType *type_stack
#endif
								)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GNValue res;

	for (int i = index; i < size; i += offset) {
		res = GNValue::getTrue();
#ifdef TREE_EVAL_
#ifdef FUNC_CALL_
		res = (end_size > 1) ? EvaluateRecvFunc(end_exp, 1, end_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecvFunc(post_exp, 1, post_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols) : res;
#else
		res = (end_size > 1) ? EvaluateRecvNonFunc(end_exp, 1, end_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecvNonFunc(post_exp, 1, post_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols) : res;
#endif
#else
#ifdef FUNC_CALL_
		res = (end_size > 0) ? EvaluateItrFunc(end_exp, end_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, stack + index, offset) : res;
		res = (post_size > 0 && res.isTrue()) ? EvaluateItrFunc(post_exp, post_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, stack + index, offset) : res;
#else
		res = (end_size > 0) ? EvaluateItrNonFunc(end_exp, end_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, val_stack + index, type_stack + index, offset) : res;
		res = (post_size > 0 && res.isTrue()) ? EvaluateItrNonFunc(post_exp, post_size, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
		out_bound[i].lkey = (res.isTrue()) ? (in_bound[i].lkey + outer_base_idx) : (-1);
		out_bound[i].rkey = (res.isTrue()) ? (in_bound[i].rkey + inner_base_idx) : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void ExpressionFilterWrapper2(GNValue *outer_table, GNValue *inner_table,
								RESULT *in_bound, RESULT *out_bound,
								ulong *mark_location, int size,
								uint outer_cols, uint inner_cols,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GTreeNode *where_exp, int where_size,
								int outer_base_idx, int inner_base_idx)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilter2<<<grid_size, block_size>>>(outer_table, inner_table,
													in_bound, out_bound,
													mark_location, size,
													outer_cols, inner_cols,
													end_exp, end_size,
													post_exp, post_size,
													where_exp, where_size,
													outer_base_idx, inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
													, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
													, val_stack, type_stack
#endif
												);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void ExpressionFilterAsyncWrapper2(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx,
									cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilter2<<<grid_size, block_size, 0, stream>>>(outer_table, inner_table,
															in_bound, out_bound,
															mark_location, size,
															outer_cols, inner_cols,
															end_exp, end_size,
															post_exp, post_size,
															where_exp, where_size,
															outer_base_idx, inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
															, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
															, val_stack, type_stack
#endif
															);
	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

__global__ void ExpressionFilter3(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
									, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
									, int64_t *val_stack, ValueType *type_stack
#endif
									)
{
	extern __shared__ GNValue shared_tmp[];
	GNValue *tmp_outer = shared_tmp;
	GNValue *tmp_inner = shared_tmp + blockDim.x * outer_cols;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	GNValue res;
	int outer_idx, inner_idx;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += offset) {
		outer_idx = in_bound[i].lkey;
		inner_idx = in_bound[i].rkey;
		// Load outer tuples to shared memory
		for (int j = 0; j < outer_cols; j++) {
			tmp_outer[threadIdx.x + j] = outer_table[outer_idx * outer_cols + j];
		}
		// Load inner tuples to shared memory
		for (int j = 0; j < inner_cols; j++) {
			tmp_inner[threadIdx.x + j] = inner_table[inner_idx * inner_cols + j];
		}

		__syncthreads();

		res = GNValue::getTrue();
#ifdef TREE_EVAL_
#ifdef FUNC_CALL_
		res = (end_size > 1) ? EvaluateRecvFunc(end_exp, 1, end_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecvFunc(post_exp, 1, post_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols) : res;
#else
		res = (end_size > 1) ? EvaluateRecvNonFunc(end_exp, 1, end_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecvNonFunc(post_exp, 1, post_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols) : res;
#endif
#else
#ifdef FUNC_CALL_
		res = (end_size > 0) ? EvaluateItrFunc(end_exp, end_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, stack + index, offset) : res;
		res = (post_size > 0 && res.isTrue()) ? EvaluateItrFunc(post_exp, post_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, stack + index, offset) : res;
#else
		res = (end_size > 0) ? EvaluateItrNonFunc(end_exp, end_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, val_stack + index, type_stack + index, offset) : res;
		res = (post_size > 0 && res.isTrue()) ? EvaluateItrNonFunc(post_exp, post_size, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
		out_bound[i].lkey = (res.isTrue()) ? (in_bound[i].lkey + outer_base_idx) : (-1);
		out_bound[i].rkey = (res.isTrue()) ? (in_bound[i].rkey + inner_base_idx) : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
		__syncthreads();
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void ExpressionFilterWrapper3(GNValue *outer_table, GNValue *inner_table,
								RESULT *in_bound, RESULT *out_bound,
								ulong *mark_location, int size,
								uint outer_cols, uint inner_cols,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GTreeNode *where_exp, int where_size,
								int outer_base_idx, int inner_base_idx)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = SHARED_SIZE_/(sizeof(GNValue) * (outer_cols + inner_cols));

	//block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilter3<<<grid_size, block_size, block_x * sizeof(GNValue) * (outer_cols + inner_cols)>>>(outer_table, inner_table,
																										in_bound, out_bound,
																										mark_location, size,
																										outer_cols, inner_cols,
																										end_exp, end_size,
																										post_exp, post_size,
																										where_exp, where_size,
																										outer_base_idx, inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																										, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																										, val_stack, type_stack
#endif
																										);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void ExpressionFilterAsyncWrapper3(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx,
									cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = SHARED_SIZE_/(sizeof(GNValue) * (outer_cols + inner_cols));

	//block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilter3<<<grid_size, block_size, block_x * sizeof(GNValue) * (outer_cols + inner_cols), stream>>>(outer_table, inner_table,
																												in_bound, out_bound,
																												mark_location, size,
																												outer_cols, inner_cols,
																												end_exp, end_size,
																												post_exp, post_size,
																												where_exp, where_size,
																												outer_base_idx, inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																												, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																												, val_stack, type_stack
#endif
																												);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

__forceinline__ __device__ void KeyGenerate(GNValue *tuple, int *key_indices, int index_num, uint64_t *packed_key)
{
	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	if (key_indices != NULL) {
		for (int i = 0; i < index_num; i++) {
			//uint64_t key_value = static_cast<uint64_t>(tuple[key_indices[i]].getValue() + INT64_MAX + 1);

			switch (tuple[key_indices[i]].getValueType()) {
				case VALUE_TYPE_TINYINT: {
					uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[key_indices[i]].getValue() + INT8_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					uint64_t key_value = static_cast<uint16_t>((int16_t)tuple[key_indices[i]].getValue() + INT16_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					uint64_t key_value = static_cast<uint32_t>((int32_t)tuple[key_indices[i]].getValue() + INT32_MAX + 1);

					if (key_value == tuple[key_indices[i]].getValue())
						printf("Error at inner\n");

					for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= ((0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8));
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					uint64_t key_value = static_cast<uint64_t>((int64_t)tuple[key_indices[i]].getValue() + INT64_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				default: {
					return;
				}
			}
		}
	} else {
		for (int i = 0; i < index_num; i++) {
			switch (tuple[i].getValueType()) {
				case VALUE_TYPE_TINYINT: {
					uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[i].getValue() + INT8_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					uint64_t key_value = static_cast<uint16_t>((int16_t)tuple[i].getValue() + INT16_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					uint64_t key_value = static_cast<uint32_t>((int32_t)tuple[i].getValue() + INT32_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					uint64_t key_value = static_cast<uint64_t>((int64_t)tuple[i].getValue() + INT64_MAX + 1);

					for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
						packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
						intra_key_offset--;
						if (intra_key_offset < 0) {
							intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
							key_offset++;
						}
					}

					break;
				}
				default: {
					return;
				}
			}

		}
	}
}


__global__ void PackKey(GNValue *table, int rows, int cols, int *indices, int index_num, uint64_t *key, int key_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < rows; i += stride) {
		KeyGenerate(table + i * cols, indices, index_num, key + i * key_size);
	}
}

void PackKeyWrapper(GNValue *table, int rows, int cols, int *indices, int index_num, uint64_t *key, int key_size)
{
	int grid_x, block_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemset(key, 0, sizeof(uint64_t) * rows * key_size));
	checkCudaErrors(cudaDeviceSynchronize());

	PackKey<<<grid_size, block_size>>>(table, rows, cols, indices, index_num, key, key_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void PackKeyAsyncWrapper(GNValue *table, int rows, int cols, int *indices, int index_num, uint64_t *key, int key_size, cudaStream_t stream)
{
	int grid_x, block_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemsetAsync(key, 0, sizeof(uint64_t) * rows * key_size, stream));

	PackKey<<<grid_size, block_size, 0, stream>>>(table, rows, cols, indices, index_num, key, key_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void PackSearchKey(GNValue *table, int rows, int cols, uint64_t *keys, GTreeNode *search_exp, int *search_size, int search_num, int key_size
#if defined(FUNC_CALL_) && defined(POST_EXP_)
								,GNValue *stack
#elif defined(POST_EXP_)
								,int64_t *val_stack,
								ValueType *type_stack
#endif
								)
{
	int index = threadIdx.x + blockIdx.x *blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GNValue tmp_table[4];
	int search_ptr = 0;

	for (int i = index; i < rows; i += stride) {
		search_ptr = 0;
		for (int j = 0; j < search_num; search_ptr += search_size[j], j++) {
#ifdef POST_EXP_
#ifdef FUNC_CALL_
			tmp_table[j] = EvaluateItrFunc(search_exp + search_ptr, search_size[j], table + i * cols, NULL, stack + index, stride);
#else
			tmp_table[j] = EvaluateItrNonFunc(search_exp + search_ptr, search_size[j], table + i * cols, NULL, val_stack + index, type_stack + index, stride);
#endif
#else
#ifdef FUNC_CALL_
			tmp_table[j] = EvaluateRecvFunc(search_exp + search_ptr, 1, search_size[j], table + i * cols, NULL);
#else
			tmp_table[j] = EvaluateRecvNonFunc(search_exp + search_ptr, 1, search_size[j], table + i * cols, NULL);
#endif
#endif
		}

		KeyGenerate(tmp_table, NULL, search_num, keys + i * key_size);
	}
}

void PackSearchKeyWrapper(GNValue *table, int rows, int cols, uint64_t *key, GTreeNode *search_exp, int *search_size, int search_num, int key_size)
{
	int block_x, grid_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemset(key, 0, sizeof(uint64_t) * rows * key_size));
	checkCudaErrors(cudaDeviceSynchronize());

	PackSearchKey<<<grid_size, block_size>>>(table, rows, cols, key, search_exp, search_size, search_num, key_size
#if defined(FUNC_CALL_) && defined(POST_EXP_)
											,stack
#elif defined(POST_EXP_)
											,val_stack,
											type_stack
#endif
											);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void PackSearchKeyAsyncWrapper(GNValue *table, int rows, int cols, uint64_t *key, GTreeNode *search_exp, int *search_size, int search_num, int key_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemsetAsync(key, 0, sizeof(uint64_t) * rows * key_size, stream));

	PackSearchKey<<<grid_size, block_size, 0, stream>>>(table, rows, cols, key, search_exp, search_size, search_num, key_size
#if defined(FUNC_CALL_) && defined(POST_EXP_)
														,stack
#elif defined(POST_EXP_)
														,val_stack,
														type_stack
#endif
														);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void debugGTrees(const GTreeNode *expression, int size)
{
	std::cout << "DEBUGGING INFORMATION..." << std::endl;
	for (int index = 0; index < size; index++) {
		switch (expression[index].type) {
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				std::cout << "[" << index << "] CONJUNCTION AND" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				std::cout << "[" << index << "] CONJUNCTION OR" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				std::cout << "[" << index << "] COMPARE EQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				std::cout << "[" << index << "] COMPARE NOTEQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				std::cout << "[" << index << "] COMPARE LESS THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				std::cout << "[" << index << "] COMPARE GREATER THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE LESS THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE GREATER THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LIKE: {
				std::cout << "[" << index << "] COMPARE LIKE" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_IN: {
				std::cout << "[" << index << "] COMPARE IN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				std::cout << "[" << index << "] TUPLE(";
				std::cout << expression[index].column_idx << "," << expression[index].tuple_idx;
				std::cout << ")" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				std::cout << "[" << index << "] VALUE TUPLE = " << expression[index].value.debug2()  << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_NULL:
			case EXPRESSION_TYPE_INVALID:
			default: {
				std::cout << "NULL value" << std::endl;
				break;
			}
		}
	}
}
}
}
