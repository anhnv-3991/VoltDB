#include "index_join_gpu.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/nodedata.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#include <thrust/execution_policy.h>
#include "gcommon/gpu_common.h"

namespace voltdb {

extern "C" {
__forceinline__ __device__ int LowerBound(GTreeNode * search_exp, int *search_exp_size, int search_exp_num,
											int * key_indices, int key_index_size,
											GNValue *outer_table, GNValue *inner_table,
											int outer_cols, int inner_cols,
											int left, int right,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
											GNValue *stack,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
											int64_t *val_stack,
											ValueType *type_stack,
#endif
											int offset)
{
	int middle = -1;
	int search_ptr;
	int result = -1;
	int inner_idx;

#ifndef FUNC_CALL_
	int key_idx;
	int64_t outer_tmp[8], inner_tmp;
	ValueType outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	GNValue tmp;

	res_i = -1;
	res_d = -1;
#else
	GNValue outer_tmp[8];
	int res;
#endif

	search_ptr = 0;
	for (int i = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
#ifndef FUNC_CALL_
#ifdef POST_EXP_
		tmp = EvaluateItrNonFunc(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = EvaluateRecvNonFunc(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
		outer_tmp[i] = EvaluateItrFunc(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = EvaluateRecvFunc(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
			key_idx = key_indices[i];

			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
#else
		res = 0;
		for (int i = 0; (res == 0) && (i < search_exp_num); i++) {
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);
		}
#endif


#ifndef FUNC_CALL_
		right = (res_i <= 0 && res_d <= 0) ? (middle - 1) : right;	//move to left
		left = (res_i > 0 || res_d > 0) ? (middle + 1) : left;		//move to right
		result = (res_i <= 0 && res_d <= 0) ? middle : result;
#else
		right = (res == VALUE_COMPARE_GREATERTHAN) ? right : (middle - 1);
		left = (res == VALUE_COMPARE_GREATERTHAN) ? (middle + 1) : left;
		result = (res == VALUE_COMPARE_GREATERTHAN) ? result : middle;
#endif
	}
	return result;
}

__forceinline__ __device__ int UpperBound(GTreeNode * search_exp, int *search_exp_size, int search_exp_num,
											int * key_indices, int key_index_size,
											GNValue *outer_table, GNValue *inner_table,
											int outer_cols, int inner_cols,
											int left, int right,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
											GNValue *stack,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
											int64_t *val_stack,
											ValueType *type_stack,
#endif
											int offset)
{
	int middle = -1;
	int search_ptr;
	int result = right;
	int inner_idx;

#ifndef FUNC_CALL_
	int key_idx;
	int64_t outer_tmp[8], inner_tmp;
	ValueType outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	GNValue tmp;

	res_i = -1;
	res_d = -1;
#else
	GNValue outer_tmp[8];
	int res;
#endif
	search_ptr = 0;
	for (int i = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
#ifndef FUNC_CALL_
#ifdef POST_EXP_
		tmp = EvaluateItrNonFunc(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = EvaluateRecvNonFunc(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
		outer_tmp[i] = EvaluateItrFunc(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = EvaluateRecvFunc(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {

			key_idx = key_indices[i];
			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
#else
		res = 0;
		for (int i = 0; res == 0 && i < search_exp_num; i++) {
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);
		}
#endif


#ifndef FUNC_CALL_
		right = (res_i < 0 || res_d < 0) ? (middle - 1) : right;	//move to left
		left = (res_i >= 0 && res_d >= 0) ? (middle + 1) : left;		//move to right
		result = (res_i < 0 || res_d < 0) ? middle : result;
#else
		right = (res == VALUE_COMPARE_LESSTHAN) ? (middle - 1) : right;
		left = (res == VALUE_COMPARE_LESSTHAN) ? left : (middle + 1);
		result = (res == VALUE_COMPARE_LESSTHAN) ? middle : result;
#endif
	}

	return result;
}


__global__ void PrejoinFilter(GNValue *outer_dev,
								uint outer_rows,
								uint outer_cols,
								GTreeNode *prejoin_dev,
								uint prejoin_size,
								bool *result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								,int64_t *val_stack,
								ValueType *type_stack
#endif
								)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i+= offset) {
		GNValue res = GNValue::getTrue();

#ifdef 	TREE_EVAL_
#ifdef FUNC_CALL_
		res = (prejoin_size > 1) ? EvaluateRecvFunc(prejoin_dev, 1, prejoin_size, outer_dev + i * outer_cols, NULL) : res;
#else
		res = (prejoin_size > 1) ? EvaluateRecvNonFunc(prejoin_dev, 1, prejoin_size, outer_dev + i * outer_cols, NULL) : res;
#endif
#elif	POST_EXP_
#ifndef FUNC_CALL_
		res = (prejoin_size > 1) ? EvaluateItrNonFunc(prejoin_dev, prejoin_size, outer_dev + i * outer_cols, NULL, val_stack + index, type_stack + index, offset) : res;
#else
		res = (prejoin_size > 1) ? EvaluateItrFunc(prejoin_dev, prejoin_size, outer_dev + i * outer_cols, NULL, stack + index, offset) : res;
#endif
#endif
		result[i] = res.isTrue();
	}
}


__global__ void IndexFilterLowerBound(GNValue *outer_dev, GNValue *inner_dev,
										  ulong *index_psum, ResBound *res_bound,
										  uint outer_part_size, uint outer_cols,
										  uint inner_part_size, uint inner_cols,
										  GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
										  int *key_indices, int key_index_size,
										  IndexLookupType lookup_type,
										  bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
										  ,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
										  ,int64_t *val_stack,
										  ValueType *type_stack
#endif
										  )

{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_part_size; i += offset) {
		res_bound[i].left = -1;
		res_bound[i].outer = -1;

		if (prejoin_res_dev[i]) {
			res_bound[i].outer = i;

			switch (lookup_type) {
			case INDEX_LOOKUP_TYPE_EQ:
			case INDEX_LOOKUP_TYPE_GT:
			case INDEX_LOOKUP_TYPE_GTE:
			case INDEX_LOOKUP_TYPE_LT: {
				res_bound[i].left = LowerBound(search_exp_dev, search_exp_size, search_exp_num,
												key_indices, key_index_size,
												outer_dev + i * outer_cols, inner_dev,
												outer_cols, inner_cols,
												0, inner_part_size - 1,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												stack + index,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL))
												val_stack + index,
												type_stack + index,
#endif
												offset);
				break;
			}
			case INDEX_LOOKUP_TYPE_LTE: {
				res_bound[i].left = 0;
				break;
			}
			default:
				break;
			}
		}
	}
}

__global__ void IndexFilterUpperBound(GNValue *outer_dev, GNValue *inner_dev,
										  ulong *index_psum, ResBound *res_bound,
										  uint outer_part_size, uint outer_cols,
										  uint inner_part_size, uint inner_cols,
										  GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
										  int *key_indices, int key_index_size,
										  IndexLookupType lookup_type,
										  bool *prejoin_res_dev
#if (defined(POST_EXP_) && !defined(FUNC_CALL_))
										  ,int64_t *val_stack,
										  ValueType *type_stack
#elif (defined(POST_EXP_) && defined(FUNC_CALL_))
										  ,GNValue *stack
#endif
										  )

{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_part_size; i += offset) {
		index_psum[i] = 0;
		res_bound[i].right = -1;

		if (prejoin_res_dev[i]) {
			switch (lookup_type) {
			case INDEX_LOOKUP_TYPE_EQ:
			case INDEX_LOOKUP_TYPE_LTE: {
				res_bound[i].right = UpperBound(search_exp_dev, search_exp_size, search_exp_num,
												key_indices, key_index_size,
												outer_dev + i * outer_cols, inner_dev,
												outer_cols, inner_cols,
												0, inner_part_size - 1,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												stack + index,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												val_stack + index,
												type_stack + index,
#endif
												offset);
				break;
			}
			case INDEX_LOOKUP_TYPE_GT:
			case INDEX_LOOKUP_TYPE_GTE: {
				res_bound[i].right = inner_part_size - 1;
				break;
			}
			case INDEX_LOOKUP_TYPE_LT: {
				res_bound[i].right = res_bound[i].left - 1;
				res_bound[i].left = 0;
				break;
			}
			default:
				break;
			}
		}

		index_psum[i] = (res_bound[i].right >= 0 && res_bound[i].left >= 0) ? (res_bound[i].right - res_bound[i].left + 1) : 0;
	}

	if (index == 0)
		index_psum[outer_part_size] = 0;
}

__global__ void ExpressionFilter(GNValue *outer_dev, GNValue *inner_dev,
									RESULT *result_dev, ulong *index_psum,
									ulong *exp_psum, uint outer_part_size,
									uint outer_cols, uint inner_cols,
									uint jr_size,
									GTreeNode *end_dev, int end_size,
									GTreeNode *post_dev, int post_size,
									GTreeNode *where_dev, int where_size,
									ResBound *res_bound,
									int outer_base_idx, int inner_base_idx,
									bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
									,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
									,int64_t *val_stack,
									ValueType *type_stack
#endif
								)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_part_size; i += offset) {
		exp_psum[i] = 0;
		ulong writeloc = index_psum[index];
		int count = 0;
		int res_left = -1, res_right = -1;
		GNValue res = GNValue::getTrue();

		res_left = res_bound[i].left;
		res_right = res_bound[i].right;

		while (res_left >= 0 && res_left <= res_right && writeloc < jr_size) {
#ifdef	TREE_EVAL_
#ifdef FUNC_CALL_
			res = (end_size > 1) ? EvaluateRecvFunc(end_dev, 1, end_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? EvaluateRecvFunc(post_dev, 1, post_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols) : res;
#else
			res = (end_size > 1) ? EvaluateRecvNonFunc(end_dev, 1, end_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? EvaluateRecvNonFunc(post_dev, 1, post_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols) : res;
#endif

#elif	POST_EXP_


#ifdef 	FUNC_CALL_
			res = (end_size > 0) ? EvaluateItrFunc(end_dev, end_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, stack + index, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? EvaluateItrFunc(post_dev, post_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, stack + index, offset) : res;
#else
			res = (end_size > 0) ? EvaluateItrNonFunc(end_dev, end_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, val_stack + index, type_stack + index, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? EvaluateItrNonFunc(post_dev, post_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
			result_dev[writeloc].lkey = (res.isTrue()) ? (i + outer_base_idx) : (-1);
			result_dev[writeloc].rkey = (res.isTrue()) ? (res_left + inner_base_idx) : (-1);
			count += (res.isTrue()) ? 1 : 0;
			writeloc++;
			res_left++;
		}
		exp_psum[i] = count;
	}

	if (index == 0) {
		exp_psum[outer_part_size] = 0;
	}
}

__global__ void Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i].lkey = in[in_location[i]].outer;
		out[i].rkey = in[in_location[i]].left + local_offset[i];
	}
}

void DecomposeWrapper(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	Decompose<<<grid_size, block_x>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void DecomposeAsyncWrapper(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	Decompose<<<grid_size, block_x, 0, stream>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

void PrejoinFilterWrapper(GNValue *outer_table, uint outer_rows, uint outer_cols, GTreeNode *prejoin_exp, uint prejoin_size, bool *result)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

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

	PrejoinFilter<<<grid_size, block_size>>>(outer_table, outer_rows, outer_cols, prejoin_exp, prejoin_size, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,val_stack,
												type_stack
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

void PrejoinFilterAsyncWrapper(GNValue *outer_table, uint outer_rows, uint outer_cols, GTreeNode *prejoin_exp, uint prejoin_size, bool *result, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

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

	PrejoinFilter<<<grid_size, block_size, 0, stream>>>(outer_table, outer_rows, outer_cols, prejoin_exp, prejoin_size, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,val_stack,
												type_stack
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

void IndexFilterWrapper(GNValue *outer_dev, GNValue *inner_dev,
							ulong *index_psum, ResBound *res_bound,
							uint outer_rows, uint outer_cols,
							uint inner_rows, uint inner_cols,
							GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
							int *key_indices, int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

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

	IndexFilterLowerBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_rows, outer_cols,
														inner_rows, inner_cols,
														search_exp_dev, search_exp_size,
														search_exp_num, key_indices,
														key_index_size, lookup_type,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
#endif
													);

	checkCudaErrors(cudaGetLastError());

	IndexFilterUpperBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_rows, outer_cols,
														inner_rows, inner_cols,
														search_exp_dev, search_exp_size,
														search_exp_num, key_indices,
														key_index_size, lookup_type,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
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

void IndexFilterAsyncWrapper(GNValue *outer_dev, GNValue *inner_dev,
							ulong *index_psum, ResBound *res_bound,
							uint outer_rows, uint outer_cols,
							uint inner_rows, uint inner_cols,
							GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
							int *key_indices, int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev,
							cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

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

	IndexFilterLowerBound<<<grid_size, block_size, 0, stream>>>(outer_dev, inner_dev,
																index_psum, res_bound,
																outer_rows, outer_cols,
																inner_rows, inner_cols,
																search_exp_dev, search_exp_size,
																search_exp_num, key_indices,
																key_index_size, lookup_type,
																prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																,val_stack,
																type_stack
#endif
																);

	checkCudaErrors(cudaGetLastError());

	IndexFilterUpperBound<<<grid_size, block_size, 0, stream>>>(outer_dev, inner_dev,
																index_psum, res_bound,
																outer_rows, outer_cols,
																inner_rows, inner_cols,
																search_exp_dev, search_exp_size,
																search_exp_num, key_indices,
																key_index_size, lookup_type,
																prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																,val_stack,
																type_stack
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

void ExpressionFilterWrapper(GNValue *outer_dev, GNValue *inner_dev,
								RESULT *result_dev,
								ulong *index_psum, ulong *exp_psum,
								uint outer_rows,
								uint outer_cols, uint inner_cols,
								uint jr_size,
								GTreeNode *end_dev, int end_size,
								GTreeNode *post_dev, int post_size,
								GTreeNode *where_dev, int where_size,
								ResBound *res_bound,
								int outer_base_idx, int inner_base_idx,
								bool *prejoin_res_dev)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

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

	ExpressionFilter<<<grid_size, block_size>>>(outer_dev, inner_dev,
												result_dev, index_psum,
												exp_psum, outer_rows,
												outer_cols, inner_cols,
												jr_size, end_dev,
												end_size, post_dev,
												post_size, where_dev,
												where_size, res_bound,
												outer_base_idx, inner_base_idx,
												prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												, val_stack
												, type_stack
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

void ExpressionFilterAsyncWrapper(GNValue *outer_dev, GNValue *inner_dev,
									RESULT *result_dev,
									ulong *index_psum, ulong *exp_psum,
									uint outer_rows,
									uint outer_cols, uint inner_cols,
									uint jr_size,
									GTreeNode *end_dev, int end_size,
									GTreeNode *post_dev, int post_size,
									GTreeNode *where_dev, int where_size,
									ResBound *res_bound,
									int outer_base_idx, int inner_base_idx,
									bool *prejoin_res_dev,
									cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

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

	ExpressionFilter<<<grid_size, block_size, 0, stream>>>(outer_dev, inner_dev,
															result_dev, index_psum,
															exp_psum, outer_rows,
															outer_cols, inner_cols,
															jr_size, end_dev,
															end_size, post_dev,
															post_size, where_dev,
															where_size, res_bound,
															outer_base_idx, inner_base_idx,
															prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
															, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
															, val_stack
															, type_stack
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

void RebalanceAsync3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	ExclusiveScanAsyncWrapper(in, in_size, out_size, stream);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemsetAsync(location, 0, sizeof(ulong) * (*out_size), stream));

	//checkCudaErrors(cudaStreamSynchronize(stream));

	MarkLocationAsyncWrapper(location, in, in_size, stream);

	InclusiveScanAsyncWrapper(location, *out_size, stream);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	ComputeOffsetAsyncWrapper(in, location, local_offset, *out_size, stream);

	DecomposeAsyncWrapper(in_bound, *out_bound, location, local_offset, *out_size, stream);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	ExclusiveScanWrapper(in, in_size, out_size);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemset(location, 0, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaDeviceSynchronize());

	MarkLocationWrapper(location, in, in_size);

	InclusiveScanWrapper(location, *out_size);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	ComputeOffsetWrapper(in, location, local_offset, *out_size);

	DecomposeWrapper(in_bound, *out_bound, location, local_offset, *out_size);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *mark;
	ulong size_no_zeros;
	ResBound *tmp_bound;
	ulong sum;

	/* Remove zeros elements */
	ulong *no_zeros;

	checkCudaErrors(cudaMalloc(&mark, (in_size + 1) * sizeof(ulong)));

	MarkNonZerosWrapper(index_count, in_size, mark);

	ExclusiveScanWrapper(mark, in_size + 1, &size_no_zeros);

	if (size_no_zeros == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));

		return;
	}

	checkCudaErrors(cudaMalloc(&no_zeros, (size_no_zeros + 1) * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&tmp_bound, size_no_zeros * sizeof(ResBound)));

	RemoveZerosWrapper(index_count, in_bound, no_zeros, tmp_bound, mark, in_size);

	ExclusiveScanWrapper(no_zeros, size_no_zeros + 1, &sum);

	if (sum == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));
		checkCudaErrors(cudaFree(no_zeros));
		checkCudaErrors(cudaFree(tmp_bound));

		return;
	}

	ulong *tmp_location, *local_offset;

	checkCudaErrors(cudaMalloc(&tmp_location, sum * sizeof(ulong)));

	checkCudaErrors(cudaMemset(tmp_location, 0, sizeof(ulong) * sum));
	checkCudaErrors(cudaDeviceSynchronize());

	MarkTmpLocationWrapper(tmp_location, no_zeros, size_no_zeros);

	InclusiveScanWrapper(tmp_location, sum);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	ComputeOffsetWrapper(no_zeros, tmp_location, local_offset, sum);

	DecomposeWrapper(tmp_bound, *out_bound, tmp_location, local_offset, sum);
	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));

}



void RebalanceAsync(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *mark;
	ulong size_no_zeros;
	ResBound *tmp_bound;
	ulong sum;

	/* Remove zeros elements */
	ulong *no_zeros;

	checkCudaErrors(cudaMalloc(&mark, (in_size + 1) * sizeof(ulong)));

	MarkNonZerosAsyncWrapper(index_count, in_size, mark, stream);

	ExclusiveScanAsyncWrapper(mark, in_size + 1, &size_no_zeros, stream);

	if (size_no_zeros == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));

		return;
	}

	checkCudaErrors(cudaMalloc(&no_zeros, (size_no_zeros + 1) * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&tmp_bound, size_no_zeros * sizeof(ResBound)));

	RemoveZerosAsyncWrapper(index_count, in_bound, no_zeros, tmp_bound, mark, in_size, stream);

	ExclusiveScanAsyncWrapper(no_zeros, size_no_zeros + 1, &sum, stream);

	if (sum == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));
		checkCudaErrors(cudaFree(no_zeros));
		checkCudaErrors(cudaFree(tmp_bound));

		return;
	}

	ulong *tmp_location, *local_offset;

	checkCudaErrors(cudaMalloc(&tmp_location, sum * sizeof(ulong)));

	checkCudaErrors(cudaMemsetAsync(tmp_location, 0, sizeof(ulong) * sum, stream));

	MarkTmpLocationAsyncWrapper(tmp_location, no_zeros, size_no_zeros, stream);

	InclusiveScanAsyncWrapper(tmp_location, sum, stream);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	ComputeOffsetAsyncWrapper(no_zeros, tmp_location, local_offset, sum, stream);

	DecomposeAsyncWrapper(tmp_bound, *out_bound, tmp_location, local_offset, sum, stream);
	checkCudaErrors(cudaStreamSynchronize(stream));
	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));
}

__global__ void Decompose2(ulong *in, ResBound *in_bound, RESULT *out_bound, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		if (in_bound[i].left >= 0 && in_bound[i].right >= 0 && in_bound[i].outer >= 0) {
			int write_location = in[i];

			for (int j = in_bound[i].left; j <= in_bound[i].right; j++) {
				out_bound[write_location].lkey = in_bound[i].outer;
				out_bound[write_location].rkey = j;
				write_location++;
			}
		}
	}
}

void Rebalance2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExclusiveScanWrapper(in, in_size, out_size);

	if (*out_size == 0)
		return;

	checkCudaErrors(cudaMalloc(out_bound, sizeof(RESULT) * (*out_size)));
	Decompose2<<<grid_size, block_size>>>(in, in_bound, *out_bound, in_size - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void RebalanceAsync2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExclusiveScanAsyncWrapper(in, in_size, out_size, stream);

	if (*out_size == 0)
		return;

	checkCudaErrors(cudaMalloc(out_bound, sizeof(RESULT) * (*out_size)));
	Decompose2<<<grid_size, block_size, 0, stream>>>(in, in_bound, *out_bound, in_size - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

}
}

