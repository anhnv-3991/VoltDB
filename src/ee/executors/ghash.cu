
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
#include "gcommon/gpu_common.h"

#include "ghash.h"


extern "C" {
#define MASK_BITS 0x9e3779b9


__forceinline__ __device__ uint64_t Hasher(uint64_t *packed_key, int key_size)
{
	uint64_t seed = 0;

	for (int i = 0; i <  key_size; i++) {
		seed ^= packed_key[i] + MASK_BITS + (seed << 6) + (seed >> 2);
	}

	return seed;
}


__forceinline__ __device__ GNValue HashEvaluateItrFunc(GTreeNode *tree_expression,
															int tree_size,
															GNValue *outer_tuple,
															GNValue *inner_tuple,
															GNValue *stack,
															int offset)
{
	int top = 0;
	stack[0] = GNValue::getNullValue();

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx];
					top += offset;
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx];
					top += offset;
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = tree_expression[i].value;
				top += offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_and(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_or(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_equal(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_notEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_add(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_subtract(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_divide(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_multiply(stack[top - offset]);
				top -= offset;

				break;
			}
			default: {
				return GNValue::getFalse();
			}
		}
	}

	return stack[0];
}

__global__ void GhashCount(uint64_t *packed_key, int tuple_num, int key_size, ulong *hash_count, uint64_t max_buckets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < tuple_num; i += stride) {
		uint64_t hash = Hasher(packed_key + i * key_size, key_size);
		uint64_t bucket_offset = hash % max_buckets;
		hash_count[bucket_offset * stride + index]++;
	}

}


__global__ void Ghash(uint64_t *packed_key, ulong *hash_count, GHashNode hash_table)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int i;
	int key_size = hash_table.key_size;
	int max_buckets = hash_table.bucket_num;

	for (i = index; i <= max_buckets; i+= stride) {
		hash_table.bucket_location[i] = hash_count[i * stride];
	}

	__syncthreads();

	for (i = index; i < hash_table.size; i += stride) {
		uint64_t hash = Hasher(packed_key + i * key_size, key_size);
		uint64_t bucket_offset = hash % max_buckets;
		ulong hash_idx = hash_count[bucket_offset * stride + index];

		hash_table.hashed_idx[hash_idx] = i;

		for (int j = 0; j < key_size; j++) {
			hash_table.hashed_key[hash_idx * key_size + j] = packed_key[i * key_size + j];
		}

		hash_count[bucket_offset * stride + index]++;
	}
}



__global__ void HashJoin(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode outer_hash, GHashNode inner_hash,
							int base_outer_idx, int base_inner_idx,
							ulong *index_count, int size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
							GNValue *stack,
#elif defined(POST_EXP_)
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int bucket_idx = blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outer_idx, inner_idx;
	int outer_tuple_idx, inner_tuple_idx;
	int end_outer_idx, end_inner_idx;
	bool end_check;

	if (index < size && bucket_idx < outer_hash.bucket_num) {
		write_location = index_count[index];
		for (outer_idx = threadIdx.x + outer_hash.bucket_location[bucket_idx], end_outer_idx = outer_hash.bucket_location[bucket_idx + 1]; outer_idx < end_outer_idx; outer_idx += blockDim.x) {
			for (inner_idx = inner_hash.bucket_location[bucket_idx], end_inner_idx = inner_hash.bucket_location[bucket_idx + 1]; inner_idx < end_inner_idx; inner_idx++) {
				outer_tuple_idx = outer_hash.hashed_idx[outer_idx];
				inner_tuple_idx = inner_hash.hashed_idx[inner_idx];

				//key_check = equalityChecker(&outer_hash.hashedKey[outer_idx * outer_hash.key_size], &inner_hash.hashedKey[inner_idx * outer_hash.key_size], outer_hash.key_size);
#ifdef POST_EXP_
#ifdef FUNC_CALL_
				end_check = (end_size > 0) ? (bool)(EvaluateItrFunc(end_expression, end_size,
																	outer_table + outer_tuple_idx * outer_cols,
																	inner_table + inner_tuple_idx * inner_cols,
																	stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : true;
				end_check = (end_check && post_size > 0) ? (bool)(EvaluateItrFunc(post_expression, post_size,
																					outer_table + outer_tuple_idx * outer_cols,
																					inner_table + inner_tuple_idx * inner_cols,
																					stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : end_check;
#else
				end_check = (end_size > 0) ? (bool)(EvaluateItrNonFunc(end_expression, end_size,
																		outer_table + outer_tuple_idx * outer_cols,
																		inner_table + inner_tuple_idx * inner_cols,
																		val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : true;
				end_check = (end_check && post_size > 0) ? (bool)(EvaluateItrNonFunc(post_expression, post_size,
																						outer_table + outer_tuple_idx * outer_cols,
																						inner_table + inner_tuple_idx * inner_cols,
																						val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : end_check;
#endif

#else
#ifdef FUNC_CALL_
				end_check = (end_size > 0) ? (bool)(EvaluateRecvFunc(end_expression, 1, end_size,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols).getValue()) : true;
				end_check = (end_check && post_size > 0) ? (bool)(EvaluateRecvFunc(post_expression, 1, post_size,
																					outer_table + outer_idx * outer_cols,
																					inner_table + inner_idx * inner_cols).getValue()) : end_check;
#else
				end_check = (end_size > 0) ? (bool)(EvaluateRecvNonFunc(end_expression, 1, end_size,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols).getValue()) : true;
				end_check = (end_check && post_size > 0) ? (bool)(EvaluateRecvNonFunc(post_expression, 1, post_size,
																						outer_table + outer_idx * outer_cols,
																						inner_table + inner_idx * inner_cols).getValue()) : end_check;
#endif
#endif

				result[write_location].lkey = (end_check) ? (outer_tuple_idx + base_outer_idx) : (-1);
				result[write_location].rkey = (end_check) ? (inner_tuple_idx + base_inner_idx) : (-1);
				write_location++;
			}
		}
	}
}

__global__ void HashJoinShared(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp,	int post_size,
								GHashNode outer_hash, GHashNode inner_hash,
								int base_outer_idx, int base_inner_idx,
								ulong *index_count, int size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
								GNValue *stack,
#elif defined(POST_EXP_)
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int bucket_idx = blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outer_idx, inner_idx;
	int outer_tuple_idx, inner_tuple_idx;
	int end_outer_idx, end_inner_idx;
	GNValue end_check;
	__shared__ GNValue tmp_inner[SHARED_MEM];
	int real_size = 0;
	int shared_size = SHARED_MEM;
	int tmp = 0;

	if (index < size && blockIdx.x < outer_hash.bucket_num) {
		write_location = index_count[index];
		for (inner_idx = inner_hash.bucket_location[bucket_idx], end_inner_idx = inner_hash.bucket_location[bucket_idx + 1]; inner_idx < end_inner_idx; inner_idx += (shared_size / inner_cols)) {
			//inner_tuple_idx = inner_hash.hashed_idx[inner_idx];
			tmp = shared_size/inner_cols;
			real_size = ((inner_idx + tmp) < end_inner_idx) ? (tmp * inner_cols) : ((end_inner_idx - inner_idx) * inner_cols);
			for (int i = threadIdx.x; i < real_size; i += blockDim.x) {
				tmp_inner[i] = inner_table[inner_hash.hashed_idx[inner_idx + i / inner_cols] * inner_cols + i % inner_cols];
			}
			__syncthreads();

			for (outer_idx = threadIdx.x + outer_hash.bucket_location[bucket_idx], end_outer_idx = outer_hash.bucket_location[bucket_idx + 1]; outer_idx < end_outer_idx; outer_idx += blockDim.x) {
				outer_tuple_idx = outer_hash.hashed_idx[outer_idx];
				for (int tmp_inner_idx = 0; tmp_inner_idx < real_size/inner_cols; tmp_inner_idx++) {
					inner_tuple_idx = inner_hash.hashed_idx[inner_idx + tmp_inner_idx];
#ifdef POST_EXP_
#ifdef FUNC_CALL_
					end_check = (end_size > 0) ? (EvaluateItrFunc(end_exp, end_size,
																		outer_table + outer_tuple_idx * outer_cols,
																		tmp_inner + tmp_inner_idx * inner_cols,
																		stack + index, gridDim.x * blockDim.x)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_size > 0) ? (EvaluateItrFunc(post_exp, post_size,
																							outer_table + outer_tuple_idx * outer_cols,
																							tmp_inner + tmp_inner_idx * inner_cols,
																							stack + index, gridDim.x * blockDim.x)) : end_check;
#else
					end_check = (end_size > 0) ? EvaluateItrNonFunc(end_exp, end_size,
																	outer_table + outer_tuple_idx * outer_cols,
																	tmp_inner + tmp_inner_idx * inner_cols,
																	val_stack + index, type_stack + index, gridDim.x * blockDim.x) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_size > 0) ? (EvaluateItrNonFunc(post_exp, post_size,
																							outer_table + outer_tuple_idx * outer_cols,
																							tmp_inner + tmp_inner_idx * inner_cols,
																							val_stack + index, type_stack + index, gridDim.x * blockDim.x)) : end_check;
#endif
#else
#ifdef FUNC_CALL_
					end_check = (end_size > 0) ? (EvaluateRecvFunc(end_exp, 1, end_size,
																	outer_table + outer_tuple_idx * outer_cols,
																	tmp_inner + tmp_inner_idx * inner_cols)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_size > 0) ? (EvaluateRecvFunc(post_exp, 1, post_size,
																							outer_table + outer_tuple_idx * outer_cols,
																							tmp_inner + tmp_inner_idx * inner_cols)) : end_check;
#else
					end_check = (end_size > 0) ? (EvaluateRecvNonFunc(end_exp, 1, end_size,
																		outer_table + outer_tuple_idx * outer_cols,
																		tmp_inner + tmp_inner_idx * inner_cols)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_size > 0) ? (EvaluateRecvNonFunc(post_exp, 1, post_size,
																								outer_table + outer_tuple_idx * outer_cols,
																								tmp_inner + tmp_inner_idx * inner_cols)) : end_check;
#endif
#endif

					result[write_location].lkey = (end_check.isTrue()) ? (outer_tuple_idx + base_outer_idx) : (-1);
					result[write_location].rkey = (end_check.isTrue()) ? (inner_tuple_idx + base_inner_idx) : (-1);
					write_location++;
				}
			}
			__syncthreads();
		}

	}
}

__global__ void HashJoinLegacy(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								int size,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GHashNode inner_hash,
								int base_outer_idx, int base_inner_idx,
								ulong *write_location,
								ResBound *index_bound,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
								GNValue *stack,
#elif defined(POST_EXP_)
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	GNValue res;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += offset) {
		ulong location = write_location[i];
		int outer_idx = index_bound[i].outer;

		for (int left = index_bound[i].left, right = index_bound[i].right; left < right; left++) {
			int inner_idx = inner_hash.hashed_idx[left];

			res = GNValue::getTrue();

#ifdef POST_EXP_
#ifdef FUNC_CALL_
			res = (end_size > 0) ? EvaluateItrFunc(end_exp, end_size,
													outer_table + outer_idx * outer_cols,
													inner_table + inner_idx * inner_cols,
													stack + index, offset) : res;
			res = (res.isTrue() && post_size > 0) ? EvaluateItrFunc(post_exp, post_size,
																	outer_table + outer_idx * outer_cols,
																	inner_table + inner_idx * inner_cols,
																	stack + index, offset) : res;
#else
			res = (end_size > 0) ? EvaluateItrNonFunc(end_exp, end_size,
														outer_table + outer_idx * outer_cols,
														inner_table + inner_idx * inner_cols,
														val_stack + index, type_stack + index, offset) : res;

			res = (res.isTrue() && post_size > 0) ? EvaluateItrNonFunc(post_exp, post_size,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols,
																		val_stack + index, type_stack + index, offset) : res;
#endif
#else
#ifdef FUNC_CALL_
			res = (end_size > 0) ? EvaluateRecvFunc(end_exp, 1, end_size,
													outer_table + outer_idx * outer_cols,
													inner_table + inner_idx * inner_cols) : res;
			res = (res.isTrue() && post_size > 0) ? EvaluateRecvFunc(post_exp, 1, post_size,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols) : res;
#else
			res = (end_size > 0) ? EvaluateRecvNonFunc(end_exp, 1, end_size,
														outer_table + outer_idx * outer_cols,
														inner_table + inner_idx * inner_cols) : res;
			res = (res.isTrue() && post_size > 0) ? EvaluateRecvNonFunc(post_exp, 1, post_size,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols) : res;
#endif
#endif

			result[location].lkey = (res.isTrue()) ? (outer_idx + base_outer_idx) : (-1);
			result[location].rkey = (res.isTrue()) ? (inner_idx + base_inner_idx) : (-1);
			location++;
		}
		__syncthreads();
	}
}


void GhashWrapper(uint64_t *packed_key, GHashNode hash_table)
{
	int block_x, grid_x;

	block_x = (hash_table.size < BLOCK_SIZE_X) ? hash_table.size : BLOCK_SIZE_X;
	grid_x = 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *histogram;
	ulong sum;

	checkCudaErrors(cudaMalloc(&histogram, sizeof(ulong) * (block_x * grid_x * hash_table.bucket_num + 1)));
	checkCudaErrors(cudaMemset(histogram, 0, sizeof(ulong) * (block_x * grid_x * hash_table.bucket_num + 1)));
	checkCudaErrors(cudaDeviceSynchronize());

	GhashCount<<<grid_size, block_size>>>(packed_key, hash_table.size, hash_table.key_size, histogram, hash_table.bucket_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	ExclusiveScanWrapper(histogram, block_x * grid_x * hash_table.bucket_num + 1, &sum);

	Ghash<<<grid_size, block_size>>>(packed_key, histogram, hash_table);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(histogram));
}

void GhashAsyncWrapper(uint64_t *packed_key, GHashNode hash_table, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (hash_table.size < BLOCK_SIZE_X) ? hash_table.size : BLOCK_SIZE_X;
	grid_x = 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *histogram;
	ulong sum;

	checkCudaErrors(cudaMalloc(&histogram, sizeof(ulong) * (block_x * grid_x * hash_table.bucket_num + 1)));
	checkCudaErrors(cudaMemsetAsync(histogram, 0, sizeof(ulong) * (block_x * grid_x * hash_table.bucket_num + 1), stream));

	GhashCount<<<grid_size, block_size, 0, stream>>>(packed_key, hash_table.size, hash_table.key_size, histogram, hash_table.bucket_num);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

	ExclusiveScanAsyncWrapper(histogram, block_x * grid_x * hash_table.bucket_num + 1, &sum, stream);

	Ghash<<<grid_size, block_size, 0, stream>>>(packed_key, histogram, hash_table);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

	checkCudaErrors(cudaFree(histogram));
}

__global__ void HashIndexCount(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int bucket_idx = blockIdx.x + blockIdx.y * gridDim.x;

	if (blockIdx.x < outer_hash.bucket_num && index < size) {
		int left_bound = outer_hash.bucket_location[bucket_idx];
		int right_bound = outer_hash.bucket_location[bucket_idx + 1];
		int bucket_size = inner_hash.bucket_location[bucket_idx + 1] - inner_hash.bucket_location[bucket_idx];
		ulong count = 0;

		for (int i = threadIdx.x + left_bound; i < right_bound; i += blockDim.x) {
			count += bucket_size;
		}
		index_count[index] = count;
	}
}

void IndexCountWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size)
{
	int block_x, grid_x, grid_y;

	grid_x = (outer_hash.bucket_num < size) ? outer_hash.bucket_num : size;
	grid_y = (outer_hash.bucket_num - 1)/grid_x + 1;
	block_x = (outer_hash.size/grid_x < BLOCK_SIZE_X) ? outer_hash.size/grid_x : BLOCK_SIZE_X;

	checkCudaErrors(cudaMemset(index_count, 0, sizeof(ulong) * size));
	checkCudaErrors(cudaDeviceSynchronize());

	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCount<<<grid_size, block_size>>>(outer_hash, inner_hash, index_count, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void IndexCountAsyncWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size, cudaStream_t stream)
{
	int block_x, grid_x, grid_y;

	grid_x = (outer_hash.bucket_num < size) ? outer_hash.bucket_num : size;
	grid_y = (outer_hash.bucket_num - 1)/grid_x + 1;
	block_x = (outer_hash.size/grid_x < BLOCK_SIZE_X) ? outer_hash.size/grid_x : BLOCK_SIZE_X;

	checkCudaErrors(cudaMemsetAsync(index_count, 0, sizeof(ulong) * size, stream));

	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCount<<<grid_size, block_size, 0, stream>>>(outer_hash, inner_hash, index_count, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

void HashJoinWrapper(GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_exp, int end_size,
						GTreeNode *post_exp, int post_size,
						GHashNode outer_hash, GHashNode inner_hash,
						int base_outer_idx, int base_inner_idx,
						ulong *index_count, int size,
						RESULT *result)
{
	int block_x, grid_x, grid_y;

	grid_x = (outer_hash.bucket_num < size) ? outer_hash.bucket_num : size;
	grid_y = (outer_hash.bucket_num - 1)/grid_x + 1;
	block_x = (outer_hash.size/grid_x < BLOCK_SIZE_X) ? outer_hash.size/grid_x : BLOCK_SIZE_X;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, 1, 1);

#ifndef SHARED_
	HashJoin<<<grid_size, block_size>>>(outer_table, inner_table,
										outer_cols, inner_cols,
										end_exp, end_size,
										post_exp, post_size,
										outer_hash, inner_hash,
										base_outer_idx, base_inner_idx,
										index_count, size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
										stack,
#elif defined(POST_EXP_)
										val_stack,
										type_stack,
#endif
										result);
#else
	HashJoinShared<<<grid_size, block_size>>>(outer_table, inner_table,
												outer_cols, inner_cols,
												end_exp, end_size,
												post_exp, post_size,
												outer_hash, inner_hash,
												base_outer_idx, base_inner_idx,
												index_count, size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
												stack,
#elif defined(POST_EXP_)
												val_stack,
												type_stack,
#endif
												result);
#endif
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void HashJoinAsyncWrapper(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_exp, int end_size,
							GTreeNode *post_exp, int post_size,
							GHashNode outer_hash, GHashNode inner_hash,
							int base_outer_idx, int base_inner_idx,
							ulong *index_count, int size,
							RESULT *result, cudaStream_t stream)
{
	int block_x, grid_x, grid_y;

	grid_x = (outer_hash.bucket_num < size) ? outer_hash.bucket_num : size;
	grid_y = (outer_hash.bucket_num - 1)/grid_x + 1;
	block_x = (outer_hash.size/grid_x < BLOCK_SIZE_X) ? outer_hash.size/grid_x : BLOCK_SIZE_X;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, 1, 1);

#ifndef SHARED_
	HashJoin<<<grid_size, block_size, 0, stream>>>(outer_table, inner_table,
													outer_cols, inner_cols,
													end_exp, end_size,
													post_exp, post_size,
													outer_hash, inner_hash,
													base_outer_idx, base_inner_idx,
													index_count, size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
													stack,
#elif defined(POST_EXP_)
													val_stack,
													type_stack,
#endif
													result);
#else
	HashJoinShared<<<grid_size, block_size, SHARED_MEM * sizeof(GNValue), stream>>>(outer_table, inner_table,
																						outer_cols, inner_cols,
																						end_exp, end_size,
																						post_exp, post_size,
																						outer_hash, inner_hash,
																						base_outer_idx, base_inner_idx,
																						index_count, size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
																						stack,
#elif defined(POST_EXP_)
																						val_stack,
																						type_stack,
#endif
																						result);
#endif
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));


#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void HashJoinLegacyWrapper(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							int size,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode inner_hash,
							int base_outer_idx, int base_inner_idx,
							ulong *index_count,
							ResBound *index_bound,
							RESULT *result)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size < partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	HashJoinLegacy<<<grid_size, block_size>>>(outer_table, inner_table,
												outer_cols, inner_cols,
												size,
												end_expression, end_size,
												post_expression, post_size,
												inner_hash,
												base_outer_idx, base_inner_idx,
												index_count, index_bound,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
												stack,
#elif defined(POST_EXP_)
												val_stack,
												type_stack,
#endif
												result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void HashJoinLegacyAsyncWrapper(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								int size,
								GTreeNode *end_expression, int end_size,
								GTreeNode *post_expression,	int post_size,
								GHashNode inner_hash,
								int base_outer_idx, int base_inner_idx,
								ulong *index_count,
								ResBound *index_bound,
								RESULT *result, cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size < partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	HashJoinLegacy<<<grid_size, block_size, 0, stream>>>(outer_table, inner_table,
															outer_cols, inner_cols,
															size,
															end_expression, end_size,
															post_expression, post_size,
															inner_hash,
															base_outer_idx, base_inner_idx,
															index_count, index_bound,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
															stack,
#elif defined(POST_EXP_)
															val_stack,
															type_stack,
#endif
															result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

__global__ void HashIndexCountLegacy(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int key_size = inner_hash.key_size;
	int bucket_number = inner_hash.bucket_num;
	uint64_t hash_val;
	uint64_t bucket_offset;

	for (int i = index; i < outer_rows - 1; i += blockDim.x * gridDim.x) {
		hash_val = Hasher(outer_key + i * key_size, key_size);
		bucket_offset = hash_val % bucket_number;

		out_bound[i].outer = i;
		out_bound[i].left = inner_hash.bucket_location[bucket_offset];
		out_bound[i].right = inner_hash.bucket_location[bucket_offset + 1];
		index_count[i] = inner_hash.bucket_location[bucket_offset + 1] - inner_hash.bucket_location[bucket_offset];
	}

	__syncthreads();

	if (index == 0)
		index_count[outer_rows - 1] = 0;
}

void IndexCountLegacyAsyncWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCountLegacy<<<grid_size, block_size, 0, stream>>>(outer_key, outer_rows, inner_hash, index_count, out_bound);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

void IndexCountLegacyWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound)
{
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCountLegacy<<<grid_size, block_size>>>(outer_key, outer_rows, inner_hash, index_count, out_bound);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void HDecompose(RESULT *output, ResBound *in_bound, GHashNode in_hash, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		output[i].lkey = in_bound[in_location[i]].outer;
		output[i].rkey = in_hash.hashed_idx[in_bound[in_location[i]].left + local_offset[i]];
	}
}

void HDecomposeWrapper(RESULT *output, ResBound *in_bound, GHashNode in_hash, ulong *in_location, ulong *local_offset, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	HDecompose<<<grid_size, block_size>>>(output, in_bound, in_hash, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void HDecomposeAsyncWrapper(RESULT *output, ResBound *in_bound, GHashNode in_hash, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	HDecompose<<<grid_size, block_size, 0, stream>>>(output, in_bound, in_hash, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

void HRebalance2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size)
{
	ExclusiveScanWrapper(index_count, in_size, out_size);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaMemset(location, 0, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaDeviceSynchronize());


	MarkLocationWrapper(location, index_count, in_size);


	InclusiveScanWrapper(location, *out_size);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	ComputeOffsetWrapper(index_count, location, local_offset, *out_size);

	HDecomposeWrapper(*out_bound, in_bound, inner_hash, location, local_offset, *out_size);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void HRebalanceAsync2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	ExclusiveScanAsyncWrapper(index_count, in_size, out_size, stream);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaMemsetAsync(location, 0, sizeof(ulong) * (*out_size), stream));

	MarkLocationAsyncWrapper(location, index_count, in_size, stream);

	InclusiveScanAsyncWrapper(location, *out_size, stream);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	ComputeOffsetAsyncWrapper(index_count, location, local_offset, *out_size, stream);

	HDecomposeAsyncWrapper(*out_bound, in_bound, inner_hash, location, local_offset, *out_size, stream);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void HRebalance(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size)
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
	HDecomposeWrapper(*out_bound, tmp_bound, inner_hash, tmp_location, local_offset, sum);

	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));

}

void HRebalanceAsync(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
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
	checkCudaErrors(cudaMemset(tmp_location, 0, sizeof(ulong) * sum));
	checkCudaErrors(cudaDeviceSynchronize());

	MarkTmpLocationAsyncWrapper(tmp_location, no_zeros, size_no_zeros, stream);

	InclusiveScanAsyncWrapper(tmp_location, sum, stream);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	ComputeOffsetAsyncWrapper(no_zeros, tmp_location, local_offset, sum, stream);
	HDecomposeAsyncWrapper(*out_bound, tmp_bound, inner_hash, tmp_location, local_offset, sum, stream);

	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));

}

__global__ void HashIndexCount2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound)
{
	int max_buckets = outer_hash.bucket_num;
	int size = outer_hash.size;

	for (int bucket_idx = blockIdx.x; bucket_idx < max_buckets; bucket_idx += gridDim.x) {
		int size_of_bucket = inner_hash.bucket_location[bucket_idx + 1] - inner_hash.bucket_location[bucket_idx];

		for (int outer_idx = threadIdx.x + outer_hash.bucket_location[bucket_idx], end_outer_idx = outer_hash.bucket_location[bucket_idx + 1]; outer_idx < end_outer_idx; outer_idx += blockDim.x) {
			index_count[outer_idx] = size_of_bucket;
			out_bound[outer_idx].outer = outer_hash.hashed_idx[outer_idx];
			out_bound[outer_idx].left = inner_hash.bucket_location[bucket_idx];
			out_bound[outer_idx].right = inner_hash.bucket_location[bucket_idx + 1];
		}
		__syncthreads();
	}

	if (threadIdx.x + blockIdx.x * blockDim.x == 0)
		index_count[size] = 0;
}

void IndexCountWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound)
{
	int block_x, grid_x;

	block_x = (outer_hash.bucket_num < BLOCK_SIZE_X) ? outer_hash.bucket_num : BLOCK_SIZE_X;
	grid_x = (outer_hash.bucket_num - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCount2<<<grid_size, block_size>>>(outer_hash, inner_hash, index_count, out_bound);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void IndexCountAsyncWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (outer_hash.bucket_num < BLOCK_SIZE_X) ? outer_hash.bucket_num : BLOCK_SIZE_X;
	grid_x = (outer_hash.bucket_num - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	HashIndexCount2<<<grid_size, block_size, 0, stream>>>(outer_hash, inner_hash, index_count, out_bound);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

}
