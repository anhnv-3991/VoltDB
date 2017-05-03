
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
#include "gpu_common.h"

#include "ghash.h"


extern "C" {
#define MASK_BITS 0x9e3779b9



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



__global__ void HashJoin(GTable outer, GTable inner,
							GTree end_pred, GTree post_pred,
							GHashNode outer_hash, GHashNode inner_hash,
							ulong *index_count, int size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
							GNValue *stack,
#elif defined(POST_EXP_)
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result)
{
	int64_t *outer_table = outer.block_list->gdata, *inner_table = inner.block_list->gdata;
	int outer_cols = outer.column_num, inner_cols = inner.column_num;
	GColumnInfo *outer_schema = outer.schema, *inner_schema = inner.schema;

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

#ifdef POST_EXP_
#ifdef FUNC_CALL_
				end_check = (end_pred.size > 0) ? (bool)(EvaluateItrFunc(end_pred,
																			outer_table + outer_tuple_idx * outer_cols, inner_table + inner_tuple_idx * inner_cols,
																			outer_schema, inner_schema,
																			stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : true;
				end_check = (end_check && post_pred.size > 0) ? (bool)(EvaluateItrFunc(post_pred,
																						outer_table + outer_tuple_idx * outer_cols, inner_table + inner_tuple_idx * inner_cols,
																						outer_schema, inner_schema,
																						stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : end_check;
#else
				end_check = (end_pred.size > 0) ? (bool)(EvaluateItrNonFunc(end_pred,
																				outer_table + outer_tuple_idx * outer_cols, inner_table + inner_tuple_idx * inner_cols,
																				outer_schema, inner_schema,
																				val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : true;
				end_check = (end_check && post_pred.size > 0) ? (bool)(EvaluateItrNonFunc(post_pred,
																							outer_table + outer_tuple_idx * outer_cols, inner_table + inner_tuple_idx * inner_cols,
																							outer_schema, inner_schema,
																							val_stack + index, type_stack + index, gridDim.x * blockDim.x * gridDim.y).getValue()) : end_check;
#endif

#else
#ifdef FUNC_CALL_
				end_check = (end_pred.size > 0) ? (bool)(EvaluateRecvFunc(end_pred, 1,
																			outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																			outer_schema, inner_schema).getValue()) : true;
				end_check = (end_check && post_pred.size > 0) ? (bool)(EvaluateRecvFunc(post_pred, 1,
																							outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																							outer_schema, inner_schema).getValue()) : end_check;
#else
				end_check = (end_pred.size > 0) ? (bool)(EvaluateRecvNonFunc(end_pred, 1,
																				outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																				outer_schema, inner_schema).getValue()) : true;
				end_check = (end_check && post_pred.size > 0) ? (bool)(EvaluateRecvNonFunc(post_pred, 1,
																							outer_table + outer_idx * outer_cols, inner_table + inner_idx * inner_cols,
																							outer_schema, inner_schema).getValue()) : end_check;
#endif
#endif

				result[write_location].lkey = (end_check) ? (outer_tuple_idx) : (-1);
				result[write_location].rkey = (end_check) ? (inner_tuple_idx) : (-1);
				write_location++;
			}
		}
	}
}

__global__ void HashJoinShared(GTable outer, GTable inner,
								GTree end_exp, GTree post_exp,
								GHashNode outer_hash, GHashNode inner_hash,
								ulong *index_count, int size,
#if defined(FUNC_CALL_) && defined(POST_EXP_)
								GNValue *stack,
#elif defined(POST_EXP_)
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result)
{
	int64_t *outer_table = outer.block_list->gdata, *inner_table = inner.block_list->gdata;
	int outer_cols = outer.column_num, inner_cols = inner.column_num;
	GColumnInfo *outer_schema = outer.schema, *inner_schema = inner.schema;

	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int bucket_idx = blockIdx.x + blockIdx.y * gridDim.x;

	ulong write_location;
	int outer_idx, inner_idx;
	int outer_tuple_idx, inner_tuple_idx;
	int end_outer_idx, end_inner_idx;
	GNValue end_check;
	__shared__ int64_t tmp_inner[SHARED_MEM];
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
					end_check = (end_exp.size > 0) ? (EvaluateItrFunc(end_exp,
																		outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																		outer_schema, inner_schema,
																		stack + index, gridDim.x * blockDim.x)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_exp.size > 0) ? (EvaluateItrFunc(post_exp,
																							outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																							outer_schema, inner_schema,
																							stack + index, gridDim.x * blockDim.x)) : end_check;
#else
					end_check = (end_exp.size > 0) ? EvaluateItrNonFunc(end_exp,
																		outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																		outer_schema, inner_schema,
																		val_stack + index, type_stack + index, gridDim.x * blockDim.x) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_exp.size > 0) ? (EvaluateItrNonFunc(post_exp,
																								outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																								outer_schema, inner_schema,
																								val_stack + index, type_stack + index, gridDim.x * blockDim.x)) : end_check;
#endif
#else
#ifdef FUNC_CALL_
					end_check = (end_exp.size > 0) ? (EvaluateRecvFunc(end_exp, 1,
																		outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																		outer_schema, inner_schema)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_exp.size > 0) ? (EvaluateRecvFunc(post_exp, 1,
																								outer_table + outer_tuple_idx * outer_cols, tmp_inner + tmp_inner_idx * inner_cols,
																								outer_schema, inner_schema)) : end_check;
#else
					end_check = (end_exp.size > 0) ? (EvaluateRecvNonFunc(end_exp, 1,
																			outer_table + outer_tuple_idx * outer_cols,
																			tmp_inner + tmp_inner_idx * inner_cols,
																			outer_schema, inner_schema)) : GNValue::getTrue();
					end_check = (end_check.isTrue() && post_exp.size > 0) ? (EvaluateRecvNonFunc(post_exp, 1,
																								outer_table + outer_tuple_idx * outer_cols,
																								tmp_inner + tmp_inner_idx * inner_cols,
																								outer_schema, inner_schema)) : end_check;
#endif
#endif

					result[write_location].lkey = (end_check.isTrue()) ? (outer_tuple_idx) : (-1);
					result[write_location].rkey = (end_check.isTrue()) ? (inner_tuple_idx) : (-1);
					write_location++;
				}
			}
			__syncthreads();
		}

	}
}

__global__ void HashJoinLegacy(GTable outer, GTable inner,
								GTree end_exp, GTree post_exp,
								GHashNode inner_hash,
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
	int64_t *outer_table = outer.block_list->gdata, *inner_table = inner.block_list->gdata;
	int outer_cols = outer.column_num, inner_cols = inner.column_num;
	GColumnInfo *outer_schema = outer.schema, *inner_schema = inner.schema;
	int outer_rows = outer.block_list->rows;

	GNValue res;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i += offset) {
		ulong location = write_location[i];
		int outer_idx = index_bound[i].outer;

		for (int left = index_bound[i].left, right = index_bound[i].right; left < right; left++) {
			int inner_idx = inner_hash.hashed_idx[left];

			res = GNValue::getTrue();

#ifdef POST_EXP_
#ifdef FUNC_CALL_
			res = (end_exp.size > 0) ? EvaluateItrFunc(end_exp,
														outer_table + outer_idx * outer_cols,
														inner_table + inner_idx * inner_cols,
														outer_schema, inner_schema,
														stack + index, offset) : res;
			res = (res.isTrue() && post_exp.size > 0) ? EvaluateItrFunc(post_exp,
																		outer_table + outer_idx * outer_cols,
																		inner_table + inner_idx * inner_cols,
																		outer_schema, inner_schema,
																		stack + index, offset) : res;
#else
			res = (end_exp.size > 0) ? EvaluateItrNonFunc(end_exp,
															outer_table + outer_idx * outer_cols,
															inner_table + inner_idx * inner_cols,
															outer_schema, inner_schema,
															val_stack + index, type_stack + index, offset) : res;

			res = (res.isTrue() && post_exp.size > 0) ? EvaluateItrNonFunc(post_exp,
																			outer_table + outer_idx * outer_cols,
																			inner_table + inner_idx * inner_cols,
																			outer_schema, inner_schema,
																			val_stack + index, type_stack + index, offset) : res;
#endif
#else
#ifdef FUNC_CALL_
			res = (end_exp.size > 0) ? EvaluateRecvFunc(end_exp, 1,
															outer_table + outer_idx * outer_cols,
															inner_table + inner_idx * inner_cols,
															outer_schema, inner_schema) : res;
			res = (res.isTrue() && post_exp.size > 0) ? EvaluateRecvFunc(post_exp, 1,
																			outer_table + outer_idx * outer_cols,
																			inner_table + inner_idx * inner_cols,
																			outer_schema, inner_schema) : res;
#else
			res = (end_exp.size > 0) ? EvaluateRecvNonFunc(end_exp, 1,
															outer_table + outer_idx * outer_cols,
															inner_table + inner_idx * inner_cols,
															outer_schema, inner_schema) : res;
			res = (res.isTrue() && post_exp.size > 0) ? EvaluateRecvNonFunc(post_exp, 1,
																				outer_table + outer_idx * outer_cols,
																				inner_table + inner_idx * inner_cols,
																				outer_schema, inner_schema) : res;
#endif
#endif

			result[location].lkey = (res.isTrue()) ? (outer_idx) : (-1);
			result[location].rkey = (res.isTrue()) ? (inner_idx) : (-1);
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


}
