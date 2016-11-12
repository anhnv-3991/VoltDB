
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


#include "ghash.h"


extern "C" {
#define MASK_BITS 0x9e3779b9


__device__ void keyGenerate(GNValue *tuple, int *keyIndices, int indexNum, uint64_t *packedKey)
{
	int keyOffset = 0;
	int intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
	GNValue tmp_val;

	if (keyIndices != NULL) {
		for (int i = 0; i < indexNum; i++) {
			tmp_val = tuple[keyIndices[i]];
			int64_t value = tmp_val.getValue();
			uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);

			switch (tmp_val.getValueType()) {
				case VALUE_TYPE_TINYINT: {
					for (int j = sizeof(uint8_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					for (int j = sizeof(uint16_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= ((0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8));
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					for (int j = sizeof(uint64_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				default: {
					printf("Error: no match type. Type = %d\n", tmp_val.getValueType());
				}
			}
		}
	} else {
		for (int i = 0; i < indexNum; i++) {
			tmp_val = tuple[i];
			int64_t value = tmp_val.getValue();
			uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);

			switch (tmp_val.getValueType()) {
				case VALUE_TYPE_TINYINT: {
					for (int j = sizeof(uint8_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}
					break;
				}
				case VALUE_TYPE_SMALLINT: {
					for (int j = sizeof(uint16_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_INTEGER: {
					for (int j = sizeof(uint32_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				case VALUE_TYPE_BIGINT: {
					for (int j = sizeof(uint64_t) - 1; j >= 0; j--) {
						packedKey[keyOffset] |= (0xFF & (keyValue >> (j * 8))) << (intraKeyOffset * 8);
						intraKeyOffset--;
						if (intraKeyOffset < 0) {
							intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
							keyOffset++;
						}
					}

					break;
				}
				default:
					printf("Error: cannot detect type at index %d\n", i);
			}

		}
	}
}

__device__ uint64_t hasher(uint64_t *packedKey, int keySize)
{
	uint64_t seed = 0;

	for (int i = 0; i <  keySize; i++) {
		seed ^= packedKey[i] + MASK_BITS + (seed << 6) + (seed >> 2);
	}

	return seed;
}

__device__ bool equalityChecker(uint64_t *leftKey, uint64_t *rightKey, int keySize)
{
	bool res = true;

	for (int i = 0; i < keySize; i++) {
		res &= (leftKey[i] == rightKey[i]);
	}

	return res;
}

__device__ GNValue hashEvaluate(GTreeNode *tree_expression,
									int tree_size,
									GNValue *outer_tuple,
									GNValue *inner_tuple,
									GNValue *stack,
									int offset)
{
	int top = 0;

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
				printf("Wrong parameter at i = %d\n", i);
				return GNValue::getFalse();
			}
		}
	}

	return stack[0];
}

__global__ void packKey(GNValue *index_table, int tuple_num, int col_num, int *indices, int index_num, uint64_t *packedKey, int keySize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int stride = blockDim.x * gridDim.x * gridDim.y;

	for (int i = index; i < tuple_num * keySize; i += stride) {
		packedKey[i] = 0;
	}

	__syncthreads();

	for (int i = index; i < tuple_num; i += stride) {
		keyGenerate(index_table + i * col_num, indices, index_num, packedKey + i * keySize);
	}
}


__global__ void packSearchKey(GNValue *outer_table, int outer_rows, int outer_cols,
								uint64_t *searchPackedKey, GTreeNode *searchKeyExp,
								int *searchKeySize, int searchExpNum,
								int keySize, GNValue *stack)
{
	int index = threadIdx.x + blockIdx.x *blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int stride = blockDim.x * gridDim.x * gridDim.y;
	GNValue tmp_outer[4];
	int search_ptr = 0;

	for (int i = index; i < outer_rows * keySize; i += stride) {
		searchPackedKey[i] = 0;
	}

	__syncthreads();

	for (int i = index; i < outer_rows; i += stride) {
		search_ptr = 0;
		for (int j = 0; j < searchExpNum; search_ptr += searchKeySize[j], j++) {
			tmp_outer[j] = hashEvaluate(searchKeyExp + search_ptr, searchKeySize[j], outer_table + i * outer_cols, NULL, stack + index, stride);
			//tmp_outer[j].debug();
		}

		keyGenerate(tmp_outer, NULL, searchExpNum, searchPackedKey + i * keySize);
	}
}


__global__ void ghashCount(uint64_t *packedKey, int tupleNum, int keySize, ulong *hashCount, uint64_t maxNumberOfBuckets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int stride = blockDim.x * gridDim.x * gridDim.y;
	int i;

	for (i = index; i <= maxNumberOfBuckets * blockDim.x; i += stride) {
		hashCount[i] = 0;
	}

	__syncthreads();

	for (i = index; i < tupleNum; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		hashCount[bucketOffset * stride + index]++;
	}
}


__global__ void ghash(uint64_t *packedKey, ulong *hashCount, GHashNode hashTable)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int stride = blockDim.x * gridDim.x * gridDim.y;
	int i;
	int keySize = hashTable.keySize;
	int maxNumberOfBuckets = hashTable.bucketNum;

	for (i = index; i <= maxNumberOfBuckets; i += stride) {
		hashTable.bucketLocation[i] = hashCount[i * stride];
	}

	__syncthreads();

	for (i = index; i < hashTable.size; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		ulong hashIdx = hashCount[bucketOffset * stride + index];

		hashTable.hashedIdx[hashIdx] = i;

		for (int j = 0; j < keySize; j++) {
			hashTable.hashedKey[hashIdx * keySize + j] = packedKey[i * keySize + j];
		}

		hashCount[bucketOffset * stride + index]++;

		//printf("index %d at bucket %d\n", i, (int)bucketOffset);
	}
}


__global__ void hashIndexCount(GHashNode outerHash, GHashNode innerHash, int lowerBound, int upperBound, ulong *indexCount, int size)
{
	int bucketIdx;
	int endOuterIdx, endInnerIdx;
	int outerIdx, innerIdx;
	ulong count_res = 0;
	int threadGlobalIndex = threadIdx.x + blockIdx.x * blockDim.x + blockDim.x * gridDim.x * blockIdx.y;
	int stride = gridDim.x * gridDim.y;
	int keySize = outerHash.keySize;
	//bool res_check;

	for (int i = threadGlobalIndex; i <= size; i += stride * blockDim.x) {
		indexCount[threadGlobalIndex] = 0;
	}

	__syncthreads();

	if (threadGlobalIndex < size) {

		for (bucketIdx = lowerBound + blockIdx.x + gridDim.x * blockIdx.y; bucketIdx < upperBound; bucketIdx += stride) {
			for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
				for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {

					count_res += (equalityChecker(outerHash.hashedKey + outerIdx * keySize, innerHash.hashedKey + innerIdx * keySize, keySize)) ? 1 : 0;
				}
			}
		}

		indexCount[threadGlobalIndex] = count_res;
	}
}

__global__ void hashJoin(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode outerHash,
							GHashNode innerHash,
							int lowerBound,
							int upperBound,
							ulong *indexCount,
							int size,
							GNValue *stack,
							RESULT *result)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	int stride = gridDim.x * gridDim.y;
	int globalStride = gridDim.x * gridDim.y * blockDim.x;

	bool key_check;
	GNValue exp_check;
	ulong write_location;
	int bucketIdx;
	int endOuterIdx, endInnerIdx;
	int outerIdx, innerIdx;
	int keySize = outerHash.keySize;
	int outerTupleIdx, innerTupleIdx;
	bool writeLocChanged = false;

	if (index < size) {
		write_location = indexCount[index];

		for (bucketIdx = lowerBound + blockIdx.x + blockIdx.y * gridDim.x; bucketIdx < upperBound; bucketIdx += stride) {
			for (outerIdx = threadIdx.x + outerHash.bucketLocation[bucketIdx], endOuterIdx = outerHash.bucketLocation[bucketIdx + 1]; outerIdx < endOuterIdx; outerIdx += blockDim.x) {
				for (innerIdx = innerHash.bucketLocation[bucketIdx], endInnerIdx = innerHash.bucketLocation[bucketIdx + 1]; innerIdx < endInnerIdx; innerIdx++) {
					outerTupleIdx = outerHash.hashedIdx[outerIdx];
					innerTupleIdx = innerHash.hashedIdx[innerIdx];

					key_check = equalityChecker(&outerHash.hashedKey[outerIdx * keySize], &innerHash.hashedKey[innerIdx * keySize], keySize);
					exp_check = (key_check) ? GNValue::getTrue() : GNValue::getFalse();
					exp_check = (exp_check.isTrue() && end_size > 0) ? hashEvaluate(end_expression, end_size,
																						outer_table + outerTupleIdx * outer_cols,
																						inner_table + innerTupleIdx * inner_cols,
																						stack + index, globalStride) : exp_check;
					exp_check = (exp_check.isTrue() && post_size > 0) ? hashEvaluate(post_expression, post_size,
																						outer_table + outerTupleIdx * outer_cols,
																						inner_table + innerTupleIdx * inner_cols,
																						stack + index, globalStride) : exp_check;

					result[write_location].lkey = (exp_check.isTrue()) ? outerTupleIdx : ((!writeLocChanged) ? result[write_location].lkey : (-1));
					result[write_location].rkey = (exp_check.isTrue()) ? innerTupleIdx : ((!writeLocChanged) ? result[write_location].lkey : (-1));
					write_location += (key_check) ? 1 : 0;
					writeLocChanged = (key_check) ? true : false;

				}
			}
		}
	}
}

void packKeyWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *index_table,
					int tuple_num,
					int col_num,
					int *indices,
					int index_num,
					uint64_t *packedKey,
					int keySize)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	packKey<<<gridSize, blockSize>>>(index_table, tuple_num, col_num, indices, index_num, packedKey, keySize);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (packKey) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void ghashCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						uint64_t *packedKey,
						int keyNum,
						int keySize,
						ulong *hashCount,
						uint64_t maxNumberOfBuckets
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghashCount<<<gridSize, blockSize>>>(packedKey, keyNum, keySize, hashCount, maxNumberOfBuckets);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghashCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void ghashWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					uint64_t *packedKey,
					ulong *hashCount,
					GHashNode hashTable
					)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghash<<<gridSize, blockSize>>>(packedKey, hashCount, hashTable);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghash) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void packSearchKeyWrapper(int block_x, int block_y,
							int grid_x, int grid_y,
							GNValue *outer_table, int outer_rows, int outer_cols,
							uint64_t *searchPackedKey, GTreeNode *searchKeyExp,
							int *searchKeySize, int searchExpNum,
							int keySize, GNValue *stack)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	packSearchKey<<<gridSize, blockSize>>>(outer_table, outer_rows, outer_cols, searchPackedKey, searchKeyExp, searchKeySize, searchExpNum, keySize, stack);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghash) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());

}

void indexCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashIndexCount<<<gridSize, blockSize>>>(outerHash, innerHash, lowerBound, upperBound, indexCount,size);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashIndexCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void hashJoinWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table,
						GNValue *inner_table,
						int outer_cols,
						int inner_cols,
						GTreeNode *end_expression,
						int end_size,
						GTreeNode *post_expression,
						int post_size,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size,
						GNValue *stack,
						RESULT *result
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashJoin<<<gridSize, blockSize>>>(outer_table, inner_table,
										outer_cols, inner_cols,
										end_expression, end_size,
										post_expression, post_size,
										outerHash, innerHash,
										lowerBound, upperBound,
										indexCount, size,
										stack, result);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashIndexCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void hprefixSumWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}
}
