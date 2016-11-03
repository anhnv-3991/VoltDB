#include "ghash.h"


extern "C" {
#define MASK_BITS 0x9e3779b9


__device__ void keyGenerate(GNValue *tuple, int *keyIndices, int indexNum, uint64_t *packedKey)
{
	int keyOffset = 0;
	int intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);

	if (keyIndices != NULL) {
		for (int i = 0; i < indexNum; i++) {
			int64_t value = tuple[keyIndices[i]].getValue();
			uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);

			for (int j = sizeof(int64_t) - 1; k >= 0; k--) {
				packedKey[keyOffset] |= (0xFF & (keyValue >> (k * 8))) << (intraKeyOffset * 8);
				intraKeyOffset--;
				if (intraKeyOffset < 0) {
					intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
					keyOffset++;
				}
			}
		}
	} else {
		for (int i = 0; i < indexNum; i++) {
			int64_t value = tuple[i].getValue();
			uint64_t keyValue = static_cast<uint64_t>(value + INT64_MAX + 1);

			for (int j = sizeof(int64_t) - 1; k >= 0; k--) {
				packedKey[keyOffset] |= (0xFF & (keyValue >> (k * 8))) << (intraKeyOffset * 8);
				intraKeyOffset--;
				if (intraKeyOffset < 0) {
					intraKeyOffset = static_cast<int>(sizeof(uint64_t) - 1);
					keyOffset++;
				}
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
				return GNValue::getFalse();
			}
		}
	}

	return stack[0];
}

__global__ void packKey(GNValue *index_table, int tuple_num, int col_num, int *indices, int index_num, uint64_t *packedKey, int keySize)
{
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < tuple_num; i += blockDim.x * gridDim.x) {
		keyGenerate(index_table + i, indices, index_num, packedKey + i * keySize);
	}
}

__global__ void ghashCount(uint64_t *packedKey, int tupleNum, int keySize, uint64_t *hashCount, uint64_t maxNumberOfBuckets)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < tupleNum; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		hashCount[bucketOffset * stride + index]++;
	}

	__syncthreads();

	if (i == keyNum) {
		hashCount[i] = 0;
	}
}

__global__ void ghash(uint64_t *packedKey, int tupleNum, int keySize, uint64_t *hashCount, uint64_t maxNumberOfBuckets, uint64_t *hashedIndex, uint64_t *bucketLocation)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	if (index < maxNumberOfBuckets) {
		bucketLocation[index] = hashCount[index * stride];
	}

	if (index == maxNumberOfBuckets) {
		bucketLocation[index] = hashCount[(index - 1) * stride];
	}

	__syncthreads();

	for (int i = index; i < tupleNum; i += stride) {
		uint64_t hash = hasher(packedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		hashedIndex[hashCount[bucketOffset * stride + index]] = i;
		hashCount[bucketOffset * stride + index]++;
	}
}

__global__ void hashIndexCount(GNValue *outer_table, int outer_rows, uint64_t *searchPackedKey,
								GTreeNode *searchKeyExp, int *searchKeySize, int searchExpNum,
								uint64_t *packedKey, uint64_t *bucketLocation, uint64_t *hashedIndex,
								ulong *indexCount, int keySize, int maxNumberOfBuckets, GNValue *stack)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	GNValue tmp_outer[4];
	int search_ptr = 0;
	int count_res = 0;

	for (int i = index; i < outer_rows; i += stride) {
		for (int j = 0; j < searchExpNum; search_ptr += searchKeySize[j], j++) {
			tmp_outer[j] = hashEvaluate(searchKeyExp + search_ptr, searchKeySize[j], outer_table + i, NULL, stack + index, stride);
		}

		keyGenerate(tmp_outer, NULL, searchExpNum, searchPackedKey + i * keySize);

		uint64_t bucketOffset = hasher(searchPackedKey + i * keySize, keySize) % maxNumberOfBuckets;
		uint64_t start = bucketLocation[bucketOffset];
		uint64_t end = bucketLocation[bucketOffset + 1];

		for (uint64_t j = start; j < end; j++) {
			count_res += (equalityChecker(searchPackedKey + i * keySize, packedKey + j, keySize)) ? 1 : 0;
		}
	}

	indexCount[index] = count_res;
	if (index == outer_rows - 1) {
		indexCount[outer_rows] = 0;
	}
}

__global__ void hashJoin(GNValue *outer_table, GNValue *inner_table, int outer_rows,
							int outer_cols, int inner_cols,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							uint64_t *searchPackedKey,
							uint64_t *packedKey,
							uint64_t *bucketLocation,
							uint64_t *hashedIndex,
							ulong *indexCount,
							int keySize,
							int maxNumberOfBuckets,
							GNValue *stack,
							RESULT *result)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	bool key_check;
	GNValue exp_check;
	ulong write_location = indexCount[index];

	for (int i = index; i < outer_rows; i += stride) {
		uint64_t bucketOffset = hasher(searchPackedKey + i * keySize, keySize) % maxNumberOfBuckets;
		uint64_t start = bucketLocation[bucketOffset];
		uint64_t end = bucketLocation[bucketOffset + 1];

		for (uint64_t j = start; j < end; j++) {
			key_check = equalityChecker(searchPackedKey + i * keySize, packedKey + j, keySize);
			exp_check = (key_check) ? hashEvaluate(end_expression, end_size, outer_table + i * outer_cols, inner_table + j * inner_cols, stack + index, stride) : GNValue::getFalse();
			exp_check = (exp_check.isTrue()) ? hashEvaluate(post_expression, post_size, outer_table + i * outer_cols, inner_table + j * inner_cols, stack + index, stride) : GNValue::getFalse();
			result[write_location].lkey = (exp_check.isTrue()) ? i : (-1);
			result[write_location].rkey = (exp_check.isTrue()) ? hashedIndex[j] : (-1);
			write_location++;
		}
	}
}

void packKeyWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue **index_table,
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
						uint64_t *hashCount,
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
					int keyNum,
					uint64_t *hashCount,
					int keySize,
					int numberOfBuckets,
					uint64_t *hashedIndex,
					uint64_t *bucketLocation
					)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	ghash<<<gridSize, blockSize>>>(packedKey, keyNum, hashCount, keySize, numberOfBuckets, hashedIndex, bucketLocation);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (ghash) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void indexCountWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *outer_table,
					int outer_rows,
					int col_num,
					uint64_t *searchPackedKey,
					GTreeNode *searchKeyExp,
					int *searchKeySize,
					int searchExpNum,
					uint64_t *packedKey,
					uint64_t *bucketLocation,
					uint64_t *hashedIndex,
					int *indexCount,
					int keySize,
					int maxNumberOfBuckets,
					GNValue *stack
					)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashIndexCount<<<gridSize, blockSize>>>(outer_table, outer_rows, searchPackedKey,
											searchKeyExp, searchKeySize, searchExpNum,
											packedKey, bucketLocation,
											hashedIndex, indexCount,
											keySize, maxNumberOfBuckets, stack);
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
						int outer_rows,
						int outer_cols,
						int inner_cols,
						GTreeNode *end_expression,
						int end_size,
						GTreeNode *post_expression,
						int post_size,
						uint64_t *searchPackedKey,
						uint64_t *packedKey,
						uint64_t *bucketLocation,
						uint64_t *hashedIndex,
						ulong *indexCount,
						int keySize,
						int maxNumberOfBuckets,
						RESULT *result
						)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashJoin<<<gridSize, blockSize>>>(outer_table, inner_table,
										outer_rows, outer_cols, inner_cols,
										end_expression, end_size,
										post_expression, post_size,
										searchPackedKey,
										packedKey, bucketLocation,
										hashedIndex, indexCount,
										keySize, maxNumberOfBuckets,
										stack, result);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (hashIndexCount) error: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void prefixSumWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}
}
