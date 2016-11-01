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

__device__ GNValue evaluate2(GTreeNode *tree_expression,
								int tree_size,
								GNValue *outer_tuple,
								GNValue *inner_tuple,
								GNValue *stack,
								int offset);

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

__global__ void hashIndexCount(GNValue *outer_table, int tuple_num, int col_num, uint64_t *searchPackedKey,
								GTreeNode *searchKeyExp, int *searchKeySize, int searchExpNum,
								uint64_t *packedKey, uint64_t *bucketLocation, uint64_t *hashedIndex,
								int *indexCount, int keySize, int numberOfBuckets, GNValue *stack)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	GNValue tmp_outer[4];
	int search_ptr = 0;
	int count_res = 0;

	for (int i = index; i < tuple_num; i += stride) {
		for (int j = 0; j < searchExpNum; search_ptr += searchKeySize[j], j++) {
			tmp_outer[j] = evaluate2(searchKeyExp + search_ptr, searchKeySize[j], outer_table + i, NULL, stack + index, stride);
		}

		keyGenerate(tmp_outer, NULL, searchExpNum, searchPackedKey + i * keySize);

		uint64_t hash = hasher(searchPackedKey + i * keySize, keySize);
		uint64_t bucketOffset = hash % maxNumberOfBuckets;
		int start = bucketLocation[bucketOffset];
		int end = bucketLocation[bucketOffset + 1];

		for (int j = start; j < end; j++) {
			count_res += (equalityChecker(searchPackedKey + i * keySize, packedKey + j, keySize)) ? 1 : 0;
		}
	}

	indexCount[index] = count_res;
	if (index == tuple_num - 1) {
		indexCount[tuple_num] = 0;
	}
}

__global__ void hashJoin()
{

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
					int tuple_num,
					int col_num,
					GTreeNode *searchKeyExp,
					int *searchKeySize,
					int searchExpNum,
					uint64_t *packedKey,
					uint64_t *bucketLocation,
					uint64_t *hashedIndex,
					int *indexCount,
					int keySize,
					int numberOfBuckets
					)
{
	dim3 gridSize(grid_x, grid_y, 1);
	dim3 blockSize(block_x, block_y, 1);

	hashIndexCount<<<gridSize, blockSize>>>(outer_table, tuple_num, col_num,
											searchKeyExp, searchKeySize, searchExpNum,
											packedKey, bucketLocation,
											hashedIndex, indexCount,
											keySize, numberOfBuckets);
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
						int *indexCount,
						RESULT *result
						);
}
