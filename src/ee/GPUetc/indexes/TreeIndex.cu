#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUetc/indexes/KeyIndex.h"
#include "GPUetc/indexes/TreeIndex.h"

namespace voltdb {
GTreeIndex::GTreeIndex() {
	schema_ = NULL;
	sorted_idx_ = NULL;
	key_idx_ = NULL;
	key_size_ = 0;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void initialize(int *sorted_idx, int base_idx, int left, int right) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index + left; i <= right; i += stride) {
		sorted_idx[i] = i - index + base_idx;
	}
}

extern "C" __global__ void initialize(int64_t *key_list, GColumnInfo *int *sorted_idx, int64_t *table, int tuple_num, GColumnInfo *table_schema, int *key_schema, int key_schema_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < tuple_num; i += stride) {
		sorted_idx[i] = i;
	}
}

GTreeIndex::GTreeIndex(GTable table, int block_id, int *key_idx, int key_size) {
	schema_ = table.getSchema();
	key_idx_ = key_idx;
	key_size_ = key_size;


	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries
	quickSort<<<1, 1>>>(*this, 0, rows_ - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GTreeIndex::addEntry(int new_tuple_idx) {
	int entry_idx;

	upperBoundSearch<<<1, 1>>>(*this, table_ + new_tuple_idx * columns_, &entry_idx);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx + 1, sorted_idx_ + entry_idx, sizeof(int) * (rows_ - entry_idx + 1), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx, &rows_, sizeof(int), cudaMemcpyHostToDevice));
	rows_ += 1;
}

/* Add multiple new indexes.
 * New table are already stored in table_ at indexes started from base_idx.
 *
 * */
void GTreeIndex::addBatchEntry(int base_idx, int size) {
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	initialize<<<grid_x, block_x>>>(sorted_idx_, base_idx, rows_, rows_ + size - 1);
	quickSort<<<grid_x, block_x>>>(*this, rows_, rows_ + size - 1);
	checkCudaErrors(cudaDeviceSynchronize());

	merge(0, rows_ - 1, rows_, rows_ + size - 1);
	rows_ += size;
}


/* Merge new array to the old array
 * Both the new and old arrays are already sorted
 */

void GTreeIndex::merge(int old_left, int old_right, int new_left, int new_right) {
	int old_size, new_size;

	old_size = old_right - old_left + 1;
	new_size = new_right - new_left + 1;

	int block_x, grid_x;

	block_x = (new_size < BLOCK_SIZE_X) ? new_size : BLOCK_SIZE_X;
	grid_x = (new_size - 1) / block_x + 1;

	int *write_location;

	checkCudaErrors(cudaMalloc(&write_location, (old_size + new_size) * sizeof(int)));
	batchSearchUpper<<<grid_x, block_x>>>(*this, new_left, new_right, old_left, old_right, write_location + old_size);
	constructWriteLocation<<<grid_x, block_x>>>(write_location + old_size, new_size);

	block_x = (old_size < BLOCK_SIZE_X) ? old_size : BLOCK_SIZE_X;
	grid_x = (old_size - 1)/block_x + 1;

	batchSearchLower<<<grid_x, block_x>>>(*this, old_left, old_right, new_left, new_right, write_location);
	constructWriteLocation<<<grid_x, block_x>>>(write_location, old_size);

	block_x = (old_size + new_size < BLOCK_SIZE_X) ? (old_size + new_size) : BLOCK_SIZE_X;
	grid_x = (old_size + new_size - 1)/block_x + 1;

	int *new_sorted_idx;

	checkCudaErrors(cudaMalloc(&new_sorted_idx, (old_size + new_size) * sizeof(int)));
	rearrange<<<grid_x, block_x>>>(sorted_idx_, new_sorted_idx, write_location, old_size + new_size);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(sorted_idx_));
	sorted_idx_ = new_sorted_idx;
}

extern "C" {
//Initialize the index array
__global__ void initialize(int *sorted_idx, int base_idx, int left, int right) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index + left; i <= right; i += stride) {
		sorted_idx[i] = i - index + base_idx;
	}
}

//Search for the upper bounds of an array of keys

__global__ void batchSearchUpper(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int64_t *table = indexes.getTable();
	int columns = indexes.getColumns();
	int *key_idx = indexes.getKeyIdx();
	int key_size = indexes.getKeySize();
	GColumnInfo *schema = indexes.getSchema();

	for (int i = index + key_left; i <= key_right; i+= stride) {

		GKeyIndex key(table + i * columns, schema, key_idx, key_size);

		output[i] = indexes.upperBound(key, left, right);
	}
}

//Search for the lower bounds of an array of keys
__global__ void batchSearchLower(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int64_t *table = indexes.getTable();
	int columns = indexes.getColumns();
	int *key_idx = indexes.getKeyIdx();
	int key_size = indexes.getKeySize();
	GColumnInfo *schema = indexes.getSchema();

	for (int i = index + key_left; i <= key_right; i += stride) {
		GKeyIndex key(table + i * columns, schema, key_idx, key_size);

		output[i] = indexes.lowerBound(key, left, right);
	}
}

__global__ void constructWriteLocation(int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += stride) {
		location[i] += i;
	}
}

// Merge the new to the old
__global__ void rearrange(int *input, int *output, int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		output[location[i]] = input[i];
	}
}

__global__ void upperBoundSearch(GTreeIndex indexes, int64_t *tuple, int *entry_idx) {
	GColumnInfo *schema = indexes.getSchema();
	int *key_idx = indexes.getKeyIdx();
	int key_size = indexes.getKeySize();
	int rows = indexes.getRows();

	GKeyIndex key(tuple, schema, key_idx, key_size);

	*entry_idx = indexes.upperBound(key, 0, rows - 1);
}

__global__ void lowerBoundSearch(GTreeIndex indexes, int64_t *tuple, int *entry_idx) {
	GColumnInfo *schema = indexes.getSchema();
	int *key_idx = indexes.getKeyIdx();
	int key_size = indexes.getKeySize();
	int rows = indexes.getRows();

	GKeyIndex key(tuple, schema, key_idx, key_size);

	*entry_idx = indexes.lowerBound(key, 0, rows - 1);
}

//Quick Sort
__global__ void quickSort(GTreeIndex indexes, int left, int right) {
	if (right <= left)
		return;

	int64_t *table = indexes.getTable();
	GColumnInfo *schema = indexes.getSchema();
	int columns = indexes.getColumns();
	int *key_idx = indexes.getKeyIdx();
	int key_size = indexes.getKeySize();
	int *sorted_idx = indexes.getSortedIdx();

	int pivot = (left + right)/2;
	GKeyIndex pivot_key(table + pivot * columns, schema, key_idx, key_size);
	int left_ptr, right_ptr;

	left_ptr = sorted_idx[left];
	right_ptr = sorted_idx[right];

	while (left_ptr <= right_ptr) {
		GTreeIndexKey left_key(table + columns * sorted_idx[left_ptr], schema, key_idx, key_size);
		GTreeIndexKey right_key(table + columns * sorted_idx[right_ptr], schema, key_idx, key_size);

		while (GKeyIndex::KeyComparator(left_key, pivot_key) < 0) {
			left_ptr++;
			left_key.setTuple(table + columns * sorted_idx[left_ptr]);
		}

		while (GKeyIndex::KeyComparator(right_key, pivot_key) > 0) {
			right_ptr--;
			right_key.setTuple(table + columns * sorted_idx[right_ptr]);
		}

		if (left_ptr <= right_ptr) {
			int tmp = sorted_idx[left_ptr];
			sorted_idx[left_ptr] = sorted_idx[right_ptr];
			sorted_idx[right_ptr] = tmp;
		}

	}

	if (left < left_ptr) {
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		quickSort<<<1, 1, 0, s1>>>(indexes, left, right_ptr);
		cudaStreamDestroy(s1);
	}

	if (right > right_ptr) {
		cudaStream_t s2;
		cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
		quickSort<<<1, 1, 0, s2>>>(indexes, left_ptr, right);
		cudaStreamDestroy(s2);
	}
}
}
}
