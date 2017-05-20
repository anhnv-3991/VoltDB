#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUetc/indexes/KeyIndex.h"
#include "GPUetc/indexes/TreeIndex.h"

namespace voltdb {
GTreeIndex::GTreeIndex() {
	key_schema_ = NULL;
	sorted_idx_ = NULL;
	key_idx_ = NULL;
	key_size_ = 0;
	key_num_ = 0;
	packed_key_ = NULL;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}


GTreeIndex::GTreeIndex(int key_size, int key_num)
{
	key_size_ = key_size;
	key_num_ = key_num;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * key_num_));
	checkCudaErrors(cudaMalloc(&packed_key_, sizeof(int64_t) * key_num_ * key_size_));
	checkCudaErrors(cudaMalloc(&key_schema_, sizeof(GColumnInfo) * key_size_));
	checkCudaErrors(cudaMalloc(&key_idx_, sizeof(int) * key_size_));
}

GTreeIndex::GTreeIndex(int *sorted_idx, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *key_schema, int key_num)
{
	sorted_idx_ = sorted_idx;
	key_idx_ = key_idx;
	key_size_ = key_size;
	packed_key_ = packed_key;
	key_schema_ = key_schema;
	key_num_ = key_num;
}

extern "C" __global__ void setKeySchema(GColumnInfo *key_schema, GColumnInfo *table_schema, int *key_idx, int key_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < key_size; i += stride) {
		key_schema[i] = table_schema[key_idx[i]];
	}
}

extern "C" __global__ void initialize(GTreeIndex table_index, int64_t *table, GColumnInfo *schema, int columns, int left, int right)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index + left; i <= right; i += stride) {
		GTuple tuple(table + i * columns, schema, columns);

		table_index.insertKeyTupleNoSort(tuple, i);
	}
}


extern "C" __global__ void mergeSort(GTreeIndex table_index, int left, int right, int *input, int *output, int chunk_size)
{
	int half_size = (right - left)/2 + 1;

	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < half_size; i += blockDim.x * gridDim.x) {
		int left_key_ptr = left + i;
		int right_key_ptr = left_key_ptr + half_size;
		int left_root = left + (i/chunk_size) * chunk_size;
		int right_root = left_root + half_size;
		int left_size = chunk_size;
		int right_size = (right_root + chunk_size > right) ? right_root + chunk_size : right - right_root + 1;
		int base_idx = left + (i / chunk_size) * chunk_size * 2;
		int local_idx = left_key_ptr - left_root;

		GTreeIndexKey left_key = table_index.getKeyAtSortedIndex(left_key_ptr);
		GTreeIndexKey right_key = table_index.getKeyAtSortedIndex(right_key_ptr);
		int new_left_key_ptr = table_index.lowerBound(left_key, right_root, right_root + right_size - 1);
		int new_right_key_ptr = table_index.upperBound(right_key, left_root, left_root + left_size - 1);

		new_left_key_ptr = (new_left_key_ptr != -1) ? (new_left_key_ptr - left_key_ptr + local_idx) : (-1);
		new_right_key_ptr = (new_right_key_ptr != -1) ? (new_right_key_ptr - right_key_ptr + local_idx) : (-1);

		output[new_left_key_ptr + base_idx] = input[left_key_ptr];
		output[new_right_key_ptr + base_idx] = input[right_key_ptr];
	}
}

void GTreeIndex::createIndex(int64_t *table, GColumnInfo *schema, int rows, int columns)
{
	key_num_ = rows;

	int block_x = (key_num_ < BLOCK_SIZE_X) ? key_num_ : BLOCK_SIZE_X;
	int grid_x = (key_num_ - 1) / block_x + 1;

	setKeySchema<<<grid_x, block_x>>>(key_schema_, schema, key_idx_, key_size_);

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	block_x = (key_num_ < BLOCK_SIZE_X) ? key_num_ : BLOCK_SIZE_X;
	grid_x = (key_num_ - 1)/block_x + 1;
	initialize<<<grid_x, block_x>>>(current_index, table, schema, columns, 0, key_num_ - 1);

	int *tmp_sorted_idx, *tmp;

	checkCudaErrors(cudaMalloc(&tmp_sorted_idx, sizeof(int) * key_num_));

	for (int chunk_size = 1; chunk_size <= key_num_/2; chunk_size <<= 1) {
		mergeSort<<<grid_x, block_x>>>(*this, 0, key_num_ - 1, sorted_idx_, tmp_sorted_idx, chunk_size);
		checkCudaErrors(cudaDeviceSynchronize());

		tmp = sorted_idx_;
		sorted_idx_ = tmp_sorted_idx;
		tmp_sorted_idx = sorted_idx_;
	}

	checkCudaErrors(cudaFree(tmp_sorted_idx));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" __global__ void upperBoundSearch(GTreeIndex indexes, GTuple new_tuple, int *key_schema, int key_size, int *entry_idx)
{
	int key_num = indexes.getKeyNum();
	GTreeIndexKey key = indexes.getKeyAtIndex(key_num);

	key.createKey(new_tuple, key_schema, key_size);

	*entry_idx = indexes.upperBound(key, 0, key_num - 1);
}

extern "C" __global__ void lowerBoundSearch(GTreeIndex indexes, GTuple new_tuple, int *key_schema, int key_size, int *entry_idx)
{
	int key_num = indexes.getKeyNum();
	GTreeIndexKey key = indexes.getKeyAtIndex(key_num);

	key.createKey(new_tuple, key_schema, key_size);

	*entry_idx = indexes.lowerBound(key, 0, key_num - 1);
}

void GTreeIndex::addEntry(GTuple new_tuple) {
	int entry_idx, *entry_idx_dev;

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	checkCudaErrors(cudaMalloc(&entry_idx_dev, sizeof(int)));
	upperBoundSearch<<<1, 1>>>(current_index, new_tuple, key_idx_, key_size_, entry_idx_dev);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&entry_idx, entry_idx_dev, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx + 1, sorted_idx_ + entry_idx, sizeof(int) * (key_num_ - entry_idx + 1), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + entry_idx, &key_num_, sizeof(int), cudaMemcpyHostToDevice));
	key_num_++;
}

/* Add multiple new indexes.
 * New table are already stored in table_ at indexes started from base_idx.
 *
 * */
void GTreeIndex::addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns)
{
	GTreeIndex new_index(sorted_idx_ + key_num_, key_idx_, key_size_, packed_key_ + key_num_ * key_size_, key_schema_, rows);

	new_index.createIndex(table, schema, rows, columns);

	merge(0, key_num_ - 1, key_num_, key_num_ + rows - 1);
	key_num_ += rows;
}

extern "C" __global__ void batchSearchUpper(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTreeIndexKey key;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {
		key = indexes.getKeyAtIndex(i + key_left);

		output[i] = indexes.upperBound(key, left, right);
	}
}

//Search for the lower bounds of an array of keys
extern "C" __global__ void batchSearchLower(GTreeIndex indexes, int key_left, int key_right, int left, int right, int *output) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTreeIndexKey key;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {
		key = indexes.getKeyAtIndex(i + key_left);

		output[i] = indexes.lowerBound(key, left, right);
	}
}

extern "C" __global__ void constructWriteLocation(int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += stride) {
		location[i] += i;
	}
}

extern "C" __global__ void rearrange(int *input, int *output, int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		output[location[i]] = input[i];
	}
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

	GTreeIndex current_index(sorted_idx_, key_idx_, key_size_, packed_key_, key_schema_, key_num_);

	int *write_location;

	checkCudaErrors(cudaMalloc(&write_location, (old_size + new_size) * sizeof(int)));
	batchSearchUpper<<<grid_x, block_x>>>(current_index, new_left, new_right, old_left, old_right, write_location + old_size);
	constructWriteLocation<<<grid_x, block_x>>>(write_location + old_size, new_size);

	block_x = (old_size < BLOCK_SIZE_X) ? old_size : BLOCK_SIZE_X;
	grid_x = (old_size - 1)/block_x + 1;

	batchSearchLower<<<grid_x, block_x>>>(current_index, old_left, old_right, new_left, new_right, write_location);
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

void GTreeIndex::removeIndex() {
	checkCudaErrors(cudaFree(sorted_idx_));
	checkCudaErrors(cudaFree(key_idx_));
}
}
