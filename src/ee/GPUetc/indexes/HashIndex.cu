#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUetc/indexes/KeyIndex.h"
#include "GPUetc/indexes/HashIndex.h"

namespace voltdb {
GHashIndex::GHashIndex() {
	key_idx_ = NULL;
	key_size_ = 0;
	bucket_locations_ = NULL;
	bucket_num_ = 0;
	packed_key_ = NULL;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries
	checkCudaErrors(cudaGetLastError());
}

GHashIndex::GHashIndex(int key_num, int key_size)
{
	key_idx_ = NULL;
	key_size_ = key_size;
	bucket_locations_ = NULL;
	bucket_num_ = 0;
	key_num_ = key_num;

	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * key_num_));
	checkCudaErrors(cudaMalloc(&packed_key_, sizeof(uint64_t) * key_num_ * key_size_));
	checkCudaErrors(cudaGetLastError());
}


extern "C" __global__ void hashInitialize(GHashIndex table_index, GTable table, int left, int right) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index + left; i <= right; i += stride) {
		GTuple tuple(table, i);

		table_index.insertKeyTupleNoSort(tuple, i);
	}
}


extern "C" __global__ void hashCount(GHashIndex indexes, ulong *hash_count, uint64_t max_buckets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int tuple_num = indexes.getRows();

	for (int i = index; i < tuple_num; i += stride) {
		GHashIndexKey search_key(indexes, i);
		uint64_t hash = search_key.KeyHasher();
		uint64_t bucket_offset = hash % max_buckets;
		hash_count[bucket_offset * stride + index]++;
	}
}

extern "C" __global__ void bucketsLocate(ulong *hash_count, int *bucket_locations, int bucket_num)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i <= bucket_num; i += stride) {
		bucket_location[i] = hash_count[i * stride];
	}
}

extern "C" __global__ void gHash(GHashIndex indexes, ulong *hash_count, int *sorted_idx, int bucket_num, int start_idx, int end_idx)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i <= end_idx - start_idx + 1; i += stride) {
		GHashIndexKey key(indexes, i + start_idx);
		uint64_t hash = key.KeyHasher();
		uint64_t bucket_offset = hash % bucket_num;
		ulong hash_idx = hash_count[bucket_offset * stride + index];

		sorted_idx[hash_idx] = i;

		hash_count[bucket_offset * stride + index]++;
	}
}


GHashIndex::GHashIndex(GTable table, int *key_idx, int key_size, int bucket_num) {

	key_num_ = table.rows_;
	key_idx_ = key_idx;
	key_size_ = key_size;
	bucket_num_ = bucket_num;

	int block_x = (key_size < BLOCK_SIZE_X) ? key_size : BLOCK_SIZE_X;
	int grid_x = (key_size - 1) / block_x + 1;

	checkCudaErrors(cudaMalloc(&bucket_locations_, sizeof(int) * (bucket_num_ + 1)));

	block_x = (key_num_ < BLOCK_SIZE_X) ? key_num_ : BLOCK_SIZE_X;
	grid_x = (key_num_ - 1)/block_x + 1;
	hashInitialize<<<grid_x, block_x>>>(*this, table, 0, key_num_);

	ulong *hash_count;

	checkCudaErrors(cudaMalloc(&hash_count, sizeof(ulong) * (bucket_num_ * block_x * grid_x + 1)));

	ulong total;

	hashCount<<<grid_x, block_x>>>(*this, hash_count, bucket_num_);
	GUtilities::ExclusiveScan(hash_count, bucket_num_ * block_x * grid_x + 1, &total);
	checkCudaErrors(cudaMalloc(&sorted_idx_, sizeof(int) * DEFAULT_PART_SIZE_));	//Default 1024 * 1024 entries

	int *tmp_bucket_location;

	checkCudaErrors(cudaMalloc(&tmp_bucket_locations, sizeof(int) * (bucket_num_ + 1)));

	bucketsLocate<<<grid_x, block_x>>>(*this, tmp_bucket_locations, bucket_num_);
	gHash<<<grid_x, block_x>>>(*this, hash_count, sorted_idx_, bucket_num_, 0, key_num_ - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(hash_count));
}

GHashIndex::~GHashIndex()
{
	if (key_num_ != 0) {
		checkCudaErrors(cudaFree(sorted_idx_));
		checkCudaErrors(cudaFree(bucket_locations_));
		checkCudaErrors(cudaFree(packed_key_));
	}

}

__global__ void hashBucketSearch(GTuple key_tuple, int *bucket_idx)
{
	GHashIndexKey key(key_tuple);

	*bucket_idx = key.KeyHasher();
}

__global__ void hashUpdate(ulong *bucket_location, int bucket_idx, int bucket_num)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < bucket_num - bucket_idx; i += stride) {
		bucket_location[i + bucket_idx + 1]++;
	}
}

void GHashIndex::addEntry(GTuple new_tuple)
{
	int bucket_idx;
	int *dev_bucket_idx;

	checkCudaErrors(cudaMalloc(&dev_bucket_idx, sizeof(int)));

	hashBucketSearch<<<1, 1>>>(new_tuple, dev_bucket_idx);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&bucket_idx, dev_bucket_idx, sizeof(int)));

	ulong copy_location;

	checkCudaErrors(cudaMemcpy(&copy_location, bucket_locations_ + bucket_idx + 1, sizeof(ulong), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(sorted_idx_ + copy_location + 1, sorted_idx_ + copy_location, sizeof(int) * (key_num_ - copy_location + 1), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(sorted_idx_ + copy_location, &key_num_, sizeof(int), cudaMemcpyHostToDevice));
	key_num_ += 1;
}

/* Add multiple new indexes.
 * New table are already stored in table_ at indexes started from base_idx.
 */
void GHashIndex::addBatchEntry(GTable table, int start_idx, int size) {
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	hashInitialize<<<grid_x, block_x>>>(*this, table, start_idx, start_idx + size - 1);

	ulong *hash_count;

	checkCudaErrors(cudaMalloc(&hash_count, sizeof(ulong) * (bucket_num_ * block_x * grid_x + 1)));

	ulong total;

	hashCount<<<grid_x, block_x>>>(*this, hash_count, bucket_num_);
	GUtilities::ExclusiveScan(hash_count, bucket_num_ * block_x * grid_x + 1, &total);

	int *bucket_locations;

	checkCudaErrors(cudaMalloc(&bucket_locations, sizeof(int) * (bucket_num_ + 1)));
	bucketsLocate<<<grid_x, block_x>>>(*this, bucket_locations, bucket_num_);
	gHash<<<grid_x, block_x>>>(*this, hash_count, sorted_idx_, bucket_num_, start_idx, start_idx + size - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	quickSort<<<grid_x, block_x>>>(*this, start_idx, start_idx + size - 1);
	checkCudaErrors(cudaDeviceSynchronize());

	merge(0, key_num_ - 1, key_num_, key_num_ + size - 1, bucket_locations);
	key_num_ += size;
}

extern "C" __global__ void hashSearchUpper(GHashIndex indexes, int key_left, int key_right, int *output, int *bucket_locations)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {

		GHashIndexKey key(indexes, i + key_left);

		uint64_t bucket_idx = key.KeyHasher();

		output[i] = bucket_locations[bucket_idx + 1];
	}
}

extern "C" __global__ void hashSearchLower(GHashIndex indexes, int key_left, int key_right, int *output, int *bucket_locations)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i <= key_right - key_left + 1; i += stride) {
		GHashIndexKey key(indexes, i + key_left);

		uint64_t bucket_idx = key.KeyHasher();

		output[i] = bucket_locations[bucket_idx];
	}
}

extern "C" __global__ void constructHashLocation(int *location, int size)
{
	int index = threaIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += stride) {
		location[i] += i;
	}
}

extern "C" __global__ void hashArrange(int *input, int *output, int *location, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < size; i+= stride) {
		output[location[i]] = input[i];
	}
}
/* Merge new array to the old array
 * Both the new and old arrays are already sorted
 */
void GHashIndex::merge(int old_left, int old_right, int new_left, int new_right, int *bucket_locations) {
	int old_size, new_size;

	old_size = old_right - old_left + 1;
	new_size = new_right - new_left + 1;

	int block_x, grid_x;

	block_x = (new_size < BLOCK_SIZE_X) ? new_size : BLOCK_SIZE_X;
	grid_x = (new_size - 1) / block_x + 1;

	int *write_location;

	checkCudaErrors(cudaMalloc(&write_location, (old_size + new_size) * sizeof(int)));
	hashSearchUpper<<<grid_x, block_x>>>(*this, new_left, new_right, write_location + old_size, bucket_locations);
	constructHashLocation<<<grid_x, block_x>>>(write_location + old_size, new_size);

	block_x = (old_size < BLOCK_SIZE_X) ? old_size : BLOCK_SIZE_X;
	grid_x = (old_size - 1)/block_x + 1;

	batchSearchLower<<<grid_x, block_x>>>(*this, old_left, old_right, new_left, new_right, write_location);
	constructHashLocation<<<grid_x, block_x>>>(write_location, old_size);

	block_x = (old_size + new_size < BLOCK_SIZE_X) ? (old_size + new_size) : BLOCK_SIZE_X;
	grid_x = (old_size + new_size - 1)/block_x + 1;

	int *new_sorted_idx;

	checkCudaErrors(cudaMalloc(&new_sorted_idx, (old_size + new_size) * sizeof(int)));
	hashArrange<<<grid_x, block_x>>>(sorted_idx_, new_sorted_idx, write_location, old_size + new_size);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(sorted_idx_));
	sorted_idx_ = new_sorted_idx;
}

int GHashIndex::getBucketNum()
{
	return bucket_num_;
}

}
