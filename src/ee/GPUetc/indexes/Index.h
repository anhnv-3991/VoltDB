#ifndef GINDEX_H_
#define GINDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>

namespace voltdb {

class GIndex {
public:
	GIndex();

	void addEntry(int new_tuple_idx);

	void addBatchEntry(int base_idx, int size);

	void merge(int old_left, int old_right, int new_left, int new_right);

	__forceinline__ __device__ int64_t *getTable();

	__forceinline__ __device__ GColumnInfo *getSchema();

	__forceinline__ __device__ int getColumns();

	__forceinline__ __device__ int getRows();

	__forceinline__ __device__ int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx();

	__forceinline__ __device__ int getKeySize();
private:
	int key_num_;	//Number of key values (equal to the number of rows)
	int *sorted_idx_;
	int64_t *packed_key_;
	int *key_idx_;
	int key_size_;	//Number of columns selected as keys
};

}

#endif
