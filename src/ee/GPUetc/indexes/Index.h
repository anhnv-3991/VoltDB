#ifndef GINDEX_H_
#define GINDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "TreeIndex.h"
#include "HashIndex.h"

namespace voltdb {

class GIndex {
	friend class GTreeIndexKey;
	friend class GHashIndexKey;
public:
	GIndex();

	void addEntry(int new_tuple_idx);

	void addBatchEntry(int base_idx, int size);

	void merge(int old_left, int old_right, int new_left, int new_right);


	__forceinline__ __device__ int getColumns() {
		return key_size_;
	}

	__forceinline__ __device__ int getRows() {
		return key_num_;
	}

	int *getSortedIdx() {
		return sorted_idx_;
	}

	__forceinline__ __device__ int *getKeyIdx() {
		return key_idx_;
	}

	__forceinline__ __device__ int getKeySize() {
		return key_size_;
	}
protected:
	int key_num_;	//Number of key values (equal to the number of rows)
	int *sorted_idx_;
	int *key_idx_;	// Index of columns selected as keys
	int key_size_;	// Number of columns selected as keys
};

}

#endif
