#ifndef GINDEX_H_
#define GINDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/storage/gtuple.h"

namespace voltdb {

class GIndex {
public:
	__forceinline__ __host__ __device__ GIndex();

	virtual void addEntry(GTuple new_tuple) = 0;

	virtual void addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns) = 0;

	virtual void merge(int old_left, int old_right, int new_left, int new_right) = 0;

	virtual void removeIndex() = 0;

	virtual ~GIndex();
};

}

#endif
