#ifndef KEY_INDEX_H_
#define KEY_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/nodedata.h"

namespace voltdb {
class GKeyIndex {
public:
	__forceinline__ __device__ GKeyIndex();
protected:
	int size_;
};

__forceinline__ __device__ GKeyIndex::GKeyIndex() {
	size_ = 0;
}
}

#endif
