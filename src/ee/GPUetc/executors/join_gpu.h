#ifndef JOIN_GPU_H_
#define JOIN_GPU_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include "GPUetc/common/GPUTUPLE.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/scan.h>

namespace voltdb {
extern "C" {
void prefixSumFilterWrapper(GTable outer, GTable inner, ulong *count_psum,
								GTree pre_join_pred, GTree join_pred, GTree where_pred);

void prefixSumWrapper(ulong *count_psum, uint gpu_size, ulong *jr_size);

void expFilterWrapper(GTable outer, GTable inner,
						RESULT *jresult_dev, ulong *count_psum,
						GTree pre_join_pred, GTree join_pred, GTree where_pred);
}
}

#endif
