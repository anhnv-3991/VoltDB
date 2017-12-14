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
void prefixSumFilterWrapper(int grid_x, int grid_y,
								int block_x, int block_y,
								GNValue *outer_dev, GNValue *inner_dev,
								ulong *count_psum,
								int outer_part_size,
								int inner_part_size,
								GTreeNode *preJoinPred_dev,
								int preJoin_size,
								GTreeNode *joinPred_dev,
								int join_size,
								GTreeNode *where_dev,
								int where_size,
								int64_t *val_stack,
								ValueType *type_stack);

void prefixSumWrapper(ulong *count_psum, uint gpu_size, ulong *jr_size);

void expFilterWrapper(int grid_x, int grid_y,
						int block_x, int block_y,
						GNValue *outer_dev, GNValue *inner_dev,
						RESULT *jresult_dev,
						ulong *count_psum,
						int outer_part_size, int inner_part_size,
						ulong jr_size,
						int outer_base_idx, int inner_base_idx,
						GTreeNode *preJoinPred_dev,
						int preJoin_size,
						GTreeNode *joinPred_dev,
						int join_size,
						GTreeNode *where_dev,
						int where_size,
						int64_t *val_stack,
						ValueType *type_stack);
}
}

#endif
