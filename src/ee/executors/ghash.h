#ifndef GHASH_H_
#define GHASH_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

using namespace voltdb;

extern "C" {
void packKeyWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue **index_table,
					int tuple_num,
					int col_num,
					int *indices,
					int index_num,
					uint64_t *packedKey,
					int keySize);

void ghashCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						uint64_t *packedKey,
						int keyNum,
						int keySize,
						uint64_t *hashCount,
						uint64_t maxNumberOfBuckets
						);

void ghashWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					uint64_t *packedKey,
					int keyNum,
					uint64_t *hashCount,
					int keySize,
					int numberOfBuckets,
					uint64_t *hashedIndex,
					uint64_t *bucketLocation
					);

void indexCountWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *outer_table,
					int outer_rows,
					int col_num,
					uint64_t *searchPackedKey,
					GTreeNode *searchKeyExp,
					int *searchKeySize,
					int searchExpNum,
					uint64_t *packedKey,
					uint64_t *bucketLocation,
					uint64_t *hashedIndex,
					int *indexCount,
					int keySize,
					int maxNumberOfBuckets,
					GNValue *stack
					);

void hashJoinWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table,
						GNValue *inner_table,
						int outer_rows,
						int outer_cols,
						int inner_cols,
						GTreeNode *end_expression,
						int end_size,
						GTreeNode *post_expression,
						int post_size,
						uint64_t *searchPackedKey,
						uint64_t *packedKey,
						uint64_t *bucketLocation,
						uint64_t *hashedIndex,
						ulong *indexCount,
						int keySize,
						int maxNumberOfBuckets,
						RESULT *result
						);

void prefixSumWrapper(ulong *input, int ele_num, ulong *sum);
}

#endif
