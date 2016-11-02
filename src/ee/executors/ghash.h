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
					GNValue *index_table,
					int *indices,
					uint64_t *packedKey
					);

void ghashCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						uint64_t *packedKey,
						int keySize,
						int *hashCount,
						int numberOfBuckets
						);

void ghashWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					uint64_t *packedKey,
					int *hashCount,
					int keySize,
					int numberOfBuckets
					);

void indexCountWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *outer_table,
					GTreeNode *search_key_exp,
					int *hashCount,
					int *indexCount,
					int keySize,
					int numberOfBucket
					);

void hashJoinWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table,
						GNValue *inner_table,
						int outer_cols,
						int inner_cols,
						int *indexCount,
						RESULT *result
						);
}

#endif
