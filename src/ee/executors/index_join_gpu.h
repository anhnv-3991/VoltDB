#ifndef INDEX_JOIN_GPU_H_
#define INDEX_JOIN_GPU_H_

#include <iostream>
#include <stdio.h>
#include <stdint.h>
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
void prejoin_filterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							uint outer_part_size,
							GTreeNode *prejoin_dev,
							uint prejoin_size,
							bool *result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
							,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
							,int64_t *val_stack,
							ValueType *type_stack
#endif
							);

void index_filterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							GNValue *inner_dev,
							ulong *index_psum,
							ResBound *res_bound,
							uint outer_part_size,
							uint inner_part_size,
							GTreeNode *search_exp_dev,
							int *search_exp_size,
							int search_exp_num,
							int *key_indices,
							int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
							,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
							,int64_t *val_stack,
							ValueType *type_stack
#endif
							);

void exp_filterWrapper(int grid_x, int grid_y,
						int block_x, int block_y,
						GNValue *outer_dev,
						GNValue *inner_dev,
						RESULT *result_dev,
						ulong *index_psum,
						ulong *exp_psum,
						uint outer_part_size,
						uint jr_size,
						GTreeNode *end_dev,
						int end_size,
						GTreeNode *post_dev,
						int post_size,
						GTreeNode *where_dev,
						int where_size,
						ResBound *res_bound,
						int outer_base_idx,
						int inner_base_idx,
						bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
						,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
						,int64_t *val_stack,
						ValueType *type_stack
#endif
						);

void write_outWrapper(int grid_x, int grid_y,
						int block_x, int block_y,
						RESULT *out,
						RESULT *in,
						ulong *count_dev,
						ulong *count_dev2,
						uint outer_part_size,
						uint out_size,
						uint in_size);

void prefix_sumWrapper(ulong *input, int ele_num, ulong *sum);
}

#endif
