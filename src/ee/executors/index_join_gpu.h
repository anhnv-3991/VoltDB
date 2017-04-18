#ifndef INDEX_JOIN_GPU_H_
#define INDEX_JOIN_GPU_H_

#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/nodedata.h"

namespace voltdb {

extern "C" {
void PrejoinFilterWrapper(GNValue *outer_table, uint outer_rows, uint outer_cols, GTreeNode *prejoin_exp, uint prejoin_size, bool *result);
void PrejoinFilterAsyncWrapper(GNValue *outer_table, uint outer_rows, uint outer_cols, GTreeNode *prejoin_exp, uint prejoin_size, bool *result, cudaStream_t stream);

void IndexFilterWrapper(GNValue *outer_dev, GNValue *inner_dev,
							ulong *index_psum, ResBound *res_bound,
							uint outer_rows, uint outer_cols,
							uint inner_rows, uint inner_cols,
							GTreeNode *search_exp_dev,
							int *search_exp_size, int search_exp_num,
							int *key_indices, int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev);

void IndexFilterAsyncWrapper(GNValue *outer_dev, GNValue *inner_dev,
								ulong *index_psum, ResBound *res_bound,
								uint outer_rows, uint outer_cols,
								uint inner_rows, uint inner_cols,
								GTreeNode *search_exp_dev,
								int *search_exp_size, int search_exp_num,
								int *key_indices, int key_index_size,
								IndexLookupType lookup_type,
								bool *prejoin_res_dev,
								cudaStream_t stream);

void ExpressionFilterWrapper(GNValue *outer_dev, GNValue *inner_dev,
								RESULT *result_dev, ulong *index_psum,
								ulong *exp_psum,
								uint outer_rows,
								uint outer_cols, uint inner_cols,
								uint jr_size,
								GTreeNode *end_dev, int end_size,
								GTreeNode *post_dev, int post_size,
								GTreeNode *where_dev, int where_size,
								ResBound *res_bound,
								int outer_base_idx, int inner_base_idx,
								bool *prejoin_res_dev);

void ExpressionFilterAsyncWrapper(GNValue *outer_dev, GNValue *inner_dev,
									RESULT *result_dev, ulong *index_psum,
									ulong *exp_psum,
									uint outer_rows,
									uint outer_cols, uint inner_cols,
									uint jr_size,
									GTreeNode *end_dev, int end_size,
									GTreeNode *post_dev, int post_size,
									GTreeNode *where_dev, int where_size,
									ResBound *res_bound,
									int outer_base_idx, int inner_base_idx,
									bool *prejoin_res_dev,
									cudaStream_t stream);


void DecomposeWrapper(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size);
void DecomposeAsyncWrapper(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream);

void Rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
void RebalanceAsync(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

void Rebalance2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
void RebalanceAsync2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

void Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
void RebalanceAsync3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

}
}
#endif
