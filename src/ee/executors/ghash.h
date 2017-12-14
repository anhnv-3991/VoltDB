#ifndef GHASH_H_
#define GHASH_H_

#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/nodedata.h"

using namespace voltdb;

extern "C" {
void PackKeyWrapper(GNValue *index_table, int tuple_num, int col_num, int *indices, int index_num, uint64_t *packedKey, int keySize);
void PackKeyAsyncWrapper(GNValue *index_table, int tuple_num, int col_num, int *indices, int index_num, uint64_t *packedKey, int keySize, cudaStream_t stream);

void GhashWrapper(uint64_t *packedKey, GHashNode hashTable);
void GhashAsyncWrapper(uint64_t *packedKey, GHashNode hashTable, cudaStream_t stream);

void PackSearchKeyWrapper(GNValue *outer_table, int outer_rows, int outer_cols,
							uint64_t *packed_key, GTreeNode *search_key_exp,
							int *search_key_size, int search_exp_num,
							int key_size);
void PackSearchKeyAsyncWrapper(GNValue *outer_table, int outer_rows, int outer_cols,
								uint64_t *packed_key, GTreeNode *search_key_exp,
								int *search_key_size, int search_exp_num,
								int key_size, cudaStream_t stream);

void IndexCountWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size);
void IndexCountAsyncWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size, cudaStream_t stream);

void IndexCountLegacyWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound);
void IndexCountLegacyAsyncWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream);

void IndexCountWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound);
void IndexCountAsyncWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream);

void HashJoinWrapper(GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_exp, int end_size,
						GTreeNode *post_exp, int post_size,
						GHashNode outer_hash, GHashNode inner_hash,
						int base_outer_idx, int base_inner_idx,
						ulong *index_count, int size,
						RESULT *result);
void HashJoinAsyncWrapper(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							GTreeNode *end_exp, int end_size,
							GTreeNode *post_exp, int post_size,
							GHashNode outer_hash, GHashNode inner_hash,
							int base_outer_idx, int base_inner_idx,
							ulong *index_count, int size,
							RESULT *result, cudaStream_t stream);

void HashJoinLegacyWrapper(GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							int outer_rows,
							GTreeNode *end_exp, int end_size,
							GTreeNode *post_exp, int post_size,
							GHashNode inner_hash,
							int base_outer_idx, int base_inner_idx,
							ulong *index_count,
							ResBound *index_bound,
							RESULT *result);
void HashJoinLegacyAsyncWrapper(GNValue *outer_table, GNValue *inner_table,
								int outer_cols, int inner_cols,
								int outer_rows,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GHashNode inner_hash,
								int base_outer_idx, int base_inner_idx,
								ulong *index_count,
								ResBound *index_bound,
								RESULT *result, cudaStream_t stream);

void HRebalance(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size);
void HRebalanceAsync(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

void HRebalance2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size);
void HRebalanceAsync2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

}

#endif
