#ifndef GHASH_H_
#define GHASH_H_

#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/storage/gtable.h"

using namespace voltdb;

extern "C" {
void PackKeyWrapper(GTable index_table, int *indices, int index_num, uint64_t *packedKey, int keySize);
void PackKeyAsyncWrapper(GTable index_table, int *indices, int index_num, uint64_t *packedKey, int keySize, cudaStream_t stream);

void GhashWrapper(uint64_t *packedKey, GHashNode hashTable);
void GhashAsyncWrapper(uint64_t *packedKey, GHashNode hashTable, cudaStream_t stream);

void PackSearchKeyWrapper(GTable outer_table, uint64_t *packed_key, GTreeNode *search_key_exp,
							int *search_key_size, int search_exp_num, int key_size);
void PackSearchKeyAsyncWrapper(GTable outer_table, uint64_t *packed_key, GTreeNode *search_key_exp,
								int *search_key_size, int search_exp_num,
								int key_size, cudaStream_t stream);

void IndexCountWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size);
void IndexCountAsyncWrapper(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, int size, cudaStream_t stream);

void IndexCountLegacyWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound);
void IndexCountLegacyAsyncWrapper(uint64_t *outer_key, int outer_rows, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream);

void IndexCountWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound);
void IndexCountAsyncWrapper2(GHashNode outer_hash, GHashNode inner_hash, ulong *index_count, ResBound *out_bound, cudaStream_t stream);

void HashJoinWrapper(GTable outer_table, GTable inner_table,
						GTree end_exp, GTree post_exp,
						GHashNode outer_hash, GHashNode inner_hash,
						ulong *index_count, int size,
						RESULT *result);
void HashJoinAsyncWrapper(GTable outer_table, GTable inner_table,
							GTree end_exp, GTree post_exp,
							GHashNode outer_hash, GHashNode inner_hash,
							ulong *index_count, int size,
							RESULT *result, cudaStream_t stream);

void HashJoinLegacyWrapper(GTable outer_table, GTable inner_table,
							GTree end_exp, GTree post_exp,
							GHashNode inner_hash,
							ulong *index_count,
							ResBound *index_bound,
							RESULT *result);
void HashJoinLegacyAsyncWrapper(GTable outer_table, GTable inner_table,
								GTree end_exp, GTree post_exp,
								GHashNode inner_hash,
								ulong *index_count,
								ResBound *index_bound,
								RESULT *result, cudaStream_t stream);

void HRebalance(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size);
void HRebalanceAsync(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

void HRebalance2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size);
void HRebalanceAsync2(ulong *index_count, ResBound *in_bound, GHashNode inner_hash, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

}

#endif
