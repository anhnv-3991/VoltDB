#ifndef GPUHJ_H_
#define GPUHJ_H_

#include <cuda.h>
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include <sys/time.h>
#include "GPUetc/storage/gtable.h"

namespace voltdb {

class GPUHJ {
public:
	GPUHJ();

	GPUHJ(GTable outer_table,
			GTable inner_table,
			std::vector<TreeExpression> search_idx,
			std::vector<int> indices,
			TreeExpression end_expression,
			TreeExpression post_expression,
			TreeExpression initial_expression,
			TreeExpression skipNullExpr,
			TreeExpression prejoin_expression,
			TreeExpression where_expression,
			IndexLookupType lookup_type,
			int mSizeIndex);

	~GPUHJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

	static const uint64_t MAX_BUCKETS[];
private:
	GTable outer_table_, inner_table_;
	RESULT *join_result_;
	int *indices_;
	int result_size_;
	int indices_size_, *search_exp_size_, search_exp_num_;
	IndexLookupType lookup_type_;
	uint64_t maxNumberOfBuckets_;
	int keySize_;
	int m_sizeIndex_;

	GTreeNode *search_exp_;
	GExpression end_expression_;
	GExpression post_expression_;
	GExpression initial_expression_;
	GExpression skipNullExpr_;
	GExpression prejoin_expression_;
	GExpression where_expression_;

	uint getPartitionSize() const;
	bool getTreeNodes(GTree *expression, const TreeExpression tree_expression);
	bool getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void freeArrays2(GTree expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue)
	{
		int64_t tmp = gnvalue.getMdata();
		char gtmp[16];

		memcpy(gtmp, &tmp, sizeof(int64_t));
		nvalue->setMdataFromGPU(gtmp);
		nvalue->setValueTypeFromGPU(gnvalue.getValueType());
	}

	void PackKey(GTable index_table, int *indices, int index_num, uint64_t *packedKey, int keySize);
	void PackKey(GTable index_table, int *indices, int index_num, uint64_t *packedKey, int keySize, cudaStream_t stream);

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
};

}
#endif
