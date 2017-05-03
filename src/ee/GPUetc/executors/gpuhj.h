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
	GTable outer_chunk_, inner_chunk_;
	RESULT *join_result_;
	int result_size_;
	int *search_exp_size_, search_exp_num_;
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

	void IndexCount(ulong *index_count, ResBound *out_bound);
	void IndexCount(ulong *index_count, ResBound *out_bound, cudaStream_t stream);

	void HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size);
	void HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream);


	void decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size);
	void decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream);

	void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);

	void Rebalance2(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance2(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
};

}
#endif
