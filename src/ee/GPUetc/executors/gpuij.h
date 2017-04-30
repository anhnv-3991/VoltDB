#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/common/nodedata.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/storage/gtable.h"
#include "GPUetc/expressions/gexpression.h"

namespace voltdb {


class GPUIJ {
public:
	GPUIJ();

	GPUIJ(GTable outer_table,
			GTable inner_table,
			std::vector<TreeExpression> search_idx,
			TreeExpression end_expression,
			TreeExpression post_expression,
			TreeExpression initial_expression,
			TreeExpression skipNullExpr,
			TreeExpression prejoin_expression,
			TreeExpression where_expression,
			IndexLookupType lookup_type);

	~GPUIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	GTable outer_table_, inner_table_;
	GTable outer_chunk_, inner_chunk_;
	GTable search_table_;
	RESULT *join_result_;
	int result_size_;
	int *search_exp_size_, search_exp_num_;
	IndexLookupType lookup_type_;

	GTreeNode *search_exp_;
	GExpression end_expression_;
	GExpression post_expression_;
	GExpression initial_expression_;
	GExpression skipNullExpr_;
	GExpression prejoin_expression_;
	GExpression where_expression_;

	//For profiling
	std::vector<unsigned long> allocation_, prejoin_, index_, expression_, ipsum_, epsum_, wtime_, joins_only_, rebalance_;
	struct timeval all_start_, all_end_;

	void profiling();

	uint getPartitionSize() const;
	bool getTreeNodes(GTree *expression, const TreeExpression tree_expression);
	bool getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void freeArrays2(GTree expression);

	unsigned long timeDiff(struct timeval start, struct timeval end);

	void PrejoinFilter(bool *result);
	void PrejoinFilter(bool *result, cudaStream_t stream);

	void Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size);
	void Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream);

	void IndexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev);

	void IndexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream);

	void ExpressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev);

	void ExpressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream);

	void ExpressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size);

	void ExpressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream);

	void ExpressionFilterShared(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size);

	void ExpressionFilterShared(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream);

	void Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size);
	void Rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
	void Rebalance2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
	void Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream);
};
}

#endif
