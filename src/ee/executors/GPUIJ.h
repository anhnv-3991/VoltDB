#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/expressions/nodedata.h"

using namespace voltdb;

#define DEFAULT_PART_SIZE_ (256 * 1024)
class GPUIJ {
public:
	GPUIJ();

	GPUIJ(IndexData *outer_table,
			IndexData *inner_table,
			int outer_size,
			int inner_size,
			std::vector<int> search_idx,
			std::vector<int> indices,
			TreeExpression end_expression,
			TreeExpression post_expression,
			TreeExpression initial_expression,
			TreeExpression skipNullExpr,
			TreeExpression prejoin_expression,
			TreeExpression where_expression);

	~GPUIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	IndexData *outer_table_;
	IndexData *inner_table_;
	int outer_size_, inner_size_;
	RESULT *join_result_;
	int result_size_;
	int end_size_, post_size_, initial_size_, skipNull_size_, prejoin_size_, where_size_, search_keys_size_, indices_size_;
	int *search_keys_, *indices_;


	GTreeNode *end_expression_;
	GTreeNode *post_expression_;
	GTreeNode *initial_expression_;
	GTreeNode *skipNullExpr_;
	GTreeNode *prejoin_expression_;
	GTreeNode *where_expression_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
	bool getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue);
	void debugGTrees(const GTreeNode *expression, int size);
};

#endif
