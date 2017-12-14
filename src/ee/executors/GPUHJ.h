#ifndef GPUHJ_H_
#define GPUHJ_H_

#include <cuda.h>
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include <sys/time.h>

namespace voltdb {

class GPUHJ {
public:
	GPUHJ();

	GPUHJ(GNValue *outer_table,
			GNValue *inner_table,
			int outer_rows,
			int outer_cols,
			int inner_rows,
			int inner_cols,
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
	GNValue *outer_table_, *inner_table_;
	int outer_rows_, inner_rows_, outer_cols_, inner_cols_;
	RESULT *join_result_;
	int *indices_;
	int result_size_;
	int end_size_, post_size_, initial_size_, skipNull_size_, prejoin_size_, where_size_, indices_size_, *search_exp_size_, search_exp_num_;
	IndexLookupType lookup_type_;
	uint64_t maxNumberOfBuckets_;
	int keySize_;
	int m_sizeIndex_;

	GTreeNode *search_exp_;
	GTreeNode *end_expression_;
	GTreeNode *post_expression_;
	GTreeNode *initial_expression_;
	GTreeNode *skipNullExpr_;
	GTreeNode *prejoin_expression_;
	GTreeNode *where_expression_;

	uint getPartitionSize() const;
	bool getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression);
	bool getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue)
	{
		int64_t tmp = gnvalue.getMdata();
		char gtmp[16];

		memcpy(gtmp, &tmp, sizeof(int64_t));
		nvalue->setMdataFromGPU(gtmp);
		nvalue->setValueTypeFromGPU(gnvalue.getValueType());
	}
};

}
#endif