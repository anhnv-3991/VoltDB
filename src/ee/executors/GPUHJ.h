#ifndef GPUHJ_H_
#define GPUHJ_H_

#include <cuda.h>
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/expressions/nodedata.h"
#include "GPUTUPLE.h"

using namespace voltdb;

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
			IndexLookupType lookup_type);

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
	void setNValue(NValue *nvalue, GNValue &gnvalue);
	void debugGTrees(const GTreeNode *expression, int size);

	void GNValueDebug(GNValue &column_data)	{
		NValue value;
		long double gtmp = column_data.getMdata();
		char tmp[16];
		memcpy(tmp, &gtmp, sizeof(long double));
		value.setMdataFromGPU(tmp);
		//value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug();
	}

};

const uint64_t GPUHJ::MAX_BUCKETS[] = {
	        3,
	        7,
	        13,
	        31,
	        61,
	        127,
	        251,
	        509,
	        1021,
	        2039,
	        4093,
	        8191,
	        16381,
	        32749,
	        65521,
	        131071,
	        262139,
	        524287,
	        1048573,
	        2097143,
	        4194301,
	        8388593,
	        16777213,
	        33554393,
	        67108859,
	        134217689,
	        268435399,
	        536870909,
	        1073741789,
	        2147483647,
	        4294967291,
	        8589934583
	};

#endif