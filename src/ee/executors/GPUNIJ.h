/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H
#define GPUNIJ_H

#include <cuda.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

class GPUNIJ{
public:
	GPUNIJ();

	GPUNIJ(GNValue *outer_table,
			GNValue *inner_table,
			int outer_rows,
			int outer_cols,
			int inner_rows,
			int inner_cols,
			TreeExpression preJoinPredicate,
			TreeExpression joinPredicate,
			TreeExpression wherePredicate);

	~GPUNIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	GNValue *outer_table_, *inner_table_;
	int outer_rows_, inner_rows_, outer_cols_, inner_cols_, outer_size_, inner_size_;
	RESULT *join_result_;
	int result_size_;
	int preJoin_size_, join_size_, where_size_;

	GTreeNode *preJoinPredicate_;
	GTreeNode *joinPredicate_;
	GTreeNode *wherePredicate_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
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
//		value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug();
	}

};

#endif
