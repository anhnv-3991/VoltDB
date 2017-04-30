/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H
#define GPUNIJ_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/storage/gtable.h"

using namespace voltdb;

class GPUNIJ{
public:
	GPUNIJ();

	GPUNIJ(GTable outer_table,
			GTable inner_table,
			TreeExpression pre_join_predicate,
			TreeExpression join_predicate,
			TreeExpression where_predicate);

	~GPUNIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	GTable outer_table_, inner_table_;
	RESULT *join_result_;
	int result_size_;

	GTree pre_join_predicate_;
	GTree join_predicate_;
	GTree where_predicate_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
	bool getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression);
	bool getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue);
	void debugGTrees(const GTree tree);

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
