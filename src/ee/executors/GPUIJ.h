#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/treeexpression.h"

using namespace voltdb;

#define DEFAULT_PART_SIZE_ (256 * 1024)
class GPUIJ {
public:
	GPUIJ();
	~GPUIJ();

	GPUIJ(IndexData *outer_table,
			IndexData *inner_table,
			int outer_size,
			int inner_size,
			TreeExpression *end_expression,
			TreeExpression *post_expression,
			TreeExpression *initial_expression,
			TreeExpression *skipNullExpr,
			TreeExpression *prejoin_expression,
			TreeExpression *where_expression
			);

	bool join();

private:
	IndexData *outer_table_;
	IndexData *inner_table_;
	int outer_size_, inner_size_;
	RESULT *join_result_;
	int result_size_;

	TreeExpression *end_expression_;
	TreeExpression *post_expression_;
	TreeExpression *initial_expression_;
	TreeExpression *skipNullExpr_;
	TreeExpression *prejoin_expression_;
	TreeExpression *where_expression_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
};

#endif
