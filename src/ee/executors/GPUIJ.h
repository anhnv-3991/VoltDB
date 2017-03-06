#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/expressions/nodedata.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"

namespace voltdb {


class GPUIJ {
public:
	GPUIJ();

	GPUIJ(GNValue *outer_table,
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

	~GPUIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	GNValue *outer_table_, *inner_table_;
	int outer_rows_, inner_rows_, outer_cols_, inner_cols_, outer_size_, inner_size_;
	RESULT *join_result_;
	int *indices_;
	int result_size_;
	int end_size_, post_size_, initial_size_, skipNull_size_, prejoin_size_, where_size_, indices_size_, *search_exp_size_, search_exp_num_;
	IndexLookupType lookup_type_;

	GTreeNode *search_exp_;
	GTreeNode *end_expression_;
	GTreeNode *post_expression_;
	GTreeNode *initial_expression_;
	GTreeNode *skipNullExpr_;
	GTreeNode *prejoin_expression_;
	GTreeNode *where_expression_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
	bool getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression);
	bool getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue);
	void debugGTrees(const GTreeNode *expression, int size);

	unsigned long timeDiff(struct timeval start, struct timeval end)
	{
		return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	}

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

	GNValue evalTest(GTreeNode *tree_expression,
							int root,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple)
	{
		printf("Tree_size = %d, root = %d\n", tree_size, root);
		if (root >= tree_size) {

			return GNValue::getNullValue();
		}
		GNValue left, right;
		GTreeNode tmp_node = tree_expression[root];

		switch (tmp_node.type) {
		case EXPRESSION_TYPE_VALUE_TUPLE: {
			if (tmp_node.tuple_idx == 0) {
				printf("Left value = %d\n", (int)(outer_tuple[tmp_node.column_idx]).getValue());
				return outer_tuple[tmp_node.column_idx];
			}
			else if (tmp_node.tuple_idx == 1) {
				printf("Right value = %d\n", (int)(inner_tuple[tmp_node.column_idx]).getValue());
				return inner_tuple[tmp_node.column_idx];
			}
			break;
		}
		case EXPRESSION_TYPE_VALUE_CONSTANT:
		case EXPRESSION_TYPE_VALUE_PARAMETER: {
			return tmp_node.value;
		}
		case EXPRESSION_TYPE_CONJUNCTION_AND: {
			printf("Operator AND\n");
			//left = GNValue::getTrue();
			//right = GNValue::getTrue();
			left = evalTest(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
			right = evalTest(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);

			return left.op_and(right);
		}
	//	case EXPRESSION_TYPE_CONJUNCTION_OR: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_or(right);
	//	}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			printf("Operator equal\n");
			left = evalTest(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
			right = evalTest(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);

			return left.op_equal(right);
		}
	//	case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_notEqual(right);
	//	}
	//	case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_lessThan(right);
	//	}
	//	case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_lessThanOrEqual(right);
	//	}
	//	case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_greaterThan(right);
	//	}
	//	case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_greaterThanOrEqual(right);
	//	}
		case EXPRESSION_TYPE_OPERATOR_PLUS: {
			printf("Operator plus\n");
			right = evalTest(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);
			left = evalTest(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);

			GNValue test = left.op_add(right);

			printf("test = %d\n", (int)test.getValue());
			return left.op_add(right);
		}
	//	case EXPRESSION_TYPE_OPERATOR_MINUS: {
	//
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_subtract(right);
	//	}
	//	case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_multiply(right);
	//	}
	//	case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
	//		left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//		right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, outer_idx, inner_idx);
	//
	//		return left.op_divide(right);
	//	}
		default: {
			return GNValue::getNullValue();
		}
		}
	}

	GNValue evaluateTest(GTreeNode *tree_expression,
								int root,
								int tree_size,
								GNValue *outer_tuple,
								GNValue *inner_tuple)
	{
		printf("Root = %d\n", root);
		if (root == 0)
			return GNValue::getTrue();

		if (root >= tree_size)
			return GNValue::getNullValue();

		GTreeNode tmp_node = tree_expression[root];

		if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
			if (tmp_node.tuple_idx == 0) {
				printf("Outer tuple at %d is %d\n", tmp_node.column_idx, (int)outer_tuple[tmp_node.column_idx].getValue());
				return outer_tuple[tmp_node.column_idx];
			} else if (tmp_node.tuple_idx == 1) {
				printf("Inner tuple at %d is %d\n", tmp_node.column_idx, (int)inner_tuple[tmp_node.column_idx].getValue());
				return inner_tuple[tmp_node.column_idx];
			}
		} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
			return tmp_node.value;
		}


		GNValue left = evaluateTest(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
		GNValue right = evaluateTest(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);
		GNValue res;


		switch (tmp_node.type) {
		case EXPRESSION_TYPE_CONJUNCTION_AND: {
			res = left.op_and(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator AND\n", root);
			break;
		}
		case EXPRESSION_TYPE_CONJUNCTION_OR: {
			res = left.op_or(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator OR\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			res = left.op_equal(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator EQUAL\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
			res = left.op_notEqual(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator NOTEQUAL\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
			res = left.op_lessThan(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator LESSTHAN\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
			res = left.op_lessThanOrEqual(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator LESSTHANOREQUAL\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
			res = left.op_greaterThan(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator GREATERTHAN\n", root);
			break;
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
			res = left.op_greaterThanOrEqual(right);
			if (!res.isTrue())
				printf("Failed at root = %d operator GREATERTHANOREQUALTO\n", root);
			break;
		}
		case EXPRESSION_TYPE_OPERATOR_PLUS: {
			res = left.op_add(right);
			break;
		}
		case EXPRESSION_TYPE_OPERATOR_MINUS: {
			res = left.op_subtract(right);
			break;
		}
		case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
			res = left.op_multiply(right);
			break;
		}
		case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
			res = left.op_divide(right);
			break;
		}
		default:
			return GNValue::getNullValue();
		}

		return res;
	}

};
}

#endif
