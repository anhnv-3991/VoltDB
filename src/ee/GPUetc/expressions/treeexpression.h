#ifndef TREE_EXPRESSION_
#define TREE_EXPRESSION_

#include <stdio.h>
#include <cmath>

#include "common/types.h"
#include "expressions/abstractexpression.h"
#include "expressions/comparisonexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "GPUetc/expressions/nodedata.h"

using namespace std;

namespace voltdb {

class TreeExpression {
public:
	struct TreeNode {
		ExpressionType type;	//type of
		int column_idx;		//Index of column in tuple, -1 if not tuple value
		int tuple_idx;			//0: left, 1: right
		GNValue value;		// Value of const, = NULL if not const
	};

	TreeExpression();
	CUDAH TreeExpression(const AbstractExpression *expression) {
		int vector_size = length(expression, 1);

		std::vector<TreeNode> tmp_tree_(vector_size + 1);

		for (int i = 1; i < tmp_tree_.size(); i++) {
			tmp_tree_[i].type = EXPRESSION_TYPE_INVALID;
		}

		buildTree(expression, &tmp_tree_, 1);

		tree_ = tmp_tree_;
	}

	TreeExpression(const std::vector<TreeNode> tree) {
		tree_ = tree;
	}

	std::vector<TreeNode> getTree()
	{
		return tree_;
	}

	int getColNumLeft(void) {
		int col_num = 0;

		for (int i = 1; i < tree_.size(); i++) {
			if (tree_[i].type == EXPRESSION_TYPE_VALUE_TUPLE && tree_[i].tuple_idx == 0)
				col_num++;
		}

		return col_num;
	}

	int getColNumRight(void) {
		int col_num = 0;

		for (int i = 1; i < tree_.size(); i++) {
			if (tree_[i].type == EXPRESSION_TYPE_VALUE_TUPLE && tree_[i].tuple_idx == 1)
				col_num++;
		}

		return col_num;
	}

	void debug(void) {
		for (int index = 1; index < tree_.size(); index++) {
			switch (tree_[index].type) {
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					cout << "[" << index << "] CONJUNCTION AND" << endl;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					cout << "[" << index << "] CONJUNCTION OR" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					cout << "[" << index << "] COMPARE EQUAL" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					cout << "[" << index << "] COMPARE NOTEQUAL" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					cout << "[" << index << "] COMPARE LESS THAN" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					cout << "[" << index << "] COMPARE GREATER THAN" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					cout << "[" << index << "] COMPARE LESS THAN OR EQUAL TO" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					cout << "[" << index << "] COMPARE GREATER THAN OR EQUAL TO" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LIKE: {
					cout << "[" << index << "] COMPARE LIKE" << endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_IN: {
					cout << "[" << index << "] COMPARE IN" << endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					cout << "[" << index << "] TUPLE(";
					cout << tree_[index].column_idx << "," << tree_[index].tuple_idx;
					cout << ")" << endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT: {
					NValue tmp;

					setNValue(&tmp, tree_[index].value);
					cout << "[" << index << "] VALUE TUPLE = " << tmp.debug().c_str()  << endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_NULL:
				case EXPRESSION_TYPE_INVALID:
				default: {
					cout << "NULL value" << endl;
					break;
				}
			}
		}
	}

	CUDAH GNValue eval(int root_index, int out_idx, int in_idx, IndexData outer_tuple[], IndexData inner_tuple[], int out_size, int in_size) const {
		if (root_index == 0 || root_index >= tree_.size()) {
			return GNValue::getNullValue();
		}

		if (tree_[root_index].type == EXPRESSION_TYPE_INVALID) {
			return GNValue::getNullValue();
		}

		if (out_idx >= out_size || in_idx >= in_size || out_idx < 0 || in_idx < 0) {
			return GNValue::getNullValue();
		}

		switch (tree_[root_index].type) {
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				return tree_[root_index].value;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_[root_index].tuple_idx == 0) {
					return outer_tuple[out_idx].gn[tree_[root_index].column_idx];
				} else if (tree_[root_index].tuple_idx == 1) {
					return inner_tuple[in_idx].gn[tree_[root_index].column_idx];
				}

				return GNValue::getNullValue();
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if ((eval_left.getValueType() != VALUE_TYPE_BOOLEAN || eval_right.getValueType() != VALUE_TYPE_BOOLEAN)) {
						cout << "Error: CONJUNCTION AND: Wrong operands" << endl;
						return GNValue::getNullValue();
				}

				return eval_left.op_and(eval_right);
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != VALUE_TYPE_BOOLEAN || eval_right.getValueType() != VALUE_TYPE_BOOLEAN) {
						cout << "Error: CONJUNCTION OR: Wrong operands" << endl;
						return GNValue::getNullValue();
				}

				return eval_left.op_or(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE EQUAL: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_equal(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE NOTEQUAL: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_notEqual(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE LESSTHAN: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_lessThan(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE LESSTHANOREQUALTO: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_lessThanOrEqual(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE GREATERTHAN: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_lessThanOrEqual(eval_right);
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				GNValue eval_left, eval_right;

				eval_left = this->eval(root_index * 2, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				eval_right = this->eval(root_index * 2 + 1, out_idx, in_idx, outer_tuple, inner_tuple, out_size, in_size);
				if (eval_left.getValueType() != eval_right.getValueType()) {
						cout << "Error: COMPARE GREATERTHANOREQUALTO: Wrong operands" << endl;

						return GNValue::getNullValue();
				}

				return eval_left.op_greaterThanOrEqual(eval_right);
			}
			default: {
				break;
			}

		}

		return GNValue::getNullValue();
	}

private:
	std::vector<TreeNode> tree_;
	int length(const AbstractExpression *expression, int root) {
		if (expression == NULL) {
			cout << "Empty expression" << endl;
			return 0;
		}

		int left, right;

		if (expression->getLeft() != NULL)
			left = length(expression->getLeft(), root * 2);
		else
			left = 0;

		if (expression->getRight() != NULL)
			right = length(expression->getRight(), root * 2 + 1);
		else
			right = 0;

		if (left == 0 && right == 0)
			return root;
		else if (left > right)
			return left;
		else
			return right;
	}

	bool buildTree(const AbstractExpression *expression, std::vector<TreeNode> *inTree, int index) {
		if (expression == NULL) {
			return true;
		}

		if (inTree->size() < index) {
			return false;
		}

		bool res = true;

		inTree->at(index).type = expression->getExpressionType();

		switch (expression->getExpressionType()) {
			case EXPRESSION_TYPE_CONJUNCTION_AND:
			case EXPRESSION_TYPE_CONJUNCTION_OR:
			case EXPRESSION_TYPE_COMPARE_EQUAL:
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
			case EXPRESSION_TYPE_COMPARE_LESSTHAN:
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
			case EXPRESSION_TYPE_COMPARE_LIKE:
			case EXPRESSION_TYPE_COMPARE_IN:
			case EXPRESSION_TYPE_OPERATOR_PLUS:
			case EXPRESSION_TYPE_OPERATOR_MINUS:
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY:
			case EXPRESSION_TYPE_OPERATOR_CONCAT:
			case EXPRESSION_TYPE_OPERATOR_MOD:
			case EXPRESSION_TYPE_OPERATOR_CAST: {
				res = buildTree(expression->getLeft(), inTree, index * 2);
				if (!res) {
					cout << "Error: cannot build left child at index = " << index << endl;
					return res;
				}

				res = buildTree(expression->getRight(), inTree, index * 2 + 1);
				if (!res) {
					cout << "Error: cannot build right child at index = " << index << endl;
					return res;
				}

				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				inTree->at(index).tuple_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getTupleId();
				inTree->at(index).column_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getColumnId();

				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				NValue nvalue = expression->eval(NULL, NULL);

				setGNValue(&(inTree->at(index).value), nvalue);

				break;
			}
			case EXPRESSION_TYPE_INVALID: {
				res = false;

				break;
			}
			case EXPRESSION_TYPE_VALUE_NULL:
			default: {
				break;
			}
		}

		return res;
	}

	void setGNValue(GNValue *gnvalue, NValue &nvalue)
	{
		gnvalue->setMdata(nvalue.getMdataForGPU());
		gnvalue->setSourceInlined(nvalue.getSourceInlinedForGPU());
		gnvalue->setValueType(nvalue.getValueTypeForGPU());
	}

	void setNValue(NValue *nvalue, GNValue &gnvalue)
	{
		nvalue->setMdataFromGPU(gnvalue.getMdata());
		nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
		nvalue->setValueTypeFromGPU(gnvalue.getValueType());
	}

	void GNValueDebug(GNValue column_data) const
	{
		NValue value;
		value.setMdataFromGPU(column_data.getMdata());
		value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug() << std::endl;
	}
};
}

#endif
