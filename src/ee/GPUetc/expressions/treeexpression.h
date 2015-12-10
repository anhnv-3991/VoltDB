#ifndef tree__
#define tree__

#include <stdio.h>
#include <cmath>

#include "common/types.h"
#include "expressions/abstractexpression.h"
#include "expressions/comparisonexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "GPUetc/expressions/nodedata.h"

namespace voltdb {

class TreeExpression {
public:
	TreeExpression();

	TreeExpression(const AbstractExpression *expression) {
		int tree_size = 0;

		tree_size =	getExpressionLength(expression, 1);
		GTreeNode tmp;

		memset(&tmp, 0, sizeof(GTreeNode));

		for (int i = 0; i < tree_size + 1; i++) {
			tree_.push_back(tmp);
		}

		buildTree(expression, 1);
	}

	void debug(void) {
		int tree_size = (int)(tree_.size());

		if (tree_size <= 1) {
			std::cout << "Empty expression" << std::endl;
		}

		for (int index = 1; index < tree_size; index++) {
			switch (tree_[index].type) {
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					std::cout << "[" << index << "] CONJUNCTION AND" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					std::cout << "[" << index << "] CONJUNCTION OR" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					std::cout << "[" << index << "] COMPARE EQUAL" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					std::cout << "[" << index << "] COMPARE NOTEQUAL" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					std::cout << "[" << index << "] COMPARE LESS THAN" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					std::cout << "[" << index << "] COMPARE GREATER THAN" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					std::cout << "[" << index << "] COMPARE LESS THAN OR EQUAL TO" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					std::cout << "[" << index << "] COMPARE GREATER THAN OR EQUAL TO" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LIKE: {
					std::cout << "[" << index << "] COMPARE LIKE" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_IN: {
					std::cout << "[" << index << "] COMPARE IN" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					std::cout << "[" << index << "] TUPLE(";
					std::cout << tree_[index].column_idx << "," << tree_[index].tuple_idx;
					std::cout << ")" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT: {
					NValue tmp;
					GNValue tmp_gnvalue = tree_[index].value;

					setNValue(&tmp, tmp_gnvalue);
					std::cout << "[" << index << "] VALUE TUPLE = " << tmp.debug().c_str()  << std::endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_NULL:
				case EXPRESSION_TYPE_INVALID:
				default: {
					std::cout << "NULL value" << std::endl;
					break;
				}
			}
		}
	}


	int getColNumLeft(void) const {
		int col_num = 0;

		for (int i = 1; i < tree_.size(); i++) {
			if (tree_[i].type == EXPRESSION_TYPE_VALUE_TUPLE && tree_[i].tuple_idx == 0)
				col_num++;
		}

		return col_num;
	}

	int getColNumRight(void) const {
		int col_num = 0;

		for (int i = 1; i < tree_.size(); i++) {
			if (tree_[i].type == EXPRESSION_TYPE_VALUE_TUPLE && tree_[i].tuple_idx == 1)
				col_num++;
		}

		return col_num;
	}

	int getSize(void) const {
		return (int)(tree_.size());
	}

	void getNodesArray(GTreeNode *output) const {
		for (int i = 1; i < tree_.size(); i++) {
			output[i] = tree_[i];
		}
	}

private:
	std::vector<GTreeNode> tree_;

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

	void GNValueDebug(GNValue column_data)
	{
		NValue value;
		value.setMdataFromGPU(column_data.getMdata());
		value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug() << std::endl;
	}

	int getExpressionLength(const AbstractExpression *expression, int root) {
		if (expression == NULL) {
			return 0;
		}

		int left, right;

		if (expression->getLeft() != NULL)
			left = getExpressionLength(expression->getLeft(), root * 2);
		else
			left = 0;

		if (expression->getRight() != NULL)
			right = getExpressionLength(expression->getRight(), root * 2 + 1);
		else
			right = 0;

		if (left == 0 && right == 0)
			return root;
		else if (left > right)
			return left;
		else
			return right;
	}

	bool buildTree(const AbstractExpression *expression, int index) {
		if (expression == NULL) {
			return true;
		}

		if (tree_.size() < index) {
			return false;
		}

		bool res = true;

		tree_[index].type = expression->getExpressionType();

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
				res = buildTree(expression->getLeft(), index * 2);
				if (!res) {
					std::cout << "Error: cannot build left child at index = " << index << std::endl;
					return res;
				}

				res = buildTree(expression->getRight(), index * 2 + 1);
				if (!res) {
					std::cout << "Error: cannot build right child at index = " << index << std::endl;
					return res;
				}

				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				tree_[index].tuple_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getTupleId();
				tree_[index].column_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getColumnId();

				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				NValue nvalue = expression->eval(NULL, NULL);

				setGNValue(&(tree_[index].value), nvalue);

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
};
};

#endif
