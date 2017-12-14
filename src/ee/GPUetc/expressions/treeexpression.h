#ifndef tree__
#define tree__

#include <stdio.h>
#include <cmath>

#include "common/types.h"
#include "expressions/abstractexpression.h"
#include "expressions/comparisonexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "GPUetc/common/nodedata.h"


namespace voltdb {

class TreeExpression {
public:
	TreeExpression();

	TreeExpression(const AbstractExpression *expression) {
		int tree_size = 0;

#ifndef TREE_EVAL_
		tree_size =	getExpressionLength(expression);
		//int tmp_size = 1;
		//tree_size =	getTreeSize(expression, tmp_size) + 1;
		int root = 0;
		tree_ = std::vector<GTreeNode>(tree_size);

		buildPostExpression(expression, &root);
		//buildTreeExpression(expression, 1);
#else
		int tmp_size = 1;
		tree_size = getTreeSize(expression, tmp_size) + 1;
		printf("Tree size = %d\n", tree_size);
		tree_ = std::vector<GTreeNode>(tree_size);
		buildTreeExpression(expression, 1);
#endif
	}

	void debug(void) {
		int tree_size = (int)(tree_.size());

		if (tree_size < 1) {
			std::cout << "Empty expression" << std::endl;
		}

		for (int index = 0; index < tree_size; index++) {
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
					std::cout << "[" << index << "] CONSTANT = " << tmp.debug().c_str()  << std::endl;
					break;
				}
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					NValue tmp;
					GNValue tmp_gnvalue = tree_[index].value;

					setNValue(&tmp, tmp_gnvalue);
					std::cout << "[" << index << "] PARAMETER = " << tmp.debug().c_str()  << std::endl;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					std::cout << "[" << index << "]" << "OPERATOR PLUS" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					std::cout << "[" << index << "]" << "OPERATOR MINUS" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					std::cout << "[" << index << "]" << "OPERATOR DIVIDE" << std::endl;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					std::cout << "[" << index << "]" << "OPERATOR MULTIPLY" << std::endl;
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
		for (int i = 0; i < tree_.size(); i++) {
			output[i] = tree_[i];
		}
	}

private:
	std::vector<GTreeNode> tree_;

	void setGNValue(GNValue *gnvalue, NValue &nvalue)
	{
		gnvalue->setMdata(nvalue.getValueTypeForGPU(), nvalue.getMdataForGPU());
//		gnvalue->setSourceInlined(nvalue.getSourceInlinedForGPU());
		gnvalue->setValueType(nvalue.getValueTypeForGPU());
	}

	void setNValue(NValue *nvalue, GNValue &gnvalue)
	{
		long double gtmp = gnvalue.getMdata();
		char tmp[16];
		memcpy(tmp, &gtmp, sizeof(long double));
		nvalue->setMdataFromGPU(tmp);
//		nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
		nvalue->setValueTypeFromGPU(gnvalue.getValueType());
	}

	void GNValueDebug(GNValue column_data)
	{
		NValue value;
		long double gtmp = column_data.getMdata();
		char tmp[16];
		memcpy(tmp, &gtmp, sizeof(long double));
		value.setMdataFromGPU(tmp);
//		value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug() << std::endl;
	}

	int getExpressionLength(const AbstractExpression *expression) {
		if (expression == NULL) {
			return 0;
		}

		int left, right;

		left = getExpressionLength(expression->getLeft());

		right = getExpressionLength(expression->getRight());

		return (1 + left + right);
	}

	int getTreeSize(const AbstractExpression *expression, int size) {
		if (expression == NULL)
			return size / 2;

		int left, right;

		left = getTreeSize(expression->getLeft(), size * 2);
		right = getTreeSize(expression->getRight(), size * 2 + 1);

		return (left > right) ? left : right;
	}

	bool buildPostExpression(const AbstractExpression *expression, int *index) {
		if (expression == NULL) {
			return true;
		}

		if (tree_.size() < *index) {
			return false;
		}

		bool res = true;

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
			case EXPRESSION_TYPE_OPERATOR_DIVIDE:
			case EXPRESSION_TYPE_OPERATOR_CONCAT:
			case EXPRESSION_TYPE_OPERATOR_MOD:
			case EXPRESSION_TYPE_OPERATOR_CAST: {
				res = buildPostExpression(expression->getLeft(), index);
				if (!res) {
					std::cout << "Error: cannot build left child at index = " << *index << std::endl;
					return res;
				}

				res = buildPostExpression(expression->getRight(), index);
				if (!res) {
					std::cout << "Error: cannot build right child at index = " << *index << std::endl;
					return res;
				}

				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				tree_[*index].tuple_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getTupleId();
				tree_[*index].column_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getColumnId();

				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				NValue nvalue = expression->eval(NULL, NULL);

				setGNValue(&(tree_[*index].value), nvalue);

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

		tree_[*index].type = expression->getExpressionType();
		(*index)++;

		return res;
	}

	bool buildTreeExpression(const AbstractExpression *expression, int index) {
		if (expression == NULL) {
			printf("Null expression");
			return true;
		}

		if (tree_.size() < index) {
			printf("Out of range. index = %d. size = %d\n", index, (int)(tree_.size()));
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
			case EXPRESSION_TYPE_OPERATOR_DIVIDE:
			case EXPRESSION_TYPE_OPERATOR_CONCAT:
			case EXPRESSION_TYPE_OPERATOR_MOD:
			case EXPRESSION_TYPE_OPERATOR_CAST: {
				res = buildTreeExpression(expression->getLeft(), index * 2);
				if (!res) {
					std::cout << "Error: cannot build left child at index = " << index * 2 << " type is " << expression->getExpressionType() << std::endl;
					return res;
				}

				res = buildTreeExpression(expression->getRight(), index * 2 + 1);
				if (!res) {
					std::cout << "Error: cannot build right child at index = " << index * 2 + 1 << " type is " << expression->getExpressionType() << std::endl;
					return res;
				}

				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				tree_[index].tuple_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getTupleId();
				tree_[index].column_idx = (dynamic_cast<const TupleValueExpression *>(expression))->getColumnId();

				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
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
