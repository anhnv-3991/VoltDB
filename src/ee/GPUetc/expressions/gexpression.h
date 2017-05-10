#ifndef GEXPRESSION_H_
#define GEXPRESSION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "treeexpression.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GNValue.h"

namespace voltdb {
typedef

class GExpression {
public:
	GExpression() {
		expression_ = NULL;
		size_ = 0;
	}

	GExpression(ExpressionNode *expression) {
		int tree_size = 0;

#ifndef TREE_EVAL_
		tree_size =	getExpressionLength(expression);
		GExpression *tmp_expression = (GExpression*)malloc(sizeof(GExpression) * tree_size);
		checkCudaErrors(cudaMalloc(&expression_, sizeof(GExpression) * tree_size));

		int root = 0;
		tree_ = std::vector<GTreeNode>(tree_size);

		buildPostExpression(tmp_expression, expression, &root);
#else
		int tmp_size = 1;
		tree_size = getTreeSize(expression, tmp_size) + 1;
		GExpression *tmp_expression = (GExpression*)malloc(sizeof(GExpression) * tree_size);
		checkCudaErrors(cudaMalloc(&expression_, sizeof(GExpression) * tree_size));

		buildTreeExpression(tmp_expression, expression, 1);
#endif
		checkCudaErrors(cudaMemcpy(expression_, tmp_expression, sizeof(GExpression) * tree_size, cudaMemcpyHostToDevice));
		free(tmp_expression);
	}

	void freeExpression() {
		if (size_ > 0)
			checkCudaErrors(cudaFree(expression_));
	}

	__forceinline__ __device__ GExpression(GTreeNode *expression, int size) {
		expression_ = expression;
		size_ = size;
	}


	__forceinline__ __device__ int getSize()
	{
		return size_;
	}

	__forceinline__ __device__ GNValue evaluate(int root, int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema)
	{
		if (root == 0)
			return GNValue::getTrue();

		if (root >= size_)
			return GNValue::getNullValue();

		GTreeNode tmp_node = expression_[root];

		if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
			if (tmp_node.tuple_idx == 0) {
				return GNValue(outer_schema[tmp_node.column_idx].data_type, outer_tuple[tmp_node.column_idx]);
			} else if (tmp_node.tuple_idx == 1) {
				return GNValue(inner_schema[tmp_node.column_idx].data_type, inner_tuple[tmp_node.column_idx]);
			}
		} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
			return tmp_node.value;
		}


		GNValue left = evaluate(root * 2, outer_tuple, inner_tuple, outer_schema, inner_schema);
		GNValue right = evaluate(root * 2 + 1, outer_tuple, inner_tuple, outer_schema, inner_schema);

		switch (tmp_node.type) {
		case EXPRESSION_TYPE_CONJUNCTION_AND: {
			return left && right;
		}
		case EXPRESSION_TYPE_CONJUNCTION_OR: {
			return left || right;
		}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			return left == right;
		}
		case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
			return left != right;
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
			return left < right;
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
			return left <= right;
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
			return left > right;
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
			return left >= right;
		}
		case EXPRESSION_TYPE_OPERATOR_PLUS: {
			return left + right;
		}
		case EXPRESSION_TYPE_OPERATOR_MINUS: {
			return left - right;
		}
		case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
			return left * right;
		}
		case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
			return left / right;
		}
		default:
			return GNValue::getNullValue();
		}
	}

	__forceinline__ __device__ GNValue evaluate(int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema, GNValue *stack, int offset)
	{
		int top = 0;

		for (int i = 0; i < size_; i++) {
			GTreeNode tmp = expression_[i];

			switch (tmp.type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp.tuple_idx == 0) {
						stack[top] = GNValue(outer_schema[tmp.column_idx].data_type, outer_tuple[tmp.column_idx]);
						top += offset;
					} else if (tmp.tuple_idx == 1) {
						stack[top] = GNValue(inner_schema[tmp.column_idx].data_type, inner_tuple[tmp.column_idx]);
						top += offset;
					}
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack[top] = tmp.value;
					top += offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					stack[top - 2 * offset] = stack[top - 2 * offset] && stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					stack[top - 2 * offset] = stack[top - 2 * offset] || stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] == stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					stack[top - 2 * offset] = stack[top - 2 * offset] != stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] < stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] <= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					stack[top - 2 * offset] = stack[top - 2 * offset] > stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					stack[top - 2 * offset] = stack[top - 2 * offset] >= stack[top - offset];
					top -= offset;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] + stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					stack[top - 2 * offset] = stack[top - 2 * offset] - stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					stack[top - 2 * offset] = stack[top - 2 * offset] / stack[top - offset];
					top -= offset;

					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					stack[top - 2 * offset] = stack[top - 2 * offset] * stack[top - offset];
					top -= offset;

					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		return stack[0];
	}

	__forceinline__ __device__ GNValue evaluate2(int root, int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema)
	{
		if (root == 0)
			return GNValue::getTrue();

		if (root >= size_)
			return GNValue::getNullValue();

		GTreeNode tmp_node = expression_[root];

		switch (tmp_node.type) {
		case EXPRESSION_TYPE_VALUE_TUPLE: {
			if (tmp_node.tuple_idx == 0) {
				return GNValue(outer_schema[tmp_node.column_idx].data_type, outer_tuple[tmp_node.column_idx]);
			} else if (tmp_node.tuple_idx == 1) {
				return GNValue(inner_schema[tmp_node.column_idx].data_type, inner_tuple[tmp_node.column_idx]);
			} else
				return GNValue::getNullValue();

		}
		case EXPRESSION_TYPE_VALUE_CONSTANT:
		case EXPRESSION_TYPE_VALUE_PARAMETER: {
			return tmp_node.value;
		}
		default:
			break;
		}


		GNValue left = evaluate2(root * 2, outer_tuple, inner_tuple, outer_schema, inner_schema);
		GNValue right = evaluate2(root * 2 + 1, outer_tuple, inner_tuple, outer_schema, inner_schema);
		int64_t left_i = left.getValue(), right_i = right.getValue(), res_i;
		ValueType left_t = left.getValueType(), right_t = right.getValueType(), res_t;


		switch (tmp_node.type) {
		case EXPRESSION_TYPE_CONJUNCTION_AND: {
			assert(left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN);
			res_i = (int64_t)((bool)left_i && (bool)right_i);
			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_CONJUNCTION_OR: {
			assert(left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN);
			res_i = (int64_t)((bool)left_i || (bool)right_i);
			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
			res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));

			return GNValue(VALUE_TYPE_BOOLEAN, res_i);
		}
		case EXPRESSION_TYPE_OPERATOR_PLUS: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d, res_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);

			if (left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) {
				res_d = left_d + right_d;
				res_i = *reinterpret_cast<int64_t *>(&res_d);
				res_t = VALUE_TYPE_DOUBLE;
			} else {
				res_i = left_i + right_i;
				res_t = (left_t > right_t) ? left_t : right_t;
			}

			return GNValue(res_t, res_i);
		}
		case EXPRESSION_TYPE_OPERATOR_MINUS: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d, res_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);

			if (left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) {
				res_d = left_d - right_d;
				res_i = *reinterpret_cast<int64_t *>(&res_d);
				res_t = VALUE_TYPE_DOUBLE;
			} else {
				res_i = left_i - right_i;
				res_t = (left_t > right_t) ? left_t : right_t;
			}

			return GNValue(res_t, res_i);
		}
		case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d, res_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);

			if (left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) {
				res_d = left_d * right_d;
				res_i = *reinterpret_cast<int64_t *>(&res_d);
				res_t = VALUE_TYPE_DOUBLE;
			} else {
				res_i = left_i * right_i;
				res_t = (left_t > right_t) ? left_t : right_t;
			}

			return GNValue(res_t, res_i);
		}
		case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
			assert(left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL);
			double left_d, right_d, res_d;

			left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
			right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);

			if (left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) {
				res_d = (right_d != 0) ? left_d/right_d : 0;
				res_i = *reinterpret_cast<int64_t *>(&res_d);
				res_t = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
			} else {
				res_i = (right_i != 0) ? left_i/right_i : 0;
				res_t = (right_d != 0) ? ((left_t > right_t) ? left_t : right_t) : VALUE_TYPE_INVALID;
			}

			return GNValue(res_t, res_i);
		}
		default:
			return GNValue::getNullValue();
		}
	}

	__forceinline__ __device__ GNValue evaluate(int64_t *outer_tuple, int64_t *inner_tuple, GColumnInfo *outer_schema, GColumnInfo *inner_schema, int64_t *stack, ValueType *gtype, int offset)
	{
		ValueType ltype, rtype;
		int l_idx, r_idx;

		int top = 0;
		double left_d, right_d, res_d;
		int64_t left_i, right_i;

		for (int i = 0; i < size_; i++) {
			GTreeNode *tmp = expression_ + i;

			switch (tmp->type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp->tuple_idx == 0) {
						stack[top] = outer_tuple[tmp->column_idx];
						gtype[top] = outer_schema[tmp->column_idx].data_type;
					} else if (tmp->tuple_idx == 1) {
						stack[top] = inner_tuple[tmp->column_idx];
						gtype[top] = inner_schema[tmp->column_idx].data_type;

					}

					top += offset;
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack[top] = (tmp->value).getValue();
					gtype[top] = (tmp->value).getValueType();
					top += offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
						stack[l_idx] = (int64_t)((bool)(stack[l_idx]) && (bool)(stack[r_idx]));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
						stack[l_idx] = (int64_t)((bool)(stack[l_idx]) || (bool)(stack[r_idx]));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] =  (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
						gtype[r_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;

					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (int64_t)((ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}

				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d + right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i + right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d - right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i - right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d * right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i * right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = (right_d != 0) ? left_d / right_d : 0;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
						} else {
							stack[l_idx] = (right_i != 0) ? left_i / right_i : 0;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
							gtype[l_idx] = (right_i != 0) ? ltype : VALUE_TYPE_INVALID;
						}
					} else {
						stack[l_idx] =  0;
						gtype[r_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		GNValue retval(gtype[0], stack[0]);

		return retval;
	}

	__forceinline__ __device__ GNValue evaluate(GTuple *outer_tuple, GTuple *inner_tuple, int64_t *stack, ValueType *gtype, int offset)
	{
		ValueType ltype, rtype;
		int l_idx, r_idx;

		int top = 0;
		double left_d, right_d, res_d;
		int64_t left_i, right_i;

		for (int i = 0; i < size_; i++) {
			GTreeNode *tmp = expression_ + i;

			switch (tmp->type) {
				case EXPRESSION_TYPE_VALUE_TUPLE: {
					if (tmp->tuple_idx == 0) {
						stack[top] = outer_tuple->tuple_[tmp->column_idx];
						gtype[top] = outer_tuple->schema_[tmp->column_idx].data_type;
					} else if (tmp->tuple_idx == 1) {
						stack[top] = inner_tuple->tuple_[tmp->column_idx];
						gtype[top] = inner_tuple->schema_[tmp->column_idx].data_type;

					}

					top += offset;
					break;
				}
				case EXPRESSION_TYPE_VALUE_CONSTANT:
				case EXPRESSION_TYPE_VALUE_PARAMETER: {
					stack[top] = (tmp->value).getValue();
					gtype[top] = (tmp->value).getValueType();
					top += offset;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_AND: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
						stack[l_idx] = (int64_t)((bool)(stack[l_idx]) && (bool)(stack[r_idx]));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_CONJUNCTION_OR: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					if (gtype[l_idx] == VALUE_TYPE_BOOLEAN && gtype[r_idx] == VALUE_TYPE_BOOLEAN) {
						stack[l_idx] = (int64_t)((bool)(stack[l_idx]) || (bool)(stack[r_idx]));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_EQUAL: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] =  (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
						gtype[r_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;

					break;
				}
				case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] = 0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						stack[l_idx] = (int64_t)((ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
						gtype[l_idx] = VALUE_TYPE_BOOLEAN;
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}

				case EXPRESSION_TYPE_OPERATOR_PLUS: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d + right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i + right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MINUS: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d - right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i - right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = left_d * right_d;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = VALUE_TYPE_DOUBLE;
						} else {
							stack[l_idx] = left_i * right_i;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
						}
					} else {
						stack[l_idx] =  0;
						gtype[l_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
					l_idx = top - 2 * offset;
					r_idx = top - offset;
					ltype = gtype[l_idx];
					rtype = gtype[r_idx];
					if (ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID) {
						left_i = stack[l_idx];
						right_i = stack[r_idx];
						left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
						right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
						res_d = (right_d != 0) ? left_d / right_d : 0;
						if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
							stack[l_idx] = *reinterpret_cast<int64_t *>(&res_d);
							gtype[l_idx] = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
						} else {
							stack[l_idx] = (right_i != 0) ? left_i / right_i : 0;
							gtype[l_idx] = (ltype > rtype) ? ltype : rtype;
							gtype[l_idx] = (right_i != 0) ? ltype : VALUE_TYPE_INVALID;
						}
					} else {
						stack[l_idx] =  0;
						gtype[r_idx] = VALUE_TYPE_INVALID;
					}
					top = r_idx;
					break;
				}
				default: {
					return GNValue::getFalse();
				}
			}
		}

		GNValue retval(gtype[0], stack[0]);

		return retval;
	}

	__forceinline__ __device__ GNValue evaluate(int root, GTuple *outer_tuple, GTuple *inner_tuple)
		{
			if (root == 0)
				return GNValue::getTrue();

			if (root >= size_)
				return GNValue::getNullValue();

			GTreeNode tmp_node = expression_[root];

			if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
				if (tmp_node.tuple_idx == 0) {
					return GNValue(outer_tuple->schema_[tmp_node.column_idx].data_type, outer_tuple->tuple_[tmp_node.column_idx]);
				} else if (tmp_node.tuple_idx == 1) {
					return GNValue(inner_tuple->schema_[tmp_node.column_idx].data_type, inner_tuple->tuple_[tmp_node.column_idx]);
				}
			} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
				return tmp_node.value;
			}


			GNValue left = evaluate(root * 2, outer_tuple, inner_tuple);
			GNValue right = evaluate(root * 2 + 1, outer_tuple, inner_tuple);

			switch (tmp_node.type) {
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				return left && right;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				return left || right;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				return left == right;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				return left != right;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				return left < right;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				return left <= right;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				return left > right;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				return left >= right;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				return left + right;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				return left - right;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				return left * right;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				return left / right;
			}
			default:
				return GNValue::getNullValue();
			}
		}

private:
	int getExpressionLength(ExpressionNode *expression) {
		if (expression == NULL) {
			return 0;
		}

		int left, right;

		left = getExpressionLength(expression->left);

		right = getExpressionLength(expression->right);

		return (1 + left + right);
	}

	int getTreeSize(ExpressionNode *expression, int size) {
		if (expression == NULL)
			return size / 2;

		int left, right;

		left = getTreeSize(expression->left, size * 2);
		right = getTreeSize(expression->right, size * 2 + 1);

		return (left > right) ? left : right;
	}

	bool buildPostExpression(GExpression *output_expression, ExpressionNode *expression, int *index) {
		if (expression == NULL)
			return true;

		if (size_ < *index)
			return false;

		if (!buildPostExpression(output_expression, expression->left, index))
			return false;

		if (!buildPostExpression(output_expression, expression->right, index))
			return false;

		output_expression[*index] = expression->node;
		(*index)++;

		return true;
	}

	bool buildTreeExpression(GExpression *output_expression, ExpressionNode *expression, int index) {
		if (expression == NULL)
			return true;

		if (size_ < *index)
			return false;

		expression_[index] = expression->node;
		if (!buildTreeExpression(output_expression, expression->left, index * 2))
			return false;

		if (!buildTreeEpxression(output_expression, expression->right, index * 2 + 1))
			return false;

		return true;
	}

	GTreeNode *expression_;
	int size_;
};
}

#endif
