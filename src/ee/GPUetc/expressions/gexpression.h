#ifndef GEXPRESSION_H_
#define GEXPRESSION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/storage/gtuple.h"

namespace voltdb {

class GExpression {
public:
	GExpression();

	/* Create a new expression, allocate the GPU memory for
	 * the expression and convert the input pointer-based
	 * tree expression to the desired expression form.
	 */
	GExpression(ExpressionNode *expression);

	/* Create a new expression from an existing GPU buffer. */
	__forceinline__ __host__ __device__ GExpression(GTreeNode *expression, int size) {
		expression_ = expression;
		size_ = size;
	}

	/* Create an expression from an input pointer-based tree expression */
	bool createExpression(ExpressionNode *expression);

	void freeExpression();

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
			if (left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN) {
				res_i = (int64_t)((bool)left_i && (bool)right_i);
				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_CONJUNCTION_OR: {
			if (left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN) {
				res_i = (int64_t)((bool)left_i || (bool)right_i);
				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
				double left_d, right_d;

				left_d = (left_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (right_t == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_i = (int64_t)((left_t == VALUE_TYPE_DOUBLE || right_t == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));

				return GNValue(VALUE_TYPE_BOOLEAN, res_i);
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_OPERATOR_PLUS: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
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
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_OPERATOR_MINUS: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
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
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
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
			} else
				return GNValue::getInvalid();
		}
		case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
			if (left_t != VALUE_TYPE_INVALID && left_t != VALUE_TYPE_NULL && right_t != VALUE_TYPE_INVALID && right_t != VALUE_TYPE_NULL) {
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
			} else
				return GNValue::getInvalid();
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

	static int getExpressionLength(ExpressionNode *expression);

	static int getTreeSize(ExpressionNode *expression, int size);
private:


	bool buildPostExpression(GTreeNode *output_expression, ExpressionNode *expression, int *index);

	bool buildTreeExpression(GTreeNode *output_expression, ExpressionNode *expression, int index);

	GTreeNode *expression_;
	int size_;
};
}

#endif
