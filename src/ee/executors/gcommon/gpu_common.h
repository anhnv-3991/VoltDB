#include <iostream>
#include <stdint.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/common/nodedata.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>

namespace voltdb {

extern "C" {
__forceinline__ __device__ GNValue EvaluateRecvFunc(GTreeNode *tree_expression, int root, int tree_size, GNValue *outer_tuple, GNValue *inner_tuple)
{
	if (root == 0)
		return GNValue::getTrue();

	if (root >= tree_size)
		return GNValue::getNullValue();

	GTreeNode tmp_node = tree_expression[root];

	if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
		if (tmp_node.tuple_idx == 0) {
			return outer_tuple[tmp_node.column_idx];
		} else if (tmp_node.tuple_idx == 1) {
			return inner_tuple[tmp_node.column_idx];
		}
	} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
		return tmp_node.value;
	}


	GNValue left = EvaluateRecvFunc(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = EvaluateRecvFunc(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);


	switch (tmp_node.type) {
	case EXPRESSION_TYPE_CONJUNCTION_AND: {
		return left.op_and(right);
	}
	case EXPRESSION_TYPE_CONJUNCTION_OR: {
		return left.op_or(right);
	}
	case EXPRESSION_TYPE_COMPARE_EQUAL: {
		return left.op_equal(right);
	}
	case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
		return left.op_notEqual(right);
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
		return left.op_lessThan(right);
	}
	case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
		return left.op_lessThanOrEqual(right);
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
		return left.op_greaterThan(right);
	}
	case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
		return left.op_greaterThanOrEqual(right);
	}
	case EXPRESSION_TYPE_OPERATOR_PLUS: {
		return left.op_add(right);
	}
	case EXPRESSION_TYPE_OPERATOR_MINUS: {
		return left.op_subtract(right);
	}
	case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
		return left.op_multiply(right);
	}
	case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
		return left.op_divide(right);
	}
	default:
		return GNValue::getNullValue();
	}
}

__forceinline__ __device__ GNValue EvaluateItrFunc(GTreeNode *tree_expression, int tree_size, GNValue *outer_tuple, GNValue *inner_tuple, GNValue *stack, int offset)
{
	int top = 0;

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx];
					top += offset;
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx];
					top += offset;
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = tree_expression[i].value;
				top += offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_and(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_or(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_equal(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_notEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_lessThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThan(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_greaterThanOrEqual(stack[top - offset]);
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_add(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_subtract(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_divide(stack[top - offset]);
				top -= offset;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				stack[top - 2 * offset] = stack[top - 2 * offset].op_multiply(stack[top - offset]);
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

__forceinline__ __device__ GNValue EvaluateRecvNonFunc(GTreeNode *tree_expression, int root, int tree_size, GNValue *outer_tuple, GNValue *inner_tuple)
{
	if (root == 0)
		return GNValue::getTrue();

	if (root >= tree_size)
		return GNValue::getNullValue();

	GTreeNode tmp_node = tree_expression[root];

	switch (tmp_node.type) {
	case EXPRESSION_TYPE_VALUE_TUPLE: {
		if (tmp_node.tuple_idx == 0) {
			return outer_tuple[tmp_node.column_idx];
		} else if (tmp_node.tuple_idx == 1) {
			return inner_tuple[tmp_node.column_idx];
		} else
			return GNValue::getNullValue();

	}
	case EXPRESSION_TYPE_VALUE_CONSTANT:
	case EXPRESSION_TYPE_VALUE_PARAMETER: {
		return tmp_node.value;
	}
	}


	GNValue left = EvaluateRecvNonFunc(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = EvaluateRecvNonFunc(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);
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

__forceinline__ __device__ GNValue EvaluateItrNonFunc(GTreeNode *tree_expression, int tree_size, GNValue *outer_tuple, GNValue *inner_tuple, int64_t *stack, ValueType *gtype, int offset)
{
	ValueType ltype, rtype;
	int l_idx, r_idx;

	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i;

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx].getValue();
					gtype[top] = outer_tuple[tree_expression[i].column_idx].getValueType();
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx].getValue();
					gtype[top] = inner_tuple[tree_expression[i].column_idx].getValueType();

				}

				top += offset;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = (tree_expression[i].value).getValue();
				gtype[top] = (tree_expression[i].value).getValueType();
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


void MarkNonZerosWrapper(ulong *input, int size, ulong *output);
void MarkNonZerosAsyncWrapper(ulong *input, int size, ulong *output, cudaStream_t stream);

void RemoveZerosWrapper(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size);
void RemoveZerosAsyncWrapper(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size, cudaStream_t stream);

void MarkTmpLocationWrapper(ulong *tmp_location, ulong *input, int size);
void MarkTmpLocationAsyncWrapper(ulong *tmp_location, ulong *input, int size, cudaStream_t stream);

void MarkLocationWrapper(ulong *location, ulong *input, int size);
void MarkLocationAsyncWrapper(ulong *location, ulong *input, int size, cudaStream_t stream);

void ComputeOffsetWrapper(ulong *input1, ulong *input2, ulong *out, int size);
void ComputeOffsetAsyncWrapper(ulong *input1, ulong *input2, ulong *out, int size, cudaStream_t stream);

void ExclusiveScanWrapper(ulong *input, int ele_num, ulong *sum);
void ExclusiveScanAsyncWrapper(ulong *input, int ele_num, ulong *sum, cudaStream_t stream);

void InclusiveScanWrapper(ulong *input, int ele_num);
void InclusiveScanAsyncWrapper(ulong *input, int ele_num, cudaStream_t stream);

unsigned long timeDiff(struct timeval start, struct timeval end);
void debugGTrees(const GTreeNode *expression, int size);
void RemoveEmptyResultWrapper(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size);
void RemoveEmptyResultAsyncWrapper(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size, cudaStream_t stream);

void RemoveEmptyResultWrapper2(RESULT *out, RESULT *in, ulong *location, int size);
void RemoveEmptyResultAsyncWrapper2(RESULT *out, RESULT *in, ulong *location, int size, cudaStream_t stream);

void ExpressionFilterWrapper2(GNValue *outer_table, GNValue *inner_table,
								RESULT *in_bound, RESULT *out_bound,
								ulong *mark_location, int size,
								uint outer_cols, uint inner_cols,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GTreeNode *where_exp, int where_size,
								int outer_base_idx, int inner_base_idx);

void ExpressionFilterAsyncWrapper2(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx,
									cudaStream_t stream);

void ExpressionFilterWrapper3(GNValue *outer_table, GNValue *inner_table,
								RESULT *in_bound, RESULT *out_bound,
								ulong *mark_location, int size,
								uint outer_cols, uint inner_cols,
								GTreeNode *end_exp, int end_size,
								GTreeNode *post_exp, int post_size,
								GTreeNode *where_exp, int where_size,
								int outer_base_idx, int inner_base_idx);

void ExpressionFilterAsyncWrapper3(GNValue *outer_table, GNValue *inner_table,
									RESULT *in_bound, RESULT *out_bound,
									ulong *mark_location, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_exp, int end_size,
									GTreeNode *post_exp, int post_size,
									GTreeNode *where_exp, int where_size,
									int outer_base_idx, int inner_base_idx, cudaStream_t stream);

void PackKeyWrapper(GNValue *table, int rows, int cols, int *indices, int index_num, uint64_t *key, int key_size);
void PackKeyAsyncWrapper(GNValue *table, int rows, int cols, int *indices, int index_num, uint64_t *key, int key_size, cudaStream_t stream);

void PackSearchKeyWrapper(GNValue *table, int rows, int cols, uint64_t *key, GTreeNode *search_exp, int *exp_size, int exp_num, int key_size);
void PackSearchKeyAsyncWrapper(GNValue *table, int rows, int cols, uint64_t *key, GTreeNode *search_exp, int *exp_size, int exp_num, int key_size, cudaStream_t stream);
}

}
