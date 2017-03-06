#include "index_join_gpu.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/nodedata.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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
#include <thrust/execution_policy.h>

namespace voltdb {

/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/

extern "C" {

__forceinline__ __device__ GNValue EvaluateRecursive(GTreeNode *tree_expression,
														int root,
														int tree_size,
														GNValue *outer_tuple,
														GNValue *inner_tuple)
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


	GNValue left = EvaluateRecursive(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = EvaluateRecursive(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);


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

__forceinline__ __device__ GNValue EvaluateIterative(GTreeNode *tree_expression,
														int tree_size,
														GNValue *outer_tuple,
														GNValue *inner_tuple,
														GNValue *stack,
														int offset)
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

__forceinline__ __device__ GNValue EvaluateRecursive2(GTreeNode *tree_expression,
														int root,
														int tree_size,
														GNValue *outer_tuple,
														GNValue *inner_tuple)
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


	GNValue left = EvaluateRecursive2(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = EvaluateRecursive2(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);
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

__forceinline__ __device__ GNValue EvaluateIterative2(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int64_t *stack,
							ValueType *gtype,
							int offset)
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

__forceinline__ __device__ int LowerBound(GTreeNode * search_exp, int *search_exp_size, int search_exp_num,
											int * key_indices, int key_index_size,
											GNValue *outer_table, GNValue *inner_table,
											int outer_cols, int inner_cols,
											int left, int right,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
											GNValue *stack,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
											int64_t *val_stack,
											ValueType *type_stack,
#endif
											int offset)
{
	int middle = -1;
	int search_ptr;
	int result = -1;
	int inner_idx;

#ifndef FUNC_CALL_
	int key_idx;
	int64_t outer_tmp[8], inner_tmp;
	ValueType outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	GNValue tmp;

	res_i = -1;
	res_d = -1;
#else
	GNValue outer_tmp[8];
	int res;
#endif

	search_ptr = 0;
	for (int i = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
#ifndef FUNC_CALL_
#ifdef POST_EXP_
		tmp = EvaluateIterative2(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = EvaluateRecursive2(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
		outer_tmp[i] = EvaluateIterative(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = EvaluateRecursive(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
			key_idx = key_indices[i];

			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
#else
		res = 0;
		for (int i = 0; (res == 0) && (i < search_exp_num); i++) {
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);
		}
#endif


#ifndef FUNC_CALL_
		right = (res_i <= 0 && res_d <= 0) ? (middle - 1) : right;	//move to left
		left = (res_i > 0 || res_d > 0) ? (middle + 1) : left;		//move to right
		result = (res_i <= 0 && res_d <= 0) ? middle : result;
#else
		right = (res == VALUE_COMPARE_GREATERTHAN) ? right : (middle - 1);
		left = (res == VALUE_COMPARE_GREATERTHAN) ? (middle + 1) : left;
		result = (res == VALUE_COMPARE_GREATERTHAN) ? result : middle;
#endif
	}
	return result;
}

__forceinline__ __device__ int UpperBound(GTreeNode * search_exp, int *search_exp_size, int search_exp_num,
											int * key_indices, int key_index_size,
											GNValue *outer_table, GNValue *inner_table,
											int outer_cols, int inner_cols,
											int left, int right,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
											GNValue *stack,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
											int64_t *val_stack,
											ValueType *type_stack,
#endif
											int offset)
{
	int middle = -1;
	int search_ptr;
	int result = right;
	int inner_idx;

#ifndef FUNC_CALL_
	int key_idx;
	int64_t outer_tmp[8], inner_tmp;
	ValueType outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	GNValue tmp;

	res_i = -1;
	res_d = -1;
#else
	GNValue outer_tmp[8];
	int res;
#endif
	search_ptr = 0;
	for (int i = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
#ifndef FUNC_CALL_
#ifdef POST_EXP_
		tmp = EvaluateIterative2(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = EvaluateRecursive2(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
		outer_tmp[i] = EvaluateIterative(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = EvaluateRecursive(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {

			key_idx = key_indices[i];
			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
#else
		res = 0;
		for (int i = 0; res == 0 && i < search_exp_num; i++) {
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);
		}
#endif


#ifndef FUNC_CALL_
		right = (res_i < 0 || res_d < 0) ? (middle - 1) : right;	//move to left
		left = (res_i >= 0 && res_d >= 0) ? (middle + 1) : left;		//move to right
		result = (res_i < 0 || res_d < 0) ? middle : result;
#else
		right = (res == VALUE_COMPARE_LESSTHAN) ? (middle - 1) : right;
		left = (res == VALUE_COMPARE_LESSTHAN) ? left : (middle + 1);
		result = (res == VALUE_COMPARE_LESSTHAN) ? middle : result;
#endif
	}

	return result;
}


__global__ void PrejoinFilter(GNValue *outer_dev,
								uint outer_part_size,
								uint outer_cols,
								GTreeNode *prejoin_dev,
								uint prejoin_size,
								bool *result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								,int64_t *val_stack,
								ValueType *type_stack
#endif
								)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < outer_part_size) {
		GNValue res = GNValue::getTrue();

#ifdef 	TREE_EVAL_
#ifdef FUNC_CALL_
		res = (prejoin_size > 1) ? EvaluateRecursive(prejoin_dev, 1, prejoin_size, outer_dev + x, NULL) : res;
#else
		res = (prejoin_size > 1) ? EvaluateRecursive2(prejoin_dev, 1, prejoin_size, outer_dev + x, NULL) : res;
#endif
#elif	POST_EXP_
		int offset = blockDim.x * gridDim.x;

#ifndef FUNC_CALL_
		res = (prejoin_size > 1) ? EvaluateIterative2(prejoin_dev, prejoin_size, outer_dev + x * outer_cols, NULL, val_stack + x, type_stack + x, offset) : res;
#else
		res = (prejoin_size > 1) ? EvaluateIterative(prejoin_dev, prejoin_size, outer_dev + x * outer_cols, NULL, stack + x, offset) : res;
#endif
#endif
		result[x] = res.isTrue();
	}
}


__global__ void IndexFilterLowerBound(GNValue *outer_dev,
										  GNValue *inner_dev,
										  ulong *index_psum,
										  ResBound *res_bound,
										  uint outer_part_size,
										  uint outer_cols,
										  uint inner_part_size,
										  uint inner_cols,
										  GTreeNode *search_exp_dev,
										  int *search_exp_size,
										  int search_exp_num,
										  int *key_indices,
										  int key_index_size,
										  IndexLookupType lookup_type,
										  bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
										  ,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
										  ,int64_t *val_stack,
										  ValueType *type_stack
#endif
										  )

{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);

	int offset = blockDim.x * gridDim.x;

	res_bound[x + k].left = -1;

	if (x < outer_part_size && prejoin_res_dev[x]) {
		res_bound[x + k].outer = x;
		switch (lookup_type) {
		case INDEX_LOOKUP_TYPE_EQ:
		case INDEX_LOOKUP_TYPE_GT:
		case INDEX_LOOKUP_TYPE_GTE:
		case INDEX_LOOKUP_TYPE_LT: {
			res_bound[x + k].left = LowerBound(search_exp_dev,
												search_exp_size,
												search_exp_num,
												key_indices,
												key_index_size,
												outer_dev + x * outer_cols,
												inner_dev,
												outer_cols,
												inner_cols,
												left_bound,
												right_bound,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												stack + x,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL))
												val_stack + x,
												type_stack + x,
#endif
												offset);
			break;
		}
		case INDEX_LOOKUP_TYPE_LTE: {
			res_bound[x + k].left = left_bound;
			break;
		}
		default:
			break;
		}
	}
}

__global__ void IndexFilterUpperBound(GNValue *outer_dev,
										  GNValue *inner_dev,
										  ulong *index_psum,
										  ResBound *res_bound,
										  uint outer_part_size,
										  uint outer_cols,
										  uint inner_part_size,
										  uint inner_cols,
										  GTreeNode *search_exp_dev,
										  int *search_exp_size,
										  int search_exp_num,
										  int *key_indices,
										  int key_index_size,
										  IndexLookupType lookup_type,
										  bool *prejoin_res_dev
#if (defined(POST_EXP_) && !defined(FUNC_CALL_))
										  ,int64_t *val_stack,
										  ValueType *type_stack
#elif (defined(POST_EXP_) && defined(FUNC_CALL_))
										  ,GNValue *stack
#endif
										  )

{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);

	int offset = blockDim.x * gridDim.x;

	index_psum[x + k] = 0;
	res_bound[x + k].right = -1;

	if (x < outer_part_size && prejoin_res_dev[x]) {
		switch (lookup_type) {
		case INDEX_LOOKUP_TYPE_EQ:
		case INDEX_LOOKUP_TYPE_LTE: {
			res_bound[x + k].right = UpperBound(search_exp_dev,
													search_exp_size,
													search_exp_num,
													key_indices,
													key_index_size,
													outer_dev + x * outer_cols,
													inner_dev,
													outer_cols,
													inner_cols,
													left_bound,
													right_bound,
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
													stack + x,
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
													val_stack + x,
													type_stack + x,
#endif
													offset);
			break;
		}
		case INDEX_LOOKUP_TYPE_GT:
		case INDEX_LOOKUP_TYPE_GTE: {
			res_bound[x + k].right = right_bound;
			break;
		}
		case INDEX_LOOKUP_TYPE_LT: {
			res_bound[x + k].right = res_bound[x + k].left - 1;
			res_bound[x + k].left = left_bound;
			break;
		}
		default:
			break;
		}

		index_psum[x + k] = (res_bound[x + k].right >= 0 && res_bound[x + k].left >= 0) ? (res_bound[x + k].right - res_bound[x + k].left + 1) : 0;
	}


	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		index_psum[x + k + 1] = 0;
	}
}

__global__ void ExpressionFilter(GNValue *outer_dev,
									GNValue *inner_dev,
									RESULT *result_dev,
									ulong *index_psum,
									ulong *exp_psum,
									uint outer_part_size,
									uint outer_cols,
									uint inner_cols,
									uint jr_size,
									GTreeNode *end_dev,
									int end_size,
									GTreeNode *post_dev,
									int post_size,
									GTreeNode *where_dev,
									int where_size,
									ResBound *res_bound,
									int outer_base_idx,
									int inner_base_idx,
									bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
									,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
									,int64_t *val_stack,
									ValueType *type_stack
#endif
							)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;

	exp_psum[x + k] = 0;

	if (x < outer_part_size) {
		ulong writeloc = index_psum[x + k];
		int count = 0;
		int res_left = -1, res_right = -1;
		GNValue res = GNValue::getTrue();

		res_left = res_bound[x + k].left;
		res_right = res_bound[x + k].right;

		while (res_left >= 0 && res_left <= res_right && writeloc < jr_size) {
#ifdef	TREE_EVAL_
#ifdef FUNC_CALL_
			res = (end_size > 1) ? EvaluateRecursive(end_dev, 1, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? EvaluateRecursive(post_dev, 1, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
#else
			res = (end_size > 1) ? EvaluateRecursive2(end_dev, 1, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? EvaluateRecursive2(post_dev, 1, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
#endif

#elif	POST_EXP_
			int offset = blockDim.x * gridDim.x;

#ifdef 	FUNC_CALL_
			res = (end_size > 0) ? EvaluateIterative(end_dev, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols, stack + x, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? EvaluateIterative(post_dev, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols, stack + x, offset) : res;
#else
			res = (end_size > 0) ? EvaluateIterative2(end_dev, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols, val_stack + x, type_stack + x, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? EvaluateIterative2(post_dev, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols, val_stack + x, type_stack + x, offset) : res;
#endif
#endif
			result_dev[writeloc].lkey = (res.isTrue()) ? (x + outer_base_idx) : (-1);
			result_dev[writeloc].rkey = (res.isTrue()) ? (res_left + inner_base_idx) : (-1);
			count += (res.isTrue()) ? 1 : 0;
			writeloc++;
			res_left++;
		}
		exp_psum[x + k] = count;
	}

	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		exp_psum[x + k + 1] = 0;
	}
}

__global__ void ExpressionFilter2(GNValue *outer_dev, GNValue *inner_dev,
									RESULT *in, RESULT *out,
									ulong *exp_psum, int size,
									uint outer_cols, uint inner_cols,
									GTreeNode *end_dev, int end_size,
									GTreeNode *post_dev, int post_size,
									GTreeNode *where_dev, int where_size,
									int outer_base_idx, int inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
									,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
									,int64_t *val_stack, ValueType *type_stack
#endif
		)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GNValue res = GNValue::getTrue();
	//int count = 0;

	for (int i = index; i < size; i += offset) {
#ifdef TREE_EVAL_
#ifdef FUNC_CALL_
		res = (end_size > 1) ? EvaluateRecursive(end_dev, 1, end_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecursive(post_dev, 1, post_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols) : res;
#else
		res = (end_size > 1) ? EvaluateRecursive2(end_dev, 1, end_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateRecursive2(post_dev, 1, post_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols) : res;
#endif
#else
#ifdef FUNC_CALL_
		res = (end_size > 1) ? EvaluateIterative(end_dev, end_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols, stack + index, offset) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateIterative(post_dev, post_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols, stack + index, offset) : res;
#else
		res = (end_size > 1) ? EvaluateIterative2(end_dev, end_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols, val_stack + index, type_stack + index, offset) : res;
		res = (post_size > 1 && res.isTrue()) ? EvaluateIterative2(post_dev, post_size, outer_dev + in[i].lkey * outer_cols, inner_dev + in[i].rkey * inner_cols, val_stack + index, type_stack + index, offset) : res;
#endif
		out[i].lkey = (res.isTrue()) ? (in[i].lkey + outer_base_idx) : (-1);
		out[i].rkey = (res.isTrue()) ? (in[i].rkey + inner_base_idx) : (-1);
		//count += (res.isTrue()) ? 1 : 0;
		exp_psum[i] = (res.isTrue()) ? 1 : 0;
#endif
	}

	//exp_psum[index] = count;

	if (index == 0) {
		exp_psum[size] = 0;
	}
}


__global__ void RemoveEmptyResult(RESULT *out, RESULT *in,
									ulong *count_dev, ulong *count_dev2,
									uint outer_part_size,
									uint out_size, uint in_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	ulong writeloc_out = count_dev2[x + k];
	ulong readloc_in = count_dev[x + k];
	ulong num = count_dev[x + k + 1] - count_dev[x + k];
	int lkey, rkey;
	ulong i = 0;

	if (x < outer_part_size) {
		while (i < num) {
			lkey = (readloc_in + i < (ulong)in_size) ? in[readloc_in + i].lkey : (-1);
			rkey = (readloc_in + i < (ulong)in_size) ? in[readloc_in + i].rkey : (-1);

			if (lkey != -1 && rkey != -1) {
				out[writeloc_out].lkey = lkey;
				out[writeloc_out].rkey = rkey;
				writeloc_out++;
			}
			i++;
		}
	}
}

__global__ void RemoveEmptyResult2(RESULT *out, RESULT *in, ulong *location, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tmp_location;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location = location[i];
		if (in[i].lkey != -1 && in[i].rkey != -1) {
			out[tmp_location].lkey = in[i].lkey;
			out[tmp_location].rkey = in[i].rkey;
		}
	}
}

__global__ void MarkNonZeros(ulong *input, int size, ulong *mark)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i;

	for (i = index; i < size; i += blockDim.x * gridDim.x) {
		mark[i] = (input[i] != 0) ? 1 : 0;
	}

	if (i == size)
		mark[i] = 0;
}

__global__ void RemoveZeros(ulong *input, ulong *output, ResBound *in_bound, ResBound *out_bound, ulong *mark, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		if (input[i] != 0) {
			output[mark[i]] = input[i];
			out_bound[mark[i]] = in_bound[i];
		}
	}
}

__global__ void MarkTmpLocation(ulong *tmp_location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location[input[i]] = (i != 0) ? 1 : 0;
	}
}

__global__ void ComputeOffset(ulong *input1, ulong *input2, ulong *out, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i] = i - input1[input2[i]];
	}
}

__global__ void Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i].lkey = in[in_location[i]].outer;
		out[i].rkey = in[in_location[i]].left + local_offset[i];
	}
}

void PrejoinFilterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							uint outer_part_size,
							uint outer_cols,
							GTreeNode *prejoin_dev,
							uint prejoin_size,
							bool *result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
							, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
							, int64_t *val_stack
							, ValueType *type_stack
#endif
							)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	PrejoinFilter<<<grid_size, block_size>>>(outer_dev, outer_part_size, outer_cols, prejoin_dev, prejoin_size, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,val_stack,
												type_stack
#endif
												);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error: Async kernel (PrejoinFilter) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void IndexFilterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							GNValue *inner_dev,
							ulong *index_psum,
							ResBound *res_bound,
							uint outer_part_size,
							uint outer_cols,
							uint inner_part_size,
							uint inner_cols,
							GTreeNode *search_exp_dev,
							int *search_exp_size,
							int search_exp_num,
							int *key_indices,
							int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
							, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
							, int64_t *val_stack
							, ValueType *type_stack
#endif
							)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	IndexFilterLowerBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_part_size, outer_cols,
														inner_part_size, inner_cols,
														search_exp_dev, search_exp_size,
														search_exp_num, key_indices,
														key_index_size, lookup_type,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
#endif
														);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error: Async kernel (IndexFilterLowerBound) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());

	IndexFilterUpperBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_part_size, outer_cols,
														inner_part_size, inner_cols,
														search_exp_dev, search_exp_size,
														search_exp_num, key_indices,
														key_index_size, lookup_type,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
#endif
														);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error: Async kernel (IndexFilterUpperBound) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());

}

void ExpressionFilterWrapper(int grid_x, int grid_y,
								int block_x, int block_y,
								GNValue *outer_dev, GNValue *inner_dev,
								RESULT *result_dev,
								ulong *index_psum, ulong *exp_psum,
								uint outer_part_size,
								uint outer_cols, uint inner_cols,
								uint jr_size,
								GTreeNode *end_dev, int end_size,
								GTreeNode *post_dev, int post_size,
								GTreeNode *where_dev, int where_size,
								ResBound *res_bound,
								int outer_base_idx, int inner_base_idx,
								bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								, int64_t *val_stack
								, ValueType *type_stack
#endif
						)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	ExpressionFilter<<<grid_size, block_size>>>(outer_dev, inner_dev,
												result_dev, index_psum,
												exp_psum, outer_part_size,
												outer_cols, inner_cols,
												jr_size, end_dev,
												end_size, post_dev,
												post_size, where_dev,
												where_size, res_bound,
												outer_base_idx, inner_base_idx,
												prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												, val_stack
												, type_stack
#endif
												);

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error: Async kernel (exp_filter) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void ExpressionFilterWrapper2(int grid_x, int grid_y,
								int block_x, int block_y,
								GNValue *outer_dev, GNValue *inner_dev,
								RESULT *in_index, RESULT *out_index,
								ulong *exp_psum, int size,
								uint outer_cols, uint inner_cols,
								GTreeNode *end_dev, int end_size,
								GTreeNode *post_dev, int post_size,
								GTreeNode *where_dev, int where_size,
								int outer_base_idx, int inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								, int64_t *val_stack, ValueType *type_stack
#endif
						)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	ExpressionFilter2<<<grid_size, block_size>>>(outer_dev, inner_dev,
															in_index, out_index,
															exp_psum, size,
															outer_cols, inner_cols,
															end_dev, end_size,
															post_dev, post_size,
															where_dev, where_size,
															outer_base_idx, inner_base_idx
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
															, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
															, val_stack, type_stack
#endif
															);
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		printf("Error: Async kernel (ExpressionFilterWrapper2) failed error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());

}

void RemoveEmptyResultWrapper(int grid_x, int grid_y,
								int block_x, int block_y,
								RESULT *out,
								RESULT *in,
								ulong *count_dev,
								ulong *count_dev2,
								uint outer_part_size,
								uint out_size,
								uint in_size)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	RemoveEmptyResult<<<grid_size, block_size>>>(out, in, count_dev, count_dev2, outer_part_size, out_size, in_size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (RemoveEmptyResult) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void RemoveEmptyResultWrapper2(int grid_x, int grid_y,
								int block_x, int block_y,
								RESULT *out, RESULT *in,
								ulong *location, int size)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	RemoveEmptyResult2<<<grid_size, block_size>>>(out, in, location, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (RemoveEmptyResult2) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void ExclusiveScanWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

void InclusiveScanWrapper(ulong *input, int ele_num)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::inclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());
}

void Rebalance(int grid_x, int grid_y, int block_x, int block_y, ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	// Remove Zeros
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	ulong *mark;
	ulong size_no_zeros;
	ResBound *tmp_bound;
	ulong sum;

	/* Remove zeros elements */
	ulong *no_zeros;

	checkCudaErrors(cudaMalloc(&mark, (in_size + 1) * sizeof(ulong)));

	MarkNonZeros<<<grid_size, block_size>>>(in, in_size, mark);
	checkCudaErrors(cudaDeviceSynchronize());

	ExclusiveScanWrapper(mark, in_size + 1, &size_no_zeros);

	if (size_no_zeros == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));

		return;
	}

	checkCudaErrors(cudaMalloc(&no_zeros, (size_no_zeros + 1) * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&tmp_bound, size_no_zeros * sizeof(ResBound)));

	RemoveZeros<<<grid_size, block_size>>>(in, no_zeros, in_bound, tmp_bound, mark, in_size);
	checkCudaErrors(cudaDeviceSynchronize());

	ExclusiveScanWrapper(no_zeros, size_no_zeros + 1, &sum);

	if (sum == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));
		checkCudaErrors(cudaFree(no_zeros));
		checkCudaErrors(cudaFree(tmp_bound));

		return;
	}

	ulong *tmp_location, *local_offset;

	checkCudaErrors(cudaMalloc(&tmp_location, sum * sizeof(ulong)));
	checkCudaErrors(cudaMemset(tmp_location, 0, sizeof(ulong) * sum));
	checkCudaErrors(cudaDeviceSynchronize());

	MarkTmpLocation<<<grid_size, block_size>>>(tmp_location, no_zeros, size_no_zeros);
	checkCudaErrors(cudaDeviceSynchronize());

	InclusiveScanWrapper(tmp_location, sum);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	ComputeOffset<<<grid_size, block_size>>>(no_zeros, tmp_location, local_offset, sum);
	Decompose<<<grid_size, block_size>>>(tmp_bound, *out_bound, tmp_location, local_offset, sum);
	checkCudaErrors(cudaDeviceSynchronize());

	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));

}

}
}

