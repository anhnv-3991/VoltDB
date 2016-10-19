#include "index_join_gpu.h"

namespace voltdb {

/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/

extern "C" {

__device__ GNValue evaluate(GTreeNode *tree_expression,
							int root,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int offset)
{
	if (root == 0)
		return GNValue::getTrue();

	if (root >= tree_size)
		return GNValue::getNullValue();

	GTreeNode tmp_node = tree_expression[root];

	if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
		if (tmp_node.tuple_idx == 0) {
			return outer_tuple[tmp_node.column_idx * offset];
		} else if (tmp_node.tuple_idx == 1) {
			return inner_tuple[tmp_node.column_idx * offset];
		}
	} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT || tmp_node.type == EXPRESSION_TYPE_VALUE_PARAMETER) {
		return tmp_node.value;
	}


	GNValue left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, offset);
	GNValue right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, offset);


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

__device__ GNValue evaluate2(GTreeNode *tree_expression,
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
					stack[top] = outer_tuple[tree_expression[i].column_idx * offset];
					top += offset;
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx * offset];
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

__device__ GNValue evaluate3(GTreeNode *tree_expression,
							int root,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int offset)
{
	if (root == 0)
		return GNValue::getTrue();

	if (root >= tree_size)
		return GNValue::getNullValue();

	GTreeNode tmp_node = tree_expression[root];

	switch (tmp_node.type) {
	case EXPRESSION_TYPE_VALUE_TUPLE: {
		if (tmp_node.tuple_idx == 0) {
			return outer_tuple[tmp_node.column_idx * offset];
		} else if (tmp_node.tuple_idx == 1) {
			return inner_tuple[tmp_node.column_idx * offset];
		} else
			return GNValue::getNullValue();

	}
	case EXPRESSION_TYPE_VALUE_CONSTANT:
	case EXPRESSION_TYPE_VALUE_PARAMETER: {
		return tmp_node.value;
	}
	}


	GNValue left = evaluate3(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple, offset);
	GNValue right = evaluate3(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple, offset);
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

__device__ GNValue evaluate4(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int64_t *stack,
							ValueType *gtype,
							int offset)
{
	ValueType ltype, rtype;

	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i;

	for (int i = 0; i < tree_size; i++) {
		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx * offset].getValue();
					gtype[top] = outer_tuple[tree_expression[i].column_idx * offset].getValueType();
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx * offset].getValue();
					gtype[top] = inner_tuple[tree_expression[i].column_idx * offset].getValueType();

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
				assert(gtype[top - 2 * offset] == VALUE_TYPE_BOOLEAN && gtype[top - offset] == VALUE_TYPE_BOOLEAN);
				stack[top - 2 * offset] = (int64_t)((bool)(stack[top - 2 * offset]) && (bool)(stack[top - offset]));
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				assert(gtype[top - 2 * offset] == VALUE_TYPE_BOOLEAN && gtype[top - offset] == VALUE_TYPE_BOOLEAN);
				stack[top - 2 * offset] = (int64_t)((bool)(stack[top - 2 * offset]) || (bool)(stack[top - offset]));
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] =  (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2 * offset] = (int64_t)((ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
				gtype[top - 2 * offset] = VALUE_TYPE_BOOLEAN;
				top -= offset;
				break;
			}

			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d + right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2 * offset] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2 * offset] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2 * offset] = left_i + right_i;
					gtype[top - 2 * offset] = (ltype > rtype) ? ltype : rtype;
				}
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d - right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2 * offset] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2 * offset] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2 * offset] = left_i - right_i;
					gtype[top - 2 * offset] = (ltype > rtype) ? ltype : rtype;
				}
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d * right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2 * offset] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2 * offset] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2 * offset] = left_i * right_i;
					gtype[top - 2 * offset] = (ltype > rtype) ? ltype : rtype;
				}
				top -= offset;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				ltype = gtype[top - 2 * offset];
				rtype = gtype[top - offset];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2 * offset];
				right_i = stack[top - offset];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = (right_d != 0) ? left_d / right_d : 0;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2 * offset] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2 * offset] = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
				} else {
					stack[top - 2 * offset] = (right_i != 0) ? left_i / right_i : 0;
					gtype[top - 2 * offset] = (ltype > rtype) ? ltype : rtype;
					gtype[top - 2 * offset] = (right_i != 0) ? gtype[top - 2 * offset] : VALUE_TYPE_INVALID;
				}
				top--;
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


__device__ GNValue evaluate2_non_coalesce(GTreeNode *tree_expression,
											int tree_size,
											GNValue *outer_tuple,
											GNValue *inner_tuple,
											int offset)
{
	GNValue stack[MAX_STACK_SIZE];
	int top = 0;

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx * offset];
					top ++;
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx * offset];
					top ++;
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = tree_expression[i].value;
				top ++;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				stack[top - 2] = stack[top - 2].op_and(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				stack[top - 2] = stack[top - 2].op_or(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				stack[top - 2] = stack[top - 2].op_equal(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				stack[top - 2] = stack[top - 2].op_notEqual(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				stack[top - 2] = stack[top - 2].op_lessThan(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				stack[top - 2] = stack[top - 2].op_lessThanOrEqual(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				stack[top - 2] = stack[top - 2].op_greaterThan(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				stack[top - 2] = stack[top - 2].op_greaterThanOrEqual(stack[top - 1]);
				top --;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				stack[top - 2] = stack[top - 2].op_add(stack[top - 1]);
				top --;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				stack[top - 2] = stack[top - 2].op_subtract(stack[top - 1]);
				top --;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				stack[top - 2] = stack[top - 2].op_divide(stack[top - 1]);
				top --;

				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				stack[top - 2] = stack[top - 2].op_multiply(stack[top - 1]);
				top --;

				break;
			}
			default: {
				return GNValue::getFalse();
			}
		}
	}

	return stack[0];
}

__device__ GNValue evaluate4_non_coalesce(GTreeNode *tree_expression,
											int tree_size,
											GNValue *outer_tuple,
											GNValue *inner_tuple,
											int offset)
{
	int64_t stack[MAX_STACK_SIZE];
	ValueType gtype[MAX_STACK_SIZE];
	ValueType ltype, rtype;

	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i;

	for (int i = 0; i < tree_size; i++) {
		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[tree_expression[i].column_idx * offset].getValue();
					gtype[top] = outer_tuple[tree_expression[i].column_idx * offset].getValueType();
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[tree_expression[i].column_idx * offset].getValue();
					gtype[top] = inner_tuple[tree_expression[i].column_idx * offset].getValueType();

				}

				top ++;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = (tree_expression[i].value).getValue();
				gtype[top] = (tree_expression[i].value).getValueType();
				top ++;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				assert(gtype[top - 2] == VALUE_TYPE_BOOLEAN && gtype[top - 1] == VALUE_TYPE_BOOLEAN);
				stack[top - 2] = (int64_t)((bool)(stack[top - 2]) && (bool)(stack[top - 1]));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				assert(gtype[top - 2] == VALUE_TYPE_BOOLEAN && gtype[top - 1] == VALUE_TYPE_BOOLEAN);
				stack[top - 2] = (int64_t)((bool)(stack[top - 2]) || (bool)(stack[top - 1]));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] =  (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] = (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				stack[top - 2] = (int64_t)((ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top --;
				break;
			}

			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d + right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2] = left_i + right_i;
					gtype[top - 2] = (ltype > rtype) ? ltype : rtype;
				}
				top --;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d - right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2] = left_i - right_i;
					gtype[top - 2] = (ltype > rtype) ? ltype : rtype;
				}
				top --;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = left_d * right_d;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2] = VALUE_TYPE_DOUBLE;
				} else {
					stack[top - 2] = left_i * right_i;
					gtype[top - 2] = (ltype > rtype) ? ltype : rtype;
				}
				top --;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				ltype = gtype[top - 2];
				rtype = gtype[top - 1];
				assert(ltype != VALUE_TYPE_NULL && ltype != VALUE_TYPE_INVALID && rtype != VALUE_TYPE_NULL && rtype != VALUE_TYPE_INVALID);
				left_i = stack[top - 2];
				right_i = stack[top - 1];
				left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
				right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
				res_d = (right_d != 0) ? left_d / right_d : 0;
				if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
					stack[top - 2] = *reinterpret_cast<int64_t *>(&res_d);
					gtype[top - 2] = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
				} else {
					stack[top - 2] = (right_i != 0) ? left_i / right_i : 0;
					gtype[top - 2] = (ltype > rtype) ? ltype : rtype;
					gtype[top - 2] = (right_i != 0) ? gtype[top - 2] : VALUE_TYPE_INVALID;
				}
				top--;
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

__device__ int lowerBound(GTreeNode * search_exp,
							int *search_exp_size,
							int search_exp_num,
							int * key_indices,
							int key_index_size,
							GNValue *outer_table,
							GNValue *inner_table,
							int left,
							int right,
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
#ifdef COALESCE_
		tmp = evaluate4(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = evaluate4_non_coalesce(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, offset);
#endif
#else
		tmp = evaluate3(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL, offset);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
#ifdef COALESCE_
		outer_tmp[i] = evaluate2(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = evaluate2_non_coalesce(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, offset);
#endif
#else
		outer_tmp[i] = evaluate(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL, offset);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
			key_idx = key_indices[i];
			inner_tmp = inner_table[middle + key_idx * offset].getValue();
			inner_type = inner_table[middle + key_idx * offset].getValueType();

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
			res = outer_tmp[i].compare_withoutNull(inner_table[middle + key_indices[i] * offset]);
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

__device__ int upperBound(GTreeNode * search_exp,
							int *search_exp_size,
							int search_exp_num,
							int * key_indices,
							int key_index_size,
							GNValue *outer_table,
							GNValue *inner_table,
							int left,
							int right,
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
#ifdef COALESCE_
		tmp = evaluate4(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, val_stack, type_stack, offset);
#else
		tmp = evaluate4_non_coalesce(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, offset);
#endif
#else
		tmp = evaluate3(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL, offset);
#endif
		outer_tmp[i] = tmp.getValue();
		outer_gtype[i] = tmp.getValueType();
#else
#ifdef POST_EXP_
#ifdef COALESCE_
		outer_tmp[i] = evaluate2(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, offset);
#else
		outer_tmp[i] = evaluate2_non_coalesce(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, offset);
#endif
#else
		outer_tmp[i] = evaluate(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL, offset);
#endif
#endif
	}

	while (left <= right) {
		middle = (left + right) >> 1;

#ifndef FUNC_CALL_
		res_i = 0;
		res_d = 0;
		for (int i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {

			key_idx = key_indices[i];
			inner_tmp = inner_table[middle + key_idx * offset].getValue();
			inner_type = inner_table[middle + key_idx * offset].getValueType();

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
			res = outer_tmp[i].compare_withoutNull(inner_table[middle + key_indices[i]]);
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


__global__ void prejoin_filter(GNValue *outer_dev,
								uint outer_part_size,
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
	int offset = blockDim.x * gridDim.x;


	if (x < outer_part_size) {
		GNValue res = GNValue::getTrue();

#ifdef 	TREE_EVAL_
#ifdef FUNC_CALL_
		res = (prejoin_size > 1) ? evaluate(prejoin_dev, 1, prejoin_size, outer_dev + x, NULL, offset) : res;
#else
		res = (prejoin_size > 1) ? evaluate3(prejoin_dev, 1, prejoin_size, outer_dev + x, NULL, offset) : res;
#endif
#elif	POST_EXP_
#ifndef FUNC_CALL_
#ifdef COALESCE_
		res = (prejoin_size > 1) ? evaluate4(prejoin_dev, prejoin_size, outer_dev + x, NULL, val_stack + x, type_stack + x, offset) : res;
#else
		res = (prejoin_size > 1) ? evaluate4_non_coalesce(prejoin_dev, prejoin_size, outer_dev + x, NULL, offset) : res;
#endif
#else
#ifdef COALESCE_
		res = (prejoin_size > 1) ? evaluate2(prejoin_dev, prejoin_size, outer_dev + x, NULL, stack + x, offset) : res;
#else
		res = (prejoin_size > 1) ? evaluate2_non_coalesce(prejoin_dev, prejoin_size, outer_dev + x, NULL, offset) : res;
#endif
#endif
#endif
		result[x] = res.isTrue();
	}
}


__global__ void index_filterLowerBound(GNValue *outer_dev,
										  GNValue *inner_dev,
										  ulong *index_psum,
										  ResBound *res_bound,
										  uint outer_part_size,
										  uint inner_part_size,
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
		switch (lookup_type) {
		case INDEX_LOOKUP_TYPE_EQ:
		case INDEX_LOOKUP_TYPE_GT:
		case INDEX_LOOKUP_TYPE_GTE:
		case INDEX_LOOKUP_TYPE_LT: {
			res_bound[x + k].left = lowerBound(search_exp_dev,
												search_exp_size,
												search_exp_num,
												key_indices,
												key_index_size,
												outer_dev + x,
												inner_dev,
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

__global__ void index_filterUpperBound(GNValue *outer_dev,
										  GNValue *inner_dev,
										  ulong *index_psum,
										  ResBound *res_bound,
										  uint outer_part_size,
										  uint inner_part_size,
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
			res_bound[x + k].right = upperBound(search_exp_dev,
													search_exp_size,
													search_exp_num,
													key_indices,
													key_index_size,
													outer_dev + x,
													inner_dev,
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
//		case INDEX_LOOKUP_TYPE_LTE: {
//			res_bound[x + k].right = upperBound(search_exp_dev, search_exp_size, search_exp_num, key_indices, key_index_size, outer_dev + x * outer_cols, inner_dev, outer_cols, inner_cols, left_bound, right_bound) - 1;
//			break;
//		}
		default:
			break;
		}

		index_psum[x + k] = (res_bound[x + k].right >= 0 && res_bound[x + k].left >= 0) ? (res_bound[x + k].right - res_bound[x + k].left + 1) : 0;
	}


	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		index_psum[x + k + 1] = 0;
	}
}

__global__ void exp_filter(GNValue *outer_dev,
							GNValue *inner_dev,
							RESULT *result_dev,
							ulong *index_psum,
							ulong *exp_psum,
							uint outer_part_size,
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
	int offset = blockDim.x * gridDim.x;

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
			res = (end_size > 1) ? evaluate(end_dev, 1, end_size, outer_dev + x, inner_dev + res_left, offset) : res;
			res = (post_size > 1 && res.isTrue()) ? evaluate(post_dev, 1, post_size, outer_dev + x, inner_dev + res_left, offset) : res;
#else
			res = (end_size > 1) ? evaluate3(end_dev, 1, end_size, outer_dev + x, inner_dev + res_left, offset) : res;
			res = (post_size > 1 && res.isTrue()) ? evaluate3(post_dev, 1, post_size, outer_dev + x, inner_dev + res_left, offset) : res;
#endif

#elif	POST_EXP_
#ifdef 	FUNC_CALL_
#ifdef COALESCE_
			res = (end_size > 0) ? evaluate2(end_dev, end_size, outer_dev + x, inner_dev + res_left, stack + x, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate2(post_dev, post_size, outer_dev + x, inner_dev + res_left, stack + x, offset) : res;
#else
			res = (end_size > 0) ? evaluate2_non_coalesce(end_dev, end_size, outer_dev + x, inner_dev + res_left, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate2_non_coalesce(post_dev, post_size, outer_dev + x, inner_dev + res_left, offset) : res;
#endif
#else
#ifdef COALESCE_
			res = (end_size > 0) ? evaluate4(end_dev, end_size, outer_dev + x, inner_dev + res_left, val_stack + x, type_stack + x, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate4(post_dev, post_size, outer_dev + x, inner_dev + res_left, val_stack + x, type_stack + x, offset) : res;
#else
			res = (end_size > 0) ? evaluate4_non_coalesce(end_dev, end_size, outer_dev + x, inner_dev + res_left, offset) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate4_non_coalesce(post_dev, post_size, outer_dev + x, inner_dev + res_left, offset) : res;

#endif
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

__global__ void write_out(RESULT *out,
							RESULT *in,
							ulong *count_dev,
							ulong *count_dev2,
							uint outer_part_size,
							uint out_size,
							uint in_size)
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

void prejoin_filterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							uint outer_part_size,
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
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	prejoin_filter<<<grid_size, block_size>>>(outer_dev, outer_part_size, prejoin_dev, prejoin_size, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,val_stack,
												type_stack
#endif
												);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (prejoin_filter) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void index_filterWrapper(int grid_x, int grid_y,
							int block_x, int block_y,
							GNValue *outer_dev,
							GNValue *inner_dev,
							ulong *index_psum,
							ResBound *res_bound,
							uint outer_part_size,
							uint inner_part_size,
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
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	index_filterLowerBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_part_size,
														inner_part_size,
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
		printf("Error: Async kernel (index_filterLowerBound) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());

	index_filterUpperBound<<<grid_size, block_size>>>(outer_dev, inner_dev,
														index_psum, res_bound,
														outer_part_size,
														inner_part_size,
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
		printf("Error: Async kernel (index_filterUpperBound) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());

}

void exp_filterWrapper(int grid_x, int grid_y,
						int block_x, int block_y,
						GNValue *outer_dev,
						GNValue *inner_dev,
						RESULT *result_dev,
						ulong *index_psum,
						ulong *exp_psum,
						uint outer_part_size,
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
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	exp_filter<<<grid_size, block_size>>>(outer_dev, inner_dev,
											result_dev, index_psum,
											exp_psum, outer_part_size,
											jr_size, end_dev,
											end_size, post_dev,
											post_size, where_dev,
											where_size, res_bound,
											outer_base_idx, inner_base_idx,
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
		printf("Error: Async kernel (exp_filter) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void write_outWrapper(int grid_x, int grid_y,
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

	write_out<<<grid_size, block_size>>>(out, in, count_dev, count_dev2, outer_part_size, out_size, in_size);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (write_out) error: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void prefix_sumWrapper(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

}
}
