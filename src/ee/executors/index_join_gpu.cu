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


	GNValue left = evaluate(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = evaluate(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);


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
							GNValue *inner_tuple)
{
	GNValue stack[MAX_STACK_SIZE];
	GNValue *stack_ptr = stack;
	GNValue left, right;

	for (int i = 0; i < tree_size; i++) {

		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					*stack_ptr++ = outer_tuple[tree_expression[i].column_idx];
				} else if (tree_expression[i].tuple_idx == 1) {
					*stack_ptr++ = inner_tuple[tree_expression[i].column_idx];
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				*stack_ptr++ = tree_expression[i].value;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_and(right);
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_or(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_equal(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_notEqual(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_lessThan(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_lessThanOrEqual(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_greaterThan(right);
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_greaterThanOrEqual(right);
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_add(right);
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_subtract(right);
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_divide(right);
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				right = *--stack_ptr;
				left = *--stack_ptr;
				*stack_ptr++ = left.op_multiply(right);
				break;
			}
			default: {
				return GNValue::getFalse();
			}
		}
	}

	return *--stack_ptr;
}

__device__ GNValue evaluate3(GTreeNode *tree_expression,
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


	GNValue left = evaluate3(tree_expression, root * 2, tree_size, outer_tuple, inner_tuple);
	GNValue right = evaluate3(tree_expression, root * 2 + 1, tree_size, outer_tuple, inner_tuple);
	int64_t left_i = left.getValue(), right_i = right.getValue(), res_i;
	ValueType left_t = left.getValueType(), right_t = right.getValueType(), res_t;


	switch (tmp_node.type) {
	case EXPRESSION_TYPE_CONJUNCTION_AND: {
		assert(left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN);
		res_i = (int64_t)((bool)left_t && (bool)right_t);
		return GNValue(VALUE_TYPE_BOOLEAN, res_i);
	}
	case EXPRESSION_TYPE_CONJUNCTION_OR: {
		assert(left_t == VALUE_TYPE_BOOLEAN && right_t == VALUE_TYPE_BOOLEAN);
		res_i = (int64_t)((bool)left_t || (bool)right_t);
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

__device__ GNValue evaluate5(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple)
{
	int64_t stack[MAX_STACK_SIZE];
	ValueType gtype[MAX_STACK_SIZE], ltype, rtype;

	//memset(stack, 0, MAX_STACK_SIZE * sizeof(int64_t));
	//memset(gtype, 0, MAX_STACK_SIZE * sizeof(ValueType));
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

				top++;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				stack[top] = (tree_expression[i].value).getValue();
				gtype[top] = (tree_expression[i].value).getValueType();
				top++;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				assert(gtype[top - 2] == VALUE_TYPE_BOOLEAN && gtype[top - 1] == VALUE_TYPE_BOOLEAN);
				stack[top - 2] = (int64_t)((bool)(stack[top - 2]) && (bool)(stack[top - 1]));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				assert(gtype[top - 2] == VALUE_TYPE_BOOLEAN && gtype[top - 1] == VALUE_TYPE_BOOLEAN);
				stack[top - 2] = (int64_t)((bool)(stack[top - 2]) || (bool)(stack[top - 1]));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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
				top--;
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

__device__ bool binarySearchIdx(GTreeNode * search_exp,
									int *search_exp_size,
									int search_exp_num,
									int * key_indices,
									int key_index_size,
									GNValue *outer_table,
									GNValue *inner_table,
									int outer_cols,
									int inner_cols,
									int left_bound,
									int right_bound,
									int *res_left,
									int *res_right)
{
	int left = left_bound, right = right_bound;
	int middle = -1, i, j, key_idx;
	int inner_idx;
	//int64_t outer_tmp[8], stack[8], inner_tmp;
	int64_t outer_tmp[8];
	int64_t inner_tmp;
	int search_ptr;
	ValueType outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i, res_i2;
	double outer_d, inner_d, res_d, res_d2;
	GNValue tmp;

	*res_left = *res_right = -1;
	res_i = -1;
	res_d = -1;

	for (i = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
		//evaluate5(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, stack, gtype);
#ifndef TREE_EVAL_
		tmp = evaluate5(search_exp + search_ptr, search_exp_size[i], outer_table, NULL);
#else
		tmp = evaluate3(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);
#endif

		//outer_tmp[i] = stack[0];
		outer_tmp[i] = tmp.getValue();
		//outer_gtype[i] = gtype[0];
		outer_gtype[i] = tmp.getValueType();
	}

	while (left <= right && (res_i != 0 || res_d != 0)) {
		res_i = 0;
		res_d = 0;
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

		for (i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
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

		right = (res_i < 0 || res_d < 0) ? (middle - 1) : right;
		left = (res_i > 0 || res_d > 0) ? (middle + 1) : left;
	}

	res_i2 = res_i;
	res_d2 = res_d;
	for (left = middle - 1; (res_i == 0) && (res_d == 0) && (left >= left_bound);) {
		inner_idx = left * inner_cols;

		for (j = 0; (res_i == 0) && (res_d == 0) && (j < search_exp_num); j++) {
			key_idx = key_indices[j];
			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[j] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[j];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[j] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + j) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[j] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[j] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
		left = (res_i == 0 && res_d == 0) ? (left - 1) : left;
	}
	left++;

	res_i = res_i2;
	res_d = res_d2;
	for (right = middle + 1; (res_i == 0) && (res_d == 0) && (right <= right_bound);) {
		inner_idx = right * inner_cols;

		for (j = 0; (res_i == 0) && (res_d == 0) && (j < search_exp_num); j++) {
			key_idx = key_indices[j];
			inner_tmp = inner_table[inner_idx + key_idx].getValue();
			inner_type = inner_table[inner_idx + key_idx].getValueType();

			outer_i = (outer_gtype[j] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[j];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[j] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + j) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[j] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[j] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}
		right = (res_i == 0 && res_d == 0) ? (right + 1) : right;
	}
	right--;
	res_i = res_i2;
	res_d = res_d2;

	*res_left = (res_i == 0 && res_d == 0) ? left : (-1);
	*res_right = (res_i == 0 && res_d == 0) ? right : (-1);

	return (res_i == 0 && res_d == 0);
}

__device__ bool binarySearchIdx2(GTreeNode *search_exp,
									int *search_exp_size,
									int search_exp_num,
									int *key_indices,
									int key_index_size,
									GNValue *outer_table,
									GNValue *inner_table,
									int outer_cols,
									int inner_cols,
									int left_bound,
									int right_bound,
									int *res_left,
									int *res_right)
{
	int left = left_bound, right = right_bound;
	int middle = -1, i, j;
	int inner_idx;
	GNValue outer_tmp[8];
	int search_ptr;
	int res, res2;

	*res_left = *res_right = -1;
	res = -1;

	for (i = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
		outer_tmp[i] = evaluate(search_exp + search_ptr, 1, search_exp_size[i], outer_table, NULL);

	}

	while (left <= right && (res != 0)) {
		res = 0;
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

		for (i = 0; (res == VALUE_COMPARE_EQUAL) && (i < search_exp_num); i++)
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);

		right = (res == VALUE_COMPARE_LESSTHAN) ? (middle - 1) : right;
		left = (res == VALUE_COMPARE_GREATERTHAN) ? (middle + 1) : left;
	}

	res2 = res;

	for (left = middle - 1; (res == 0) && (left >= left_bound);) {
		inner_idx = left * inner_cols;

		for (j = 0; (res == 0) && (j < search_exp_num); j++)
			res = outer_tmp[j].compare_withoutNull(inner_table[inner_idx + key_indices[j]]);

		left = (res == 0) ? (left - 1) : left;
	}
	left++;

	res = res2;
	for (right = middle + 1; (res == 0) && (right <= right_bound);) {
		inner_idx = right * inner_cols;

		for (j = 0; (res == 0) && (j < search_exp_num); j++)
			res = outer_tmp[j].compare_withoutNull(inner_table[inner_idx + key_indices[j]]);

		right = (res == 0) ? (right + 1) : right;
	}
	right--;
	res = res2;

	*res_left = (res == 0) ? left : (-1);
	*res_right = (res == 0) ? right : (-1);

	return (res == 0);
}

__device__ bool binarySearchIdx3(GTreeNode *search_exp,
									int *search_exp_size,
									int search_exp_num,
									int *key_indices,
									int key_index_size,
									GNValue *outer_table,
									GNValue *inner_table,
									int outer_cols,
									int inner_cols,
									int left_bound,
									int right_bound,
									int *res_left,
									int *res_right)
{
	int left = left_bound, right = right_bound;
	int middle = -1, i, j;
	int inner_idx;
	GNValue outer_tmp[8];
	int search_ptr;
	int res, res2;

	*res_left = *res_right = -1;
	res = -1;

	for (i = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
		outer_tmp[i] = evaluate2(search_exp + search_ptr, search_exp_size[i], outer_table, NULL);
	}

	while (left <= right && (res != 0)) {
		res = 0;
		middle = (left + right) >> 1;
		inner_idx = middle * inner_cols;

		for (i = 0; (res == VALUE_COMPARE_EQUAL) && (i < search_exp_num); i++)
			res = outer_tmp[i].compare_withoutNull(inner_table[inner_idx + key_indices[i]]);

		right = (res == VALUE_COMPARE_LESSTHAN) ? (middle - 1) : right;
		left = (res == VALUE_COMPARE_GREATERTHAN) ? (middle + 1) : left;
	}

	res2 = res;

	for (left = middle - 1; (res == 0) && (left >= left_bound);) {
		inner_idx = left * inner_cols;

		for (j = 0; (res == 0) && (j < search_exp_num); j++)
			res = outer_tmp[j].compare_withoutNull(inner_table[inner_idx + key_indices[j]]);

		left = (res == 0) ? (left - 1) : left;
	}
	left++;

	res = res2;
	for (right = middle + 1; (res == 0) && (right <= right_bound);) {
		inner_idx = right * inner_cols;

		for (j = 0; (res == 0) && (j < search_exp_num); j++)
			res = outer_tmp[j].compare_withoutNull(inner_table[inner_idx + key_indices[j]]);

		right = (res == 0) ? (right + 1) : right;
	}
	right--;
	res = res2;

	*res_left = (res == 0) ? left : (-1);
	*res_right = (res == 0) ? right : (-1);

	return (res == 0);
}

__global__ void prejoin_filter(GNValue *outer_dev,
								uint outer_part_size,
								uint outer_cols,
								GTreeNode *prejoin_dev,
								uint prejoin_size,
								bool *result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;


	if (x < outer_part_size) {
		GNValue res = GNValue::getTrue();

#ifdef 	TREE_EVAL_
#ifdef FUNC_CALL_
		res = (prejoin_size > 1) ? evaluate(prejoin_dev, 1, prejoin_size, outer_dev + x * outer_cols, NULL) : res;
#else
		res = (prejoin_size > 1) ? evaluate3(prejoin_dev, 1, prejoin_size, outer_dev + x * outer_cols, NULL) : res;
#endif
#elif	POST_EXP_
#ifndef FUNC_CALL_
		res = (prejoin_size > 1) ? evaluate5(prejoin_dev, prejoin_size, outer_dev + x * outer_cols, NULL) : res;
#else
		res = (prejoin_size > 1) ? evaluate2(prejoin_dev, prejoin_size, outer_dev + x * outer_cols, NULL) : res;
#endif
#endif
		result[x] = res.isTrue();
	}
}

CUDAH int lowerBound(int search_exp_num,
						int * key_indices,
						int64_t *outer_tmp,
						GNValue *inner_table,
						int inner_cols,
						int left,
						int right,
						ValueType *outer_gtype)
{
	int middle = -1, i;
	int inner_idx;
	int64_t inner_tmp;
	ValueType inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	int result = -1;

	while (left <= right) {
		res_i = 0;
		res_d = 0;
		middle = (left + right) / 2;
		inner_idx = middle * inner_cols;

		for (i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
			inner_tmp = inner_table[inner_idx + key_indices[i]].getValue();
			inner_type = inner_table[inner_idx + key_indices[i]].getValueType();
			assert(outer_gtype[i] != VALUE_TYPE_NULL && outer_gtype[i] != VALUE_TYPE_INVALID);

			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}

		right = (res_i <= 0 && res_d <= 0) ? (middle - 1) : right;	//Move to left
		left = (res_i > 0 || res_d > 0) ? (middle + 1) : left;		//Move to right
		result = (res_i <= 0 && res_d <= 0) ? middle : result;
	}

	return result;
}


CUDAH int upperBound(int search_exp_num,
						int * key_indices,
						int64_t *outer_tmp,
						GNValue *inner_table,
						int inner_cols,
						int left,
						int right,
						ValueType *outer_gtype)
{
	int middle = -1, i;
	int inner_idx;
	int64_t inner_tmp;
	ValueType inner_type;
	int64_t outer_i, inner_i, res_i;
	double outer_d, inner_d, res_d;
	int result = right + 1;

	while (left <= right) {
		res_i = 0;
		res_d = 0;
		middle = (left + right) / 2;
		inner_idx = middle * inner_cols;

		for (i = 0; (res_i == 0) && (res_d == 0) && (i < search_exp_num); i++) {
			inner_tmp = inner_table[inner_idx + key_indices[i]].getValue();
			inner_type = inner_table[inner_idx + key_indices[i]].getValueType();

			assert(inner_type != VALUE_TYPE_NULL && inner_type != VALUE_TYPE_INVALID);
			outer_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? 0 : outer_tmp[i];
			inner_i = (inner_type == VALUE_TYPE_DOUBLE) ? 0: inner_tmp;
			outer_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(outer_tmp + i) : static_cast<double>(outer_i);
			inner_d = (inner_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&inner_tmp) : static_cast<double>(inner_i);

			res_i = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? 0 : (outer_i - inner_i);
			res_d = (outer_gtype[i] == VALUE_TYPE_DOUBLE || inner_type == VALUE_TYPE_DOUBLE) ? (outer_d - inner_d) : 0;
		}

		right = (res_i < 0 || res_d < 0) ? (middle - 1) : right;	//Move to left
		left = (res_i >= 0 && res_d >= 0) ? (middle + 1) : left;	//Move to right
		result = (res_i < 0 || res_d < 0) ? middle : result;
	}

	return result - 1;
}

__global__ void index_filter(GNValue *outer_dev,
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
							  bool *prejoin_res_dev)

{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);

	index_psum[x + k] = 0;
	res_bound[x + k].left = -1;
	res_bound[x + k].right = -1;

	if (x < outer_part_size && prejoin_res_dev[x]) {
		int res_left, res_right;


#ifdef POST_EXP_
#ifndef FUNC_CALL_
		binarySearchIdx(search_exp_dev,
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
							&res_left,
							&res_right);
#else
		binarySearchIdx3(search_exp_dev,
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
							&res_left,
							&res_right);
#endif
#elif TREE_EVAL_
#ifndef FUNC_CALL_
		binarySearchIdx(search_exp_dev,
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
							&res_left,
							&res_right);
#else
		binarySearchIdx2(search_exp_dev,
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
							&res_left,
							&res_right);
#endif

#endif

		switch (lookup_type) {
		case INDEX_LOOKUP_TYPE_EQ: {
//			res_bound[x + k].left = lowerBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype);
//			res_bound[x + k].right = upperBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype);
			res_bound[x + k].left = res_left;
			res_bound[x + k].right = res_right;

			break;
		}
		case INDEX_LOOKUP_TYPE_GT: {
//			res_bound[x + k].left = upperBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype);
			res_bound[x + k].left = res_right + 1;
			res_bound[x + k].right = right_bound;
			break;
		}
		case INDEX_LOOKUP_TYPE_GTE: {
//			res_bound[x + k].left = lowerBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype);
			res_bound[x + k].left = res_left;
			res_bound[x + k].right = right_bound;
			break;
		}
		case INDEX_LOOKUP_TYPE_LT: {
			res_bound[x + k].left = left_bound;
			res_bound[x + k].right = res_left - 1;
//			res_bound[x + k].right = lowerBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype) - 1;
			break;
		}
		case INDEX_LOOKUP_TYPE_LTE: {
			res_bound[x + k].left = left_bound;
			res_bound[x + k].right = res_right;
//			res_bound[x + k].right = upperBound(search_exp_num, key_indices, outer_tmp, inner_dev, inner_cols, left_bound, right_bound, outer_gtype) - 1;
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


__global__ void exp_filter(GNValue *outer_dev,
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
							bool *prejoin_res_dev)
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
			res = (end_size > 1) ? evaluate(end_dev, 1, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? evaluate(post_dev, 1, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
#else
			res = (end_size > 1) ? evaluate3(end_dev, 1, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 1 && res.isTrue()) ? evaluate3(post_dev, 1, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
#endif

#elif	POST_EXP_
#ifdef 	FUNC_CALL_

			res = (end_size > 0) ? evaluate2(end_dev, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate2(post_dev, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;

#else
			res = (end_size > 0) ? evaluate5(end_dev, end_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
			res = (post_size > 0 && res.isTrue()) ? evaluate5(post_dev, post_size, outer_dev + x * outer_cols, inner_dev + res_left * inner_cols) : res;
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
							uint outer_cols,
							GTreeNode *prejoin_dev,
							uint prejoin_size,
							bool *result)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	prejoin_filter<<<grid_size, block_size>>>(outer_dev, outer_part_size, outer_cols, prejoin_dev, prejoin_size, result);
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
							uint outer_cols,
							uint inner_part_size,
							uint inner_cols,
							GTreeNode *search_exp_dev,
							int *search_exp_size,
							int search_exp_num,
							int *key_indices,
							int key_index_size,
							IndexLookupType lookup_type,
							bool *prejoin_res_dev)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	index_filter<<<grid_size, block_size>>>(outer_dev, inner_dev,
											index_psum, res_bound,
											outer_part_size, outer_cols,
											inner_part_size, inner_cols,
											search_exp_dev, search_exp_size,
											search_exp_num, key_indices,
											key_index_size, lookup_type,
											prejoin_res_dev);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel (index_filter) error: %s\n", cudaGetErrorString(err));
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
						bool *prejoin_res_dev)
{
	dim3 grid_size(grid_x, grid_y, 1);
	dim3 block_size(block_x, block_y, 1);

	exp_filter<<<grid_size, block_size>>>(outer_dev, inner_dev,
											result_dev, index_psum,
											exp_psum, outer_part_size,
											outer_cols, inner_cols,
											jr_size, end_dev,
											end_size, post_dev,
											post_size, where_dev,
											where_size, res_bound,
											outer_base_idx, inner_base_idx,
											prejoin_res_dev);
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
