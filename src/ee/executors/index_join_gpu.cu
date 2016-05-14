#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"

using namespace voltdb;

/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/

extern "C" {
CUDAH bool evaluate2(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int outer_index,
							int inner_index,
							int outer_cols,
							int inner_cols)
{
	GNValue stack[MAX_STACK_SIZE];
	GNValue *stack_ptr = stack;
	GNValue left, right;
	GTreeNode tmp_node;
	int outer_idx = outer_index * outer_cols;
	int inner_idx = inner_index * inner_cols;

	int i;
	for (i = 0; i < tree_size; i++) {

		tmp_node = tree_expression[i];
		switch (tmp_node.type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				//*stack_ptr++ = (tmp_node.tuple_idx == 0) ? outer_tuple[outer_idx + tmp_node.column_idx] : inner_tuple[inner_idx + tmp_node.column_idx];
				if (tmp_node.tuple_idx == 0) {
					*stack_ptr++ = outer_tuple[outer_idx + tmp_node.column_idx];
				} else if (tmp_node.tuple_idx == 1) {
					*stack_ptr++ = inner_tuple[inner_idx + tmp_node.column_idx];
				}
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				*stack_ptr++ = tmp_node.value;
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
			default: {
				return false;
			}
		}
	}

	return ((--stack_ptr)->isTrue()) ? true: false;
}

CUDAH bool evaluate5(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int outer_index,
							int inner_index,
							int64_t *stack,
							ValueType *gtype)
{
	memset(stack, 0, 8 * sizeof(int64_t));
	memset(gtype, 0, 8 * sizeof(ValueType));
	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i, res_i;


	for (int i = 0; i < tree_size; i++) {
		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = outer_tuple[outer_index + tree_expression[i].column_idx].getValue();
					gtype[top] = outer_tuple[outer_index + tree_expression[i].column_idx].getValueType();
				}

				if (tree_expression[i].tuple_idx == 1) {
					stack[top] = inner_tuple[inner_index + tree_expression[i].column_idx].getValue();
					gtype[top] = outer_tuple[outer_index + tree_expression[i].column_idx].getValueType();
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
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] =  (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d == right_d) : (left_i == right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d != right_d) : (left_i != right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d < right_d) : (left_i < right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d <= right_d) : (left_i <= right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d > right_d) : (left_i > right_i);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				stack[top - 2] = (int64_t)((gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d >= right_d) : (left_i >= right_i));
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}

			case EXPRESSION_TYPE_OPERATOR_PLUS: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				res_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : (left_i + right_i);
				res_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d + right_d) : 0;
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<int64_t *>(&res_d) : res_i;
				gtype[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_BIGINT;
				top--;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MINUS: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				res_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : (left_i - right_i);
				res_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d - right_d) : 0;
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<int64_t *>(&res_d) : res_i;
				gtype[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_BIGINT;
				top--;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_MULTIPLY: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				res_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : (left_i * right_i);
				res_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d * right_d) : 0;
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<int64_t*>(&res_d) : res_i;
				gtype[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_BIGINT;
				top--;
				break;
			}
			case EXPRESSION_TYPE_OPERATOR_DIVIDE: {
				assert(gtype[top - 2] != VALUE_TYPE_NULL && gtype[top - 2] != VALUE_TYPE_INVALID && gtype[top - 1] != VALUE_TYPE_NULL && gtype[top - 1] != VALUE_TYPE_INVALID);
				left_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 2];
				right_i = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : stack[top - 1];
				left_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 2) : static_cast<double>(left_i);
				right_d = (gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(stack + top - 1) : static_cast<double>(right_i);
				if (right_i == 0 && right_d == 0)
					return false;
				res_i = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? 0 : (left_i / right_i);
				res_d = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? (left_d / right_d) : 0;
				stack[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<int64_t*>(&res_d) : res_i;
				gtype[top - 2] = (gtype[top - 2] == VALUE_TYPE_DOUBLE || gtype[top - 1] == VALUE_TYPE_DOUBLE) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_BIGINT;
				top--;
				break;
			}
			default: {
				return false;
			}
		}
	}

	return true;
}

CUDAH bool binarySearchIdx(GTreeNode * search_exp,
									int *search_exp_size,
									int search_exp_num,
									int * key_indices,
									int key_index_size,
									GNValue *outer_table,
									GNValue *inner_table,
									int search_row,
									int outer_cols,
									int inner_cols,
									int left_bound,
									int right_bound,
									int *res_left,
									int *res_right)
{
	int left = left_bound, right = right_bound;
	int middle = -1, i, j, key_idx;
	int outer_idx = search_row * outer_cols, inner_idx;
	int64_t outer_tmp[8], stack[8], inner_tmp;
	int search_ptr;
	ValueType gtype[8], outer_gtype[8], inner_type;
	int64_t outer_i, inner_i, res_i, res_i2;
	double outer_d, inner_d, res_d, res_d2;

	*res_left = *res_right = -1;
	res_i = -1;
	res_d = -1;

	for (i = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
		evaluate5(search_exp + search_ptr, search_exp_size[i], outer_table, NULL, outer_idx, 0, stack, gtype);
		outer_tmp[i] = stack[0];
		outer_gtype[i] = gtype[0];
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

__global__ void prejoin_filter(GNValue *outer_dev,
								uint outer_part_size,
								uint outer_cols,
								GTreeNode *prejoin_dev,
								uint prejoin_size,
								bool *result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int64_t stack[MAX_STACK_SIZE];
	ValueType gtype[MAX_STACK_SIZE];
	bool res = true;

	if (x < outer_part_size && prejoin_size > 0)
		res = evaluate5(prejoin_dev, prejoin_size, outer_dev, NULL, x * outer_cols, 0, stack, gtype);

	if (x < outer_part_size)
		result[x] = res;
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
		int64_t outer_tmp[8], stack[8];
		int search_ptr, i;
		ValueType gtype[8], outer_gtype[8];
		int res_left, res_right;

//		for (i = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[i], i++) {
//			evaluate5(search_exp_dev + search_ptr, search_exp_size[i], outer_dev, NULL, x * outer_cols, 0, stack, gtype);
//			outer_tmp[i] = stack[0];
//			outer_gtype[i] = gtype[0];
//			//assert(outer_gtype[i] != VALUE_TYPE_NULL && outer_gtype[i] != VALUE_TYPE_INVALID);
//		}

		binarySearchIdx(search_exp_dev,
						search_exp_size,
						search_exp_num,
						key_indices,
						key_index_size,
						outer_dev,
						inner_dev,
						x,
						outer_cols,
						inner_cols,
						left_bound,
						right_bound,
						&res_left,
						&res_right);
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
		bool res = true, eval_res = true;
		int res_left = -1, res_right = -1;
		int64_t stack[8];
		ValueType gtype[8];

		res_left = res_bound[x + k].left;
		res_right = res_bound[x + k].right;

		//printf("x = %d. Left = %d. right = %d, val1 = %d, val2 = %d\n", x, res_left, res_right, (int)(outer_dev[x * outer_cols + 1].GNValue::getValue()), (int)(outer_dev[x * outer_cols + 3].GNValue::getValue()));


		while (res_left >= 0 && res_left <= res_right && writeloc < jr_size) {
			res = true;
			eval_res = (end_size > 0) ? evaluate5(end_dev, end_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype) : eval_res;
			res &= (end_size > 0) ? (bool)stack[0] : res;
			eval_res = (post_size > 0 && res) ? evaluate5(post_dev, post_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype) : eval_res;
			res &= (post_size > 0) ? (bool)stack[0] : res;
			eval_res = (where_size > 0 && res) ? evaluate5(where_dev, where_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype) : eval_res;
			res &= (where_size > 0) ? (bool)stack[0] : res;

//			if (end_size > 0) {
//				evaluate5(end_dev, end_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype);
//				res = (bool)stack[0];
//			}
//
//			if (res && post_size > 0) {
//				evaluate5(post_dev, post_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype);
//				res = (bool)stack[0];
//			}
//
//			if (res && where_size > 0) {
//				evaluate5(where_dev, where_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype);
//				res = (bool)stack[0];
//			}

			result_dev[writeloc].lkey = (res) ? (x + outer_base_idx) : (-1);
			result_dev[writeloc].rkey = (res) ? (res_left + inner_base_idx) : (-1);

			count += (res) ? 1 : 0;
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

}
