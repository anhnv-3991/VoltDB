#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "common/types.h"
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
	memset(stack, 0, 64);
	memset(gtype, 0, 8);
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
				stack[top - 2] = (int64_t)((bool)stack[top - 2] & (bool)stack[top - 1]);
				gtype[top - 2] = VALUE_TYPE_BOOLEAN;
				top--;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				assert(gtype[top - 2] == VALUE_TYPE_BOOLEAN && gtype[top - 1] == VALUE_TYPE_BOOLEAN);
				stack[top - 2] = (int64_t)((bool)stack[top - 2] | (bool)stack[top - 1]);
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
							  int key_index_size)

{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	bool res = false;


	index_psum[x + k] = 0;
	res_bound[x + k].left = -1;
	res_bound[x + k].right = -1;

	if (x < outer_part_size) {
		res = binarySearchIdx(search_exp_dev,
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
								&res_bound[x + k].left,
								&res_bound[x + k].right);

		index_psum[x + k] = (res && res_bound[x + k].right >= 0 && res_bound[x + k].left >= 0) ? (res_bound[x + k].right - res_bound[x + k].left + 1) : 0;
	}

	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		index_psum[x + k + 1] = 0;
	}
}


__global__ void expression_filter(GNValue *outer_dev,
									GNValue *inner_dev,
									RESULT *result_dev,
									ulong *index_psum,
									ulong *exp_psum,
									uint outer_part_size,
									uint outer_cols,
									uint inner_cols,
									uint jr_size,
									GTreeNode *post_ex_dev,
									int post_size,
									ResBound *res_bound,
									int outer_base_idx,
									int inner_base_idx)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;

	exp_psum[x + k] = 0;

	if (x < outer_part_size) {
		ulong writeloc = index_psum[x + k];
		int count = 0;
		bool res = true;
		int res_left = -1, res_right = -1;
		int64_t stack[8];
		ValueType gtype[4];

		res_left = res_bound[x + k].left;
		res_right = res_bound[x + k].right;

		while (res_left >= 0 && res_left <= res_right && writeloc < jr_size) {
			res = (post_size >= 1) ? evaluate5(post_ex_dev, post_size, outer_dev, inner_dev, x * outer_cols, res_left * inner_cols, stack, gtype) : res;
			res = (bool)stack[0];

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
	int i = 0;

	if (x < outer_part_size) {
		while (i < num) {
			lkey = (readloc_in + i < in_size) ? in[readloc_in + i].lkey : (-1);
			rkey = (readloc_in + i < in_size) ? in[readloc_in + i].rkey : (-1);
			out[writeloc_out].lkey = (lkey != -1) ? lkey : (-1);
			out[writeloc_out].rkey = (rkey != -1) ? rkey : (-1);
			i++;
			writeloc_out += (lkey != -1 && rkey != -1) ? 1 : 0;
		}
	}
}

}
