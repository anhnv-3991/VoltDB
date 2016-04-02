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
__device__ bool pushStack(GNValue *stack, int *top, GNValue newVal)
{
//	if (*top >= MAX_STACK_SIZE)
//		return false;
	(*top)++;
	stack[*top] = newVal;
	return true;
}

__device__ GNValue popStack(GNValue *stack, int *top)
{
	GNValue ret = stack[*top];
	(*top)--;

	return ret;
}
//No more recursive
//__device__ bool evaluate(GTreeNode *tree_expression,
//							int tree_size,
//							GNValue *outer_tuple,
//							GNValue *inner_tuple,
//							int outer_index,
//							int inner_index,
//							int outer_cols,
//							int inner_cols,
//							GNValue *stack)
//{
//	int top = -1;
//	GNValue left, right;
//	GTreeNode tmp_node;
//	int outer_idx = outer_index * outer_cols;
//	int inner_idx = inner_index * inner_cols;
//
//
//	memset(stack, 0, MAX_STACK_SIZE);
//	int i;
//	for (i = 0; i < tree_size; i++) {
//		//printf("tree_expression type = %d at i = %d\n", tree_expression[i].type, i);
//		tmp_node = tree_expression[i];
//		if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
//			if (tmp_node.column_idx >= MAX_GNVALUE || tmp_node.column_idx < 0)
//				return false;
//
//			if (tmp_node.tuple_idx == 0) {
//				if (!pushStack(stack, &top, outer_tuple[outer_idx + tmp_node.column_idx])) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//			} else if (tmp_node.tuple_idx == 1) {
//				if (!pushStack(stack, &top, inner_tuple[inner_idx + tmp_node.column_idx])) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//			} else {
//				return false;
//			}
//
//			continue;
//		} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT) {
//			if (!pushStack(stack, &top, tmp_node.value)) {
//				printf("Error: Failed to push %d to stack! Exiting...\n, i");
//				return false;
//			}
//			continue;
//		}
//
//		// Get left operand
//		right = popStack(stack, &top);
//
//		// Get right operand
//		left = popStack(stack, &top);
//
//		switch(tmp_node.type) {
//			case EXPRESSION_TYPE_CONJUNCTION_AND: {
//				if ((left.getValueType() != VALUE_TYPE_BOOLEAN || right.getValueType() != VALUE_TYPE_BOOLEAN))
//					return false;
//
//				if (!pushStack(stack, &top, left.op_and(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_CONJUNCTION_OR: {
//				if (left.getValueType() != VALUE_TYPE_BOOLEAN || right.getValueType() != VALUE_TYPE_BOOLEAN) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_or(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_EQUAL: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_equal(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_notEqual(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_lessThan(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_lessThanOrEqual(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_greaterThan(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
//				if (left.getValueType() != right.getValueType()) {
//					return false;
//				}
//
//				if (!pushStack(stack, &top, left.op_greaterThanOrEqual(right))) {
//					printf("Error: Failed to push %d to stack! Exiting...\n", i);
//					return false;
//				}
//
//				break;
//			}
//			default: {
//				return false;
//			}
//		}
//	}
//
//	return (popStack(stack, &top).isTrue()) ? true: false;
//}

__device__ bool evaluate2(GTreeNode *tree_expression,
							int tree_size,
							GNValue *outer_tuple,
							GNValue *inner_tuple,
							int outer_index,
							int inner_index,
							int outer_cols,
							int inner_cols)
{
	GNValue stack[MAX_STACK_SIZE];
	int top = -1;
	GNValue left, right;
	GTreeNode tmp_node;
	int outer_idx = outer_index * outer_cols;
	int inner_idx = inner_index * inner_cols;


	int i;
	for (i = 0; i < tree_size; i++) {
		tmp_node = tree_expression[i];
		switch (tmp_node.type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				//tmp_tuple = (tmp_node.tuple_idx == 0) ? outer_tuple[outer_idx + tmp_node.column_idx] : inner_tuple[inner_idx + tmp_node.column_idx];
				if (tmp_node.tuple_idx == 0) {
					pushStack(stack, &top, outer_tuple[outer_idx + tmp_node.column_idx]);
				} else if (tmp_node.tuple_idx == 1) {
					pushStack(stack, &top, inner_tuple[inner_idx + tmp_node.column_idx]);
				}
				//pushStack(stack, &top, tmp_tuple);
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT:
			case EXPRESSION_TYPE_VALUE_PARAMETER: {
				pushStack(stack, &top, tmp_node.value);
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_and(right));
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_or(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_equal(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_notEqual(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_lessThan(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_lessThanOrEqual(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_greaterThan(right));
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				right = popStack(stack, &top);
				left = popStack(stack, &top);
				pushStack(stack, &top, left.op_greaterThanOrEqual(right));
				break;
			}
			default: {
				return false;
			}
		}
	}

	return (popStack(stack, &top).isTrue()) ? true: false;
}

__device__ bool evaluate4(GTreeNode *tree_expression,
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


__device__ bool binarySearchIdx(const int * search_key_indices,
								const int search_key_size,
								const int * key_indices,
								const int key_index_size,
								const GNValue *outer_table,
								const GNValue *inner_table,
								int search_row,
								int outer_cols,
								int inner_cols,
								int left_bound,
								int right_bound,
								int *res_left,
								int *res_right)
{
	int left = left_bound, right = right_bound;

	int middle = -1, res, i, j, search_idx, key_idx, res2;
	int outer_idx = search_row * outer_cols;
	int inner_idx;
	int outer_cmp, inner_cmp;

	*res_left = *res_right = -1;
	middle = left_bound;
	res = -1;

	while (left <= right && left >= left_bound && right <= right_bound && res != 0) {
		res = 0;
		middle = (left + right)/2;
		inner_idx = middle * inner_cols;

		for (i = 0; (res == 0) && (i < search_key_size); i++) {
			search_idx = search_key_indices[i];
			key_idx = key_indices[i];
			outer_cmp = outer_idx + search_idx;
			inner_cmp = inner_idx + key_idx;

			res = outer_table[outer_cmp].compare_withoutNull(inner_table[inner_cmp]);
		}

		right = (res < 0) ? (middle - 1) : right;
		left = (res > 0) ? (middle + 1) : left;
		middle = (res != 0) ? (-1) : middle;
	}

	res2 = res;
	for (left = middle - 1; (res == 0) && (left >= left_bound);) {
		inner_idx = left * inner_cols;
		for (j = 0; (res == 0) && (j < search_key_size); j++) {
			search_idx = search_key_indices[j];
			key_idx = key_indices[j];
			inner_cmp = inner_idx + key_idx;
			outer_cmp = outer_idx + search_idx;
			res = outer_table[outer_cmp].compare_withoutNull(inner_table[inner_cmp]);
		}
		left = (res == 0) ? (left - 1) : left;
	}
	left++;

	res = res2;
	for (right = middle + 1; (res == 0) && (right <= right_bound);) {
		inner_idx = right * inner_cols;
		for (j = 0; (res == 0) && (j < search_key_size); j++) {
			search_idx = search_key_indices[j];
			key_idx = key_indices[j];
			inner_cmp = inner_idx + key_idx;
			outer_cmp = outer_idx + search_idx;
			res = outer_table[outer_cmp].compare_withoutNull(inner_table[inner_cmp]);
		}
		right = (res == 0) ? (right + 1) : right;
	}
	right--;
	res = res2;

	*res_left = (res == 0) ? left : (-1);
	*res_right = (res == 0) ? right : (-1);

	return (res == 0);
}


__device__ bool binarySearchIdx2(const int * search_key_indices,
								const int search_key_size,
								const int * key_indices,
								const int key_index_size,
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

	int middle = -1, res, i, j, search_idx, key_idx, res2;
	int outer_idx = search_row * outer_cols;
	int search_keys[3], keys[3];
	int *search_ptr = search_keys, *keys_ptr = keys;
	GNValue *outer_ptr = outer_table + outer_idx, *inner_ptr, *base_inner = inner_table;

	for (i = 0; i < search_key_size; i++) {
		*search_ptr++ = search_key_indices[i];
		*keys_ptr++ = key_indices[i];
	}

	*res_left = *res_right = -1;
	res = -1;

	while (left <= right && res != 0) {
		res = 0;
		middle = (left + right) >> 1;
		inner_ptr = base_inner + middle * inner_cols;
		search_ptr = search_keys;
		keys_ptr = keys;

		for (i = 0; (res == 0) && (i < search_key_size); i++) {
			search_idx = *search_ptr++;
			key_idx = *keys_ptr++;

			res = (outer_ptr + search_idx)->compare_withoutNull(*(inner_ptr + key_idx));
		}

		right = (res < 0) ? (middle - 1) : right;
		left = (res > 0) ? (middle + 1) : left;
	}

	res2 = res;
	for (left = middle - 1; (res == 0) && (left >= left_bound);) {
		inner_ptr = base_inner + left * inner_cols;
		search_ptr = search_keys;
		keys_ptr = keys;

		for (j = 0; (res == 0) && (j < search_key_size); j++) {
			search_idx = *search_ptr++;
			key_idx = *keys_ptr++;
			res = (outer_ptr + search_idx)->compare_withoutNull(*(inner_ptr + key_idx));
		}
		left = (res == 0) ? (left - 1) : left;
	}
	left++;

	res = res2;
	for (right = middle + 1; (res == 0) && (right <= right_bound);) {
		inner_ptr = base_inner + right * inner_cols;
		search_ptr = search_keys;
		keys_ptr = keys;

		for (j = 0; (res == 0) && (j < search_key_size); j++) {
			search_idx = *search_ptr;
			key_idx = *keys_ptr++;
			res = (outer_ptr + search_idx)->compare_withoutNull(*(inner_ptr + key_idx));
		}
		right = (res == 0) ? (right + 1) : right;
	}
	right--;
	res = res2;

	*res_left = (res == 0) ? left : (-1);
	*res_right = (res == 0) ? right : (-1);

	return (res == 0);
}


__global__
void count(
          GNValue *outer_dev,
          GNValue *inner_dev,
          ulong *count_dev,
          ResBound *res_bound,
          uint outer_part_size,
          uint outer_cols,
          uint inner_part_size,
          uint inner_cols,
          uint gpu_size,
          GTreeNode *end_ex_dev,
          int end_size,
          GTreeNode *post_ex_dev,
          int post_size,
          int *search_key_indices,
          int search_keys_size,
          int *key_indices,
          int key_index_size
          )

{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	bool res = false;


	count_dev[x + k] = 0;
	res_bound[x + k].left = -1;
	res_bound[x + k].right = -1;

	if (x < outer_part_size) {
		res = binarySearchIdx2(search_key_indices,
								search_keys_size,
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

		count_dev[x + k] = (res && res_bound[x + k].right >= 0 && res_bound[x + k].left >= 0) ? (res_bound[x + k].right - res_bound[x + k].left + 1) : 0;
	}

	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		count_dev[x + k + 1] = 0;
	}
}


__global__ void join(GNValue *outer_dev,
						GNValue *inner_dev,
						RESULT *result_dev,
						ulong *count_dev,
						uint outer_part_size,
						uint outer_cols,
						uint inner_part_size,
						uint inner_cols,
						uint gpu_size,
						uint jr_size,
						GTreeNode *end_ex_dev,
						int end_size,
						GTreeNode *post_ex_dev,
						int post_size,
						ResBound *res_bound,
						int outer_base_idx,
						int inner_base_idx)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;


	if (x < outer_part_size) {
		ulong writeloc = count_dev[x + k];
		bool res = true;
		int res_left = -1, res_right = -1;

		res_left = res_bound[x + k].left;
		res_right = res_bound[x + k].right;

		while(res_left >= 0 && res_left <= res_right && writeloc < jr_size) {
			res = (post_size >= 1) ? evaluate4(post_ex_dev, post_size, outer_dev, inner_dev, x, res_left, outer_cols, inner_cols) : res;

			result_dev[writeloc].lkey = (res) ? (x + outer_base_idx) : (-1);
			result_dev[writeloc].rkey = (res) ? (res_left + inner_base_idx) : (-1);
			writeloc++;
			res_left++;
		}
	}
}
}
