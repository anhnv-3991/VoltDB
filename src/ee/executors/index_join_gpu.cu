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

__device__ GNValue eval(GTreeNode *tree_expression, int tree_size, int root, IndexData outer_tuple, IndexData inner_tuple)
{
	return GNValue::getNullValue();
}
//CUDAH int flag_test;
__device__ bool pushStack(GNValue *stack, int *top, GNValue new_val)
{
	if (*top >= MAX_STACK_SIZE - 1) {
		printf("Error: Full GPU stack!\n");
		return false;
	}

	(*top)++;
	stack[*top] = new_val;

	return true;
}

__device__ GNValue popStack(GNValue *stack, int *top)
{
	if (*top < 0) {
		printf("Error: Empty GPU stack!\n");
		return GNValue::getNullValue();
	}
	GNValue retval = stack[*top];
	(*top)--;
	return retval;
}

//No more recursive
__device__ bool evaluate(GTreeNode *tree_expression, int tree_size, IndexData outer_tuple, IndexData inner_tuple, GNValue *stack)
{
	int top = -1;
	GNValue left, right;
	GTreeNode tmp_node;


	memset(stack, 0, MAX_STACK_SIZE);
	int i;
	for (i = 0; i < tree_size; i++) {
		//printf("tree_expression type = %d at i = %d\n", tree_expression[i].type, i);
		tmp_node = tree_expression[i];
		if (tmp_node.type == EXPRESSION_TYPE_VALUE_TUPLE) {
			if (tmp_node.column_idx >= MAX_GNVALUE || tmp_node.column_idx < 0)
				return false;

			if (tmp_node.tuple_idx == 0) {
				if (!pushStack(stack, &top, outer_tuple.gn[tmp_node.column_idx])) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}
			} else if (tmp_node.tuple_idx == 1) {
				if (!pushStack(stack, &top, inner_tuple.gn[tmp_node.column_idx])) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}
			} else {
				return false;
			}

			continue;
		} else if (tmp_node.type == EXPRESSION_TYPE_VALUE_CONSTANT) {
			if (!pushStack(stack, &top, tmp_node.value)) {
				printf("Error: Failed to push %d to stack! Exiting...\n, i");
				return false;
			}
			continue;
		}

		// Get left operand
		right = popStack(stack, &top);

		// Get right operand
		left = popStack(stack, &top);

		switch(tmp_node.type) {
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				if ((left.getValueType() != VALUE_TYPE_BOOLEAN || right.getValueType() != VALUE_TYPE_BOOLEAN))
					return false;

				if (!pushStack(stack, &top, left.op_and(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				if (left.getValueType() != VALUE_TYPE_BOOLEAN || right.getValueType() != VALUE_TYPE_BOOLEAN) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_or(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_equal(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_notEqual(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_lessThan(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_lessThanOrEqual(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_greaterThan(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				if (left.getValueType() != right.getValueType()) {
					return false;
				}

				if (!pushStack(stack, &top, left.op_greaterThanOrEqual(right))) {
					printf("Error: Failed to push %d to stack! Exiting...\n", i);
					return false;
				}

				break;
			}
			default: {
				return false;
			}
		}
	}

	return (popStack(stack, &top).isTrue()) ? true: false;
}

__device__ void binarySearchIdx(const int * __restrict__ search_key_indices,
								const int search_key_size,
								const int * __restrict__ key_indices,
								const int key_index_size,
								const IndexData search_key,
								const IndexData * __restrict__ search_array,
								int left_bound,
								int right_bound,
								int *res_left,
								int *res_right)
{
	int left = left_bound, right = right_bound;

	int middle = -1, res, i, j, search_idx, key_idx;
	GNValue tmp_search, tmp_idx;

	*res_left = *res_right = -1;

	while (left <= right && left >= 0) {
		res = 0;
		middle = (left + right)/2;

		if (middle < left_bound || middle > right_bound)
			break;

		for (i = 0; i < search_key_size; i++) {
			search_idx = search_key_indices[i];
			if (search_idx >= MAX_GNVALUE)
				return;
			//tmp_search = search_key.gn[search_idx];

			//if (tmp_search.getValueType() == VALUE_TYPE_INVALID || tmp_search.getValueType() == VALUE_TYPE_NULL) {
			if (search_key.gn[search_idx].getValueType() == VALUE_TYPE_INVALID || search_key.gn[search_idx].getValueType() == VALUE_TYPE_NULL) {
				middle = -1;
				break;
			}
			if (i >= key_index_size)
				return;
			key_idx = key_indices[i];

			if (key_idx >= MAX_GNVALUE)
				return;
			tmp_idx = search_array[middle].gn[key_idx];

			//if (tmp_idx.getValueType() == VALUE_TYPE_INVALID || tmp_idx.getValueType() == VALUE_TYPE_NULL) {
			if (search_array[middle].gn[key_idx].getValueType() == VALUE_TYPE_INVALID || search_array[middle].gn[key_idx].getValueType() == VALUE_TYPE_NULL) {
				middle = -1;
				break;
			}
			//tmp_idx.debug();
			//res = tmp_search.compare_withoutNull(tmp_idx);
			res = search_key.gn[search_idx].compare_withoutNull(search_array[middle].gn[key_idx]);
			if (res != 0)
				break;
		}

		if (res < 0) {
			right = middle - 1;
			middle = -1;
		} else if (res > 0) {
			left = middle + 1;
			middle = -1;
		} else {
			break;
		}
	}

	if (middle != -1) {
		for (left = middle - 1; left >= left_bound; left--) {
			for (j = 0; j < search_key_size; j++) {
				search_idx = search_key_indices[j];
				if (search_idx >= MAX_GNVALUE)
					return;
				//tmp_search = search_key.gn[search_idx];

				if (j >= key_index_size)
					return;
				key_idx = key_indices[j];
				if (key_idx >= MAX_GNVALUE)
					return;
				//tmp_idx = search_array[left].gn[key_idx];

				//res = tmp_search.compare_withoutNull(tmp_idx);
				res = search_key.gn[search_idx].compare_withoutNull(search_array[left].gn[key_idx]);
				if (res != 0)
					break;
			}
			if (res != 0)
				break;
		}
		left++;

		for (right = middle + 1; right <= right_bound; right++) {
			for (j = 0; j < search_key_size; j++) {
				search_idx = search_key_indices[j];
				if (search_idx >= MAX_GNVALUE)
					return;
				//tmp_search = search_key.gn[search_idx];

				if (j >= key_index_size)
					return;
				key_idx = key_indices[j];
				if (key_idx >= MAX_GNVALUE)
					return;
				//tmp_idx = search_array[right].gn[key_idx];

				//res = tmp_search.compare_withoutNull(tmp_idx);
				res = search_key.gn[search_idx].compare_withoutNull(search_array[right].gn[key_idx]);
				if (res != 0)
					break;
			}
			if (res != 0)
				break;
		}
		right--;
		*res_left = left;
		*res_right = right;
	}
}

__global__
void count(
          IndexData *outer_dev,
          IndexData *inner_dev,
          ulong *count_dev,
          ResBound *res_bound,
          uint outer_part_size,
          uint inner_part_size,
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
	//bool res = true;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y < inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);

	count_dev[x + k] = 0;
	res_bound[x + k].left = -1;
	res_bound[x + k].right = -1;

	left_bound = 0;
	right_bound = BLOCK_SIZE_Y - 1;
	if (x < outer_part_size) {
//	//A global memory read is very slow.So repeating values is stored register memory
		//GNValue tmp_stack[MAX_STACK_SIZE];
		int res_left = -1, res_right = -1, res_count = 0;

		binarySearchIdx(search_key_indices,
							search_keys_size,
							key_indices,
							key_index_size,
							outer_dev[x],
							inner_dev,
							left_bound,
							right_bound,
							&res_left,
							&res_right);

		if (res_left >= 0 && res_right >= 0 && res_right < inner_part_size && res_left <= res_right)
			res_count = res_right - res_left + 1;

		count_dev[x + k] = res_count;
		res_bound[x + k].left = res_left;
		res_bound[x + k].right = res_right;
	}

	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		count_dev[x + k + 1] = 0;
	}
}


__global__ void join(IndexData *outer_dev,
						IndexData *inner_dev,
						RESULT *result_dev,
						ulong *count_dev,
						uint outer_part_size,
						uint inner_part_size,
						uint gpu_size,
						GTreeNode *end_ex_dev,
						int end_size,
						GTreeNode *post_ex_dev,
						int post_size,
						ResBound *res_bound)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;

	if (x < outer_part_size) {
		GNValue tmp_stack[MAX_STACK_SIZE];
		IndexData tmp_outer = outer_dev[x];
		ulong writeloc = count_dev[x + k];
		bool res = true;
		int res_left = -1, res_right = -1;


		res_left = res_bound[x + k].left;
		res_right = res_bound[x + k].right;

		res_left = 1;
		res_right = 2;
		if (res_left >= 0 && res_right >= 0 && res_right < inner_part_size && x % 1000 == 0) {

			evaluate(post_ex_dev, post_size, tmp_outer, inner_dev[res_left], tmp_stack);
			while(res_left <= res_right) {
//				if (end_size > 0)
//					res = evaluate(end_ex_dev, end_size, tmp_outer, inner_dev[res_left], tmp_stack);

				if (post_size > 0)
					res = evaluate(post_ex_dev, post_size, tmp_outer, inner_dev[res_left], tmp_stack);

				if (res) {
					result_dev[writeloc].lkey = x;
					result_dev[writeloc].rkey = res_left;
				} else {
					result_dev[writeloc].lkey = -1;
					result_dev[writeloc].rkey = -1;
				}
				writeloc++;
				res_left++;
			}
		}
	}
}
}
