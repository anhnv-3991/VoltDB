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

__device__ GNValue eval(GTreeNode *tree_expression,
					int tree_size,
					int root_index,
					IndexData outer_tuple,
					IndexData inner_tuple)
{
	if (root_index == 0 || root_index >= tree_size) {
		return GNValue::getNullValue();
	}

	if (tree_expression[root_index].type == EXPRESSION_TYPE_INVALID) {
		return GNValue::getNullValue();
	}

	GTreeNode tmp_node = tree_expression[root_index];

	switch (tmp_node.type) {
		case EXPRESSION_TYPE_VALUE_CONSTANT: {
			return tmp_node.value;
		}
		case EXPRESSION_TYPE_VALUE_TUPLE: {
			if (tmp_node.tuple_idx == 0) {
				return outer_tuple.gn[tmp_node.column_idx];
			} else if (tmp_node.tuple_idx == 1) {
				return inner_tuple.gn[tmp_node.column_idx];
			}

			return GNValue::getNullValue();
		}
		default: {
			break;
		}
	}

	GNValue eval_left, eval_right;

	eval_left = eval(tree_expression, tree_size, root_index * 2, outer_tuple, inner_tuple);
	eval_right = eval(tree_expression, tree_size, root_index * 2 + 1, outer_tuple, inner_tuple);

	switch (tmp_node.type) {
		case EXPRESSION_TYPE_CONJUNCTION_AND: {
			if ((eval_left.getValueType() != VALUE_TYPE_BOOLEAN || eval_right.getValueType() != VALUE_TYPE_BOOLEAN)) {
				return GNValue::getNullValue();
			}

			return eval_left.op_and(eval_right);
		}
		case EXPRESSION_TYPE_CONJUNCTION_OR: {
			if (eval_left.getValueType() != VALUE_TYPE_BOOLEAN || eval_right.getValueType() != VALUE_TYPE_BOOLEAN) {
				return GNValue::getNullValue();
			}

			return eval_left.op_or(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_EQUAL: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_equal(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_notEqual(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_lessThan(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_lessThanOrEqual(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_lessThanOrEqual(eval_right);
		}
		case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
			if (eval_left.getValueType() != eval_right.getValueType()) {
				return GNValue::getNullValue();
			}

			return eval_left.op_greaterThanOrEqual(eval_right);
		}
		default: {
			break;
		}

	}

	return GNValue::getNullValue();
}

__device__ int binarySearchIdx(const int *search_key_indices,
						int search_key_size,
						const int *key_indices,
						int key_index_size,
						const IndexData search_key,
						const IndexData *search_array,
						int left_boundary,
						int right_boundary,
						int size_of_array)
{
	if (left_boundary < 0 || right_boundary >= size_of_array || left_boundary > right_boundary) {
		return -1;
	}

	int middle = (left_boundary + right_boundary) / 2;

	int res = 0, search_res = -1;

	for (int i = 0; i < search_key_size; i++) {
		res = search_key.gn[i].compare_withoutNull(search_array[middle].gn[key_indices[i]]);
		if (res != 0)
			break;
	}

	if (res > 0) {
		search_res = binarySearchIdx(search_key_indices,
							search_key_size,
							key_indices,
							key_index_size,
							search_key,
							search_array,
							middle + 1,
							right_boundary,
							size_of_array);
	}

	if (res < 0) {
		search_res = binarySearchIdx(search_key_indices,
							search_key_size,
							key_indices,
							key_index_size,
							search_key,
							search_array,
							left_boundary,
							middle - 1,
							size_of_array);
	}

	search_res = middle;

	return search_res;
}

__global__
void count(
          IndexData *outer_dev,
          IndexData *inner_dev,
          ulong *count_dev,
          uint outer_part_size,
          uint inner_part_size,
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
//
//	__shared__ IndexData tmp_inner[BLOCK_SIZE_Y];

	/**
	TO DO : tmp_inner shoud be stored in parallel.but I have bug.
	*/

//	if (threadIdx.x == 0) {
//		for (uint j = 0; j < BLOCK_SIZE_Y && BLOCK_SIZE_Y * blockIdx.y + j < inner_part_size; j++) {
//			tmp_inner[j] = inner_dev[BLOCK_SIZE_Y * blockIdx.y + j];
//		}
//	}

//	__syncthreads();

	count_dev[x + k] = 0;

	if (x < outer_part_size) {
	//A global memory read is very slow.So repeating values is stored register memory
		IndexData tmp_outer = outer_dev[x];
//		int tmp_inner_size = inner_part_size;

		// Search for candidate value
		int candidate_idx = binarySearchIdx(search_key_indices,
																search_keys_size,
																key_indices,
																key_index_size,
																tmp_outer,
																inner_dev,
																0,
																BLOCK_SIZE_Y - 1,
																BLOCK_SIZE_Y);
		if (candidate_idx != -1) {
			GNValue res = eval(end_ex_dev, end_size, 1, tmp_outer, inner_dev[candidate_idx]);

			if (res.isTrue()) {
				res = eval(post_ex_dev, post_size, 1, tmp_outer, inner_dev[candidate_idx]);
				if (res.isTrue()) {
					count_dev[x + k] = 1;
				}
			}
		}
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
						GTreeNode *end_ex_dev,
						int end_size,
						GTreeNode *post_ex_dev,
						int post_size,
						int *search_key_indices,
						int search_keys_size,
						int *key_indices,
						int key_index_size)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;

//	__shared__ IndexData tmp_inner[BLOCK_SIZE_Y];
//
//	if (threadIdx.x == 0) {
//		for (uint j = 0; j < BLOCK_SIZE_Y && BLOCK_SIZE_Y * blockIdx.y + j < inner_part_size; j++) {
//			tmp_inner[j] = inner_dev[BLOCK_SIZE_Y * blockIdx.y + j];
//		}
//	}
//
//	__syncthreads();

	if (x < outer_part_size) {
		IndexData tmp_outer = outer_dev[x];
//		int tmp_inner_size = inner_part_size;
		ulong writeloc = count_dev[x + k];

		int candidate_idx = binarySearchIdx(search_key_indices,
																search_keys_size,
																key_indices,
																key_index_size,
																tmp_outer,
																inner_dev,
																0,
																BLOCK_SIZE_Y - 1,
																BLOCK_SIZE_Y);
		if (candidate_idx != -1) {
			GNValue res = eval(end_ex_dev, end_size, 1, tmp_outer, inner_dev[candidate_idx]);

			if (res.isTrue()) {
				res = eval(post_ex_dev, post_size, 1, tmp_outer, inner_dev[candidate_idx]);

				if (res.isTrue()) {
					result_dev[writeloc].lkey = tmp_outer.num;
					result_dev[writeloc].rkey = inner_dev[candidate_idx].num;
				}
			}
		}
	}
}

}
