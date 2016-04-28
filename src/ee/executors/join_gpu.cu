#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
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

CUDAH bool evaluate(GTreeNode *tree_expression,
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

__global__ void count(GNValue *outer_table,
						GNValue *inner_table,
						int *count_psum,
						int outer_part_size,
						int outer_cols,
						int inner_part_size,
						int inner_cols,
						GTreeNode *preJoinPred_dev,
						int preJoin_size,
						GTreeNode *joinPred_dev,
						int join_size,
						GTreeNode *where_dev,
						int where_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	bool res, ret;
	int outer_index, inner_index;


	count_psum[x + k] = 0;

	if (x < outer_part_size) {
		int64_t stack[8];
		ValueType gtype[8];
		int matched_sum = 0;

		outer_index = (x + k) * outer_cols;
		inner_index = left_bound * inner_cols;

		for (int i = left_bound; i <= right_bound; i++, inner_index += inner_cols) {
			res = true;
			ret = false;
			ret = (preJoin_size > 0) ? evaluate(preJoinPred_dev, preJoin_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;
			ret = (join_size > 0) ? evaluate(joinPred_dev, join_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;
			ret = (where_size > 0) ? evaluate(where_dev, where_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;

			matched_sum += (res) ? 1 : 0;
		}
		count_psum[x + k] = matched_sum;
	}

	if (x + k == (blockDim.x * gridDim.x * gridDim.y - 1)) {
		count_psum[x + k + 1] = 0;
	}
}

__global__ void join(GNValue *outer_table,
						GNValue *inner_table,
						RESULT *jresult_dev,
						int *count_psum,
						int outer_part_size,
						int outer_cols,
						int inner_part_size,
						int inner_cols,
						int jr_size,
						int outer_base_idx,
						int inner_base_idx,
						GTreeNode *preJoinPred_dev,
						int preJoin_size,
						GTreeNode *joinPred_dev,
						int join_size,
						GTreeNode *where_dev,
						int where_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	bool res, ret;
	int outer_index, inner_index;


	count_psum[x + k] = 0;

	if (x < outer_part_size) {
		int64_t stack[8];
		ValueType gtype[8];
		int matched_sum = 0;
		int write_location = count_psum[x + k];

		outer_index = x * outer_cols;
		inner_index = left_bound * inner_cols;

		for (int i = left_bound; i <= right_bound; i++, inner_index += inner_cols) {
			res = true;
			ret = false;
			ret = (preJoin_size > 0) ? evaluate(preJoinPred_dev, preJoin_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;
			ret = (join_size > 0) ? evaluate(joinPred_dev, join_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;
			ret = (where_size > 0) ? evaluate(where_dev, where_size, outer_table, inner_table, outer_index, inner_index, stack, gtype) : ret;
			res = (ret && res) ? (bool)stack[0] : res;

			jresult_dev[write_location].lkey = (res) ? (x + outer_base_idx) : -1;
			jresult_dev[write_location].rkey = (res) ? (i + inner_base_idx) : -1;
			write_location++;
		}
	}
}
}
