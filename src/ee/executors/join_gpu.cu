#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"
#include "join_gpu.h"


/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/


namespace voltdb {
extern "C" {
__device__ GNValue non_index_evaluate(GTreeNode *tree_expression,
										int tree_size,
										GNValue *outer_tuple,
										GNValue *inner_tuple,
										int64_t *stack,
										ValueType *gtype,
										int offset,
										int offset2)
{
	ValueType ltype, rtype;

	int top = 0;
	double left_d, right_d, res_d;
	int64_t left_i, right_i;

	for (int i = 0; i < tree_size; i++) {
		switch (tree_expression[i].type) {
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				if (tree_expression[i].tuple_idx == 0) {
					stack[top] = (outer_tuple + tree_expression[i].column_idx * offset)->getValue();
					gtype[top] = (outer_tuple + tree_expression[i].column_idx * offset)->getValueType();
				} else if (tree_expression[i].tuple_idx == 1) {
					stack[top] = (inner_tuple + tree_expression[i].column_idx * offset2)->getValue();
					gtype[top] = (inner_tuple + tree_expression[i].column_idx * offset2)->getValueType();

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

__global__ void prefixSumFilter(GNValue *outer_table,
									GNValue *inner_table,
									ulong *count_psum,
									int outer_part_size,
									int inner_part_size,
									GTreeNode *preJoinPred_dev,
									int preJoin_size,
									GTreeNode *joinPred_dev,
									int join_size,
									GTreeNode *where_dev,
									int where_size,
									int64_t *stack,
									ValueType *gtype)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	int i;
	GNValue res = GNValue::getTrue();

	count_psum[x + k] = 0;

	if (x < outer_part_size) {
		int matched_sum = 0;

		for (i = left_bound; i <= right_bound; i++) {
			res = GNValue::getTrue();
			res = (preJoin_size > 0) ? non_index_evaluate(preJoinPred_dev, preJoin_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;
			res = (join_size > 0 && res.isTrue()) ? non_index_evaluate(joinPred_dev, join_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;
			res = (where_size > 0 && res.isTrue()) ? non_index_evaluate(where_dev, where_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;

			matched_sum += (res.isTrue()) ? 1 : 0;
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
						ulong *count_psum,
						int outer_part_size,
						int inner_part_size,
						int jr_size,
						int outer_base_idx,
						int inner_base_idx,
						GTreeNode *preJoinPred_dev,
						int preJoin_size,
						GTreeNode *joinPred_dev,
						int join_size,
						GTreeNode *where_dev,
						int where_size,
						int64_t *stack,
						ValueType *gtype)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * gridDim.x * blockDim.x;
	int left_bound = BLOCK_SIZE_Y * blockIdx.y;
	int right_bound = (left_bound + BLOCK_SIZE_Y <= inner_part_size) ? (left_bound + BLOCK_SIZE_Y - 1) : (inner_part_size - 1);
	GNValue res = GNValue::getTrue();

	count_psum[x + k] = 0;

	if (x < outer_part_size) {
		int write_location = count_psum[x + k];

		for (int i = left_bound; i <= right_bound; i++) {
			res = GNValue::getTrue();
			res = (preJoin_size > 0) ? non_index_evaluate(preJoinPred_dev, preJoin_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;
			res = (join_size > 0 && res.isTrue()) ? non_index_evaluate(joinPred_dev, join_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;
			res = (where_size > 0 && res.isTrue()) ? non_index_evaluate(where_dev, where_size, outer_table + x, inner_table + i, stack + x, gtype + x, outer_part_size, inner_part_size) : res;

			jresult_dev[write_location].lkey = (res.isTrue()) ? (x + outer_base_idx) : -1;
			jresult_dev[write_location].rkey = (res.isTrue()) ? (i + inner_base_idx) : -1;
			write_location++;
		}
	}
}

void prefixSumFilterWrapper(int grid_x, int grid_y,
								int block_x, int block_y,
								GNValue *outer_dev, GNValue *inner_dev,
								ulong *count_psum,
								int outer_part_size,
								int inner_part_size,
								GTreeNode *preJoinPred_dev,
								int preJoin_size,
								GTreeNode *joinPred_dev,
								int join_size,
								GTreeNode *where_dev,
								int where_size,
								int64_t *val_stack,
								ValueType *type_stack)
{
	dim3 block(block_x, block_y, 1);
	dim3 grid(grid_x, grid_y, 1);
	prefixSumFilter<<<grid, block>>>(outer_dev, inner_dev,
										count_psum,
										outer_part_size,
										inner_part_size,
										preJoinPred_dev,
										preJoin_size,
										joinPred_dev,
										join_size,
										where_dev,
										where_size,
										val_stack,
										type_stack);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel launch (prefixSumFilter) failed. Error code: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}


void expFilterWrapper(int grid_x, int grid_y,
						int block_x, int block_y,
						GNValue *outer_dev, GNValue *inner_dev,
						RESULT *jresult_dev,
						ulong *count_psum,
						int outer_part_size, int inner_part_size,
						ulong jr_size,
						int outer_base_idx,
						int inner_base_idx,
						GTreeNode *preJoinPred_dev,
						int preJoin_size,
						GTreeNode *joinPred_dev,
						int join_size,
						GTreeNode *where_dev,
						int where_size,
						int64_t *val_stack,
						ValueType *type_stack)
{
	dim3 block(block_x, block_y, 1);
	dim3 grid(grid_x, grid_y, 1);
	join<<<grid, block>>>(outer_dev, inner_dev,
								jresult_dev,
								count_psum,
								outer_part_size,
								inner_part_size,
								jr_size,
								outer_base_idx,
								inner_base_idx,
								preJoinPred_dev,
								preJoin_size,
								joinPred_dev,
								join_size,
								where_dev,
								where_size,
								val_stack,
								type_stack);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error: Async kernel launch (expFilter) failed. Error code: %s\n", cudaGetErrorString(err));
	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void prefixSumWrapper(ulong *input, uint ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);

	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

}
}

