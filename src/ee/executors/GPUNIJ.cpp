#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "GPUNIJ.h"
#include "join_gpu.h"
#include "scan_common.h"
#include "common/types.h"
//#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"


using namespace voltdb;

GPUNIJ::GPUNIJ()
{
		outer_table_ = inner_table_ = NULL;
		outer_size_ = inner_size_ = 0;
		outer_rows_ = inner_rows_ = 0;
		outer_cols_ = inner_cols_ = 0;
		join_result_ = NULL;
		result_size_ = 0;
		preJoin_size_ = 0;
		join_size_ = 0;
		where_size_ = 0;

		preJoinPredicate_ = NULL;
		joinPredicate_ = NULL;
		wherePredicate_ = NULL;
}

GPUNIJ::GPUNIJ(GNValue *outer_table,
				GNValue *inner_table,
				int outer_rows,
				int outer_cols,
				int inner_rows,
				int inner_cols,
				TreeExpression preJoinPredicate,
				TreeExpression joinPredicate,
				TreeExpression wherePredicate)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	outer_size_ = outer_rows_ = outer_rows;
	inner_size_ = inner_rows_ = inner_rows;
	outer_cols_ = outer_cols;
	inner_cols_ = inner_cols;
	join_result_ = NULL;
	result_size_ = 0;
	preJoin_size_ = preJoinPredicate.getSize();
	join_size_ = joinPredicate.getSize();
	where_size_ = wherePredicate.getSize();

	/**** Expression data ****/
	bool ret;

	ret = getTreeNodes(&preJoinPredicate_, preJoinPredicate);
	assert(ret == true);

	ret = getTreeNodes(&joinPredicate_, joinPredicate);
	assert(ret == true);

	ret = getTreeNodes(&wherePredicate_, wherePredicate);
	assert(ret == true);
}

GPUNIJ::~GPUNIJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(preJoinPredicate_);
	freeArrays<GTreeNode>(joinPredicate_);
	freeArrays<GTreeNode>(wherePredicate_);
}

bool GPUNIJ::join(){
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction count, join;
	CUmodule module;
	char fname[256];
	char *vd;
	char path[256];

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);
	if (outer_size_ == 0 || inner_size_ == 0) {
		return true;
	}


	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size = 0, jr_size2 = 0;
	GNValue *outer_dev, *inner_dev;
	RESULT *jresult_dev;
	GTreeNode *preJoinPred_dev, *joinPred_dev, *where_dev;
	ulong *count_psum;
	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;
	std::vector<unsigned long> allocation, count_time, scan_time, join_time, joins_only;
	int64_t *val_stack;
	ValueType *type_stack;

	part_size = getPartitionSize();
	block_x = BLOCK_SIZE_X;
	block_y = BLOCK_SIZE_Y;
	grid_x = divUtility(part_size, block_x);
	grid_y = divUtility(part_size, block_y);
	block_y = 1;

	gpu_size = grid_x * grid_y * block_x * block_y + 1;
	if (gpu_size > MAX_LARGE_ARRAY_SIZE) {
		gpu_size = MAX_LARGE_ARRAY_SIZE * divUtility(gpu_size, MAX_LARGE_ARRAY_SIZE);
	} else if (gpu_size > MAX_SHORT_ARRAY_SIZE) {
		gpu_size = MAX_SHORT_ARRAY_SIZE * divUtility(gpu_size, MAX_SHORT_ARRAY_SIZE);
	} else {
		gpu_size = MAX_SHORT_ARRAY_SIZE;
	}

	printf("part_size = %d, outer_cols = %d, inner_cols = %d\n", part_size, outer_cols_, inner_cols_);

	/******** Allocate GPU buffer for table data and counting data *****/
	//res = cuMemAlloc(&outer_dev, part_size * sizeof(IndexData));
	checkCudaErrors(cudaMalloc(&outer_dev, part_size * outer_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&inner_dev, part_size * inner_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&count_psum, gpu_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&val_stack, block_x * grid_x * MAX_STACK_SIZE * sizeof(int64_t)));
	checkCudaErrors(cudaMalloc(&type_stack, block_x * grid_x * MAX_STACK_SIZE * sizeof(ValueType)));

	/******* Allocate GPU buffer for join condition *********/
	if (preJoin_size_ >= 1) {
		checkCudaErrors(cudaMalloc(&preJoinPred_dev, preJoin_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(preJoinPred_dev, preJoinPredicate_, preJoin_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (join_size_ >= 1) {
		checkCudaErrors(cudaMalloc(&joinPred_dev, join_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(joinPred_dev, joinPredicate_, join_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (where_size_ >= 1) {
		checkCudaErrors(cudaMalloc(&where_dev, where_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(where_dev, wherePredicate_, where_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	struct timeval cstart, cend, pcstart, pcend, jstart, jend, end_join;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_size_; outer_idx += part_size) {
		//Size of outer small table
		uint outer_part_size = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);

		block_x = (outer_part_size < BLOCK_SIZE_X) ? outer_part_size : BLOCK_SIZE_X;
		grid_x = divUtility(outer_part_size, block_x);

		checkCudaErrors(cudaMemcpy(outer_dev, outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));

		for (uint inner_idx = 0; inner_idx < inner_size_; inner_idx += part_size) {
			//Size of inner small table
			uint inner_part_size = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			block_y = (inner_part_size < BLOCK_SIZE_Y) ? inner_part_size : BLOCK_SIZE_Y;
			grid_y = divUtility(inner_part_size, block_y);
			block_y = 1;
			gpu_size = block_x * block_y * grid_x * grid_y + 1;

			/**** Copy IndexData to GPU memory ****/
			checkCudaErrors(cudaMemcpy(inner_dev, inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));
			gettimeofday(&cstart, NULL);
			prefixSumFilterWrapper(grid_x, grid_y, block_x, block_y,
									outer_dev, inner_dev,
									count_psum,
									outer_part_size, inner_part_size,
									preJoinPred_dev, preJoin_size_,
									joinPred_dev, join_size_,
									where_dev, where_size_,
									val_stack, type_stack);

			gettimeofday(&cend, NULL);

			gettimeofday(&pcstart, NULL);
			prefixSumWrapper(count_psum, gpu_size, &jr_size);
			gettimeofday(&pcend, NULL);

			count_time.push_back((cend.tv_sec - cstart.tv_sec) * 1000000 + (cend.tv_usec - cstart.tv_usec));
			scan_time.push_back((pcend.tv_sec - pcstart.tv_sec) * 1000000 + (pcend.tv_usec - pcstart.tv_usec));

			if (jr_size < 0) {
				printf("Scanning failed\n");
				return false;
			}

			if (jr_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back((end_join.tv_sec - cstart.tv_sec) * 1000000 + (end_join.tv_usec - cstart.tv_usec));
				continue;
				//goto free_count;
			}

			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
			gettimeofday(&jstart, NULL);
			expFilterWrapper(grid_x, grid_y, block_x, block_y,
								outer_dev, inner_dev,
								jresult_dev,
								count_psum,
								outer_part_size, inner_part_size,
								jr_size,
								outer_idx, inner_idx,
								preJoinPred_dev, preJoin_size_,
								joinPred_dev, join_size_,
								where_dev, where_size_,
								val_stack, type_stack);
			gettimeofday(&jend, NULL);

			join_time.push_back((jend.tv_sec - jstart.tv_sec) * 1000000 + (jend.tv_usec - jstart.tv_usec));

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);

			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost));
			result_size_ += jr_size;
			jr_size = 0;

			joins_only.push_back((end_join.tv_sec - cstart.tv_sec) * 1000000 + (end_join.tv_usec - cstart.tv_usec));
		}
	}


	/******** Free GPU memory, unload module, end session **************/
	checkCudaErrors(cudaFree(outer_dev));
	checkCudaErrors(cudaFree(inner_dev));
	checkCudaErrors(cudaFree(count_psum));
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
	gettimeofday(&all_end, NULL);

	unsigned long allocation_time = 0, count_t = 0, join_t = 0, ipsum_time = 0, scan_t = 0, joins_only_time = 0, all_time = 0;
	for (int i = 0; i < count_time.size(); i++) {
		count_t += count_time[i];
	}

	for (int i = 0; i < join_time.size(); i++) {
		join_t += join_time[i];
	}

	for (int i = 0; i < scan_time.size(); i++) {
		scan_t += scan_time[i];
	}

	for (int i = 0; i < joins_only.size(); i++) {
		joins_only_time += joins_only[i];
	}

	all_time = (all_end.tv_sec - all_start.tv_sec) * 1000000 + (all_end.tv_usec - all_start.tv_usec);

	allocation_time = all_time - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"count Time: %lu\n"
			"Prefix Sum Time: %lu\n"
			"Join Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, count_t, scan_t, join_t, joins_only_time, all_time);
	printf("End of join\n");
	return true;
}

void GPUNIJ::getResult(RESULT *output) const
{
	memcpy(output, join_result_, sizeof(RESULT) * result_size_);
}

int GPUNIJ::getResultSize() const
{
	return result_size_;
}

uint GPUNIJ::getPartitionSize() const
{
//	return PART_SIZE_;
	uint part_size = DEFAULT_PART_SIZE_;
	uint outer_size = outer_rows_;
	uint inner_size = inner_rows_;
	uint bigger_tuple_size = (outer_size_ > inner_size_) ? outer_size_ : inner_size_;

	if (bigger_tuple_size < part_size) {
		return bigger_tuple_size;
	}

	for (uint i = 32768; i <= DEFAULT_PART_SIZE_; i = i * 2) {
		if (bigger_tuple_size < i) {
			part_size = i;
			break;
		}
	}

	printf("getPartitionSize: PART SIZE = %d\n", part_size);
	return part_size;
}


uint GPUNIJ::divUtility(uint dividend, uint divisor) const
{
	return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

bool GPUNIJ::getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression)
{
	int tmp_size = tree_expression.getSize();
	if (tmp_size >= 1) {
		*expression = (GTreeNode *)malloc(sizeof(GTreeNode) * tmp_size);
		if (expression == NULL) {
			std::cout << "Error: malloc(expression) failed." << std::endl;
			return false;
		}
		tree_expression.getNodesArray(*expression);
	} else {
		*expression = NULL;
	}

	return true;
}

bool GPUNIJ::getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() >= 1)
		tree_expression.getNodesArray(expression);

	return true;
}

template <typename T> void GPUNIJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}


void GPUNIJ::debug(void)
{
	std::cout << "Size of outer table = " << outer_size_ << std::endl;
	if (outer_size_ != 0) {
		std::cout << "Outer table" << std::endl;
		for (int i = 0; i < outer_size_; i++) {
			for (int j = 0; j < MAX_GNVALUE; j++) {
				NValue tmp;
				setNValue(&tmp, outer_table_[i * outer_cols_ + j]);
				std::cout << tmp.debug().c_str() << std::endl;
			}
		}
	} else
		std::cout << "Empty outer table" << std::endl;

	std::cout << "Size of inner table =" << inner_size_ << std::endl;
	if (inner_size_ != 0) {
		for (int i = 0; i < inner_size_; i++) {
			for (int j = 0; j < MAX_GNVALUE; j++) {
				NValue tmp;
				setNValue(&tmp, inner_table_[i * inner_cols_ + j]);
				std::cout << tmp.debug().c_str() << std::endl;
			}
		}
	} else
		std::cout << "Empty inner table" << std::endl;

	std::cout << "Size of preJoinPredicate = " << preJoin_size_ << std::endl;
	if (preJoin_size_ != 0) {
		std::cout << "Content of preJoinPredicate" << std::endl;
		debugGTrees(preJoinPredicate_, preJoin_size_);
	} else
		std::cout << "Empty preJoinPredicate" << std::endl;

	std::cout << "Size of joinPredicate = " << join_size_ << std::endl;
	if (join_size_ != 0) {
		std::cout << "Content of joinPredicate" << std::endl;
		debugGTrees(joinPredicate_, join_size_);
	} else
		std::cout << "Empty joinPredicate" << std::endl;

	std::cout << "Size of wherePredicate = " << where_size_ << std::endl;
	if (where_size_ != 0) {
		std::cout << "Content of wherePredicate" << std::endl;
		debugGTrees(wherePredicate_, where_size_);
	} else
		std::cout << "Empty wherePredicate" << std::endl;
}

void GPUNIJ::setNValue(NValue *nvalue, GNValue &gnvalue)
{
	double tmp = gnvalue.getMdata();
	char gtmp[16];
	memcpy(gtmp, &tmp, sizeof(double));
	nvalue->setMdataFromGPU(gtmp);
//	nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
	nvalue->setValueTypeFromGPU(gnvalue.getValueType());
}

void GPUNIJ::debugGTrees(const GTreeNode *expression, int size)
{
	std::cout << "DEBUGGING INFORMATION..." << std::endl;
	for (int index = 0; index < size; index++) {
		switch (expression[index].type) {
			case EXPRESSION_TYPE_CONJUNCTION_AND: {
				std::cout << "[" << index << "] CONJUNCTION AND" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_CONJUNCTION_OR: {
				std::cout << "[" << index << "] CONJUNCTION OR" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_EQUAL: {
				std::cout << "[" << index << "] COMPARE EQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_NOTEQUAL: {
				std::cout << "[" << index << "] COMPARE NOTEQUAL" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHAN: {
				std::cout << "[" << index << "] COMPARE LESS THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHAN: {
				std::cout << "[" << index << "] COMPARE GREATER THAN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE LESS THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO: {
				std::cout << "[" << index << "] COMPARE GREATER THAN OR EQUAL TO" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_LIKE: {
				std::cout << "[" << index << "] COMPARE LIKE" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_COMPARE_IN: {
				std::cout << "[" << index << "] COMPARE IN" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_TUPLE: {
				std::cout << "[" << index << "] TUPLE(";
				std::cout << expression[index].column_idx << "," << expression[index].tuple_idx;
				std::cout << ")" << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_CONSTANT: {
				NValue tmp;
				GNValue tmp_gnvalue = expression[index].value;

				setNValue(&tmp, tmp_gnvalue);
				std::cout << "[" << index << "] VALUE TUPLE = " << tmp.debug().c_str()  << std::endl;
				break;
			}
			case EXPRESSION_TYPE_VALUE_NULL:
			case EXPRESSION_TYPE_INVALID:
			default: {
				std::cout << "NULL value" << std::endl;
				break;
			}
		}
	}
}
