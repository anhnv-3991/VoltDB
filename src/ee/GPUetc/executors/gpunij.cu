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
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUNIJ.h"
#include "join_gpu.h"
#include "GPUetc/common/GNValue.h"


using namespace voltdb;

GPUNIJ::GPUNIJ()
{
		join_result_ = NULL;
		result_size_ = 0;

		outer_table_.block_list = NULL;
		outer_table_.block_num = 0;
		outer_table_.column_num = 0;
		outer_table_.schema = NULL;

		inner_table_.block_list = NULL;
		inner_table_.block_num = 0;
		inner_table_.column_num = 0;
		inner_table_.schema = NULL;

		pre_join_predicate_.exp = NULL;
		pre_join_predicate_.size = 0;
		join_predicate_.exp = NULL;
		join_predicate_.size = 0;
		where_predicate_.exp = NULL;
		where_predicate_.size = 0;
}

GPUNIJ::GPUNIJ(GTable outer_table,
				GTable inner_table,
				TreeExpression pre_join_predicate,
				TreeExpression join_predicate,
				TreeExpression where_predicate)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	join_result_ = NULL;
	result_size_ = 0;

	/**** Expression data ****/
	bool ret;

	assert(getTreeNodes(&(pre_join_predicate_.exp), pre_join_predicate));
	pre_join_predicate_.size = pre_join_predicate.getSize();


	assert(getTreeNodes(&(join_predicate_.exp), join_predicate));
	join_predicate_.size = join_predicate.getSize();

	assert(getTreeNodes(&(where_predicate_.exp), where_predicate));
	where_predicate_.size = where_predicate.getSize();
}

GPUNIJ::~GPUNIJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(pre_join_predicate_.exp);
	freeArrays<GTreeNode>(join_predicate_.exp);
	freeArrays<GTreeNode>(where_predicate_.exp);
}

bool GPUNIJ::join(){

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);
	if (outer_table_.block_list->rows == 0 || inner_table_.block_list->rows == 0) {
		return true;
	}

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size = 0, jr_size2 = 0;
	RESULT *jresult_dev;
	GTree pre_join_pred, join_pred, where_pred;
	ulong *count_psum;
	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;
	std::vector<unsigned long> allocation, count_time, scan_time, join_time, joins_only;

	GTable outer_chunk, inner_chunk;

	outer_chunk.column_num = outer_table_.column_num;
	outer_chunk.schema = outer_table_.schema;
	inner_chunk.column_num = inner_table_.column_num;
	inner_chunk.schema = outer_table_.schema;

	part_size = getPartitionSize();
	block_x = BLOCK_SIZE_X;
	block_y = BLOCK_SIZE_Y;
	grid_x = divUtility(part_size, block_x);
	grid_y = divUtility(part_size, block_y);
	block_y = 1;

	gpu_size = grid_x * grid_y * block_x * block_y + 1;

	/******** Allocate GPU buffer for table data and counting data *****/
	checkCudaErrors(cudaMalloc(&count_psum, gpu_size * sizeof(ulong)));

	/******* Allocate GPU buffer for join condition *********/
	if (pre_join_predicate_.size >= 1) {
		checkCudaErrors(cudaMalloc(&(pre_join_pred.exp), pre_join_predicate_.size * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(pre_join_pred.exp, pre_join_predicate_.exp, pre_join_predicate_.size * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (join_predicate_.size >= 1) {
		checkCudaErrors(cudaMalloc(&(join_pred.exp), join_predicate_.size * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(join_pred.exp, join_predicate_.exp, join_predicate_.size * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (where_predicate_.size >= 1) {
		checkCudaErrors(cudaMalloc(&(where_pred.exp), where_predicate_.size * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(where_pred.exp, where_predicate_.exp, where_predicate_.size * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	struct timeval cstart, cend, pcstart, pcend, jstart, jend, end_join;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_table_.block_num; outer_idx++) {
		outer_chunk.block_list = outer_table_.block_list + outer_idx;

		for (uint inner_idx = 0; inner_idx < inner_table_.block_num; inner_idx++) {
			inner_chunk.block_list = inner_table_.block_list + inner_idx;

			/**** Copy IndexData to GPU memory ****/
			gettimeofday(&cstart, NULL);
			prefixSumFilterWrapper(outer_chunk, inner_chunk, count_psum,
									pre_join_pred, join_pred, where_pred);

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
			expFilterWrapper(outer_chunk, inner_chunk,
								jresult_dev, count_psum,
								pre_join_pred, join_pred, where_pred);
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
	checkCudaErrors(cudaFree(count_psum));
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
//	uint outer_size = outer_table_.;
//	uint inner_size = inner_rows_;
//	uint bigger_tuple_size = (outer_size_ > inner_size_) ? outer_size_ : inner_size_;
//
//	if (bigger_tuple_size < part_size) {
//		return bigger_tuple_size;
//	}
//
//	for (uint i = 32768; i <= DEFAULT_PART_SIZE_; i = i * 2) {
//		if (bigger_tuple_size < i) {
//			part_size = i;
//			break;
//		}
//	}
//
//	printf("getPartitionSize: PART SIZE = %d\n", part_size);
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
//	int outer_size = 0;
//
//	for (int i = 0; i < outer_table_.block_num; i++) {
//		outer_size += outer_table_.block_list[i].rows;
//	}
//
//	std::cout << "Size of outer table = " << outer_size << std::endl;
//	if (outer_size != 0) {
//		std::cout << "Outer table" << std::endl;
//		for (int i = 0; i < outer_tabler_.block_list->rows; i++) {
//			for (int j = 0; j < MAX_GNVALUE; j++) {
//				NValue tmp;
//				setNValue(&tmp, outer_table_.[i * outer_cols_ + j]);
//				std::cout << tmp.debug().c_str() << std::endl;
//			}
//		}
//	} else
//		std::cout << "Empty outer table" << std::endl;
//
//	std::cout << "Size of inner table =" << inner_size_ << std::endl;
//	if (inner_size_ != 0) {
//		for (int i = 0; i < inner_size_; i++) {
//			for (int j = 0; j < MAX_GNVALUE; j++) {
//				NValue tmp;
//				setNValue(&tmp, inner_table_[i * inner_cols_ + j]);
//				std::cout << tmp.debug().c_str() << std::endl;
//			}
//		}
//	} else
//		std::cout << "Empty inner table" << std::endl;

	std::cout << "Size of preJoinPredicate = " << pre_join_predicate_.size << std::endl;
	if (pre_join_predicate_.size != 0) {
		std::cout << "Content of preJoinPredicate" << std::endl;
		debugGTrees(pre_join_predicate_);
	} else
		std::cout << "Empty preJoinPredicate" << std::endl;

	std::cout << "Size of joinPredicate = " << join_predicate_.size << std::endl;
	if (join_predicate_.size != 0) {
		std::cout << "Content of joinPredicate" << std::endl;
		debugGTrees(join_predicate_);
	} else
		std::cout << "Empty joinPredicate" << std::endl;

	std::cout << "Size of wherePredicate = " << where_predicate_.size << std::endl;
	if (where_predicate_.size != 0) {
		std::cout << "Content of wherePredicate" << std::endl;
		debugGTrees(where_predicate_);
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

void GPUNIJ::debugGTrees(const GTree tree)
{
	GTreeNode *expression = tree.exp;
	int size = tree.size;

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


