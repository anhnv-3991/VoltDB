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
#include "gpunij.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/gexpression.h"
#include "utilities.h"


namespace voltdb {

GPUNIJ::GPUNIJ()
{
		result_ = NULL;
		result_size_ = 0;
		result_size_ = 0;
		all_time_ = 0;
}

GPUNIJ::GPUNIJ(GTable outer_table,
				GTable inner_table,
				ExpressionNode *pre_join_predicate,
				ExpressionNode *join_predicate,
				ExpressionNode *where_predicate)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	result_ = NULL;
	result_size_ = 0;
	all_time_ = 0;

	/**** Expression data ****/
	pre_join_predicate_ = GExpression(pre_join_predicate);
	join_predicate_ = GExpression(join_predicate);
	where_predicate_ = GExpression(where_predicate);
}

GPUNIJ::~GPUNIJ()
{
	freeArrays<RESULT>(result_);
	pre_join_predicate_.freeExpression();
	join_predicate_.freeExpression();
	where_predicate_.freeExpression();
}

bool GPUNIJ::join(){

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size = 0, jr_size2 = 0;
	RESULT *jresult_dev;
	ulong *count_psum;
	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;


	part_size = getPartitionSize();
	block_x = BLOCK_SIZE_X;
	block_y = BLOCK_SIZE_Y;
	grid_x = (part_size - 1)/block_x + 1;
	grid_y = (part_size - 1)/block_y + 1;
	block_y = 1;

	gpu_size = grid_x * grid_y * block_x * block_y + 1;

	/******** Allocate GPU buffer for table data and counting data *****/
	checkCudaErrors(cudaMalloc(&count_psum, gpu_size * sizeof(ulong)));

	/******* Allocate GPU buffer for join condition *********/
	struct timeval cstart, cend, pcstart, pcend, jstart, jend, end_join;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_table_.getBlockNum(); outer_idx++) {
		outer_table_.moveToIndex(outer_idx);

		for (uint inner_idx = 0; inner_idx < inner_table_.getBlockNum(); inner_idx++) {
			inner_table_.moveToIndex(inner_idx);

			/**** Copy IndexData to GPU memory ****/
			gettimeofday(&cstart, NULL);
			FirstEvaluation(count_psum);

			gettimeofday(&cend, NULL);

			gettimeofday(&pcstart, NULL);
			GUtilities::ExclusiveScan(count_psum, gpu_size, &jr_size);
			gettimeofday(&pcend, NULL);

			count_time_.push_back(GUtilities::timeDiff(cstart, cend));
			scan_time_.push_back(GUtilities::timeDiff(pcstart, pcend));


			if (jr_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only_.push_back(GUtilities::timeDiff(cstart, end_join));
				continue;
			}

			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
			gettimeofday(&jstart, NULL);
			SecondEvaluation(jresult_dev, count_psum);
			gettimeofday(&jend, NULL);

			join_time_.push_back((jend.tv_sec - jstart.tv_sec) * 1000000 + (jend.tv_usec - jstart.tv_usec));

			result_ = (RESULT *)realloc(result_, (result_size_ + jr_size) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);

			checkCudaErrors(cudaMemcpy(result_ + result_size_, jresult_dev, jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost));
			result_size_ += jr_size;
			jr_size = 0;

			joins_only_.push_back(GUtilities::timeDiff(cstart, end_join));
		}
	}


	/******** Free GPU memory, unload module, end session **************/
	checkCudaErrors(cudaFree(count_psum));
	gettimeofday(&all_end, NULL);

	all_time_ = GUtilities::timeDiff(all_start, all_end);
	return true;
}

void GPUNIJ::profiling()
{
	unsigned long allocation_time = 0, count_t = 0, join_t = 0, scan_t = 0, joins_only_time = 0;

	for (int i = 0; i < count_time_.size(); i++) {
		count_t += count_time_[i];
	}

	for (int i = 0; i < join_time_.size(); i++) {
		join_t += join_time_[i];
	}

	for (int i = 0; i < scan_time_.size(); i++) {
		scan_t += scan_time_[i];
	}

	for (int i = 0; i < joins_only_.size(); i++) {
		joins_only_time += joins_only_[i];
	}

	allocation_time = all_time_ - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"count Time: %lu\n"
			"Prefix Sum Time: %lu\n"
			"Join Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, count_t, scan_t, join_t, joins_only_time, all_time_);
}

void GPUNIJ::getResult(RESULT *output) const
{
	memcpy(output, result_, sizeof(RESULT) * result_size_);
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


extern "C" __global__ void firstEvaluation(GTable outer, GTable inner,
										int outer_rows, int inner_rows,
										ulong *pre_join_count,
										GExpression pre_join_pred, GExpression join_pred, GExpression where_pred,
										int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	bool res;
	int count = 0;
	GTuple outer_tuple, inner_tuple;

	for (int i = index; i < outer_rows; i += stride) {
		outer_tuple = outer.getGTuple(i);

		for (int j = 0; j < inner_rows; j++) {
			inner_tuple = inner.getGTuple(j);
			res = true;
			res = (res && pre_join_pred.getSize() > 0 && (pre_join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;
			res = (res && join_pred.getSize() > 0 && (join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;
			res = (res && where_pred.getSize() > 0 && (where_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;

			count += (res) ? 1 : 0;
		}
	}

	if (index < outer_rows)
		pre_join_count[index] = count;
	if (index == 0)
		pre_join_count[outer_rows] = 0;
}

void GPUNIJ::FirstEvaluation(ulong *first_count)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x));

	firstEvaluation<<<grid_x, block_x>>>(outer_table_, inner_table_,
										outer_table_.getCurrentRowNum(), inner_table_.getCurrentRowNum(),
										first_count,
										pre_join_predicate_, join_predicate_, where_predicate_,
										val_stack, type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

extern "C" __global__ void secondEvaluation(GTable outer, GTable inner,
											int outer_rows, int inner_rows,
											ulong *write_location, RESULT *output,
											GExpression pre_join_pred, GExpression join_pred, GExpression where_pred,
											int64_t *val_stack, ValueType *type_stack)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	bool res;
	GTuple outer_tuple, inner_tuple;

	for (int i = index; i < outer_rows; i += stride) {
		int location = write_location[i];

		outer_tuple = outer.getGTuple(i);

		for (int j = 0; j < inner_rows; j++) {
			inner_tuple = inner.getGTuple(j);

			res = true;
			res = (res && pre_join_pred.getSize() > 0 && (pre_join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;
			res = (res && join_pred.getSize() > 0 && (join_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;
			res = (res && where_pred.getSize() > 0 && (where_pred.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, stride)).isTrue()) ? true : false;

			output[location].lkey = (res) ? i : (-1);
			output[location].rkey = (res) ? j : (-1);
			location++;
		}
	}
}

void GPUNIJ::SecondEvaluation(RESULT *join_result, ulong *write_location)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1) / block_x + 1;

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x));

	secondEvaluation<<<grid_x, block_x>>>(outer_table_, inner_table_,
											outer_table_.getCurrentRowNum(), inner_table_.getCurrentRowNum(),
											write_location, join_result,
											pre_join_predicate_, join_predicate_, where_predicate_,
											val_stack, type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

template <typename T> void GPUNIJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}


//void GPUNIJ::debug(void)
//{
////	int outer_size = 0;
////
////	for (int i = 0; i < outer_table_.block_num; i++) {
////		outer_size += outer_table_.block_list[i].rows;
////	}
////
////	std::cout << "Size of outer table = " << outer_size << std::endl;
////	if (outer_size != 0) {
////		std::cout << "Outer table" << std::endl;
////		for (int i = 0; i < outer_tabler_.block_list->rows; i++) {
////			for (int j = 0; j < MAX_GNVALUE; j++) {
////				NValue tmp;
////				setNValue(&tmp, outer_table_.[i * outer_cols_ + j]);
////				std::cout << tmp.debug().c_str() << std::endl;
////			}
////		}
////	} else
////		std::cout << "Empty outer table" << std::endl;
////
////	std::cout << "Size of inner table =" << inner_size_ << std::endl;
////	if (inner_size_ != 0) {
////		for (int i = 0; i < inner_size_; i++) {
////			for (int j = 0; j < MAX_GNVALUE; j++) {
////				NValue tmp;
////				setNValue(&tmp, inner_table_[i * inner_cols_ + j]);
////				std::cout << tmp.debug().c_str() << std::endl;
////			}
////		}
////	} else
////		std::cout << "Empty inner table" << std::endl;
//
//	std::cout << "Size of preJoinPredicate = " << pre_join_predicate_.size << std::endl;
//	if (pre_join_predicate_.size != 0) {
//		std::cout << "Content of preJoinPredicate" << std::endl;
//		debugGTrees(pre_join_predicate_);
//	} else
//		std::cout << "Empty preJoinPredicate" << std::endl;
//
//	std::cout << "Size of joinPredicate = " << join_predicate_.size << std::endl;
//	if (join_predicate_.size != 0) {
//		std::cout << "Content of joinPredicate" << std::endl;
//		debugGTrees(join_predicate_);
//	} else
//		std::cout << "Empty joinPredicate" << std::endl;
//
//	std::cout << "Size of wherePredicate = " << where_predicate_.size << std::endl;
//	if (where_predicate_.size != 0) {
//		std::cout << "Content of wherePredicate" << std::endl;
//		debugGTrees(where_predicate_);
//	} else
//		std::cout << "Empty wherePredicate" << std::endl;
//}

}


