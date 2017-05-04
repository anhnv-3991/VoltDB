#include "gpuhj.h"
#include "common/types.h"
#include "GPUetc/storage/gtable.h"

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
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>
#include <math.h>


#include <inttypes.h>

#include "utilities.h"

namespace voltdb {



const uint64_t GPUHJ::MAX_BUCKETS[] = {
	        3,				//0
	        7,				//1
	        13,				//2
	        31,				//3
	        61,				//4
	        127,			//5
	        251,			//6
	        509,			//7
	        1021,			//8
	        2039,			//9
	        4093,			//10
	        8191,			//11
	        16381,			//12
	        32749,			//13
	        65521,			//14
	        131071,			//15
	        262139,			//16
	        524287,			//17
	        1048573,		//18
	        2097143,		//19
	        4194301,		//20
	        8388593,		//21
	        16777213,
	        33554393,
	        67108859,
	        134217689,
	        268435399,
	        536870909,
	        1073741789,
	        2147483647,
	        4294967291,
	        8589934583
	};

GPUHJ::GPUHJ()
{
		join_result_ = NULL;
		result_size_ = 0;
		search_exp_size_ = NULL;
		search_exp_num_ = 0;
		maxNumberOfBuckets_ = 0;

		search_exp_ = NULL;
		m_sizeIndex_ = 0;
		lookup_type_ = INDEX_LOOKUP_TYPE_EQ;
}

GPUHJ::GPUHJ(GTable outer_table,
				GTable inner_table,
				std::vector<TreeExpression> search_exp,
				std::vector<int> indices,
				TreeExpression end_expression,
				TreeExpression post_expression,
				TreeExpression initial_expression,
				TreeExpression skipNullExpr,
				TreeExpression prejoin_expression,
				TreeExpression where_expression,
				IndexLookupType lookup_type,
				int mSizeIndex)
{
	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	join_result_ = NULL;
	result_size_ = 0;
	search_exp_num_ = search_exp.size();
	lookup_type_ = lookup_type;
	m_sizeIndex_ = mSizeIndex;

	//Fix the size of bucket at 16
	maxNumberOfBuckets_ = MAX_BUCKETS[m_sizeIndex_];

	printf("New M_SIZE_INDEX = %d\n", m_sizeIndex_);

	int tmp_size = 0;

	int *search_exp_size_tmp = (int *)malloc(sizeof(int) * search_exp_num_);
	assert(search_exp_size_tmp != NULL);
	for (int i = 0; i < search_exp_num_; i++) {
		search_exp_size_tmp[i] = search_exp[i].getSize();
		tmp_size += search_exp_size_tmp[i];
	}

	GTreeNode *search_exp_tmp = (GTreeNode *)malloc(sizeof(GTreeNode) * tmp_size);
	assert(search_exp_tmp != NULL);
	GTreeNode *exp_ptr = search_exp_tmp;
	for (int i = 0; i < search_exp_num_; i++) {
		getTreeNodes(exp_ptr, search_exp[i]);
		exp_ptr += search_exp_size_tmp[i];
	}

	checkCudaErrors(cudaMalloc(&search_exp_, tmp_size * sizeof(GTreeNode)));
	checkCudaErrors(cudaMalloc(&search_exp_size_, search_exp_num_ * sizeof(int)));

	checkCudaErrors(cudaMemcpy(search_exp_, search_exp_tmp, tmp_size * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(search_exp_size_, search_exp_size_tmp, search_exp_num_ * sizeof(int), cudaMemcpyHostToDevice));

	free(search_exp_size_tmp);
	free(search_exp_tmp);

	/**** Expression data ****/

	end_expression_ = GExpression(end_expression);
	post_expression_ = GExpression(post_expression);
	initial_expression_ = GExpression(initial_expression);
	skipNullExpr_ = GExpression(skipNullExpr);
	prejoin_expression_ = GExpression(prejoin_expression);
	where_expression_ = GExpression(where_expression);
}

GPUHJ::~GPUHJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(search_exp_);
	freeArrays<int>(search_exp_size_);
	end_expression_.freeExpression();
	post_expression_.freeExpression();
	initial_expression_.freeExpression();
	skipNullExpr_.freeExpression();
	prejoin_expression_.freeExpression();
	where_expression_.freeExpression();
}

template <typename T> void GPUHJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}

void GPUHJ::getResult(RESULT *output) const
{
	memcpy(output, join_result_, sizeof(RESULT) * result_size_);
}

int GPUHJ::getResultSize() const
{
	return result_size_;
}

bool GPUHJ::getTreeNodes(GTreeNode *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() >= 1)
		tree_expression.getNodesArray(expression);

	return true;
}

void GPUHJ::debug(void)
{
//	std::cout << "Size of outer table = " << outer_rows_ << std::endl;
//	if (outer_rows_ != 0) {
//		std::cout << "Outer table" << std::endl;
//		for (int i = 0; i < outer_rows_; i++) {
//			for (int j = 0; j < MAX_GNVALUE; j++) {
//				NValue tmp;
//				setNValue(&tmp, outer_table_[i * outer_cols_ + j]);
//				std::cout << tmp.debug().c_str() << std::endl;
//			}
//		}
//	} else
//		std::cout << "Empty outer table" << std::endl;
//
//	std::cout << "Size of inner table =" << inner_rows_ << std::endl;
//	if (inner_rows_ != 0) {
//		for (int i = 0; i < inner_rows_; i++) {
//			for (int j = 0; j < MAX_GNVALUE; j++) {
//				NValue tmp;
//				setNValue(&tmp, inner_table_[i * inner_cols_ + j]);
//				std::cout << tmp.debug().c_str() << std::endl;
//			}
//		}
//	} else
//		std::cout << "Empty inner table" << std::endl;
//
//	std::cout << "Size of end_expression = " << end_size_ << std::endl;
//	if (end_size_ != 0) {
//		std::cout << "Content of end_expression" << std::endl;
//		debugGTrees(end_expression_, end_size_);
//	} else
//		std::cout << "Empty end expression" << std::endl;
//
//	std::cout << "Size of post_expression = " << post_size_ << std::endl;
//	if (post_size_ != 0) {
//		std::cout << "Content of post_expression" << std::endl;
//		debugGTrees(post_expression_, post_size_);
//	} else
//		std::cout << "Empty post expression" << std::endl;
//
//	std::cout << "Size of initial_expression = " << initial_size_ << std::endl;
//	if (initial_size_ != 0) {
//		std::cout << "Content of initial_expression" << std::endl;
//		debugGTrees(initial_expression_, initial_size_);
//	} else
//		std::cout << "Empty initial expression" << std::endl;
//
//	std::cout << "Size of skip null expression = " << skipNull_size_ << std::endl;
//	if (skipNull_size_ != 0) {
//		std::cout << "Content of skip null_expression" << std::endl;
//		debugGTrees(skipNullExpr_, skipNull_size_);
//	} else
//		std::cout << "Empty skip null expression" << std::endl;
//
//	std::cout << "Size of prejoin_expression = " << prejoin_size_ << std::endl;
//	if (prejoin_size_ != 0) {
//		std::cout << "Content of prejoin_expression" << std::endl;
//		debugGTrees(prejoin_expression_, prejoin_size_);
//	} else
//		std::cout << "Empty prejoin expression " << std::endl;
//
//	std::cout << "Size of where expression = " << where_size_ << std::endl;
//	if (where_size_ != 0) {
//		std::cout << "Content of where_expression" << std::endl;
//		debugGTrees(where_expression_, where_size_);
//	} else
//		std::cout << "Empty where expression" << std::endl;
//
//	std::cout << "Size of search_exp_ array = " << search_exp_num_ << std::endl;
//	int search_exp_ptr = 0;
//	if (search_exp_num_ != 0) {
//		std::cout << "Content of search_exp" << std::endl;
//		for (int i = 0; i < search_exp_num_; i++) {
//			std::cout << "search_exp[" << i << std::endl;
//			debugGTrees(search_exp_ + search_exp_ptr, search_exp_size_[i]);
//			search_exp_ptr += search_exp_size_[i];
//		}
//	} else
//		std::cout << "Empty search keys array" << std::endl;
//
//	std::cout << "Size of innner_indices = " << indices_size_ << std::endl;
//	if (indices_size_ != 0) {
//		std::cout << "Content of inner indices" << std::endl;
//		for (int i = 0; i < indices_size_; i++) {
//			std::cout << "indices[" << i << "] = " << indices_[i] << std::endl;
//		}
//	} else
//		std::cout << "Empty indices array" << std::endl;
}




uint GPUHJ::getPartitionSize() const
{
//	return PART_SIZE_;
	uint part_size = DEFAULT_PART_SIZE_;
//	uint outer_size = outer_rows_;
//	uint inner_size = inner_rows_;
//	uint bigger_tuple_size = (outer_size > inner_size) ? outer_size : inner_size;
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


bool GPUHJ::join()
{


	checkCudaErrors(cudaProfilerStart());
	ulong *index_count, jr_size;
	RESULT *jresult_dev;
	struct timeval start_all, end_all;

	int partition_size;

	struct timeval index_count_start, index_count_end, prefix_start, prefix_end, join_start, join_end, rebalance_start, rebalance_end, remove_start, remove_end;

	gettimeofday(&start_all, NULL);

	/******* Hash the outer table *******/

	partition_size = getPartitionSize();
	checkCudaErrors(cudaMalloc(&index_count, sizeof(ulong) * (partition_size + 1)));

	ResBound *in_bound;

	checkCudaErrors(cudaMalloc(&in_bound, sizeof(ResBound) * partition_size));

	printf("Start Joining\n");

	for (int outer_idx = 0; outer_idx < outer_table_.getBlockNum(); outer_idx++) {
		outer_table_.moveToIndex(outer_idx);

		for (int inner_idx = 0; inner_idx < inner_table_.getBlockNum(); inner_idx++) {
			inner_table_.moveToIndex(inner_idx);

			gettimeofday(&index_count_start, NULL);
			IndexCount(index_count, in_bound);
			gettimeofday(&index_count_end, NULL);

			index_hcount_.push_back(GUtilities::timeDiff(index_count_start, index_count_end));

			RESULT *tmp_bound, *out_bound;
			ulong out_size;
			ulong *exp_psum;

			gettimeofday(&rebalance_start, NULL);
#ifdef DECOMPOSED1_
			Rebalance(index_count, in_bound, &tmp_bound, outer_table_.getCurrentRowNum(), &out_size);
#else
			Rebalance2(index_count, in_bound, &tmp_bound, outer_table_.getCurrentRowNum() + 1, &out_size);
#endif
			gettimeofday(&rebalance_end, NULL);
			rebalance_cost_.push_back(GUtilities::timeDiff(rebalance_start, rebalance_end));

			if (out_size == 0) {
				continue;
			}

			printf("out_size = %lu\n", out_size);
			checkCudaErrors(cudaMalloc(&exp_psum, (out_size + 1) * sizeof(ulong)));
			checkCudaErrors(cudaMalloc(&out_bound, out_size * sizeof(RESULT)));

			gettimeofday(&join_start, NULL);
			HashJoinLegacy(tmp_bound, out_bound, exp_psum, out_size);
			gettimeofday(&join_end, NULL);
			join_time_.push_back(GUtilities::timeDiff(join_start, join_end));

			gettimeofday(&prefix_start, NULL);
			GUtilities::ExclusiveScan(exp_psum, out_size + 1, &jr_size);
			gettimeofday(&prefix_end, NULL);

			prefix_sum_.push_back(GUtilities::timeDiff(prefix_start, prefix_end));

			checkCudaErrors(cudaFree(tmp_bound));

			if (jr_size == 0) {
				checkCudaErrors(cudaFree(exp_psum));
				checkCudaErrors(cudaFree(out_bound));
				continue;
			}

			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));

			gettimeofday(&remove_start, NULL);
			GUtilities::RemoveEmptyResult(jresult_dev, out_bound, exp_psum, out_size);
			gettimeofday(&remove_end, NULL);
			remove_empty_.push_back(GUtilities::timeDiff(remove_start, remove_end));

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));

			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));
#ifdef DECOMPOSED1_
			checkCudaErrors(cudaFree(exp_psum));
			checkCudaErrors(cudaFree(out_bound));
#endif
			checkCudaErrors(cudaFree(jresult_dev));
			result_size_ += jr_size;
			jr_size = 0;
		}
	}

	gettimeofday(&end_all, NULL);

	checkCudaErrors(cudaFree(index_count));

	checkCudaErrors(cudaProfilerStop());

	total_ = GUtilities::timeDiff(start_all, end_all);
	return true;
}

void GPUHJ::profiling()
{
	unsigned long index_count_final, prefix_sum_final, join_final;
	unsigned long rebalance_final, remove_empty_total;

	index_count_final = 0;
	for (int i = 0; i < index_hcount_.size(); i++) {
		index_count_final += index_hcount_[i];
	}

	prefix_sum_final = 0;
	for (int i = 0; i < prefix_sum_.size(); i++) {
		prefix_sum_final += prefix_sum_[i];
	}

	rebalance_final = 0;
	for (int i = 0; i < rebalance_cost_.size(); i++) {
		rebalance_final += rebalance_cost_[i];
	}

	remove_empty_total = 0;
	for (int i = 0; i < remove_empty_.size(); i++) {
		remove_empty_total += remove_empty_[i];
	}

	join_final = 0;
	for (int i = 0; i < join_time_.size(); i++) {
		join_final += join_time_[i];
	}

	ulong join_total, data_copy;

	join_total = index_count_final + prefix_sum_final + join_final + remove_empty_total;

	data_copy = total_ - join_total;

	printf("\n*** Execution time *****************************\n"
			"index Count: %lu\n"
			"prefix_sum: %lu\n"
			"Join: %lu\n"
			"*************************************************\n"
#ifdef DECOMPOSED1_
			"Rebalance total: %lu\n"
#endif
			"Exp evaluation: %lu\n"
			"Remove empty total: %lu\n"
			"Data copy: %lu\n"
			"Total time: %lu\n", index_count_final, prefix_sum_final, join_final,
#ifdef DECOMPOSED1_
								rebalance_final,
#endif
								join_total, remove_empty_total, data_copy, total_);

}

extern "C" __global__ void EvaluateSearchPredicate(GTable outer_table, GTreeNode *search_keys,
													int outer_rows,
													int *search_size, int search_num,
													int64_t *val_stack, ValueType *type_stack, GTable output, GHashIndex output_index)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	GTuple tuple_res, outer_tuple;

	for (int i = index; i < outer_rows; i += stride) {
		tuple_res = output.getGTuple(i);
		outer_tuple = outer_table.getGTuple(i);

		for (int j = 0, search_ptr = 0; j < search_num; search_ptr += search_size[j], j++) {
			GExpression search_exp(search_keys + search_ptr, search_size[j]);
			GNValue eval_result = search_exp.evaluate(&outer_tuple, NULL, val_stack, type_stack, stride);

			tuple_res.insertColumnValue(eval_result, j);
		}

		output_index.insertKeyTupleNoSort(tuple_res, i);
	}
}

extern "C" __global__ void indexCount(GHashIndex outer_index, GHashIndex inner_index, ulong *index_count, ResBound *out_bound)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int outer_rows = outer_index.getKeyRows();

	for (int i = index; i < outer_rows; i += stride) {
		GHashIndexKey key(outer_index, i);
		int bucket_id = key.KeyHasher();

		out_bound[i].left = inner_index.getBucketLocation(bucket_id);
		out_bound[i].right = inner_index.getBucketLocation(bucket_id + 1);

		index_count[i] = out_bound[i].right - out_bound[i].left + 1;
	}
}

void GPUHJ::IndexCount(ulong *index_count, ResBound *out_bound)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	GColumnInfo *search_schema;

	checkCudaErrors(cudaMalloc(&search_schema, sizeof(GColumnInfo) * search_exp_num_));
	GTable search_table(outer_table_.getDatabaseId(), NULL, search_schema, search_exp_num_, outer_table_.getCurrentRowNum());
	GHashIndex tmp_index(outer_table_.getCurrentRowNum(), search_exp_num_);

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * outer_rows * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * outer_rows * MAX_STACK_SIZE));
	EvaluateSearchPredicate<<<grid_x, block_x>>>(outer_table_, search_exp_,
													outer_table_.getCurrentRowNum(),
													search_exp_size_, search_exp_num_,
													val_stack, type_stack, search_table, tmp_index);
	GIndex tmp_idx = inner_table_.getIndex();
	GHashIndex *inner_index = dynamic_cast<GHashIndex*>(&tmp_idx);
	indexCount<<<grid_x, block_x>>>(tmp_index, *inner_index, index_count, out_bound);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(search_schema));
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

void GPUHJ::IndexCount(ulong *index_count, ResBound *out_bound, cudaStream_t stream)
{
	int outer_rows = outer_table_.getCurrentRowNum();
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

	GTable search_table(outer_table_.getDatabaseId(), NULL, search_exp_num_);

	GHashIndex tmp_index(outer_table_.getCurrentRowNum(), search_exp_num_);

	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * outer_rows * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * outer_rows * MAX_STACK_SIZE));

	EvaluateSearchPredicate<<<grid_x, block_x, 0, stream>>>(outer_table_, search_exp_,
																outer_table_.getCurrentRowNum(),
																search_exp_size_, search_exp_num_,
																val_stack, type_stack, search_table, tmp_index);
	GIndex tmp_idx = inner_table_.getIndex();
	GHashIndex *inner_index = dynamic_cast<GHashIndex*>(&tmp_idx);
	indexCount<<<grid_x, block_x, 0, stream>>>(tmp_index, *inner_index, index_count, out_bound);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	search_table.removeTable();
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

extern "C" __global__ void hashJoinLegacy(GTable outer, GTable inner,
											RESULT *in_bound, RESULT *out_bound,
											ulong *mark_location, int size,
											GExpression end_exp, GExpression post_exp, GExpression where_exp,
											int64_t *val_stack, ValueType *type_stack
											)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GNValue res;
	GTuple outer_tuple, inner_tuple;

	for (int i = index; i < size; i += offset) {
		outer_tuple = outer.getGTuple(in_bound[i].lkey);
		inner_tuple = inner.getGTuple(in_bound[i].rkey);
		res = GNValue::getTrue();

		res = (end_exp.getSize() > 0) ? end_exp.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, offset) : res;
		res = (post_exp.getSize() > 0 && res.isTrue()) ? post_exp.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, offset) : res;
		res = (where_exp.getSize() > 0 && res.isTrue()) ? where_exp.evaluate(&outer_tuple, &inner_tuple, val_stack + index, type_stack + index, offset) : res;

		out_bound[i].lkey = (res.isTrue()) ? in_bound[i].lkey : (-1);
		out_bound[i].rkey = (res.isTrue()) ? in_bound[i].rkey : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void GPUHJ::HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size < partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;


	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	hashJoinLegacy<<<grid_size, block_size>>>(outer_table_, inner_table_,
												in_bound, out_bound,
												mark_location, size,
												end_expression_, post_expression_, where_expression_,
												val_stack,
												type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}

void GPUHJ::HashJoinLegacy(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size < partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;


	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	hashJoinLegacy<<<grid_size, block_size, 0, stream>>>(outer_table_, inner_table_,
															in_bound, out_bound,
															mark_location, size,
															end_expression_, post_expression_, where_expression_,
															val_stack,
															type_stack);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
}


__global__ void HDecompose(RESULT *output, ResBound *in_bound, int *sorted_idx, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		output[i].lkey = in_bound[in_location[i]].outer;
		output[i].rkey = sorted_idx[in_bound[in_location[i]].left + local_offset[i]];
	}
}

void GPUHJ::decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	GIndex tmp_index = inner_table_.getIndex();
	GHashIndex *inner_idx = dynamic_cast<GHashIndex *>(&tmp_index);
	int *sorted_idx = inner_idx->getSortedIdx();

	HDecompose<<<grid_size, block_size>>>(output, in_bound, sorted_idx, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GPUHJ::decompose(RESULT *output, ResBound *in_bound, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	GIndex tmp_index = inner_table_.getIndex();
	GHashIndex *inner_idx = dynamic_cast<GHashIndex *>(&tmp_index);
	int *sorted_idx = inner_idx->getSortedIdx();

	HDecompose<<<grid_size, block_size, 0, stream>>>(output, in_bound, sorted_idx, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

void GPUHJ::Rebalance2(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	GUtilities::ExclusiveScan(index_count, in_size, out_size);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaMemset(location, 0, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaDeviceSynchronize());


	GUtilities::MarkLocation(location, index_count, in_size);


	GUtilities::InclusiveScan(location, *out_size);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::ComputeOffset(index_count, location, local_offset, *out_size);

	decompose(*out_bound, in_bound, location, local_offset, *out_size);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void GPUHJ::Rebalance2(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	GUtilities::ExclusiveScan(index_count, in_size, out_size, stream);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));
	checkCudaErrors(cudaMemsetAsync(location, 0, sizeof(ulong) * (*out_size), stream));

	GUtilities::MarkLocation(location, index_count, in_size, stream);

	GUtilities::InclusiveScan(location, *out_size, stream);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::ComputeOffset(index_count, location, local_offset, *out_size, stream);

	decompose(*out_bound, in_bound, location, local_offset, *out_size, stream);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void GPUHJ::Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{

	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *mark;
	ulong size_no_zeros;
	ResBound *tmp_bound;
	ulong sum;

	/* Remove zeros elements */
	ulong *no_zeros;

	checkCudaErrors(cudaMalloc(&mark, (in_size + 1) * sizeof(ulong)));

	GUtilities::MarkNonZeros(index_count, in_size, mark);

	GUtilities::ExclusiveScan(mark, in_size + 1, &size_no_zeros);

	if (size_no_zeros == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));

		return;
	}

	checkCudaErrors(cudaMalloc(&no_zeros, (size_no_zeros + 1) * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&tmp_bound, size_no_zeros * sizeof(ResBound)));

	GUtilities::RemoveZeros(index_count, in_bound, no_zeros, tmp_bound, mark, in_size);

	GUtilities::ExclusiveScan(no_zeros, size_no_zeros + 1, &sum);

	if (sum == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));
		checkCudaErrors(cudaFree(no_zeros));
		checkCudaErrors(cudaFree(tmp_bound));

		return;
	}

	ulong *tmp_location, *local_offset;

	checkCudaErrors(cudaMalloc(&tmp_location, sum * sizeof(ulong)));
	checkCudaErrors(cudaMemset(tmp_location, 0, sizeof(ulong) * sum));
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::MarkTmpLocation(tmp_location, no_zeros, size_no_zeros);

	GUtilities::InclusiveScan(tmp_location, sum);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	GUtilities::ComputeOffset(no_zeros, tmp_location, local_offset, sum);
	decompose(*out_bound, tmp_bound, tmp_location, local_offset, sum);

	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));

}

void GPUHJ::Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{

	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ulong *mark;
	ulong size_no_zeros;
	ResBound *tmp_bound;
	ulong sum;

	/* Remove zeros elements */
	ulong *no_zeros;

	checkCudaErrors(cudaMalloc(&mark, (in_size + 1) * sizeof(ulong)));

	GUtilities::MarkNonZeros(index_count, in_size, mark, stream);

	GUtilities::ExclusiveScan(mark, in_size + 1, &size_no_zeros, stream);

	if (size_no_zeros == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));

		return;
	}

	checkCudaErrors(cudaMalloc(&no_zeros, (size_no_zeros + 1) * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&tmp_bound, size_no_zeros * sizeof(ResBound)));

	GUtilities::RemoveZeros(index_count, in_bound, no_zeros, tmp_bound, mark, in_size, stream);

	GUtilities::ExclusiveScan(no_zeros, size_no_zeros + 1, &sum, stream);

	if (sum == 0) {
		*out_size = 0;
		checkCudaErrors(cudaFree(mark));
		checkCudaErrors(cudaFree(no_zeros));
		checkCudaErrors(cudaFree(tmp_bound));

		return;
	}

	ulong *tmp_location, *local_offset;

	checkCudaErrors(cudaMalloc(&tmp_location, sum * sizeof(ulong)));
	checkCudaErrors(cudaMemset(tmp_location, 0, sizeof(ulong) * sum));
	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::MarkTmpLocation(tmp_location, no_zeros, size_no_zeros, stream);

	GUtilities::InclusiveScan(tmp_location, sum, stream);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	GUtilities::ComputeOffset(no_zeros, tmp_location, local_offset, sum, stream);
	decompose(*out_bound, tmp_bound, tmp_location, local_offset, sum, stream);

	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));

}
}
