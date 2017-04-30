#include "GPUHJ.h"
#include "ghash.h"
#include "common/types.h"
#include "gpu_common.h"
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
		outer_table_.block_list = inner_table_.block_list = NULL;
		join_result_ = NULL;
		result_size_ = 0;
		indices_size_ = 0;
		search_exp_size_ = NULL;
		search_exp_num_ = 0;
		indices_ = NULL;
		maxNumberOfBuckets_ = 0;

		search_exp_ = NULL;
		end_expression_.exp = NULL;
		end_expression_.size = 0;
		post_expression_.exp = NULL;
		post_expression_.size = 0;
		initial_expression_.exp = NULL;
		initial_expression_.size = 0;
		skipNullExpr_.exp = NULL;
		skipNullExpr_.size = 0;
		prejoin_expression_.exp = NULL;
		prejoin_expression_.size = 0;
		where_expression_.exp = NULL;
		where_expression_.size = 0;
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
	indices_size_ = indices.size();
	lookup_type_ = lookup_type;
	m_sizeIndex_ = mSizeIndex;

	//Fix the size of bucket at 16
	maxNumberOfBuckets_ = MAX_BUCKETS[m_sizeIndex_];

	printf("New M_SIZE_INDEX = %d\n", m_sizeIndex_);

	bool ret = true;
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
		getTreeNodes2(exp_ptr, search_exp[i]);
		exp_ptr += search_exp_size_tmp[i];
	}

	checkCudaErrors(cudaMalloc(&search_exp_, tmp_size * sizeof(GTreeNode)));
	checkCudaErrors(cudaMalloc(&search_exp_size_, search_exp_num_ * sizeof(int)));
	checkCudaErrors(cudaMalloc(&indices_, indices_size_ * sizeof(int)));

	checkCudaErrors(cudaMemcpy(search_exp_, search_exp_tmp, tmp_size * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(search_exp_size_, search_exp_size_tmp, search_exp_num_ * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(indices_, &indices[0], indices_size_ * sizeof(int), cudaMemcpyHostToDevice));

	free(search_exp_size_tmp);
	free(search_exp_tmp);

	/**** Expression data ****/

	assert(getTreeNodes(&end_expression_, end_expression));

	assert(getTreeNodes(&post_expression_, post_expression));

	assert(getTreeNodes(&initial_expression_, initial_expression));

	assert(getTreeNodes(&skipNullExpr_, skipNullExpr));

	assert(getTreeNodes(&prejoin_expression_, prejoin_expression));

	assert(getTreeNodes(&where_expression_, where_expression));

	int size = 0;

	for (int i = 0; i < indices_size_; i++) {
		switch(inner_table_.schema[indices_[i]].data_type) {
		case VALUE_TYPE_TINYINT: {
			size += sizeof(int8_t);
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			size += sizeof(int16_t);
			break;
		}
		case VALUE_TYPE_INTEGER: {
			size += sizeof(int32_t);
			break;
		}
		case VALUE_TYPE_BIGINT: {
			size += sizeof(int64_t);
			break;
		}
		}
	}

	keySize_ = (size - 1)/8 + 1;
	printf("KEYSIZE = %d\n", keySize_);
}

bool GPUHJ::getTreeNodes(GTree *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() > 0) {
		checkCudaErrors(cudaMalloc(&expression->exp, tree_expression.getSize() * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(expression->exp, tree_expression.getNodesArray2(), tree_expression.getSize() * sizeof(GTreeNode), cudaMemcpyHostToDevice));
		expression->size = tree_expression.getSize();
	}

	return true;
}

GPUHJ::~GPUHJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(search_exp_);
	freeArrays<int>(search_exp_size_);
	freeArrays<int>(indices_);
	freeArrays2(end_expression_);
	freeArrays2(post_expression_);
	freeArrays2(initial_expression_);
	freeArrays2(skipNullExpr_);
	freeArrays2(where_expression_);
}

template <typename T> void GPUHJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}

void GPUHJ::freeArrays2(GTree expression)
{
	if (expression.size > 0) {
		checkCudaErrors(cudaFree(expression.exp));
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

bool GPUHJ::getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression)
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
	return false;
//
//	checkCudaErrors(cudaProfilerStart());
//	GTable outer_chunk, inner_chunk;
//	ulong *index_count, jr_size;
//	RESULT *jresult_dev;
//	int block_x, grid_x, grid_y;
//	struct timeval start_all, end_all;
//#if defined(FUNC_CALL_) && defined(POST_EXP_)
//	GNValue *stack;
//#elif defined(POST_EXP_)
//	int64_t *val_stack;
//	ValueType *type_stack;
//#endif
//	int partition_size, size_of_buckets, bucketStride;
//	double tmp; //For calculating block size and grid size in power of 2
//	struct timeval inner_pack_start, inner_pack_end, inner_hash_count_start, inner_hash_count_end, inner_hash_start, inner_hash_end, inner_prefix_start, inner_prefix_end;
//	std::vector<unsigned long> inner_pack, inner_hasher;
//
//	struct timeval outer_pack_start, outer_pack_end, outer_hash_count_start, outer_hash_count_end, outer_prefix_start, outer_prefix_end, outer_hash_start, outer_hash_end;
//	std::vector<unsigned long> outer_pack, outer_hasher;
//
//	struct timeval index_count_start, index_count_end, prefix_start, prefix_end, join_start, join_end, rebalance_start, rebalance_end, remove_start, remove_end;
//	std::vector<unsigned long> index_hcount, prefix_sum, join_time, rebalance_cost, remove_empty;
//
//	gettimeofday(&start_all, NULL);
//
//	/******* Hash the outer table *******/
//#ifdef METHOD_1_
//	partition_size = getPartitionSize();
//	GHashNode *inner_hash_host;
//	GHashNode outer_hash_dev, inner_hash_dev;
//	bool *inner_hashed;
//	int part_num_inner;
//
//	part_num_inner = (inner_rows_ - 1)/partition_size + 1;
////	tmp = part_num_inner;
////	m_sizeIndex_ -= (int)log2(tmp);
////
////	maxNumberOfBuckets_ = MAX_BUCKETS[m_sizeIndex_];
//	printf("New m_sizeIndex = %d\n", m_sizeIndex_);
//	inner_hash_host = (GHashNode *)malloc(sizeof(GHashNode) * part_num_inner);
//	inner_hashed = (bool*)malloc(sizeof(bool) * part_num_inner);
//
//
//	outer_hash_dev.bucket_num = maxNumberOfBuckets_;
//	outer_hash_dev.key_size = keySize_;
//
//
//	for (int i = 0; i < part_num_inner; i++) {
//		inner_hashed[i] = false;
//		inner_hash_host[i].bucket_num = maxNumberOfBuckets_;
//		inner_hash_host[i].key_size = keySize_;
//		inner_hash_host[i].size = partition_size;
//		inner_hash_host[i].bucket_location = (int*)malloc(sizeof(int) * (maxNumberOfBuckets_ + 1));
//		inner_hash_host[i].hashed_idx = (int*)malloc(sizeof(int) * partition_size);
//		inner_hash_host[i].hashed_key = (uint64_t*)malloc(sizeof(uint64_t) * partition_size * keySize_);
//	}
//
//	inner_hash_dev.bucket_num = maxNumberOfBuckets_;
//	inner_hash_dev.key_size = keySize_;
//
//	checkCudaErrors(cudaMalloc(&outer_key, sizeof(uint64_t) * partition_size * keySize_));
//	checkCudaErrors(cudaMalloc(&(outer_hash_dev.hashed_idx), sizeof(int) * partition_size));
//	checkCudaErrors(cudaMalloc(&(outer_hash_dev.hashed_key), sizeof(uint64_t) * partition_size * keySize_));
//	checkCudaErrors(cudaMalloc(&(outer_hash_dev.bucket_location), sizeof(int) * (maxNumberOfBuckets_ + 1)));
//
//	checkCudaErrors(cudaMalloc(&inner_key, sizeof(uint64_t) * partition_size * keySize_));
//	checkCudaErrors(cudaMalloc(&(inner_hash_dev.hashed_idx), sizeof(int) * partition_size));
//	checkCudaErrors(cudaMalloc(&(inner_hash_dev.hashed_key), sizeof(uint64_t) * partition_size * keySize_));
//	checkCudaErrors(cudaMalloc(&(inner_hash_dev.bucket_location), sizeof(int) * (maxNumberOfBuckets_ + 1)));
//
//	tmp = (outer_rows_ - 1)/maxNumberOfBuckets_ + 1;
//	size_of_buckets = (int)pow(2, (int)(log2(tmp)));
//
//	checkCudaErrors(cudaMalloc(&index_count, sizeof(ulong) * (partition_size + 1)));
//	checkCudaErrors(cudaMalloc(&outer_dev, sizeof(GNValue) * partition_size * outer_cols_));
//	checkCudaErrors(cudaMalloc(&inner_dev, sizeof(GNValue) * partition_size * inner_cols_));
//
//	ResBound *in_bound;
//
//	checkCudaErrors(cudaMalloc(&in_bound, sizeof(ResBound) * partition_size));
//
//	printf("Start Joining\n");
//	for (int base_outer_idx = 0; base_outer_idx < outer_rows_; base_outer_idx += partition_size) {
//		/*** Hash the outer partition ***/
//		int outer_part_size = (base_outer_idx + partition_size < outer_rows_) ? partition_size : (outer_rows_ - base_outer_idx);
//
//		checkCudaErrors(cudaMemcpy(outer_dev, outer_table_ + base_outer_idx * outer_cols_, sizeof(GNValue) * outer_part_size * outer_cols_, cudaMemcpyHostToDevice));
//
//		gettimeofday(&outer_pack_start, NULL);
//		PackSearchKeyWrapper(outer_dev, outer_part_size, outer_cols_, outer_key, search_exp_dev, search_exp_size, search_exp_num_, keySize_);
//		gettimeofday(&outer_pack_end, NULL);
//		outer_pack.push_back(timeDiff(outer_pack_start, outer_pack_end));
//
//		gettimeofday(&outer_hash_start, NULL);
//		outer_hash_dev.size = outer_part_size;
//		GhashWrapper(outer_key, outer_hash_dev);
//		gettimeofday(&outer_hash_end, NULL);
//		outer_hasher.push_back(timeDiff(outer_hash_start, outer_hash_end));
//
//		for (int base_inner_idx = 0, j = 0; base_inner_idx < inner_rows_; base_inner_idx += partition_size, j++) {
//			int inner_part_size = (base_inner_idx + partition_size < inner_rows_) ? partition_size : (inner_rows_ - base_inner_idx);
//
//			checkCudaErrors(cudaMemcpy(inner_dev, inner_table_ + base_inner_idx * inner_cols_, sizeof(GNValue) * inner_part_size * inner_cols_, cudaMemcpyHostToDevice));
//
//			if (!inner_hashed[j]) {
//				inner_hashed[j] = true;
//				gettimeofday(&inner_pack_start, NULL);
//				PackKeyWrapper(inner_dev, inner_part_size, inner_cols_, indices_dev, indices_size_, inner_key, keySize_);
//				gettimeofday(&inner_pack_end, NULL);
//				inner_pack.push_back(timeDiff(inner_pack_start, inner_pack_end));
//
//				gettimeofday(&inner_hash_start, NULL);
//				inner_hash_dev.size = inner_part_size;
//				GhashWrapper(inner_key, inner_hash_dev);
//				gettimeofday(&inner_hash_end, NULL);
//				inner_hasher.push_back(timeDiff(inner_hash_start, inner_hash_end));
//
//				checkCudaErrors(cudaMemcpy(inner_hash_host[j].bucket_location, inner_hash_dev.bucket_location, sizeof(int) * (maxNumberOfBuckets_ + 1), cudaMemcpyDeviceToHost));
//				checkCudaErrors(cudaMemcpy(inner_hash_host[j].hashed_idx, inner_hash_dev.hashed_idx, sizeof(int) * inner_part_size, cudaMemcpyDeviceToHost));
//				checkCudaErrors(cudaMemcpy(inner_hash_host[j].hashed_key, inner_hash_dev.hashed_key, sizeof(uint64_t) * inner_part_size * keySize_, cudaMemcpyDeviceToHost));
//				inner_hash_host[j].size = inner_part_size;
//			} else {
//				checkCudaErrors(cudaMemcpy(inner_hash_dev.bucket_location, inner_hash_host[j].bucket_location, sizeof(int) * (maxNumberOfBuckets_ + 1), cudaMemcpyHostToDevice));
//				checkCudaErrors(cudaMemcpy(inner_hash_dev.hashed_idx, inner_hash_host[j].hashed_idx, sizeof(int) * inner_part_size, cudaMemcpyHostToDevice));
//				checkCudaErrors(cudaMemcpy(inner_hash_dev.hashed_key, inner_hash_host[j].hashed_key, sizeof(uint64_t) * inner_part_size * keySize_, cudaMemcpyHostToDevice));
//				inner_hash_dev.size = inner_hash_host[j].size;
//			}
//
//			gettimeofday(&index_count_start, NULL);
//#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
//			IndexCountWrapper(outer_hash_dev, inner_hash_dev, index_count, partition_size + 1);
//#else
//			IndexCountWrapper2(outer_hash_dev, inner_hash_dev, index_count, in_bound);
//#endif
//			gettimeofday(&index_count_end, NULL);
//			index_hcount.push_back(timeDiff(index_count_start, index_count_end));
//
//#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
//			gettimeofday(&prefix_start, NULL);
//			ExclusiveScanWrapper(index_count, partition_size + 1, &jr_size);
//			gettimeofday(&prefix_end, NULL);
//			prefix_sum.push_back(timeDiff(prefix_start, prefix_end));
//
//			if (jr_size < 0) {
//				printf("Scanning failed\n");
//				return false;
//			}
//
//			if (jr_size == 0) {
//				continue;
//			}
//
//			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
//
//			gettimeofday(&join_start, NULL);
//			HashJoinWrapper(outer_dev, inner_dev,
//								outer_cols_, inner_cols_,
//								end_dev, end_size_,
//								post_dev, post_size_,
//								outer_hash_dev, inner_hash_dev,
//								base_outer_idx, base_inner_idx,
//								index_count, outer_part_size,
//								jresult_dev);
//			gettimeofday(&join_end, NULL);
//			join_time.push_back(timeDiff(join_start, join_end));
//
//			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));
//
//			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));
//			checkCudaErrors(cudaFree(jresult_dev));
//			result_size_ += jr_size;
//			jr_size = 0;
//#else
//			RESULT *tmp_bound, *out_bound;
//			ulong out_size;
//			ulong *exp_psum;
//
//			gettimeofday(&rebalance_start, NULL);
//#ifdef DECOMPOSED1_
//			HRebalance2(index_count, in_bound, inner_hash_dev, &tmp_bound, outer_part_size + 1, &out_size);
//#else
//			HRebalance(index_count, in_bound, inner_hash_dev, &tmp_bound, outer_part_size, &out_size);
//#endif
//			gettimeofday(&rebalance_end, NULL);
//			rebalance_cost.push_back(timeDiff(rebalance_start, rebalance_end));
//
//			if (out_size == 0) {
//				continue;
//			}
//
//			checkCudaErrors(cudaMalloc(&exp_psum, (out_size + 1) * sizeof(ulong)));
//			checkCudaErrors(cudaMalloc(&out_bound, out_size * sizeof(RESULT)));
//
//			gettimeofday(&join_start, NULL);
//
//#ifndef SHARED_
//			ExpressionFilterWrapper2(outer_dev, inner_dev,
//										tmp_bound, out_bound,
//										exp_psum, out_size,
//										outer_cols_, inner_cols_,
//										end_dev, end_size_,
//										post_dev, post_size_,
//										where_dev, where_size_,
//										base_outer_idx, base_inner_idx);
//#else
//			ExpressionFilterWrapper3(outer_dev, inner_dev,
//										tmp_bound, out_bound,
//										exp_psum, out_size,
//										outer_cols_, inner_cols_,
//										end_dev, end_size_,
//										post_dev, post_size_,
//										where_dev, where_size_,
//										base_outer_idx, base_inner_idx);
//#endif
//			gettimeofday(&join_end, NULL);
//			join_time.push_back(timeDiff(join_start, join_end));
//
//			gettimeofday(&prefix_start, NULL);
//			ExclusiveScanWrapper(exp_psum, out_size + 1, &jr_size);
//			gettimeofday(&prefix_end, NULL);
//
//			prefix_sum.push_back(timeDiff(prefix_start, prefix_end));
//
//			checkCudaErrors(cudaFree(tmp_bound));
//
//			if (jr_size == 0) {
//				printf("EMPTY RESULT******************************************\n");
//				checkCudaErrors(cudaFree(exp_psum));
//				checkCudaErrors(cudaFree(out_bound));
//				continue;
//			}
//
//			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
//
//			gettimeofday(&remove_start, NULL);
//			RemoveEmptyResultWrapper2(jresult_dev, out_bound, exp_psum, out_size);
//			gettimeofday(&remove_end, NULL);
//			remove_empty.push_back(timeDiff(remove_start, remove_end));
//			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));
//
//			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));
//
//			checkCudaErrors(cudaFree(exp_psum));
//			checkCudaErrors(cudaFree(out_bound));
//			checkCudaErrors(cudaFree(jresult_dev));
//			result_size_ += jr_size;
//			jr_size = 0;
//#endif
//		}
//	}
//
//#else
//	partition_size = getPartitionSize();
//	checkCudaErrors(cudaMalloc(&index_count, sizeof(ulong) * (partition_size + 1)));
//
//	ResBound *in_bound;
//
//	checkCudaErrors(cudaMalloc(&in_bound, sizeof(ResBound) * partition_size));
//
//	printf("Start Joining\n");
//
//	outer_chunk.column_num = outer_table_.column_num;
//	outer_chunk.schema = outer_table_.schema;
//	inner_chunk.column_num = inner_table_.column_num;
//	inner_chunk.schema = inner_table_.schema;
//	int64_t *outer_key = NULL;
//	for (int outer_idx = 0; outer_idx < outer_table_.block_num; outer_idx++) {
//		/*** Hash the outer partition ***/
//		outer_chunk.block_list = outer_table_.block_list + outer_idx;
//
//		for (int inner_idx = 0; inner_idx < inner_table_.block_num; inner_idx++) {
//
//			inner_chunk.block_list = inner_table_.block_list + inner_idx;
//
//			gettimeofday(&index_count_start, NULL);
//			//IndexCountLegacyWrapper(outer_key, outer_chunk.block_list->rows, inner_hash_dev, index_count, in_bound);
//			gettimeofday(&index_count_end, NULL);
//
//			index_hcount.push_back(timeDiff(index_count_start, index_count_end));
//
//#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
//			gettimeofday(&prefix_start, NULL);
//			ExclusiveScanWrapper(index_count, partition_size + 1, &jr_size);
//			gettimeofday(&prefix_end, NULL);
//
//			prefix_sum.push_back(timeDiff(prefix_start, prefix_end));
//
//			if (jr_size < 0) {
//				printf("Scanning failed\n");
//				return false;
//			}
//
//			if (jr_size == 0) {
//				continue;
//			}
//
//			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
//
//			gettimeofday(&join_start, NULL);
//			HashJoinLegacyWrapper(outer_dev, inner_dev,
//									outer_cols_, inner_cols_,
//									outer_part_size, outer_key,
//									end_dev, end_size_,
//									post_dev, post_size_, inner_hash_dev,
//									base_outer_idx, base_inner_idx,
//									index_count, in_bound,
//									jresult_dev);
//			gettimeofday(&join_end, NULL);
//			join_time.push_back(timeDiff(join_start, join_end));
//#else
//			RESULT *tmp_bound, *out_bound;
//			ulong out_size;
//			ulong *exp_psum;
//
//			gettimeofday(&rebalance_start, NULL);
//#ifdef DECOMPOSED1_
//			//HRebalance(index_count, in_bound, inner_hash_dev, &tmp_bound, outer_part_size, &out_size);
//#else
//			HRebalance2(index_count, in_bound, inner_hash_dev, &tmp_bound, outer_part_size + 1, &out_size);
//#endif
//			gettimeofday(&rebalance_end, NULL);
//			rebalance_cost.push_back(timeDiff(rebalance_start, rebalance_end));
//
//			if (out_size == 0) {
//				continue;
//			}
//
//			printf("out_size = %lu\n", out_size);
//			checkCudaErrors(cudaMalloc(&exp_psum, (out_size + 1) * sizeof(ulong)));
//			checkCudaErrors(cudaMalloc(&out_bound, out_size * sizeof(RESULT)));
//
//			gettimeofday(&join_start, NULL);
////			ExpressionFilterWrapper2(outer_dev, inner_dev,
////										tmp_bound, out_bound,
////										exp_psum, out_size,
////										outer_cols_, inner_cols_,
////										end_dev, end_size_,
////										post_dev, post_size_,
////										where_dev, where_size_,
////										base_outer_idx, base_inner_idx);
//			gettimeofday(&join_end, NULL);
//			join_time.push_back(timeDiff(join_start, join_end));
//
//			gettimeofday(&prefix_start, NULL);
//			ExclusiveScanWrapper(exp_psum, out_size + 1, &jr_size);
//			gettimeofday(&prefix_end, NULL);
//
//			prefix_sum.push_back(timeDiff(prefix_start, prefix_end));
//
//			checkCudaErrors(cudaFree(tmp_bound));
//
//			if (jr_size == 0) {
//				checkCudaErrors(cudaFree(exp_psum));
//				checkCudaErrors(cudaFree(out_bound));
//				continue;
//			}
//
//			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
//
//			gettimeofday(&remove_start, NULL);
//			RemoveEmptyResultWrapper2(jresult_dev, out_bound, exp_psum, out_size);
//			gettimeofday(&remove_end, NULL);
//			remove_empty.push_back(timeDiff(remove_start, remove_end));
//#endif
//
//			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));
//
//			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT), cudaMemcpyDeviceToHost));
//#ifdef DECOMPOSED1_
//			checkCudaErrors(cudaFree(exp_psum));
//			checkCudaErrors(cudaFree(out_bound));
//#endif
//			checkCudaErrors(cudaFree(jresult_dev));
//			result_size_ += jr_size;
//			jr_size = 0;
//		}
//	}
//
//#endif
//
//	gettimeofday(&end_all, NULL);
//	unsigned long inner_pack_final, inner_hash_final;
//	unsigned long outer_pack_final, outer_hash_final;
//	unsigned long index_count_final, prefix_sum_final, join_final;
//	unsigned long rebalance_final, remove_empty_total;
//
//	inner_pack_final = 0;
//	for (int i = 0; i < inner_pack.size(); i++) {
//		printf("InnerPack time at %d is %lu\n", i, inner_pack[i]);
//		inner_pack_final += inner_pack[i];
//	}
//
//
//	inner_hash_final = 0;
//	for (int i = 0; i < inner_hasher.size(); i++) {
//		inner_hash_final += inner_hasher[i];
//	}
//
//	outer_pack_final = 0;
//	for (int i = 0; i < outer_pack.size(); i++) {
//		outer_pack_final += outer_pack[i];
//	}
//
//	outer_hash_final = 0;
//	for (int i = 0; i < outer_hasher.size(); i++) {
//		outer_hash_final += outer_hasher[i];
//	}
//
//	index_count_final = 0;
//	for (int i = 0; i < index_hcount.size(); i++) {
//		//printf("index count time = %lu\n", index_hcount[i]);
//		index_count_final += index_hcount[i];
//	}
//
//	prefix_sum_final = 0;
//	for (int i = 0; i < prefix_sum.size(); i++) {
//		//printf("Prefix sum time = %lu\n", prefix_sum[i]);
//		prefix_sum_final += prefix_sum[i];
//	}
//
//	rebalance_final = 0;
//	for (int i = 0; i < rebalance_cost.size(); i++) {
//		rebalance_final += rebalance_cost[i];
//	}
//
//	remove_empty_total = 0;
//	for (int i = 0; i < remove_empty.size(); i++) {
//		remove_empty_total += remove_empty[i];
//	}
//
//	join_final = 0;
//	for (int i = 0; i < join_time.size(); i++) {
//		join_final += join_time[i];
//	}
//
//	ulong inner_hash_total, outer_hash_total, join_total, data_copy, total;
//
//	inner_hash_total = inner_pack_final + inner_hash_final;
//	outer_hash_total = outer_pack_final + outer_hash_final;
//	join_total = index_count_final + prefix_sum_final + join_final + remove_empty_total;
//	total = timeDiff(start_all, end_all);
//	data_copy = total - inner_hash_total - outer_hash_total - join_total;
//
//	printf("\n*** Execution time *****************************\n"
//			"Inner Pack: %lu\n"
//			"Inner Hash: %lu\n\n"
//
//			"Outer Pack: %lu\n"
//			"Outer Hash: %lu\n\n"
//
//			"index Count: %lu\n"
//			"prefix_sum: %lu\n"
//			"Join: %lu\n"
//			"*************************************************\n"
//			"Inner hash Total: %lu\n"
//			"Outer hash Total: %lu\n"
//#ifdef DECOMPOSED1_
//			"Rebalance total: %lu\n"
//#endif
//			"Exp evaluation: %lu\n"
//			"Remove empty total: %lu\n"
//			"Data copy: %lu\n"
//			"Total time including data copy: %lu\n"
//			"Total time excluding data copy: %lu\n"
//			"Total time: %lu\n", inner_pack_final, inner_hash_final,
//								outer_pack_final, outer_hash_final,
//								index_count_final, prefix_sum_final, join_final,
//								inner_hash_total, outer_hash_total,
//#ifdef DECOMPOSED1_
//								rebalance_final,
//#endif
//								join_total, remove_empty_total, data_copy, total - inner_hash_total, total - inner_hash_total - data_copy, total);
//
//
//	checkCudaErrors(cudaFree(outer_dev));
//	checkCudaErrors(cudaFree(inner_dev));
//
//	if (initial_size_ > 0)
//		checkCudaErrors(cudaFree(initial_dev));
//
//	if (end_size_ > 0)
//		checkCudaErrors(cudaFree(end_dev));
//
//	if (post_size_ > 0)
//		checkCudaErrors(cudaFree(post_dev));
//
//	if (where_size_ > 0)
//		checkCudaErrors(cudaFree(where_dev));
//
//	checkCudaErrors(cudaFree(search_exp_dev));
//	checkCudaErrors(cudaFree(search_exp_size));
//	checkCudaErrors(cudaFree(indices_dev));
//	checkCudaErrors(cudaFree(index_count));
//
//#ifdef METHOD_1_
//	checkCudaErrors(cudaFree(outer_key));
//	checkCudaErrors(cudaFree(inner_key));
//	checkCudaErrors(cudaFree(outer_hash_dev.bucket_location));
//	checkCudaErrors(cudaFree(outer_hash_dev.hashed_idx));
//	checkCudaErrors(cudaFree(outer_hash_dev.hashed_key));
//	checkCudaErrors(cudaFree(inner_hash_dev.bucket_location));
//	checkCudaErrors(cudaFree(inner_hash_dev.hashed_idx));
//	checkCudaErrors(cudaFree(inner_hash_dev.hashed_key));
//
//#ifdef DECOMPOSED1_
//	checkCudaErrors(cudaFree(in_bound));
//#endif
//
//	int max = 0, maxId = 0, blockMax;
//	for (int i = 0; i < part_num_inner; i++) {
//		for (int k = 0; k < maxNumberOfBuckets_; k++) {
//			if (inner_hash_host[i].bucket_location[k + 1] - inner_hash_host[i].bucket_location[k] > max) {
//				max = inner_hash_host[i].bucket_location[k + 1] - inner_hash_host[i].bucket_location[k];
//				maxId = k;
//				blockMax = i;
//			}
//		}
//		free(inner_hash_host[i].bucket_location);
//		free(inner_hash_host[i].hashed_idx);
//		free(inner_hash_host[i].hashed_key);
//	}
//
//	printf("Max bucket of Outer at index %d and block %d is %d\n", maxId, blockMax, max);
//
//	free(inner_hash_host);
//	free(inner_hashed);
//#else
//	checkCudaErrors(cudaFree(outer_key));
//	checkCudaErrors(cudaFree(inner_key));
//	checkCudaErrors(cudaFree(inner_hash_dev.bucket_location));
//	checkCudaErrors(cudaFree(inner_hash_dev.hashed_idx));
//	checkCudaErrors(cudaFree(inner_hash_dev.hashed_key));
//
//
//	for (int i = 0; i < part_num_inner; i++) {
//		free(inner_hash_host[i].bucket_location);
//		free(inner_hash_host[i].hashed_idx);
//		free(inner_hash_host[i].hashed_key);
//	}
//	free(inner_hash_host);
//	free(inner_hashed);
//
//#endif
//
//	checkCudaErrors(cudaProfilerStop());
//
//	return true;
}



__global__ void PackKey(GTable index_table, int *indices, int index_num, uint64_t *key, int key_size)
{
	int64_t *table = index_table.block_list->gdata;
	GColumnInfo *schema = index_table.schema;
	int cols = index_table.column_num;
	int rows = index_table.block_list->rows;

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < rows; i += stride) {
		KeyGenerate(table + i * cols, schema, indices, index_num, key + i * key_size);
	}
}

void PackKeyWrapper(GTable table, int *indices, int index_num, uint64_t *key, int key_size)
{
	int rows = table.block_list->rows;
	int grid_x, block_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemset(key, 0, sizeof(uint64_t) * rows * key_size));
	checkCudaErrors(cudaDeviceSynchronize());

	PackKey<<<grid_size, block_size>>>(table, indices, index_num, key, key_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void PackKeyAsyncWrapper(GTable table, int *indices, int index_num, uint64_t *key, int key_size, cudaStream_t stream)
{
	int rows = table.block_list->rows;
	int grid_x, block_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemsetAsync(key, 0, sizeof(uint64_t) * rows * key_size, stream));

	PackKey<<<grid_size, block_size, 0, stream>>>(table, indices, index_num, key, key_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void EvalSearchKey(GTable input, GTable output, GTreeNode *search_exp, int *search_size, int search_num
#if defined(FUNC_CALL_) && defined(POST_EXP_)
								,GNValue *stack
#elif defined(POST_EXP_)
								,int64_t *val_stack,
								ValueType *type_stack
#endif
								)
{
	int input_cols = input.column_num;
	int input_rows = input.block_list->rows, output_rows = output.block_list->rows;
	int64_t *input_table = input.block_list->gdata, *output_table = output.block_list->gdata;
	GColumnInfo *input_schema = input.schema, *output_schema = output.schema;

	int index = threadIdx.x + blockIdx.x *blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int search_ptr = 0;
	GTree pred;
	GNValue tmp;

	for (int i = index; i < input_rows; i += stride) {
		search_ptr = 0;
		for (int j = 0; j < search_num; search_ptr += search_size[j], j++) {
			pred.exp = search_exp + search_ptr;
			pred.size = search_size[j];
#ifdef POST_EXP_
#ifdef FUNC_CALL_
			tmp = EvaluateItrFunc(pred, input_table + i * input_cols, NULL, input_schema, NULL, stack + index, stride);
#else
			tmp = EvaluateItrNonFunc(pred, input_table + i * input_cols, NULL, input_schema, NULL, val_stack + index, type_stack + index, stride);
#endif
#else
#ifdef FUNC_CALL_
			tmp = EvaluateRecvFunc(pred, 1, input_table + i * cols, NULL, schema, NULL);
#else
			tmp = EvaluateRecvNonFunc(pred, 1, input_table + i * cols, NULL, schema, NULL);
#endif
#endif
			output_table[i * output_rows + j] = tmp.getValue();
			output_schema[j].data_type = tmp.getValueType();
		}
	}
}

__global__ void PackSearchKey(GTable input, uint64_t *key, int key_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int size = input.block_list->rows;

	for (int i = index; i < size; i += stride)
		KeyGenerate(input.block_list->gdata, input.schema, NULL, input.column_num, key + i * key_size);
}

void PackSearchKeyWrapper(GTable table, uint64_t *key, GTreeNode *search_exp, int *search_size, int search_num, int key_size)
{
	int rows = table.block_list->rows;
	int block_x, grid_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif
	GTable out_table;
	GBlock new_block;

	checkCudaErrors(cudaMalloc(&(new_block.gdata), sizeof(int64_t) * rows * search_num));
	new_block.rows = 0;
	new_block.columns = 0;
	new_block.block_size = sizeof(int64_t) * rows * search_num;
	out_table.block_list = &new_block;
	checkCudaErrors(cudaMalloc(&(out_table.schema), sizeof(GColumnInfo) * search_num));
	out_table.column_num = search_num;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemset(key, 0, sizeof(uint64_t) * rows * key_size));
	checkCudaErrors(cudaDeviceSynchronize());

	EvalSearchKey<<<grid_size, block_size>>>(table, out_table, search_exp, search_size, search_num
#if defined(FUNC_CALL_) && defined(POST_EXP_)
											,stack
#elif defined(POST_EXP_)
											,val_stack,
											type_stack
#endif
											);

	PackSearchKey<<<grid_size, block_size>>>(out_table, key, key_size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

	checkCudaErrors(cudaFree(out_table.block_list->gdata));
	checkCudaErrors(cudaFree(out_table.schema));
}

void PackSearchKeyAsyncWrapper(GTable table, uint64_t *key, GTreeNode *search_exp, int *search_size, int search_num, int key_size, cudaStream_t stream)
{
	int rows = table.block_list->rows;
	int block_x, grid_x;

	block_x = (rows < BLOCK_SIZE_X) ? rows : BLOCK_SIZE_X;
	grid_x = (rows - 1)/block_x + 1;

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif defined(POST_EXP_)
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif
	GTable out_table;
	GBlock new_block;

	checkCudaErrors(cudaMalloc(&(new_block.gdata), sizeof(int64_t) * rows * search_num));
	new_block.rows = 0;
	new_block.columns = 0;
	new_block.block_size = sizeof(int64_t) * rows * search_num;
	out_table.block_list = &new_block;
	checkCudaErrors(cudaMalloc(&(out_table.schema), sizeof(GColumnInfo) * search_num));
	out_table.column_num = search_num;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	checkCudaErrors(cudaMemsetAsync(key, 0, sizeof(uint64_t) * rows * key_size, stream));

	EvalSearchKey<<<grid_size, block_size, 0, stream>>>(table, out_table, search_exp, search_size, search_num
#if defined(FUNC_CALL_) && defined(POST_EXP_)
														,stack
#elif defined(POST_EXP_)
														,val_stack,
														type_stack
#endif
														);
	PackSearchKey<<<grid_size, block_size, 0, stream>>>(out_table, key, key_size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if defined(FUNC_CALL_) && defined(POST_EXP_)
	checkCudaErrors(cudaFree(stack));
#elif defined(POST_EXP_)
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

	checkCudaErrors(cudaFree(out_table.block_list->gdata));
	checkCudaErrors(cudaFree(out_table.schema));

}
}
