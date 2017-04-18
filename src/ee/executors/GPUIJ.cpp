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
#include "GPUIJ.h"
#include "scan_common.h"
#include "index_join_gpu.h"
#include "gcommon/gpu_common.h"

namespace voltdb {

GPUIJ::GPUIJ()
{
		outer_table_ = inner_table_ = NULL;
		outer_size_ = inner_size_ = 0;
		outer_rows_ = inner_rows_ = 0;
		outer_cols_ = inner_cols_ = 0;
		join_result_ = NULL;
		result_size_ = 0;
		end_size_ = 0;
		post_size_ = 0;
		initial_size_ = 0;
		skipNull_size_ = 0;
		prejoin_size_ = 0;
		where_size_ = 0;
		indices_size_ = 0;
		search_exp_size_ = NULL;
		search_exp_num_ = 0;
		indices_ = NULL;

		search_exp_ = NULL;
		end_expression_ = NULL;
		post_expression_ = NULL;
		initial_expression_ = NULL;
		skipNullExpr_ = NULL;
		prejoin_expression_ = NULL;
		where_expression_ = NULL;
}

GPUIJ::GPUIJ(GNValue *outer_table,
				GNValue *inner_table,
				int outer_rows,
				int outer_cols,
				int inner_rows,
				int inner_cols,
				std::vector<TreeExpression> search_exp,
				std::vector<int> indices,
				TreeExpression end_expression,
				TreeExpression post_expression,
				TreeExpression initial_expression,
				TreeExpression skipNullExpr,
				TreeExpression prejoin_expression,
				TreeExpression where_expression,
				IndexLookupType lookup_type)
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
	end_size_ = end_expression.getSize();
	post_size_ = post_expression.getSize();
	initial_size_ = initial_expression.getSize();
	skipNull_size_ = skipNullExpr.getSize();
	prejoin_size_ = prejoin_expression.getSize();
	where_size_ = where_expression.getSize();
	search_exp_num_ = search_exp.size();
	indices_size_ = indices.size();
	lookup_type_ = lookup_type;


	bool ret = true;
	int tmp_size = 0;

	search_exp_size_ = (int *)malloc(sizeof(int) * search_exp_num_);
	assert(search_exp_size_ != NULL);
	for (int i = 0; i < search_exp_num_; i++) {
		search_exp_size_[i] = search_exp[i].getSize();
		tmp_size += search_exp_size_[i];
	}

	search_exp_ = (GTreeNode *)malloc(sizeof(GTreeNode) * tmp_size);
	assert(search_exp_ != NULL);
	GTreeNode *exp_ptr = search_exp_;
	for (int i = 0; i < search_exp_num_; i++) {
		getTreeNodes2(exp_ptr, search_exp[i]);
		exp_ptr += search_exp_size_[i];
	}

	indices_ = (int *)malloc(sizeof(int) * indices_size_);
	assert(indices_ != NULL);
	for (int i = 0; i < indices_size_; i++) {
		indices_[i] = indices[i];
	}

	/**** Expression data ****/

	ret = getTreeNodes(&end_expression_, end_expression);
	assert(ret == true);

	ret = getTreeNodes(&post_expression_, post_expression);
	assert(ret == true);

	ret = getTreeNodes(&initial_expression_, initial_expression);
	assert(ret == true);

	ret = getTreeNodes(&skipNullExpr_, skipNullExpr);
	assert(ret == true);

	ret = getTreeNodes(&prejoin_expression_, prejoin_expression);
	assert(ret == true);

	ret = getTreeNodes(&where_expression_, where_expression);
	assert(ret == true);
}

GPUIJ::~GPUIJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(search_exp_);
	freeArrays<int>(search_exp_size_);
	freeArrays<int>(indices_);
	freeArrays<GTreeNode>(end_expression_);
	freeArrays<GTreeNode>(post_expression_);
	freeArrays<GTreeNode>(initial_expression_);
	freeArrays<GTreeNode>(skipNullExpr_);
	freeArrays<GTreeNode>(where_expression_);
}

bool GPUIJ::join(){
	int loop_count = 0, loop_count2 = 0;
	cudaError_t res;

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	//ulong jr_size = 0, jr_size2 = 0;
	ulong jr_size, jr_size2;
	int *indices_dev, *search_exp_size;
	GTreeNode *initial_dev, *skipNull_dev, *prejoin_dev, *where_dev, *end_dev, *post_dev, *search_exp_dev;

#ifdef ASYNC_
	GNValue *outer_dev[2], *inner_dev[2];
	RESULT *jresult_dev[2], *write_dev[2];
	jresult_dev[0] = jresult_dev[1] = NULL;
	write_dev[0] = write_dev[1] = NULL;
	ulong *index_psum[2], *exp_psum[2];
	index_psum[0] = index_psum[1] = NULL;
	exp_psum[0] = exp_psum[1] = NULL;
	ResBound *res_bound[2];
	bool *prejoin_res_dev[2];
#else
	GNValue *outer_dev, *inner_dev;
	RESULT *jresult_dev, *write_dev;
	jresult_dev = write_dev = NULL;
	ulong *index_psum, *exp_psum;
	ResBound *res_bound;
	bool *prejoin_res_dev;
#endif

	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;
	std::vector<unsigned long> allocation, prejoin, index, expression, ipsum, epsum, wtime, joins_only, rebalance;

	part_size = getPartitionSize();
	printf("Part size = %u\n", part_size);
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

	/******** Allocate GPU buffer for table data and counting data *****/

#ifdef ASYNC_
	checkCudaErrors(cudaMalloc(&prejoin_res_dev[0], part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&prejoin_res_dev[1], part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&outer_dev[0], part_size * outer_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&outer_dev[1], part_size * outer_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&inner_dev[0], part_size * inner_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&inner_dev[1], part_size * inner_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&index_psum[0], gpu_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&index_psum[1], gpu_size * sizeof(ulong)));

#ifndef DECOMPOSED1_
	checkCudaErrors(cudaMalloc(&exp_psum[0], gpu_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&exp_psum[1], gpu_size * sizeof(ulong)));
#endif
	checkCudaErrors(cudaMalloc(&res_bound[0], gpu_size * sizeof(ResBound)));
	checkCudaErrors(cudaMalloc(&res_bound[1], gpu_size * sizeof(ResBound)));

#else
	//checkCudaErrors(cudaMalloc(&outer_dev, part_size * outer_cols_ * sizeof(GNValue)));
	//checkCudaErrors(cudaMalloc(&inner_dev, part_size * inner_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&outer_dev, sizeof(GNValue) * outer_cols_ * outer_rows_));
	checkCudaErrors(cudaMalloc(&inner_dev, sizeof(GNValue) * inner_cols_ * inner_rows_));
	checkCudaErrors(cudaMalloc(&prejoin_res_dev, part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&index_psum, gpu_size * sizeof(ulong)));

#ifndef DECOMPOSED1_
	checkCudaErrors(cudaMalloc(&exp_psum, gpu_size * sizeof(ulong)));
#endif
	checkCudaErrors(cudaMalloc(&res_bound, gpu_size * sizeof(ResBound)));
#endif




	/******* Allocate GPU buffer for join condition *********/

	if (prejoin_size_ > 0) {
		checkCudaErrors(cudaMalloc(&prejoin_dev, prejoin_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(prejoin_dev, prejoin_expression_, prejoin_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (initial_size_ > 0) {
		checkCudaErrors(cudaMalloc(&initial_dev, initial_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(initial_dev, initial_expression_, initial_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (skipNull_size_ > 0) {
		checkCudaErrors(cudaMalloc(&skipNull_dev, skipNull_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(skipNull_dev, skipNullExpr_, skipNull_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (end_size_ > 0) {
		checkCudaErrors(cudaMalloc(&end_dev, end_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(end_dev, end_expression_, end_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (post_size_ > 0) {
		checkCudaErrors(cudaMalloc(&post_dev, post_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(post_dev, post_expression_, post_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	if (where_size_ > 0) {
		checkCudaErrors(cudaMalloc(&where_dev, where_size_ * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(where_dev, where_expression_, where_size_ * sizeof(GTreeNode), cudaMemcpyHostToDevice));
	}

	/******* Allocate GPU buffer for search keys and index keys *****/
	int tmp_size = 0;
	for (int i = 0; i < search_exp_num_; i++) {
		tmp_size += search_exp_size_[i];
	}

	checkCudaErrors(cudaMalloc(&search_exp_dev, sizeof(GTreeNode) * tmp_size));
	checkCudaErrors(cudaMalloc(&search_exp_size, sizeof(int) * search_exp_num_));

	checkCudaErrors(cudaMemcpy(search_exp_dev, search_exp_, sizeof(GTreeNode) * tmp_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(search_exp_size, search_exp_size_, sizeof(int) * search_exp_num_, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&indices_dev, sizeof(int) * indices_size_));
	checkCudaErrors(cudaMemcpy(indices_dev, indices_, sizeof(int) * indices_size_, cudaMemcpyHostToDevice));

	printf("Block_x = %d, block_y = %d, grid_x = %d, grid_y = %d\n", block_x, block_y, grid_x, grid_y);

	int stream_idx = 0;
	struct timeval pre_start, pre_end, istart, iend, pistart, piend, estart, eend, pestart, peend, wstart, wend, end_join, balance_start, balance_end;

	int init_outer_size = (part_size < outer_size_) ? part_size : outer_size_;
	int init_inner_size = (part_size < inner_size_) ? part_size : inner_size_;

#ifdef ASYNC_
	cudaStream_t stream[2];

	checkCudaErrors(cudaStreamCreate(&stream[0]));
	checkCudaErrors(cudaStreamCreate(&stream[1]));
#endif

	uint outer_part_size = 0, inner_part_size = 0;
	uint outer_idx, inner_idx;
	uint outer_dev_idx, inner_dev_idx;
	int old_result_size = 0;
	RESULT *old_result = NULL;
	GNValue *old_outer = NULL, *old_inner = NULL;

#ifdef ASYNC_
	checkCudaErrors(cudaHostRegister(outer_table_, sizeof(GNValue) * outer_cols_ * outer_rows_, cudaHostRegisterDefault));
	checkCudaErrors(cudaHostRegister(inner_table_, sizeof(GNValue) * inner_cols_ * inner_rows_, cudaHostRegisterDefault));
#endif

	//checkCudaErrors(cudaMemcpy(outer_dev, outer_table_, sizeof(GNValue) * outer_cols_ * outer_rows_, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(inner_dev, inner_table_, sizeof(GNValue) * inner_cols_ * inner_rows_, cudaMemcpyHostToDevice));

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (outer_idx = 0, outer_dev_idx = 0, stream_idx = 0; outer_idx < outer_size_; outer_idx += part_size, stream_idx = 1 - stream_idx, outer_dev_idx = 1 - outer_dev_idx) {
		//Size of outer small table
		uint old_outer_size = outer_part_size;
		outer_part_size = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);
#ifdef ASYNC_
//		if (old_outer != NULL) {
//			checkCudaErrors(cudaHostUnregister(old_outer));
//			old_outer = NULL;
//		}
//		checkCudaErrors(cudaHostRegister(outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue), cudaHostRegisterDefault));
//		old_outer = outer_table_ + outer_idx * outer_cols_;
		checkCudaErrors(cudaMemcpyAsync(outer_dev[outer_dev_idx], outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice, stream[stream_idx]));
#else
		checkCudaErrors(cudaMemcpy(outer_dev, outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));
#endif

		/* Evaluate prejoin predicate */
		gettimeofday(&pre_start, NULL);
#ifdef ASYNC_
		PrejoinFilterAsyncWrapper(outer_dev[outer_dev_idx], outer_part_size, outer_cols_, prejoin_dev, prejoin_size_, prejoin_res_dev[outer_dev_idx], stream[stream_idx]);
#else
		//PrejoinFilterWrapper(outer_dev, outer_part_size, outer_cols_, prejoin_dev, prejoin_size_, prejoin_res_dev);
		PrejoinFilterWrapper(outer_dev + outer_idx * outer_cols_, outer_part_size, outer_cols_, prejoin_dev, prejoin_size_, prejoin_res_dev);
#endif
		gettimeofday(&pre_end, NULL);
		prejoin.push_back(timeDiff(pre_start, pre_end));

		joins_only.push_back(timeDiff(pre_start, pre_end));

		for (inner_idx = 0, inner_dev_idx = 0; inner_idx < inner_size_; inner_idx += part_size, inner_dev_idx = 1 - inner_dev_idx) {
			//Size of inner small table
			uint old_inner_size = inner_part_size;

			inner_part_size = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			gpu_size = outer_part_size + 1;
			/**** Copy IndexData to GPU memory ****/
#ifdef ASYNC_
//			if (old_inner != NULL) {
//				checkCudaErrors(cudaHostUnregister(old_inner));
//				old_inner = NULL;
//			}
//			checkCudaErrors(cudaHostRegister(inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue), cudaHostRegisterDefault));
//			old_inner = inner_table_ + inner_idx * inner_cols_;
			checkCudaErrors(cudaMemcpyAsync(inner_dev[inner_dev_idx], inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice, stream[stream_idx]));
#else
			checkCudaErrors(cudaMemcpy(inner_dev, inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));
#endif

			/* Binary search for index */
			gettimeofday(&istart, NULL);
#ifdef ASYNC_
			IndexFilterAsyncWrapper(outer_dev[outer_dev_idx], inner_dev[inner_dev_idx],
									index_psum[stream_idx], res_bound[stream_idx],
									outer_part_size, outer_cols_,
									inner_part_size, inner_cols_,
									search_exp_dev, search_exp_size, search_exp_num_,
									indices_dev, indices_size_,
									lookup_type_, prejoin_res_dev[outer_dev_idx], stream[stream_idx]);
#else
			//IndexFilterWrapper(outer_dev + outer_idx * outer_cols_, inner_dev + inner_idx * inner_cols_,
			IndexFilterWrapper(outer_dev, inner_dev,
								index_psum, res_bound,
								outer_part_size, outer_cols_,
								inner_part_size, inner_cols_,
								search_exp_dev, search_exp_size, search_exp_num_,
								indices_dev, indices_size_,
								lookup_type_, prejoin_res_dev);
#endif

			gettimeofday(&iend, NULL);
			index.push_back(timeDiff(istart, iend));

#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
			/* Prefix sum on the result */
			gettimeofday(&pistart, NULL);
#ifdef ASYNC_
			ExclusiveScanAsyncWrapper(index_psum[stream_idx], gpu_size, &jr_size, stream[stream_idx]);
#else
			ExclusiveScanWrapper(index_psum, gpu_size, &jr_size);
#endif
			gettimeofday(&piend, NULL);


			ipsum.push_back(timeDiff(pistart, piend));

			if (jr_size < 0) {
				printf("Scanning failed\n");
				return false;
			}

			if (jr_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back(timeDiff(istart, end_join));
				continue;
			}

#ifdef ASYNC_
			checkCudaErrors(cudaMalloc(&jresult_dev[stream_idx], jr_size * sizeof(RESULT)));
#else
			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));
#endif

			gettimeofday(&estart, NULL);
#ifdef ASYNC_
			ExpressionFilterAsyncWrapper(outer_dev[outer_dev_idx], inner_dev[inner_dev_idx],
											jresult_dev[stream_idx], index_psum[stream_idx], exp_psum[stream_idx],
											outer_part_size,
											outer_cols_, inner_cols_,
											jr_size,
											end_dev, end_size_,
											post_dev, post_size_,
											where_dev, where_size_,
											res_bound[stream_idx],
											outer_idx, inner_idx,
											prejoin_res_dev, stream[stream_idx]);
#else
			ExpressionFilterWrapper(outer_dev, inner_dev,
									jresult_dev, index_psum, exp_psum,
									outer_part_size,
									outer_cols_, inner_cols_,
									jr_size,
									end_dev, end_size_,
									post_dev, post_size_,
									where_dev, where_size_,
									res_bound,
									outer_idx, inner_idx,
									prejoin_res_dev);
#endif
			gettimeofday(&eend, NULL);


			gettimeofday(&pestart, NULL);
#ifdef ASYNC_
			ExclusiveScanAsyncWrapper(exp_psum[stream_idx], gpu_size, &jr_size2, stream[stream_idx]);
#else
			ExclusiveScanWrapper(exp_psum, gpu_size, &jr_size2);
#endif
			gettimeofday(&peend, NULL);

			expression.push_back(timeDiff(estart, eend));
			epsum.push_back(timeDiff(pestart, peend));

			if (jr_size2 == 0) {
				checkCudaErrors(cudaFree(jresult_dev));
				continue;
			}

#ifdef ASYNC_
			checkCudaErrors(cudaMalloc(&write_dev[stream_idx], jr_size2 * sizeof(RESULT)));
#else
			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));
#endif

			gettimeofday(&wstart, NULL);
#ifdef ASYNC_
			RemoveEmptyResultAsyncWrapper(write_dev[stream_idx], jresult_dev[stream_idx], index_psum[stream_idx], exp_psum[stream_idx], jr_size, stream[stream_idx]);
#else
			RemoveEmptyResultWrapper(write_dev, jresult_dev, index_psum, exp_psum, jr_size);
#endif
			gettimeofday(&wend, NULL);
#elif defined(DECOMPOSED1_)
			RESULT *tmp_result;
			ulong tmp_size = 0;

			gettimeofday(&balance_start, NULL);
#ifdef ASYNC_
			RebalanceAsync3(index_psum[stream_idx], res_bound[stream_idx], &tmp_result, gpu_size, &tmp_size, stream[stream_idx]);
#else
			Rebalance3(index_psum, res_bound, &tmp_result, gpu_size, &tmp_size);
#endif
			gettimeofday(&balance_end, NULL);

			rebalance.push_back(timeDiff(balance_start, balance_end));

			if (tmp_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back(timeDiff(istart, end_join));
				continue;
			}

#ifdef ASYNC_
			if (jresult_dev[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(jresult_dev[1 - stream_idx]));
				jresult_dev[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&jresult_dev[stream_idx], tmp_size * sizeof(RESULT)));

			if (exp_psum[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(exp_psum[1 - stream_idx]));
				exp_psum[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&exp_psum[stream_idx], (tmp_size + 1) * sizeof(ulong)));
#else
			checkCudaErrors(cudaMalloc(&jresult_dev, tmp_size * sizeof(RESULT)));
			checkCudaErrors(cudaMalloc(&exp_psum, (tmp_size + 1) * sizeof(ulong)));
#endif

			gettimeofday(&estart, NULL);
#ifdef ASYNC_
			ExpressionFilterAsyncWrapper2(outer_dev[outer_dev_idx], inner_dev[inner_dev_idx],
											tmp_result, jresult_dev[stream_idx],
											exp_psum[stream_idx], tmp_size,
											outer_cols_, inner_cols_,
											end_dev, end_size_,
											post_dev, post_size_,
											where_dev, where_size_,
											outer_idx, inner_idx, stream[stream_idx]);
#else
			//ExpressionFilterWrapper2(outer_dev + outer_idx * outer_cols_, inner_dev + inner_idx * inner_cols_,
			ExpressionFilterWrapper2(outer_dev, inner_dev,
										tmp_result, jresult_dev,
										exp_psum, tmp_size,
										outer_cols_, inner_cols_,
										end_dev, end_size_,
										post_dev, post_size_,
										where_dev, where_size_,
										outer_idx, inner_idx);
#endif
			gettimeofday(&eend, NULL);

			expression.push_back(timeDiff(estart, eend));

			gettimeofday(&pestart, NULL);
#ifdef ASYNC_
			ExclusiveScanAsyncWrapper(exp_psum[stream_idx], tmp_size + 1, &jr_size2, stream[stream_idx]);
#else
			ExclusiveScanWrapper(exp_psum, tmp_size + 1, &jr_size2);
#endif
			gettimeofday(&peend, NULL);

			epsum.push_back(timeDiff(pestart, peend));

			checkCudaErrors(cudaFree(tmp_result));

			if (jr_size2 == 0) {
				continue;
			}
#ifdef ASYNC_
			if (write_dev[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(write_dev[1 - stream_idx]));
				write_dev[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&write_dev[stream_idx], jr_size2 * sizeof(RESULT)));
#else
			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));
#endif

			gettimeofday(&wstart, NULL);
#ifdef ASYNC_
			RemoveEmptyResultAsyncWrapper2(write_dev[stream_idx], jresult_dev[stream_idx], exp_psum[stream_idx], tmp_size, stream[stream_idx]);
#else
			RemoveEmptyResultWrapper2(write_dev, jresult_dev, exp_psum, tmp_size);
#endif
			gettimeofday(&wend, NULL);
#elif defined(DECOMPOSED2_)
			RESULT *tmp_result;
			ulong tmp_size = 0;

			gettimeofday(&balance_start, NULL);
#ifdef ASYNC_
			RebalanceAsync2(index_psum[stream_idx], res_bound[stream_idx], &tmp_result, gpu_size, &tmp_size, stream[stream_idx]);
#else
			Rebalance2(index_psum, res_bound, &tmp_result, gpu_size, &tmp_size);
#endif
			gettimeofday(&balance_end, NULL);

			rebalance.push_back(timeDiff(balance_start, balance_end));

			if (tmp_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back(timeDiff(istart, end_join));
				continue;
			}

#ifdef ASYNC_
			if (jresult_dev[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(jresult_dev[1 - stream_idx]));
				jresult_dev[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&jresult_dev[stream_idx], tmp_size * sizeof(RESULT)));

			if (exp_psum[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(exp_psum[1 - stream_idx]));
				exp_psum[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&exp_psum[stream_idx], (tmp_size + 1) * sizeof(ulong)));
#else
			checkCudaErrors(cudaMalloc(&jresult_dev, tmp_size * sizeof(RESULT)));
			checkCudaErrors(cudaMalloc(&exp_psum, (tmp_size + 1) * sizeof(ulong)));
#endif


			gettimeofday(&estart, NULL);
#ifdef ASYNC_
			ExpressionFilterAsyncWrapper2(outer_dev[outer_dev_idx], inner_dev[inner_dev_idx],
											tmp_result, jresult_dev[stream_idx],
											exp_psum[stream_idx], tmp_size,
											outer_cols_, inner_cols_,
											end_dev, end_size_,
											post_dev, post_size_,
											where_dev, where_size_,
											outer_idx, inner_idx, stream[stream_idx]);

#else
			ExpressionFilterWrapper2(outer_dev, inner_dev,
										tmp_result, jresult_dev,
										exp_psum, tmp_size,
										outer_cols_, inner_cols_,
										end_dev, end_size_,
										post_dev, post_size_,
										where_dev, where_size_,
										outer_idx, inner_idx);
#endif
			gettimeofday(&eend, NULL);

			expression.push_back(timeDiff(estart, eend));

			gettimeofday(&pestart, NULL);
#ifdef ASYNC_
			ExclusiveScanAsyncWrapper(exp_psum[stream_idx], tmp_size + 1, &jr_size2, stream[stream_idx]);
#else
			ExclusiveScanWrapper(exp_psum, tmp_size + 1, &jr_size2);
#endif
			gettimeofday(&peend, NULL);

			epsum.push_back(timeDiff(pestart, peend));

			checkCudaErrors(cudaFree(tmp_result));

			if (jr_size2 == 0)
				continue;
#ifdef ASYNC_
			if (write_dev[1 - stream_idx] != NULL) {
				checkCudaErrors(cudaFree(write_dev[1 - stream_idx]));
				write_dev[1 - stream_idx] = NULL;
			}
			checkCudaErrors(cudaMalloc(&write_dev[stream_idx], jr_size2 * sizeof(RESULT)));
#else
			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));
#endif

			gettimeofday(&wstart, NULL);
#ifdef ASYNC_
			RemoveEmptyResultAsyncWrapper2(write_dev[stream_idx], jresult_dev[stream_idx], exp_psum[stream_idx], tmp_size, stream[stream_idx]);
#else
			RemoveEmptyResultWrapper2(write_dev, jresult_dev, exp_psum, tmp_size);
#endif
			gettimeofday(&wend, NULL);
#endif

			wtime.push_back(timeDiff(wstart, wend));

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size2) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);

			//checkCudaErrors(cudaHostRegister(join_result_ + result_size_, jr_size2 * sizeof(RESULT), cudaHostRegisterDefault));
#ifdef ASYNC_
			if (old_result != NULL) {
				checkCudaErrors(cudaHostUnregister(old_result));
				old_result = NULL;
			}
			checkCudaErrors(cudaHostRegister(join_result_ + result_size_, jr_size2 * sizeof(RESULT), cudaHostRegisterDefault));
			old_result = join_result_ + result_size_;
			checkCudaErrors(cudaMemcpyAsync(join_result_ + result_size_, write_dev[stream_idx], jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost, stream[stream_idx]));
#else
			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, write_dev, jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost));
#endif
			//checkCudaErrors(cudaHostUnregister(join_result_ + result_size_));

			result_size_ += jr_size2;
			jr_size = 0;
			jr_size2 = 0;


			joins_only.push_back(timeDiff(istart, end_join));

#ifndef ASYNC_
			checkCudaErrors(cudaFree(exp_psum));
			checkCudaErrors(cudaFree(jresult_dev));
			checkCudaErrors(cudaFree(write_dev));
#endif
		}
	}


	checkCudaErrors(cudaDeviceSynchronize());
	/******** Free GPU memory, unload module, end session **************/

#ifdef ASYNC_
	checkCudaErrors(cudaFree(outer_dev[0]));
	checkCudaErrors(cudaFree(outer_dev[1]));
	checkCudaErrors(cudaFree(inner_dev[0]));
	checkCudaErrors(cudaFree(inner_dev[1]));
	checkCudaErrors(cudaFree(index_psum[0]));
	checkCudaErrors(cudaFree(index_psum[1]));
	checkCudaErrors(cudaFree(res_bound[0]));
	checkCudaErrors(cudaFree(res_bound[1]));
	checkCudaErrors(cudaFree(prejoin_res_dev[0]));
	checkCudaErrors(cudaFree(prejoin_res_dev[1]));

#ifndef DECOMPOSED1_
	checkCudaErrors(cudaFree(exp_psum[0]));
	checkCudaErrors(cudaFree(exp_psum[1]));
#else
	checkCudaErrors(cudaFree(exp_psum[stream_idx]));
	checkCudaErrors(cudaFree(write_dev[stream_idx]));
	checkCudaErrors(cudaFree(jresult_dev[stream_idx]));

	checkCudaErrors(cudaStreamDestroy(stream[0]));
	checkCudaErrors(cudaStreamDestroy(stream[1]));

//	if (old_outer != NULL)
//		checkCudaErrors(cudaHostUnregister(old_outer));
//	if (old_inner != NULL)
//		checkCudaErrors(cudaHostUnregister(old_inner));
	if (old_result != NULL)
		checkCudaErrors(cudaHostUnregister(old_result));
	checkCudaErrors(cudaHostUnregister(outer_table_));
	checkCudaErrors(cudaHostUnregister(inner_table_));
#endif
#else
	checkCudaErrors(cudaFree(outer_dev));
	checkCudaErrors(cudaFree(inner_dev));
	checkCudaErrors(cudaFree(index_psum));
#ifndef DECOMPOSED1_
	checkCudaErrors(cudaFree(exp_psum));
#endif
	checkCudaErrors(cudaFree(res_bound));
	checkCudaErrors(cudaFree(prejoin_res_dev));
#endif

	checkCudaErrors(cudaFree(search_exp_dev));
	checkCudaErrors(cudaFree(search_exp_size));
	checkCudaErrors(cudaFree(indices_dev));
	if (initial_size_ > 0)
		checkCudaErrors(cudaFree(initial_dev));

	if (skipNull_size_ > 0)
		checkCudaErrors(cudaFree(skipNull_dev));

	if (prejoin_size_ > 0)
		checkCudaErrors(cudaFree(prejoin_dev));

	if (where_size_ > 0)
		checkCudaErrors(cudaFree(where_dev));

	if (end_size_ > 0)
		checkCudaErrors(cudaFree(end_dev));

	if (post_size_ > 0)
		checkCudaErrors(cudaFree(post_dev));

	gettimeofday(&all_end, NULL);

	unsigned long allocation_time = 0, prejoin_time = 0, index_time = 0, expression_time = 0, ipsum_time = 0, epsum_time = 0, wtime_time = 0, joins_only_time = 0, all_time = 0;
	for (int i = 0; i < prejoin.size(); i++) {
		prejoin_time += prejoin[i];
	}

	for (int i = 0; i < index.size(); i++) {
		index_time += index[i];
	}

	for (int i = 0; i < expression.size(); i++) {
		expression_time += expression[i];
	}

	for (int i = 0; i < ipsum.size(); i++) {
		ipsum_time += ipsum[i];
	}

	for (int i = 0; i < epsum.size(); i++) {
		epsum_time += epsum[i];
	}

	for (int i = 0; i < wtime.size(); i++) {
		wtime_time += wtime[i];
	}

#if (defined(DECOMPOSED1_) || defined(DECOMPOSED2_))
	unsigned long rebalance_cost = 0;
	for (int i = 0; i < rebalance.size(); i++) {
		rebalance_cost += rebalance[i];
	}
#endif


	for (int i = 0; i < joins_only.size(); i++) {
		joins_only_time += joins_only[i];
	}

	all_time = (all_end.tv_sec - all_start.tv_sec) * 1000000 + (all_end.tv_usec - all_start.tv_usec);

	allocation_time = all_time - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"Prejoin filter Time: %lu\n"
			"Index Search Time: %lu\n"
#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
			"Index Prefix Sum Time: %lu\n"
#else
			"Rebalance Cost: %lu\n"
#endif
			"Expression filter Time: %lu\n"
			"Expression Prefix Sum Time: %lu\n"
			"Write back time Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, prejoin_time, index_time,
#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
			ipsum_time,
#else
			rebalance_cost,
#endif
			expression_time, epsum_time, wtime_time, joins_only_time, all_time);

	//exit(0);
	return true;
}

void GPUIJ::getResult(RESULT *output) const
{
	memcpy(output, join_result_, sizeof(RESULT) * result_size_);
}

int GPUIJ::getResultSize() const
{
	return result_size_;
}

uint GPUIJ::getPartitionSize() const
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


uint GPUIJ::divUtility(uint dividend, uint divisor) const
{
	return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

bool GPUIJ::getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression)
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

bool GPUIJ::getTreeNodes2(GTreeNode *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() >= 1)
		tree_expression.getNodesArray(expression);

	return true;
}

template <typename T> void GPUIJ::freeArrays(T *expression)
{
	if (expression != NULL) {
		free(expression);
	}
}


void GPUIJ::debug(void)
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

	std::cout << "Size of end_expression = " << end_size_ << std::endl;
	if (end_size_ != 0) {
		std::cout << "Content of end_expression" << std::endl;
		debugGTrees(end_expression_, end_size_);
	} else
		std::cout << "Empty end expression" << std::endl;

	std::cout << "Size of post_expression = " << post_size_ << std::endl;
	if (post_size_ != 0) {
		std::cout << "Content of post_expression" << std::endl;
		debugGTrees(post_expression_, post_size_);
	} else
		std::cout << "Empty post expression" << std::endl;

	std::cout << "Size of initial_expression = " << initial_size_ << std::endl;
	if (initial_size_ != 0) {
		std::cout << "Content of initial_expression" << std::endl;
		debugGTrees(initial_expression_, initial_size_);
	} else
		std::cout << "Empty initial expression" << std::endl;

	std::cout << "Size of skip null expression = " << skipNull_size_ << std::endl;
	if (skipNull_size_ != 0) {
		std::cout << "Content of skip null_expression" << std::endl;
		debugGTrees(skipNullExpr_, skipNull_size_);
	} else
		std::cout << "Empty skip null expression" << std::endl;

	std::cout << "Size of prejoin_expression = " << prejoin_size_ << std::endl;
	if (prejoin_size_ != 0) {
		std::cout << "Content of prejoin_expression" << std::endl;
		debugGTrees(prejoin_expression_, prejoin_size_);
	} else
		std::cout << "Empty prejoin expression " << std::endl;

	std::cout << "Size of where expression = " << where_size_ << std::endl;
	if (where_size_ != 0) {
		std::cout << "Content of where_expression" << std::endl;
		debugGTrees(where_expression_, where_size_);
	} else
		std::cout << "Empty where expression" << std::endl;

	std::cout << "Size of search_exp_ array = " << search_exp_num_ << std::endl;
	int search_exp_ptr = 0;
	if (search_exp_num_ != 0) {
		std::cout << "Content of search_exp" << std::endl;
		for (int i = 0; i < search_exp_num_; i++) {
			std::cout << "search_exp[" << i << std::endl;
			debugGTrees(search_exp_ + search_exp_ptr, search_exp_size_[i]);
			search_exp_ptr += search_exp_size_[i];
		}
	} else
		std::cout << "Empty search keys array" << std::endl;

	std::cout << "Size of innner_indices = " << indices_size_ << std::endl;
	if (indices_size_ != 0) {
		std::cout << "Content of inner indices" << std::endl;
		for (int i = 0; i < indices_size_; i++) {
			std::cout << "indices[" << i << "] = " << indices_[i] << std::endl;
		}
	} else
		std::cout << "Empty indices array" << std::endl;
}
}
