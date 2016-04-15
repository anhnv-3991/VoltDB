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
#include "GPUIJ.h"
#include "scan_common.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cudaProfiler.h>


using namespace voltdb;

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
				TreeExpression where_expression)
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

int compareTime(const void *a, const void *b)
{
	long int x = *((long int*)a);
	long int y = *((long int*)b);

	return (x > y) ? 1 : ((x < y) ? -1 : 0);
}

bool GPUIJ::join(){
	int loop_count = 0, loop_count2 = 0;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction index_filter, expression_filter, write_out;
	CUmodule module, c_module;
	char fname[256];
	char *vd;
	char path[256];

	struct timeval all_start, all_end;
	gettimeofday(&all_start, NULL);
	if (outer_size_ == 0 || inner_size_ == 0) {
		return true;
	}


	/****************** Initialize GPU ********************/
	if ((vd = getenv("VOLT_HOME")) != NULL) {
		snprintf(path, 256, "%s/voltdb", vd);
	} else if ((vd = getenv("HOME")) != NULL) {
		snprintf(path, 256, "%s/voltdb", vd);
	} else
		return false;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		printf("cuInit failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	sprintf(fname, "%s/index_join_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleLoad(join) failed: res = %lu\n file name = %s\n", (unsigned long)res, fname);
		return false;
	}

	res = cuModuleGetFunction(&index_filter, module, "index_filter");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(index_filter) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuModuleGetFunction(&expression_filter, module, "expression_filter");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(expression_filter) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuModuleGetFunction(&write_out, module, "write_out");
	if (res != CUDA_SUCCESS) {
		printf("cuModuleGetFunction(write_out) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size = 0, jr_size2 = 0;
	CUdeviceptr outer_dev, inner_dev, jresult_dev, write_dev, index_psum, exp_psum, presum_dev, end_ex_dev, post_ex_dev, search_exp_dev, indices_dev, res_bound, search_exp_size;
	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;
	std::vector<unsigned long> allocation, index, expression, ipsum, epsum, wtime, joins_only;

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

	/******** Allocate GPU buffer for table data and counting data *****/
	//res = cuMemAlloc(&outer_dev, part_size * sizeof(IndexData));
	res = cuMemAlloc(&outer_dev, part_size * outer_cols_ * sizeof(GNValue));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(outer_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&inner_dev, part_size * inner_cols_ * sizeof(GNValue));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(inner_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	printf("Original GPU SIZE = %d\n", gpu_size);

	res = cuMemAlloc(&index_psum, gpu_size * sizeof(ulong));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(index_psum) failed: res = %lu\n gpu_size = %u\n", (unsigned long)res, gpu_size);
		return false;
	}

	res = cuMemAlloc(&exp_psum, gpu_size * sizeof(ulong));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(exp_psum) failed: res = %lu\n gpu_size = %u\n", (unsigned long)res, gpu_size);
		return false;
	}

	res = cuMemAlloc(&res_bound, gpu_size * sizeof(ResBound));
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(res_bound) failed: res = %lu\n gpu_size = %u\n", (unsigned long)res, gpu_size);
		return false;
	}

	//cudaMemset(&count_dev, 0, gpu_size * sizeof(ulong));

	/******* Allocate GPU buffer for join condition *********/
	if (end_size_ >= 1) {
		res = cuMemAlloc(&end_ex_dev, end_size_ * sizeof(GTreeNode));
		if (res != CUDA_SUCCESS) {
			printf("cuMemAlloc(end_ex_dev) failed: res = %lu\n", (unsigned long)res);
			return false;
		}

		res = cuMemcpyHtoD(end_ex_dev, end_expression_, end_size_ * sizeof(GTreeNode));
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD(end_ex_dev, end_expression) failed: res = %lu\n", (unsigned long)res);
			return false;
		}
	}

	if (post_size_ >= 1) {
		res = cuMemAlloc(&post_ex_dev, post_size_ * sizeof(GTreeNode));
		if (res != CUDA_SUCCESS) {
			printf("cuMemAlloc(post_ex_dev) failed: res = %lu\n", (unsigned long)res);
			return false;
		}

		res = cuMemcpyHtoD(post_ex_dev, post_expression_, post_size_ * sizeof(GTreeNode));
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD(post_ex_dev, post_expression) failed: res = %lu\n", (unsigned long)res);
			return false;
		}
	}

	/******* Allocate GPU buffer for search keys and index keys *****/
	int tmp_size = 0;
	for (int i = 0; i < search_exp_num_; i++) {
		tmp_size += search_exp_size_[i];
	}
	res = cuMemAlloc(&search_exp_dev, sizeof(GTreeNode) * tmp_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(search_exp_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&search_exp_size, sizeof(int) * search_exp_num_);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(search_exp_size) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemcpyHtoD(search_exp_dev, search_exp_, sizeof(GTreeNode) * tmp_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD(search_exp_dev, search_exp_) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemcpyHtoD(search_exp_size, search_exp_size_, sizeof(int) * search_exp_num_);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD(search_exp_size, search_exp_size_) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&indices_dev, sizeof(int) * indices_size_);
	if (res != CUDA_SUCCESS) {
		printf("cuMemAlloc(indices_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemcpyHtoD(indices_dev, indices_, sizeof(int) * indices_size_);
	if (res != CUDA_SUCCESS) {
		printf("cuMemcpyHtoD(indices_dev, indices_) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	struct timeval istart, iend, pistart, piend, estart, eend, pestart, peend, wstart, wend, end_join;
	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_size_; outer_idx += part_size) {
		//Size of outer small table
		uint outer_part_size = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);

		block_x = (outer_part_size < BLOCK_SIZE_X) ? outer_part_size : BLOCK_SIZE_X;
		grid_x = divUtility(outer_part_size, block_x);

		res = cuMemcpyHtoD(outer_dev, outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue));
		if (res != CUDA_SUCCESS) {
			printf("cuMemcpyHtoD(outer_dev, outer_table_ + %u) failed: res = %lu\n", outer_idx, (unsigned long)res);
			return false;
		}
		loop_count++;
		for (uint inner_idx = 0; inner_idx < inner_size_; inner_idx += part_size) {
			//Size of inner small table
			uint inner_part_size = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			block_y = (inner_part_size < BLOCK_SIZE_Y) ? inner_part_size : BLOCK_SIZE_Y;
			grid_y = divUtility(inner_part_size, block_y);
			block_y = 1;
			gpu_size = block_x * block_y * grid_x * grid_y + 1;

			loop_count2++;
			/**** Copy IndexData to GPU memory ****/
			res = cuMemcpyHtoD(inner_dev, inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue));
			if (res != CUDA_SUCCESS) {
				printf("cuMemcpyHtoD(inner_dev, inner_table_ + %u) failed: res = %lu\n", inner_idx, (unsigned long)res);
				return false;
			}

			void *index_filter_arg[] = {
										(void *)&outer_dev,
										(void *)&inner_dev,
										(void *)&index_psum,
										(void *)&res_bound,
										(void *)&outer_part_size,
										(void *)&outer_cols_,
										(void *)&inner_part_size,
										(void *)&inner_cols_,
										(void *)&search_exp_dev,
										(void *)&search_exp_size,
										(void *)&search_exp_num_,
										(void *)&indices_dev,
										(void *)&indices_size_
										};

			gettimeofday(&istart, NULL);
			res = cuLaunchKernel(index_filter, grid_x, grid_y, 1, block_x, block_y, 1, 0, NULL, index_filter_arg, NULL);
			if (res != CUDA_SUCCESS) {
				printf("cuLaunchKernel(index_filter) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuCtxSynchronize();
			if (res != CUDA_SUCCESS) {
				printf("cuCtxSynchronize(index_filter) failed: res = %lu\n", (unsigned long)res);
				return false;
			}
			gettimeofday(&iend, NULL);

			gettimeofday(&pistart, NULL);
			if (!((new GPUSCAN<ulong, ulong4>)->presum(&index_psum, gpu_size))) {
				printf("Prefix(&index_filter_dev, gpu_size) sum error.\n");
				return false;
			}

			if (!((new GPUSCAN<ulong, ulong4>)->getValue(index_psum, gpu_size, &jr_size))) {
				printf("getValue(index_filter_dev, gpu_size, &jr_size) error");
				return false;
			}

			gettimeofday(&piend, NULL);

			index.push_back((iend.tv_sec - istart.tv_sec) * 1000000 + (iend.tv_usec - istart.tv_usec));
			ipsum.push_back((piend.tv_sec - pistart.tv_sec) * 1000000 + (piend.tv_usec - pistart.tv_usec));

			if (jr_size < 0) {
				printf("Scanning failed\n");
				return false;
			}

			if (jr_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back((end_join.tv_sec - istart.tv_sec) * 1000000 + (end_join.tv_usec - istart.tv_usec));
				continue;
				//goto free_count;
			}

			res = cuMemAlloc(&jresult_dev, jr_size * sizeof(RESULT));
			if (res != CUDA_SUCCESS) {
				printf("cuMemAlloc(jresult_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			void *exp_filter_arg[] = {
											(void *)&outer_dev,
											(void *)&inner_dev,
											(void *)&jresult_dev,
											(void *)&index_psum,
											(void *)&exp_psum,
											(void *)&outer_part_size,
											(void *)&outer_cols_,
											(void *)&inner_cols_,
											(void *)&jr_size,
											(void *)&post_ex_dev,
											(void *)&post_size_,
											(void *)&res_bound,
											(void *)&outer_idx,
											(void *)&inner_idx
											};

			gettimeofday(&estart, NULL);

			res = cuLaunchKernel(expression_filter, grid_x, grid_y, 1, block_x, block_y, 1, 0, NULL, exp_filter_arg, NULL);
			if (res != CUDA_SUCCESS) {
				printf("cuLaunchKernel(expression_filter) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuCtxSynchronize();
			if (res != CUDA_SUCCESS) {
				printf("cuCtxSynchronize(expression_filter) failed: res = %lu\n", (unsigned long)res);
				return false;
			}
			gettimeofday(&eend, NULL);

			gettimeofday(&pestart, NULL);
			if (!((new GPUSCAN<ulong, ulong4>)->presum(&exp_psum, gpu_size))) {
				printf("Prefix(&exp_sum, gpu_size) sum error.\n");
				return false;
			}

			if (!((new GPUSCAN<ulong, ulong4>)->getValue(exp_psum, gpu_size, &jr_size2))) {
				printf("getValue(exp_sum, gpu_size, &jr_size) error");
				return false;
			}
			gettimeofday(&peend, NULL);

			expression.push_back((eend.tv_sec - estart.tv_sec) * 1000000 + (eend.tv_usec - estart.tv_usec));
			epsum.push_back((peend.tv_sec - pestart.tv_sec) * 1000000 + (peend.tv_usec - pestart.tv_usec));
			gettimeofday(&wstart, NULL);

			res = cuMemAlloc(&write_dev, jr_size2 * sizeof(RESULT));
			if (res != CUDA_SUCCESS) {
				printf("cuMemAlloc(write_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			void *write_back_arg[] = {
										(void *)&write_dev,
										(void *)&jresult_dev,
										(void *)&index_psum,
										(void *)&exp_psum,
										(void *)&outer_part_size,
										(void *)&jr_size2,
										(void *)&jr_size
									};

			res = cuLaunchKernel(write_out, grid_x, grid_y, 1, block_x, block_y, 1, 0, NULL, write_back_arg, NULL);
			if (res != CUDA_SUCCESS) {
				printf("cuLaunchKernel(write_out) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuCtxSynchronize();
			if (res != CUDA_SUCCESS) {
				printf("cuCtxSynchronize(write_out) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size2) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);

			res = cuMemcpyDtoH(join_result_ + result_size_, write_dev, jr_size2 * sizeof(RESULT));
			if (res != CUDA_SUCCESS) {
				printf("cuMemcpyDtoH(join_result_[%u], jresult_dev) failed: res = %lu\n", result_size_, (unsigned long)res);
				return false;
			}

			res = cuMemFree(jresult_dev);
			if (res != CUDA_SUCCESS) {
				printf("cuMemFree(jresult_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}
			res = cuMemFree(write_dev);
			if (res != CUDA_SUCCESS) {
				printf("cuMemFree(write_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			result_size_ += jr_size2;
			jr_size = 0;
			gettimeofday(&wend, NULL);
			wtime.push_back((wend.tv_sec - wstart.tv_sec) * 1000000 + (wend.tv_usec - wstart.tv_usec));

			joins_only.push_back((end_join.tv_sec - istart.tv_sec) * 1000000 + (end_join.tv_usec - istart.tv_usec));
		}
	}


	/******** Free GPU memory, unload module, end session **************/
	res = cuMemFree(outer_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(outer_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(inner_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(inner_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(search_exp_dev);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(search_exp_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(search_exp_size);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(search_exp_size) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(index_psum);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(count_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(exp_psum);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(count_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(res_bound);
	if (res != CUDA_SUCCESS) {
		printf("cuMemFree(res_bound) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		printf("cuModuleUnload(module) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		printf("cuCtxDestroy(ctx) failed: res = %lu\n", (unsigned long)res);
		return false;
	}
	gettimeofday(&all_end, NULL);

	unsigned long allocation_time = 0, index_time = 0, expression_time = 0, ipsum_time = 0, epsum_time = 0, wtime_time = 0, joins_only_time = 0, all_time = 0;
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

	for (int i = 0; i < joins_only.size(); i++) {
		joins_only_time += joins_only[i];
	}

	all_time = (all_end.tv_sec - all_start.tv_sec) * 1000000 + (all_end.tv_usec - all_start.tv_usec);

	allocation_time = all_time - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"Index Search Time: %lu\n"
			"Index Prefix Sum Time: %lu\n"
			"Expression filter Time: %lu\n"
			"Expression Prefix Sum Time: %lu\n"
			"Write back time Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, index_time, ipsum_time, expression_time, epsum_time, wtime_time, joins_only_time, all_time);
	printf("End of join\n");
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

void GPUIJ::setNValue(NValue *nvalue, GNValue &gnvalue)
{
	nvalue->setMdataFromGPU(gnvalue.getMdata());
	nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
	nvalue->setValueTypeFromGPU(gnvalue.getValueType());
}

void GPUIJ::debugGTrees(const GTreeNode *expression, int size)
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
