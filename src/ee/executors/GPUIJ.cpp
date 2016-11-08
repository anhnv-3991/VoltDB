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
	ulong jr_size = 0, jr_size2 = 0;
	GNValue *outer_dev, *inner_dev;
	RESULT *jresult_dev, *write_dev;
	int *indices_dev, *search_exp_size;
	ulong *index_psum, *exp_psum;
	ResBound *res_bound;
	bool *prejoin_res_dev;
	GTreeNode *initial_dev, *skipNull_dev, *prejoin_dev, *where_dev, *end_dev, *post_dev, *search_exp_dev;
	uint block_x = 0, block_y = 0, grid_x = 0, grid_y = 0;
	std::vector<unsigned long> allocation, prejoin, index, expression, ipsum, epsum, wtime, joins_only;
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;
#endif

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
	//res = cuMemAlloc(&outer_dev, part_size * sizeof(IndexData));
	checkCudaErrors(cudaMalloc(&outer_dev, part_size * outer_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&inner_dev, part_size * inner_cols_ * sizeof(GNValue)));
	checkCudaErrors(cudaMalloc(&prejoin_res_dev, part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&index_psum, gpu_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&exp_psum, gpu_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(&res_bound, gpu_size * sizeof(ResBound)));
	//cudaMemset(&count_dev, 0, gpu_size * sizeof(ulong));

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

#ifdef POST_EXP_
#ifdef FUNC_CALL_
	checkCudaErrors(cudaMalloc(&stack, block_x * grid_x * sizeof(GNValue) * MAX_STACK_SIZE));
#else
	checkCudaErrors(cudaMalloc(&val_stack, block_x * grid_x * sizeof(int64_t) * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, block_x * grid_x * sizeof(ValueType) * MAX_STACK_SIZE));
#endif
#endif

	struct timeval pre_start, pre_end, istart, iend, pistart, piend, estart, eend, pestart, peend, wstart, wend, end_join;
	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_size_; outer_idx += part_size) {
		//Size of outer small table
		uint outer_part_size = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);

		block_x = (outer_part_size < BLOCK_SIZE_X) ? outer_part_size : BLOCK_SIZE_X;
		grid_x = divUtility(outer_part_size, block_x);

		checkCudaErrors(cudaMemcpy(outer_dev, outer_table_ + outer_idx * outer_cols_, outer_part_size * outer_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));

		loop_count++;
		for (uint inner_idx = 0; inner_idx < inner_size_; inner_idx += part_size) {
			//Size of inner small table
			uint inner_part_size = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			block_y = (inner_part_size < BLOCK_SIZE_Y) ? inner_part_size : BLOCK_SIZE_Y;
			grid_y = divUtility(inner_part_size, block_y);
			block_y = 1;
			gpu_size = block_x * block_y * grid_x * grid_y + 1;

			//printf("Block_x = %d, block_y = %d, grid_x = %d, grid_y = %d\n", block_x, block_y, grid_x, grid_y);
			loop_count2++;
			/**** Copy IndexData to GPU memory ****/
			checkCudaErrors(cudaMemcpy(inner_dev, inner_table_ + inner_idx * inner_cols_, inner_part_size * inner_cols_ * sizeof(GNValue), cudaMemcpyHostToDevice));

			/* Evaluate prejoin predicate */
			gettimeofday(&pre_start, NULL);
			prejoin_filterWrapper(grid_x, grid_y, block_x, block_y, outer_dev, outer_part_size, prejoin_dev, prejoin_size_, prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
					,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
					,val_stack,
					type_stack
#endif
					);
			gettimeofday(&pre_end, NULL);

			prejoin.push_back((pre_end.tv_sec - pre_start.tv_sec) * 1000000 + (pre_end.tv_usec - pre_start.tv_usec));

			/* Binary search for index */
			gettimeofday(&istart, NULL);
			index_filterWrapper(grid_x, grid_y, block_x, block_y, outer_dev, inner_dev, index_psum, res_bound, outer_part_size, inner_part_size,
								search_exp_dev, search_exp_size, search_exp_num_, indices_dev, indices_size_, lookup_type_, prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								,val_stack,
								type_stack
#endif
								);
			gettimeofday(&iend, NULL);

			/* Prefix sum on the result */
			gettimeofday(&pistart, NULL);
			prefix_sumWrapper(index_psum, gpu_size, &jr_size);
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
			}

			//printf("jr_size = %lu\n", jr_size);
			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));

			gettimeofday(&estart, NULL);
			exp_filterWrapper(grid_x, grid_y, block_x, block_y, outer_dev, inner_dev, jresult_dev, index_psum, exp_psum,
								outer_part_size, jr_size, end_dev, end_size_, post_dev, post_size_,
								where_dev, where_size_, res_bound, outer_idx, inner_idx, prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
								,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
								,val_stack,
								type_stack
#endif
								);
			gettimeofday(&eend, NULL);


			gettimeofday(&pestart, NULL);
			prefix_sumWrapper(exp_psum, gpu_size, &jr_size2);
			gettimeofday(&peend, NULL);

			expression.push_back((eend.tv_sec - estart.tv_sec) * 1000000 + (eend.tv_usec - estart.tv_usec));
			epsum.push_back((peend.tv_sec - pestart.tv_sec) * 1000000 + (peend.tv_usec - pestart.tv_usec));

			gettimeofday(&wstart, NULL);

			if (jr_size2 == 0) {
				//printf("Empty2\n");
				checkCudaErrors(cudaFree(jresult_dev));
				continue;
			}

			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));

			write_outWrapper(grid_x, grid_y, block_x, block_y, write_dev, jresult_dev, index_psum, exp_psum, outer_part_size, jr_size2, jr_size);
			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size2) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);

			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, jresult_dev, jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost));

			checkCudaErrors(cudaFree(jresult_dev));
			checkCudaErrors(cudaFree(write_dev));

			result_size_ += jr_size2;
			jr_size = 0;
			jr_size2 = 0;
			gettimeofday(&wend, NULL);
			wtime.push_back((wend.tv_sec - wstart.tv_sec) * 1000000 + (wend.tv_usec - wstart.tv_usec));

			joins_only.push_back((end_join.tv_sec - pre_start.tv_sec) * 1000000 + (end_join.tv_usec - pre_start.tv_usec));
		}
	}


	/******** Free GPU memory, unload module, end session **************/
	checkCudaErrors(cudaFree(outer_dev));
	checkCudaErrors(cudaFree(inner_dev));
	checkCudaErrors(cudaFree(search_exp_dev));
	checkCudaErrors(cudaFree(search_exp_size));
	checkCudaErrors(cudaFree(indices_dev));
	checkCudaErrors(cudaFree(index_psum));
	checkCudaErrors(cudaFree(exp_psum));
	checkCudaErrors(cudaFree(res_bound));
	checkCudaErrors(cudaFree(prejoin_res_dev));

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

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

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

	for (int i = 0; i < joins_only.size(); i++) {
		joins_only_time += joins_only[i];
	}

	all_time = (all_end.tv_sec - all_start.tv_sec) * 1000000 + (all_end.tv_usec - all_start.tv_usec);

	allocation_time = all_time - joins_only_time;
	printf("**********************************\n"
			"Allocation & data movement time: %lu\n"
			"Prejoin filter Time: %lu\n"
			"Index Search Time: %lu\n"
			"Index Prefix Sum Time: %lu\n"
			"Expression filter Time: %lu\n"
			"Expression Prefix Sum Time: %lu\n"
			"Write back time Time: %lu\n"
			"Joins Only Time: %lu\n"
			"Total join time: %lu\n"
			"*******************************\n",
			allocation_time, prejoin_time, index_time, ipsum_time, expression_time, epsum_time, wtime_time, joins_only_time, all_time);
	printf("End of join\n");
	cudaDeviceReset();
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
	double tmp = gnvalue.getMdata();
	char gtmp[16];
	memcpy(gtmp, &tmp, sizeof(double));
	nvalue->setMdataFromGPU(gtmp);
//	nvalue->setSourceInlinedFromGPU(gnvalue.getSourceInlined());
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
