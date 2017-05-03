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
#include "gpuij.h"
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>
#include <thrust/system/cuda/execution_policy.h>
#include "GPUetc/expressions/gexpression.h"


namespace voltdb {

GPUIJ::GPUIJ()
{
		outer_table_.block_list = inner_table_.block_list = NULL;
		join_result_ = NULL;
		result_size_ = 0;
		indices_size_ = 0;
		search_exp_size_ = NULL;
		search_exp_num_ = 0;
		indices_ = NULL;

		search_exp_ = NULL;
		lookup_type_ = INDEX_LOOKUP_TYPE_EQ;
}

GPUIJ::GPUIJ(GTable outer_table,
				GTable inner_table,
				std::vector<TreeExpression> search_exp,
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
	join_result_ = NULL;
	result_size_ = 0;
	search_exp_num_ = search_exp.size();
	lookup_type_ = lookup_type;


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

	/******* Allocate GPU buffer for search keys and index keys *****/
	checkCudaErrors(cudaMalloc(&search_exp_, sizeof(GTreeNode) * tmp_size));
	checkCudaErrors(cudaMalloc(&search_exp_size_, sizeof(int) * search_exp_num_));

	checkCudaErrors(cudaMemcpy(search_exp_, search_exp_tmp, sizeof(GTreeNode) * tmp_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(search_exp_size_, search_exp_size_tmp, sizeof(int) * search_exp_num_, cudaMemcpyHostToDevice));

	/**** Expression data ****/

	end_expression_ = GExpression(end_expression);

	post_expression_ = GExpression(post_expression);

	initial_expression_ = GExpression(initial_expression);

	skipNullExpr_ = GExpression(skipNullExpr);

	prejoin_expression_ = GExpression(prejoin_expression);

	where_expression_ = GExpression(where_expression);
}

GPUIJ::~GPUIJ()
{
	freeArrays<RESULT>(join_result_);
	freeArrays<GTreeNode>(search_exp_);
	freeArrays<int>(search_exp_size_);
}

bool GPUIJ::join(){
	int loop_count = 0, loop_count2 = 0;
	cudaError_t res;

	gettimeofday(&all_start_, NULL);

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size = 0, part_size = 0;
	ulong jr_size, jr_size2;

	RESULT *jresult_dev, *write_dev;
	jresult_dev = write_dev = NULL;
	ulong *index_psum, *exp_psum;
	ResBound *res_bound;
	bool *prejoin_res_dev;

	part_size = getPartitionSize();

	int block_x, grid_x;

	block_x = (part_size < BLOCK_SIZE_X) ? part_size : BLOCK_SIZE_X;
	grid_x = (part_size - 1)/block_x + 1;
	gpu_size = DEFAULT_PART_SIZE_ + 1;

	/******** Allocate GPU buffer for table data and counting data *****/
	checkCudaErrors(cudaMalloc(&prejoin_res_dev, part_size * sizeof(bool)));
	checkCudaErrors(cudaMalloc(&index_psum, gpu_size * sizeof(ulong)));

#ifndef DECOMPOSED1_
	checkCudaErrors(cudaMalloc(&exp_psum, gpu_size * sizeof(ulong)));
#endif
	checkCudaErrors(cudaMalloc(&res_bound, gpu_size * sizeof(ResBound)));

	struct timeval pre_start, pre_end, istart, iend, pistart, piend, estart, eend, pestart, peend, wstart, wend, end_join, balance_start, balance_end;

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (int outer_idx = 0; outer_idx < outer_table_.getBlockNum(); outer_idx++) {
		//Size of outer small table
		outer_chunk_ = outer_table_[outer_idx];
		gpu_size = outer_table_.getBlockTuple(outer_idx) + 1;

		/* Evaluate prejoin predicate */
		gettimeofday(&pre_start, NULL);
		PrejoinFilter(prejoin_res_dev);
		gettimeofday(&pre_end, NULL);
		prejoin_.push_back(timeDiff(pre_start, pre_end));

		joins_only_.push_back(timeDiff(pre_start, pre_end));

		for (int inner_idx = 0; inner_idx < inner_table_.getBlockNum(); inner_idx++) {
			/* Binary search for index */
			inner_chunk_ = inner_table_[inner_idx];
			gettimeofday(&istart, NULL);

			IndexFilter(index_psum, res_bound, prejoin_res_dev);

			gettimeofday(&iend, NULL);
			index_.push_back(timeDiff(istart, iend));

#if !defined(DECOMPOSED1_) && !defined(DECOMPOSED2_)
			/* Prefix sum on the result */
			gettimeofday(&pistart, NULL);
			GUtilities::ExclusiveScan(index_psum, gpu_size, &jr_size);
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

			checkCudaErrors(cudaMalloc(&jresult_dev, jr_size * sizeof(RESULT)));

			gettimeofday(&estart, NULL);
			ExpressionFilter(jresult_dev, index_psum, exp_psum, jr_size, res_bound, prejoin_res_dev);
			gettimeofday(&eend, NULL);


			gettimeofday(&pestart, NULL);
			GUtilities::ExclusiveScan(exp_psum, gpu_size, &jr_size2);
			gettimeofday(&peend, NULL);

			expression.push_back(timeDiff(estart, eend));
			epsum.push_back(timeDiff(pestart, peend));

			if (jr_size2 == 0) {
				checkCudaErrors(cudaFree(jresult_dev));
				continue;
			}

			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));


			gettimeofday(&wstart, NULL);
			GUtilities::RemoveEmptyResult(write_dev, jresult_dev, index_psum, exp_psum, jr_size);
			gettimeofday(&wend, NULL);
#elif defined(DECOMPOSED1_)
			RESULT *tmp_result;
			ulong tmp_size = 0;

			gettimeofday(&balance_start, NULL);
			Rebalance3(index_psum, res_bound, &tmp_result, gpu_size, &tmp_size);
			gettimeofday(&balance_end, NULL);

			rebalance_.push_back(timeDiff(balance_start, balance_end));

			if (tmp_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only_.push_back(timeDiff(istart, end_join));
				continue;
			}
			checkCudaErrors(cudaMalloc(&jresult_dev, tmp_size * sizeof(RESULT)));
			checkCudaErrors(cudaMalloc(&exp_psum, (tmp_size + 1) * sizeof(ulong)));

			gettimeofday(&estart, NULL);
			ExpressionFilter(tmp_result, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&eend, NULL);

			expression_.push_back(timeDiff(estart, eend));

			gettimeofday(&pestart, NULL);
			GUtilities::ExclusiveScan(exp_psum, tmp_size + 1, &jr_size2);
			gettimeofday(&peend, NULL);

			epsum_.push_back(timeDiff(pestart, peend));

			checkCudaErrors(cudaFree(tmp_result));

			if (jr_size2 == 0) {
				continue;
			}
			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));

			gettimeofday(&wstart, NULL);
			GUtilities::RemoveEmptyResult(write_dev, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&wend, NULL);
#elif defined(DECOMPOSED2_)
			RESULT *tmp_result;
			ulong tmp_size = 0;

			gettimeofday(&balance_start, NULL);
			Rebalance2(index_psum, res_bound, &tmp_result, gpu_size, &tmp_size);
			gettimeofday(&balance_end, NULL);

			rebalance.push_back(timeDiff(balance_start, balance_end));

			if (tmp_size == 0) {
				gettimeofday(&end_join, NULL);
				joins_only.push_back(timeDiff(istart, end_join));
				continue;
			}

			checkCudaErrors(cudaMalloc(&jresult_dev, tmp_size * sizeof(RESULT)));
			checkCudaErrors(cudaMalloc(&exp_psum, (tmp_size + 1) * sizeof(ulong)));

			gettimeofday(&estart, NULL);
			ExpressionFilter(tmp_result, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&eend, NULL);

			expression.push_back(timeDiff(estart, eend));

			gettimeofday(&pestart, NULL);
			GUtilities::ExclusiveScan(exp_psum, tmp_size + 1, &jr_size2);
			gettimeofday(&peend, NULL);

			epsum.push_back(timeDiff(pestart, peend));

			checkCudaErrors(cudaFree(tmp_result));

			if (jr_size2 == 0)
				continue;
			checkCudaErrors(cudaMalloc(&write_dev, jr_size2 * sizeof(RESULT)));

			gettimeofday(&wstart, NULL);
			GUtilities::RemoveEmptyResult(write_dev, jresult_dev, exp_psum, tmp_size);
			gettimeofday(&wend, NULL);
#endif

			wtime_.push_back(timeDiff(wstart, wend));

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size2) * sizeof(RESULT));

			gettimeofday(&end_join, NULL);
			checkCudaErrors(cudaMemcpy(join_result_ + result_size_, write_dev, jr_size2 * sizeof(RESULT), cudaMemcpyDeviceToHost));

			result_size_ += jr_size2;
			jr_size = 0;
			jr_size2 = 0;

			joins_only_.push_back(timeDiff(istart, end_join));
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	/******** Free GPU memory, unload module, end session **************/

#ifndef DECOMPOSED1_
	checkCudaErrors(cudaFree(exp_psum));
#endif
	checkCudaErrors(cudaFree(res_bound));
	checkCudaErrors(cudaFree(prejoin_res_dev));
	gettimeofday(&all_end_, NULL);

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
	uint outer_size = outer_table_.getTupleCount();
	uint inner_size = inner_table_.getTupleCount();
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


bool GPUIJ::getTreeNodes(GTree *expression, const TreeExpression tree_expression)
{
	if (tree_expression.getSize() > 0) {
		checkCudaErrors(cudaMalloc(&expression->exp, tree_expression.getSize() * sizeof(GTreeNode)));
		checkCudaErrors(cudaMemcpy(expression->exp, tree_expression.getNodesArray2(), tree_expression.getSize() * sizeof(GTreeNode), cudaMemcpyHostToDevice));
		expression->size = tree_expression.getSize();
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

void GPUIJ::freeArrays2(GTree expression)
{
	if (expression.size > 0)
		checkCudaErrors(cudaFree(expression.exp));

}



void GPUIJ::debug(void)
{
//	std::cout << "Size of outer table = " << outer_size_ << std::endl;
//	if (outer_size_ != 0) {
//		std::cout << "Outer table" << std::endl;
//		for (int i = 0; i < outer_size_; i++) {
//			for (int j = 0; j < MAX_GNVALUE; j++) {
//				NValue tmp;
//				setNValue(&tmp, outer_table_[i * outer_cols_ + j]);
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


void GPUIJ::profiling()
{
	unsigned long allocation_time = 0, prejoin_time = 0, index_time = 0, expression_time = 0, ipsum_time = 0, epsum_time = 0, wtime_time = 0, joins_only_time = 0, all_time = 0;

	for (int i = 0; i < prejoin_.size(); i++) {
		prejoin_time += prejoin_[i];
	}

	for (int i = 0; i < index_.size(); i++) {
		index_time += index_[i];
	}

	for (int i = 0; i < expression_.size(); i++) {
		expression_time += expression_[i];
	}

	for (int i = 0; i < ipsum_.size(); i++) {
		ipsum_time += ipsum_[i];
	}

	for (int i = 0; i < epsum_.size(); i++) {
		epsum_time += epsum_[i];
	}

	for (int i = 0; i < wtime_.size(); i++) {
		wtime_time += wtime_[i];
	}

#if (defined(DECOMPOSED1_) || defined(DECOMPOSED2_))
	unsigned long rebalance_cost = 0;
	for (int i = 0; i < rebalance_.size(); i++) {
		rebalance_cost += rebalance_[i];
	}
#endif


	for (int i = 0; i < joins_only_.size(); i++) {
		joins_only_time += joins_only_[i];
	}

	all_time = (all_end_.tv_sec - all_start_.tv_sec) * 1000000 + (all_end_.tv_usec - all_start_.tv_usec);

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

}

unsigned long GPUIJ::timeDiff(struct timeval start, struct timeval end)
{
	return GUtilities::timeDiff(start, end);
}

extern "C" __global__ void PrejoinFilterDev(GTable outer, GExpression prejoin, bool *result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
											,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
											,int64_t *val_stack, ValueType *type_stack
#endif
)
{
	int64_t *outer_dev = outer.getBlock(0)->gdata;
	int outer_cols = outer.getColumnCount();
	int outer_rows = outer.getBlock(0)->rows;
	GColumnInfo *schema = outer.getSchema();

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i+= offset) {
		GNValue res = GNValue::getTrue();

#ifdef 	TREE_EVAL_
#ifdef FUNC_CALL_
		res = (prejoin.getSize() > 1) ? prejoin.evaluate(1, outer_dev + i * outer_cols, NULL, schema, NULL) : res;
#else
		res = (prejoin.getSize() > 1) ? prejoin.evaluate2(1, outer_dev + i * outer_cols, NULL, schema, NULL) : res;
#endif
#elif	POST_EXP_
#ifndef FUNC_CALL_
		res = (prejoin.getSize() > 1) ? prejoin.evaluate(outer_dev + i * outer_cols, NULL, schema, NULL, val_stack + index, type_stack + index, offset) : res;
#else
		res = (prejoin.getSize() > 1) ? prejoin.evaluate(prejoin, outer_dev + i * outer_cols, NULL, schema, NULL, stack + index, offset) : res;
#endif
#endif
		result[i] = res.isTrue();
	}
}

void GPUIJ::PrejoinFilter(bool *result)
{
	int outer_rows = outer_chunk_.getBlock().rows;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	PrejoinFilterDev<<<grid_size, block_size>>>(outer_chunk_, prejoin_expression_, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,val_stack,
												type_stack
#endif
												);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void GPUIJ::PrejoinFilter(bool *result, cudaStream_t stream)
{
	int outer_rows = outer_chunk_.getBlock().rows;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	PrejoinFilterDev<<<grid_size, block_size, 0, stream>>>(outer_chunk_, prejoin_expression_, result
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
															,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
															,val_stack,
															type_stack
#endif
															);
	checkCudaErrors(cudaGetLastError());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

extern "C" __global__ void decomposeDev(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i].lkey = in[in_location[i]].outer;
		out[i].rkey = in[in_location[i]].left + local_offset[i];
	}
}

void GPUIJ::Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	decomposeDev<<<grid_size, block_x>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GPUIJ::Decompose(ResBound *in, RESULT *out, ulong *in_location, ulong *local_offset, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	decomposeDev<<<grid_size, block_x, 0, stream>>>(in, out, in_location, local_offset, size);
	checkCudaErrors(cudaGetLastError());
}


extern "C" __global__ void IndexFilterLowerBound(GTable search_table, GTable inner,
												  ulong *index_psum, ResBound *res_bound,
												  GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
												  int *key_indices, int key_index_size,
												  IndexLookupType lookup_type,
												  bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												  ,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												  ,int64_t *val_stack,
												  ValueType *type_stack
#endif
										  )

{
	int outer_rows = search_table.getBlock().rows, inner_rows = inner.getBlock().rows;
	GTreeIndex inner_idx = inner.getIndex();

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i += offset) {
		res_bound[i].left = -1;
		res_bound[i].outer = -1;

		if (prejoin_res_dev[i]) {
			res_bound[i].outer = i;

			GTuple tuple(search_table, i);
			GKeyIndex outer_key(tuple);

			switch (lookup_type) {
			case INDEX_LOOKUP_TYPE_EQ:
			case INDEX_LOOKUP_TYPE_GT:
			case INDEX_LOOKUP_TYPE_GTE:
			case INDEX_LOOKUP_TYPE_LT: {
				res_bound[i].left = inner_idx.lowerBound(outer_key, 0, inner_rows - 1);
				break;
			}
			case INDEX_LOOKUP_TYPE_LTE: {
				res_bound[i].left = 0;
				break;
			}
			default:
				break;
			}
		}
	}
}

extern "C" __global__ void IndexFilterUpperBound(GTable search_table, GTable inner,
												  ulong *index_psum, ResBound *res_bound,
												  GTreeNode *search_exp_dev, int *search_exp_size, int search_exp_num,
												  IndexLookupType lookup_type,
												  bool *prejoin_res_dev
#if (defined(POST_EXP_) && !defined(FUNC_CALL_))
												  ,int64_t *val_stack,
												  ValueType *type_stack
#elif (defined(POST_EXP_) && defined(FUNC_CALL_))
												  ,GNValue *stack
#endif
										  )

{
	int outer_rows = search_table.getBlock().rows, inner_rows = inner.getBlock().rows;
	GTreeIndex inner_idx = inner.getIndex();

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i += offset) {
		index_psum[i] = 0;
		res_bound[i].right = -1;

		if (prejoin_res_dev[i]) {
			GTuple tuple(search_table, i);
			GKeyIndex outer_key(tuple);

			switch (lookup_type) {
			case INDEX_LOOKUP_TYPE_EQ:
			case INDEX_LOOKUP_TYPE_LTE: {
				res_bound[i].right = inner_idx.upperBound(outer_key, 0, inner_rows - 1);
				break;
			}
			case INDEX_LOOKUP_TYPE_GT:
			case INDEX_LOOKUP_TYPE_GTE: {
				res_bound[i].right = inner.block_list->rows - 1;
				break;
			}
			case INDEX_LOOKUP_TYPE_LT: {
				res_bound[i].right = res_bound[i].left - 1;
				res_bound[i].left = 0;
				break;
			}
			default:
				break;
			}
		}

		index_psum[i] = (res_bound[i].right >= 0 && res_bound[i].left >= 0) ? (res_bound[i].right - res_bound[i].left + 1) : 0;
	}

	if (index == 0)
		index_psum[outer_rows] = 0;
}


extern "C" __global__ void constructSearchTable(GTable outer_table, GTable search_table,
												GTreeNode * search_exp, int *search_exp_size, int search_exp_num,
												int64_t *val_stack, ValueType *type_stack,
												int offset)
{
	int outer_rows = outer_table.getTupleCount();
	GNValue tmp;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i += stride) {
		for (int j = 0, search_ptr = 0; i < search_exp_num; search_ptr += search_exp_size[j], j++) {
			GTuple tuple(search_table, i);
			GExpression pred(search_exp + search_ptr, search_exp_size[i]);

			tmp = pred.evaluate(outer_table, NULL, outer_schema, NULL, val_stack, type_stack, offset);
			tuple.attachColumn(tmp);
		}
	}
}

void GPUIJ::IndexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev)
{
	GTable search_table;
	int outer_rows = outer_chunk_.getBlock()->rows;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);
	GTable searchTable;

	IndexFilterLowerBound<<<grid_size, block_size>>>(outer_chunk_, inner_chunk_,
														index_psum, res_bound,
														search_exp_, search_exp_size_,
														search_exp_num_,
														lookup_type_,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
#endif
													);

	checkCudaErrors(cudaGetLastError());

	IndexFilterUpperBound<<<grid_size, block_size>>>(outer_chunk_, inner_chunk_,
														index_psum, res_bound,
														search_exp_, search_exp_size_,
														search_exp_num_,
														lookup_type_,
														prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
														,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
														,val_stack,
														type_stack
#endif
														);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void GPUIJ::IndexFilter(ulong *index_psum, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream)
{
	int outer_rows = outer_chunk_.getBlock().rows;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	IndexFilterLowerBound<<<grid_size, block_size, 0, stream>>>(outer_chunk_, inner_chunk_,
																index_psum, res_bound,
																search_exp_, search_exp_size_,
																search_exp_num_,
																lookup_type_,
																prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																,val_stack,
																type_stack
#endif
																);

	checkCudaErrors(cudaGetLastError());

	IndexFilterUpperBound<<<grid_size, block_size, 0, stream>>>(outer_chunk_, inner_chunk_,
																index_psum, res_bound,
																search_exp_, search_exp_size_,
																search_exp_num_,
																lookup_type_,
																prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																,stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																,val_stack,
																type_stack
#endif
																);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}


extern "C" __global__ void ExpressionFilterDev2(GTable outer, GTable inner,
												RESULT *in_bound, RESULT *out_bound,
												ulong *mark_location, int size,
												GExpression end_exp, GExpression post_exp, GExpression where_exp
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,int64_t *val_stack, ValueType *type_stack
#endif
											)
{
	int outer_cols = outer.getColumnCount(), inner_cols = inner.getColumnCount();
	int64_t *outer_table = outer.getBlock().data, *inner_table = inner.getBlock().data;
	GColumnInfo *outer_schema = outer.getSchema(), *inner_schema = inner.getSchema();
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	GNValue res;

	for (int i = index; i < size; i += offset) {
		res = GNValue::getTrue();
#ifdef TREE_EVAL_
#ifdef FUNC_CALL_
		res = (end_exp.getSize() > 1) ? end_exp.evaluate(1, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema) : res;
		res = (post_exp.getSize() > 1 && res.isTrue()) ? post_exp.evaluate(1, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema) : res;
#else
		res = (end_exp.getSize() > 1) ? end_exp.evaluate2(1, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema) : res;
		res = (post_exp.getSize() > 1 && res.isTrue()) ? post_exp.evaluate2(1, outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema) : res;
#endif
#else
#ifdef FUNC_CALL_
		res = (end_exp.getSize() > 0) ? end_exp.evaluate(outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
		res = (post_exp.getSize() > 0 && res.isTrue()) ? post_exp.evaluate(outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
#else
		res = (end_exp.getSize() > 0) ? end_exp.evaluate(outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
		res = (post_exp.getSize() > 0 && res.isTrue()) ? post_exp.evaluate(outer_table + in_bound[i].lkey * outer_cols, inner_table + in_bound[i].rkey * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
		out_bound[i].lkey = (res.isTrue()) ? in_bound[i].lkey : (-1);
		out_bound[i].rkey = (res.isTrue()) ? in_bound[i].rkey : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void GPUIJ::ExpressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilterDev2<<<grid_size, block_size>>>(outer_chunk_, inner_chunk_,
													in_bound, out_bound,
													mark_location, size,
													end_expression_, post_expression_, where_expression_
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
													, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
													, val_stack, type_stack
#endif
												);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void GPUIJ::ExpressionFilter(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilter2<<<grid_size, block_size, 0, stream>>>(outer_chunk_, inner_chunk_,
															in_bound, out_bound,
															mark_location, size,
															end_expression_, post_expression_, where_expression_
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
															, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
															, val_stack, type_stack
#endif
															);
	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

extern "C"__global__ void ExpressionFilterSharedDev(GTable outer, GTable inner,
													RESULT *in_bound, RESULT *out_bound,
													ulong *mark_location, int size,
													GExpression end_exp, GExpression post_exp, GExpression where_exp
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
													, GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
													, int64_t *val_stack, ValueType *type_stack
#endif
												)
{
	int outer_cols = outer.getColumnCount(), inner_cols = inner.getColumnCount();
	int64_t *outer_table = outer.getBlock().data, *inner_table = inner.getBlock().data;
	GColumnInfo *outer_schema = outer.getSchema(), *inner_schema = inner.getSchema();

	extern __shared__ int64_t shared_tmp[];
	int64_t *tmp_outer = shared_tmp;
	int64_t *tmp_inner = shared_tmp + blockDim.x * outer_cols;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	GNValue res;
	int outer_idx, inner_idx;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < size; i += offset) {
		outer_idx = in_bound[i].lkey;
		inner_idx = in_bound[i].rkey;
		// Load outer tuples to shared memory
		for (int j = 0; j < outer_cols; j++) {
			tmp_outer[threadIdx.x + j] = outer_table[outer_idx * outer_cols + j];
		}
		// Load inner tuples to shared memory
		for (int j = 0; j < inner_cols; j++) {
			tmp_inner[threadIdx.x + j] = inner_table[inner_idx * inner_cols + j];
		}

		__syncthreads();

		res = GNValue::getTrue();
#ifdef TREE_EVAL_
#ifdef FUNC_CALL_
		res = (end_exp.getSize() > 1) ? end_exp.evaluate(1, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema) : res;
		res = (post_exp.getSize() > 1 && res.isTrue()) ? post_exp.evaluate(1, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema) : res;
#else
		res = (end_exp.getSize() > 1) ? end_exp.evaluate2(1, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema) : res;
		res = (post_exp.getSize() > 1 && res.isTrue()) ? post_exp.evaluate2(1, tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema) : res;
#endif
#else
#ifdef FUNC_CALL_
		res = (end_exp.getSize() > 0) ? end_exp.evaluate(tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
		res = (post_exp.getSize() > 0 && res.isTrue()) ? post_exp.evaluate(tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
#else
		res = (end_exp.getSize() > 0) ? end_exp.evaluate(tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
		res = (post_exp.getSize() > 0 && res.isTrue()) ? post_exp.evaluate(tmp_outer + threadIdx.x * outer_cols, tmp_inner + threadIdx.x * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
		out_bound[i].lkey = (res.isTrue()) ? (in_bound[i].lkey) : (-1);
		out_bound[i].rkey = (res.isTrue()) ? (in_bound[i].rkey) : (-1);
		mark_location[i] = (res.isTrue()) ? 1 : 0;
		__syncthreads();
	}

	if (index == 0) {
		mark_location[size] = 0;
	}
}

void GPUIJ::ExpressionFilterShared(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;
	int outer_cols = outer_chunk_.getColumnCount(), inner_cols = inner_chunk_.getColumnCount();

	block_x = SHARED_SIZE_/(sizeof(int64_t) * (outer_cols + inner_cols));

	//block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilterSharedDev<<<grid_size, block_size, block_x * sizeof(GNValue) * (outer_cols + inner_cols)>>>(outer_chunk_, inner_chunk_,
																												in_bound, out_bound,
																												mark_location, size,
																												end_expression_, post_expression_, where_expression_
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																												, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																												, val_stack, type_stack
#endif
																										);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}

void GPUIJ::ExpressionFilterShared(RESULT *in_bound, RESULT *out_bound, ulong *mark_location, int size, cudaStream_t stream)
{
	int partition_size = DEFAULT_PART_SIZE_;

	int block_x, grid_x;

	int outer_cols = outer_chunk_.getColumnCount(), inner_cols = inner_chunk_.getColumnCount();

	block_x = SHARED_SIZE_/(sizeof(int64_t) * (outer_cols + inner_cols));

	//block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size <= partition_size) ? (size - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilterSharedDev<<<grid_size, block_size, block_x * sizeof(GNValue) * (outer_cols + inner_cols), stream>>>(outer_chunk_, inner_chunk_,
																														in_bound, out_bound,
																														mark_location, size,
																														end_expression_, post_expression_, where_expression_
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																														, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																														, val_stack, type_stack
#endif
																														);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif

}



extern "C" __global__ void ExpressionFilterDev(GTable outer, GTable inner,
												RESULT *result, ulong *index_psum,
												ulong *exp_psum, uint result_size,
												GExpression end_dev, GExpression post_dev, GExpression where_dev,
												ResBound *res_bound, bool *prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
												,GNValue *stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
												,int64_t *val_stack,
												ValueType *type_stack
#endif
)
{

	int64_t *outer_dev = outer.getBlock().data, *inner_dev = inner.getBlock().data;
	int outer_cols = outer.getColumnCount(), inner_cols = inner.getColumnCount();
	int outer_rows = outer.getBlock().rows;
	GColumnInfo *outer_schema = outer.getSchema(), *inner_schema = inner.getSchema();

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = index; i < outer_rows; i += offset) {
		exp_psum[i] = 0;
		ulong writeloc = index_psum[index];
		int count = 0;
		int res_left = -1, res_right = -1;
		GNValue res = GNValue::getTrue();

		res_left = res_bound[i].left;
		res_right = res_bound[i].right;

		while (res_left >= 0 && res_left <= res_right && writeloc < result_size) {
#ifdef	TREE_EVAL_
#ifdef FUNC_CALL_
			res = (end_dev.getSize() > 1) ? end_dev.evaluate(1, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema) : res;
			res = (post_dev.getSize() > 1 && res.isTrue()) ? post_dev.evaluate(1, post_size, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema) : res;
#else
			res = (end_dev.getSize() > 1) ? end_dev.evaluate2(1, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema) : res;
			res = (post_dev.getSize() > 1 && res.isTrue()) ? post_dev.evaluate2(1, outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema) : res;
#endif

#elif	POST_EXP_


#ifdef 	FUNC_CALL_
			res = (end_dev.getSize() > 0) ? end_dev.evaluate(outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
			res = (post_dev.getSize() > 0 && res.isTrue()) ? post_dev.evaluate(outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema, stack + index, offset) : res;
#else
			res = (end_dev.getSize() > 0) ? end_dev.evaluate(outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
			res = (post_dev.getSize() > 0 && res.isTrue()) ? end_dev.evaluate(outer_dev + i * outer_cols, inner_dev + res_left * inner_cols, outer_schema, inner_schema, val_stack + index, type_stack + index, offset) : res;
#endif
#endif
			result_dev[writeloc].lkey = (res.isTrue()) ? i : (-1);
			result_dev[writeloc].rkey = (res.isTrue()) ? res_left : (-1);
			count += (res.isTrue()) ? 1 : 0;
			writeloc++;
			res_left++;
		}
		exp_psum[i] = count;
	}

	if (index == 0) {
		exp_psum[outer_part_size] = 0;
	}
}

void GPUIJ::ExpressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev)
{
	int outer_rows = outer_chunk_.block_list->rows;
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilterDev<<<grid_size, block_size>>>(outer_chunk_, inner_chunk_,
													result, index_psum,
													exp_psum,
													result_size,
													end_expression_, post_expression_, where_expression_,
													res_bound, prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
													, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
													, val_stack
													, type_stack
#endif
												);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void GPUIJ::ExpressionFilter(ulong *index_psum, ulong *exp_psum, RESULT *result, int result_size, ResBound *res_bound, bool *prejoin_res_dev, cudaStream_t stream)
{
	int outer_rows = outer_chunk_.getBlock().rows;
	int partition_size = DEFAULT_PART_SIZE_;
	int block_x, grid_x;

	block_x = (outer_rows < BLOCK_SIZE_X) ? outer_rows : BLOCK_SIZE_X;
	grid_x = (outer_rows < partition_size) ? (outer_rows - 1)/block_x + 1 : (partition_size - 1)/block_x + 1;

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	GNValue *stack;

	checkCudaErrors(cudaMalloc(&stack, sizeof(GNValue) * block_x * grid_x * MAX_STACK_SIZE));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	int64_t *val_stack;
	ValueType *type_stack;

	checkCudaErrors(cudaMalloc(&val_stack, sizeof(int64_t) * block_x * grid_x * MAX_STACK_SIZE));
	checkCudaErrors(cudaMalloc(&type_stack, sizeof(ValueType) * block_x * grid_x * MAX_STACK_SIZE));
#endif

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	ExpressionFilterDev<<<grid_size, block_size, 0, stream>>>(outer_chunk_, inner_chunk_,
																result, index_psum,
																exp_psum, result_size,
																end_expression_, post_expression_, where_expression_,
																res_bound, prejoin_res_dev
#if (defined(POST_EXP_) && defined(FUNC_CALL_))
																, stack
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
																, val_stack
																, type_stack
#endif
															);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));

#if (defined(POST_EXP_) && defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(stack));
#elif (defined(POST_EXP_) && !defined(FUNC_CALL_))
	checkCudaErrors(cudaFree(val_stack));
	checkCudaErrors(cudaFree(type_stack));
#endif
}

void GPUIJ::Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	ExclusiveScanAsyncWrapper(in, in_size, out_size, stream);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemsetAsync(location, 0, sizeof(ulong) * (*out_size), stream));

	GUtilities::MarkLocation(location, in, in_size, stream);

	GUtilities::InclusiveScan(location, *out_size, stream);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::ComputeOffset(in, location, local_offset, *out_size, stream);

	Decompose(in_bound, *out_bound, location, local_offset, *out_size, stream);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void GPUIJ::Rebalance3(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	GUtilities::ExclusiveScan(in, in_size, out_size);

	if (*out_size == 0) {
		return;
	}

	ulong *location;

	checkCudaErrors(cudaMalloc(&location, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaMemset(location, 0, sizeof(ulong) * (*out_size)));

	checkCudaErrors(cudaDeviceSynchronize());

	GUtilities::MarkLocation(location, in, in_size);

	GUtilities::InclusiveScan(location, *out_size);

	ulong *local_offset;

	checkCudaErrors(cudaMalloc(&local_offset, *out_size * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, *out_size * sizeof(RESULT)));

	GUtilities::ComputeOffset(in, location, local_offset, *out_size);

	Decompose(in_bound, *out_bound, location, local_offset, *out_size);

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(location));
}

void GPUIJ::Rebalance(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
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

	ExclusiveScanWrapper(mark, in_size + 1, &size_no_zeros);

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

	Decompose(tmp_bound, *out_bound, tmp_location, local_offset, sum);
	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));
}



void GPUIJ::Rebalance2(ulong *index_count, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
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

	checkCudaErrors(cudaMemsetAsync(tmp_location, 0, sizeof(ulong) * sum, stream));

	GUtilities::MarkTmpLocation(tmp_location, no_zeros, size_no_zeros, stream);

	GUtilities::InclusiveScan(tmp_location, sum, stream);

	checkCudaErrors(cudaMalloc(&local_offset, sum * sizeof(ulong)));
	checkCudaErrors(cudaMalloc(out_bound, sum * sizeof(RESULT)));

	GUtilities::ComputeOffset(no_zeros, tmp_location, local_offset, sum, stream);

	Decompose(tmp_bound, *out_bound, tmp_location, local_offset, sum, stream);
	checkCudaErrors(cudaStreamSynchronize(stream));
	*out_size = sum;

	checkCudaErrors(cudaFree(local_offset));
	checkCudaErrors(cudaFree(tmp_location));
	checkCudaErrors(cudaFree(no_zeros));
	checkCudaErrors(cudaFree(mark));
	checkCudaErrors(cudaFree(tmp_bound));
}

extern "C" __global__ void DecomposeDev2(ulong *in, ResBound *in_bound, RESULT *out_bound, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		if (in_bound[i].left >= 0 && in_bound[i].right >= 0 && in_bound[i].outer >= 0) {
			int write_location = in[i];

			for (int j = in_bound[i].left; j <= in_bound[i].right; j++) {
				out_bound[write_location].lkey = in_bound[i].outer;
				out_bound[write_location].rkey = j;
				write_location++;
			}
		}
	}
}

void GPUIJ::Rebalance2(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	GUtilities::ExclusiveScan(in, in_size, out_size);

	if (*out_size == 0)
		return;

	checkCudaErrors(cudaMalloc(out_bound, sizeof(RESULT) * (*out_size)));
	DecomposeDev2<<<grid_size, block_size>>>(in, in_bound, *out_bound, in_size - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GPUIJ::Rebalance(ulong *in, ResBound *in_bound, RESULT **out_bound, int in_size, ulong *out_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	GUtilities::ExclusiveScan(in, in_size, out_size, stream);

	if (*out_size == 0)
		return;

	checkCudaErrors(cudaMalloc(out_bound, sizeof(RESULT) * (*out_size)));
	DecomposeDev2<<<grid_size, block_size, 0, stream>>>(in, in_bound, *out_bound, in_size - 1);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

}
