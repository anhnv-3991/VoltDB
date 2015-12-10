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


using namespace voltdb;

GPUIJ::GPUIJ()
{
		outer_table_ = inner_table_ = NULL;
		outer_size_ = inner_size_ = 0;
		join_result_ = NULL;
		result_size_ = 0;
		end_size_ = 0;
		post_size_ = 0;
		initial_size_ = 0;
		skipNull_size_ = 0;
		prejoin_size_ = 0;
		where_size_ = 0;

		end_expression_ = NULL;
		post_expression_ = NULL;
		initial_expression_ = NULL;
		skipNullExpr_ = NULL;
		prejoin_expression_ = NULL;
		where_expression_ = NULL;
}

GPUIJ::GPUIJ(IndexData *outer_table,
				IndexData *inner_table,
				int outer_size,
				int inner_size,
				std::vector<int> search_idx,
				std::vector<int> indices,
				TreeExpression end_expression,
				TreeExpression post_expression,
				TreeExpression initial_expression,
				TreeExpression skipNullExpr,
				TreeExpression prejoin_expression,
				TreeExpression where_expression)
{
	assert(outer_size >= 0 && inner_size >= 0);
	assert(outer_table != NULL && inner_table != NULL);

	/**** Table data *********/
	outer_table_ = outer_table;
	inner_table_ = inner_table;
	outer_size_ = outer_size;
	inner_size_ = inner_size;
	join_result_ = NULL;
	result_size_ = 0;
	end_size_ = end_expression.getSize();
	post_size_ = post_expression.getSize();
	initial_size_ = initial_expression.getSize();
	skipNull_size_ = skipNullExpr.getSize();
	prejoin_size_ = prejoin_expression.getSize();
	where_size_ = where_expression.getSize();
	search_keys_size_ = search_idx.size();
	indices_size_ = indices.size();


	bool ret = true;

	search_keys_ = (int *)malloc(sizeof(int) * search_keys_size_);
	assert(search_keys_ != NULL);
	for (int i = 0; i < search_keys_size_; i++) {
		search_keys_[i] = search_idx[i];
	}

	indices_ = (int *)malloc(sizeof(int) * indices_size_);
	assert(indices_ != NULL);
	for (int i = 0; i < indices_size_; i++) {
		indices_[i] = indices[i];
	}

	/**** Expression data ****/

	ret = getTreeNodes(&end_expression_, end_expression);
	assert(ret == true);
	//std::cout << "End of constructing end expression" << std::endl;

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
	freeArrays<int>(search_keys_);
	freeArrays<int>(indices_);
	freeArrays<GTreeNode>(end_expression_);
	freeArrays<GTreeNode>(post_expression_);
	freeArrays<GTreeNode>(initial_expression_);
	freeArrays<GTreeNode>(skipNullExpr_);
	freeArrays<GTreeNode>(where_expression_);
}

bool GPUIJ::join(){
	CUresult res;
	CUdevice dev;
	CUcontext ctx;
	CUfunction func, c_func;
	CUmodule module, c_module;
	char fname[256];
	char *vd;
	char path[256];

	/****************** Initialize GPU ********************/
	if ((vd = getenv("VOLT_HOME")) != NULL) {
		snprintf(path, 256, "%s/voltdb", vd);
		std::cout << "PATH = " << path << std::endl;
	} else if ((vd = getenv("HOME")) != NULL) {
		snprintf(path, 256, "%s/voltdb", vd);
		std::cout << "PATH = " << path << std::endl;
	} else
		return false;

	res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuInit failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuDeviceGet failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuCtxCreate(&ctx, 0, dev);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuCtxCreate failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	sprintf(fname, "%s/index_join_gpu.cubin", path);
	res = cuModuleLoad(&module, fname);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuModuleLoad(join) failed: res = %lu\n file name = %s\n", (unsigned long)res, fname);
		return false;
	}

	res = cuModuleGetFunction(&func, module, "join");
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuModuleGetFunction(join) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuModuleGetFunction(&c_func, module, "count");
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuModuleGetFunction(count) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	/******** Calculate size of blocks, grids, and GPU buffers *********/
	uint gpu_size, part_size;
	ulong jr_size;
	CUdeviceptr outer_dev, inner_dev, jresult_dev, count_dev, presum_dev, end_ex_dev, post_ex_dev, search_keys_dev, indices_dev;
	uint block_x, block_y, grid_x, grid_y;

	part_size = getPartitionSize();
	block_x = BLOCK_SIZE_X;
	block_y = BLOCK_SIZE_Y;
	grid_x = divUtility(part_size, block_x);
	grid_y = divUtility(part_size, block_y);

	gpu_size = grid_x * grid_y * block_x * block_y + 1;
	if (gpu_size > MAX_LARGE_ARRAY_SIZE) {
		gpu_size = MAX_LARGE_ARRAY_SIZE * divUtility(gpu_size, MAX_LARGE_ARRAY_SIZE);
	} else if (gpu_size > MAX_SHORT_ARRAY_SIZE) {
		gpu_size = MAX_SHORT_ARRAY_SIZE * divUtility(gpu_size, MAX_SHORT_ARRAY_SIZE);
	} else {
		gpu_size = MAX_SHORT_ARRAY_SIZE;
	}

	/******** Allocate GPU buffer for table data and counting data *****/
	res = cuMemAlloc(&outer_dev, part_size * sizeof(IndexData));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(outer_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&inner_dev, part_size * sizeof(IndexData));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(inner_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&count_dev, gpu_size * sizeof(ulong));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(count_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	/******* Allocate GPU buffer for join condition *********/
	res = cuMemAlloc(&end_ex_dev, end_size_ * sizeof(GTreeNode));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(end_ex_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&post_ex_dev, post_size_ * sizeof(GTreeNode));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(post_ex_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemcpyHtoD(end_ex_dev, end_expression_, end_size_ * sizeof(GTreeNode));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemcpyHtoD(end_ex_dev, end_expression) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemcpyHtoD(post_ex_dev, post_expression_, post_size_ * sizeof(GTreeNode));
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemcpyHtoD(post_ex_dev, post_expression) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	/******* Allocate GPU buffer for search keys and index keys *****/
	res = cuMemAlloc(&search_keys_dev, sizeof(int) * search_keys_size_);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(search_keys_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemAlloc(&indices_dev, sizeof(int) * indices_size_);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemAlloc(indices_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_size_; outer_idx += part_size) {
		for (uint inner_idx = 0; inner_idx < inner_size_; inner_idx += part_size) {
			//Size of outer small table
			uint outer_part_size = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);
			//Size of inner small table
			uint inner_part_size = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			block_x = (outer_part_size < BLOCK_SIZE_X) ? outer_part_size : BLOCK_SIZE_X;
			block_y = (inner_part_size < BLOCK_SIZE_Y) ? inner_part_size : BLOCK_SIZE_Y;
			grid_x = divUtility(outer_part_size, block_x);
			grid_y = divUtility(inner_part_size, block_y);
			gpu_size = grid_x * grid_y * block_x * block_y + 1;

			/**** Copy IndexData to GPU memory ****/
			res = cuMemcpyHtoD(outer_dev, outer_table_ + outer_idx, outer_part_size * sizeof(IndexData));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemcpyHtoD(outer_dev, outer_table_ + %u) failed: res = %lu\n", outer_idx, (unsigned long)res);
				return false;
			}

			res = cuMemcpyHtoD(inner_dev, inner_table_ + inner_idx, inner_part_size * sizeof(IndexData));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemcpyHtoD(inner_dev, inner_table_ + %u) failed: res = %lu\n", inner_idx, (unsigned long)res);
				return false;
			}

			/*** <TODO> Copy data of post_expression, initial_expression, skipNullExpr, prejoin_expression, where_expression ***/

			/*** Calculate the size of output result***/
			void *count_args[] = {
					(void *)&outer_dev,
					(void *)&inner_dev,
					(void *)&count_dev,
					(void *)&outer_part_size,
					(void *)&inner_part_size,
					(void *)&end_ex_dev,
					(void *)&end_size_,
					(void *)&post_ex_dev,
					(void *)&post_size_,
					(void *)&search_keys_dev,
					(void *)&search_keys_size_,
					(void *)&indices_dev,
					(void *)&indices_size_
			};

			res = cuLaunchKernel(c_func,
									grid_x,
									grid_y,
									1,
									block_x,
									block_y,
									1,
									0,
									NULL,
									count_args,
									NULL);
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuLaunchKernel(c_func) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuCtxSynchronize();
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuCtxSynchronize() failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			if (!((new GPUSCAN<ulong, ulong4>)->presum(&count_dev, gpu_size))) {
				fprintf(stderr, "Prefix(&count_dev, gpu_size) sum error.\n");
				return false;
			}

			if (!((new GPUSCAN<ulong, ulong4>)->getValue(count_dev, gpu_size, &jr_size))) {
				fprintf(stderr, "getValue(count_dev, gpu_size, &jr_size) error");
				return false;
			}

			if (jr_size < 0) {
				return false;
			}

			if (jr_size > 64 * 1024 * 1024) {
				fprintf(stdout, "One time result size is over???\n");
				return true;
			}

			if (result_size_ > 1024 * 1024 * 1024) {
				fprintf(stdout, "Result size is over???\n");
				return true;
			}

			join_result_ = (RESULT *)realloc(join_result_, (result_size_ + jr_size) * sizeof(RESULT));
			res = cuMemAlloc(&jresult_dev, jr_size * sizeof(RESULT));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemAlloc(jresult_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			void *join_args[] = {
					(void *)&outer_dev,
					(void *)&inner_dev,
					(void *)&jresult_dev,
					(void *)&count_dev,
					(void *)&outer_part_size,
					(void *)&inner_part_size,
					(void *)&end_ex_dev,
					(void *)&end_size_,
					(void *)&post_ex_dev,
					(void *)&post_size_,
					(void *)&search_keys_dev,
					(void *)&search_keys_size_,
					(void *)&indices_dev,
					(void *)&indices_size_
			};

			res = cuLaunchKernel(func,
									grid_x,
									grid_y,
									1,
									block_x,
									block_y,
									1,
									0,
									NULL,
									join_args,
									NULL);
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuLaunchKernel(func) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuCtxSynchronize();
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuCtxSynchronize() failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			res = cuMemcpyDtoH(join_result_ + result_size_, jresult_dev, jr_size * sizeof(RESULT));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemcpyDtoH(join_result_[%u], jresult_dev) failed: res = %lu\n", result_size_, (unsigned long)res);
				return false;
			}

			res = cuMemFree(jresult_dev);
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemFree(jresult_dev) failed: res = %lu\n", (unsigned long)res);
				return false;
			}

			result_size_ += jr_size;
			jr_size = 0;
		}
	}

	/******** Free GPU memory, unload module, end session **************/
	res = cuMemFree(outer_dev);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemFree(outer_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(inner_dev);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemFree(inner_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuMemFree(count_dev);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuMemFree(count_dev) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuModuleUnload(module);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuModuleUnload(module) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

	res = cuCtxDestroy(ctx);
	if (res != CUDA_SUCCESS) {
		fprintf(stderr, "cuCtxDestroy(ctx) failed: res = %lu\n", (unsigned long)res);
		return false;
	}

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
	uint part_size = DEFAULT_PART_SIZE_;
	uint bigger_tuple_size = (outer_size_ > inner_size_) ? outer_size_ : inner_size_;

	for (uint i = 32768; i <= DEFAULT_PART_SIZE_; i = i * 2) {
		if (bigger_tuple_size < i) {
			part_size = i;
			break;
		}
	}

	return part_size;
}


uint GPUIJ::divUtility(uint dividend, uint divisor) const
{
	return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

bool GPUIJ::getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression)
{
	int tmp_size = tree_expression.getSize();
	if (tmp_size > 1) {
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
				setNValue(&tmp, outer_table_[i].gn[j]);
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
				setNValue(&tmp, inner_table_[i].gn[j]);
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

	std::cout << "Size of search_keys_array = " << search_keys_size_ << std::endl;
	if (search_keys_size_ != 0) {
		std::cout << "Content of search_keys" << std::endl;
		for (int i = 0; i < search_keys_size_; i++) {
			std::cout << "search_keys[" << i << "] = " << search_keys_[i] << std::endl;
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
	for (int index = 1; index < size; index++) {
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
