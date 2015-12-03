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
							TreeExpression *end_expression,
							TreeExpression *post_expression,
							TreeExpression *initial_expression,
							TreeExpression *skipNullExpr,
							TreeExpression *prejoin_expression,
							TreeExpression *where_expression)
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

	/**** Expression data ****/
	end_expression_ = end_expression;
	post_expression_ = post_expression;
	initial_expression_ = initial_expression;
	skipNullExpr_ = skipNullExpr;
	prejoin_expression_ = prejoin_expression;
	where_expression_ = where_expression;
}

GPUIJ::~GPUIJ()
{
	if (join_result_ != NULL) {
		free(join_result_);
	}
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
	} else if ((vd = getenv("HOME")) != NULL) {
		snprintf(path, 256, "%s/voltdb", vd);
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
		fprintf(stderr, "cuModuleLoad(join) failed: res = %lu\n", (unsigned long)res);
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
	CUdeviceptr outer_dev, inner_dev, jresult_dev, count_dev, presum_dev, cond_dev;
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

	/*** Loop over outer tuples and inner tuples to copy table data to GPU buffer **/
	for (uint outer_idx = 0; outer_idx < outer_size_; outer_idx += part_size) {
		for (uint inner_idx = 0; inner_idx < inner_size_; inner_idx += part_size) {
			//Size of outer small table
			uint outer_part = (outer_idx + part_size < outer_size_) ? part_size : (outer_size_ - outer_idx);
			//Size of inner small table
			uint inner_part = (inner_idx + part_size < inner_size_) ? part_size : (inner_size_ - inner_idx);

			block_x = (outer_part < BLOCK_SIZE_X) ? outer_part : BLOCK_SIZE_X;
			block_y = (inner_part < BLOCK_SIZE_Y) ? inner_part : BLOCK_SIZE_Y;
			grid_x = divUtility(outer_part, block_x);
			grid_y = divUtility(inner_part, block_y);
			gpu_size = grid_x * grid_y * block_x * block_y + 1;

			/**** Copy IndexData to GPU memory ****/
			res = cuMemcpyHtoD(outer_dev, outer_table_ + outer_idx, outer_part * sizeof(IndexData));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemcpyHtoD(outer_dev, outer_table_ + %u) failed: res = %lu\n", outer_idx, (unsigned long)res);
				return false;
			}

			res = cuMemcpyHtoD(inner_dev, inner_table_ + inner_idx, inner_part * sizeof(IndexData));
			if (res != CUDA_SUCCESS) {
				fprintf(stderr, "cuMemcpyHtoD(inner_dev, inner_table_ + %u) failed: res = %lu\n", inner_idx, (unsigned long)res);
				return false;
			}

			/*** <TODO> Copy data of post_expression, initial_expression, skipNullExpr, prejoin_expression, where_expression ***/

			/*** Calculate the size of output result***/
			void *count_args[] = {
					(void *)&outer_dev,
					(void *)&inner_dev,
					(void *)end_expression_,
					(void *)&count_dev,
					(void *)&outer_part,
					(void *)&inner_part
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
					(void *)end_expression_,
					(void *)&count_dev,
					(void *)&outer_part,
					(void *)&inner_part
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
