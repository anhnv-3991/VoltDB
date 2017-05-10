#include "utilities.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/storage/gtable.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <inttypes.h>
#include <thrust/system/cuda/execution_policy.h>

namespace voltdb {

extern "C" __global__ void markNonZeros(ulong *input, int size, ulong *mark)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i <= size; i += blockDim.x * gridDim.x) {
		mark[i] = (i < size && input[i] != 0) ? 1 : 0;
	}
}

void GUtilities::MarkNonZeros(ulong *input, int size, ulong *output)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	markNonZeros<<<grid_size, block_size>>>(input, size, output);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::MarkNonZeros(ulong *input, int size, ulong *output, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 block_size(block_x, 1, 1);
	dim3 grid_size(grid_x, 1, 1);

	markNonZeros<<<grid_size, block_size, 0, stream>>>(input, size, output);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

extern "C" __global__ void removeZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		if (input[i] != 0) {
			output[output_location[i]] = input[i];
			out_bound[output_location[i]] = in_bound[i];
		}
		__syncthreads();
	}
}

void GUtilities::RemoveZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);


	removeZeros<<<grid_size, block_size>>>(input, in_bound, output, out_bound, output_location, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::RemoveZeros(ulong *input, ResBound *in_bound, ulong *output, ResBound *out_bound, ulong *output_location, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);


	removeZeros<<<grid_size, block_size, 0, stream>>>(input, in_bound, output, out_bound, output_location, size);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

extern "C" __global__ void markTmpLocation(ulong *tmp_location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location[input[i]] = (i != 0) ? 1 : 0;
	}
}

void GUtilities::MarkTmpLocation(ulong *tmp_location, ulong *input, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1) / block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	markTmpLocation<<<grid_size, block_size>>>(tmp_location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::MarkTmpLocation(ulong *tmp_location, ulong *input, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1) / block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	markTmpLocation<<<grid_size, block_size, 0, stream>>>(tmp_location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

extern "C" __global__ void markLocation1(ulong *location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index + 1; i < size - 1; i += blockDim.x * gridDim.x) {
		if (input[i] != input[i + 1])
		location[input[i]] = i;
	}

}

extern "C" __global__ void markLocation2(ulong *location, ulong *input, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = index + 1; i < size - 1; i += blockDim.x * gridDim.x) {
		if (input[i] != input[i - 1])
			location[input[i]] -= (i - 1);
	}
}

void GUtilities::MarkLocation(ulong *location, ulong *input, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	markLocation1<<<grid_size, block_size>>>(location, input, size);
	markLocation2<<<grid_size, block_size>>>(location, input, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::MarkLocation(ulong *location, ulong *input, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	markLocation1<<<grid_size, block_size, 0, stream>>>(location, input, size);
	markLocation2<<<grid_size, block_size, 0, stream>>>(location, input, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

extern "C" __global__ void computeOffset(ulong *input1, ulong *input2, ulong *out, int size)
{
	ulong index = threadIdx.x + blockIdx.x * blockDim.x;

	for (ulong i = index; i < size; i += blockDim.x * gridDim.x) {
		out[i] = i - input1[input2[i]];
	}
}

void GUtilities::ComputeOffset(ulong *input1, ulong *input2, ulong *out, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	computeOffset<<<grid_size, block_size>>>(input1, input2, out, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::ComputeOffset(ulong *input1, ulong *input2, ulong *out, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	computeOffset<<<grid_size, block_size, 0, stream>>>(input1, input2, out, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

void GUtilities::ExclusiveScan(ulong *input, int ele_num, ulong *sum)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());

	*sum = *(dev_ptr + ele_num - 1);
}

void GUtilities::ExclusiveScan(ulong *input, int ele_num, ulong *sum, cudaStream_t stream)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::exclusive_scan(thrust::system::cuda::par.on(stream), dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaStreamSynchronize(stream));

	*sum = *(dev_ptr + ele_num - 1);
}

void GUtilities::InclusiveScan(ulong *input, int ele_num)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::inclusive_scan(dev_ptr, dev_ptr + ele_num, dev_ptr);
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::InclusiveScan(ulong *input, int ele_num, cudaStream_t stream)
{
	thrust::device_ptr<ulong> dev_ptr(input);

	thrust::inclusive_scan(thrust::system::cuda::par.on(stream), dev_ptr, dev_ptr + ele_num, dev_ptr);
	//checkCudaErrors(cudaStreamSynchronize(stream));
}

unsigned long GUtilities::timeDiff(struct timeval start, struct timeval end)
{
	return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

extern "C" __global__ void removeEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < in_size) {
		ulong write_loc = out_location[index];
		ulong read_loc = in_location[index];
		ulong num = in_location[index + 1] - in_location[index];
		int lkey, rkey;
		ulong i = 0;

		while (i < num) {
			lkey = in_bound[read_loc + i].lkey;
			rkey = in_bound[read_loc + i].rkey;

			if (lkey != -1 && rkey != -1) {
				out_bound[write_loc].lkey = lkey;
				out_bound[write_loc].rkey = rkey;
				write_loc++;
			}
			i++;
		}
	}
}

void GUtilities::RemoveEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	removeEmptyResult<<<grid_size, block_size>>>(out_bound, in_bound, in_location, out_location, in_size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::RemoveEmptyResult(RESULT *out_bound, RESULT *in_bound, ulong *in_location, ulong *out_location, uint in_size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (in_size < BLOCK_SIZE_X) ? in_size : BLOCK_SIZE_X;
	grid_x = (in_size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	removeEmptyResult<<<grid_size, block_size, 0, stream>>>(out_bound, in_bound, in_location, out_location, in_size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaStreamSynchronize(stream));
}

extern "C" __global__ void removeEmptyResult2(RESULT *out, RESULT *in, ulong *location, int size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int tmp_location;

	for (int i = index; i < size; i += blockDim.x * gridDim.x) {
		tmp_location = location[i];
		if (in[i].lkey != -1 && in[i].rkey != -1) {
			out[tmp_location].lkey = in[i].lkey;
			out[tmp_location].rkey = in[i].rkey;
		}
	}
}

void GUtilities::RemoveEmptyResult(RESULT *out, RESULT *in, ulong *location, int size)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	removeEmptyResult2<<<grid_size, block_size>>>(out, in, location, size);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void GUtilities::RemoveEmptyResult(RESULT *out, RESULT *in, ulong *location, int size, cudaStream_t stream)
{
	int block_x, grid_x;

	block_x = (size < BLOCK_SIZE_X) ? size : BLOCK_SIZE_X;
	grid_x = (size - 1)/block_x + 1;

	dim3 grid_size(grid_x, 1, 1);
	dim3 block_size(block_x, 1, 1);

	removeEmptyResult2<<<grid_size, block_size, 0, stream>>>(out, in, location, size);

	checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaStreamSynchronize(stream));
}





}
