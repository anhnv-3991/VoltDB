/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <sys/time.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include "GPUTUPLE.h"
#include "scan_common.h"



////////////////////////////////////////////////////////////////////////////////
// Basic ccan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions

namespace voltdb{

//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 1024
#define LOOP_PERTHREAD 16
#define LOOP_PERTHREAD2 16

//extern "C" const uint 
////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
extern "C" const uint MAX_BATCH_ELEMENTS = THREADBLOCK_SIZE * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
extern "C" const uint MIN_LL_SIZE = 8 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
extern "C" const uint MAX_LL_SIZE = MAX_BATCH_ELEMENTS;//4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE * THREADBLOCK_SIZE;
}

using namespace voltdb;
    
template<typename TTT>
inline __device__ TTT scan1Inclusive(TTT idata, volatile TTT *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        TTT t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

template<typename TTT>
inline __device__ TTT scan1Exclusive(TTT idata, volatile TTT *s_Data, uint size)
{
  return scan1Inclusive<TTT>(idata, s_Data, size) - idata;
}

template<typename TTT,typename SSS>
inline __device__ SSS scan4Inclusive(SSS idata4, volatile TTT *s_Data, uint size)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    TTT oval = scan1Exclusive<TTT>(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as uint4
template<typename TTT,typename SSS>
inline __device__ SSS scan4Exclusive(SSS idata4, volatile TTT *s_Data, ulong size)
{
  SSS odata4 = scan4Inclusive<TTT,SSS>(idata4, s_Data, size);
  odata4.x -= idata4.x;
  odata4.y -= idata4.y;
  odata4.z -= idata4.z;
  odata4.w -= idata4.w;
  return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////

template<typename TTT>
__global__ void scanExclusiveSharedMIN(
                                       TTT *d_Dst,
                                       TTT *d_Src,
                                       uint size
)
{
  if(threadIdx.x==0&&blockIdx.x==0&&blockIdx.y==0){
    d_Dst[0] = 0;
    for(int i=1; i<size ; i++){
      d_Dst[i] = d_Src[i-1]+d_Dst[i-1];
    }
  }
}

template<typename TTT,typename SSS>
__global__ void scanExclusiveShared(
    SSS *d_Dst,
    SSS *d_Src,
    uint size
)
{
    __shared__ TTT s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    SSS idata4 = d_Src[pos];

    //Calculate exclusive scan
    SSS odata4 = scan4Exclusive<TTT,SSS>(idata4, s_Data, size);

    //Write back
    d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)

template<typename TTT>
__global__ void scanExclusiveShared2(
    TTT *d_Buf,
    TTT *d_Dst,
    TTT *d_Src,
    uint N,
    uint arrayLength
)
{
    __shared__ TTT s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    TTT idata = 0;

    if (pos < N)
        idata =
            d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] + d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];


    //Compute
    TTT odata = scan1Exclusive<TTT>(idata, s_Data, arrayLength);

    //Avoid out-of-bound access
    if (pos < N){
        d_Buf[pos] = odata;
    }

}

template<typename TTT>
__global__ void scanExclusiveShared3(
                                     TTT *e_Buf,
                                     TTT *d_Buf,
                                     TTT *d_Dst,
                                     TTT *d_Src,
                                     uint N,
                                     uint arrayLength
                                     )
{
  __shared__ TTT s_Data[2 * THREADBLOCK_SIZE];
  
  //Skip loads and stores for inactive threads of last threadblock (pos >= N)
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;
  
  //Load top elements
  //Convert results of bottom-level scan back to inclusive
  TTT idata = 0;
  
  if (pos < N)
    idata =
      d_Buf[THREADBLOCK_SIZE -1 + pos * THREADBLOCK_SIZE] + d_Dst[(4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE) * pos] + d_Src[(4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE) * pos];
  
  //Compute
  TTT odata = scan1Exclusive<TTT>(idata, s_Data, arrayLength);
  
  //Avoid out-of-bound access
  if (pos < N)
    {
      e_Buf[pos] = odata;
    }
}


//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
template<typename TTT,typename SSS>
__global__ void uniformUpdate(
    SSS *d_Data,
    TTT *d_Buffer
)
{
    __shared__ TTT buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
    {
        buf = d_Buffer[blockIdx.x];
    }

    __syncthreads();

    SSS data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

template<typename TTT,typename SSS>
__global__ void uniformUpdate2(
    SSS *d_Data,
    TTT *d_Buffer
)
{
    __shared__ TTT buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint temp = blockIdx.x/THREADBLOCK_SIZE;
    if (threadIdx.x == 0)
    {
        buf = d_Buffer[temp];
    }

    __syncthreads();

    SSS data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

template<typename TTT>
__global__ void getValue_kernel(
    TTT *d_Data,
    TTT *d_Src,
    uint loc

)
{

  d_Data[0] = d_Src[loc-1];

}

/*
template __global__ void scanExclusiveSharedMIN<uint,uint4>();
template __global__ void scanExclusiveSharedMIN<ulong,ulong4>();
template __global__ void scanExclusiveShared<uint,uint4>();
template __global__ void scanExclusiveShared<ulong,ulong4>();
template __global__ void scanExclusiveShared2<uint>();
template __global__ void scanExclusiveShared2<ulong>();
template __global__ void scanExclusiveShared3<uint>();
template __global__ void scanExclusiveShared3<ulong>();
template __global__ void uniformUpdate<uint,uint4>();
template __global__ void uniformUpdate<ulong,ulong4>();
template __global__ void uniformUpdate2<uint,uint4>();
template __global__ void uniformUpdate2<ulong,ulong4>();
template __global__ void getValue_kernel<uint>();
template __global__ void getValue_kernel<ulong>();

*/

template void GPUSCAN<uint,uint4>::initScan();
template void GPUSCAN<uint,uint4>::closeScan();
template size_t GPUSCAN<uint,uint4>::scanExclusiveMIN(uint *d_Dst,uint *d_Src,uint arrayLength);
template size_t GPUSCAN<uint,uint4>::scanExclusiveShort(uint *d_Dst,uint *d_Src,uint arrayLength);
template size_t GPUSCAN<uint,uint4>::scanExclusiveLarge(uint *d_Dst,uint *d_Src,uint arrayLength);
template size_t GPUSCAN<uint,uint4>::scanExclusiveLL(uint *d_Dst,uint *d_Src,uint arrayLength);
template void GPUSCAN<uint,uint4>::getValue_gpu(uint *d_Dst,uint *d_Src,uint loc);


template void GPUSCAN<ulong,ulong4>::initScan();
template void GPUSCAN<ulong,ulong4>::closeScan();
template size_t GPUSCAN<ulong,ulong4>::scanExclusiveMIN(ulong *d_Dst,ulong *d_Src,uint arrayLength);
template size_t GPUSCAN<ulong,ulong4>::scanExclusiveShort(ulong *d_Dst,ulong *d_Src,uint arrayLength);
template size_t GPUSCAN<ulong,ulong4>::scanExclusiveLarge(ulong *d_Dst,ulong *d_Src,uint arrayLength);
template size_t GPUSCAN<ulong,ulong4>::scanExclusiveLL(ulong *d_Dst,ulong *d_Src,uint arrayLength);
template void GPUSCAN<ulong,ulong4>::getValue_gpu(ulong *d_Dst,ulong *d_Src,uint loc);

    
template<typename T,typename S>
void GPUSCAN<T,S>::initScan(void)
{

  checkCudaErrors(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(T)));
  
  checkCudaErrors(cudaMalloc((void **)&e_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE)) * sizeof(T)));

}

template<typename T,typename S>
void GPUSCAN<T,S>::closeScan(void)
{
  checkCudaErrors(cudaFree(d_Buf));
  checkCudaErrors(cudaFree(e_Buf));
    
}

static uint factorRadix2(uint &log2L, uint L)
{
  if (!L)
    {
      log2L = 0;
      return 0;
    }
  else
    {
      for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);

      return L;
    }
}

/*
  static uint iDivUp(uint dividend, uint divisor)
  {
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
  }
*/

template<typename T,typename S>
size_t GPUSCAN<T,S>::scanExclusiveMIN(
                                      T *d_Dst,
                                      T *d_Src,
                                      uint arrayLength
                                      )
{

  //Check total batch size limit
  assert(arrayLength ==5);

  //Check all threadblocks to be fully packed with data
  //assert(arrayLength % (4 * THREADBLOCK_SIZE) == 0);

  scanExclusiveSharedMIN<T><<<1, 1>>>(
                                      d_Dst,
                                      d_Src,
                                      arrayLength
                                      );
  getLastCudaError("scanExclusiveShared() execution FAILED\n");

  return THREADBLOCK_SIZE;
}

template<typename T,typename S>
size_t GPUSCAN<T,S>::scanExclusiveShort(
                                        T *d_Dst,
                                        T *d_Src,
                                        uint arrayLength
                                        )
{
  //Check power-of-two factorization
  uint log2L;
  uint factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  //Check supported size range
  assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

  //Check total batch size limit
  assert(arrayLength <= MAX_BATCH_ELEMENTS);

  //Check all threadblocks to be fully packed with data
  //assert(arrayLength % (4 * THREADBLOCK_SIZE) == 0);

  const int blockCountShort = iDivUp(arrayLength , 4*THREADBLOCK_SIZE);

  scanExclusiveShared<T,S><<<blockCountShort, THREADBLOCK_SIZE>>>(
                                                                  (S *)d_Dst,
                                                                  (S *)d_Src,
                                                                  arrayLength
                                                                  );
  getLastCudaError("scanExclusiveShared() execution FAILED\n");

  return THREADBLOCK_SIZE;
}

template<typename T,typename S>
size_t GPUSCAN<T,S>::scanExclusiveLarge(
                                        T *d_Dst,
                                        T *d_Src,
                                        uint arrayLength
                                        )
{
  //Check power-of-two factorization
  /*
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);
  */
  assert(arrayLength%MAX_SHORT_ARRAY_SIZE == 0);

  //Check supported size range
  assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));

  //Check total batch size limit
  assert(arrayLength <= MAX_BATCH_ELEMENTS);

  scanExclusiveShared<T,S><<<arrayLength / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
                                                                                       (S *)d_Dst,
                                                                                       (S *)d_Src,
                                                                                       4 * THREADBLOCK_SIZE
                                                                                       );
  getLastCudaError("scanExclusiveShared() execution FAILED\n");

  //Not all threadblocks need to be packed with input data:
  //inactive threads of highest threadblock just don't do global reads and writes

  uint array_temp = THREADBLOCK_SIZE;
  for(uint i = 2; i<=THREADBLOCK_SIZE ; i <<= 1){
    if(i >= arrayLength/(4 * THREADBLOCK_SIZE)){
      array_temp = i;
      break;
    }
  }

  const uint blockCount2 = 1;//iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
  scanExclusiveShared2<T><<< blockCount2, THREADBLOCK_SIZE>>>(
                                                              (T *)d_Buf,
                                                              (T *)d_Dst,
                                                              (T *)d_Src,
                                                              arrayLength / (4 * THREADBLOCK_SIZE),
                                                              array_temp
                                                              );
  getLastCudaError("scanExclusiveShared2() execution FAILED\n");

  uniformUpdate<T,S><<<(arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
                                                                                   (S *)d_Dst,
                                                                                   (T *)d_Buf
                                                                                   );
  getLastCudaError("uniformUpdate() execution FAILED\n");

  return THREADBLOCK_SIZE;
}

template<typename T,typename S>
size_t GPUSCAN<T,S>::scanExclusiveLL(
                                     T *d_Dst,
                                     T *d_Src,
                                     uint arrayLength
                                     )
{
  //Check power-of-two factorization
  /*
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);
  */
  assert((arrayLength%MAX_LARGE_ARRAY_SIZE) == 0);

  //Check supported size range
  assert((arrayLength >= MIN_LL_SIZE) && (arrayLength <= MAX_LL_SIZE));

  //Check total batch size limit
  assert((arrayLength) <= MAX_BATCH_ELEMENTS);

  scanExclusiveShared<T,S><<<arrayLength / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
                                                                                       (S *)d_Dst,
                                                                                       (S *)d_Src,
                                                                                       4 * THREADBLOCK_SIZE
                                                                                       );
  getLastCudaError("scanExclusiveShared() execution FAILED\n");
  checkCudaErrors(cudaDeviceSynchronize());

  //Now ,prefix sum per THREADBLOCK_SIZE done


  //Not all threadblocks need to be packed with input data:
  //inactive threads of highest threadblock just don't do global reads and writes

  const uint blockCount2 = iDivUp (arrayLength / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
  scanExclusiveShared2<T><<< blockCount2, THREADBLOCK_SIZE>>>(
                                                              (T *)d_Buf,
                                                              (T *)d_Dst,
                                                              (T *)d_Src,
                                                              arrayLength / (4 * THREADBLOCK_SIZE),
                                                              THREADBLOCK_SIZE
                                                              );
  getLastCudaError("scanExclusiveShared2() execution FAILED\n");
  checkCudaErrors(cudaDeviceSynchronize());


  //prefix sum of last elements per THREADBLOCK_SIZE done
  //this prefix sum can caluculate under only THREADBLOCK_SIZE size.
  //so We need one more prefix sum for last elements.

  uint array_temp = THREADBLOCK_SIZE;
  for(uint i = 2; i<=THREADBLOCK_SIZE ; i <<= 1){
    if(i >= arrayLength/(4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE)){
      array_temp = i;
      break;
    }
  }

  const uint blockCount3 = 1;//(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE);
  scanExclusiveShared3<T><<< blockCount3, THREADBLOCK_SIZE>>>(
                                                              (T *)e_Buf,
                                                              (T *)d_Buf,
                                                              (T *)d_Dst,
                                                              (T *)d_Src,
                                                              arrayLength / (4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE),
                                                              array_temp
                                                              );
  getLastCudaError("scanExclusiveShared3() execution FAILED\n");
  checkCudaErrors(cudaDeviceSynchronize());


  //add d_Buf to each array of d_Dst
  uniformUpdate<T,S><<<arrayLength / (4 * THREADBLOCK_SIZE ), THREADBLOCK_SIZE>>>(
                                                                                  (S *)d_Dst,
                                                                                  (T *)d_Buf
                                                                                  );

  //add e_Buf to each array of d_Dst
  checkCudaErrors(cudaDeviceSynchronize());

  uniformUpdate2<T,S><<<arrayLength / (4 * THREADBLOCK_SIZE ), THREADBLOCK_SIZE>>>(
                                                                                   (S *)d_Dst,
                                                                                   (T *)e_Buf
                                                                                   );
  getLastCudaError("uniformUpdate() execution FAILED\n");

  checkCudaErrors(cudaDeviceSynchronize());
  return THREADBLOCK_SIZE;
}


//transport input data to output per diff
template<typename T,typename S>
void GPUSCAN<T,S>::getValue_gpu(
                                T *d_Dst,
                                T *d_Src,
                                uint loc
                                )
{

  //Check total batch size limit
  //assert((arrayLength) <= MAX_BATCH_ELEMENTS);

  const uint blockCount = 1;//iDivUp(arrayLength , LOOP_PERTHREAD2*THREADBLOCK_SIZE);
  getValue_kernel<T><<<blockCount, 1>>>(
                                        d_Dst,
                                        d_Src,
                                        loc
                                        );
  getLastCudaError("transport_gpu() execution FAILED\n");
  checkCudaErrors(cudaDeviceSynchronize());

}

