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

#ifndef SCAN_COMMON_H
#define SCAN_COMMON_H

#include <stdlib.h>
#include <cuda.h>

namespace voltdb{

////////////////////////////////////////////////////////////////////////////////
// Shortcut typename
////////////////////////////////////////////////////////////////////////////////
#define SUCCESS 1
#define FALSE 0
    
////////////////////////////////////////////////////////////////////////////////
// Implementation limits
////////////////////////////////////////////////////////////////////////////////
    extern "C" const uint MAX_BATCH_ELEMENTS;
    extern "C" const uint MIN_SHORT_ARRAY_SIZE;
    extern "C" const uint MAX_SHORT_ARRAY_SIZE;
    extern "C" const uint MIN_LARGE_ARRAY_SIZE;
    extern "C" const uint MAX_LARGE_ARRAY_SIZE;
    extern "C" const uint MIN_LL_SIZE;
    extern "C" const uint MAX_LL_SIZE;




    template<typename T,typename S>
        class GPUSCAN{
        
    public:
////////////////////////////////////////////////////////////////////////////////
// constructor
////////////////////////////////////////////////////////////////////////////////
        GPUSCAN(){};
        

////////////////////////////////////////////////////////////////////////////////
// CUDA scan
////////////////////////////////////////////////////////////////////////////////
        void initScan();

        void closeScan();

        size_t scanExclusiveMIN(
            T *d_Dst,
            T *d_Src,
            uint arrayLength
            );

        size_t scanExclusiveShort(
            T *d_Dst,
            T *d_Src,
            uint arrayLength
            );

        size_t scanExclusiveLarge(
            T *d_Dst,
            T *d_Src,
            uint arrayLength
            );
    
        size_t scanExclusiveLL(
            T *d_Dst,
            T *d_Src,
            uint arrayLength
            );

        void getValue_gpu(
            T *d_Dst,
            T *d_Src,
            uint loc
            );
        

        uint presum(
            CUdeviceptr *d_Input,
            uint arrayLength
            );

        uint getValue(
            CUdeviceptr d_Input,
            uint loc,
            T *res
            );

    private:

        uint iDivUp(uint dividend, uint divisor)
        {
            return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
        }

//Internal exclusive scan buffer
        T *d_Buf;
        T *e_Buf;
        
    };


    template<typename T,typename S>
        uint GPUSCAN<T,S>::presum(CUdeviceptr *d_Input, uint arrayLength)
    {
        
        uint N = 0;
        CUdeviceptr d_Output;

        //printf("Initializing CUDA-C scan...\n\n");
        initScan();
        //size_t szWorkgroup;

        if(arrayLength <= MIN_SHORT_ARRAY_SIZE && arrayLength > 0){

            N = 5;

            checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(T)));
      
            checkCudaErrors(cudaDeviceSynchronize());
      
            scanExclusiveMIN((T *)d_Output, (T *)(*d_Input), N);
      
            checkCudaErrors(cudaDeviceSynchronize());
      

        }else if(arrayLength <= MAX_SHORT_ARRAY_SIZE && arrayLength > MIN_SHORT_ARRAY_SIZE){    
            for(uint i = 4; i<=MAX_SHORT_ARRAY_SIZE ; i<<=1){
                if(arrayLength <= i){
                    N = i;
                    break;
                }
            }

            //printf("N = %d\n",N);

            checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(T)));

            checkCudaErrors(cudaDeviceSynchronize());

            scanExclusiveShort((T *)d_Output, (T *)(*d_Input), N);

            checkCudaErrors(cudaDeviceSynchronize());

        }else if(arrayLength <= MAX_LARGE_ARRAY_SIZE){

            N = MAX_SHORT_ARRAY_SIZE * iDivUp(arrayLength,MAX_SHORT_ARRAY_SIZE);

            //printf("N = %d\n",N);

            checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(T)));
      
            checkCudaErrors(cudaDeviceSynchronize());

            scanExclusiveLarge((T *)d_Output, (T *)(*d_Input), N);
      
            checkCudaErrors(cudaDeviceSynchronize());


        }else if(arrayLength <= MAX_LL_SIZE){

            N = MAX_LARGE_ARRAY_SIZE * iDivUp(arrayLength,MAX_LARGE_ARRAY_SIZE);

            //printf("N = %d\n",N);

            checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(T)));
            //checkCudaErrors(cudaMemset((void *)d_Output,0,N*sizeof(uint)));
        
            checkCudaErrors(cudaDeviceSynchronize());

            scanExclusiveLL((T *)d_Output, (T *)(*d_Input), N);
        
            checkCudaErrors(cudaDeviceSynchronize());

        }else{
            //cuMemFree(d_Output);
            closeScan();

            return FALSE;      
        }

        closeScan();

        cuMemFree(*d_Input);
        *d_Input = d_Output;

        return SUCCESS;

    }

    template<typename T,typename S>
        uint GPUSCAN<T,S>::getValue(CUdeviceptr d_Input , uint loc , T *res){

        CUdeviceptr d_Output;

        checkCudaErrors(cudaMalloc((void **)&d_Output, sizeof(T)));
        checkCudaErrors(cudaDeviceSynchronize());

        getValue_gpu((T *)d_Output, (T *)d_Input, loc);
        //szWorkgroup = scanExclusiveLarge((uint *)d_Output, (uint *)d_Input, pnum, N);
        checkCudaErrors(cudaDeviceSynchronize());

        if(cuMemcpyDtoH(res,d_Output,sizeof(T)) != CUDA_SUCCESS){
            printf("cuMemcpyDtoH(d_Output) error.\n");
            exit(1);
        }

        // pass or fail (cumulative... all tests in the loop)

        return SUCCESS;


    }



    
}

#endif


/*
////////////////////////////////////////////////////////////////////////////////
// CUDA scan
////////////////////////////////////////////////////////////////////////////////
extern "C" void initScan(void);
extern "C" void closeScan(void);
    
extern "C" size_t scanExclusiveMIN(
T *d_Dst,
T *d_Src,
uint arrayLength
);

extern "C" size_t scanExclusiveShort(
T *d_Dst,
T *d_Src,
uint arrayLength
);

extern "C" size_t scanExclusiveLarge(
T *d_Dst,
T *d_Src,
uint arrayLength
);

extern "C" size_t scanExclusiveLL(
T *d_Dst,
T *d_Src,
uint arrayLength
);

extern "C" void getValue_gpu(
T *d_Dst,
T *d_Src,
uint loc
);

extern "C" uint presum(
CUdeviceptr *d_Input,
uint arrayLength
);

extern "C" uint getValue(
CUdeviceptr d_Input,
uint loc,
T *res
);

*/
