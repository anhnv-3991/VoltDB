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
#include "GPUNIJ.h"
#include "scan_common.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"


using namespace voltdb;

GPUNIJ::GPUNIJ(){

  jt = NULL;
  left_CD = NULL;
  right_CD = NULL;
  total = 0;

}


bool GPUNIJ::initGPU(){ 

  char fname[256];
  char *vd;
  char path[256];
  //char *path="/home/yabuta/voltdb/voltdb";//TODO : get voltdb/voltdb path
    
  if((vd = getenv("VOLT_HOME")) != NULL){
    snprintf(path,256,"%s/voltdb",vd);
  }else if((vd = getenv("HOME")) != NULL){
    snprintf(path,256,"%s/voltdb",vd);
  }else{
    return false;
  }

  /******************** GPU init here ************************************************/

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

  /*********************************************************************************/


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  
  sprintf(fname, "%s/join_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(join) failed res=%lu\n",(unsigned long)res);
    return false;
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(oijoin) failed\n");
    return false;
  }
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(oicount) failed\n");
    return false;
  }

  return true;
}


void GPUNIJ::finish(){

  if(jt!=NULL){
    free(jt);
  }
  if(left_CD!=NULL){
    free(left_CD);
  }
  if(right_CD!=NULL){
    free(right_CD);
  }
  //finish GPU   ****************************************************

  res = cuModuleUnload(module);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleUnload module failed: res = %lu\n", (unsigned long)res);
  }  

  /*
  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
  }
  */
  
}



void
GPUNIJ::printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

uint GPUNIJ::iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}



//HrightとHleftをそれぞれ比較する。GPUで並列化するforループもここにあるもので行う。
bool GPUNIJ::join()
{

  //int i, j;
  uint gpu_size;
  ulong jt_size;
  CUdeviceptr lt_dev, rt_dev, jt_dev,count_dev, pre_dev;
  CUdeviceptr ltn_dev, rtn_dev;
  unsigned int block_x, block_y, grid_x, grid_y;

  /************** block_x * block_y is decided by BLOCK_SIZE. **************/

  block_x = BLOCK_SIZE_X;
  block_y = BLOCK_SIZE_Y;
    grid_x = PART / block_x;
  if (PART % block_x != 0)
    grid_x++;
  grid_y = PART / block_y;
  if (PART % block_y != 0)
    grid_y++;
  block_y = 1;

  gpu_size = grid_x * grid_y * block_x * block_y+1;
  if(gpu_size>MAX_LARGE_ARRAY_SIZE){
    gpu_size = MAX_LARGE_ARRAY_SIZE * iDivUp(gpu_size,MAX_LARGE_ARRAY_SIZE);
  }else if(gpu_size > MAX_SHORT_ARRAY_SIZE){
    gpu_size = MAX_SHORT_ARRAY_SIZE * iDivUp(gpu_size,MAX_SHORT_ARRAY_SIZE);
  }else{
    gpu_size = MAX_SHORT_ARRAY_SIZE;
  }


  /********************************************************************************/

  res = cuMemAlloc(&lt_dev, PART * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    return false;
  }
  res = cuMemAlloc(&rt_dev, PART * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    return false;
  }
  res = cuMemAlloc(&count_dev, gpu_size * sizeof(ulong));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    return false;
  }

  /********************** upload lt , rt and count***********************/

  for(uint ll = 0; ll < left ; ll += PART){
    for(uint rr = 0; rr < right ; rr += PART){

      uint lls=PART,rrs=PART;
      if((ll+PART) >= left){
        lls = left - ll;
      }
      if((rr+PART) >= right){
        rrs = right - rr;
      }

      block_x = lls < BLOCK_SIZE_X ? lls : BLOCK_SIZE_X;
      block_y = rrs < BLOCK_SIZE_Y ? rrs : BLOCK_SIZE_Y;      
      grid_x = lls / block_x;
      if (lls % block_x != 0)
        grid_x++;

      //thread of axial y is fixed 1. 
      grid_y = rrs / block_y;
      if (rrs % block_y != 0)
        grid_y++;
      block_y = 1;


      printf("\nStarting...\nll = %d\trr = %d\tlls = %d\trrs = %d\n",ll,rr,lls,rrs);
      printf("grid_x = %d\tgrid_y = %d\tblock_x = %d\tblock_y = %d\n",grid_x,grid_y,block_x,block_y);
      gpu_size = grid_x * grid_y * block_x + 1;
      printf("gpu_size = %d\n",gpu_size);


      res = cuMemcpyHtoD(lt_dev, &(left_CD[ll]), lls * sizeof(COLUMNDATA));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (lt) failed: res = %lu\n", (unsigned long)res);//conv(res));
        return false;
      }
      res = cuMemcpyHtoD(rt_dev, &(right_CD[rr]), rrs * sizeof(COLUMNDATA));
      if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
        return false;
      }
      
      
      void *count_args[]={
        (void *)&lt_dev,
        (void *)&rt_dev,
        (void *)expression,
        (void *)&count_dev,
        (void *)&lls,
        (void *)&rrs
      };
      
      res = cuLaunchKernel(
                           c_function,    // CUfunction f
                           grid_x,        // gridDimX
                           grid_y,        // gridDimY
                           1,             // gridDimZ
                           block_x,       // blockDimX
                           1,       // blockDimY
                           1,             // blockDimZ
                           0,             // sharedMemBytes
                           NULL,          // hStream
                           count_args,   // keunelParams
                           NULL           // extra
                           );
      if(res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(count) failed: res = %lu\n", (unsigned long int)res);
        return false;
      }      
      
      res = cuCtxSynchronize();
      if(res != CUDA_SUCCESS) {
        printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
        return false;
      }  
        

      /**************************** prefix sum *************************************/
      if(!((new GPUSCAN<ulong,ulong4>)->presum(&count_dev,gpu_size))){
        printf("count scan error.\n");
        return false;
      }
      /********************************************************************/      

      if(!(new GPUSCAN<ulong,ulong4>)->getValue(count_dev,gpu_size,&jt_size)){
        printf("transport error.\n");
        return false;
      }


      /************************************************************************
      jt memory alloc and jt upload

      ************************************************************************/

      printf("jt_size %lu\n",jt_size);

      if(jt_size <0){
        return false;
      }else if(jt_size > 64*1024*1024){
        printf("one time result size is over.\n");
        return true;

      }else if(total > 1024*1024*1024){
        printf("result size is over.\n");
        return true;
      }else if(jt_size==0){
        total += jt_size;
        jt_size = 0;
      }else{
        jt = (RESULT *)realloc(jt,(total+jt_size)*sizeof(RESULT));
        res = cuMemAlloc(&jt_dev, jt_size*sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemAlloc (join) failed\n");
          return false;
        }      

        void *kernel_args[]={
          (void *)&lt_dev,
          (void *)&rt_dev,
          (void *)&jt_dev,
          (void *)expression,
          (void *)&count_dev,
          (void *)&lls,
          (void *)&rrs,    
        };

        res = cuLaunchKernel(
                             function,      // CUfunction f
                             grid_x,        // gridDimX
                             grid_y,        // gridDimY
                             1,             // gridDimZ
                             block_x,       // blockDimX
                             1,       // blockDimY
                             1,             // blockDimZ
                             0,             // sharedMemBytes
                             NULL,          // hStream
                             kernel_args,   // keunelParams
                             NULL           // extra
                             );
        if(res != CUDA_SUCCESS) {
          printf("cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
          return false;
        }  
        
        res = cuCtxSynchronize();
        if(res != CUDA_SUCCESS) {
          printf("cuCtxSynchronize() failed: res = %lu\n", (unsigned long int)res);
          return false;
        }  

          
        res = cuMemcpyDtoH(&(jt[total]), jt_dev, jt_size * sizeof(RESULT));
        if (res != CUDA_SUCCESS) {
          printf("cuMemcpyDtoH (jt) failed: res = %lu\n", (unsigned long)res);
          return false;
        }
        cuMemFree(jt_dev);
        total += jt_size;
//        printf("End...\n jt_size = %d\ttotal = %d\n",jt_size,total);
        jt_size = 0;
        
      }
    }
    

  }



  /***************************************************************/

  //free GPU memory***********************************************


  res = cuMemFree(lt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (lt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(rt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(count_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    return false;
  }

  return true;

}


