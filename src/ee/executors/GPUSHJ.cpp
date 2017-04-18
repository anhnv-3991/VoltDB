#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
//#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "scan_common.h"
#include "GPUSHJ.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"


GPUSHJ::GPUSHJ(){

  jt = NULL;
  total = 0;

}

bool GPUSHJ::initGPU(){ 

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
  //GPU仕様のために
  /*
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
  */

  /*********************************************************************************/


  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */

  
  sprintf(fname, "%s/hjoin_gpu.cubin", path);
  res = cuModuleLoad(&module, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(join) failed res=%lu\n",(unsigned long)res);
    return false;
  }
  res = cuModuleGetFunction(&function, module, "join");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(join) failed\n");
    return false;
  }
  res = cuModuleGetFunction(&c_function, module, "count");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count) failed\n");
    return false;
  }

  sprintf(fname, "%s/partitioning.cubin", path);
  res = cuModuleLoad(&pmodule, fname);
  if (res != CUDA_SUCCESS) {
    printf("cuModuleLoad(partitioning) failed res=%lu\n",(unsigned long)res);
    return false;
  }
  res = cuModuleGetFunction(&pc_function, pmodule, "rcount_partitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(count_partitioning) failed\n");
    return false;
  }
  res = cuModuleGetFunction(&p_function, pmodule, "rpartitioning");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(partitioning) failed\n");
    return false;
  }
  res = cuModuleGetFunction(&sp_function, pmodule, "countPartition");
  if (res != CUDA_SUCCESS) {
    printf("cuModuleGetFunction(countPartition) failed\n");
    return false;
  }


  return true;
}


void GPUSHJ::finish(){

  free(jt);
  free(left_CD);
  free(right_CD);

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
GPUSHJ::printDiff(struct timeval begin, struct timeval end)
{
  long diff;
  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

uint GPUSHJ::iDivUp(uint dividend, uint divisor)
{
  return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


bool GPUSHJ::join(){

  //uint *count;
  ulong jt_size;
  CUresult res;
  CUdeviceptr lt_dev, rt_dev, jt_dev, bucket_dev, buckArray_dev ,idxcount_dev;
  CUdeviceptr prt_dev,rL_dev;
  CUdeviceptr ltn_dev, rtn_dev, jt_size_dev;
  CUdeviceptr c_dev;
  unsigned int block_x, grid_x;

  /********************************************************************
   *lt,rt,countのメモリを割り当てる。
   *
   */
  /* lt */
  res = cuMemAlloc(&lt_dev, left * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (lefttuple) failed\n");
    return false;
  }
  /* rt */
  res = cuMemAlloc(&rt_dev, right * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (righttuple) failed\n");
    return false;
  }
  /*count */
  res = cuMemAlloc(&c_dev, (left+1) * sizeof(ulong));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    return false;
  }
  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/

  res = cuMemcpyHtoD(lt_dev, left_CD, left * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (lt) failed: res = %lu\n", (unsigned long)res);//conv(res));
    return false;
  }
  res = cuMemcpyHtoD(rt_dev, right_CD, right * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (rt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }

  /***************************************************************************/

  int p_num = 0;
  int t_num;

  int pt = right*PART_STANDARD;
  //if(right%PART_STANDARD!=0) pt++;

  for(uint i=PARTITION ; i<=pow(PARTITION,4); i*=PARTITION){
    if(i<=pt&&pt<=i*2){
      p_num = i;
    }
  }

  if(p_num==0){
    double temp = right*PART_STANDARD;
    if(temp < 2){
      p_num = 1;
    }else if(floor(log2(temp))==ceil(log2(temp))){
      p_num = (int)temp;
    }else{
      p_num = pow(2,(int)log2(temp) + 1);
    }
  }

  t_num = right/RIGHT_PER_TH;
  if(left%RIGHT_PER_TH != 0){
    t_num++;
  }


  /*hash table create*/

  res = cuMemAlloc(&prt_dev, right * sizeof(COLUMNDATA));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (prt) failed\n");
    return false;
  }

  res = cuMemAlloc(&rL_dev, t_num * PARTITION * sizeof(ulong));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (rL) failed\n");
    return false;
  }
  checkCudaErrors(cudaMemset((void *)rL_dev,0,t_num*PARTITION*sizeof(ulong)));


  t_num = right/RIGHT_PER_TH;
  if(right%RIGHT_PER_TH != 0){
    t_num++;
  }

  printf("t_num=%d\tp_num=%d\n",t_num,p_num);

  int p_block_x = t_num < PART_C_NUM ? t_num : PART_C_NUM;
  int p_grid_x = t_num / p_block_x;
  if (t_num % p_block_x != 0)
    p_grid_x++;

  int p_n=0;
  CUdeviceptr hashtemp;

  for(uint loop=0 ; pow(PARTITION,loop)<p_num || (p_num==1&&loop==0) ; loop++){

    if(p_num<pow(PARTITION,loop+1)){
      p_n = p_num/pow(PARTITION,loop);
    }else{
      p_n = PARTITION;
    }

    printf("p_grid=%d\tp_block=%d\tp_n=%d\n",p_grid_x,p_block_x,p_n);

    void *count_rpartition_args[]={

      (void *)&rt_dev,
      (void *)&rL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&right,
      (void *)&loop

    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         pc_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         count_rpartition_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rcount hash) failed: res = %lu\n", (unsigned long int)res);
      return false;
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash count) failed: res = %lu\n", (unsigned long int)res);
      return false;
    }

    /**************************** prefix sum *************************************/

    if(!((new GPUSCAN<ulong,ulong4>)->presum(&rL_dev,t_num*p_n))){
      printf("lL presum error\n");
      return false;
    }
    /********************************************************************/
    void *rpartition_args[]={
      (void *)&rt_dev,
      (void *)&prt_dev,
      (void *)&rL_dev,
      (void *)&p_n,
      (void *)&t_num,
      (void *)&right,
      (void *)&loop
    };
    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         p_function,    // CUfunction f
                         p_grid_x,        // gridDimX
                         1,        // gridDimY
                         1,             // gridDimZ
                         p_block_x,       // blockDimX
                         1,       // blockDimY
                         1,             // blockDimZ
                         0,             // sharedMemBytes
                         NULL,          // hStream
                         rpartition_args,   // keunelParams
                         NULL           // extra
                         );
    if(res != CUDA_SUCCESS) {
      printf("cuLaunchKernel(rhash partition) failed: res = %lu\n", (unsigned long int)res);
      return false;
    }
    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(rhash partition) failed: res = %lu\n", (unsigned long int)res);
      return false;
    }

    printf("...loop finish\n");

    hashtemp = rt_dev;
    rt_dev = prt_dev;
    prt_dev = hashtemp;

  }

  p_block_x = 256;
  p_grid_x = right/p_block_x;
  if(right%p_block_x!=0){
    p_grid_x++;
  }

  CUdeviceptr rstartPos_dev;
  int rpos_size = MAX_LARGE_ARRAY_SIZE*iDivUp(p_num+1,MAX_LARGE_ARRAY_SIZE);

  res = cuMemAlloc(&rstartPos_dev, rpos_size * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (rstartPos) failed\n");
    return false;
  }
  checkCudaErrors(cudaMemset((void *)rstartPos_dev,0,rpos_size*sizeof(uint)));

  printf("rpos = %d %d %d\n",rpos_size,p_block_x,p_grid_x);

  void *rspartition_args[]={

    (void *)&rt_dev,
    (void *)&rstartPos_dev,
    (void *)&p_num,
    (void *)&right
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
  res = cuLaunchKernel(
                       sp_function,    // CUfunction f
                       p_grid_x,        // gridDimX
                       1,        // gridDimY
                       1,             // gridDimZ
                       p_block_x,       // blockDimX
                       1,       // blockDimY
                       1,             // blockDimZ
                       0,             // sharedMemBytes
                       NULL,          // hStream
                       rspartition_args,   // keunelParams
                       NULL           // extra
                       );
  if(res != CUDA_SUCCESS){
    printf("cuLaunchKernel(rhash count partition) failed: res = %lu\n", (unsigned long int)res);
    return false;
  }
  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(rhash count partition) failed: res = %lu\n", (unsigned long int)res);
    return false;
  }

  if(!((new GPUSCAN<uint,uint4>)->presum(&rstartPos_dev,p_num+1))){
    printf("rstartpos presum error\n");
    return false;
  }


  res = cuMemFree(prt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (prt) failed: res = %lu\n", (unsigned long)res);
    return false;
  }
  res = cuMemFree(rL_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rL) failed: res = %lu\n", (unsigned long)res);
    return false;
  }


  /*
    条件に合致するタプルの数をあらかじめ求めておく
    これによってダウンロードするタプルの数を減らせる
   */



  /******************************************************************
    count the number of match tuple
    
  *******************************************************************/


  block_x = left < BLOCK_SIZE_X ? left : BLOCK_SIZE_X;
  grid_x = left / block_x;
  if (left % block_x != 0)
    grid_x++;


  void *count_args[]={
    
    (void *)&lt_dev,
    (void *)&rt_dev,
    (void *)&c_dev,
    (void *)expression,
    (void *)&rstartPos_dev,
    (void *)&p_num,
    (void *)&left
      
  };

  //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う

  res = cuLaunchKernel(
                       c_function,    // CUfunction f
                       grid_x,        // gridDimX
                       1,        // gridDimY
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
    printf("count cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
    return false;
  }      

  res = cuCtxSynchronize();
  if(res != CUDA_SUCCESS) {
    printf("cuCtxSynchronize(count) failed: res = %lu\n", (unsigned long int)res);
    return false;
  }


  /***************************************************************************************/


  /**************************** prefix sum *************************************/

  if(!((new GPUSCAN<ulong,ulong4>)->presum(&c_dev,(uint)left+1))){
    printf("count scan error\n");
    return false;
  }

  /********************************************************************/


  /************************************************************************
   join

   jt memory alloc and jt upload
  ************************************************************************/

  if(!(new GPUSCAN<ulong,ulong4>)->getValue(c_dev,(uint)left+1,&jt_size)){
    printf("transport error.\n");
    return false;
  }

  //printf("jt_size = %d\n",jt_size);

  if(jt_size < 0){
    printf("join size is under 0.\n");
    return false;
  }else if(jt_size == 0){
    printf("no tuple is matched.\n");
    return true;
  }else if(jt_size > 256 * 1024 * 1024){
    printf("the number of match tuple is over limit\n");
    return false;
  }else{
  
    jt = (RESULT *)malloc(jt_size * sizeof(RESULT));
    res = cuMemAlloc(&jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemAlloc (join) failed\n");
      return false;
    }

    void *kernel_args[]={
      (void *)&lt_dev,
      (void *)&rt_dev,
      (void *)&jt_dev,
      (void *)expression,
      (void *)&rstartPos_dev,
      (void *)&c_dev,
      (void *)&p_num,
      (void *)&left    
    };


    //グリッド・ブロックの指定、変数の指定、カーネルの実行を行う
    res = cuLaunchKernel(
                         function,      // CUfunction f
                         grid_x,        // gridDimX
                         1,        // gridDimY
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
      printf("join cuLaunchKernel() failed: res = %lu\n", (unsigned long int)res);
      return false;
    }  



    res = cuCtxSynchronize();
    if(res != CUDA_SUCCESS) {
      printf("cuCtxSynchronize(join) failed: res = %lu\n", (unsigned long int)res);
      return false;
    }  

    res = cuMemcpyDtoH(jt, jt_dev, jt_size * sizeof(RESULT));
    if (res != CUDA_SUCCESS) {
      printf("cuMemcpyDtoH (p) failed: res = %lu\n", (unsigned long)res);
      return false;
    }

  }

  /***************************************************************
  free GPU memory
  ***************************************************************/

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
  res = cuMemFree(jt_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (jointuple) failed: res = %lu\n", (unsigned long)res);
    return false;
  }  
  res = cuMemFree(c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    return false;
  }

  res = cuMemFree(rstartPos_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (rstartPos) failed: res = %lu\n", (unsigned long)res);
    return false;
  }

  total = jt_size;
  
  return true;


}

