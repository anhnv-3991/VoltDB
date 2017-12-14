/*
count the number of match tuple in each partition and each thread

*/

#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"

using namespace voltdb;

extern "C" {

__global__
void rcount_partitioning(
          COLUMNDATA *t,
          ulong *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;

  int x = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int part[SHARED_MAX];

  for(uint i=threadIdx.x; i<SHARED_MAX ; i+=blockDim.x){
    part[i] = 0;
  }
  __syncthreads();

  //int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;

  int DEF = blockIdx.x * blockDim.x * RIGHT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;
  // Matching phase
  int hash = 0;
  COLUMNDATA tt;

  if(x < t_n){

    for(uint i=0; i<RIGHT_PER_TH&&(DEF+threadIdx.x*RIGHT_PER_TH+i)<rows_n; i++){
      //caution : success for some reason. Not t[hoge].gn.getHashValue
      tt = t[DEF+threadIdx.x*RIGHT_PER_TH+i];
      //hash = tt.gn.getHashValue( loop*RADIX , p_n);
      if(hash == -1) return;
      part[hash*Dim + threadIdx.x]++;
      
    }
    for(uint j=0 ; j*Dim+threadIdx.x<p_n*Dim ; j++){
      L[t_n*j + blockIdx.x*blockDim.x + threadIdx.x] = (ulong)part[j*Dim+threadIdx.x];
    }
  }
}


__global__
void rpartitioning(
          COLUMNDATA *t,
          COLUMNDATA *pt,
          ulong *L,
          int p_num,
          int t_num,
          int rows_num,
          int loop
          ) 

{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int rows_n = rows_num;
  int p_n = p_num;
  int t_n = t_num;


  //int PER_TH = (table_type==LEFT) ? LEFT_PER_TH:RIGHT_PER_TH;
  int DEF = blockIdx.x * blockDim.x * RIGHT_PER_TH;
  int Dim = (gridDim.x-1 == blockIdx.x) ? (t_n - blockIdx.x*blockDim.x):blockDim.x;

  __shared__ uint part[SHARED_MAX];
  for(uint j=0 ; j*Dim+threadIdx.x<p_n*Dim ; j++){
    part[j*Dim+threadIdx.x]=(uint)L[t_n*j+blockIdx.x*blockDim.x+threadIdx.x];
  }
  
  __syncthreads();

  // Matching phase
  int hash = 0;
  int temp = 0;
  COLUMNDATA tt;

  if(x < t_n){

    for(uint i=0; i<RIGHT_PER_TH&&(DEF+threadIdx.x*RIGHT_PER_TH+i)<rows_n; i++){
      //caution : success for some reason
      tt = t[DEF+threadIdx.x*RIGHT_PER_TH+i];
      //hash = tt.gn.getHashValue( loop*RADIX , p_n);
      temp = part[hash*Dim + threadIdx.x]++;
      pt[temp] = tt;
    } 
  }

}

__global__
void countPartition(
          COLUMNDATA *t,
          uint *startpos,
          int p_num,
          int rows_num
          ) 
{

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  if(x<rows_num){

    //caution : success for some reason. Not t[hoge].gn.getHashValue
    GNValue tt;
    tt = t[x].gn;
    int p = 0;
   // p=tt.getHashValue( 0 , p_num);
    atomicAdd(&(startpos[p]),1);
  }

  if(x==rows_num-1){
    startpos[p_num+1]=0;
  }


}



}
