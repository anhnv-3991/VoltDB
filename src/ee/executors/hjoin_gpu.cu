#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

extern "C" {

__global__
void count(
          COLUMNDATA *lt,
          COLUMNDATA *prt,
          ulong *count,
          GComparisonExpression ex,
          int *r_p,
          int p_num,
          int left
          ) 
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;  


  if(x < left){

    //caution : success for some reason. Not unuse if 
    GNValue tlgnv;
    if(x == left-1){
      tlgnv = lt[x].gn;
    }else{
      tlgnv = lt[x].gn;
    }

    uint temp = 0;
    int idx = tlgnv.getHashValue( 0 , p_num);
    int temp2 = r_p[idx+1];


    for(int y=r_p[idx] ; y<temp2 ; y++){
      if(ex.eval(tlgnv,prt[y].gn)){
        temp++;
      }
    }

    count[x] = (ulong)temp;

  }


  if(x == left-1){
    count[x+1] = 0;
  }

}


__global__ 
void join(
          COLUMNDATA *lt,
          COLUMNDATA *prt,
          RESULT *jt,
          GComparisonExpression ex,
          int *r_p,
          ulong *count,
          int p_num,
          int left
          ) 
{


  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if(x < left){

    ulong writeloc = count[x];

    GNValue tlgnv;
    if(x == left-1){
      tlgnv = lt[x].gn;
    }else{
      tlgnv = lt[x].gn;
    }

    int idx = tlgnv.getHashValue( 0 , p_num);
    int temp2 = r_p[idx+1];

    for(int y=r_p[idx] ; y<temp2 ; y ++){
      if(ex.eval(tlgnv,prt[y].gn)){
        jt[writeloc].lkey = lt[x].num;
        jt[writeloc].rkey = prt[y].num;
        writeloc++;
      }
    }

  }

}    

}
