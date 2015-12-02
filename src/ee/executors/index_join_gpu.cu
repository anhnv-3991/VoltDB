#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <sys/time.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

/**
count() is counting match tuple.
And in CPU, caluculate starting position using scan.
finally join() store match tuple to result array .

*/



extern "C" {


__global__
void count(
          COLUMNDATA *oCD,
          COLUMNDATA *iCD,
          GComparisonExpression ex,
          ulong *count,
          int ltn,
          int rtn
          )

{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  __shared__ COLUMNDATA tiCD[BLOCK_SIZE_Y];

  /**
     TO DO : tiCD shoud be stored in parallel.but I have bug.
   */

  if(threadIdx.x==0){
    for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
      tiCD[j] = iCD[BLOCK_SIZE_Y*blockIdx.y + j];
    }
  }

  /*
  for(int i = threadIdx.x; i<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+i<rtn ; i+=blockDim.x){
    tiCD[i] = iCD[BLOCK_SIZE_Y*blockIdx.y + i];
    //    memcpy(&tiGTT[i*its],iGTT + (block_size_y*blockIdx.y+i)*its,its);
  }
  */

  __syncthreads();

  count[x+k] = 0;

  if(x<ltn){


    //A global memory read is very slow.So repeating values is stored register memory
    COLUMNDATA toCD=oCD[x];
    int rtn_g = rtn;
    int mcount = 0;
    for(uint y = 0; y<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+y<rtn_g;y++){
      if(ex.eval(toCD.gn,tiCD[y].gn)){
        mcount++;
      }
    }

    count[x+k] = mcount;
  }

  if(x+k == (blockDim.x*gridDim.x*gridDim.y-1)){
    count[x+k+1] = 0;
  }

}


__global__ void join(
          COLUMNDATA *oCD,
          COLUMNDATA *iCD,
          RESULT *p,
          GComparisonExpression ex,
          ulong *count,
          int ltn,
          int rtn
          )
{

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * gridDim.x * blockDim.x;

  __shared__ COLUMNDATA tiCD[BLOCK_SIZE_Y];

  if(threadIdx.x==0){
    for(uint j=0 ; j<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+j<rtn ; j++){
      tiCD[j] = iCD[BLOCK_SIZE_Y*blockIdx.y + j];
    }
  }

  /*
  for(int i = threadIdx.x; i<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+i<rtn ; i+=blockDim.x){
    tiCD[i] = iCD[BLOCK_SIZE_Y*blockIdx.y + i];
    //    memcpy(&tiGTT[i*its],iGTT + (block_size_y*blockIdx.y+i)*its,its);
  }
  */
  __syncthreads();


  if(x<ltn){

    COLUMNDATA toCD = oCD[x];
    int rtn_g = rtn;
    ulong writeloc = count[x+k];
    for(uint y = 0; y<BLOCK_SIZE_Y && BLOCK_SIZE_Y*blockIdx.y+y<rtn_g;y++){
      if(ex.eval(toCD.gn,tiCD[y].gn)){
        p[writeloc].lkey = oCD[x].num;
        p[writeloc].rkey = iCD[BLOCK_SIZE_Y*blockIdx.y+y].num;
        writeloc++;
      }
    }
  }
}

}
