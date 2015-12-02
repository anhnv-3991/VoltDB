/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUNIJ_H
#define GPUNIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

class GPUNIJ{

public:

    GPUNIJ();

    bool initGPU();
    void finish();
    bool join();


/**
   outer tuple = left
   inner tuple = right
 */

    bool setTableData(COLUMNDATA *oCD,
                      COLUMNDATA *iCD,
                      int outerSize,
                      int innerSize,
                      GComparisonExpression *GC){
        
        assert(outerSize >= 0 && innerSize >= 0);
        assert(oCD != NULL && iCD != NULL);

        left_CD = oCD;
        right_CD = iCD;
        left = outerSize;
        right = innerSize;
        
        expression = GC;

        PART = 256 * 1024;
        
        uint biggerTupleSize = left;
        if(left < right) biggerTupleSize = right;

        for(int i=32768 ; i<=256 * 1024 ; i = i*2){
            if(biggerTupleSize<=i){
                PART = i;
                break;
            }
        }
        printf("PART : %d\n",PART);

        //NIJ is always true. SHJ may become false.
        return true;

    }

    RESULT *getResult(){
        return jt;
    }

    int getResultSize(){
        return total;
    }


private:

//for partition execution
   
    RESULT *jt;
    int total;

    uint left,right;
    COLUMNDATA *left_CD;
    COLUMNDATA *right_CD;

    GComparisonExpression *expression;

    int PART;

    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function;
    CUmodule module,c_module;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};

#endif
