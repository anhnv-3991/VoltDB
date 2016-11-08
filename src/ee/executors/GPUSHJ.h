/********************************
タプルの情報はここでまとめておく。

元のプログラムでは構造体のリストだったが、
GPUで動かすため配列のほうが向いていると思ったので
配列に変更している
********************************/

#ifndef GPUSHJ_H
#define GPUSHJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
//#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"

using namespace voltdb;

class GPUSHJ{

public:


    GPUSHJ();

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

        if(GC->getET() == EXPRESSION_TYPE_COMPARE_EQUAL){
            expression = GC;
            return true;
        }else{
            return false;
        }
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

    //int PART;

    CUresult res;
    CUdevice dev;
    CUcontext ctx;
    CUfunction function,c_function,p_function,pc_function,sp_function;
    CUmodule module,pmodule;
    
    void printDiff(struct timeval begin, struct timeval end);

    uint iDivUp(uint dividend, uint divisor);


};

#endif
