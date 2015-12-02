
#ifndef MAKEEXPRESSIONTREE_H
#define MAKEEXPRESSIONTREE_H

#include <stdio.h>
#include <cmath>

#include "common/types.h"
#include "expressions/abstractexpression.h"
#include "expressions/comparisonexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "GPUetc/expressions/Gabstractexpression.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"
#include "GPUetc/expressions/Gtuplevalueexpression.h"
#include "GPUetc/expressions/nodedata.h"


namespace voltdb{


/**
This class make a tree including expression information for GPGPU.
Each node has ExpressionType, startPos and etc.
The reason is that expression data is integrated in one array because we must send necessary data to GPU.
So I make one array including Expressions. 
The startPos of the structure of this class is the start position of an Expression in the array of GPU.

I assume this class used to seach an Expression in the GPU array.

*/


/**
rule of allocating tree pos

    root
      0

    1   2

   3 4 5 6

*/


class makeExpressionTree{

public:
  makeExpressionTree(){};

  /*
    reclusive function. first is maketree(ab,0,1)
   */

  void maketree(const AbstractExpression *ab,int pos,int dep){

      if(ab == NULL){
          if(pos == 0){
              enode[pos].et = EXPRESSION_TYPE_INVALID;
              
          }else{
              enode[pos].et = EXPRESSION_TYPE_INVALID;
          }
      }else{
          enode[pos].et = ab->getExpressionType();
      }

      if(dep >= 4){
          return;
      }

      int nextpos = static_cast<int>((pow(2,dep)-1) + (pos+1-pow(2,dep)/2)*2);

      if(enode[pos].et == EXPRESSION_TYPE_VALUE_TUPLE || ab == NULL){
          maketree(NULL,nextpos,dep+1);
          maketree(NULL,nextpos+1,dep+1);
      }else{
          maketree(ab->getLeft(),nextpos,dep+1);
          maketree(ab->getRight(),nextpos+1,dep+1);

      }
  }

  void setSize(){
      enode[0].startPos = 0;
      enode[0].endPos = expressionSize(enode[0].et);

      for(int i = 1 ; i<15 ; i++){
          enode[i].startPos = enode[i-1].endPos;
          enode[i].endPos = enode[i].startPos + expressionSize(enode[i].et);
      }
  }


  void allocate(const AbstractExpression *ab, char *data,int pos, int dep) {

      //printf("pos = %d  dep = %d\n",pos,dep);

      int nextpos = static_cast<int>((pow(2,dep)-1) + (pos+1-pow(2,dep)/2)*2);

      switch(enode[pos].et){
      case EXPRESSION_TYPE_OPERATOR_PLUS:
      case EXPRESSION_TYPE_OPERATOR_MINUS:
      case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
      case EXPRESSION_TYPE_OPERATOR_DIVIDE:
          break;
      case EXPRESSION_TYPE_OPERATOR_NOT:
          break;
      case EXPRESSION_TYPE_OPERATOR_IS_NULL:
          break;
      case EXPRESSION_TYPE_COMPARE_EQUAL:
      case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
      case EXPRESSION_TYPE_COMPARE_LESSTHAN:
      case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
      case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
      case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
      {
          GComparisonExpression *tmpGCE = 
              new GComparisonExpression(enode[pos].et,enode[nextpos].startPos,enode[nextpos+1].startPos,enode[nextpos].et,enode[nextpos+1].et);
          tmpGCE->setInBytes(ab->getInBytes());
          tmpGCE->setValueSize(ab->getValueSize());
          tmpGCE->setValueType(ab->getValueType());
          memcpy(&data[enode[pos].startPos],tmpGCE,sizeof(GComparisonExpression));
          delete tmpGCE;
          break;
      }
      case EXPRESSION_TYPE_CONJUNCTION_AND:
      case EXPRESSION_TYPE_CONJUNCTION_OR:
          break;
      case EXPRESSION_TYPE_VALUE_TUPLE:
      {
          TupleValueExpression *tmpTV = reinterpret_cast<TupleValueExpression*>(const_cast<AbstractExpression*>(ab));
          GTupleValueExpression *tmpGTVE = new GTupleValueExpression(tmpTV->getTupleId(),tmpTV->getColumnId());
          tmpGTVE->setInBytes(ab->getInBytes());
          tmpGTVE->setValueSize(ab->getValueSize());
          tmpGTVE->setValueType(ab->getValueType());
          memcpy(&data[enode[pos].startPos],tmpGTVE,sizeof(GTupleValueExpression));
          delete tmpGTVE;
          break;
      }
      default:
          break;
      }

      if(dep >= 4){
          return;
      }

      if(enode[pos].et == EXPRESSION_TYPE_VALUE_TUPLE || ab == NULL){
          allocate(NULL ,data ,nextpos ,dep+1);
          allocate(NULL ,data ,nextpos+1 ,dep+1);
      }else{
          allocate(ab->getLeft() ,data ,nextpos ,dep+1);
          allocate(ab->getRight() ,data ,nextpos+1 ,dep+1);
      }

  }
  

  int getSize(){
      return enode[14].endPos;
  }

  EXPRESSIONNODE getENode(int idx){
    assert(idx < 15);
    return enode[idx];
  }


private:


  int expressionSize(ExpressionType et){

    switch(et){
    case EXPRESSION_TYPE_OPERATOR_PLUS:
    case EXPRESSION_TYPE_OPERATOR_MINUS:
    case EXPRESSION_TYPE_OPERATOR_MULTIPLY :
    case EXPRESSION_TYPE_OPERATOR_DIVIDE:
        return 0;
        //return sizeof(OperatorExpression);
    case EXPRESSION_TYPE_OPERATOR_NOT:
        return 0;
        //return sizeof(OperatorNOTExpression);
    case EXPRESSION_TYPE_OPERATOR_IS_NULL:
        return 0;
        //return sizeof(OperatorIsNullExpression);
    case EXPRESSION_TYPE_COMPARE_EQUAL:
    case EXPRESSION_TYPE_COMPARE_NOTEQUAL:
    case EXPRESSION_TYPE_COMPARE_LESSTHAN:
    case EXPRESSION_TYPE_COMPARE_GREATERTHAN:
    case EXPRESSION_TYPE_COMPARE_LESSTHANOREQUALTO:
    case EXPRESSION_TYPE_COMPARE_GREATERTHANOREQUALTO:
        return sizeof(GComparisonExpression);
    case EXPRESSION_TYPE_CONJUNCTION_AND:
    case EXPRESSION_TYPE_CONJUNCTION_OR:
        return 0;
        //return sizeof(ConjunctionExpression);
    case EXPRESSION_TYPE_VALUE_TUPLE:
      return sizeof(GTupleValueExpression);
    default:
      return 0;     
    }
  }

  
  EXPRESSIONNODE enode[15];

};

}


#endif
