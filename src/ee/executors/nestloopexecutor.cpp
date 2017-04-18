/* This file is part of VoltDB.
 * Copyright (C) 2008-2015 VoltDB Inc.
 *
 * This file contains original code and/or modifications of original code.
 * Any modifications made by VoltDB Inc. are licensed under the following
 * terms and conditions:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */
/* Copyright (C) 2008 by H-Store Project
 * Brown University
 * Massachusetts Institute of Technology
 * Yale University
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#include <vector>
#include <string>
#include <stack>
#include "nestloopexecutor.h"
#include "common/debuglog.h"
#include "common/common.h"
#include "common/tabletuple.h"
#include "common/FatalException.hpp"
#include "common/types.h"
#include "executors/aggregateexecutor.h"
#include "execution/ProgressMonitorProxy.h"
#include "expressions/abstractexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "expressions/comparisonexpression.h"
#include "storage/table.h"
#include "storage/temptable.h"
#include "storage/tableiterator.h"
#include "plannodes/nestloopnode.h"
#include "plannodes/limitnode.h"
#include "plannodes/aggregatenode.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUNIJ.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"


#ifdef VOLT_DEBUG_ENABLED
#include <ctime>
#include <sys/times.h>
#include <unistd.h>
#endif


using namespace std;
using namespace voltdb;

bool NestLoopExecutor::p_init(AbstractPlanNode* abstract_node,
                              TempTableLimits* limits)
{
    VOLT_TRACE("init NestLoop Executor");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(abstract_node);
    assert(node);

    // Create output table based on output schema from the plan
    setTempOutputTable(limits);

    assert(m_tmpOutputTable);

    // NULL tuple for outer join
    if (node->getJoinType() == JOIN_TYPE_LEFT) {
        Table* inner_table = node->getInputTable(1);
        assert(inner_table);
        m_null_tuple.init(inner_table->schema());
    }

    // Inline aggregation can be serial, partial or hash
    m_aggExec = voltdb::getInlineAggregateExecutor(m_abstractNode);

    return true;
}


/**
copy m_data,SourceInlined and ValueType from NValue to GNValue
 */
//void setGNValue(COLUMNDATA *cd,NValue NV){
//  //void setGNValue(int rowdata,char *columndata,int columnsize,NValue NV){
//
//  //memcpy(columndata[rowdata*columnsize],NV.getMdataForGPU(),columnsize);
//
//  cd->gn.setMdata(NV.getValueTypeForGPU(), NV.getMdataForGPU());
//  cd->gn.setSourceInlined(NV.getSourceInlinedForGPU());
//  cd->gn.setValueType(NV.getValueTypeForGPU());
//}


void setGNValue(GNValue *column_data, NValue value)
{
	column_data->setMdata(value.getValueTypeForGPU(), value.getMdataForGPU());
//	column_data->setSourceInlined(value.getSourceInlinedForGPU());
	column_data->setValueType(value.getValueTypeForGPU());
}

void swap(RESULT *a,RESULT *b)
{
  RESULT temp;
  temp = *a;
  *a=*b;
  *b=temp;
}

void qsort(RESULT *jt,int p,int q)
{
  int i,j;
  int pivot;

  i = p;
  j = q;

  pivot = jt[(p+q)/2].lkey;

  while(1){
    while(jt[i].lkey < pivot) i++;
    while(pivot < jt[j].lkey) j--;
    if(i>=j) break;
    swap(&jt[i],&jt[j]);
    i++;
    j--;
  }
  if(p < i-1) qsort(jt,p,i-1);
  if(j+1 < q) qsort(jt,j+1,q);

}



bool NestLoopExecutor::p_execute(const NValueArray &params) {
    VOLT_DEBUG("executing NestLoop...");

    NestLoopPlanNode* node = dynamic_cast<NestLoopPlanNode*>(m_abstractNode);
    assert(node);
    assert(node->getInputTableCount() == 2);

    // output table must be a temp table
    assert(m_tmpOutputTable);

    Table* outer_table = node->getInputTable();
    assert(outer_table);

    Table* inner_table = node->getInputTable(1);
    assert(inner_table);

    VOLT_TRACE ("input table left:\n %s", outer_table->debug().c_str());
    VOLT_TRACE ("input table right:\n %s", inner_table->debug().c_str());

    //
    // Pre Join Expression
    //
    AbstractExpression *preJoinPredicate = node->getPreJoinPredicate();
    TreeExpression post_preJoinTree(preJoinPredicate);
    printf("Pre-Join Predicate::");
    post_preJoinTree.debug();
    if (preJoinPredicate) {
        VOLT_TRACE ("Pre Join predicate: %s", preJoinPredicate == NULL ?
                    "NULL" : preJoinPredicate->debug(true).c_str());
    }
    //
    // Join Expression
    //
    AbstractExpression *joinPredicate = node->getJoinPredicate();
    TreeExpression post_joinTree(joinPredicate);
    printf("Join Predicate::");
    post_joinTree.debug();
    if (joinPredicate) {
        VOLT_TRACE ("Join predicate: %s", joinPredicate == NULL ?
                    "NULL" : joinPredicate->debug(true).c_str());
    }
    //
    // Where Expression
    //
    AbstractExpression *wherePredicate = node->getWherePredicate();
    TreeExpression post_whereTree(wherePredicate);
    printf("Where Predicate::");
    post_whereTree.debug();
    if (wherePredicate) {
        VOLT_TRACE ("Where predicate: %s", wherePredicate == NULL ?
                    "NULL" : wherePredicate->debug(true).c_str());
    }

    LimitPlanNode* limit_node = dynamic_cast<LimitPlanNode*>(node->getInlinePlanNode(PLAN_NODE_TYPE_LIMIT));
    int limit = -1;
    int tuple_ctr = 0;
    int offset = -1;
    if (limit_node) {
        limit_node->getLimitAndOffsetByReference(params, limit, offset);
    }

    int outer_cols = outer_table->columnCount();
    int inner_cols = inner_table->columnCount();
    TableTuple outer_tuple(node->getInputTable(0)->schema());
    TableTuple inner_tuple(node->getInputTable(1)->schema());

    TableIterator iterator0 = outer_table->iterator();
    TableIterator iterator1 = inner_table->iterator();
    //int tuple_ctr = 0;
    //int tuple_skipped = 0;
    ProgressMonitorProxy pmp(m_engine, this, inner_table);

    TableTuple join_tuple;
    if (m_aggExec != NULL) {
        VOLT_TRACE("Init inline aggregate...");
        const TupleSchema * aggInputSchema = node->getTupleSchemaPreAgg();
        join_tuple = m_aggExec->p_execute_init(params, &pmp, aggInputSchema, m_tmpOutputTable);
    } else {
        join_tuple = m_tmpOutputTable->tempTuple();
    }

    GNValue *outer_data = NULL, *inner_data = NULL;
    TableTuple *tmp_outer_tuple = NULL,*tmp_inner_tuple = NULL;

    int outer_size = (int)outer_table->activeTupleCount();
    int inner_size = (int)inner_table->activeTupleCount();

    outer_data = (GNValue *)malloc(sizeof(GNValue) * outer_size * outer_cols);
    inner_data = (GNValue *)malloc(sizeof(GNValue) * inner_size * inner_cols);
    tmp_outer_tuple = (TableTuple *)malloc(sizeof(TableTuple) * outer_size);
    tmp_inner_tuple = (TableTuple *)malloc(sizeof(TableTuple) * inner_size);
    /*
      Recently ,only joinPredicate is implemented 
     */

    /**
       get left NValue of condition
    */
    int idx = 0;
    int tmp_idx = 0;
    int block = 0;
    int outer_block_size = (outer_size < DEFAULT_PART_SIZE_) ? outer_size : DEFAULT_PART_SIZE_;
    int inner_block_size = (inner_size < DEFAULT_PART_SIZE_) ? inner_size : DEFAULT_PART_SIZE_;

    while (iterator0.next(outer_tuple)) {
    	tmp_outer_tuple[idx] = outer_tuple;
    	for (int i = 0; i < outer_cols; i++)
    		setGNValue(&outer_data[tmp_idx + i * outer_block_size + block * outer_block_size * outer_cols], outer_tuple.getNValue(i));

    	idx++;
    	tmp_idx++;
    	if (idx % DEFAULT_PART_SIZE_ == 0) {
    		block++;
    		tmp_idx = 0;
    	}
    }
    idx = 0;
    tmp_idx = 0;
    block = 0;

    std::cout << std::endl;
    while (iterator1.next(inner_tuple)) {
    	tmp_inner_tuple[idx] = inner_tuple;
    	for (int i = 0; i < inner_cols; i++)
    		setGNValue(&inner_data[tmp_idx + i * inner_block_size + block * inner_block_size * inner_cols], inner_tuple.getNValue(i));

    	idx++;
    	tmp_idx++;
    	if (idx % DEFAULT_PART_SIZE_ == 0) {
    		block++;
    		tmp_idx = 0;
    	}
    }

    RESULT *join_result = NULL;
    int result_size = 0;
    bool ret;
    bool earlyReturned = false;

    GPUNIJ gn(outer_data, inner_data, outer_size, outer_cols, inner_size, inner_cols, post_preJoinTree, post_joinTree, post_whereTree);

    ret = gn.join();
    if (!ret) {
    	printf("Error: join failed\n");
    } else {
    	result_size = gn.getResultSize();
    	join_result = (RESULT *)malloc(sizeof(RESULT) * result_size);
    	gn.getResult(join_result);

		for (int i = 0; i < result_size && (limit == -1 || tuple_ctr < limit); i++, tuple_ctr++) {
			int l = join_result[i].lkey;
			int r = join_result[i].rkey;

			if (l >= 0 && r >= 0 && l < outer_size && r < inner_size) {
				join_tuple.setNValues(0, tmp_outer_tuple[l], 0, outer_cols);
				join_tuple.setNValues(outer_cols, tmp_inner_tuple[r], 0, inner_cols);
			}

			if (m_aggExec != NULL) {
				if (m_aggExec->p_execute_tuple(join_tuple)){
					// Get enough rows for LIMIT
					earlyReturned = true;
					break;
				}
			} else {
				m_tmpOutputTable->insertTempTuple(join_tuple);
				pmp.countdownProgress();
			}

			if (earlyReturned) {
				break;
			}
		}
    }



    /*

    while ((limit == -1 || tuple_ctr < limit) && iterator0.next(outer_tuple)){
        pmp.countdownProgress();

        // populate output table's temp tuple with outer table's values
        // probably have to do this at least once - avoid doing it many
        // times per outer tuple
        join_tuple.setNValues(0, outer_tuple, 0, outer_cols);

        // did this loop body find at least one match for this tuple?
        bool match = false;
        // For outer joins if outer tuple fails pre-join predicate
        // (join expression based on the outer table only)
        // it can't match any of inner tuples
        if (preJoinPredicate == NULL || preJoinPredicate->eval(&outer_tuple, NULL).isTrue()) {

            // By default, the delete as we go flag is false.
            TableIterator iterator1 = inner_table->iterator();
            while ((limit == -1 || tuple_ctr < limit) && iterator1.next(inner_tuple)) {
                pmp.countdownProgress();
                // Apply join filter to produce matches for each outer that has them,
                // then pad unmatched outers, then filter them all
                if (joinPredicate == NULL || joinPredicate->eval(&outer_tuple, &inner_tuple).isTrue()){
                    match = true;
                    // Filter the joined tuple
                    if (wherePredicate == NULL || wherePredicate->eval(&outer_tuple, &inner_tuple).isTrue()) {
                        // Check if we have to skip this tuple because of offset
                        if (tuple_skipped < offset) {
                            tuple_skipped++;
                            continue;
                        }
                        ++tuple_ctr;
                        // Matched! Complete the joined tuple with the inner column values.
                        join_tuple.setNValues(outer_cols, inner_tuple, 0, inner_cols);
                        if (m_aggExec != NULL) {
                            if (m_aggExec->p_execute_tuple(join_tuple)){
                                // Get enough rows for LIMIT
                                earlyReturned = true;
                                break;
                            }
                        } else {
                            m_tmpOutputTable->insertTempTuple(join_tuple);
                            pmp.countdownProgress();
                        }
                    }
                }
            } // END INNER WHILE LOOP
        } // END IF PRE JOIN CONDITION

        //
        // Left Outer Join
        //
        if (join_type == JOIN_TYPE_LEFT && !match && (limit == -1 || tuple_ctr < limit)) {
            // Still needs to pass the filter
            if (wherePredicate == NULL || wherePredicate->eval(&outer_tuple, &null_tuple).isTrue()) {
                // Check if we have to skip this tuple because of offset
                if (tuple_skipped < offset) {
                    tuple_skipped++;
                    continue;
                }
                ++tuple_ctr;
                join_tuple.setNValues(outer_cols, null_tuple, 0, inner_cols);
                if (m_aggExec != NULL) {
                    if (m_aggExec->p_execute_tuple(join_tuple)) {
                        earlyReturned = true;
                    }
                } else {
                    m_tmpOutputTable->insertTempTuple(join_tuple);
                    pmp.countdownProgress();
                }
            }
        } // END IF LEFT OUTER JOIN

        if (earlyReturned) {
            // Get enough rows for LIMIT inlined with aggregation
            break;
        }

    } // END OUTER WHILE LOOP
    */

    if (m_aggExec != NULL) {
        m_aggExec->p_execute_finish();
    }

    free(tmp_outer_tuple);
    free(tmp_inner_tuple);
    free(outer_data);
    free(inner_data);
    cleanupInputTempTable(inner_table);
    cleanupInputTempTable(outer_table);

    return (true);
}

