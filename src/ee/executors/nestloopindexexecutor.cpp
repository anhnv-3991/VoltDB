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
#include "nestloopindexexecutor.h"
#include "common/debuglog.h"
#include "common/tabletuple.h"
#include "common/FatalException.hpp"

#include "execution/VoltDBEngine.h"
#include "executors/aggregateexecutor.h"
#include "execution/ProgressMonitorProxy.h"
#include "expressions/abstractexpression.h"
#include "expressions/tuplevalueexpression.h"

#include "plannodes/nestloopindexnode.h"
#include "plannodes/indexscannode.h"
#include "plannodes/limitnode.h"
#include "plannodes/aggregatenode.h"

#include "storage/table.h"
#include "storage/persistenttable.h"
#include "storage/temptable.h"
#include "storage/tableiterator.h"

#include "indexes/tableindex.h"
#include "expressions/comparisonexpression.h"
#include <sys/time.h>


using namespace std;
using namespace voltdb;

bool NestLoopIndexExecutor::p_init(AbstractPlanNode* abstractNode,
                                   TempTableLimits* limits)
{
    VOLT_TRACE("init NLIJ Executor");
    assert(limits);

    NestLoopIndexPlanNode* node = dynamic_cast<NestLoopIndexPlanNode*>(m_abstractNode);
    assert(node);
    m_indexNode =
        dynamic_cast<IndexScanPlanNode*>(m_abstractNode->getInlinePlanNode(PLAN_NODE_TYPE_INDEXSCAN));
    assert(m_indexNode);
    VOLT_TRACE("<NestLoopIndexPlanNode> %s, <IndexScanPlanNode> %s",
               m_abstractNode->debug().c_str(), m_indexNode->debug().c_str());

    m_joinType = node->getJoinType();
    m_lookupType = m_indexNode->getLookupType();
    m_sortDirection = m_indexNode->getSortDirection();

    // Inline aggregation can be serial, partial or hash
    m_aggExec = voltdb::getInlineAggregateExecutor(m_abstractNode);

    //
    // We need exactly one input table and a target table
    //
    assert(node->getInputTableCount() == 1);

    // Create output table based on output schema from the plan
    setTempOutputTable(limits);

    // output must be a temp table
    assert(m_tmpOutputTable);

    node->getOutputColumnExpressions(m_outputExpressions);

    //
    // Make sure that we actually have search keys
    //
    int num_of_searchkeys = (int)m_indexNode->getSearchKeyExpressions().size();
    //nshi commented this out in revision 4495 of the old repo in index scan executor
    VOLT_TRACE ("<Nested Loop Index exec, INIT...> Number of searchKeys: %d \n", num_of_searchkeys);

    for (int ctr = 0; ctr < num_of_searchkeys; ctr++) {
        if (m_indexNode->getSearchKeyExpressions()[ctr] == NULL) {
            VOLT_ERROR("The search key expression at position '%d' is NULL for"
                       " internal PlanNode '%s' of PlanNode '%s'",
                       ctr, m_indexNode->debug().c_str(), node->debug().c_str());
            return false;
        }
    }

    assert(node->getInputTable());

    PersistentTable* inner_table = dynamic_cast<PersistentTable*>(m_indexNode->getTargetTable());
    assert(inner_table);

    // Grab the Index from our inner table
    // We'll throw an error if the index is missing
    TableIndex* index = inner_table->index(m_indexNode->getTargetIndexName());
    if (index == NULL) {
        VOLT_ERROR("Failed to retreive index '%s' from inner table '%s' for"
                   " internal PlanNode '%s'",
                   m_indexNode->getTargetIndexName().c_str(),
                   inner_table->name().c_str(), m_indexNode->debug().c_str());
        return false;
    }

    // NULL tuple for outer join
    if (node->getJoinType() == JOIN_TYPE_LEFT) {
        Table* inner_out_table = m_indexNode->getOutputTable();
        assert(inner_out_table);
        m_null_tuple.init(inner_out_table->schema());
    }

    m_indexValues.init(index->getKeySchema());
    return true;
}

bool NestLoopIndexExecutor::p_execute(const NValueArray &params)
{
//	std::cout << "NestLoopIndexExecutor execute" << std::endl;
//	printf("Size of GTreeNode = %d, size of GNValue = %d\n", (int)sizeof(GTreeNode), (int)sizeof(GNValue));
//	struct timeval start, finish;
//
//	gettimeofday(&start, NULL);
//
//    NestLoopIndexPlanNode* node = dynamic_cast<NestLoopIndexPlanNode*>(m_abstractNode);
//    assert(node);
//
//    // output table must be a temp table
//    assert(m_tmpOutputTable);
//
//    PersistentTable* inner_table = dynamic_cast<PersistentTable*>(m_indexNode->getTargetTable());
//    assert(inner_table);
//
//    TableIndex* index = inner_table->index(m_indexNode->getTargetIndexName());
//    std::cout << "Target Index Name = " << m_indexNode->getTargetIndexName() << std::endl;
//    assert(index);
//    IndexCursor indexCursor(index->getTupleSchema());
//
//    // NULL tuple for outer join
//    if (node->getJoinType() == JOIN_TYPE_LEFT) {
//        Table* inner_out_table = m_indexNode->getOutputTable();
//        assert(inner_out_table);
//        m_null_tuple.init(inner_out_table->schema());
//    }
//
//    //outer_table is the input table that have tuples to be iterated
//    assert(node->getInputTableCount() == 1);
//    Table* outer_table = node->getInputTable();
//    assert(outer_table);
//    VOLT_TRACE("executing NestLoopIndex with outer table: %s, inner table: %s",
//               outer_table->debug().c_str(), inner_table->debug().c_str());
//
//    //
//    // Substitute parameter to SEARCH KEY Note that the expressions
//    // will include TupleValueExpression even after this substitution
//    //
//    int num_of_searchkeys = (int)m_indexNode->getSearchKeyExpressions().size();
//
//    // end expression
//    // where table1.field = table2.field
//    AbstractExpression* end_expression = m_indexNode->getEndExpression();
//    if (end_expression) {
//        VOLT_TRACE("End Expression:\n%s", end_expression->debug(true).c_str());
//    }
//
//    // post expression
//    AbstractExpression* post_expression = m_indexNode->getPredicate();
//    if (post_expression != NULL) {
//        VOLT_TRACE("Post Expression:\n%s", post_expression->debug(true).c_str());
//    }
//
//    // initial expression
//    AbstractExpression* initial_expression = m_indexNode->getInitialExpression();
//    if (initial_expression != NULL) {
//        VOLT_TRACE("Initial Expression:\n%s", initial_expression->debug(true).c_str());
//    }
//
//    // SKIP NULL EXPRESSION
//    AbstractExpression* skipNullExpr = m_indexNode->getSkipNullPredicate();
//
//    // For reverse scan edge case NULL values and forward scan underflow case.
//    if (skipNullExpr != NULL) {
//        VOLT_DEBUG("Skip NULL Expression:\n%s", skipNullExpr->debug(true).c_str());
//    }
//
//    // pre join expression
//    AbstractExpression* prejoin_expression = node->getPreJoinPredicate();
//    if (prejoin_expression != NULL) {
//        VOLT_TRACE("Prejoin Expression:\n%s", prejoin_expression->debug(true).c_str());
//    }
//
//    // where expression
//    AbstractExpression* where_expression = node->getWherePredicate();
//    if (where_expression != NULL) {
//        VOLT_TRACE("Where Expression:\n%s", where_expression->debug(true).c_str());
//    }
//
//    LimitPlanNode* limit_node = dynamic_cast<LimitPlanNode*>(node->getInlinePlanNode(PLAN_NODE_TYPE_LIMIT));
//    int tuple_ctr = 0;
//    //int tuple_skipped = 0;
//    int limit = -1;
//    int offset = -1;
//    if (limit_node) {
//        limit_node->getLimitAndOffsetByReference(params, limit, offset);
//    }
//
//    //2015.11.04 - added for GPU Join
//    std::vector<int> inner_indices = index->getColumnIndices();
//    if (inner_indices.empty()) {
//    	cout << "Empty indexed expression" << endl;
//    }
//
//    TableTuple outer_tuple(outer_table->schema());
//    TableTuple inner_tuple(inner_table->schema());
//    TableIterator outer_iterator = outer_table->iteratorDeletingAsWeGo();
//    int num_of_outer_cols = outer_table->columnCount();
//    assert (outer_tuple.sizeInValues() == outer_table->columnCount());
//    assert (inner_tuple.sizeInValues() == inner_table->columnCount());
//   // const TableTuple &null_tuple = m_null_tuple.tuple();
//    //nt num_of_inner_cols = (m_joinType == JOIN_TYPE_LEFT)? null_tuple.sizeInValues() : 0;
//    ProgressMonitorProxy pmp(m_engine, this, inner_table);
//
//    TableTuple join_tuple;
//    if (m_aggExec != NULL) {
//        VOLT_TRACE("Init inline aggregate...");
//        const TupleSchema * aggInputSchema = node->getTupleSchemaPreAgg();
//        join_tuple = m_aggExec->p_execute_init(params, &pmp, aggInputSchema, m_tmpOutputTable);
//    } else {
//        join_tuple = m_tmpOutputTable->tempTuple();
//    }

    //bool earlyReturned = false;


//	/************ Build Expression Tree *****************************/
//	TreeExpression end_ex_tree(end_expression);
//
//	TreeExpression post_ex_tree(post_expression);
//
//	TreeExpression initial_ex_tree(initial_expression);
//
//	TreeExpression skipNull_ex_tree(skipNullExpr);
//
//	TreeExpression prejoin_ex_tree(prejoin_expression);
//
//	TreeExpression where_ex_tree(where_expression);
//
//
//	/******************** Add for GPU join **************************/
//
//    /******************** GET COLUMN DATA ***************************/
//    int outer_size = (int)outer_table->activeTupleCount();
//    int inner_size = (int)inner_table->activeTupleCount();
//
//	NValue inner_nv_tmp;
//
//	/********** Get column data for end_expression (search keys) & post_expression from outer table ***************/
//	std::vector<int> search_keys(0);
//
//	struct timeval join_start, join_end, write_start, write_end;
//
//	std::vector<TreeExpression> gsearchKeyExpressions;
//
//	for (int ctr = 0; ctr < num_of_searchkeys; ctr++) {
//		TreeExpression tmp(m_indexNode->getSearchKeyExpressions()[ctr]);
//		gsearchKeyExpressions.push_back(tmp);
//		tmp.debug();
//		std::cout << std::endl;
//	}
//
//	for (int ctr = 0; ctr < inner_indices.size(); ctr++) {
//		std::cout << "Inner index " << ctr << " = " << inner_indices[ctr] << std::endl;
//	}
//
//	int col_outer = outer_tuple.sizeInValues();
//	printf("Number of outer_columns: %d and %d\n", col_outer, num_of_outer_cols);
//
//	GTable outer, inner;
//
//    outer.block_list = &(outer_table->getGBlockList()[0]);
//    outer.column_num = (int)(outer_table->activeTupleCount());
//    outer.schema = &(outer_table->getGSchema()[0]);
//
//    inner.block_list = &(inner_table->getGBlockList()[0]);
//    inner.column_num = (int)(inner_table->activeTupleCount());
//    inner.schema = &(inner_table->getGSchema()[0]);
//
//	bool ret = true;
//	RESULT *join_result = NULL;
//	int result_size = 0;
//	IndexLookupType lookup_type = m_lookupType;
//    /* Copy data to GPU memory */
//	int real_result_size = 0;
//	if (outer_size != 0 && inner_size != 0) {
//
//		gettimeofday(&join_start, NULL);
//		GPUIJ gn(outer, inner, gsearchKeyExpressions, inner_indices, end_ex_tree,
//					post_ex_tree, initial_ex_tree, skipNull_ex_tree, prejoin_ex_tree, where_ex_tree, lookup_type);
//		GPUHJ gn(index_data_out,
//					index_data_in,
//					outer_size,
//					col_outer,
//					inner_size,
//					col_inner,
//					gsearchKeyExpressions,
//					inner_indices,
//					end_ex_tree,
//					post_ex_tree,
//					initial_ex_tree,
//					skipNull_ex_tree,
//					prejoin_ex_tree,
//					where_ex_tree,
//					lookup_type,
//					m_sizeIndex
//					);

//		printf("Start joining\n");
//		ret = gn.join();
//		gettimeofday(&join_end, NULL);
//
//
//		if (ret != true) {
//			std::cout << "Error: gpu join failed." << std::endl;
//		} else {
//			result_size = gn.getResultSize();
//
//
//			join_result = (RESULT *)malloc(sizeof(RESULT) * result_size);
//			gn.getResult(join_result);
//
//			printf("Size of result: %d\n", result_size);
//			gettimeofday(&write_start, NULL);
//			for (int i = 0; i < result_size && (limit == -1 || tuple_ctr < limit); i++, tuple_ctr++) {
//				int l = join_result[i].lkey;
//				int r = join_result[i].rkey;
//
//				if (l >= 0 && r >= 0 && l < outer_size && r < inner_size) {
//					real_result_size++;
//					//printf("Index of output outer = %d and inner = %d\n", l, r);
//					//join_tuple.setNValues(0, tmp_outer_tuple[l], 0, num_of_outer_cols);
//
//					for (int col_ctr = num_of_outer_cols; col_ctr < join_tuple.sizeInValues(); ++col_ctr) {
//						//std::cout << m_outputExpressions[col_ctr]->debug() << std::endl;;
//						//join_tuple.setNValue(col_ctr,
//						//		  m_outputExpressions[col_ctr]->eval(&tmp_outer_tuple[l], &tmp_inner_tuple[r]));
//					}
//				}

//                if (m_aggExec != NULL) {
//                    if (m_aggExec->p_execute_tuple(join_tuple)) {
//                    	// Get enough rows for LIMIT
//
//                        earlyReturned = true;
//                        break;
//                    }
//                } else {
//                    m_tmpOutputTable->insertTempTuple(join_tuple);
//                    pmp.countdownProgress();
//                }
//
//				if (earlyReturned) {
//					break;
//				}
//			}
//			gettimeofday(&write_end, NULL);
//		}
//	}
//
//	printf("Elapsed time for joining: %ld\n", (join_end.tv_sec - join_start.tv_sec) * 1000000 + (join_end.tv_usec - join_start.tv_usec));
//	printf("Elapsed time for writing result: %ld\n", (write_end.tv_sec - write_start.tv_sec) * 1000000 + (write_end.tv_usec - write_start.tv_usec));
//	printf("Real result size = %d\n", real_result_size);
//
//    /******************* End of adding GPU join ********************/
//
//    if (m_aggExec != NULL) {
//        m_aggExec->p_execute_finish();
//    }
//
//    VOLT_TRACE ("result table:\n %s", m_tmpOutputTable->debug().c_str());
//    VOLT_TRACE("Finished NestLoopIndex");
//
//    cleanupInputTempTable(inner_table);
//    cleanupInputTempTable(outer_table);
//
//    if (outer_size != 0 && inner_size != 0 && result_size != 0) {
//    	free(join_result);
//    }
//
//    gettimeofday(&finish, NULL);
//
//    printf("Elapsed time: %lu microseconds\n", ((finish.tv_sec - start.tv_sec) * 1000000 + finish.tv_usec - start.tv_usec));

    //exit(0); //2016.08.26 - Added to force quit for visual profiling
    return (true);
}

NestLoopIndexExecutor::~NestLoopIndexExecutor() { }
