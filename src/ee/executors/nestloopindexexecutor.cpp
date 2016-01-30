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
#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUNIJ.h"
#include "GPUSHJ.h"
#include "GPUIJ.h"
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/Gcomparisonexpression.h"
#include "expressions/comparisonexpression.h"
#include "GPUetc/expressions/treeexpression.h"
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

void setGNValue(GNValue *column_data, NValue &value)
{
	column_data->setMdata(value.getMdataForGPU());
	column_data->setSourceInlined(value.getSourceInlinedForGPU());
	column_data->setValueType(value.getValueTypeForGPU());
}

//Test the value of IndexData
void GNValueDebug(GNValue &column_data)
{
	NValue value;
	value.setMdataFromGPU(column_data.getMdata());
	value.setSourceInlinedFromGPU(column_data.getSourceInlined());
	value.setValueTypeFromGPU(column_data.getValueType());

	std::cout << value.debug();
}


bool NestLoopIndexExecutor::p_execute(const NValueArray &params)
{
	struct timeval start, finish;

	gettimeofday(&start, NULL);

    NestLoopIndexPlanNode* node = dynamic_cast<NestLoopIndexPlanNode*>(m_abstractNode);
    assert(node);

    // output table must be a temp table
    assert(m_tmpOutputTable);

    PersistentTable* inner_table = dynamic_cast<PersistentTable*>(m_indexNode->getTargetTable());
    assert(inner_table);

    TableIndex* index = inner_table->index(m_indexNode->getTargetIndexName());
    assert(index);
    IndexCursor indexCursor(index->getTupleSchema());

    // NULL tuple for outer join
    if (node->getJoinType() == JOIN_TYPE_LEFT) {
        Table* inner_out_table = m_indexNode->getOutputTable();
        assert(inner_out_table);
        m_null_tuple.init(inner_out_table->schema());
    }

    //outer_table is the input table that have tuples to be iterated
    assert(node->getInputTableCount() == 1);
    Table* outer_table = node->getInputTable();
    assert(outer_table);
    VOLT_TRACE("executing NestLoopIndex with outer table: %s, inner table: %s",
               outer_table->debug().c_str(), inner_table->debug().c_str());

    //
    // Substitute parameter to SEARCH KEY Note that the expressions
    // will include TupleValueExpression even after this substitution
    //
    int num_of_searchkeys = (int)m_indexNode->getSearchKeyExpressions().size();
//    for (int ctr = 0; ctr < num_of_searchkeys; ctr++) {
//        VOLT_TRACE("Search Key[%d]:\n%s",
//                   ctr, m_indexNode->getSearchKeyExpressions()[ctr]->debug(true).c_str());
//    }
//    std::cout << "************************* START *************************" << std::endl;
//    for (int ctr = 0; ctr < num_of_searchkeys; ctr++) {
//    	std::cout << "Search key " << m_indexNode->getSearchKeyExpressions()[ctr]->debug(true).c_str() << std::endl;
//    }
//    std::cout << "************************** END *************************" << std::endl;

    // end expression
    // where table1.field = table2.field
    AbstractExpression* end_expression = m_indexNode->getEndExpression();
    if (end_expression) {
        VOLT_TRACE("End Expression:\n%s", end_expression->debug(true).c_str());
    }

    // post expression
    AbstractExpression* post_expression = m_indexNode->getPredicate();
    if (post_expression != NULL) {
        VOLT_TRACE("Post Expression:\n%s", post_expression->debug(true).c_str());
    }

    // initial expression
    AbstractExpression* initial_expression = m_indexNode->getInitialExpression();
    if (initial_expression != NULL) {
        VOLT_TRACE("Initial Expression:\n%s", initial_expression->debug(true).c_str());
    }

    // SKIP NULL EXPRESSION
    AbstractExpression* skipNullExpr = m_indexNode->getSkipNullPredicate();

    // For reverse scan edge case NULL values and forward scan underflow case.
    if (skipNullExpr != NULL) {
        VOLT_DEBUG("Skip NULL Expression:\n%s", skipNullExpr->debug(true).c_str());
    }

    // pre join expression
    AbstractExpression* prejoin_expression = node->getPreJoinPredicate();
    if (prejoin_expression != NULL) {
        VOLT_TRACE("Prejoin Expression:\n%s", prejoin_expression->debug(true).c_str());
    }

    // where expression
    AbstractExpression* where_expression = node->getWherePredicate();
    if (where_expression != NULL) {
        VOLT_TRACE("Where Expression:\n%s", where_expression->debug(true).c_str());
    }

    LimitPlanNode* limit_node = dynamic_cast<LimitPlanNode*>(node->getInlinePlanNode(PLAN_NODE_TYPE_LIMIT));
    int tuple_ctr = 0;
    //int tuple_skipped = 0;
    int limit = -1;
    int offset = -1;
    if (limit_node) {
        limit_node->getLimitAndOffsetByReference(params, limit, offset);
    }

    //2015.11.04 - added for GPU Join
    std::vector<int> inner_indices = index->getColumnIndices();
    if (inner_indices.empty()) {
    	cout << "Empty indexed expression" << endl;
    }

    TableTuple outer_tuple(outer_table->schema());
    TableTuple inner_tuple(inner_table->schema());
    TableIterator outer_iterator = outer_table->iteratorDeletingAsWeGo();
    int num_of_outer_cols = outer_table->columnCount();
    assert (outer_tuple.sizeInValues() == outer_table->columnCount());
    assert (inner_tuple.sizeInValues() == inner_table->columnCount());
    //const TableTuple &null_tuple = m_null_tuple.tuple();
    //int num_of_inner_cols = (m_joinType == JOIN_TYPE_LEFT)? null_tuple.sizeInValues() : 0;
    ProgressMonitorProxy pmp(m_engine, this, inner_table);

    TableTuple join_tuple;
    if (m_aggExec != NULL) {
        VOLT_TRACE("Init inline aggregate...");
        const TupleSchema * aggInputSchema = node->getTupleSchemaPreAgg();
        join_tuple = m_aggExec->p_execute_init(params, &pmp, aggInputSchema, m_tmpOutputTable);
    } else {
        join_tuple = m_tmpOutputTable->tempTuple();
    }

    //bool earlyReturned = false;



	/************ Build Expression Tree *****************************/
	TreeExpression end_ex_tree(end_expression);
	printf("End ex tree:::");
	end_ex_tree.debug();

	TreeExpression post_ex_tree(post_expression);
	printf("Post ex tree:::");
	post_ex_tree.debug();

	TreeExpression initial_ex_tree(initial_expression);
	printf("Initial ex tree:::");
	initial_ex_tree.debug();

	TreeExpression skipNull_ex_tree(skipNullExpr);
	printf("skipNull ex tree:::");
	skipNull_ex_tree.debug();

	TreeExpression prejoin_ex_tree(prejoin_expression);
	printf("Prejoin ex tree:::");
	prejoin_ex_tree.debug();

	TreeExpression where_ex_tree(where_expression);
	printf("Where ex tree:::");
	where_ex_tree.debug();

	/******************** Add for GPU join **************************/

    /******************** GET COLUMN DATA ***************************/
    int outer_size = (int)outer_table->activeTupleCount();
    int inner_size = (int)inner_table->activeTupleCount();

	NValue inner_nv_tmp;

	/********** Get column data for end_expression (search keys) & post_expression from outer table ***************/
	TableIterator search_it_out = outer_table->iterator(), search_it_in = inner_table->iterator();
	IndexData *index_data_out = (IndexData *)malloc(sizeof(IndexData) * outer_size);
	IndexData *index_data_in = (IndexData *)malloc(sizeof(IndexData) * inner_size);
	TableTuple *tmp_outer_tuple = (TableTuple *)malloc(sizeof(TableTuple) * outer_size);
	TableTuple *tmp_inner_tuple = (TableTuple *)malloc(sizeof(TableTuple) * inner_size);
	std::vector<int> search_keys(0);
	int idx;



	for (int ctr = 0; ctr < num_of_searchkeys; ctr++) {
		int tmp = (dynamic_cast<TupleValueExpression *>(m_indexNode->getSearchKeyExpressions()[ctr]))->getColumnId();
		//std::cout << "Search key " << ctr << " = " << tmp << std::endl;
		search_keys.push_back(tmp);
	}

//	for (int ctr = 0; ctr < inner_indices.size(); ctr++) {
//		std::cout << "Inner index " << ctr << " = " << inner_indices[ctr] << std::endl;
//	}
	idx = 0;
	while (search_it_out.next(outer_tuple)) {
//		if (prejoin_expression == NULL || prejoin_expression->eval(&outer_tuple, NULL).isTrue()) {
			//index_data_out[idx].num = idx;
			tmp_outer_tuple[idx] = outer_tuple;
			for (int i = 0; i < outer_tuple.sizeInValues(); i++) {
				NValue tmp_value = outer_tuple.getNValue(i);

				setGNValue(&(index_data_out[idx].gn[i]), tmp_value);
			}
			idx++;
//		}
	}

//	for (int ctr = 0; ctr < outer_size; ctr++) {
//		for (int i = 0; i < outer_tuple.sizeInValues(); i++) {
//			GNValueDebug(index_data_out[ctr].gn[i]);
//		}
//		std::cout << std::endl;
//	}

	/********** Get column data for end_expression (index keys) & post_expression from inner table ********************************/
	IndexCursor index_cursor2(index->getTupleSchema());

	/* Move to smallest key */
	bool begin = true;
	TableTuple tmp_tuple(inner_table->schema());

	index->moveToEnd(begin, index_cursor2);

	idx = 0;
	while (!(inner_tuple = index->nextValue(index_cursor2)).isNullTuple()) {
//		if (prejoin_expression == NULL || prejoin_expression->eval(&tmp_tuple, NULL).isTrue()) {
			//index_data_in[idx].num = idx;
			tmp_inner_tuple[idx] = inner_tuple;
			for (int i = 0; i < inner_tuple.sizeInValues(); i++) {
				NValue tmp_value = inner_tuple.getNValue(i);

				setGNValue(&(index_data_in[idx].gn[i]), tmp_value);
			}
			idx++;
		//}
	}

	bool ret = true;
	RESULT *join_result = NULL;
	int result_size = 0;
    /* Copy data to GPU memory */
	struct timeval join_start, join_end;

	if (outer_size != 0 && inner_size != 0) {

		GPUIJ gn(index_data_out, index_data_in, outer_size, inner_size, search_keys, inner_indices, end_ex_tree,
					post_ex_tree, initial_ex_tree, skipNull_ex_tree, prejoin_ex_tree, where_ex_tree);

		gettimeofday(&join_start, NULL);
		ret = gn.join();
		gettimeofday(&join_end, NULL);
		if (ret != true) {
			std::cout << "Error: gpu join failed." << std::endl;
		} else {
			result_size = gn.getResultSize();

			join_result = (RESULT *)malloc(sizeof(RESULT) * result_size);
			gn.getResult(join_result);

			//printf("Size of result = %d\n", result_size);
			//printf("Start writing output...\n");
			for (int i = 0; i < result_size && (limit == -1 || tuple_ctr < limit); i++, tuple_ctr++) {
				int l = join_result[i].lkey;
				int r = join_result[i].rkey;
				if (l >= 0 && r >= 0 && l < outer_size && r < inner_size) {
					join_tuple.setNValues(0, tmp_outer_tuple[l], 0, num_of_outer_cols);
					//printf("i = %d; lkey = %d; rkey = %d\n", i, join_result[i].lkey, join_result[i].rkey);
					for (int col_ctr = num_of_outer_cols; col_ctr < join_tuple.sizeInValues(); ++col_ctr) {
						//std::cout << m_outputExpressions[col_ctr]->debug() << std::endl;;
						join_tuple.setNValue(col_ctr,
								  m_outputExpressions[col_ctr]->eval(&tmp_outer_tuple[l], &tmp_inner_tuple[r]));
					}
				}

                if (m_aggExec != NULL) {
                    if (m_aggExec->p_execute_tuple(join_tuple)) {
                        // Get enough rows for LIMIT
                        //earlyReturned = true;
                        break;
                    }
                } else {
                    m_tmpOutputTable->insertTempTuple(join_tuple);
                    pmp.countdownProgress();
                }
			}
		}
	}

	printf("Elapsed time for joining: %ld\n", (join_end.tv_sec - join_start.tv_sec) * 1000000 + (join_end.tv_usec - join_start.tv_usec));
    /******************* End of adding GPU join ********************/


//    VOLT_TRACE("<num_of_outer_cols>: %d\n", num_of_outer_cols);
//    while ((limit == -1 || tuple_ctr < limit) && outer_iterator.next(outer_tuple)) {
//        VOLT_TRACE("outer_tuple:%s",
//                   outer_tuple.debug(outer_table->name()).c_str());
//        pmp.countdownProgress();
//        // Set the outer tuple columns. Must be outside the inner loop
//        // in case of the empty inner table
//        join_tuple.setNValues(0, outer_tuple, 0, num_of_outer_cols);
//
//        // did this loop body find at least one match for this tuple?
//        bool match = false;
//        // For outer joins if outer tuple fails pre-join predicate
//        // (join expression based on the outer table only)
//        // it can't match any of inner tuples
//        if (prejoin_expression == NULL || prejoin_expression->eval(&outer_tuple, NULL).isTrue()) {
//            int activeNumOfSearchKeys = num_of_searchkeys;
//            VOLT_TRACE ("<Nested Loop Index exec, WHILE-LOOP...> Number of searchKeys: %d \n", num_of_searchkeys);
//            IndexLookupType localLookupType = m_lookupType;
//            SortDirectionType localSortDirection = m_sortDirection;
//            VOLT_TRACE("Lookup type: %d\n", m_lookupType);
//            VOLT_TRACE("SortDirectionType: %d\n", m_sortDirection);
//
//            // did setting the search key fail (usually due to overflow)
//            bool keyException = false;
//            //
//            // Now use the outer table tuple to construct the search key
//            // against the inner table
//            //
//            const TableTuple& index_values = m_indexValues.tuple();
//            index_values.setAllNulls();
//            for (int ctr = 0; ctr < activeNumOfSearchKeys; ctr++) {
//                // in a normal index scan, params would be substituted here,
//                // but this scan fills in params outside the loop
//            	//std::cout << "Find candidate value for search index" << std::endl;
//                NValue candidateValue = m_indexNode->getSearchKeyExpressions()[ctr]->eval(&outer_tuple, NULL);
//                //std::cout << "End of finding candidate value for search index"  << std::endl;
//                try {
//                    index_values.setNValue(ctr, candidateValue);
//                }
//                catch (const SQLException &e) {
//                    // This next bit of logic handles underflow and overflow while
//                    // setting up the search keys.
//                    // e.g. TINYINT > 200 or INT <= 6000000000
//
//                    // re-throw if not an overflow or underflow
//                    // currently, it's expected to always be an overflow or underflow
//                    if ((e.getInternalFlags() & (SQLException::TYPE_OVERFLOW | SQLException::TYPE_UNDERFLOW)) == 0) {
//                        throw e;
//                    }
//
//                    // handle the case where this is a comparison, rather than equality match
//                    // comparison is the only place where the executor might return matching tuples
//                    // e.g. TINYINT < 1000 should return all values
//                    if ((localLookupType != INDEX_LOOKUP_TYPE_EQ) &&
//                        (ctr == (activeNumOfSearchKeys - 1))) {
//
//                        if (e.getInternalFlags() & SQLException::TYPE_OVERFLOW) {
//                            if ((localLookupType == INDEX_LOOKUP_TYPE_GT) ||
//                                (localLookupType == INDEX_LOOKUP_TYPE_GTE)) {
//
//                                // gt or gte when key overflows breaks out
//                                // and only returns for left-outer
//                                keyException = true;
//                                break; // the outer while loop
//                            }
//                            else {
//                                // overflow of LT or LTE should be treated as LTE
//                                // to issue an "initial" forward scan
//                                localLookupType = INDEX_LOOKUP_TYPE_LTE;
//                            }
//                        }
//                        if (e.getInternalFlags() & SQLException::TYPE_UNDERFLOW) {
//                            if ((localLookupType == INDEX_LOOKUP_TYPE_LT) ||
//                                (localLookupType == INDEX_LOOKUP_TYPE_LTE)) {
//                                // overflow of LT or LTE should be treated as LTE
//                                // to issue an "initial" forward scans
//                                localLookupType = INDEX_LOOKUP_TYPE_LTE;
//                            }
//                            else {
//                                // don't allow GTE because it breaks null handling
//                                localLookupType = INDEX_LOOKUP_TYPE_GT;
//                            }
//                        }
//
//                        // if here, means all tuples with the previous searchkey
//                        // columns need to be scaned.
//                        activeNumOfSearchKeys--;
//                        if (localSortDirection == SORT_DIRECTION_TYPE_INVALID) {
//                            localSortDirection = SORT_DIRECTION_TYPE_ASC;
//                        }
//                    }
//                    // if a EQ comparison is out of range, then the tuple from
//                    // the outer loop returns no matches (except left-outer)
//                    else {
//                        keyException = true;
//                    }
//                    break;
//                } // End catch block for under- or overflow when setting index key
//            } // End for each active search key
//            VOLT_TRACE("Searching %s", index_values.debug("").c_str());
//            //std::cout << "Index values = " << index_values.debug("").c_str() << std::endl;
//
//            //std::cout << "End of generating search keys" << std::endl;
//            // if a search value didn't fit into the targeted index key, skip this key
//            if (!keyException) {
//                //
//                // Our index scan on the inner table is going to have three parts:
//                //  (1) Lookup tuples using the search key
//                //
//                //  (2) For each tuple that comes back, check whether the
//                //      end_expression is false.  If it is, then we stop
//                //      scanning. Otherwise...
//                //
//                //  (3) Check whether the tuple satisfies the post expression.
//                //      If it does, then add it to the output table
//                //
//                // Use our search key to prime the index iterator
//                // The loop through each tuple given to us by the iterator
//                //
//                // Essentially cut and pasted this if ladder from
//                // index scan executor
//                if (num_of_searchkeys > 0)
//                {
//                    if (localLookupType == INDEX_LOOKUP_TYPE_EQ) {
//                    	//cout << "Move to Key" << endl;
//                        index->moveToKey(&index_values, indexCursor);
//                    }
//                    else if (localLookupType == INDEX_LOOKUP_TYPE_GT) {
//                    	//cout << "Move to Greater than key" << endl;
//                        index->moveToGreaterThanKey(&index_values, indexCursor);
//                    }
//                    else if (localLookupType == INDEX_LOOKUP_TYPE_GTE) {
//                    	//cout << "Move to Key or Greater" << endl;
//                        index->moveToKeyOrGreater(&index_values, indexCursor);
//                    }
//                    else if (localLookupType == INDEX_LOOKUP_TYPE_LT) {
//                    	//cout << "Move to Less than Key" << endl;
//                        index->moveToLessThanKey(&index_values, indexCursor);
//                    } else if (localLookupType == INDEX_LOOKUP_TYPE_LTE) {
//                    	//cout << "move to Greater Than Key" << endl;
//                        // find the entry whose key is greater than search key,
//                        // do a forward scan using initialExpr to find the correct
//                        // start point to do reverse scan
//                        bool isEnd = index->moveToGreaterThanKey(&index_values, indexCursor);
//                        if (isEnd) {
//                            index->moveToEnd(false, indexCursor);
//                        } else {
//                            while (!(inner_tuple = index->nextValue(indexCursor)).isNullTuple()) {
//                                pmp.countdownProgress();
//                                if (initial_expression != NULL && !initial_expression->eval(&outer_tuple, &inner_tuple).isTrue()) {
//                                    // just passed the first failed entry, so move 2 backward
//                                    index->moveToBeforePriorEntry(indexCursor);
//                                    break;
//                                }
//                            }
//                            if (inner_tuple.isNullTuple()) {
//                                index->moveToEnd(false, indexCursor);
//                            }
//                        }
//                    }
//                    else {
//                        return false;
//                    }
//                } else {
//                    bool toStartActually = (localSortDirection != SORT_DIRECTION_TYPE_DESC);
//                    index->moveToEnd(toStartActually, indexCursor);
//                }
//
//                AbstractExpression* skipNullExprIteration = skipNullExpr;
//
//                while ((limit == -1 || tuple_ctr < limit) &&
//                       ((localLookupType == INDEX_LOOKUP_TYPE_EQ &&
//                        !(inner_tuple = index->nextValueAtKey(indexCursor)).isNullTuple()) ||
//                       ((localLookupType != INDEX_LOOKUP_TYPE_EQ || num_of_searchkeys == 0) &&
//                        !(inner_tuple = index->nextValue(indexCursor)).isNullTuple())))
//                {
//                    VOLT_TRACE("inner_tuple:%s",
//                               inner_tuple.debug(inner_table->name()).c_str());
//                    pmp.countdownProgress();
//
//                    //
//                    // First check to eliminate the null index rows for UNDERFLOW case only
//                    //
//                    if (skipNullExprIteration != NULL) {
//                        if (skipNullExprIteration->eval(&outer_tuple, &inner_tuple).isTrue()) {
//                            VOLT_DEBUG("Index scan: find out null rows or columns.");
//                            continue;
//                        } else {
//                            skipNullExprIteration = NULL;
//                        }
//                    }
//
//                    //
//                    // First check whether the end_expression is now false
//                    //
//                    if (end_expression != NULL &&
//                        !end_expression->eval(&outer_tuple, &inner_tuple).isTrue())
//                    {
//                        VOLT_TRACE("End Expression evaluated to false, stopping scan\n");
//                        break;
//                    }
//                    //
//                    // Then apply our post-predicate to do further filtering
//                    //
//                    if (post_expression == NULL ||
//                        post_expression->eval(&outer_tuple, &inner_tuple).isTrue())
//                    {
//                        match = true;
//                        // Still need to pass where filtering
//                        if (where_expression == NULL || where_expression->eval(&outer_tuple, &inner_tuple).isTrue()) {
//                            // Check if we have to skip this tuple because of offset
//                            if (tuple_skipped < offset) {
//                                tuple_skipped++;
//                                continue;
//                            }
//                            ++tuple_ctr;
//                            //
//                            // Try to put the tuple into our output table
//                            // Append the inner values to the end of our join tuple
//                            //
//                            for (int col_ctr = num_of_outer_cols;
//                                 col_ctr < join_tuple.sizeInValues();
//                                 ++col_ctr)
//                            {
//                                // For the sake of consistency, we don't try to do
//                                // output expressions here with columns from both tables.
//                                join_tuple.setNValue(col_ctr,
//                                          m_outputExpressions[col_ctr]->eval(&outer_tuple, &inner_tuple));
//                            }
//                            VOLT_TRACE("join_tuple tuple: %s",
//                                       join_tuple.debug(m_tmpOutputTable->name()).c_str());
//                            VOLT_TRACE("MATCH: %s",
//                                   join_tuple.debug(m_tmpOutputTable->name()).c_str());
//
//                            if (m_aggExec != NULL) {
//                                if (m_aggExec->p_execute_tuple(join_tuple)) {
//                                    // Get enough rows for LIMIT
//                                    earlyReturned = true;
//                                    break;
//                                }
//                            } else {
//                                m_tmpOutputTable->insertTempTuple(join_tuple);
//                                pmp.countdownProgress();
//                            }
//
//                        }
//                    }
//                } // END INNER WHILE LOOP
//
//                if (earlyReturned) {
//                    break;
//                }
//            } // END IF INDEX KEY EXCEPTION CONDITION
//        } // END IF PRE JOIN CONDITION
//
//        //
//        // Left Outer Join
//        //
//        if (m_joinType == JOIN_TYPE_LEFT && !match
//                && (limit == -1 || tuple_ctr < limit) )
//        {
//            if (where_expression == NULL || where_expression->eval(&outer_tuple, &null_tuple).isTrue()) {
//                // Check if we have to skip this tuple because of offset
//                if (tuple_skipped < offset) {
//                    tuple_skipped++;
//                    continue;
//                }
//                ++tuple_ctr;
//                join_tuple.setNValues(num_of_outer_cols, m_null_tuple.tuple(), 0, num_of_inner_cols);
//
//                if (m_aggExec != NULL) {
//                    if (m_aggExec->p_execute_tuple(join_tuple)) {
//                        // Get enough rows for LIMIT
//                        earlyReturned = true;
//                        break;
//                    }
//                } else {
//                    m_tmpOutputTable->insertTempTuple(join_tuple);
//                    pmp.countdownProgress();
//                }
//            }
//        }
//    } // END OUTER WHILE LOOP

    if (m_aggExec != NULL) {
        m_aggExec->p_execute_finish();
    }

    VOLT_TRACE ("result table:\n %s", m_tmpOutputTable->debug().c_str());
    VOLT_TRACE("Finished NestLoopIndex");

    cleanupInputTempTable(inner_table);
    cleanupInputTempTable(outer_table);

    //printf("End of JOIN\n");
    if (outer_size != 0) {
    	free(index_data_out);
    	free(tmp_outer_tuple);
    }
    if (inner_size != 0) {
    	free(index_data_in);
    	free(tmp_inner_tuple);
    }

    if (outer_size != 0 && inner_size != 0 && result_size != 0) {
    	free(join_result);
    }

    gettimeofday(&finish, NULL);

    printf("Elapsed time: %lu microseconds\n", ((finish.tv_sec - start.tv_sec) * 1000000 + finish.tv_usec - start.tv_usec));

    return (true);
}

NestLoopIndexExecutor::~NestLoopIndexExecutor() { }
