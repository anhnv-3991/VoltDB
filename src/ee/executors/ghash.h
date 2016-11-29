#ifndef GHASH_H_
#define GHASH_H_

#include "GPUTUPLE.h"
#include "common/types.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/cudaheader.h"
#include "GPUetc/expressions/nodedata.h"

using namespace voltdb;

extern "C" {
void packKeyWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					GNValue *index_table,
					int tuple_num,
					int col_num,
					int *indices,
					int index_num,
					uint64_t *packedKey,
					int keySize);

void ghashCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						uint64_t *packedKey,
						int keyNum,
						int keySize,
						ulong *hashCount,
						uint64_t maxNumberOfBuckets
						);

void ghashWrapper(int block_x, int block_y,
					int grid_x, int grid_y,
					uint64_t *packedKey,
					ulong *hashCount,
					GHashNode hashTable
					);

void packSearchKeyWrapper(int block_x, int block_y,
							int grid_x, int grid_y,
							GNValue *outer_table, int outer_rows, int outer_cols,
							uint64_t *searchPackedKey, GTreeNode *searchKeyExp,
							int *searchKeySize, int searchExpNum,
							int keySize,
#ifdef FUNC_CALL_
							GNValue *stack
#else
							int64_t *val_stack,
							ValueType *type_stack
#endif
							);

void indexCountWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size
						);

void hashJoinWrapper(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table,
						GNValue *inner_table,
						int outer_cols,
						int inner_cols,
						GTreeNode *end_expression,
						int end_size,
						GTreeNode *post_expression,
						int post_size,
						GHashNode outerHash,
						GHashNode innerHash,
						int lowerBound,
						int upperBound,
						ulong *indexCount,
						int size,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result
						);

void ghashPhysicalWrapper(int block_x, int block_y, int grid_x, int grid_y,
							GNValue *inputTable, GNValue *outputTable,
							int colNum, int rowNum, GHashNode hashTable);

void hashPhysicalJoinWrapper(int block_x, int block_y,
								int grid_x, int grid_y,
								GNValue *outer_table,
								GNValue *inner_table,
								int outer_cols,
								int inner_cols,
								GTreeNode *end_expression,
								int end_size,
								GTreeNode *post_expression,
								int post_size,
								GHashNode outerHash,
								GHashNode innerHash,
								int lowerBound,
								int upperBound,
								ulong *indexCount,
								int size,
#ifdef FUNC_CALL_
								GNValue *stack,
#else
								int64_t *val_stack,
								ValueType *type_stack,
#endif
								RESULT *result
								);

void hashJoinWrapper2(int block_x, int block_y, int grid_x, int grid_y,
						GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_expression, int end_size,
						GTreeNode *post_expression, int post_size,
						GHashNode outerHash, GHashNode innerHash,
						int baseOuterIdx, int baseInnerIdx,
						ulong *indexCount, int partitionSize,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result);

void hashJoinWrapper3(int block_x, int block_y,
						int grid_x, int grid_y,
						GNValue *outer_table, GNValue *inner_table,
						int outer_cols, int inner_cols,
						GTreeNode *end_expression, int end_size,
						GTreeNode *post_expression, int post_size,
						GHashNode outerHash, GHashNode innerHash,
						int lowerBound, int upperBound,
						int outerBaseIdx, int innerBaseIdx,
						ulong *indexCount, int size,
#ifdef FUNC_CALL_
						GNValue *stack,
#else
						int64_t *val_stack,
						ValueType *type_stack,
#endif
						RESULT *result
						);

void indexCountLegacyWrapper(int block_x, int block_y,
								int grid_x, int grid_y,
								uint64_t *outerKey,
								int outer_rows,
								GHashNode innerHash,
								ulong *indexCount,
								int size);

void hashJoinLegacyWrapper(int block_x, int block_y, int grid_x, int grid_y,
							GNValue *outer_table, GNValue *inner_table,
							int outer_cols, int inner_cols,
							int outer_rows,
							uint64_t *outerKey,
							GTreeNode *end_expression, int end_size,
							GTreeNode *post_expression,	int post_size,
							GHashNode innerHash,
							int baseOuterIdx, int baseInnerIdx,
							ulong *indexCount, int size,
#ifdef FUNC_CALL_
							GNValue *stack,
#else
							int64_t *val_stack,
							ValueType *type_stack,
#endif
							RESULT *result);


void hprefixSumWrapper(ulong *input, int ele_num, ulong *sum);
}

#endif
