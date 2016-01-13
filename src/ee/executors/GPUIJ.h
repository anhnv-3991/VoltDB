#ifndef GPUIJ_H
#define GPUIJ_H

#include <cuda.h>
#include "GPUTUPLE.h"
#include "GPUetc/common/GNValue.h"
#include "GPUetc/expressions/treeexpression.h"
#include "GPUetc/expressions/nodedata.h"

using namespace voltdb;

#define DEFAULT_PART_SIZE_ (256 * 1024)
#define PART_SIZE_ 1024
class GPUIJ {
public:
	GPUIJ();

	GPUIJ(IndexData *outer_table,
			IndexData *inner_table,
			int outer_size,
			int inner_size,
			std::vector<int> search_idx,
			std::vector<int> indices,
			TreeExpression end_expression,
			TreeExpression post_expression,
			TreeExpression initial_expression,
			TreeExpression skipNullExpr,
			TreeExpression prejoin_expression,
			TreeExpression where_expression);

	~GPUIJ();

	bool join();

	void getResult(RESULT *output) const;

	int getResultSize() const;

	void debug();

private:
	IndexData *outer_table_;
	IndexData *inner_table_;
	int outer_size_, inner_size_;
	RESULT *join_result_;
	int *search_keys_, *indices_;
	int result_size_;
	int end_size_, post_size_, initial_size_, skipNull_size_, prejoin_size_, where_size_, search_keys_size_, indices_size_;


	GTreeNode *end_expression_;
	GTreeNode *post_expression_;
	GTreeNode *initial_expression_;
	GTreeNode *skipNullExpr_;
	GTreeNode *prejoin_expression_;
	GTreeNode *where_expression_;

	uint getPartitionSize() const;
	uint divUtility(uint divident, uint divisor) const;
	bool getTreeNodes(GTreeNode **expression, const TreeExpression tree_expression);
	template <typename T> void freeArrays(T *expression);
	void setNValue(NValue *nvalue, GNValue &gnvalue);
	void debugGTrees(const GTreeNode *expression, int size);
	void GNValueDebug(GNValue &column_data)
	{
		NValue value;
		value.setMdataFromGPU(column_data.getMdata());
		value.setSourceInlinedFromGPU(column_data.getSourceInlined());
		value.setValueTypeFromGPU(column_data.getValueType());

		std::cout << value.debug();
	}

	CUDAH int binarySearchIdxTest(int *search_key_indices,
									int search_key_size,
									int *key_indices,
									int key_index_size,
									IndexData search_key,
									IndexData *search_array,
									int size_of_search_array)
	{
		int left = 0, right = size_of_search_array - 1;

		int middle = -1, res, i, search_idx, key_idx;
		GNValue tmp_search, tmp_idx;

		while (left <= right && left >= 0) {
			res = 0;
			middle = (left + right)/2;

			if (middle < 0 || middle >= size_of_search_array)
				break;

			for (i = 0; i < search_key_size; i++) {
				search_idx = search_key_indices[i];
				if (search_idx >= MAX_GNVALUE)
					return -1;
				tmp_search = search_key.gn[search_idx];
				printf("Search = ");
				GNValueDebug(tmp_search);
				printf(";;;");

				if (i >= key_index_size)
					return -1;
				key_idx = key_indices[i];
				if (key_idx >= MAX_GNVALUE)
					return -1;
				tmp_idx = search_array[middle].gn[key_idx];
				printf("Index = ");
				GNValueDebug(tmp_idx);
				printf("\n");

				res = tmp_search.compare_withoutNull(tmp_idx);
				if (res != 0)
					break;
			}

			if (res < 0) {
				right = middle - 1;
				middle = -1;
			}
			else if (res > 0) {
				left = middle + 1;
				middle = -1;
			}
			else {
				break;
			}
		}

		return middle;
	}
};

#endif
