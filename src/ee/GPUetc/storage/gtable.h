#ifndef GTABLE_H_
#define GTABLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/indexes/TreeIndex.h"
#include "GPUetc/indexes/HashIndex.h"
#include "GPUetc/indexes/KeyIndex.h"


namespace voltdb {

class GTable;

class GTuple {
	friend class GKeyIndex;
	friend class GTreeIndexKey;
	friend class GHashIndexKey;
	friend class GTreeIndex;
	friend class GExpression;
public:
	__forceinline__ __device__ GTuple();
	__forceinline__ __device__ GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns);
	__forceinline__ __device__ GTuple(GTable table, int tuple_idx);

	__forceinline__ __device__ bool attachColumn(GNValue new_value);


protected:
	int64_t *tuple_;
	GColumnInfo *schema_;
	int columns_;
	int max_columns_;
};

__forceinline__ __device__ GTuple::GTuple()
{
	tuple_ = NULL;
	schema_ = NULL;
	columns_ = 0;
	max_columns_ = 0;
}


__forceinline__ __device__ GTuple::GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns)
{
	tuple_ = tuple;
	schema_ = schema_buff;
	columns_ = 0;
	max_columns_ = 0;
}

__forceinline__ __device__ GTuple::GTuple(GTable table, int rows_index)
{
	columns_ = 0;
	max_columns_ = table.block_dev_.columns;
	schema_ = table.schema_;
	tuple_ = table.block_dev_.data + columns * rows_index;
}


__forceinline__ __device__ bool GTuple::attachColumn(GNValue new_value)
{
	if (columns_ < max_columns_) {
		tuple_[columns_] = new_value.getValue();
		schema_[columns_].data_type = new_value.getValueType();
		columns_++;

		return true;
	}

	return false;
}

class GTable {
	friend class GTuple;
	friend class GTreeIndex;
	friend class GHashIndex;

public:
	typedef struct {
		int64_t *data;
		int rows;
		int columns;
		int block_size;
	} GBlock;

	GTable();

	GTable(int database_id, char *name, GColumnInfo *schema, int column_num);
	GTable(int database_id, char *name, GColumnInfo *schema, int column_num, int rows);

	/********************************
	 * Device-side functions
	 *******************************/
	__forceinline__ __device__ GColumnInfo *getSchema() {
		return schema_;
	}

	__forceinline__ __device__ GBlock getBlock() {
		return block_dev_;
	}

	GIndex getIndex() {
		return index_;
	}


	__forceinline__ __host__ __device__ int getColumnCount() {
		return columns_;
	}

	__forceinline__ __device__ int getTupleCount() {
		return rows_;
	}

	__forceinline__ __device__ int getBlockTupleCount(int blockId) {
		return block_list_[blockId].rows;
	}

	/*****************************
	 * Host-side functions
	 *****************************/
	void addBlock() {
		block_list_host_ = (GBlock*)realloc(block_list_host_, block_num_ + 1);

		checkCudaErrors(cudaMalloc(&block_list_host_[block_num_].data, MAX_BLOCK_SIZE));
		block_list_host_[block_num_].columns = columns_;
		block_list_host_[block_num_].rows = 0;
		block_list_host_[block_num_].block_size = MAX_BLOCK_SIZE;
		block_num_++;
	}

	void removeBlock(int block_id) {
		if (block_id < block_num_) {
			checkCudaErrors(cudaFree(block_list_host_[block_id].data));
			memcpy(block_list_host_ + block_id, block_list_host_ + block_id + 1, sizeof(GBlock) * (block_num_ - block_id));
			free(block_list_host_ + block_num_ - 1);
		}
	}

	GBlock *getBlock(int blockId) {
		return block_list_ + blockId;
	}

	int getBlockNum() {
		return block_num_;
	}

	int getIndexCount() {
		return index_num_;
	}

	char *getTableName() {
		return name_;
	}

	int getDatabaseId() {
		return database_id_;
	}

	int getBlockTuple(int block_id) {
		assert(block_id < block_num_);

		return block_list_host_[block_id].rows;
	}

	bool isBlockFull(int block_id) {
		return (block_list_host_[block_id].rows >= block_list_host_[block_id].block_size/(columns_ * sizeof(int64_t)));
	}

	int getCurrentRowNum() {
		return block_dev_.rows;
	}

	void deleteAllTuples();
	void deleteTuple(int blockId, int tupleId);
	void insertTuple(int64_t *tuple);
	void insertToAllIndexes(int blockId, int tupleId);

	void addIndex(int *key_idx, int key_size);
	void removeIndex();

	void moveToIndex(int idx) {
		assert(idx < block_num_);
		block_dev_ = block_list_host_[idx];
		index_ = indexes_[0].block_indexes[idx];
	}


private:
	void nextFreeTuple(int *blockId, int *tupleId);
	void insertToIndex(int blockId, int tupleId, GIndex *index);


	int database_id_;
	char *name_;

	GBlock *block_list_host_;
	int columns_;
	int rows_;
	int block_num_;
	GIndex *indexes_;
	int index_num_;

protected:
	GColumnInfo *schema_;
	GBlock block_dev_;
	GIndex index_;
};
}

#endif
