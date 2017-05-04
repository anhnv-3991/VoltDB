#ifndef GTABLE_H_
#define GTABLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "GPUetc/indexes/TreeIndex.h"
#include "GPUetc/indexes/HashIndex.h"
#include "GPUetc/indexes/KeyIndex.h"


namespace voltdb {

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

	/* Allocate an empty table */
	GTable();

	/* Allocate a table.
	 * Also preallocate buffers on the GPU memory
	 * for the table and the table schema
	 */
	GTable(int database_id, char *name, int column_num);

	/* Allocate a table.
	 * Also preallocate buffers on the GPU memory
	 * for the table data and the table schema.
	 * Copy schema from the host memory to the
	 * schema on the GPU memory.
	 */
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
		return block_list_host_[blockId].rows;
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

	void removeTable() {
		checkCudaErrors(cudaFree(schema_));

		for (int i = 0; i < block_num_; i++) {
			checkCudaErrors(cudaFree(block_list_host_[i].data));
		}
		if (block_num_ > 0)
			free(block_list_host_);

		for (int i = 0; i < index_num_; i++) {
			indexes_[i].removeIndex();
		}
	}

	GBlock *getBlock(int blockId) {
		return block_list_host_ + blockId;
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

	int getCurrentRowNum() const {
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
		index_ = indexes_[0];
	}

	__forceinline__ __device__ GTuple getGTuple(int index) {
		return GTuple(block_dev_.data + columns_ * index, schema_, columns_);
	}

protected:
	GColumnInfo *schema_;
	GBlock block_dev_;
	GIndex index_;
	int columns_;

private:
	void nextFreeTuple(int *blockId, int *tupleId);
	void insertToIndex(int blockId, int tupleId, GIndex *index);


	int database_id_;
	char *name_;

	GBlock *block_list_host_;
	int rows_;
	int block_num_;
	GIndex *indexes_;
	int index_num_;
};
}

#endif
