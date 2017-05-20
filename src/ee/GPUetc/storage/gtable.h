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

class GTable {
	friend class GTuple;
	friend class GTreeIndex;
	friend class GHashIndex;

public:
	typedef enum {
		GTREE_INDEX_,
		GHASH_INDEX_,
	} GIndexType;

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

	/*****************************
	 * Host-side functions
	 *****************************/
	void addBlock();

	void removeBlock(int block_id);

	void removeTable();

	GBlock *getBlock(int blockId);

	GTreeIndex *getCurrentIndex();

	int getBlockNum();

	int getIndexCount();

	char *getTableName();

	int getDatabaseId();

	int getBlockTuple(int block_id);

	bool isBlockFull(int block_id);

	int getCurrentRowNum() const;

	void deleteAllTuples();

	void deleteTuple(int blockId, int tupleId);

	void insertTuple(int64_t *tuple);

	void insertToAllIndexes(int blockId, int tupleId);

	void addIndex(int *key_idx, int key_size, GIndexType type);

	void removeIndex();

	void moveToIndex(int idx);

	/********************************
	 * Device-side functions
	 *******************************/
	__forceinline__ __device__ GColumnInfo *getSchema() {
		return schema_;
	}

	__forceinline__ __device__ GBlock getBlock() {
		return block_dev_;
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

	__forceinline__ __device__ GTuple getGTuple(int index) {
		return GTuple(block_dev_.data + columns_ * index, schema_, columns_);
	}

protected:
	GColumnInfo *schema_;
	GBlock block_dev_;
	GTreeIndex *index_;
	int columns_;

private:
	void nextFreeTuple(int *blockId, int *tupleId);
	void insertToIndex(int block_id, int tuple_id, int index_id);


	int database_id_;
	char *name_;

	GBlock *block_list_host_;
	int rows_;
	int block_num_;
	GTreeIndex *indexes_;
	int index_num_;
};
}

#endif
