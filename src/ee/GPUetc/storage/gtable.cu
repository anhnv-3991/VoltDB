#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "gtable.h"
#include "GPUetc/common/GPUTUPLE.h"

namespace voltdb {

GTable::GTable() {
	database_id_ = 0;
	name_ = NULL;
	block_list_host_ = NULL;
	schema_ = NULL;
	columns_ = 0;
	rows_ = 0;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;
}

GTable::GTable(int database_id, char *name, int column_num)
{
	database_id_ = database_id;
	name_ = name;
	block_list_host_ = NULL;
	columns_ = column_num;
	rows_ = 0;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;

	block_list_host_ = (GBlock *)malloc(sizeof(GBlock));
	checkCudaErrors(cudaMalloc(&block_list_host_[0].data, sizeof(int64_t) * MAX_BLOCK_SIZE));
	checkCudaErrors(cudaMalloc(&schema_, sizeof(GColumnInfo) * column_num));
	block_list_host_[0].rows = 0;
	block_list_host_[0].columns = column_num;
	block_list_host_[0].block_size = MAX_BLOCK_SIZE;
}

GTable::GTable(int database_id, char *name, GColumnInfo *schema, int column_num, int rows)
{
	database_id_ = database_id;
	name_ = name;
	block_list_host_ = NULL;
	columns_ = column_num;
	rows_ = rows;
	block_num_ = 0;
	indexes_ = NULL;
	index_num_ = 0;

	block_list_host_ = (GBlock *)malloc(sizeof(GBlock));
	checkCudaErrors(cudaMalloc(&block_list_host_[0].data, sizeof(int64_t) * MAX_BLOCK_SIZE));
	checkCudaErrors(cudaMalloc(&schema_, sizeof(GColumnInfo) * column_num));
	checkCudaErrors(cudaMemcpy(schema_, schema, sizeof(GColumnInfo) * column_num, cudaMemcpyHostToDevice));
	block_list_host_[0].rows = rows;
	block_list_host_[0].columns = column_num;
	block_list_host_[0].block_size = MAX_BLOCK_SIZE;
}

void GTable::deleteAllTuples()
{
	for (int i = 0; i < block_num_; i++) {
		checkCudaErrors(cudaFree(block_list_host_[i].data));
	}
	free(block_list_host_);
	block_num_ = 0;
	rows_ = 0;
}

void GTable::deleteTuple(int blockId, int tupleId)
{
	if (tupleId < 0 || tupleId > block_list_host_[blockId].rows) {
		printf("Error: tupleId out of range\n");
		return;
	}

	GBlock *target_block = block_list_host_ + blockId;
	int64_t *target_data = target_block->data;

	checkCudaErrors(cudaMemcpy(target_data + tupleId * columns_, target_data + (tupleId + 1) * columns_, (target_block->rows - tupleId) * columns_ * sizeof(int64_t), cudaMemcpyDeviceToDevice));
	target_block->rows -= 1;
}

void GTable::insertTuple(int64_t *tuple)
{
	int block_id, tuple_id;

	nextFreeTuple(&block_id, &tuple_id);

	int64_t *target_location = block_list_host_[block_id].data + tuple_id * columns_;

	checkCudaErrors(cudaMemcpy(target_location, tuple, columns_ * sizeof(int64_t), cudaMemcpyHostToDevice));
	block_list_host_[block_id].rows++;
	insertToAllIndexes(block_id, tuple_id);
}

void GTable::insertToAllIndexes(int block_id, int tuple_id)
{
	for (int i = 0; i < index_num_; i++) {
		insertToIndex(block_id, tuple_id, i);
	}
}

void GTable::insertToIndex(int block_id, int tuple_id, int index_id)
{
	return;
}

/* INCOMPLETED */
void GTable::addIndex(int *key_idx, int key_size, GIndexType type)
{
	indexes_ = (GIndex*)realloc(indexes_, sizeof(GIndex) * (index_num_ + 1));
	index_num_++;
}

void GTable::removeIndex()
{
	printf("Error: unsupported operation\n");
	exit(1);
}

void GTable::nextFreeTuple(int *block_id, int *tuple_id)
{
	for (int i = 0; i < block_num_; i++) {
		if (!isBlockFull(i)) {
			*block_id = i;
			*tuple_id = block_list_host_[i].rows;
			return;
		}
	}

	//All current blocks are full, allocate a new one
	GBlock new_block;

	checkCudaErrors(cudaMalloc(&new_block.data, MAX_BLOCK_SIZE));
	new_block.columns = columns_;
	new_block.rows = 0;
	new_block.block_size = MAX_BLOCK_SIZE;

	block_list_host_ = (GBlock*)realloc(block_list_host_, sizeof(GBlock) * (block_num_ + 1));
	block_num_++;
}
}
