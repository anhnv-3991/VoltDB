#ifndef TREE_INDEX_H_
#define TREE_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"

namespace voltdb {

class GTreeIndex;
extern class GTable;

/* Class for index keys.
 * Each index key contains multiple column values.
 * The value type of each value is indicated in schema_.
 */
class GTreeIndexKey: public GKeyIndex {
public:
	__forceinline__ __device__ GTreeIndexKey();

	/* Constructing object from individual tuple, schema, index schema.
	 * Private variables of this object are set to packed_key and packed_schema.
	 * Used for building keys from the index table.
	 */
	__forceinline__ __device__ GTreeIndexKey(int64_t *tuple, GColumnInfo *schema, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *packed_schema);

	/* Constructing object from a tuple and an index schema.
	 * Private variables of this object are set to packed_key and packed schema.
	 * Used for building keys from the index table.
	 */
	__forceinline__ __device__ GTreeIndexKey(GTuple tuple, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *packed_schema);


	/* Get key values at index key_idx from a list of key 'index'.
	 */
	__forceinline__ __device__ GTreeIndexKey(GTreeIndex index, int key_idx);

	/* Constructing object from a tuple without index schema.
	 * Iterate over columns of the tuple and copy the value of columns to the
	 * corresponding key.
	 * Used for building keys from the non-index table.
	 */
	__forceinline__ __device__ GTreeIndexKey(GTuple tuple, int64_t *packed_key, GColumnInfo *packed_schema);

	/* Extract a key from an array of key
	 */
	__forceinline__ __device__ GTreeIndexKey(int64_t *keys, GColumnInfo *schema, int key_size);

	/* Extract a key from a tuple */
	__forceinline__ __device__ GTreeIndexKey(GTuple tuple);

	/* Comparator for GTreeIndexKey objects.
	 * Iterate through keys and compare.
	 * Return -1 if left is smaller than right, 0 if equal, and 1 if larger.
	 */
	static __forceinline__ __device__ int KeyComparator(GTreeIndexKey left, GTreeIndexKey right);

	/* Insert a key value to the key tuple at a specified key column.
	 * The value of the key value is of type int64_t.
	 * This is used to construct key values of a tuple.
	 * The type of the key value is ignored.
	 */
	__forceinline__ __device__ void insertKeyValue(int64_t value, int key_col);

	__forceinline__ __device__ void setKey(GTreeIndex key_list, int index);

private:
	GColumnInfo *schema_;
};

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey()
{
	schema_ = NULL;
	size_ = 0;
	packed_key_ = NULL;
}

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(int64_t *tuple, GColumnInfo *schema, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *packed_schema)
{
	schema_ = packed_schema;
	size_ = key_size;
	packed_key_ = packed_key;

	for (int i = 0; i < key_size; i++) {
		packed_key_[i] = tuple[key_idx[i]];
		schema_[i] = schema[key_idx[i]];
	}
}

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(GTuple tuple, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *packed_schema)
{
	schema_ = packed_schema;
	size_ = key_size;
	packed_key_ = packed_key;

	for (int i = 0; i < key_size; i++) {
		packed_key_[i] = tuple.tuple_[key_idx[i]];
		schema_[i] = tuple.schema_[key_idx[i]];
	}
}

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(GTreeIndex index, int key_idx)
{
	schema_ = index.key_schema_;
	size_ = index.key_size_;
	packed_key_ = index.packed_key_ + size_ * index.sorted_idx_[key_idx];
}


__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(GTuple tuple, int64_t *packed_key, GColumnInfo *packed_schema)
{
	schema_ = packed_schema;
	size_ = tuple.columns_;
	packed_key_ = packed_key;

	for (int i = 0; i < size_; i++) {
		packed_key_[i] = tuple.tuple_[i];
		schema_[i] = tuple.schema_[i];
	}
}

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(GTuple tuple)
{
	packed_key_ = tuple.tuple_;
	schema_ = tuple.schema_;
	size_ = tuple.columns_;
}

__forceinline__ __device__ GTreeIndexKey::GTreeIndexKey(int64_t *keys, GColumnInfo *schema, int key_size)
{
	packed_key_ = keys;
	schema_ = schema;
	size_ = key_size;
}

static __forceinline__ __device__ int GTreeIndexKey::KeyComparator(GTreeIndexKey left, GTreeIndexKey right) {
	int64_t res_i = 0;
	double res_d = 0;
	ValueType left_type, right_type;

	for (int i = 0; i < right.size_ && res_i == 0 && res_d == 0; i++) {
		left_type = left.schema_[i].data_type;
		right_type = right.schema_[i].data_type;

		if (left_type != VALUE_TYPE_INVALID && right_type != VALUE_TYPE_INVALID
				&& left_type != VALUE_TYPE_NULL && right_type != VALUE_TYPE_NULL) {
			int64_t left_i = (left_type == VALUE_TYPE_DOUBLE) ? 0 : left.tuple_[i];
			int64_t right_i = (right_type == VALUE_TYPE_DOUBLE) ? 0 : right.tuple_[i];
			double left_d = (left.schema_[i].data_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(left_i) : static_cast<double>(left_i);
			double right_d = (right.schema_[i].data_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(right_i) : static_cast<double>(right_i);

			res_i = (left_type == VALUE_TYPE_DOUBLE || right_type == VALUE_TYPE_DOUBLE) ? 0 : (left_i - right_i);
			res_d = (left_type == VALUE_TYPE_DOUBLE || right_type == VALUE_TYPE_DOUBLE) ? (left_d - right_d) : 0;
		}
	}

	return (res_i > 0 || res_d > 0) ? 1 : ((res_i < 0 || res_d < 0) ? -1 : 0);
}

__forceinline__ __device__ void GTreeIndexKey::insertKeyValue(int64_t value, int key_col)
{
	packed_key_[key_col] = value;
}


__forceinline__ __device__ void GTreeIndexKey::setKey(GTreeIndex key_list, int index)
{
	packed_key_ = key_list + index * key_list.key_size_;
	schema_ = key_list.key_schema_;
}

/* Class for tree index.
 * Each index contains a list of key values and a sorted index array.
 * A schema array indicate the type of each key value.
 */
class GTreeIndex: public GIndex {
	friend class GTreeIndexKey;
public:
	GTreeIndex();
	GTreeIndex(GTable table, int *key_schema, int key_size);

	void addEntry(GTuple new_tuple);

	void addBatchEntry(GTable table, int start_idx, int size);

	void merge(int old_left, int old_right, int new_left, int new_right);

	__forceinline__ __device__ GColumnInfo *getSchema();

	__forceinline__ __device__ int getKeyNum();

	__forceinline__ __device__ int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx();

	__forceinline__ __device__ int getKeySize();

	__forceinline__ __device__ int lowerBound(GTreeIndexKey key, int left, int right);

	__forceinline__ __device__ int upperBound(GTreeIndexKey key, int left, int right);

	/* Insert key values of a tuple to the 'location' of the key list 'packed_key_'.
	 */
	__forceinline__ __device__ void insertKeyTupleNoSort(GTuple tuple, int location);
	__forceinline__ __device__ void swap(int left, int right);

protected:
	int64_t *packed_key_;
	GColumnInfo *key_schema_;	// Schemas of columns selected as keys
};


__forceinline__ __device__ GColumnInfo *GTreeIndex::getSchema() {
	return schema_;
}


__forceinline__ __device__ int GTreeIndex::getKeyNum() {
	return key_num_;
}

__forceinline__ __device__ int *GTreeIndex::getSortedIdx() {
	return sorted_idx_;
}

__forceinline__ __device__ int *GTreeIndex::getKeyIdx() {
	return key_idx_;
}

__forceinline__ __device__ int GTreeIndex::getKeySize() {
	return key_size_;
}

__forceinline__ __device__ int GTreeIndex::lowerBound(GTreeIndexKey key, int left, int right) {
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(*this, middle);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res > 0) ? (middle + 1) : left;
		result = (compare_res <= 0) ? middle : result;
	}
	return result;
}


__forceinline__ __device__ int GTreeIndex::upperBound(GTreeIndexKey key, int left, int right) {
	int middle = -1;
	int result = key_num_ - 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(*this, middle);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res >= 0) ? (middle + 1) : left;
		result = (compare_res < 0) ? middle : result;
	}

	return result;
}

__forceinline__ __device__ void GTreeIndex::insertKeyTupleNoSort(GTuple tuple, int location)
{
	for (int i = 0; i < key_size_; i++) {
		packed_key_[location * key_size_ + i] = tuple.tuple_[key_idx_[i]];
	}
	sorted_idx_[location] = location;
}

__forceinline__ __device__ void GTreeIndex::swap(int left, int right)
{
	int tmp = sorted_idx_[left];

	sorted_idx_[left] = sorted_idx_[right];
	sorted_idx_[right] = tmp;
}

}

#endif
