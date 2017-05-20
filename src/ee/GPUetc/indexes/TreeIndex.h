#ifndef TREE_INDEX_H_
#define TREE_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"
#include "GPUetc/common/nodedata.h"
#include "GPUetc/common/GPUTUPLE.h"
#include "Index.h"
#include "GPUetc/storage/gtuple.h"

namespace voltdb {

/* Class for index keys.
 * Each index key contains multiple column values.
 * The value type of each value is indicated in schema_.
 */

class GTreeIndexKey: public GKeyIndex {
public:
	__forceinline__ __device__ GTreeIndexKey() {
		schema_ = NULL;
		size_ = 0;
		packed_key_ = NULL;
	}

	/* Construct a key object from a packed key buffer an a schema buffer.
	 */
	__forceinline__ __device__ GTreeIndexKey(int64_t *packed_key, GColumnInfo *schema, int key_size) {
		packed_key_ = packed_key;
		schema_ = schema;
		size_ = key_size;
	}

	__forceinline__ __device__ void createKey(int64_t *tuple, GColumnInfo *schema, int *key_schema, int key_size) {
		for (int i = 0; i < key_size; i++) {
			packed_key_[i] = tuple[key_schema[i]];
			schema_[i] = schema[key_schema[i]];
		}
	}

	__forceinline__ __device__ void createKey(GTuple tuple, int *key_schema, int key_size) {
		for (int i = 0; i < key_size; i++) {
			packed_key_[i] = tuple.tuple_[key_schema[i]];
			schema_[i] = tuple.schema_[key_schema[i]];
		}
	}

	__forceinline__ __device__ void createKey(int64_t *tuple, GColumnInfo *schema) {
		for (int i = 0; i < size_; i++) {
			packed_key_[i] = tuple[i];
			schema_[i] = schema[i];
		}
	}

	__forceinline__ __device__ void createKey(GTuple tuple) {
		for (int i = 0; i < size_; i++) {
			packed_key_[i] = tuple.tuple_[i];
			schema_[i] = tuple.schema_[i];
		}
	}

	static __forceinline__ __device__ int KeyComparator(GTreeIndexKey left, GTreeIndexKey right) {
		int64_t res_i = 0;
		double res_d = 0;
		ValueType left_type, right_type;

		for (int i = 0; i < right.size_ && res_i == 0 && res_d == 0; i++) {
			left_type = left.schema_[i].data_type;
			right_type = right.schema_[i].data_type;

			if (left_type != VALUE_TYPE_INVALID && right_type != VALUE_TYPE_INVALID
					&& left_type != VALUE_TYPE_NULL && right_type != VALUE_TYPE_NULL) {
				int64_t left_i = (left_type == VALUE_TYPE_DOUBLE) ? 0 : left.packed_key_[i];
				int64_t right_i = (right_type == VALUE_TYPE_DOUBLE) ? 0 : right.packed_key_[i];
				double left_d = (left.schema_[i].data_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(left_i) : static_cast<double>(left_i);
				double right_d = (right.schema_[i].data_type == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(right_i) : static_cast<double>(right_i);

				res_i = (left_type == VALUE_TYPE_DOUBLE || right_type == VALUE_TYPE_DOUBLE) ? 0 : (left_i - right_i);
				res_d = (left_type == VALUE_TYPE_DOUBLE || right_type == VALUE_TYPE_DOUBLE) ? (left_d - right_d) : 0;
			}
		}

		return (res_i > 0 || res_d > 0) ? 1 : ((res_i < 0 || res_d < 0) ? -1 : 0);
	}

	/* Insert a key value to the key tuple at a specified key column.
	 * The value of the key value is of type int64_t.
	 * This is used to construct key values of a tuple.
	 * The type of the key value is ignored.
	 */
	__forceinline__ __device__ void insertKeyValue(int64_t value, int key_col) {
		packed_key_[key_col] = value;
	}


private:
	GColumnInfo *schema_;
	int64_t *packed_key_;
};

/* Class for tree index.
 * Each index contains a list of key values and a sorted index array.
 * A schema array indicate the type of each key value.
 */
class GTreeIndex {
	friend class GTreeIndexKey;
public:
	GTreeIndex();
	GTreeIndex(int key_size, int key_num);
	GTreeIndex(int *sorted_idx, int *key_idx, int key_size, int64_t *packed_key, GColumnInfo *key_schema, int key_num);

	void createIndex(int64_t *table, GColumnInfo *schema, int rows, int columns);

	void addEntry(GTuple new_tuple);

	void addBatchEntry(int64_t *table, GColumnInfo *schema, int rows, int columns);

	void merge(int old_left, int old_right, int new_left, int new_right);


	__forceinline__ __device__ GTreeIndexKey getKeyAtSortedIndex(int key_index) {
		return GTreeIndexKey(packed_key_ + sorted_idx_[key_index] * key_size_, key_schema_, key_size_);
	}

	__forceinline__ __device__ GTreeIndexKey getKeyAtIndex(int key_index) {
		return GTreeIndexKey(packed_key_ + key_index * key_size_, key_schema_, key_size_);
	}

	__forceinline__ __device__ GColumnInfo *getSchema();

	__forceinline__ __device__ int getKeyNum();

	__forceinline__ __device__ int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx();

	__forceinline__ __device__ int getKeySize();

	__forceinline__ __device__ int64_t *getPackedKey();

	__forceinline__ __device__ int lowerBound(GTreeIndexKey key, int left, int right);

	__forceinline__ __device__ int upperBound(GTreeIndexKey key, int left, int right);

	__forceinline__ __device__ int lowerBound(GTreeIndexKey key, int root, int size, int stride);

	__forceinline__ __device__ int upperBound(GTreeIndexKey key, int root, int size, int stride);

	/* Insert key values of a tuple to the 'location' of the key list 'packed_key_'.
	 */
	__forceinline__ __device__ void insertKeyTupleNoSort(GTuple tuple, int location);
	__forceinline__ __device__ void swap(int left, int right);


	void removeIndex();
protected:
	int key_num_;	//Number of key values (equal to the number of rows)
	int *sorted_idx_;
	int *key_idx_;	// Index of columns selected as keys
	int key_size_;	// Number of columns selected as keys
	int64_t *packed_key_;
	GColumnInfo *key_schema_;	// Schemas of columns selected as keys
};

extern "C" __global__ void quickSort(GTreeIndex *indexes, int left, int right);

__forceinline__ __device__ GColumnInfo *GTreeIndex::getSchema() {
	return key_schema_;
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

__forceinline__ __device__ int64_t *GTreeIndex::getPackedKey() {
	return packed_key_;
}

__forceinline__ __device__ int GTreeIndex::lowerBound(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = -1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res <= 0) ? (middle - 1) : right;
		left = (compare_res > 0) ? (middle + 1) : left;
		result = (compare_res <= 0) ? middle : result;
	}
	return result;
}


__forceinline__ __device__ int GTreeIndex::upperBound(GTreeIndexKey key, int left, int right)
{
	int middle = -1;
	int result = right - 1;
	int compare_res = 0;

	while (left <= right) {
		middle = (left + right) >> 1;

		//Form the middle key
		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		right = (compare_res < 0) ? (middle - 1) : right;
		left = (compare_res >= 0) ? (middle + 1) : left;
		result = (compare_res < 0) ? middle : result;
	}

	return result;
}

__forceinline__ __device__ int GTreeIndex::lowerBound(GTreeIndexKey key, int root, int size, int stride)
{
	int middle = -1;
	int result = -1;
	int compare_res = 0;
	int ptr = size / 2;

	while (size > 0) {
		middle = root + ptr * stride;

		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		size /= 2;
		ptr = (compare_res <= 0) ? (ptr - size) : (ptr + size);
		result = (compare_res <= 0) ? (root + middle * stride) : result;
	}

	return result;
}

__forceinline__ __device__ int GTreeIndex::upperBound(GTreeIndexKey key, int root, int size, int stride)
{
	int middle = -1;
	int result = root + (size - 1) * stride;
	int compare_res = 0;
	int ptr = size / 2;

	while (size > 0) {
		middle = root + size / 2 * stride;

		GTreeIndexKey middle_key(packed_key_ + middle * key_size_, key_schema_, key_size_);

		compare_res = GTreeIndexKey::KeyComparator(key, middle_key);

		size /= 2;
		ptr = (compare_res < 0) ? (ptr - size) : (ptr + size);
		result = (compare_res < 0) ? (root + middle * stride) : result;
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
