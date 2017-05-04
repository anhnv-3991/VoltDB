#ifndef HASH_INDEX_H_
#define HASH_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"
#include "Index.h"
#include "GPUetc/storage/gtuple.h"

namespace voltdb {

class GHashIndexKey: public GKeyIndex {
public:
#define MASK_BITS 0x9e3779b9
	__forceinline__ __device__ GHashIndexKey();

	/* Constructing key object from raw tuple, schema of the tuple, and the index schema.
	 * Keys are accumulated to packed_key based on the type of the columns.
	 * Used for constructing keys from the index table.
	 */
	__forceinline__ __device__ GHashIndexKey(int64_t *tuple, GColumnInfo *schema, int *key_indices, int index_num, uint64_t *packed_key);

	/* Constructing key object from tuple and schema of the tuple.
	 * Keys are accumulated to packed_key.
	 * Used for constructing keys from the non-index table.
	 */
	__forceinline__ __device__ GHashIndexKey(int64_t *tuple, GColumnInfo *schema, int column_num, uint64_t *packed_key);

	/* Constructing key object from raw tuple and schema of the tuple.
	 * Keys are accumulated to packed_key.
	 * Used for constructing keys from the non-index table.
	 */
	__forceinline__ __device__ GHashIndexKey(GTuple tuple, int *key_idx, int size, uint64_t *packed_key);

	/* Constructing key object from a tuple.
	 * Used for constructing keys from the non-index table.
	 */
	__forceinline__ __device__ GHashIndexKey(GTuple tuple, uint64_t *packed_key);

	__forceinline__ __device__ GHashIndexKey(GHashIndex index, int key_idx);

	__forceinline__ __device__ uint64_t KeyHasher();

private:
	uint64_t *packed_key_;
};

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(int64_t *tuple, GColumnInfo *schema, int *key_indices, int index_num, uint64_t *packed_key)
{
	size_ = index_num;
	packed_key_ = packed_key;

	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	for (int i = 0; i < index_num; i++) {
		switch (schema[key_indices[i]].data_type) {
		case VALUE_TYPE_TINYINT: {
			uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[key_indices[i]] + INT8_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			uint64_t key_value = static_cast<uint16_t>((int16_t)tuple[key_indices[i]] + INT16_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_INTEGER: {
			uint64_t key_value = static_cast<uint32_t>((int32_t)tuple[key_indices[i]] + INT32_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= ((0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8));
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_BIGINT: {
			uint64_t key_value = static_cast<uint64_t>((int64_t)tuple[key_indices[i]] + INT64_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		default: {
			return;
		}
		}
	}
}

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(GTuple tuple, int *key_indices, int size, uint64_t *packed_key)
{
	size_ = size;
	packed_key_ = packed_key;

	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	for (int i = 0; i < size; i++) {
		switch (tuple.schema_[key_indices[i]].data_type) {
		case VALUE_TYPE_TINYINT: {
			uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[key_indices[i]] + INT8_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			uint64_t key_value = static_cast<uint16_t>((int16_t)tuple[key_indices[i]] + INT16_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_INTEGER: {
			uint64_t key_value = static_cast<uint32_t>((int32_t)tuple[key_indices[i]] + INT32_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= ((0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8));
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_BIGINT: {
			uint64_t key_value = static_cast<uint64_t>((int64_t)tuple[key_indices[i]] + INT64_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		default: {
			return;
		}
		}
	}
}

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(int64_t *tuple, GColumnInfo *schema, int column_num, uint64_t *packed_key)
{
	size_ = column_num;
	packed_key_ = packed_key;

	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	for (int i = 0; i < index_num; i++) {
		switch (schema[i].data_type) {
		case VALUE_TYPE_TINYINT: {
			uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[i] + INT8_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			uint64_t key_value = static_cast<uint16_t>((int16_t)tuple[i] + INT16_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_INTEGER: {
			uint64_t key_value = static_cast<uint32_t>((int32_t)tuple[i] + INT32_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_BIGINT: {
			uint64_t key_value = static_cast<uint64_t>((int64_t)tuple[i] + INT64_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		default: {
			return;
		}
		}
	}
}

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(GTuple tuple, uint64_t *packed_key)
{
	size_ = tuple.columns_;
	packed_key_ = packed_key;

	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	for (int i = 0; i < index_num; i++) {
		switch (tuple.schema_[i].data_type) {
		case VALUE_TYPE_TINYINT: {
			uint64_t key_value = static_cast<uint8_t>((int8_t)tuple.tuple_[i] + INT8_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}
			break;
		}
		case VALUE_TYPE_SMALLINT: {
			uint64_t key_value = static_cast<uint16_t>((int16_t)tuple.tuple_[i] + INT16_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_INTEGER: {
			uint64_t key_value = static_cast<uint32_t>((int32_t)tuple.tuple_[i] + INT32_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		case VALUE_TYPE_BIGINT: {
			uint64_t key_value = static_cast<uint64_t>((int64_t)tuple.tuple_[i] + INT64_MAX + 1);

			for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
				packed_key[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
				intra_key_offset--;
				if (intra_key_offset < 0) {
					intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
					key_offset++;
				}
			}

			break;
		}
		default: {
			return;
		}
		}
	}
}

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(GHashIndex index, int key_idx)
{
	packed_key_ = index.packed_key_ + key_idx * index.key_size_;
	size_ = index.key_size_;
}

__forceinline__ __device__ uint64_t GHashIndexKey::KeyHasher()
{
	uint64_t seed = 0;

	for (int i = 0; i < size_; i++) {
		seed ^= packed_key_[i] + MASK_BITS + (seed << 6) + (seed >> 2);
	}

	return seed;
}

class GHashIndex: public GIndex {
	friend class GHashIndexKey;
public:
	GHashIndex();
	GHashIndex(int key_num, int key_size);
	GHashIndex(GTable table, int *key_idx, int key_size, int block_num);
	~GHashIndex();

	void addEntry(GTuple new_tuple);

	void addBatchEntry(GTable table, int base_idx, int size);

	void merge(int old_left, int old_right, int new_left, int new_right, int *new_bucket_locations);

	int getBucketNum();

	__forceinline__ __device__ int getKeyRows();

	__forceinline__ __device__ int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx();

	__forceinline__ __device__ int getKeySize();

	__forceinline__ __device__ int *getBucketLocations();

	__forceinline__ __device__ void insertKeyTupleNoSort(GTuple new_key, int location);

	__forceinline__ __device__ int getBucketLocation(int bucket_idx);
	__forceinline__ __device__ int getColumns() {
		return key_size_;
	}

	__forceinline__ __device__ int getRows() {
		return key_num_;
	}

	int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx() {
		return key_idx_;
	}

	__forceinline__ __device__ int getKeySize() {
		return key_size_;
	}

	void removeIndex() {
		checkCudaErrors(cudaFree(sorted_idx));
		checkCudaErrors(cudaFree(key_idx));
	}


protected:
	int key_num_;	//Number of key values (equal to the number of rows)
	int *sorted_idx_;
	int *key_idx_;	// Index of columns selected as keys
	int key_size_;	// Number of columns selected as keys
	uint64_t *packed_key_;
	int *bucket_locations_;
	int bucket_num_;
};


__forceinline__ __device__ int GHashIndex::getKeyRows()
{
	return key_num_;
}

__forceinline__ __device__ int *GHashIndex::getSortedIdx()
{
	return sorted_idx_;
}

__forceinline__ __device__ int *GHashIndex::getKeyIdx()
{
	return key_idx_;
}

__forceinline__ __device__ int GHashIndex::getKeySize()
{
	return key_size_;
}


__forceinline__ __device__ int *GHashIndex::getBucketLocations()
{
	return bucket_locations_;
}

__forceinline__ __device__ void GHashIndex::insertKeyTupleNoSort(GTuple tuple, int location)
{
	GHashIndexKey key(tuple, packed_key_ + location * key_size_);
	sorted_idx_[location] = location;
}

__forceinline__ __device__ int GHashIndex::getBucketLocation(int bucket_idx)
{
	return bucket_locations_[bucket_idx];
}

}

#endif
