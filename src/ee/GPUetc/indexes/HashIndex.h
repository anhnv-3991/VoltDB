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
	__forceinline__ __device__ GHashIndexKey() {
		packed_key_ = NULL;
		size_ = 0;
	}

	__forceinline__ __host__ __device__ GHashIndexKey(uint64_t *packed_key, int size) {
		packed_key_ = packed_key;
		size_ = size;
	}

	/* Constructing key object from raw tuple, schema of the tuple, and the index schema.
	 * Keys are accumulated to packed_key based on the type of the columns.
	 * Used for constructing keys from the index table.
	 */
	__forceinline__ __device__ bool createKey(int64_t *tuple, GColumnInfo *schema, int *key_indices, int key_size) {
		if (size_ != key_size) {
			return false;
		}

		int key_offset = 0;
		int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

		for (int i = 0; i < size_; i++) {
			switch (schema[key_indices[i]].data_type) {
			case VALUE_TYPE_TINYINT: {
				uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[key_indices[i]] + INT8_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= ((0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8));
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			default: {
				return false;
			}
			}
		}
		return true;
	}

	/* Constructing key object from tuple and schema of the tuple.
	 * Keys are accumulated to packed_key.
	 * Used for constructing keys from the non-index table.
	 */
	__forceinline__ __device__ bool createKey(int64_t *tuple, GColumnInfo *schema, int column_num) {
		if (size_ != column_num)
			return false;

		int key_offset = 0;
		int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

		for (int i = 0; i < size_; i++) {
			switch (schema[i].data_type) {
			case VALUE_TYPE_TINYINT: {
				uint64_t key_value = static_cast<uint8_t>((int8_t)tuple[i] + INT8_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			default: {
				return false;
			}
			}
		}

		return true;
	}

	/* Constructing key object from raw tuple and schema of the tuple.
	 * Keys are accumulated to packed_key.
	 * Used for constructing keys from the non-index table.
	 */
	__forceinline__ __device__ bool createKey(GTuple tuple) {
		if (size_ != tuple.columns_)
			return false;

		int key_offset = 0;
		int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

		for (int i = 0; i < size_; i++) {
			switch (tuple.schema_[i].data_type) {
			case VALUE_TYPE_TINYINT: {
				uint64_t key_value = static_cast<uint8_t>((int8_t)tuple.tuple_[i] + INT8_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
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
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			default: {
				return false;
			}
			}
		}

		return true;
	}

	/* Constructing key object from a tuple.
	 * Used for constructing keys from the index table.
	 */
	__forceinline__ __device__ bool createKey(GTuple tuple, int *key_indices, int key_size) {
		if (size_ != key_size)
			return false;

		int key_offset = 0;
		int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

		for (int i = 0; i < key_size; i++) {
			switch (tuple.schema_[key_indices[i]].data_type) {
			case VALUE_TYPE_TINYINT: {
				uint64_t key_value = static_cast<uint8_t>((int8_t)tuple.tuple_[key_indices[i]] + INT8_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint8_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				uint64_t key_value = static_cast<uint16_t>((int16_t)tuple.tuple_[key_indices[i]] + INT16_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint16_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			case VALUE_TYPE_INTEGER: {
				uint64_t key_value = static_cast<uint32_t>((int32_t)tuple.tuple_[key_indices[i]] + INT32_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint32_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= ((0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8));
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			case VALUE_TYPE_BIGINT: {
				uint64_t key_value = static_cast<uint64_t>((int64_t)tuple.tuple_[key_indices[i]] + INT64_MAX + 1);

				for (int j = static_cast<int>(sizeof(uint64_t)) - 1; j >= 0; j--) {
					packed_key_[key_offset] |= (0xFF & (key_value >> (j * 8))) << (intra_key_offset * 8);
					intra_key_offset--;
					if (intra_key_offset < 0) {
						intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
						key_offset++;
					}
				}

				break;
			}
			default: {
				return false;
			}
			}
		}

		return true;
	}

	__forceinline__ __host__ __device__ uint64_t KeyHasher() {
		uint64_t seed = 0;

		for (int i = 0; i < size_; i++) {
			seed ^= packed_key_[i] + MASK_BITS + (seed << 6) + (seed >> 2);
		}

		return seed;
	}

private:
	uint64_t *packed_key_;
};

class GHashIndex {
	friend class GHashIndexKey;
public:
	GHashIndex();
	GHashIndex(int key_num, int key_size, int bucket_num);
	GHashIndex(uint64_t *packed_key, int *bucket_locations, int *sorted_idx, int *key_idx, int key_num, int key_size, int bucket_num);

	bool setKeySchema(int *key_schema, int key_size);
	void createIndex(int64_t *table, GColumnInfo *schema, int rows, int columns);

	~GHashIndex();

	void addEntry(GTuple new_tuple);
	void addBatchEntry(int64_t *table, GColumnInfo *table_schema, int rows, int columns);

	void merge(int old_left, int old_right, int new_left, int new_right);

	int getBucketNum();

	__forceinline__ __device__ GHashIndexKey getKeyAtSortedIndex(int key_index) {
		return GHashIndexKey(packed_key_ + sorted_idx_[key_index] * key_size_, key_size_);
	}

	__forceinline__ __device__ GHashIndexKey getKeyAtIndex(int key_index) {
		return GHashIndexKey(packed_key_ + key_index * key_size_, key_size_);
	}

	__forceinline__ __device__ int getKeyRows();

	__forceinline__ __host__ __device__ int *getSortedIdx();

	__forceinline__ __device__ int *getKeyIdx();

	__forceinline__ __device__ int getKeySize();

	__forceinline__ __device__ int *getBucketLocations();

	__forceinline__ __device__ void insertKeyTupleNoSort(GTuple new_key, int location);

	__forceinline__ __device__ int getBucketLocation(int bucket_idx);
	__forceinline__ __device__ int getColumns() {
		return key_size_;
	}

	void removeIndex();


protected:
	uint64_t *packed_key_;
	int *bucket_locations_;
	int *sorted_idx_;
	int *key_idx_;	// Index of columns selected as keys
	int key_num_;	//Number of key values (equal to the number of rows)
	int key_size_;	// Number of columns selected as keys
	int bucket_num_;
	int *new_bucket_locations_;
};


__forceinline__ __device__ int GHashIndex::getKeyRows()
{
	return key_num_;
}

__forceinline__ __host__ __device__ int *GHashIndex::getSortedIdx()
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
	GHashIndexKey key(packed_key_ + location * key_size_, key_size_);

	key.createKey(tuple);

	sorted_idx_[location] = location;
}

__forceinline__ __device__ int GHashIndex::getBucketLocation(int bucket_idx)
{
	return bucket_locations_[bucket_idx];
}

}

#endif
