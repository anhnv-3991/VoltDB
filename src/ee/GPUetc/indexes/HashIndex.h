#ifndef HASH_INDEX_H_
#define HASH_INDEX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "KeyIndex.h"

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
	template<typename signedType, typename unsignedType, int64_t typeMaxValue>
	__forceinline__ __device__ uint64_t convertSignedToUnsigned(signedType value);

	template<typename keyValueType>
	__forceinline__ __device__ void insertKey(int *key_offset, int *intra_key_offset, uint8_t key_value);
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
				uint64_t key_value = convertSignedToUnsigned<int8_t, uint8_t, INT8_MAX>(tuple[key_indices[i]]);
				insertKey<uint8_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				uint64_t key_value = convertSignedToUnsigned<int16_t, uint16_t, INT16_MAX>(tuple[key_indices[i]]);
				insertKey<uint16_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_INTEGER: {
				uint64_t key_value = convertSignedToUnsigned<int32_t, uint32_t, INT32_MAX>(tuple[key_indices[i]]);
				insertKey<uint32_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_BIGINT: {
				uint64_t key_value = convertSignedToUnsigned<int64_t, uint64_t, INT64_MAX>(tuple[key_indices[i]]);
				insertKey<uint64_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			default: {
				return;
			}
		}
	}
}

__forceinline__ __device__ GHashIndexKey::GHashIndexKey(GTuple tuple, int *key_idx, int size, uint64_t *packed_key)
{
	size_ = size;
	packed_key_ = packed_key;

	int key_offset = 0;
	int intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);

	for (int i = 0; i < size; i++) {
		switch (tuple.schema_[key_idx[i]].data_type) {
			case VALUE_TYPE_TINYINT: {
				uint64_t key_value = convertSignedToUnsigned<int8_t, uint8_t, INT8_MAX>(tuple.tuple_[key_idx[i]]);
				insertKey<uint8_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				uint64_t key_value = convertSignedToUnsigned<int16_t, uint16_t, INT16_MAX>(tuple.tuple_[key_idx[i]]);
				insertKey<uint16_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_INTEGER: {
				uint64_t key_value = convertSignedToUnsigned<int32_t, uint32_t, INT32_MAX>(tuple.tuple_[key_idx[i]]);
				insertKey<uint32_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_BIGINT: {
				uint64_t key_value = convertSignedToUnsigned<int64_t, uint64_t, INT64_MAX>(tuple.tuple_[key_idx[i]]);
				insertKey<uint64_t>(&key_offset, &intra_key_offset, key_value);
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
				uint64_t key_value = convertSignedToUnsigned<int8_t, uint8_t, INT8_MAX>(tuple[i]);
				insertKey<uint8_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				uint64_t key_value = convertSignedToUnsigned<int16_t, uint16_t, INT16_MAX>(tuple[i]);
				insertKey<uint16_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_INTEGER: {
				uint64_t key_value = convertSignedToUnsigned<int32_t, uint32_t, INT32_MAX>(tuple[i]);
				insertKey<uint32_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_BIGINT: {
				uint64_t key_value = convertSignedToUnsigned<int64_t, uint64_t, INT64_MAX>(tuple[i]);
				insertKey<int64_t, uint64_t, INT64_MAX>(&key_offset, &intra_key_offset, key_value);
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
				uint64_t key_value = convertSignedToUnsigned<int8_t, uint8_t, INT8_MAX>(tuple.tuple_[i]);
				insertKey<uint8_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				uint64_t key_value = convertSignedToUnsigned<int16_t, uint16_t, INT16_MAX>(tuple.tuple_[i]);
				insertKey<uint16_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_INTEGER: {
				uint64_t key_value = convertSignedToUnsigned<int32_t, uint32_t, INT32_MAX>(tuple.tuple_[i]);
				insertKey<uint32_t>(&key_offset, &intra_key_offset, key_value);
				break;
			}
			case VALUE_TYPE_BIGINT: {
				uint64_t key_value = convertSignedToUnsigned<int64_t, uint64_t, INT64_MAX>(tuple.tuple_[i]);
				insertKey<int64_t, uint64_t, INT64_MAX>(&key_offset, &intra_key_offset);
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

template<typename signedType, typename unsignedType, int64_t typeMaxValue>
__forceinline__ __device__ uint64_t GHashIndexKey::convertSignedToUnsigned(signedType value) {
	return static_cast<unsignedType>((signedType)value + typeMaxValue + 1);
}



template<typename keyValueType>
__forceinline__ __device__ void GHashIndexKey::insertKey(int *key_offset, int *intra_key_offset, uint8_t key_value) {
	for (int j = static_cast<int>(sizeof(keyValueType)) - 1; j >= 0; j--) {
		packed_key_[*key_offset] |= (0xFF & (key_value >> (j * 8))) << (*intra_key_offset * 8);
		*intra_key_offset--;
		if (*intra_key_offset < 0) {
			*intra_key_offset = static_cast<int>(sizeof(uint64_t) - 1);
			*key_offset++;
		}
	}
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
protected:
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
