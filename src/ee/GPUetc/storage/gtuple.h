#ifndef GTUPLE_H_
#define GTUPLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "GPUetc/common/GNValue.h"

namespace voltdb {

class GTuple {
	friend class GKeyIndex;
	friend class GTreeIndexKey;
	friend class GHashIndexKey;
	friend class GTreeIndex;
	friend class GExpression;
	friend class GNValue;
	friend class GTable;
public:
	__forceinline__ __device__ GTuple();
	__forceinline__ __device__ GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns);

	__forceinline__ __device__ bool insertColumnValue(GNValue new_value, int column_idx);


protected:
	int64_t *tuple_;
	GColumnInfo *schema_;
	int columns_;
};

__forceinline__ __device__ GTuple::GTuple()
{
	tuple_ = NULL;
	schema_ = NULL;
	columns_ = 0;
}


__forceinline__ __device__ GTuple::GTuple(int64_t *tuple, GColumnInfo *schema_buff, int max_columns)
{
	tuple_ = tuple;
	schema_ = schema_buff;
	columns_ = 0;
}


__forceinline__ __device__ bool GTuple::insertColumnValue(GNValue new_value, int column_idx)
{
	if (column_idx >= columns_)
		return false;

	tuple_[column_idx] = new_value.m_data_;
	schema_[column_idx].data_type = new_value.type_;

	return true;
}
}

#endif
