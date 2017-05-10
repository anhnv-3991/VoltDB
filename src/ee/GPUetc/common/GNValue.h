#ifndef GNVALUE_H_
#define GNVALUE_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/types.h"


namespace voltdb {

class GNValue {
	friend class GTuple;
public:
	__forceinline__ __device__ GNValue();
	__forceinline__ __device__ GNValue(ValueType type);
	__forceinline__ __device__ GNValue(ValueType type, int64_t mdata);
	__forceinline__ __device__ GNValue(ValueType type, const char *input);

	__forceinline__ __device__ bool isNull() const;
	__forceinline__ __device__ bool isTrue() const;
	__forceinline__ __device__ bool isFalse() const;

	__forceinline__ __host__ __device__ void setValue(ValueType type, const char *data);
	__forceinline__ __host__ __device__ void setValueType(ValueType type);

	__forceinline__ __device__ int64_t getValue();
	__forceinline__ __device__ ValueType getValueType();

	__forceinline__ __device__ GNValue opNegate(void) const;
	__forceinline__ __device__ GNValue opAnd(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opOr(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opEqual(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opNotEqual(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opLessThan(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opLessThanOrEqual(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opGreaterThan(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opGreaterThanOrEqual(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opAdd(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opMultiply(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opSubtract(const GNValue rhs) const;
	__forceinline__ __device__ GNValue opDivide(const GNValue rhs) const;

	__forceinline__ __device__ static GNValue getTrue();
	__forceinline__ __device__ static GNValue getFalse();
	__forceinline__ __device__ static GNValue getInvalid();
	__forceinline__ __device__ static GNValue getNullValue();


	__forceinline__ __device__ void setNull();
	__forceinline__ __device__ void debug() const;

	__forceinline__ __device__ GNValue operator~() const;
	__forceinline__ __device__ GNValue operator&&(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator||(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator==(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator!=(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator<(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator<=(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator>(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator>=(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator+(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator*(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator-(const GNValue rhs) const;
	__forceinline__ __device__ GNValue operator/(const GNValue rhs) const;

protected:
	int64_t m_data_;
	ValueType type_;
};

__forceinline__ __device__ GNValue::GNValue()
{
	m_data_ = 0;
	type_ = VALUE_TYPE_INVALID;
}

__forceinline__ __device__ GNValue::GNValue(ValueType type)
{
	m_data_ = 0;
	type_ = type;
}

__forceinline__ __device__ GNValue::GNValue(ValueType type, int64_t mdata)
{
	m_data_ = mdata;
	type_ = type;
}

__device__ GNValue::GNValue(ValueType type, const char *input)
{
	type_ = type;

	switch (type) {
	case VALUE_TYPE_BOOLEAN:
	case VALUE_TYPE_TINYINT: {
		m_data_ = *reinterpret_cast<const int8_t *>(input);
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		m_data_ = *reinterpret_cast<const int16_t *>(input);
		break;
	}
	case VALUE_TYPE_INTEGER: {
		m_data_ = *reinterpret_cast<const int32_t *>(input);
		break;
	}
	case VALUE_TYPE_BIGINT:
	case VALUE_TYPE_DOUBLE:
	case VALUE_TYPE_TIMESTAMP: {
		m_data_ = *reinterpret_cast<const int64_t *>(input);
		break;
	}
	default:
		m_data_ = 0;
		type_ = VALUE_TYPE_INVALID;
		break;
	}
}

__forceinline__ __device__ bool GNValue::isNull() const
{
	return (type_ == VALUE_TYPE_NULL);
}

__forceinline__ __device__ bool GNValue::isTrue() const
{
	return (type_ == VALUE_TYPE_BOOLEAN && (bool)m_data_);
}

__forceinline__ __device__ bool GNValue::isFalse() const
{
	return (type_ == VALUE_TYPE_BOOLEAN && !(bool)m_data_);
}

__forceinline__ __host__ __device__ void GNValue::setValue(ValueType type, const char *input)
{
	switch (type) {
	case VALUE_TYPE_BOOLEAN:
	case VALUE_TYPE_TINYINT: {
		m_data_ = *reinterpret_cast<const int8_t *>(input);
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		m_data_ = *reinterpret_cast<const int16_t *>(input);
		break;
	}
	case VALUE_TYPE_INTEGER: {
		m_data_ = *reinterpret_cast<const int32_t *>(input);
		break;
	}
	case VALUE_TYPE_BIGINT:
	case VALUE_TYPE_DOUBLE:
	case VALUE_TYPE_TIMESTAMP: {
		m_data_ = *reinterpret_cast<const int64_t *>(input);
		break;
	}
	default: {
		break;
	}
	}
}

__forceinline__ __host__ __device__ void GNValue::setValueType(ValueType type)
{
	type_ = type;
}

__forceinline__ __device__ int64_t GNValue::getValue()
{
	return m_data_;
}

__forceinline__ __device__ ValueType GNValue::getValueType()
{
	return type_;
}

__forceinline__ __device__ GNValue GNValue::opNegate() const
{
	if (type_ == VALUE_TYPE_BOOLEAN) {
		return ((bool)m_data_) ? GNValue::getFalse() : GNValue::getTrue();
	}

	return GNValue::getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opAnd(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool left = (bool)m_data_, right = (bool)(rhs.m_data_);

		return (left && right) ? getTrue() : getFalse();
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opOr(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool left = (bool)m_data_, right = (bool)(rhs.m_data_);

		return (left || right) ? getTrue() : getFalse();
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opEqual(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d == right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ == rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opNotEqual(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d != right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ != rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opLessThan(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d < right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ < rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opLessThanOrEqual(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d <= right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ <= rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opGreaterThan(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d > right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ > rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opGreaterThanOrEqual(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d >= right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ >= rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opAdd(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d + right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ + rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opMultiply(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d * right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ * rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opSubtract(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d - right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ - rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::opDivide(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = (right_d != 0) ? left_d / right_d : 0;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = (rhs.m_data_ != 0) ? m_data_ / rhs.m_data_ : 0;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::getTrue()
{
	bool value = true;
	return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)value);
}

__forceinline__ __device__ GNValue GNValue::getFalse()
{
	bool value = false;
	return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)value);
}

__forceinline__ __device__ GNValue GNValue::getInvalid()
{
	return GNValue(VALUE_TYPE_INVALID);
}

__forceinline__ __device__ GNValue GNValue::getNullValue()
{
	return GNValue(VALUE_TYPE_NULL);
}


__forceinline__ __device__ void GNValue::setNull()
{
	m_data_ = 0;
	type_ = VALUE_TYPE_NULL;
}

__forceinline__ __device__ void GNValue::debug() const
{
	switch (type_) {
	case VALUE_TYPE_INVALID: {
		printf("VALUE TYPE INVALID");
		break;
	}
	case VALUE_TYPE_NULL: {
		printf("VALUE TYPE NULL");
		break;
	}
	case VALUE_TYPE_FOR_DIAGNOSTICS_ONLY_NUMERIC: {
		printf("VALUE TYPE FOR DIAGNOSTICS ONLY NUMERIC");
		break;
	}
	case VALUE_TYPE_TINYINT: {
		printf("VALUE TYPE TINYINT: %d", (int)m_data_);
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		printf("VALUE TYPE SMALLINT: %d", (int)m_data_);
		break;
	}
	case VALUE_TYPE_INTEGER: {
		printf("VALUE TYPE INTEGER: %d", (int)m_data_);
		break;
	}
	case VALUE_TYPE_BIGINT: {
		printf("VALUE TYPE BIGINT: %d", (int)m_data_);
		break;
	}
	case VALUE_TYPE_DOUBLE: {
		printf("VALUE TYPE DOUBLE: %lf", *reinterpret_cast<const double *>(&m_data_));
		break;
	}
	case VALUE_TYPE_VARCHAR: {
		printf("VALUE TYPE VARCHAR");
		break;
	}
	case VALUE_TYPE_TIMESTAMP: {
		printf("VALUE TYPE TIMESTAMP");
		break;
	}
	case VALUE_TYPE_DECIMAL: {
		printf("VALUE TYPE DECIMAL");
		break;
	}
	case VALUE_TYPE_BOOLEAN: {
		printf("VALUE TYPE BOOLEAN");
		break;
	}
	case VALUE_TYPE_ADDRESS: {
		printf("VALUE TYPE ADDRESS");
		break;
	}
	case VALUE_TYPE_VARBINARY: {
		printf("VALUE TYPE VARBINARY");
		break;
	}
	case VALUE_TYPE_ARRAY: {
		printf("VALUE TYPE VARBINARY");
		break;
	}
	default: {
		printf("UNDETECTED TYPE");
		break;
	}
	}
}

__forceinline__ __device__ GNValue GNValue::operator~() const
{
	if (type_ == VALUE_TYPE_BOOLEAN) {
		return ((bool)m_data_) ? GNValue::getFalse() : GNValue::getTrue();
	}

	return GNValue::getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator&&(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool left = (bool)m_data_, right = (bool)(rhs.m_data_);

		return (left && right) ? getTrue() : getFalse();
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator||(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool left = (bool)m_data_, right = (bool)(rhs.m_data_);

		return (left || right) ? getTrue() : getFalse();
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator==(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d == right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ == rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator!=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d != right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ != rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator<(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d < right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ < rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator<=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d <= right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ <= rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator>(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d > right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ > rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator>=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			return (left_d >= right_d) ? getTrue() : getFalse();
		} else {
			return (m_data_ >= rhs.m_data_) ? getTrue() : getFalse();
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator+(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d + right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ + rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator*(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d * right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ * rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator-(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = left_d - right_d;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = m_data_ - rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

__forceinline__ __device__ GNValue GNValue::operator/(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		double left_d = (type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&m_data_) : static_cast<double>(m_data_);
		double right_d = (rhs.type_ == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<const double*>(&rhs.m_data_) : static_cast<double>(rhs.m_data_);
		int64_t res_i;
		double res_d;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			res_d = (right_d != 0) ? left_d / right_d : 0;
			res_i = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, res_i);
		} else {
			res_i = (rhs.m_data_ != 0) ? m_data_ / rhs.m_data_ : 0;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}


}
#endif
