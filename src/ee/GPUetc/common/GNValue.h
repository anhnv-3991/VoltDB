/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef GNVALUE_HPP_
#define GNVALUE_HPP_

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif


#include <cassert>
#include <cfloat>
#include <climits>
#include <math.h>
#include <cstdlib>
#include <stdint.h>
#include <limits.h>
#include <string>
#include <algorithm>
#include <vector>
#include <stdio.h>

#include "boost/scoped_ptr.hpp"
#include "boost/functional/hash.hpp"
#include "ttmath/ttmathint.h"
#include "common/types.h"
#include "common/value_defs.h"
#include "common/StringRef.h"
#include "GPUetc/common/Gvalue_defs.h"
#include "GPUetc/cudaheader.h"

namespace voltdb {

/*
 * Objects are length preceded with a short length value or a long length value
 * depending on how many bytes are needed to represent the length. These
 * define how many bytes are used for the short value vs. the long value.
 */
#define SHORT_OBJECT_LENGTHLENGTH static_cast<char>(1)
#define LONG_OBJECT_LENGTHLENGTH static_cast<char>(4)
#define OBJECT_NULL_BIT static_cast<char>(1 << 6)
#define OBJECT_CONTINUATION_BIT static_cast<char>(1 << 7)
#define OBJECT_MAX_LENGTH_SHORT_LENGTH 63

#define FULL_STRING_IN_MESSAGE_THRESHOLD 100

//The int used for storage and return values
typedef ttmath::Int<2> TTInt;
//Long integer with space for multiplication and division without carry/overflow
//typedef ttmath::Int<4> TTLInt;


/**
 * A class to wrap all scalar values regardless of type and
 * storage. An NValue is not the representation used in the
 * serialization of VoltTables nor is it the representation of how
 * scalar values are stored in tables. NValue does have serialization
 * and deserialization mechanisms for both those storage formats.
 * NValues are designed to be immutable and for the most part not
 * constructable from raw data types. Access to the raw data is
 * restricted so that all operations have to go through the member
 * functions that can perform the correct casting and error
 * checking. ValueFactory can be used to construct new NValues, but
 * that should be avoided if possible.
 */
class GNValue {

  public:
    /* Create a default NValue */
    inline CUDAH GNValue();
    inline CUDAH GNValue(const ValueType type) {
		::memset(&m_data, 0, sizeof(int64_t));
		m_valueType = type;
		m_sourceInlined = false;
    }

	inline CUDAH GNValue(const ValueType type, int64_t mdata) {
		m_data = mdata;
		m_valueType = type;
		m_sourceInlined = false;
	}


    /* Check if the value represents SQL NULL */
    CUDAH bool isNull() const;

    inline CUDAH void setNull();

    CUDAH bool getSourceInlined() const {
    	return m_sourceInlined;
    }

    CUDAH int64_t getMdata() const {
    	return m_data;
    }

    inline CUDAH int compare_withoutNull(const GNValue rhs) const;

    /* Boolean operations */
    inline CUDAH GNValue op_negate(void) const;
    inline CUDAH GNValue op_and(const GNValue rhs) const;
    inline CUDAH GNValue op_or(const GNValue rhs) const;
    /* Return a boolean NValue with the comparison result */

    inline CUDAH GNValue op_equal(const GNValue rhs) const;
    inline CUDAH GNValue op_notEqual(const GNValue rhs) const;
    inline CUDAH GNValue op_lessThan(const GNValue rhs) const;
    inline CUDAH GNValue op_lessThanOrEqual(const GNValue rhs) const;
    inline CUDAH GNValue op_greaterThan(const GNValue rhs) const;
    inline CUDAH GNValue op_greaterThanOrEqual(const GNValue rhs) const;
    inline CUDAH GNValue op_add(const GNValue rhs) const;
    inline CUDAH GNValue op_multiply(const GNValue rhs) const;
    inline CUDAH GNValue op_subtract(const GNValue rhs) const;
    inline CUDAH GNValue op_divide(const GNValue rhs) const;


/**
 * Retrieve a boolean NValue that is true
 */
    inline CUDAH static GNValue getTrue() {
    	int64_t tmp = true;
        GNValue retval(VALUE_TYPE_BOOLEAN, tmp);
        return retval;
    }

/**
 * Retrieve a boolean NValue that is false
 */
    inline CUDAH static GNValue getFalse() {
    	int64_t tmp = false;
        GNValue retval(VALUE_TYPE_BOOLEAN, tmp);
        return retval;
    }

/**
 * Returns C++ true if this NValue is a boolean and is true
 * If it is NULL, return false.
 */
    inline CUDAH bool isTrue() const {
        return (bool)getValue();
    }

/**
 * Returns C++ false if this NValue is a boolean and is true
 * If it is NULL, return false.
 */
    inline CUDAH bool isFalse() const {
    	return !((bool)getValue());
    }


    inline CUDAH static void getNullValueByPointer(GNValue *retval,ValueType type) {
        retval->setValueType(type);
        retval->setNull();
    }

    inline CUDAH static GNValue getNullValue(){
        GNValue retval(VALUE_TYPE_NULL);
        //retval.tagAsNull();
        retval.setNull();
        return retval;
    }

    inline CUDAH static GNValue getNullValue(ValueType type) {
        GNValue retval(type);
        retval.setNull();
        return retval;
    }


    inline CUDAH void setMdata(ValueType type, const char *input){

    	switch (type) {
    	case VALUE_TYPE_BOOLEAN:
    	case VALUE_TYPE_TINYINT: {
    		m_data = *reinterpret_cast<const int8_t *>(input);
    		break;
    	}
    	case VALUE_TYPE_SMALLINT: {
    		m_data = *reinterpret_cast<const int16_t *>(input);
    		break;
    	}
    	case VALUE_TYPE_INTEGER: {
    		m_data = *reinterpret_cast<const int32_t *>(input);
			break;
    	}
    	case VALUE_TYPE_BIGINT:
    	case VALUE_TYPE_DOUBLE:
    	case VALUE_TYPE_TIMESTAMP: {
    		m_data = *reinterpret_cast<const int64_t *>(input);
			break;
    	}
    	default: {
    		break;
    	}
    	}
    }

    inline CUDAH void setSourceInlined(bool sourceInlined)
    {
        m_sourceInlined = sourceInlined;
    }

    /**
     * Set the type of the value that will be stored in this instance.
     * The last of the 16 bytes of storage allocated in an NValue
     * is used to store the type
     */
    inline CUDAH void setValueType(ValueType type) {
        m_valueType = type;
    }

    /**
     * Get the type of the value. This information is private
     * to prevent code outside of NValue from branching based on the type of a value.
     */
    inline CUDAH ValueType getValueType() const {
        return m_valueType;
    }


    inline CUDAH void debug() const {
    	switch (m_valueType) {
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
				printf("VALUE TYPE TINYINT: %d", (int)getValue());
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				printf("VALUE TYPE SMALLINT: %d", (int)getValue());
				break;
			}
			case VALUE_TYPE_INTEGER: {
				printf("VALUE TYPE INTEGER: %d", (int)getValue());
				break;
			}
			case VALUE_TYPE_BIGINT: {
				printf("VALUE TYPE BIGINT: %d", (int)getValue());
				break;
			}
			case VALUE_TYPE_DOUBLE: {
				int64_t tmp = getValue();
				printf("VALUE TYPE DOUBLE: %lf", *reinterpret_cast<double *>(&tmp));
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

    inline CUDAH int64_t getValue() const {
    	return m_data;
    }
  private:
    int64_t m_data;
    ValueType m_valueType;
    bool m_sourceInlined;

    /**
     * Private constructor that initializes storage and the specifies the type of value
     * that will be stored in this instance
     */

    template<typename T>
    inline CUDAH int compareValue (const T lhsValue, const T rhsValue) const {
		if (lhsValue == rhsValue)
			return VALUE_COMPARE_EQUAL;

		if (lhsValue > rhsValue)

			return VALUE_COMPARE_GREATERTHAN;
		return VALUE_COMPARE_LESSTHAN;
    }

    inline CUDAH int compareDoubleValue (const double lhsValue, const double rhsValue) const {
		if (lhsValue == rhsValue)
			return VALUE_COMPARE_EQUAL;

		if (lhsValue > rhsValue)
			return VALUE_COMPARE_GREATERTHAN;

		return VALUE_COMPARE_LESSTHAN;
    }

};

/**
 * Public constructor that initializes to an NValue that is unusable
 * with other NValues.  Useful for declaring storage for an NValue.
 */
inline CUDAH GNValue::GNValue() {
    m_data = 0;
    m_valueType = VALUE_TYPE_INVALID;
    m_sourceInlined = false;
}



/**
 * Set this NValue to null.
 */
//inline CUDAH void GNValue::setNull() {
//    tagAsNull(); // This gets overwritten for DECIMAL -- but that's OK.
//    switch (getValueType())
//    {
//    case VALUE_TYPE_BOOLEAN:
//        // HACK BOOL NULL
//        *reinterpret_cast<int8_t*>(m_data) = GINT8_NULL;
//        break;
//    case VALUE_TYPE_NULL:
//    case VALUE_TYPE_INVALID:
//        return;
//    case VALUE_TYPE_TINYINT:
//        getTinyInt() = GINT8_NULL;
//        break;
//    case VALUE_TYPE_SMALLINT:
//        getSmallInt() = GINT16_NULL;
//        break;
//    case VALUE_TYPE_INTEGER:
//        getInteger() = GINT32_NULL;
//        break;
//    case VALUE_TYPE_TIMESTAMP:
//        getTimestamp() = GINT64_NULL;
//        break;
//    case VALUE_TYPE_BIGINT:
//        getBigInt() = GINT64_NULL;
//        break;
//    case VALUE_TYPE_DOUBLE:
//        getDouble() = GDOUBLE_MIN;
//        break;
//    case VALUE_TYPE_VARCHAR:
//    case VALUE_TYPE_VARBINARY:
//    case VALUE_TYPE_DECIMAL:
//        break;
//    default: {
//        break;
//    }
//    }
//}

inline CUDAH void GNValue::setNull() {
	m_valueType = VALUE_TYPE_NULL;
}


inline CUDAH int GNValue::compare_withoutNull(const GNValue rhs) const {
    assert(isNull() == false && rhs.isNull() == false);

    int64_t left_i = getValue(), right_i = rhs.getValue();
    ValueType ltype = getValueType(), rtype = rhs.getValueType();
    int res_i, res_d;

	double left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
	double right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);

	res_d = (left_d > right_d) ? VALUE_COMPARE_GREATERTHAN : ((left_d < right_d) ? VALUE_COMPARE_LESSTHAN : VALUE_COMPARE_EQUAL);

	res_i = (left_i > right_i) ? VALUE_COMPARE_GREATERTHAN : ((left_i < right_i) ? VALUE_COMPARE_LESSTHAN : VALUE_COMPARE_EQUAL);

	return (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) ? res_d : res_i;
}


inline CUDAH bool GNValue::isNull() const {

    //assert(m_valueType != VALUE_TYPE_VARCHAR && m_valueType != VALUE_TYPE_VARBINARY && m_valueType != VALUE_TYPE_DECIMAL);
/*
    if (getValueType() == VALUE_TYPE_DECIMAL) {
        TTInt min;
        min.SetMin();
        return getDecimal() == min;
    }
*/
    //return m_data[13] == OBJECT_NULL_BIT;
	return m_valueType == VALUE_TYPE_NULL;
}

inline CUDAH GNValue GNValue::op_negate(void) const {
	bool tmp = (bool)(getValue());
	return (tmp) ? getFalse() : getTrue();
}

inline CUDAH GNValue GNValue::op_and(const GNValue rhs) const {
	bool left = (bool)(getValue()), right = (bool)(rhs.getValue());
	return (left && right) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_or(const GNValue rhs) const {
	bool left = (bool)(getValue()), right = (bool)(rhs.getValue());
	return (left || right) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_equal(const GNValue rhs) const {
	int res = compare_withoutNull(rhs);
	return (res == VALUE_COMPARE_EQUAL) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_notEqual(const GNValue rhs) const {
	int res = compare_withoutNull(rhs);
	return (res != VALUE_COMPARE_EQUAL) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_lessThan(const GNValue rhs) const{
	int res = compare_withoutNull(rhs);
	return (res == VALUE_COMPARE_LESSTHAN) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_lessThanOrEqual(const GNValue rhs) const {
	int res = compare_withoutNull(rhs);
	return (res == VALUE_COMPARE_LESSTHAN || res == VALUE_COMPARE_EQUAL) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_greaterThan(const GNValue rhs) const {
	int res = compare_withoutNull(rhs);
	return (res == VALUE_COMPARE_GREATERTHAN) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_greaterThanOrEqual(const GNValue rhs) const {
	int res = compare_withoutNull(rhs);
	return (res == VALUE_COMPARE_GREATERTHAN || res == VALUE_COMPARE_EQUAL) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_add(const GNValue rhs) const {
    int64_t left_i = getValue(), right_i = rhs.getValue();
    ValueType ltype = getValueType(), rtype = rhs.getValueType();
	double left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
	double right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
	int64_t res_i;
	double res_d;
	ValueType res_type;

	if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
		res_d = left_d + right_d;
		res_i = *reinterpret_cast<int64_t *>(&res_d);
		res_type = VALUE_TYPE_DOUBLE;
	} else {
		res_i = left_i + right_i;
		res_d = 0;
		res_type = (ltype > rtype) ? ltype : rtype;
	}

	return GNValue(res_type, res_i);
}

inline CUDAH GNValue GNValue::op_multiply(const GNValue rhs) const {
    int64_t left_i = getValue(), right_i = rhs.getValue();
    ValueType ltype = getValueType(), rtype = rhs.getValueType();
	double left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
	double right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
	int64_t res_i;
	double res_d;
	ValueType res_type;

	if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
		res_d = left_d * right_d;
		res_i = *reinterpret_cast<int64_t *>(&res_d);
		res_type = VALUE_TYPE_DOUBLE;
	} else {
		res_i = left_i * right_i;
		res_d = 0;
		res_type = (ltype > rtype) ? ltype : rtype;
	}

	return GNValue(res_type, res_i);
}

inline CUDAH GNValue GNValue::op_divide(const GNValue rhs) const {
    int64_t left_i = getValue(), right_i = rhs.getValue();
    ValueType ltype = getValueType(), rtype = rhs.getValueType();
	double left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
	double right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
	int64_t res_i;
	double res_d;
	ValueType res_type;

	if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
		res_d = (right_d != 0) ? left_d / right_d : 0;
		res_i = *reinterpret_cast<int64_t *>(&res_d);
		res_type = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
	} else {
		res_i = (right_i != 0) ? left_i / right_i : 0;
		res_d = 0;
		res_type = (ltype > rtype) ? ltype : rtype;
		res_type = (right_i != 0) ? res_type  : VALUE_TYPE_INVALID;
	}

	return GNValue(res_type, res_i);
}

inline CUDAH GNValue GNValue::op_subtract(const GNValue rhs) const {
    int64_t left_i = getValue(), right_i = rhs.getValue();
    ValueType ltype = getValueType(), rtype = rhs.getValueType();
	double left_d = (ltype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&left_i) : static_cast<double>(left_i);
	double right_d = (rtype == VALUE_TYPE_DOUBLE) ? *reinterpret_cast<double *>(&right_i) : static_cast<double>(right_i);
	int64_t res_i;
	double res_d;
	ValueType res_type;

	if (ltype == VALUE_TYPE_DOUBLE || rtype == VALUE_TYPE_DOUBLE) {
		res_d = left_d - right_d;
		res_i = *reinterpret_cast<int64_t *>(&res_d);
		res_type = VALUE_TYPE_DOUBLE;
	} else {
		res_i = left_i - right_i;
		res_d = 0;
		res_type = (ltype > rtype) ? ltype : rtype;
	}

	return GNValue(res_type, res_i);
}

} // namespace voltdb

#endif /* GNVALUE_HPP_ */
