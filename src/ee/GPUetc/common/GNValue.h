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
    CUDAH GNValue();

    /* Check if the value represents SQL NULL */
    CUDAH bool isNull() const;

    CUDAH void setNull();

    CUDAH bool getSourceInlined() const {
    	return m_sourceInlined;
    }

    CUDAH const char *getMdata() const {
    	return m_data;
    }

    inline CUDAH int compare_withoutNull(const GNValue rhs) const;

    /* Boolean operations */
    inline CUDAH GNValue op_negate(void) const;
    inline CUDAH GNValue op_and(const GNValue rhs) const;
    inline CUDAH GNValue op_or(const GNValue rhs) const;
    /* Return a boolean NValue with the comparison result */

    inline CUDAH bool op_equals_withoutNull(const GNValue *rhs) const;
    inline CUDAH bool op_notEquals_withoutNull(const GNValue *rhs) const;
    inline CUDAH bool op_lessThan_withoutNull(const GNValue *rhs) const;
    inline CUDAH bool op_lessThanOrEqual_withoutNull(const GNValue *rhs) const;
    inline CUDAH bool op_greaterThan_withoutNull(const GNValue *rhs) const;
    inline CUDAH bool op_greaterThanOrEqual_withoutNull(const GNValue *rhs) const;

    inline CUDAH GNValue op_equal(const GNValue rhs) const;
    inline CUDAH GNValue op_notEqual(const GNValue rhs) const;
    inline CUDAH GNValue op_lessThan(const GNValue rhs) const;
    inline CUDAH GNValue op_lessThanOrEqual(const GNValue rhs) const;
    inline CUDAH GNValue op_greaterThan(const GNValue rhs) const;
    inline CUDAH GNValue op_greaterThanOrEqual(const GNValue rhs) const;

    /*
    static const uint16_t kMaxDecPrec = 38;
    static const uint16_t kMaxDecScale = 12;
    static const int64_t kMaxScaleFactor = 1000000000000;
    */
/**
 * Retrieve a boolean NValue that is true
 */
    inline CUDAH static GNValue getTrue() {
        GNValue retval(VALUE_TYPE_BOOLEAN);
        retval.getBoolean() = true;
        return retval;
    }

/**
 * Retrieve a boolean NValue that is false
 */
    inline CUDAH static GNValue getFalse() {
        GNValue retval(VALUE_TYPE_BOOLEAN);
        retval.getBoolean() = false;
        return retval;
    }

/**
 * Returns C++ true if this NValue is a boolean and is true
 * If it is NULL, return false.
 */
    inline CUDAH bool isTrue() const {
        if (isBooleanNULL()) {
            return false;
        }
        return getBoolean();
    }

/**
 * Returns C++ false if this NValue is a boolean and is true
 * If it is NULL, return false.
 */
    inline CUDAH bool isFalse() const {
        if (isBooleanNULL()) {
            return false;
        }
        return !getBoolean();
    }

    inline CUDAH bool isBooleanNULL() const {
        assert(getValueType() == VALUE_TYPE_BOOLEAN);
        return *reinterpret_cast<const int8_t*>(m_data) == INT8_NULL;
    }

    CUDAH static void getNullValueByPointer(GNValue *retval) {
        retval->setValueType(VALUE_TYPE_NULL);
        retval->tagAsNull();
    }

    CUDAH static void getNullValueByPointer(GNValue *retval,ValueType type) {
        retval->setValueType(type);
        retval->setNull();
    }

    CUDAH static GNValue getNullValue(){
        GNValue retval(VALUE_TYPE_NULL);
        //retval.tagAsNull();
        retval.setNull();
        return retval;
    }

    CUDAH static GNValue getNullValue(ValueType type) {
        GNValue retval(type);
        retval.setNull();
        return retval;
    }

    CUDAH int getHashValue(int shift, int partition){
        assert(m_valueType != VALUE_TYPE_VARCHAR && m_valueType != VALUE_TYPE_VARBINARY && m_valueType != VALUE_TYPE_DECIMAL);
        
        if(getValueType() == VALUE_TYPE_DOUBLE){
            int64_t integer;
            integer = reinterpret_cast<int64_t&>(getDouble());
            return static_cast<int>((integer>>shift)%partition);
        }else if(getValueType() == VALUE_TYPE_DECIMAL){
/*
            TTInt Value = getDecimal();
            return static_cast<int>((Value>>shift)%partition);
*/
        }else{
            int64_t Value;
            
            switch (getValueType()) {
            case VALUE_TYPE_TINYINT:
            {
                Value = static_cast<int64_t>(getTinyInt());
                //the type of (Value>>shift)%partition is int64_t , but the value is smaller than MAX int32_t.
                //So static_cast<int> will not be error.
                return static_cast<int>((Value>>shift)%partition);
            }
            case VALUE_TYPE_SMALLINT:
            {
                Value = static_cast<int64_t>(getSmallInt());
                return static_cast<int>((Value>>shift)%partition);
            }
            case VALUE_TYPE_INTEGER:
            {
                Value = static_cast<int64_t>(getInteger());
                return static_cast<int>((Value>>shift)%partition);
            }
            case VALUE_TYPE_BIGINT:
            {
                Value = getBigInt();
                return static_cast<int>((Value>>shift)%partition);
            }
            case VALUE_TYPE_TIMESTAMP:
            {
                Value = getTimestamp();
                return static_cast<int>((Value>>shift)%partition);
            }
            default:
            {

            }
            }


        }
        return -1;//TODO: maybe -1 is worse value.
    }

    CUDAH static void initFromTupleStorage(const void *storage, ValueType type, GNValue *retval);

    CUDAH void setMdata(const char *input){
        memcpy(m_data,input,16);
    }

    CUDAH void setSourceInlined(bool sourceInlined)
    {
        m_sourceInlined = sourceInlined;
    }

    /**
     * Set the type of the value that will be stored in this instance.
     * The last of the 16 bytes of storage allocated in an NValue
     * is used to store the type
     */
    CUDAH void setValueType(ValueType type) {
        m_valueType = type;
    }

    /**
     * Get the type of the value. This information is private
     * to prevent code outside of NValue from branching based on the type of a value.
     */
    CUDAH ValueType getValueType() const {
        return m_valueType;
    }


    CUDAH void debug() const {
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
				printf("VALUE TYPE TINYINT");
				break;
			}
			case VALUE_TYPE_SMALLINT: {
				printf("VALUE TYPE SMALLINT");
				break;
			}
			case VALUE_TYPE_INTEGER: {
				printf("VALUE TYPE INTEGER");
				break;
			}
			case VALUE_TYPE_BIGINT: {
				printf("VALUE TYPE BIGINT");
				break;
			}
			case VALUE_TYPE_DOUBLE: {
				printf("VALUE TYPE DOUBLE");
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
  private:

    
    /**
     * 16 bytes of storage for GNValue data.
     */
    char m_data[16];
    ValueType m_valueType;
    bool m_sourceInlined;

    /**
     * Private constructor that initializes storage and the specifies the type of value
     * that will be stored in this instance
     */
    CUDAH GNValue(const ValueType type) {
        ::memset( m_data, 0, 16);
        setValueType(type);
        m_sourceInlined = false;
    }

    inline CUDAH void tagAsNull() { m_data[13] = OBJECT_NULL_BIT; }


    CUDAH const int8_t& getTinyInt() const {
        assert(getValueType() == VALUE_TYPE_TINYINT);
        return *reinterpret_cast<const int8_t*>(m_data);
    }
  
    CUDAH int8_t& getTinyInt() {
        assert(getValueType() == VALUE_TYPE_TINYINT);
        return *reinterpret_cast<int8_t*>(m_data);
    }

    CUDAH const int16_t& getSmallInt() const {
        //assert(getValueType() == VALUE_TYPE_SMALLINT);
        return *reinterpret_cast<const int16_t*>(m_data);
    }
  
    CUDAH int16_t& getSmallInt() {
        assert(getValueType() == VALUE_TYPE_SMALLINT);
        return *reinterpret_cast<int16_t*>(m_data);
    }

    CUDAH const int32_t& getInteger() const {
        assert(getValueType() == VALUE_TYPE_INTEGER);
        return *reinterpret_cast<const int32_t*>(m_data);
    }
  
    CUDAH int32_t& getInteger() {
        assert(getValueType() == VALUE_TYPE_INTEGER);
        return *reinterpret_cast<int32_t*>(m_data);
    }
  
    CUDAH const int64_t& getBigInt() const {
        assert((getValueType() == VALUE_TYPE_BIGINT) ||
               (getValueType() == VALUE_TYPE_TIMESTAMP) ||
               (getValueType() == VALUE_TYPE_ADDRESS));
    	return *reinterpret_cast<const int64_t*>(m_data);
    }

    CUDAH int64_t& getBigInt() {
        assert((getValueType() == VALUE_TYPE_BIGINT) ||
               (getValueType() == VALUE_TYPE_TIMESTAMP) ||
               (getValueType() == VALUE_TYPE_ADDRESS));
        return *reinterpret_cast<int64_t*>(m_data);
    }

    CUDAH const int64_t& getTimestamp() const {
        assert(getValueType() == VALUE_TYPE_TIMESTAMP);
        return *reinterpret_cast<const int64_t*>(m_data);
    }

    CUDAH int64_t& getTimestamp() {
        assert(getValueType() == VALUE_TYPE_TIMESTAMP);
        return *reinterpret_cast<int64_t*>(m_data);
    }

    CUDAH const double getDouble() const {
        assert(getValueType() == VALUE_TYPE_DOUBLE);
        return *reinterpret_cast<const double*>(m_data);
    }

    CUDAH double& getDouble() {
        assert(getValueType() == VALUE_TYPE_DOUBLE);
        return *reinterpret_cast<double*>(m_data);
    }

/*
    CUDAH const TTInt& getDecimal() const {
        assert(getValueType() == VALUE_TYPE_DECIMAL);
        const void* retval = reinterpret_cast<const void*>(m_data);
        return *reinterpret_cast<const TTInt*>(retval);
    }

    CUDAH TTInt& getDecimal() {
        assert(getValueType() == VALUE_TYPE_DECIMAL);
        void* retval = reinterpret_cast<void*>(m_data);
        return *reinterpret_cast<TTInt*>(retval);
    }
*/
    CUDAH int8_t getObjectLengthLength() const {
        return m_data[12];
    }

    CUDAH int32_t getObjectLength_withoutNull() const {
        assert(isNull() == false);
        assert(getValueType() == VALUE_TYPE_VARCHAR || getValueType() == VALUE_TYPE_VARBINARY);
        // now safe to read and return the length preceding value.
        return *reinterpret_cast<const int32_t *>(&m_data[8]);
    }

    CUDAH void* getObjectValue_withoutNull() const {
        void* value;
        if (m_sourceInlined) {
            value = *reinterpret_cast<char* const*>(m_data) + getObjectLengthLength();
        }
        else {
//            StringRef* sref = *reinterpret_cast<StringRef* const*>(m_data);
//            value = sref->get() + getObjectLengthLength();
        	printf("Not support non-inlined string. Return null...\n");
        	value = NULL;
        }
        return value;
    }

    CUDAH const bool& getBoolean() const {
        assert(getValueType() == VALUE_TYPE_BOOLEAN);
        return *reinterpret_cast<const bool*>(m_data);
    }

    CUDAH bool& getBoolean() {
        assert(getValueType() == VALUE_TYPE_BOOLEAN);
        return *reinterpret_cast<bool*>(m_data);
    }


    CUDAH int64_t castAsBigIntAndGetValue() const {
        assert(isNull() == false);

        const ValueType type = getValueType();
        assert(type != VALUE_TYPE_NULL);
        switch (type) {
        case VALUE_TYPE_TINYINT:
            return static_cast<int64_t>(getTinyInt());
        case VALUE_TYPE_SMALLINT:
            return static_cast<int64_t>(getSmallInt());
        case VALUE_TYPE_INTEGER:
            return static_cast<int64_t>(getInteger());
        case VALUE_TYPE_BIGINT:
            return getBigInt();
        case VALUE_TYPE_TIMESTAMP:
            return getTimestamp();
        case VALUE_TYPE_DOUBLE:
            if (getDouble() > (double)LONG_MAX || getDouble() < (double)GVOLT_INT64_MIN) {
                //throwCastSQLValueOutOfRangeException<double>(getDouble(), VALUE_TYPE_DOUBLE, VALUE_TYPE_BIGINT);
                return 0;//TO DO: undefined return value.
            }
            return static_cast<int64_t>(getDouble());
        case VALUE_TYPE_ADDRESS:
            return getBigInt();
        default:
            //throwCastSQLException(type, VALUE_TYPE_BIGINT);
            return 0; // NOT REACHED
        }
    }


    template<typename T>
        CUDAH int compareValue (const T lhsValue, const T rhsValue) const {
        if (lhsValue == rhsValue) {
            return VALUE_COMPARE_EQUAL;
        } else if (lhsValue > rhsValue){
            return VALUE_COMPARE_GREATERTHAN;
        } else {
            return VALUE_COMPARE_LESSTHAN;
        }
    }

    CUDAH int compareDoubleValue (const double lhsValue, const double rhsValue) const {
        // Treat NaN values as equals and also make them smaller than neagtive infinity.
        // This breaks IEEE754 for expressions slightly.
        
/*
        if(std::isnan(lhsValue)){
            return std::isnan(rhsValue) ? VALUE_COMPARE_EQUAL : VALUE_COMPARE_LESSTHAN;
        }
        else if (std::isnan(rhsValue)) {
          return VALUE_COMPARE_GREATERTHAN;
          }
        */

        if (lhsValue > rhsValue) {
            return VALUE_COMPARE_GREATERTHAN;
        }
        else if (lhsValue < rhsValue) {
            return VALUE_COMPARE_LESSTHAN;
        }
        else {
            return VALUE_COMPARE_EQUAL;
        }
    }

    CUDAH int compareTinyInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_TINYINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getTinyInt()), rhs.getDouble());
/*
        } 
        else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getTinyInt()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);        
*/
        }else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getTinyInt());
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareSmallInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_SMALLINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getSmallInt()), rhs.getDouble());
/*
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getSmallInt()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
*/
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getSmallInt());
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareInteger (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_INTEGER);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getInteger()), rhs.getDouble());
/*
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL){
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(static_cast<int64_t>(getInteger()));
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
*/
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = static_cast<int64_t>(getInteger());
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue,rhsValue);
        }

    }


    CUDAH int compareBigInt (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_BIGINT);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getBigInt()), rhs.getDouble());
/*
        } else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(getBigInt());
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
*/
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = getBigInt();
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
        return 0;
    }

    CUDAH int compareTimestamp (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_TIMESTAMP);

        // get the right hand side as a bigint
        if (rhs.getValueType() == VALUE_TYPE_DOUBLE) {
            return compareDoubleValue(static_cast<double>(getTimestamp()), rhs.getDouble());
/*
        }else if (rhs.getValueType() == VALUE_TYPE_DECIMAL) {
            const TTInt rhsValue = rhs.getDecimal();
            TTInt lhsValue(getTimestamp());
            lhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(lhsValue, rhsValue);
*/
        } else {
            int64_t lhsValue, rhsValue;
            lhsValue = getTimestamp();
            rhsValue = rhs.castAsBigIntAndGetValue();
            return compareValue<int64_t>(lhsValue, rhsValue);
        }
    }

    CUDAH int compareDoubleValue (const GNValue rhs) const {

        assert(m_valueType == VALUE_TYPE_DOUBLE);

        const double lhsValue = getDouble();
        double rhsValue;

        switch (rhs.getValueType()) {
        case VALUE_TYPE_DOUBLE:
            rhsValue = rhs.getDouble();
            break;
        case VALUE_TYPE_TINYINT:
            rhsValue = static_cast<double>(rhs.getTinyInt()); break;
        case VALUE_TYPE_SMALLINT:
            rhsValue = static_cast<double>(rhs.getSmallInt()); break;
        case VALUE_TYPE_INTEGER:
            rhsValue = static_cast<double>(rhs.getInteger()); break;
        case VALUE_TYPE_BIGINT:
            rhsValue = static_cast<double>(rhs.getBigInt()); break;
        case VALUE_TYPE_TIMESTAMP:
            rhsValue = static_cast<double>(rhs.getTimestamp()); break;
        case VALUE_TYPE_DECIMAL:
        {
/*
            TTInt scaledValue = rhs.getDecimal();
            TTInt whole(scaledValue);
            TTInt fractional(scaledValue);
            whole /= kMaxScaleFactor;
            fractional %= kMaxScaleFactor;
            rhsValue = static_cast<double>(whole.ToInt()) +
                (static_cast<double>(fractional.ToInt())/static_cast<double>(kMaxScaleFactor));
            break;
*/

        }
        default:

            return -3;
        }

        return compareDoubleValue(lhsValue, rhsValue);
    }

/*
    CUDAH int compareDecimalValue (const GNValue rhs) const {

        assert(m_valueType == VALUE_TYPE_DECIMAL);
        switch (rhs.getValueType()) {
        case VALUE_TYPE_DECIMAL:
        {
            return -3;//compareValue<TTInt>(getDecimal(), rhs.getDecimal());
        }
        case VALUE_TYPE_DOUBLE:
        {
            const double rhsValue = rhs.getDouble();
            TTInt scaledValue = getDecimal();
            TTInt whole(scaledValue);
            TTInt fractional(scaledValue);
            whole /= kMaxScaleFactor;
            fractional %= kMaxScaleFactor;
            const double lhsValue = static_cast<double>(whole.ToInt()) +
                (static_cast<double>(fractional.ToInt())/static_cast<double>(kMaxScaleFactor));

            return compareValue<double>(lhsValue, rhsValue);
        }
        // create the equivalent decimal value
        case VALUE_TYPE_TINYINT:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getTinyInt()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_SMALLINT:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getSmallInt()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_INTEGER:
        {
            TTInt rhsValue(static_cast<int64_t>(rhs.getInteger()));
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_BIGINT:
        {
            TTInt rhsValue(rhs.getBigInt());
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        case VALUE_TYPE_TIMESTAMP:
        {
            TTInt rhsValue(rhs.getTimestamp());
            rhsValue *= kMaxScaleFactor;
            return compareValue<TTInt>(getDecimal(), rhsValue);
        }
        default:
        {

            return -3;
        }
        }
    }

*/
    CUDAH int GStrcmp(const char *left, const char *right, int length) const
    {
    	int i = 0;

    	while (i < length) {
    		if (left[i] < right[i]) {
    			return -1;
    		} else if (left[i] > right[i]) {
    			return 1;
    		}
    		i++;
    	}

    	return 0;
    }

    //Compare string value
    CUDAH int compareStringValue (const GNValue rhs) const {
        assert(m_valueType == VALUE_TYPE_VARCHAR);

        ValueType rhsType = rhs.getValueType();
        if ((rhsType != VALUE_TYPE_VARCHAR) && (rhsType != VALUE_TYPE_VARBINARY)) {

        }

        assert(m_valueType == VALUE_TYPE_VARCHAR);

        const int32_t leftLength = getObjectLength_withoutNull();
        const int32_t rightLength = rhs.getObjectLength_withoutNull();
        const char* left = reinterpret_cast<const char*>(getObjectValue_withoutNull());
        const char* right = reinterpret_cast<const char*>(rhs.getObjectValue_withoutNull());
        int min_length = (leftLength <= rightLength) ? leftLength : rightLength;

        const int result = GStrcmp(left, right, min_length);
        if (result == 0 && leftLength != rightLength) {
            if (leftLength > rightLength) {
                return  VALUE_COMPARE_GREATERTHAN;
            } else {
                return VALUE_COMPARE_LESSTHAN;
            }
        }
        else if (result > 0) {
            return VALUE_COMPARE_GREATERTHAN;
        }
        else if (result < 0) {
            return VALUE_COMPARE_LESSTHAN;
        }

        return VALUE_COMPARE_EQUAL;
    }
    //End of comparing string value

};

/**
 * Public constructor that initializes to an NValue that is unusable
 * with other NValues.  Useful for declaring storage for an NValue.
 */
inline CUDAH GNValue::GNValue() {
    ::memset( m_data, 0, 16);
    setValueType(VALUE_TYPE_INVALID);
    m_sourceInlined = false;
}



/**
 * Set this NValue to null.
 */
inline CUDAH void GNValue::setNull() {
    tagAsNull(); // This gets overwritten for DECIMAL -- but that's OK.
    switch (getValueType())
    {
    case VALUE_TYPE_BOOLEAN:
        // HACK BOOL NULL
        *reinterpret_cast<int8_t*>(m_data) = GINT8_NULL;
        break;
    case VALUE_TYPE_NULL:
    case VALUE_TYPE_INVALID:
        return;
    case VALUE_TYPE_TINYINT:
        getTinyInt() = GINT8_NULL;
        break;
    case VALUE_TYPE_SMALLINT:
        getSmallInt() = GINT16_NULL;
        break;
    case VALUE_TYPE_INTEGER:
        getInteger() = GINT32_NULL;
        break;
    case VALUE_TYPE_TIMESTAMP:
        getTimestamp() = GINT64_NULL;
        break;
    case VALUE_TYPE_BIGINT:
        getBigInt() = GINT64_NULL;
        break;
    case VALUE_TYPE_DOUBLE:
        getDouble() = GDOUBLE_MIN;
        break;
    case VALUE_TYPE_VARCHAR:
    case VALUE_TYPE_VARBINARY:
    case VALUE_TYPE_DECIMAL:
        break;
    default: {
        break;
    }
    }
}



/**
 * Initialize an NValue of the specified type from the tuple
 * storage area provided. If this is an Object type then the third
 * argument indicates whether the object is stored in the tuple inline.
 */
inline CUDAH void GNValue::initFromTupleStorage(const void *storage, ValueType type,GNValue *retval)
{

    assert(type != VALUE_TYPE_VARCHAR && type != VALUE_TYPE_VARBINARY && type != VALUE_TYPE_DECIMAL);
/*
    if(type == VALUE_TYPE_VARCHAR || type == VALUE_TYPE_VARBINARY || type == VALUE_TYPE_DECIMAL){
        retval->tagAsNull();
        return;
    }
*/

    retval->setValueType(type);

    switch (type)
    {
    case VALUE_TYPE_INTEGER:
        if ((retval->getInteger() = *reinterpret_cast<const int32_t*>(storage)) == GINT32_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_BIGINT:
        if ((retval->getBigInt() = *reinterpret_cast<const int64_t*>(storage)) == GINT64_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_DOUBLE:
        if ((retval->getDouble() = *reinterpret_cast<const double*>(storage)) <= GDOUBLE_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_TIMESTAMP:
        if ((retval->getTimestamp() = *reinterpret_cast<const int64_t*>(storage)) == GINT64_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_TINYINT:
        if ((retval->getTinyInt() = *reinterpret_cast<const int8_t*>(storage)) == GINT8_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_SMALLINT:
        if ((retval->getSmallInt() = *reinterpret_cast<const int16_t*>(storage)) == GINT16_NULL) {
            retval->tagAsNull();
        }
        break;
    case VALUE_TYPE_VARCHAR:
    case VALUE_TYPE_VARBINARY:
    case VALUE_TYPE_DECIMAL:
    default:
        retval->tagAsNull();
    }
    //return retval;

}


inline CUDAH int GNValue::compare_withoutNull(const GNValue rhs) const {
    assert(isNull() == false && rhs.isNull() == false);
    //assert(m_valueType != VALUE_TYPE_VARCHAR && m_valueType != VALUE_TYPE_VARBINARY && m_valueType != VALUE_TYPE_DECIMAL);

    switch (m_valueType) {
    case VALUE_TYPE_BIGINT:
        return compareBigInt(rhs);
    case VALUE_TYPE_INTEGER:
        return compareInteger(rhs);
    case VALUE_TYPE_SMALLINT:
        return compareSmallInt(rhs);
   case VALUE_TYPE_TINYINT:
        return compareTinyInt(rhs);
    case VALUE_TYPE_TIMESTAMP:
        return compareTimestamp(rhs);
    case VALUE_TYPE_DOUBLE:
        return compareDoubleValue(rhs);
    case VALUE_TYPE_VARCHAR:
        return compareStringValue(rhs);
    case VALUE_TYPE_VARBINARY:
        //return compareBinaryValue(rhs);
    case VALUE_TYPE_DECIMAL:
        //return compareDecimalValue(rhs);

    default: {
        /*
          throwDynamicSQLException(
          "non comparable types lhs '%s' rhs '%s'",
          getValueTypeString().c_str(),
          rhs.getValueTypeString().c_str());
        */
        return -3;
    }
        /* no break */
    }
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
    return m_data[13] == OBJECT_NULL_BIT;
}


// without null comparison
inline CUDAH bool GNValue::op_equals_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp == 0;
}

inline CUDAH bool GNValue::op_notEquals_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp != 0;
}

inline CUDAH bool GNValue::op_lessThan_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp < 0;
}

inline CUDAH bool GNValue::op_lessThanOrEqual_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp <= 0;

}

inline CUDAH bool GNValue::op_greaterThan_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp > 0;
}

inline CUDAH bool GNValue::op_greaterThanOrEqual_withoutNull(const GNValue *rhs) const {
    int temp = compare_withoutNull(*rhs);
    if(temp == -3) return false;
    return temp >= 0;
}

inline CUDAH GNValue GNValue::op_negate(void) const {
	return (getBoolean()) ? getFalse() : getTrue();
}

inline CUDAH GNValue GNValue::op_and(const GNValue rhs) const {
	bool tmp = this->getBoolean() & rhs.getBoolean();

	return (tmp) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_or(const GNValue rhs) const {
	bool tmp = this->getBoolean() | rhs.getBoolean();

	return (tmp) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_equal(const GNValue rhs) const {
	return (op_equals_withoutNull(&rhs)) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_notEqual(const GNValue rhs) const {
	return (op_notEquals_withoutNull(&rhs)) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_lessThan(const GNValue rhs) const{
	return (op_lessThan_withoutNull(&rhs)) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_lessThanOrEqual(const GNValue rhs) const {
	return (op_lessThanOrEqual_withoutNull(&rhs)) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_greaterThan(const GNValue rhs) const {
	return (op_greaterThan_withoutNull(&rhs)) ? getTrue() : getFalse();
}

inline CUDAH GNValue GNValue::op_greaterThanOrEqual(const GNValue rhs) const {
	return (op_greaterThanOrEqual_withoutNull(&rhs)) ? getTrue() : getFalse();
}

} // namespace voltdb

#endif /* NVALUE_HPP_ */
