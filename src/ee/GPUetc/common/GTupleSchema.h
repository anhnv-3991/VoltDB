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

#ifndef GTUPLESCHEMA_H_
#define GTUPLESCHEMA_H_

#include <iostream>
#include <cassert>
#include <cstring>
#include <stdint.h>
#include <string>

#include "common/types.h"
#include "common/value_defs.h"

#include "GPUetc/common/GNValue.h"
#include "GPUetc/common/Gtabletuple.h"
#include "GPUetc/cudaheader.h"

namespace voltdb {

/**
 * Represents the schema of a tuple or table row. Used to define table rows, as
 * well as index keys. Note: due to arbitrary size embedded array data, this class
 * cannot be created on the stack; all constructors are private.
 */
class GTupleSchema {
public:
    // holds per column info
    struct ColumnInfo {
        uint32_t offset;
        uint32_t length;   // does not include length prefix for ObjectTypes
        char type;
        char allowNull;
        bool inlined;      // Stored inside the tuple or outside the tuple.

        bool inBytes;

        CUDAH const ValueType getVoltType() const {
            return static_cast<ValueType>(type);
        }
    };

    /** Return the number of columns in the schema for the tuple. */
    inline CUDAH uint16_t columnCount() const { return m_columnCount; };

    inline CUDAH uint16_t getUninlinedObjectColumnCount() const { return m_uninlinedObjectColumnCount; }

    /** Return the number of bytes used by one tuple. */
    inline CUDAH uint32_t tupleLength() const;


    CUDAH const ColumnInfo* getColumnInfo(int columnIndex) const;
    CUDAH ColumnInfo* getColumnInfo(int columnIndex);
    GTupleSchema *createGTupleSchema(const TupleSchema *schema);

private:


    // can't (shouldn't) call constructors or destructor
    // prevents TupleSchema from being created on the stack
    CUDAH GTupleSchema() {};
    CUDAH ~GTupleSchema() {};

    // number of columns
    uint16_t m_columnCount;
    uint16_t m_uninlinedObjectColumnCount;

    /*
     * Data storage for column info and for indices of string columns
     */
    char m_data[0];
};

inline CUDAH const GTupleSchema::ColumnInfo* GTupleSchema::getColumnInfo(int columnIndex) const {
    return &reinterpret_cast<const ColumnInfo*>(m_data + (sizeof(uint16_t) * m_uninlinedObjectColumnCount))[columnIndex];
}

inline CUDAH GTupleSchema::ColumnInfo* GTupleSchema::getColumnInfo(int columnIndex) {
    return &reinterpret_cast<ColumnInfo*>(m_data + (sizeof(uint16_t) * m_uninlinedObjectColumnCount))[columnIndex];
}

inline CUDAH uint32_t GTupleSchema::tupleLength() const {
    // index "m_count" has the offset for the end of the tuple
    // index "m_count-1" has the offset for the last column
    return getColumnInfo(m_columnCount)->offset;
}


} // namespace voltdb


#endif // TUPLESCHEMA_H_
