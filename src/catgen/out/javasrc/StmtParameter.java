/* This file is part of VoltDB.
 * Copyright (C) 2008-2015 VoltDB Inc.
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

/* WARNING: THIS FILE IS AUTO-GENERATED
            DO NOT MODIFY THIS SOURCE
            ALL CHANGES MUST BE MADE IN THE CATALOG GENERATOR */

package org.voltdb.catalog;

/**
 * A parameter for a parameterized SQL statement
 */
public class StmtParameter extends CatalogType {

    int m_sqltype;
    int m_javatype;
    boolean m_isarray;
    int m_index;

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "sqltype",
            "javatype",
            "isarray",
            "index",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "sqltype":
            return getSqltype();
        case "javatype":
            return getJavatype();
        case "isarray":
            return getIsarray();
        case "index":
            return getIndex();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The SQL type of the parameter (int/float/date/etc) */
    public int getSqltype() {
        return m_sqltype;
    }

    /** GETTER: The Java class of the parameter (int/float/date/etc) */
    public int getJavatype() {
        return m_javatype;
    }

    /** GETTER: Is the parameter an array of value */
    public boolean getIsarray() {
        return m_isarray;
    }

    /** GETTER: The index of the parameter in the set of statement parameters */
    public int getIndex() {
        return m_index;
    }

    /** SETTER: The SQL type of the parameter (int/float/date/etc) */
    public void setSqltype(int value) {
        m_sqltype = value;
    }

    /** SETTER: The Java class of the parameter (int/float/date/etc) */
    public void setJavatype(int value) {
        m_javatype = value;
    }

    /** SETTER: Is the parameter an array of value */
    public void setIsarray(boolean value) {
        m_isarray = value;
    }

    /** SETTER: The index of the parameter in the set of statement parameters */
    public void setIndex(int value) {
        m_index = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "sqltype":
            assert(value != null);
            m_sqltype = Integer.parseInt(value);
            break;
        case "javatype":
            assert(value != null);
            m_javatype = Integer.parseInt(value);
            break;
        case "isarray":
            assert(value != null);
            m_isarray = Boolean.parseBoolean(value);
            break;
        case "index":
            assert(value != null);
            m_index = Integer.parseInt(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        StmtParameter other = (StmtParameter) obj;

        other.m_sqltype = m_sqltype;
        other.m_javatype = m_javatype;
        other.m_isarray = m_isarray;
        other.m_index = m_index;
    }

    public boolean equals(Object obj) {
        // this isn't really the convention for null handling
        if ((obj == null) || (obj.getClass().equals(getClass()) == false))
            return false;

        // Do the identity check
        if (obj == this)
            return true;

        // this is safe because of the class check
        // it is also known that the childCollections var will be the same
        //  from the class check
        StmtParameter other = (StmtParameter) obj;

        // are the fields / children the same? (deep compare)
        if (m_sqltype != other.m_sqltype) return false;
        if (m_javatype != other.m_javatype) return false;
        if (m_isarray != other.m_isarray) return false;
        if (m_index != other.m_index) return false;

        return true;
    }

}
