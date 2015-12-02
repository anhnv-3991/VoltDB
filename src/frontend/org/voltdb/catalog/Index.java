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
 * A
 */
public class Index extends CatalogType {

    boolean m_unique;
    boolean m_assumeUnique;
    boolean m_countable;
    int m_type;
    CatalogMap<ColumnRef> m_columns;
    String m_expressionsjson = new String();

    @Override
    void initChildMaps() {
        m_columns = new CatalogMap<ColumnRef>(getCatalog(), this, "columns", ColumnRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "unique",
            "assumeUnique",
            "countable",
            "type",
            "expressionsjson",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "columns",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "unique":
            return getUnique();
        case "assumeUnique":
            return getAssumeunique();
        case "countable":
            return getCountable();
        case "type":
            return getType();
        case "columns":
            return getColumns();
        case "expressionsjson":
            return getExpressionsjson();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: May the index contain duplicate keys? */
    public boolean getUnique() {
        return m_unique;
    }

    /** GETTER: User allow unique index on partition table without including partition key? */
    public boolean getAssumeunique() {
        return m_assumeUnique;
    }

    /** GETTER: Index counter feature */
    public boolean getCountable() {
        return m_countable;
    }

    /** GETTER: What data structure is the index using and what kinds of keys does it support? */
    public int getType() {
        return m_type;
    }

    /** GETTER: Columns referenced by the index */
    public CatalogMap<ColumnRef> getColumns() {
        return m_columns;
    }

    /** GETTER: A serialized representation of the optional expression trees */
    public String getExpressionsjson() {
        return m_expressionsjson;
    }

    /** SETTER: May the index contain duplicate keys? */
    public void setUnique(boolean value) {
        m_unique = value;
    }

    /** SETTER: User allow unique index on partition table without including partition key? */
    public void setAssumeunique(boolean value) {
        m_assumeUnique = value;
    }

    /** SETTER: Index counter feature */
    public void setCountable(boolean value) {
        m_countable = value;
    }

    /** SETTER: What data structure is the index using and what kinds of keys does it support? */
    public void setType(int value) {
        m_type = value;
    }

    /** SETTER: A serialized representation of the optional expression trees */
    public void setExpressionsjson(String value) {
        m_expressionsjson = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "unique":
            assert(value != null);
            m_unique = Boolean.parseBoolean(value);
            break;
        case "assumeUnique":
            assert(value != null);
            m_assumeUnique = Boolean.parseBoolean(value);
            break;
        case "countable":
            assert(value != null);
            m_countable = Boolean.parseBoolean(value);
            break;
        case "type":
            assert(value != null);
            m_type = Integer.parseInt(value);
            break;
        case "expressionsjson":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_expressionsjson = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Index other = (Index) obj;

        other.m_unique = m_unique;
        other.m_assumeUnique = m_assumeUnique;
        other.m_countable = m_countable;
        other.m_type = m_type;
        other.m_columns.copyFrom(m_columns);
        other.m_expressionsjson = m_expressionsjson;
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
        Index other = (Index) obj;

        // are the fields / children the same? (deep compare)
        if (m_unique != other.m_unique) return false;
        if (m_assumeUnique != other.m_assumeUnique) return false;
        if (m_countable != other.m_countable) return false;
        if (m_type != other.m_type) return false;
        if ((m_columns == null) != (other.m_columns == null)) return false;
        if ((m_columns != null) && !m_columns.equals(other.m_columns)) return false;
        if ((m_expressionsjson == null) != (other.m_expressionsjson == null)) return false;
        if ((m_expressionsjson != null) && !m_expressionsjson.equals(other.m_expressionsjson)) return false;

        return true;
    }

}
