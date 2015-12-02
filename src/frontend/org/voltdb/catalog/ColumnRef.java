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
public class ColumnRef extends CatalogType {

    int m_index;
    Catalog.CatalogReference<Column> m_column = new CatalogReference<>();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "index",
            "column",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "index":
            return getIndex();
        case "column":
            return getColumn();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The index within the set */
    public int getIndex() {
        return m_index;
    }

    /** GETTER: The table column being referenced */
    public Column getColumn() {
        return m_column.get();
    }

    /** SETTER: The index within the set */
    public void setIndex(int value) {
        m_index = value;
    }

    /** SETTER: The table column being referenced */
    public void setColumn(Column value) {
        m_column.set(value);
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "index":
            assert(value != null);
            m_index = Integer.parseInt(value);
            break;
        case "column":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_column.setUnresolved(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        ColumnRef other = (ColumnRef) obj;

        other.m_index = m_index;
        other.m_column.setUnresolved(m_column.getPath());
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
        ColumnRef other = (ColumnRef) obj;

        // are the fields / children the same? (deep compare)
        if (m_index != other.m_index) return false;
        if ((m_column == null) != (other.m_column == null)) return false;
        if ((m_column != null) && !m_column.equals(other.m_column)) return false;

        return true;
    }

}
