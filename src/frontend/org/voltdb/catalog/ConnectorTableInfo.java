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
 * Per-export
 */
public class ConnectorTableInfo extends CatalogType {

    Catalog.CatalogReference<Table> m_table = new CatalogReference<>();
    boolean m_appendOnly;

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "table",
            "appendOnly",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "table":
            return getTable();
        case "appendOnly":
            return getAppendonly();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Reference to the table being appended */
    public Table getTable() {
        return m_table.get();
    }

    /** GETTER: DEPRECATED: True if this table is an append-only table for export. */
    public boolean getAppendonly() {
        return m_appendOnly;
    }

    /** SETTER: Reference to the table being appended */
    public void setTable(Table value) {
        m_table.set(value);
    }

    /** SETTER: DEPRECATED: True if this table is an append-only table for export. */
    public void setAppendonly(boolean value) {
        m_appendOnly = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "table":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_table.setUnresolved(value);
            break;
        case "appendOnly":
            assert(value != null);
            m_appendOnly = Boolean.parseBoolean(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        ConnectorTableInfo other = (ConnectorTableInfo) obj;

        other.m_table.setUnresolved(m_table.getPath());
        other.m_appendOnly = m_appendOnly;
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
        ConnectorTableInfo other = (ConnectorTableInfo) obj;

        // are the fields / children the same? (deep compare)
        if ((m_table == null) != (other.m_table == null)) return false;
        if ((m_table != null) && !m_table.equals(other.m_table)) return false;
        if (m_appendOnly != other.m_appendOnly) return false;

        return true;
    }

}
