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
public class Constraint extends CatalogType {

    int m_type;
    String m_oncommit = new String();
    Catalog.CatalogReference<Index> m_index = new CatalogReference<>();
    Catalog.CatalogReference<Table> m_foreignkeytable = new CatalogReference<>();
    CatalogMap<ColumnRef> m_foreignkeycols;

    @Override
    void initChildMaps() {
        m_foreignkeycols = new CatalogMap<ColumnRef>(getCatalog(), this, "foreignkeycols", ColumnRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "type",
            "oncommit",
            "index",
            "foreignkeytable",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "foreignkeycols",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "type":
            return getType();
        case "oncommit":
            return getOncommit();
        case "index":
            return getIndex();
        case "foreignkeytable":
            return getForeignkeytable();
        case "foreignkeycols":
            return getForeignkeycols();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The type of constraint */
    public int getType() {
        return m_type;
    }

    /** GETTER: (currently unused) */
    public String getOncommit() {
        return m_oncommit;
    }

    /** GETTER: The index used by this constraint (if needed) */
    public Index getIndex() {
        return m_index.get();
    }

    /** GETTER: The table referenced by the foreign key (if needed) */
    public Table getForeignkeytable() {
        return m_foreignkeytable.get();
    }

    /** GETTER: The columns in the foreign table referenced by the constraint (if needed) */
    public CatalogMap<ColumnRef> getForeignkeycols() {
        return m_foreignkeycols;
    }

    /** SETTER: The type of constraint */
    public void setType(int value) {
        m_type = value;
    }

    /** SETTER: (currently unused) */
    public void setOncommit(String value) {
        m_oncommit = value;
    }

    /** SETTER: The index used by this constraint (if needed) */
    public void setIndex(Index value) {
        m_index.set(value);
    }

    /** SETTER: The table referenced by the foreign key (if needed) */
    public void setForeignkeytable(Table value) {
        m_foreignkeytable.set(value);
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "type":
            assert(value != null);
            m_type = Integer.parseInt(value);
            break;
        case "oncommit":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_oncommit = value;
            break;
        case "index":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_index.setUnresolved(value);
            break;
        case "foreignkeytable":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_foreignkeytable.setUnresolved(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Constraint other = (Constraint) obj;

        other.m_type = m_type;
        other.m_oncommit = m_oncommit;
        other.m_index.setUnresolved(m_index.getPath());
        other.m_foreignkeytable.setUnresolved(m_foreignkeytable.getPath());
        other.m_foreignkeycols.copyFrom(m_foreignkeycols);
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
        Constraint other = (Constraint) obj;

        // are the fields / children the same? (deep compare)
        if (m_type != other.m_type) return false;
        if ((m_oncommit == null) != (other.m_oncommit == null)) return false;
        if ((m_oncommit != null) && !m_oncommit.equals(other.m_oncommit)) return false;
        if ((m_index == null) != (other.m_index == null)) return false;
        if ((m_index != null) && !m_index.equals(other.m_index)) return false;
        if ((m_foreignkeytable == null) != (other.m_foreignkeytable == null)) return false;
        if ((m_foreignkeytable != null) && !m_foreignkeytable.equals(other.m_foreignkeytable)) return false;
        if ((m_foreignkeycols == null) != (other.m_foreignkeycols == null)) return false;
        if ((m_foreignkeycols != null) && !m_foreignkeycols.equals(other.m_foreignkeycols)) return false;

        return true;
    }

}
