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
public class Column extends CatalogType {

    int m_index;
    int m_type;
    int m_size;
    boolean m_nullable;
    String m_name = new String();
    String m_defaultvalue = new String();
    int m_defaulttype;
    CatalogMap<ConstraintRef> m_constraints;
    Catalog.CatalogReference<MaterializedViewInfo> m_matview = new CatalogReference<>();
    int m_aggregatetype;
    Catalog.CatalogReference<Column> m_matviewsource = new CatalogReference<>();
    boolean m_inbytes;

    @Override
    void initChildMaps() {
        m_constraints = new CatalogMap<ConstraintRef>(getCatalog(), this, "constraints", ConstraintRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "index",
            "type",
            "size",
            "nullable",
            "name",
            "defaultvalue",
            "defaulttype",
            "matview",
            "aggregatetype",
            "matviewsource",
            "inbytes",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "constraints",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "index":
            return getIndex();
        case "type":
            return getType();
        case "size":
            return getSize();
        case "nullable":
            return getNullable();
        case "name":
            return getName();
        case "defaultvalue":
            return getDefaultvalue();
        case "defaulttype":
            return getDefaulttype();
        case "constraints":
            return getConstraints();
        case "matview":
            return getMatview();
        case "aggregatetype":
            return getAggregatetype();
        case "matviewsource":
            return getMatviewsource();
        case "inbytes":
            return getInbytes();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The column's order in the table */
    public int getIndex() {
        return m_index;
    }

    /** GETTER: The type of the column (int/double/date/etc) */
    public int getType() {
        return m_type;
    }

    /** GETTER: (currently unused) */
    public int getSize() {
        return m_size;
    }

    /** GETTER: Is the column nullable? */
    public boolean getNullable() {
        return m_nullable;
    }

    /** GETTER: Name of column */
    public String getName() {
        return m_name;
    }

    /** GETTER: Default value of the column */
    public String getDefaultvalue() {
        return m_defaultvalue;
    }

    /** GETTER: Type of the default value of the column */
    public int getDefaulttype() {
        return m_defaulttype;
    }

    /** GETTER: Constraints that use this column */
    public CatalogMap<ConstraintRef> getConstraints() {
        return m_constraints;
    }

    /** GETTER: If part of a materialized view, ref of view info */
    public MaterializedViewInfo getMatview() {
        return m_matview.get();
    }

    /** GETTER: If part of a materialized view, represents aggregate type */
    public int getAggregatetype() {
        return m_aggregatetype;
    }

    /** GETTER: If part of a materialized view, represents source column */
    public Column getMatviewsource() {
        return m_matviewsource.get();
    }

    /** GETTER: If a varchar column and size was specified in bytes */
    public boolean getInbytes() {
        return m_inbytes;
    }

    /** SETTER: The column's order in the table */
    public void setIndex(int value) {
        m_index = value;
    }

    /** SETTER: The type of the column (int/double/date/etc) */
    public void setType(int value) {
        m_type = value;
    }

    /** SETTER: (currently unused) */
    public void setSize(int value) {
        m_size = value;
    }

    /** SETTER: Is the column nullable? */
    public void setNullable(boolean value) {
        m_nullable = value;
    }

    /** SETTER: Name of column */
    public void setName(String value) {
        m_name = value;
    }

    /** SETTER: Default value of the column */
    public void setDefaultvalue(String value) {
        m_defaultvalue = value;
    }

    /** SETTER: Type of the default value of the column */
    public void setDefaulttype(int value) {
        m_defaulttype = value;
    }

    /** SETTER: If part of a materialized view, ref of view info */
    public void setMatview(MaterializedViewInfo value) {
        m_matview.set(value);
    }

    /** SETTER: If part of a materialized view, represents aggregate type */
    public void setAggregatetype(int value) {
        m_aggregatetype = value;
    }

    /** SETTER: If part of a materialized view, represents source column */
    public void setMatviewsource(Column value) {
        m_matviewsource.set(value);
    }

    /** SETTER: If a varchar column and size was specified in bytes */
    public void setInbytes(boolean value) {
        m_inbytes = value;
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
        case "type":
            assert(value != null);
            m_type = Integer.parseInt(value);
            break;
        case "size":
            assert(value != null);
            m_size = Integer.parseInt(value);
            break;
        case "nullable":
            assert(value != null);
            m_nullable = Boolean.parseBoolean(value);
            break;
        case "name":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_name = value;
            break;
        case "defaultvalue":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_defaultvalue = value;
            break;
        case "defaulttype":
            assert(value != null);
            m_defaulttype = Integer.parseInt(value);
            break;
        case "matview":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_matview.setUnresolved(value);
            break;
        case "aggregatetype":
            assert(value != null);
            m_aggregatetype = Integer.parseInt(value);
            break;
        case "matviewsource":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_matviewsource.setUnresolved(value);
            break;
        case "inbytes":
            assert(value != null);
            m_inbytes = Boolean.parseBoolean(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Column other = (Column) obj;

        other.m_index = m_index;
        other.m_type = m_type;
        other.m_size = m_size;
        other.m_nullable = m_nullable;
        other.m_name = m_name;
        other.m_defaultvalue = m_defaultvalue;
        other.m_defaulttype = m_defaulttype;
        other.m_constraints.copyFrom(m_constraints);
        other.m_matview.setUnresolved(m_matview.getPath());
        other.m_aggregatetype = m_aggregatetype;
        other.m_matviewsource.setUnresolved(m_matviewsource.getPath());
        other.m_inbytes = m_inbytes;
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
        Column other = (Column) obj;

        // are the fields / children the same? (deep compare)
        if (m_index != other.m_index) return false;
        if (m_type != other.m_type) return false;
        if (m_size != other.m_size) return false;
        if (m_nullable != other.m_nullable) return false;
        if ((m_name == null) != (other.m_name == null)) return false;
        if ((m_name != null) && !m_name.equals(other.m_name)) return false;
        if ((m_defaultvalue == null) != (other.m_defaultvalue == null)) return false;
        if ((m_defaultvalue != null) && !m_defaultvalue.equals(other.m_defaultvalue)) return false;
        if (m_defaulttype != other.m_defaulttype) return false;
        if ((m_constraints == null) != (other.m_constraints == null)) return false;
        if ((m_constraints != null) && !m_constraints.equals(other.m_constraints)) return false;
        if ((m_matview == null) != (other.m_matview == null)) return false;
        if ((m_matview != null) && !m_matview.equals(other.m_matview)) return false;
        if (m_aggregatetype != other.m_aggregatetype) return false;
        if ((m_matviewsource == null) != (other.m_matviewsource == null)) return false;
        if ((m_matviewsource != null) && !m_matviewsource.equals(other.m_matviewsource)) return false;
        if (m_inbytes != other.m_inbytes) return false;

        return true;
    }

}
