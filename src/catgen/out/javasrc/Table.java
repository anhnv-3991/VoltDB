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
public class Table extends CatalogType {

    CatalogMap<Column> m_columns;
    CatalogMap<Index> m_indexes;
    CatalogMap<Constraint> m_constraints;
    boolean m_isreplicated;
    Catalog.CatalogReference<Column> m_partitioncolumn = new CatalogReference<>();
    int m_estimatedtuplecount;
    CatalogMap<MaterializedViewInfo> m_views;
    Catalog.CatalogReference<Table> m_materializer = new CatalogReference<>();
    String m_signature = new String();
    int m_tuplelimit;
    CatalogMap<Statement> m_tuplelimitDeleteStmt;

    @Override
    void initChildMaps() {
        m_columns = new CatalogMap<Column>(getCatalog(), this, "columns", Column.class, m_parentMap.m_depth + 1);
        m_indexes = new CatalogMap<Index>(getCatalog(), this, "indexes", Index.class, m_parentMap.m_depth + 1);
        m_constraints = new CatalogMap<Constraint>(getCatalog(), this, "constraints", Constraint.class, m_parentMap.m_depth + 1);
        m_views = new CatalogMap<MaterializedViewInfo>(getCatalog(), this, "views", MaterializedViewInfo.class, m_parentMap.m_depth + 1);
        m_tuplelimitDeleteStmt = new CatalogMap<Statement>(getCatalog(), this, "tuplelimitDeleteStmt", Statement.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "isreplicated",
            "partitioncolumn",
            "estimatedtuplecount",
            "materializer",
            "signature",
            "tuplelimit",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "columns",
            "indexes",
            "constraints",
            "views",
            "tuplelimitDeleteStmt",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "columns":
            return getColumns();
        case "indexes":
            return getIndexes();
        case "constraints":
            return getConstraints();
        case "isreplicated":
            return getIsreplicated();
        case "partitioncolumn":
            return getPartitioncolumn();
        case "estimatedtuplecount":
            return getEstimatedtuplecount();
        case "views":
            return getViews();
        case "materializer":
            return getMaterializer();
        case "signature":
            return getSignature();
        case "tuplelimit":
            return getTuplelimit();
        case "tuplelimitDeleteStmt":
            return getTuplelimitdeletestmt();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The set of columns in the table */
    public CatalogMap<Column> getColumns() {
        return m_columns;
    }

    /** GETTER: The set of indexes on the columns in the table */
    public CatalogMap<Index> getIndexes() {
        return m_indexes;
    }

    /** GETTER: The set of constraints on the table */
    public CatalogMap<Constraint> getConstraints() {
        return m_constraints;
    }

    /** GETTER: Is the table replicated? */
    public boolean getIsreplicated() {
        return m_isreplicated;
    }

    /** GETTER: On which column is the table partitioned */
    public Column getPartitioncolumn() {
        return m_partitioncolumn.get();
    }

    /** GETTER: A rough estimate of the number of tuples in the table; used for planning */
    public int getEstimatedtuplecount() {
        return m_estimatedtuplecount;
    }

    /** GETTER: Information about materialized views based on this table's content */
    public CatalogMap<MaterializedViewInfo> getViews() {
        return m_views;
    }

    /** GETTER: If this is a materialized view, this field stores the source table */
    public Table getMaterializer() {
        return m_materializer.get();
    }

    /** GETTER: Catalog version independent signature of the table consisting of name and schema */
    public String getSignature() {
        return m_signature;
    }

    /** GETTER: A maximum number of rows in a table */
    public int getTuplelimit() {
        return m_tuplelimit;
    }

    /** GETTER: Delete statement to execute if tuple limit will be exceeded */
    public CatalogMap<Statement> getTuplelimitdeletestmt() {
        return m_tuplelimitDeleteStmt;
    }

    /** SETTER: Is the table replicated? */
    public void setIsreplicated(boolean value) {
        m_isreplicated = value;
    }

    /** SETTER: On which column is the table partitioned */
    public void setPartitioncolumn(Column value) {
        m_partitioncolumn.set(value);
    }

    /** SETTER: A rough estimate of the number of tuples in the table; used for planning */
    public void setEstimatedtuplecount(int value) {
        m_estimatedtuplecount = value;
    }

    /** SETTER: If this is a materialized view, this field stores the source table */
    public void setMaterializer(Table value) {
        m_materializer.set(value);
    }

    /** SETTER: Catalog version independent signature of the table consisting of name and schema */
    public void setSignature(String value) {
        m_signature = value;
    }

    /** SETTER: A maximum number of rows in a table */
    public void setTuplelimit(int value) {
        m_tuplelimit = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "isreplicated":
            assert(value != null);
            m_isreplicated = Boolean.parseBoolean(value);
            break;
        case "partitioncolumn":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_partitioncolumn.setUnresolved(value);
            break;
        case "estimatedtuplecount":
            assert(value != null);
            m_estimatedtuplecount = Integer.parseInt(value);
            break;
        case "materializer":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_materializer.setUnresolved(value);
            break;
        case "signature":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_signature = value;
            break;
        case "tuplelimit":
            assert(value != null);
            m_tuplelimit = Integer.parseInt(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Table other = (Table) obj;

        other.m_columns.copyFrom(m_columns);
        other.m_indexes.copyFrom(m_indexes);
        other.m_constraints.copyFrom(m_constraints);
        other.m_isreplicated = m_isreplicated;
        other.m_partitioncolumn.setUnresolved(m_partitioncolumn.getPath());
        other.m_estimatedtuplecount = m_estimatedtuplecount;
        other.m_views.copyFrom(m_views);
        other.m_materializer.setUnresolved(m_materializer.getPath());
        other.m_signature = m_signature;
        other.m_tuplelimit = m_tuplelimit;
        other.m_tuplelimitDeleteStmt.copyFrom(m_tuplelimitDeleteStmt);
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
        Table other = (Table) obj;

        // are the fields / children the same? (deep compare)
        if ((m_columns == null) != (other.m_columns == null)) return false;
        if ((m_columns != null) && !m_columns.equals(other.m_columns)) return false;
        if ((m_indexes == null) != (other.m_indexes == null)) return false;
        if ((m_indexes != null) && !m_indexes.equals(other.m_indexes)) return false;
        if ((m_constraints == null) != (other.m_constraints == null)) return false;
        if ((m_constraints != null) && !m_constraints.equals(other.m_constraints)) return false;
        if (m_isreplicated != other.m_isreplicated) return false;
        if ((m_partitioncolumn == null) != (other.m_partitioncolumn == null)) return false;
        if ((m_partitioncolumn != null) && !m_partitioncolumn.equals(other.m_partitioncolumn)) return false;
        if (m_estimatedtuplecount != other.m_estimatedtuplecount) return false;
        if ((m_views == null) != (other.m_views == null)) return false;
        if ((m_views != null) && !m_views.equals(other.m_views)) return false;
        if ((m_materializer == null) != (other.m_materializer == null)) return false;
        if ((m_materializer != null) && !m_materializer.equals(other.m_materializer)) return false;
        if ((m_signature == null) != (other.m_signature == null)) return false;
        if ((m_signature != null) && !m_signature.equals(other.m_signature)) return false;
        if (m_tuplelimit != other.m_tuplelimit) return false;
        if ((m_tuplelimitDeleteStmt == null) != (other.m_tuplelimitDeleteStmt == null)) return false;
        if ((m_tuplelimitDeleteStmt != null) && !m_tuplelimitDeleteStmt.equals(other.m_tuplelimitDeleteStmt)) return false;

        return true;
    }

}
