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
 * Information
 */
public class MaterializedViewInfo extends CatalogType {

    Catalog.CatalogReference<Table> m_dest = new CatalogReference<>();
    CatalogMap<ColumnRef> m_groupbycols;
    String m_predicate = new String();
    String m_groupbyExpressionsJson = new String();
    String m_aggregationExpressionsJson = new String();
    String m_indexForMinMax = new String();

    @Override
    void initChildMaps() {
        m_groupbycols = new CatalogMap<ColumnRef>(getCatalog(), this, "groupbycols", ColumnRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "dest",
            "predicate",
            "groupbyExpressionsJson",
            "aggregationExpressionsJson",
            "indexForMinMax",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "groupbycols",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "dest":
            return getDest();
        case "groupbycols":
            return getGroupbycols();
        case "predicate":
            return getPredicate();
        case "groupbyExpressionsJson":
            return getGroupbyexpressionsjson();
        case "aggregationExpressionsJson":
            return getAggregationexpressionsjson();
        case "indexForMinMax":
            return getIndexforminmax();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The table which will be updated when the source table is updated */
    public Table getDest() {
        return m_dest.get();
    }

    /** GETTER: The columns involved in the group by of the aggregation */
    public CatalogMap<ColumnRef> getGroupbycols() {
        return m_groupbycols;
    }

    /** GETTER: A filtering predicate */
    public String getPredicate() {
        return m_predicate;
    }

    /** GETTER: A serialized representation of the groupby expression trees */
    public String getGroupbyexpressionsjson() {
        return m_groupbyExpressionsJson;
    }

    /** GETTER: A serialized representation of the aggregation expression trees */
    public String getAggregationexpressionsjson() {
        return m_aggregationExpressionsJson;
    }

    /** GETTER: The name of index on srcTable which can be used to maintain min()/max() */
    public String getIndexforminmax() {
        return m_indexForMinMax;
    }

    /** SETTER: The table which will be updated when the source table is updated */
    public void setDest(Table value) {
        m_dest.set(value);
    }

    /** SETTER: A filtering predicate */
    public void setPredicate(String value) {
        m_predicate = value;
    }

    /** SETTER: A serialized representation of the groupby expression trees */
    public void setGroupbyexpressionsjson(String value) {
        m_groupbyExpressionsJson = value;
    }

    /** SETTER: A serialized representation of the aggregation expression trees */
    public void setAggregationexpressionsjson(String value) {
        m_aggregationExpressionsJson = value;
    }

    /** SETTER: The name of index on srcTable which can be used to maintain min()/max() */
    public void setIndexforminmax(String value) {
        m_indexForMinMax = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "dest":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_dest.setUnresolved(value);
            break;
        case "predicate":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_predicate = value;
            break;
        case "groupbyExpressionsJson":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_groupbyExpressionsJson = value;
            break;
        case "aggregationExpressionsJson":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_aggregationExpressionsJson = value;
            break;
        case "indexForMinMax":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_indexForMinMax = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        MaterializedViewInfo other = (MaterializedViewInfo) obj;

        other.m_dest.setUnresolved(m_dest.getPath());
        other.m_groupbycols.copyFrom(m_groupbycols);
        other.m_predicate = m_predicate;
        other.m_groupbyExpressionsJson = m_groupbyExpressionsJson;
        other.m_aggregationExpressionsJson = m_aggregationExpressionsJson;
        other.m_indexForMinMax = m_indexForMinMax;
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
        MaterializedViewInfo other = (MaterializedViewInfo) obj;

        // are the fields / children the same? (deep compare)
        if ((m_dest == null) != (other.m_dest == null)) return false;
        if ((m_dest != null) && !m_dest.equals(other.m_dest)) return false;
        if ((m_groupbycols == null) != (other.m_groupbycols == null)) return false;
        if ((m_groupbycols != null) && !m_groupbycols.equals(other.m_groupbycols)) return false;
        if ((m_predicate == null) != (other.m_predicate == null)) return false;
        if ((m_predicate != null) && !m_predicate.equals(other.m_predicate)) return false;
        if ((m_groupbyExpressionsJson == null) != (other.m_groupbyExpressionsJson == null)) return false;
        if ((m_groupbyExpressionsJson != null) && !m_groupbyExpressionsJson.equals(other.m_groupbyExpressionsJson)) return false;
        if ((m_aggregationExpressionsJson == null) != (other.m_aggregationExpressionsJson == null)) return false;
        if ((m_aggregationExpressionsJson != null) && !m_aggregationExpressionsJson.equals(other.m_aggregationExpressionsJson)) return false;
        if ((m_indexForMinMax == null) != (other.m_indexForMinMax == null)) return false;
        if ((m_indexForMinMax != null) && !m_indexForMinMax.equals(other.m_indexForMinMax)) return false;

        return true;
    }

}
