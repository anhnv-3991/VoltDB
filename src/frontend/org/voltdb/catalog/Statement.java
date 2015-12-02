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
public class Statement extends CatalogType {

    String m_sqltext = new String();
    int m_querytype;
    boolean m_readonly;
    boolean m_singlepartition;
    boolean m_replicatedtabledml;
    boolean m_iscontentdeterministic;
    boolean m_isorderdeterministic;
    String m_nondeterminismdetail = new String();
    CatalogMap<StmtParameter> m_parameters;
    CatalogMap<PlanFragment> m_fragments;
    int m_cost;
    int m_seqscancount;
    String m_explainplan = new String();
    String m_tablesread = new String();
    String m_tablesupdated = new String();
    String m_indexesused = new String();
    String m_cachekeyprefix = new String();

    @Override
    void initChildMaps() {
        m_parameters = new CatalogMap<StmtParameter>(getCatalog(), this, "parameters", StmtParameter.class, m_parentMap.m_depth + 1);
        m_fragments = new CatalogMap<PlanFragment>(getCatalog(), this, "fragments", PlanFragment.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "sqltext",
            "querytype",
            "readonly",
            "singlepartition",
            "replicatedtabledml",
            "iscontentdeterministic",
            "isorderdeterministic",
            "nondeterminismdetail",
            "cost",
            "seqscancount",
            "explainplan",
            "tablesread",
            "tablesupdated",
            "indexesused",
            "cachekeyprefix",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "parameters",
            "fragments",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "sqltext":
            return getSqltext();
        case "querytype":
            return getQuerytype();
        case "readonly":
            return getReadonly();
        case "singlepartition":
            return getSinglepartition();
        case "replicatedtabledml":
            return getReplicatedtabledml();
        case "iscontentdeterministic":
            return getIscontentdeterministic();
        case "isorderdeterministic":
            return getIsorderdeterministic();
        case "nondeterminismdetail":
            return getNondeterminismdetail();
        case "parameters":
            return getParameters();
        case "fragments":
            return getFragments();
        case "cost":
            return getCost();
        case "seqscancount":
            return getSeqscancount();
        case "explainplan":
            return getExplainplan();
        case "tablesread":
            return getTablesread();
        case "tablesupdated":
            return getTablesupdated();
        case "indexesused":
            return getIndexesused();
        case "cachekeyprefix":
            return getCachekeyprefix();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The text of the sql statement */
    public String getSqltext() {
        return m_sqltext;
    }

    public int getQuerytype() {
        return m_querytype;
    }

    /** GETTER: Can the statement modify any data? */
    public boolean getReadonly() {
        return m_readonly;
    }

    /** GETTER: Does the statement only use data on one partition? */
    public boolean getSinglepartition() {
        return m_singlepartition;
    }

    /** GETTER: Should the result of this statememt be divided by partition count before returned */
    public boolean getReplicatedtabledml() {
        return m_replicatedtabledml;
    }

    /** GETTER: Is the result of this statement deterministic not accounting for row order */
    public boolean getIscontentdeterministic() {
        return m_iscontentdeterministic;
    }

    /** GETTER: Is the result of this statement deterministic even accounting for row order */
    public boolean getIsorderdeterministic() {
        return m_isorderdeterministic;
    }

    /** GETTER: Explanation for any non-determinism in the statement result */
    public String getNondeterminismdetail() {
        return m_nondeterminismdetail;
    }

    /** GETTER: The set of parameters to this SQL statement */
    public CatalogMap<StmtParameter> getParameters() {
        return m_parameters;
    }

    /** GETTER: The set of plan fragments used to execute this statement */
    public CatalogMap<PlanFragment> getFragments() {
        return m_fragments;
    }

    /** GETTER: The cost of this plan measured in arbitrary units */
    public int getCost() {
        return m_cost;
    }

    /** GETTER: The number of sequential table scans in the plan */
    public int getSeqscancount() {
        return m_seqscancount;
    }

    /** GETTER: A human-readable plan description */
    public String getExplainplan() {
        return m_explainplan;
    }

    /** GETTER: A CSV list of tables this statement reads */
    public String getTablesread() {
        return m_tablesread;
    }

    /** GETTER: A CSV list of tables this statement may update */
    public String getTablesupdated() {
        return m_tablesupdated;
    }

    /** GETTER: A CSV list of indexes this statement may use” */
    public String getIndexesused() {
        return m_indexesused;
    }

    /** GETTER: Unique string that combines with the SQL text to identify a unique corresponding plan. */
    public String getCachekeyprefix() {
        return m_cachekeyprefix;
    }

    /** SETTER: The text of the sql statement */
    public void setSqltext(String value) {
        m_sqltext = value;
    }

    public void setQuerytype(int value) {
        m_querytype = value;
    }

    /** SETTER: Can the statement modify any data? */
    public void setReadonly(boolean value) {
        m_readonly = value;
    }

    /** SETTER: Does the statement only use data on one partition? */
    public void setSinglepartition(boolean value) {
        m_singlepartition = value;
    }

    /** SETTER: Should the result of this statememt be divided by partition count before returned */
    public void setReplicatedtabledml(boolean value) {
        m_replicatedtabledml = value;
    }

    /** SETTER: Is the result of this statement deterministic not accounting for row order */
    public void setIscontentdeterministic(boolean value) {
        m_iscontentdeterministic = value;
    }

    /** SETTER: Is the result of this statement deterministic even accounting for row order */
    public void setIsorderdeterministic(boolean value) {
        m_isorderdeterministic = value;
    }

    /** SETTER: Explanation for any non-determinism in the statement result */
    public void setNondeterminismdetail(String value) {
        m_nondeterminismdetail = value;
    }

    /** SETTER: The cost of this plan measured in arbitrary units */
    public void setCost(int value) {
        m_cost = value;
    }

    /** SETTER: The number of sequential table scans in the plan */
    public void setSeqscancount(int value) {
        m_seqscancount = value;
    }

    /** SETTER: A human-readable plan description */
    public void setExplainplan(String value) {
        m_explainplan = value;
    }

    /** SETTER: A CSV list of tables this statement reads */
    public void setTablesread(String value) {
        m_tablesread = value;
    }

    /** SETTER: A CSV list of tables this statement may update */
    public void setTablesupdated(String value) {
        m_tablesupdated = value;
    }

    /** SETTER: A CSV list of indexes this statement may use” */
    public void setIndexesused(String value) {
        m_indexesused = value;
    }

    /** SETTER: Unique string that combines with the SQL text to identify a unique corresponding plan. */
    public void setCachekeyprefix(String value) {
        m_cachekeyprefix = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "sqltext":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_sqltext = value;
            break;
        case "querytype":
            assert(value != null);
            m_querytype = Integer.parseInt(value);
            break;
        case "readonly":
            assert(value != null);
            m_readonly = Boolean.parseBoolean(value);
            break;
        case "singlepartition":
            assert(value != null);
            m_singlepartition = Boolean.parseBoolean(value);
            break;
        case "replicatedtabledml":
            assert(value != null);
            m_replicatedtabledml = Boolean.parseBoolean(value);
            break;
        case "iscontentdeterministic":
            assert(value != null);
            m_iscontentdeterministic = Boolean.parseBoolean(value);
            break;
        case "isorderdeterministic":
            assert(value != null);
            m_isorderdeterministic = Boolean.parseBoolean(value);
            break;
        case "nondeterminismdetail":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_nondeterminismdetail = value;
            break;
        case "cost":
            assert(value != null);
            m_cost = Integer.parseInt(value);
            break;
        case "seqscancount":
            assert(value != null);
            m_seqscancount = Integer.parseInt(value);
            break;
        case "explainplan":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_explainplan = value;
            break;
        case "tablesread":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_tablesread = value;
            break;
        case "tablesupdated":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_tablesupdated = value;
            break;
        case "indexesused":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_indexesused = value;
            break;
        case "cachekeyprefix":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_cachekeyprefix = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Statement other = (Statement) obj;

        other.m_sqltext = m_sqltext;
        other.m_querytype = m_querytype;
        other.m_readonly = m_readonly;
        other.m_singlepartition = m_singlepartition;
        other.m_replicatedtabledml = m_replicatedtabledml;
        other.m_iscontentdeterministic = m_iscontentdeterministic;
        other.m_isorderdeterministic = m_isorderdeterministic;
        other.m_nondeterminismdetail = m_nondeterminismdetail;
        other.m_parameters.copyFrom(m_parameters);
        other.m_fragments.copyFrom(m_fragments);
        other.m_cost = m_cost;
        other.m_seqscancount = m_seqscancount;
        other.m_explainplan = m_explainplan;
        other.m_tablesread = m_tablesread;
        other.m_tablesupdated = m_tablesupdated;
        other.m_indexesused = m_indexesused;
        other.m_cachekeyprefix = m_cachekeyprefix;
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
        Statement other = (Statement) obj;

        // are the fields / children the same? (deep compare)
        if ((m_sqltext == null) != (other.m_sqltext == null)) return false;
        if ((m_sqltext != null) && !m_sqltext.equals(other.m_sqltext)) return false;
        if (m_querytype != other.m_querytype) return false;
        if (m_readonly != other.m_readonly) return false;
        if (m_singlepartition != other.m_singlepartition) return false;
        if (m_replicatedtabledml != other.m_replicatedtabledml) return false;
        if (m_iscontentdeterministic != other.m_iscontentdeterministic) return false;
        if (m_isorderdeterministic != other.m_isorderdeterministic) return false;
        if ((m_nondeterminismdetail == null) != (other.m_nondeterminismdetail == null)) return false;
        if ((m_nondeterminismdetail != null) && !m_nondeterminismdetail.equals(other.m_nondeterminismdetail)) return false;
        if ((m_parameters == null) != (other.m_parameters == null)) return false;
        if ((m_parameters != null) && !m_parameters.equals(other.m_parameters)) return false;
        if ((m_fragments == null) != (other.m_fragments == null)) return false;
        if ((m_fragments != null) && !m_fragments.equals(other.m_fragments)) return false;
        if (m_cost != other.m_cost) return false;
        if (m_seqscancount != other.m_seqscancount) return false;
        if ((m_explainplan == null) != (other.m_explainplan == null)) return false;
        if ((m_explainplan != null) && !m_explainplan.equals(other.m_explainplan)) return false;
        if ((m_tablesread == null) != (other.m_tablesread == null)) return false;
        if ((m_tablesread != null) && !m_tablesread.equals(other.m_tablesread)) return false;
        if ((m_tablesupdated == null) != (other.m_tablesupdated == null)) return false;
        if ((m_tablesupdated != null) && !m_tablesupdated.equals(other.m_tablesupdated)) return false;
        if ((m_indexesused == null) != (other.m_indexesused == null)) return false;
        if ((m_indexesused != null) && !m_indexesused.equals(other.m_indexesused)) return false;
        if ((m_cachekeyprefix == null) != (other.m_cachekeyprefix == null)) return false;
        if ((m_cachekeyprefix != null) && !m_cachekeyprefix.equals(other.m_cachekeyprefix)) return false;

        return true;
    }

}
