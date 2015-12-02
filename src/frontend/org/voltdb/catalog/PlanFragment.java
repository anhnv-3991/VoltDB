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
 * Instructions
 */
public class PlanFragment extends CatalogType {

    boolean m_hasdependencies;
    boolean m_multipartition;
    String m_plannodetree = new String();
    boolean m_nontransactional;
    String m_planhash = new String();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "hasdependencies",
            "multipartition",
            "plannodetree",
            "nontransactional",
            "planhash",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "hasdependencies":
            return getHasdependencies();
        case "multipartition":
            return getMultipartition();
        case "plannodetree":
            return getPlannodetree();
        case "nontransactional":
            return getNontransactional();
        case "planhash":
            return getPlanhash();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Dependencies must be received before this plan fragment can execute */
    public boolean getHasdependencies() {
        return m_hasdependencies;
    }

    /** GETTER: Should this plan fragment be sent to all partitions */
    public boolean getMultipartition() {
        return m_multipartition;
    }

    /** GETTER: A serialized representation of the plan-graph/plan-pipeline */
    public String getPlannodetree() {
        return m_plannodetree;
    }

    /** GETTER: True if this fragment doesn't read from or write to any persistent tables */
    public boolean getNontransactional() {
        return m_nontransactional;
    }

    /** GETTER: SHA-1 Hash of the plan assumed to be unique */
    public String getPlanhash() {
        return m_planhash;
    }

    /** SETTER: Dependencies must be received before this plan fragment can execute */
    public void setHasdependencies(boolean value) {
        m_hasdependencies = value;
    }

    /** SETTER: Should this plan fragment be sent to all partitions */
    public void setMultipartition(boolean value) {
        m_multipartition = value;
    }

    /** SETTER: A serialized representation of the plan-graph/plan-pipeline */
    public void setPlannodetree(String value) {
        m_plannodetree = value;
    }

    /** SETTER: True if this fragment doesn't read from or write to any persistent tables */
    public void setNontransactional(boolean value) {
        m_nontransactional = value;
    }

    /** SETTER: SHA-1 Hash of the plan assumed to be unique */
    public void setPlanhash(String value) {
        m_planhash = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "hasdependencies":
            assert(value != null);
            m_hasdependencies = Boolean.parseBoolean(value);
            break;
        case "multipartition":
            assert(value != null);
            m_multipartition = Boolean.parseBoolean(value);
            break;
        case "plannodetree":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_plannodetree = value;
            break;
        case "nontransactional":
            assert(value != null);
            m_nontransactional = Boolean.parseBoolean(value);
            break;
        case "planhash":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_planhash = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        PlanFragment other = (PlanFragment) obj;

        other.m_hasdependencies = m_hasdependencies;
        other.m_multipartition = m_multipartition;
        other.m_plannodetree = m_plannodetree;
        other.m_nontransactional = m_nontransactional;
        other.m_planhash = m_planhash;
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
        PlanFragment other = (PlanFragment) obj;

        // are the fields / children the same? (deep compare)
        if (m_hasdependencies != other.m_hasdependencies) return false;
        if (m_multipartition != other.m_multipartition) return false;
        if ((m_plannodetree == null) != (other.m_plannodetree == null)) return false;
        if ((m_plannodetree != null) && !m_plannodetree.equals(other.m_plannodetree)) return false;
        if (m_nontransactional != other.m_nontransactional) return false;
        if ((m_planhash == null) != (other.m_planhash == null)) return false;
        if ((m_planhash != null) && !m_planhash.equals(other.m_planhash)) return false;

        return true;
    }

}
