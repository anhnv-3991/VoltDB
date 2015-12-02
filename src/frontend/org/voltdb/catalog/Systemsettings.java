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
 * Container for deployment systemsettings element
 */
public class Systemsettings extends CatalogType {

    int m_temptablemaxsize;
    int m_snapshotpriority;
    int m_elasticduration;
    int m_elasticthroughput;
    int m_querytimeout;

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "temptablemaxsize",
            "snapshotpriority",
            "elasticduration",
            "elasticthroughput",
            "querytimeout",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "temptablemaxsize":
            return getTemptablemaxsize();
        case "snapshotpriority":
            return getSnapshotpriority();
        case "elasticduration":
            return getElasticduration();
        case "elasticthroughput":
            return getElasticthroughput();
        case "querytimeout":
            return getQuerytimeout();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The maximum allocation size for temp tables in the EE */
    public int getTemptablemaxsize() {
        return m_temptablemaxsize;
    }

    /** GETTER: The priority of snapshot work */
    public int getSnapshotpriority() {
        return m_snapshotpriority;
    }

    /** GETTER: Maximum duration time for rebalancing */
    public int getElasticduration() {
        return m_elasticduration;
    }

    /** GETTER: Target throughput in megabytes for elasticity */
    public int getElasticthroughput() {
        return m_elasticthroughput;
    }

    /** GETTER: The maximum latency for a query batch before timing out */
    public int getQuerytimeout() {
        return m_querytimeout;
    }

    /** SETTER: The maximum allocation size for temp tables in the EE */
    public void setTemptablemaxsize(int value) {
        m_temptablemaxsize = value;
    }

    /** SETTER: The priority of snapshot work */
    public void setSnapshotpriority(int value) {
        m_snapshotpriority = value;
    }

    /** SETTER: Maximum duration time for rebalancing */
    public void setElasticduration(int value) {
        m_elasticduration = value;
    }

    /** SETTER: Target throughput in megabytes for elasticity */
    public void setElasticthroughput(int value) {
        m_elasticthroughput = value;
    }

    /** SETTER: The maximum latency for a query batch before timing out */
    public void setQuerytimeout(int value) {
        m_querytimeout = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "temptablemaxsize":
            assert(value != null);
            m_temptablemaxsize = Integer.parseInt(value);
            break;
        case "snapshotpriority":
            assert(value != null);
            m_snapshotpriority = Integer.parseInt(value);
            break;
        case "elasticduration":
            assert(value != null);
            m_elasticduration = Integer.parseInt(value);
            break;
        case "elasticthroughput":
            assert(value != null);
            m_elasticthroughput = Integer.parseInt(value);
            break;
        case "querytimeout":
            assert(value != null);
            m_querytimeout = Integer.parseInt(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Systemsettings other = (Systemsettings) obj;

        other.m_temptablemaxsize = m_temptablemaxsize;
        other.m_snapshotpriority = m_snapshotpriority;
        other.m_elasticduration = m_elasticduration;
        other.m_elasticthroughput = m_elasticthroughput;
        other.m_querytimeout = m_querytimeout;
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
        Systemsettings other = (Systemsettings) obj;

        // are the fields / children the same? (deep compare)
        if (m_temptablemaxsize != other.m_temptablemaxsize) return false;
        if (m_snapshotpriority != other.m_snapshotpriority) return false;
        if (m_elasticduration != other.m_elasticduration) return false;
        if (m_elasticthroughput != other.m_elasticthroughput) return false;
        if (m_querytimeout != other.m_querytimeout) return false;

        return true;
    }

}
