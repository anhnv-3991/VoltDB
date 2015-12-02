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
 * Run-time deployment settings
 */
public class Deployment extends CatalogType {

    int m_hostcount;
    int m_kfactor;
    int m_sitesperhost;
    CatalogMap<Systemsettings> m_systemsettings;

    @Override
    void initChildMaps() {
        m_systemsettings = new CatalogMap<Systemsettings>(getCatalog(), this, "systemsettings", Systemsettings.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "hostcount",
            "kfactor",
            "sitesperhost",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "systemsettings",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "hostcount":
            return getHostcount();
        case "kfactor":
            return getKfactor();
        case "sitesperhost":
            return getSitesperhost();
        case "systemsettings":
            return getSystemsettings();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The number of hosts in the cluster */
    public int getHostcount() {
        return m_hostcount;
    }

    /** GETTER: The required k-safety factor */
    public int getKfactor() {
        return m_kfactor;
    }

    /** GETTER: The number of execution sites per host */
    public int getSitesperhost() {
        return m_sitesperhost;
    }

    /** GETTER: Values from the systemsettings element */
    public CatalogMap<Systemsettings> getSystemsettings() {
        return m_systemsettings;
    }

    /** SETTER: The number of hosts in the cluster */
    public void setHostcount(int value) {
        m_hostcount = value;
    }

    /** SETTER: The required k-safety factor */
    public void setKfactor(int value) {
        m_kfactor = value;
    }

    /** SETTER: The number of execution sites per host */
    public void setSitesperhost(int value) {
        m_sitesperhost = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "hostcount":
            assert(value != null);
            m_hostcount = Integer.parseInt(value);
            break;
        case "kfactor":
            assert(value != null);
            m_kfactor = Integer.parseInt(value);
            break;
        case "sitesperhost":
            assert(value != null);
            m_sitesperhost = Integer.parseInt(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Deployment other = (Deployment) obj;

        other.m_hostcount = m_hostcount;
        other.m_kfactor = m_kfactor;
        other.m_sitesperhost = m_sitesperhost;
        other.m_systemsettings.copyFrom(m_systemsettings);
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
        Deployment other = (Deployment) obj;

        // are the fields / children the same? (deep compare)
        if (m_hostcount != other.m_hostcount) return false;
        if (m_kfactor != other.m_kfactor) return false;
        if (m_sitesperhost != other.m_sitesperhost) return false;
        if ((m_systemsettings == null) != (other.m_systemsettings == null)) return false;
        if ((m_systemsettings != null) && !m_systemsettings.equals(other.m_systemsettings)) return false;

        return true;
    }

}
