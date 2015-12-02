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
 * Export
 */
public class Connector extends CatalogType {

    String m_loaderclass = new String();
    boolean m_enabled;
    CatalogMap<ConnectorTableInfo> m_tableInfo;
    CatalogMap<ConnectorProperty> m_config;

    @Override
    void initChildMaps() {
        m_tableInfo = new CatalogMap<ConnectorTableInfo>(getCatalog(), this, "tableInfo", ConnectorTableInfo.class, m_parentMap.m_depth + 1);
        m_config = new CatalogMap<ConnectorProperty>(getCatalog(), this, "config", ConnectorProperty.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "loaderclass",
            "enabled",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "tableInfo",
            "config",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "loaderclass":
            return getLoaderclass();
        case "enabled":
            return getEnabled();
        case "tableInfo":
            return getTableinfo();
        case "config":
            return getConfig();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The class name of the connector */
    public String getLoaderclass() {
        return m_loaderclass;
    }

    /** GETTER: Is the connector enabled */
    public boolean getEnabled() {
        return m_enabled;
    }

    /** GETTER: Per table configuration */
    public CatalogMap<ConnectorTableInfo> getTableinfo() {
        return m_tableInfo;
    }

    /** GETTER: Connector configuration properties */
    public CatalogMap<ConnectorProperty> getConfig() {
        return m_config;
    }

    /** SETTER: The class name of the connector */
    public void setLoaderclass(String value) {
        m_loaderclass = value;
    }

    /** SETTER: Is the connector enabled */
    public void setEnabled(boolean value) {
        m_enabled = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "loaderclass":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_loaderclass = value;
            break;
        case "enabled":
            assert(value != null);
            m_enabled = Boolean.parseBoolean(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Connector other = (Connector) obj;

        other.m_loaderclass = m_loaderclass;
        other.m_enabled = m_enabled;
        other.m_tableInfo.copyFrom(m_tableInfo);
        other.m_config.copyFrom(m_config);
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
        Connector other = (Connector) obj;

        // are the fields / children the same? (deep compare)
        if ((m_loaderclass == null) != (other.m_loaderclass == null)) return false;
        if ((m_loaderclass != null) && !m_loaderclass.equals(other.m_loaderclass)) return false;
        if (m_enabled != other.m_enabled) return false;
        if ((m_tableInfo == null) != (other.m_tableInfo == null)) return false;
        if ((m_tableInfo != null) && !m_tableInfo.equals(other.m_tableInfo)) return false;
        if ((m_config == null) != (other.m_config == null)) return false;
        if ((m_config != null) && !m_config.equals(other.m_config)) return false;

        return true;
    }

}
