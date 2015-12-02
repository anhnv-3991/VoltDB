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
 * Connector
 */
public class ConnectorProperty extends CatalogType {

    String m_name = new String();
    String m_value = new String();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "name",
            "value",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "name":
            return getName();
        case "value":
            return getValue();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Configuration property name */
    public String getName() {
        return m_name;
    }

    /** GETTER: Configuration property value */
    public String getValue() {
        return m_value;
    }

    /** SETTER: Configuration property name */
    public void setName(String value) {
        m_name = value;
    }

    /** SETTER: Configuration property value */
    public void setValue(String value) {
        m_value = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "name":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_name = value;
            break;
        case "value":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_value = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        ConnectorProperty other = (ConnectorProperty) obj;

        other.m_name = m_name;
        other.m_value = m_value;
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
        ConnectorProperty other = (ConnectorProperty) obj;

        // are the fields / children the same? (deep compare)
        if ((m_name == null) != (other.m_name == null)) return false;
        if ((m_name != null) && !m_name.equals(other.m_name)) return false;
        if ((m_value == null) != (other.m_value == null)) return false;
        if ((m_value != null) && !m_value.equals(other.m_value)) return false;

        return true;
    }

}
