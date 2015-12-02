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
 * A schedule for the database to follow when creating automated snapshots
 */
public class SnapshotSchedule extends CatalogType {

    boolean m_enabled;
    String m_frequencyUnit = new String();
    int m_frequencyValue;
    int m_retain;
    String m_path = new String();
    String m_prefix = new String();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "enabled",
            "frequencyUnit",
            "frequencyValue",
            "retain",
            "path",
            "prefix",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "enabled":
            return getEnabled();
        case "frequencyUnit":
            return getFrequencyunit();
        case "frequencyValue":
            return getFrequencyvalue();
        case "retain":
            return getRetain();
        case "path":
            return getPath();
        case "prefix":
            return getPrefix();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Is this auto snapshot schedule enabled? */
    public boolean getEnabled() {
        return m_enabled;
    }

    /** GETTER: Unit of time frequency is specified in */
    public String getFrequencyunit() {
        return m_frequencyUnit;
    }

    /** GETTER: Frequency in some unit */
    public int getFrequencyvalue() {
        return m_frequencyValue;
    }

    /** GETTER: How many snapshots to retain */
    public int getRetain() {
        return m_retain;
    }

    /** GETTER: Path where snapshots should be stored */
    public String getPath() {
        return m_path;
    }

    /** GETTER: Prefix for snapshot filenames */
    public String getPrefix() {
        return m_prefix;
    }

    /** SETTER: Is this auto snapshot schedule enabled? */
    public void setEnabled(boolean value) {
        m_enabled = value;
    }

    /** SETTER: Unit of time frequency is specified in */
    public void setFrequencyunit(String value) {
        m_frequencyUnit = value;
    }

    /** SETTER: Frequency in some unit */
    public void setFrequencyvalue(int value) {
        m_frequencyValue = value;
    }

    /** SETTER: How many snapshots to retain */
    public void setRetain(int value) {
        m_retain = value;
    }

    /** SETTER: Path where snapshots should be stored */
    public void setPath(String value) {
        m_path = value;
    }

    /** SETTER: Prefix for snapshot filenames */
    public void setPrefix(String value) {
        m_prefix = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "enabled":
            assert(value != null);
            m_enabled = Boolean.parseBoolean(value);
            break;
        case "frequencyUnit":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_frequencyUnit = value;
            break;
        case "frequencyValue":
            assert(value != null);
            m_frequencyValue = Integer.parseInt(value);
            break;
        case "retain":
            assert(value != null);
            m_retain = Integer.parseInt(value);
            break;
        case "path":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_path = value;
            break;
        case "prefix":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_prefix = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        SnapshotSchedule other = (SnapshotSchedule) obj;

        other.m_enabled = m_enabled;
        other.m_frequencyUnit = m_frequencyUnit;
        other.m_frequencyValue = m_frequencyValue;
        other.m_retain = m_retain;
        other.m_path = m_path;
        other.m_prefix = m_prefix;
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
        SnapshotSchedule other = (SnapshotSchedule) obj;

        // are the fields / children the same? (deep compare)
        if (m_enabled != other.m_enabled) return false;
        if ((m_frequencyUnit == null) != (other.m_frequencyUnit == null)) return false;
        if ((m_frequencyUnit != null) && !m_frequencyUnit.equals(other.m_frequencyUnit)) return false;
        if (m_frequencyValue != other.m_frequencyValue) return false;
        if (m_retain != other.m_retain) return false;
        if ((m_path == null) != (other.m_path == null)) return false;
        if ((m_path != null) && !m_path.equals(other.m_path)) return false;
        if ((m_prefix == null) != (other.m_prefix == null)) return false;
        if ((m_prefix != null) && !m_prefix.equals(other.m_prefix)) return false;

        return true;
    }

}
