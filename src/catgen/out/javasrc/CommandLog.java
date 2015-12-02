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
 * Configuration for a command log
 */
public class CommandLog extends CatalogType {

    boolean m_enabled;
    boolean m_synchronous;
    int m_fsyncInterval;
    int m_maxTxns;
    int m_logSize;
    String m_logPath = new String();
    String m_internalSnapshotPath = new String();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "enabled",
            "synchronous",
            "fsyncInterval",
            "maxTxns",
            "logSize",
            "logPath",
            "internalSnapshotPath",
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
        case "synchronous":
            return getSynchronous();
        case "fsyncInterval":
            return getFsyncinterval();
        case "maxTxns":
            return getMaxtxns();
        case "logSize":
            return getLogsize();
        case "logPath":
            return getLogpath();
        case "internalSnapshotPath":
            return getInternalsnapshotpath();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Is command logging enabled */
    public boolean getEnabled() {
        return m_enabled;
    }

    /** GETTER: Should commands be executed only once durable */
    public boolean getSynchronous() {
        return m_synchronous;
    }

    /** GETTER: How often commands should be written to disk */
    public int getFsyncinterval() {
        return m_fsyncInterval;
    }

    /** GETTER: How many txns waiting to go to disk should trigger a flush */
    public int getMaxtxns() {
        return m_maxTxns;
    }

    /** GETTER: Size of the command log in megabytes */
    public int getLogsize() {
        return m_logSize;
    }

    /** GETTER: Directory to store log files */
    public String getLogpath() {
        return m_logPath;
    }

    /** GETTER: Directory to store internal snapshots for the command log */
    public String getInternalsnapshotpath() {
        return m_internalSnapshotPath;
    }

    /** SETTER: Is command logging enabled */
    public void setEnabled(boolean value) {
        m_enabled = value;
    }

    /** SETTER: Should commands be executed only once durable */
    public void setSynchronous(boolean value) {
        m_synchronous = value;
    }

    /** SETTER: How often commands should be written to disk */
    public void setFsyncinterval(int value) {
        m_fsyncInterval = value;
    }

    /** SETTER: How many txns waiting to go to disk should trigger a flush */
    public void setMaxtxns(int value) {
        m_maxTxns = value;
    }

    /** SETTER: Size of the command log in megabytes */
    public void setLogsize(int value) {
        m_logSize = value;
    }

    /** SETTER: Directory to store log files */
    public void setLogpath(String value) {
        m_logPath = value;
    }

    /** SETTER: Directory to store internal snapshots for the command log */
    public void setInternalsnapshotpath(String value) {
        m_internalSnapshotPath = value;
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
        case "synchronous":
            assert(value != null);
            m_synchronous = Boolean.parseBoolean(value);
            break;
        case "fsyncInterval":
            assert(value != null);
            m_fsyncInterval = Integer.parseInt(value);
            break;
        case "maxTxns":
            assert(value != null);
            m_maxTxns = Integer.parseInt(value);
            break;
        case "logSize":
            assert(value != null);
            m_logSize = Integer.parseInt(value);
            break;
        case "logPath":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_logPath = value;
            break;
        case "internalSnapshotPath":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_internalSnapshotPath = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        CommandLog other = (CommandLog) obj;

        other.m_enabled = m_enabled;
        other.m_synchronous = m_synchronous;
        other.m_fsyncInterval = m_fsyncInterval;
        other.m_maxTxns = m_maxTxns;
        other.m_logSize = m_logSize;
        other.m_logPath = m_logPath;
        other.m_internalSnapshotPath = m_internalSnapshotPath;
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
        CommandLog other = (CommandLog) obj;

        // are the fields / children the same? (deep compare)
        if (m_enabled != other.m_enabled) return false;
        if (m_synchronous != other.m_synchronous) return false;
        if (m_fsyncInterval != other.m_fsyncInterval) return false;
        if (m_maxTxns != other.m_maxTxns) return false;
        if (m_logSize != other.m_logSize) return false;
        if ((m_logPath == null) != (other.m_logPath == null)) return false;
        if ((m_logPath != null) && !m_logPath.equals(other.m_logPath)) return false;
        if ((m_internalSnapshotPath == null) != (other.m_internalSnapshotPath == null)) return false;
        if ((m_internalSnapshotPath != null) && !m_internalSnapshotPath.equals(other.m_internalSnapshotPath)) return false;

        return true;
    }

}
