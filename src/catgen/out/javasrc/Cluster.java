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
public class Cluster extends CatalogType {

    CatalogMap<Database> m_databases;
    CatalogMap<Deployment> m_deployment;
    int m_localepoch;
    boolean m_securityEnabled;
    int m_httpdportno;
    boolean m_jsonapi;
    boolean m_networkpartition;
    String m_voltRoot = new String();
    String m_exportOverflow = new String();
    CatalogMap<SnapshotSchedule> m_faultSnapshots;
    int m_adminport;
    boolean m_adminstartup;
    CatalogMap<CommandLog> m_logconfig;
    int m_heartbeatTimeout;
    boolean m_useddlschema;

    @Override
    void initChildMaps() {
        m_databases = new CatalogMap<Database>(getCatalog(), this, "databases", Database.class, m_parentMap.m_depth + 1);
        m_deployment = new CatalogMap<Deployment>(getCatalog(), this, "deployment", Deployment.class, m_parentMap.m_depth + 1);
        m_faultSnapshots = new CatalogMap<SnapshotSchedule>(getCatalog(), this, "faultSnapshots", SnapshotSchedule.class, m_parentMap.m_depth + 1);
        m_logconfig = new CatalogMap<CommandLog>(getCatalog(), this, "logconfig", CommandLog.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "localepoch",
            "securityEnabled",
            "httpdportno",
            "jsonapi",
            "networkpartition",
            "voltRoot",
            "exportOverflow",
            "adminport",
            "adminstartup",
            "heartbeatTimeout",
            "useddlschema",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "databases",
            "deployment",
            "faultSnapshots",
            "logconfig",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "databases":
            return getDatabases();
        case "deployment":
            return getDeployment();
        case "localepoch":
            return getLocalepoch();
        case "securityEnabled":
            return getSecurityenabled();
        case "httpdportno":
            return getHttpdportno();
        case "jsonapi":
            return getJsonapi();
        case "networkpartition":
            return getNetworkpartition();
        case "voltRoot":
            return getVoltroot();
        case "exportOverflow":
            return getExportoverflow();
        case "faultSnapshots":
            return getFaultsnapshots();
        case "adminport":
            return getAdminport();
        case "adminstartup":
            return getAdminstartup();
        case "logconfig":
            return getLogconfig();
        case "heartbeatTimeout":
            return getHeartbeattimeout();
        case "useddlschema":
            return getUseddlschema();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The set of databases the cluster is running */
    public CatalogMap<Database> getDatabases() {
        return m_databases;
    }

    /** GETTER: Storage for settings passed in on deployment */
    public CatalogMap<Deployment> getDeployment() {
        return m_deployment;
    }

    /** GETTER: The number of seconds since the epoch that we're calling our local epoch */
    public int getLocalepoch() {
        return m_localepoch;
    }

    /** GETTER: Whether security and authentication should be enabled/disabled */
    public boolean getSecurityenabled() {
        return m_securityEnabled;
    }

    /** GETTER: The port number httpd will listen on. A 0 value implies 8080. */
    public int getHttpdportno() {
        return m_httpdportno;
    }

    /** GETTER: Is the http/json interface enabled? */
    public boolean getJsonapi() {
        return m_jsonapi;
    }

    /** GETTER: Is network partition detection enabled? */
    public boolean getNetworkpartition() {
        return m_networkpartition;
    }

    /** GETTER: Directory tree where snapshots, ppd snapshots, export data etc. will be output to */
    public String getVoltroot() {
        return m_voltRoot;
    }

    /** GETTER: Directory where export data should overflow to */
    public String getExportoverflow() {
        return m_exportOverflow;
    }

    /** GETTER: Configuration for snapshots generated in response to faults. */
    public CatalogMap<SnapshotSchedule> getFaultsnapshots() {
        return m_faultSnapshots;
    }

    /** GETTER: The port number of the admin port */
    public int getAdminport() {
        return m_adminport;
    }

    /** GETTER: Does the server start in admin mode? */
    public boolean getAdminstartup() {
        return m_adminstartup;
    }

    /** GETTER: Command log configuration */
    public CatalogMap<CommandLog> getLogconfig() {
        return m_logconfig;
    }

    /** GETTER: How long to wait, in seconds, between messages before deciding a host is dead */
    public int getHeartbeattimeout() {
        return m_heartbeatTimeout;
    }

    /** GETTER: Manage the database schemas via catalog updates or live DDL */
    public boolean getUseddlschema() {
        return m_useddlschema;
    }

    /** SETTER: The number of seconds since the epoch that we're calling our local epoch */
    public void setLocalepoch(int value) {
        m_localepoch = value;
    }

    /** SETTER: Whether security and authentication should be enabled/disabled */
    public void setSecurityenabled(boolean value) {
        m_securityEnabled = value;
    }

    /** SETTER: The port number httpd will listen on. A 0 value implies 8080. */
    public void setHttpdportno(int value) {
        m_httpdportno = value;
    }

    /** SETTER: Is the http/json interface enabled? */
    public void setJsonapi(boolean value) {
        m_jsonapi = value;
    }

    /** SETTER: Is network partition detection enabled? */
    public void setNetworkpartition(boolean value) {
        m_networkpartition = value;
    }

    /** SETTER: Directory tree where snapshots, ppd snapshots, export data etc. will be output to */
    public void setVoltroot(String value) {
        m_voltRoot = value;
    }

    /** SETTER: Directory where export data should overflow to */
    public void setExportoverflow(String value) {
        m_exportOverflow = value;
    }

    /** SETTER: The port number of the admin port */
    public void setAdminport(int value) {
        m_adminport = value;
    }

    /** SETTER: Does the server start in admin mode? */
    public void setAdminstartup(boolean value) {
        m_adminstartup = value;
    }

    /** SETTER: How long to wait, in seconds, between messages before deciding a host is dead */
    public void setHeartbeattimeout(int value) {
        m_heartbeatTimeout = value;
    }

    /** SETTER: Manage the database schemas via catalog updates or live DDL */
    public void setUseddlschema(boolean value) {
        m_useddlschema = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "localepoch":
            assert(value != null);
            m_localepoch = Integer.parseInt(value);
            break;
        case "securityEnabled":
            assert(value != null);
            m_securityEnabled = Boolean.parseBoolean(value);
            break;
        case "httpdportno":
            assert(value != null);
            m_httpdportno = Integer.parseInt(value);
            break;
        case "jsonapi":
            assert(value != null);
            m_jsonapi = Boolean.parseBoolean(value);
            break;
        case "networkpartition":
            assert(value != null);
            m_networkpartition = Boolean.parseBoolean(value);
            break;
        case "voltRoot":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_voltRoot = value;
            break;
        case "exportOverflow":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_exportOverflow = value;
            break;
        case "adminport":
            assert(value != null);
            m_adminport = Integer.parseInt(value);
            break;
        case "adminstartup":
            assert(value != null);
            m_adminstartup = Boolean.parseBoolean(value);
            break;
        case "heartbeatTimeout":
            assert(value != null);
            m_heartbeatTimeout = Integer.parseInt(value);
            break;
        case "useddlschema":
            assert(value != null);
            m_useddlschema = Boolean.parseBoolean(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Cluster other = (Cluster) obj;

        other.m_databases.copyFrom(m_databases);
        other.m_deployment.copyFrom(m_deployment);
        other.m_localepoch = m_localepoch;
        other.m_securityEnabled = m_securityEnabled;
        other.m_httpdportno = m_httpdportno;
        other.m_jsonapi = m_jsonapi;
        other.m_networkpartition = m_networkpartition;
        other.m_voltRoot = m_voltRoot;
        other.m_exportOverflow = m_exportOverflow;
        other.m_faultSnapshots.copyFrom(m_faultSnapshots);
        other.m_adminport = m_adminport;
        other.m_adminstartup = m_adminstartup;
        other.m_logconfig.copyFrom(m_logconfig);
        other.m_heartbeatTimeout = m_heartbeatTimeout;
        other.m_useddlschema = m_useddlschema;
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
        Cluster other = (Cluster) obj;

        // are the fields / children the same? (deep compare)
        if ((m_databases == null) != (other.m_databases == null)) return false;
        if ((m_databases != null) && !m_databases.equals(other.m_databases)) return false;
        if ((m_deployment == null) != (other.m_deployment == null)) return false;
        if ((m_deployment != null) && !m_deployment.equals(other.m_deployment)) return false;
        if (m_localepoch != other.m_localepoch) return false;
        if (m_securityEnabled != other.m_securityEnabled) return false;
        if (m_httpdportno != other.m_httpdportno) return false;
        if (m_jsonapi != other.m_jsonapi) return false;
        if (m_networkpartition != other.m_networkpartition) return false;
        if ((m_voltRoot == null) != (other.m_voltRoot == null)) return false;
        if ((m_voltRoot != null) && !m_voltRoot.equals(other.m_voltRoot)) return false;
        if ((m_exportOverflow == null) != (other.m_exportOverflow == null)) return false;
        if ((m_exportOverflow != null) && !m_exportOverflow.equals(other.m_exportOverflow)) return false;
        if ((m_faultSnapshots == null) != (other.m_faultSnapshots == null)) return false;
        if ((m_faultSnapshots != null) && !m_faultSnapshots.equals(other.m_faultSnapshots)) return false;
        if (m_adminport != other.m_adminport) return false;
        if (m_adminstartup != other.m_adminstartup) return false;
        if ((m_logconfig == null) != (other.m_logconfig == null)) return false;
        if ((m_logconfig != null) && !m_logconfig.equals(other.m_logconfig)) return false;
        if (m_heartbeatTimeout != other.m_heartbeatTimeout) return false;
        if (m_useddlschema != other.m_useddlschema) return false;

        return true;
    }

}
