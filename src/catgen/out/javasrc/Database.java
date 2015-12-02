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
public class Database extends CatalogType {

    String m_schema = new String();
    CatalogMap<User> m_users;
    CatalogMap<Group> m_groups;
    CatalogMap<Table> m_tables;
    CatalogMap<Program> m_programs;
    CatalogMap<Procedure> m_procedures;
    CatalogMap<Connector> m_connectors;
    CatalogMap<SnapshotSchedule> m_snapshotSchedule;
    String m_securityprovider = new String();

    @Override
    void initChildMaps() {
        m_users = new CatalogMap<User>(getCatalog(), this, "users", User.class, m_parentMap.m_depth + 1);
        m_groups = new CatalogMap<Group>(getCatalog(), this, "groups", Group.class, m_parentMap.m_depth + 1);
        m_tables = new CatalogMap<Table>(getCatalog(), this, "tables", Table.class, m_parentMap.m_depth + 1);
        m_programs = new CatalogMap<Program>(getCatalog(), this, "programs", Program.class, m_parentMap.m_depth + 1);
        m_procedures = new CatalogMap<Procedure>(getCatalog(), this, "procedures", Procedure.class, m_parentMap.m_depth + 1);
        m_connectors = new CatalogMap<Connector>(getCatalog(), this, "connectors", Connector.class, m_parentMap.m_depth + 1);
        m_snapshotSchedule = new CatalogMap<SnapshotSchedule>(getCatalog(), this, "snapshotSchedule", SnapshotSchedule.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "schema",
            "securityprovider",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "users",
            "groups",
            "tables",
            "programs",
            "procedures",
            "connectors",
            "snapshotSchedule",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "schema":
            return getSchema();
        case "users":
            return getUsers();
        case "groups":
            return getGroups();
        case "tables":
            return getTables();
        case "programs":
            return getPrograms();
        case "procedures":
            return getProcedures();
        case "connectors":
            return getConnectors();
        case "snapshotSchedule":
            return getSnapshotschedule();
        case "securityprovider":
            return getSecurityprovider();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: Full SQL DDL for the database's schema */
    public String getSchema() {
        return m_schema;
    }

    /** GETTER: The set of users */
    public CatalogMap<User> getUsers() {
        return m_users;
    }

    /** GETTER: The set of groups */
    public CatalogMap<Group> getGroups() {
        return m_groups;
    }

    /** GETTER: The set of Tables for the database */
    public CatalogMap<Table> getTables() {
        return m_tables;
    }

    /** GETTER: The set of programs allowed to access this database */
    public CatalogMap<Program> getPrograms() {
        return m_programs;
    }

    /** GETTER: The set of stored procedures/transactions for this database */
    public CatalogMap<Procedure> getProcedures() {
        return m_procedures;
    }

    /** GETTER: Export connector configuration */
    public CatalogMap<Connector> getConnectors() {
        return m_connectors;
    }

    /** GETTER: Schedule for automated snapshots */
    public CatalogMap<SnapshotSchedule> getSnapshotschedule() {
        return m_snapshotSchedule;
    }

    /** GETTER: The security provider used to authenticate users */
    public String getSecurityprovider() {
        return m_securityprovider;
    }

    /** SETTER: Full SQL DDL for the database's schema */
    public void setSchema(String value) {
        m_schema = value;
    }

    /** SETTER: The security provider used to authenticate users */
    public void setSecurityprovider(String value) {
        m_securityprovider = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "schema":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_schema = value;
            break;
        case "securityprovider":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_securityprovider = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Database other = (Database) obj;

        other.m_schema = m_schema;
        other.m_users.copyFrom(m_users);
        other.m_groups.copyFrom(m_groups);
        other.m_tables.copyFrom(m_tables);
        other.m_programs.copyFrom(m_programs);
        other.m_procedures.copyFrom(m_procedures);
        other.m_connectors.copyFrom(m_connectors);
        other.m_snapshotSchedule.copyFrom(m_snapshotSchedule);
        other.m_securityprovider = m_securityprovider;
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
        Database other = (Database) obj;

        // are the fields / children the same? (deep compare)
        if ((m_schema == null) != (other.m_schema == null)) return false;
        if ((m_schema != null) && !m_schema.equals(other.m_schema)) return false;
        if ((m_users == null) != (other.m_users == null)) return false;
        if ((m_users != null) && !m_users.equals(other.m_users)) return false;
        if ((m_groups == null) != (other.m_groups == null)) return false;
        if ((m_groups != null) && !m_groups.equals(other.m_groups)) return false;
        if ((m_tables == null) != (other.m_tables == null)) return false;
        if ((m_tables != null) && !m_tables.equals(other.m_tables)) return false;
        if ((m_programs == null) != (other.m_programs == null)) return false;
        if ((m_programs != null) && !m_programs.equals(other.m_programs)) return false;
        if ((m_procedures == null) != (other.m_procedures == null)) return false;
        if ((m_procedures != null) && !m_procedures.equals(other.m_procedures)) return false;
        if ((m_connectors == null) != (other.m_connectors == null)) return false;
        if ((m_connectors != null) && !m_connectors.equals(other.m_connectors)) return false;
        if ((m_snapshotSchedule == null) != (other.m_snapshotSchedule == null)) return false;
        if ((m_snapshotSchedule != null) && !m_snapshotSchedule.equals(other.m_snapshotSchedule)) return false;
        if ((m_securityprovider == null) != (other.m_securityprovider == null)) return false;
        if ((m_securityprovider != null) && !m_securityprovider.equals(other.m_securityprovider)) return false;

        return true;
    }

}
