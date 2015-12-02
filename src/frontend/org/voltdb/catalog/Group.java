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

public class Group extends CatalogType {

    CatalogMap<UserRef> m_users;
    boolean m_admin;
    boolean m_defaultproc;
    boolean m_defaultprocread;
    boolean m_sql;
    boolean m_sqlread;
    boolean m_allproc;

    @Override
    void initChildMaps() {
        m_users = new CatalogMap<UserRef>(getCatalog(), this, "users", UserRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "admin",
            "defaultproc",
            "defaultprocread",
            "sql",
            "sqlread",
            "allproc",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "users",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "users":
            return getUsers();
        case "admin":
            return getAdmin();
        case "defaultproc":
            return getDefaultproc();
        case "defaultprocread":
            return getDefaultprocread();
        case "sql":
            return getSql();
        case "sqlread":
            return getSqlread();
        case "allproc":
            return getAllproc();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    public CatalogMap<UserRef> getUsers() {
        return m_users;
    }

    /** GETTER: Can perform all database operations */
    public boolean getAdmin() {
        return m_admin;
    }

    /** GETTER: Can invoke default procedures */
    public boolean getDefaultproc() {
        return m_defaultproc;
    }

    /** GETTER: Can invoke read-only default procedures */
    public boolean getDefaultprocread() {
        return m_defaultprocread;
    }

    /** GETTER: Can invoke the adhoc system procedures */
    public boolean getSql() {
        return m_sql;
    }

    /** GETTER: Can invoke read-only adhoc system procedures */
    public boolean getSqlread() {
        return m_sqlread;
    }

    /** GETTER: Can invoke any user defined procedures */
    public boolean getAllproc() {
        return m_allproc;
    }

    /** SETTER: Can perform all database operations */
    public void setAdmin(boolean value) {
        m_admin = value;
    }

    /** SETTER: Can invoke default procedures */
    public void setDefaultproc(boolean value) {
        m_defaultproc = value;
    }

    /** SETTER: Can invoke read-only default procedures */
    public void setDefaultprocread(boolean value) {
        m_defaultprocread = value;
    }

    /** SETTER: Can invoke the adhoc system procedures */
    public void setSql(boolean value) {
        m_sql = value;
    }

    /** SETTER: Can invoke read-only adhoc system procedures */
    public void setSqlread(boolean value) {
        m_sqlread = value;
    }

    /** SETTER: Can invoke any user defined procedures */
    public void setAllproc(boolean value) {
        m_allproc = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "admin":
            assert(value != null);
            m_admin = Boolean.parseBoolean(value);
            break;
        case "defaultproc":
            assert(value != null);
            m_defaultproc = Boolean.parseBoolean(value);
            break;
        case "defaultprocread":
            assert(value != null);
            m_defaultprocread = Boolean.parseBoolean(value);
            break;
        case "sql":
            assert(value != null);
            m_sql = Boolean.parseBoolean(value);
            break;
        case "sqlread":
            assert(value != null);
            m_sqlread = Boolean.parseBoolean(value);
            break;
        case "allproc":
            assert(value != null);
            m_allproc = Boolean.parseBoolean(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Group other = (Group) obj;

        other.m_users.copyFrom(m_users);
        other.m_admin = m_admin;
        other.m_defaultproc = m_defaultproc;
        other.m_defaultprocread = m_defaultprocread;
        other.m_sql = m_sql;
        other.m_sqlread = m_sqlread;
        other.m_allproc = m_allproc;
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
        Group other = (Group) obj;

        // are the fields / children the same? (deep compare)
        if ((m_users == null) != (other.m_users == null)) return false;
        if ((m_users != null) && !m_users.equals(other.m_users)) return false;
        if (m_admin != other.m_admin) return false;
        if (m_defaultproc != other.m_defaultproc) return false;
        if (m_defaultprocread != other.m_defaultprocread) return false;
        if (m_sql != other.m_sql) return false;
        if (m_sqlread != other.m_sqlread) return false;
        if (m_allproc != other.m_allproc) return false;

        return true;
    }

}
