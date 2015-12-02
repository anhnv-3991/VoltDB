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

public class User extends CatalogType {

    CatalogMap<GroupRef> m_groups;
    String m_shadowPassword = new String();

    @Override
    void initChildMaps() {
        m_groups = new CatalogMap<GroupRef>(getCatalog(), this, "groups", GroupRef.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "shadowPassword",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "groups",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "groups":
            return getGroups();
        case "shadowPassword":
            return getShadowpassword();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    public CatalogMap<GroupRef> getGroups() {
        return m_groups;
    }

    /** GETTER: SHA-1 double hashed hex encoded version of the password */
    public String getShadowpassword() {
        return m_shadowPassword;
    }

    /** SETTER: SHA-1 double hashed hex encoded version of the password */
    public void setShadowpassword(String value) {
        m_shadowPassword = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "shadowPassword":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_shadowPassword = value;
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        User other = (User) obj;

        other.m_groups.copyFrom(m_groups);
        other.m_shadowPassword = m_shadowPassword;
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
        User other = (User) obj;

        // are the fields / children the same? (deep compare)
        if ((m_groups == null) != (other.m_groups == null)) return false;
        if ((m_groups != null) && !m_groups.equals(other.m_groups)) return false;
        if ((m_shadowPassword == null) != (other.m_shadowPassword == null)) return false;
        if ((m_shadowPassword != null) && !m_shadowPassword.equals(other.m_shadowPassword)) return false;

        return true;
    }

}
