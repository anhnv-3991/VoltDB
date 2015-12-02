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
public class ConstraintRef extends CatalogType {

    Catalog.CatalogReference<Constraint> m_constraint = new CatalogReference<>();

    @Override
    void initChildMaps() {
    }

    public String[] getFields() {
        return new String[] {
            "constraint",
        };
    };

    String[] getChildCollections() {
        return new String[] {
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "constraint":
            return getConstraint();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The constraint that is referenced */
    public Constraint getConstraint() {
        return m_constraint.get();
    }

    /** SETTER: The constraint that is referenced */
    public void setConstraint(Constraint value) {
        m_constraint.set(value);
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "constraint":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_constraint.setUnresolved(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        ConstraintRef other = (ConstraintRef) obj;

        other.m_constraint.setUnresolved(m_constraint.getPath());
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
        ConstraintRef other = (ConstraintRef) obj;

        // are the fields / children the same? (deep compare)
        if ((m_constraint == null) != (other.m_constraint == null)) return false;
        if ((m_constraint != null) && !m_constraint.equals(other.m_constraint)) return false;

        return true;
    }

}
