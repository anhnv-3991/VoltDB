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
 * A stored procedure (transaction) in the system
 */
public class Procedure extends CatalogType {

    String m_classname = new String();
    CatalogMap<UserRef> m_authUsers;
    CatalogMap<GroupRef> m_authGroups;
    boolean m_readonly;
    boolean m_singlepartition;
    boolean m_everysite;
    boolean m_systemproc;
    boolean m_defaultproc;
    boolean m_hasjava;
    boolean m_hasseqscans;
    String m_language = new String();
    Catalog.CatalogReference<Table> m_partitiontable = new CatalogReference<>();
    Catalog.CatalogReference<Column> m_partitioncolumn = new CatalogReference<>();
    int m_partitionparameter;
    CatalogMap<AuthProgram> m_authPrograms;
    CatalogMap<Statement> m_statements;
    CatalogMap<ProcParameter> m_parameters;

    @Override
    void initChildMaps() {
        m_authUsers = new CatalogMap<UserRef>(getCatalog(), this, "authUsers", UserRef.class, m_parentMap.m_depth + 1);
        m_authGroups = new CatalogMap<GroupRef>(getCatalog(), this, "authGroups", GroupRef.class, m_parentMap.m_depth + 1);
        m_authPrograms = new CatalogMap<AuthProgram>(getCatalog(), this, "authPrograms", AuthProgram.class, m_parentMap.m_depth + 1);
        m_statements = new CatalogMap<Statement>(getCatalog(), this, "statements", Statement.class, m_parentMap.m_depth + 1);
        m_parameters = new CatalogMap<ProcParameter>(getCatalog(), this, "parameters", ProcParameter.class, m_parentMap.m_depth + 1);
    }

    public String[] getFields() {
        return new String[] {
            "classname",
            "readonly",
            "singlepartition",
            "everysite",
            "systemproc",
            "defaultproc",
            "hasjava",
            "hasseqscans",
            "language",
            "partitiontable",
            "partitioncolumn",
            "partitionparameter",
        };
    };

    String[] getChildCollections() {
        return new String[] {
            "authUsers",
            "authGroups",
            "authPrograms",
            "statements",
            "parameters",
        };
    };

    public Object getField(String field) {
        switch (field) {
        case "classname":
            return getClassname();
        case "authUsers":
            return getAuthusers();
        case "authGroups":
            return getAuthgroups();
        case "readonly":
            return getReadonly();
        case "singlepartition":
            return getSinglepartition();
        case "everysite":
            return getEverysite();
        case "systemproc":
            return getSystemproc();
        case "defaultproc":
            return getDefaultproc();
        case "hasjava":
            return getHasjava();
        case "hasseqscans":
            return getHasseqscans();
        case "language":
            return getLanguage();
        case "partitiontable":
            return getPartitiontable();
        case "partitioncolumn":
            return getPartitioncolumn();
        case "partitionparameter":
            return getPartitionparameter();
        case "authPrograms":
            return getAuthprograms();
        case "statements":
            return getStatements();
        case "parameters":
            return getParameters();
        default:
            throw new CatalogException("Unknown field");
        }
    }

    /** GETTER: The full class name for the Java class for this procedure */
    public String getClassname() {
        return m_classname;
    }

    /** GETTER: Users authorized to invoke this procedure */
    public CatalogMap<UserRef> getAuthusers() {
        return m_authUsers;
    }

    /** GETTER: Groups authorized to invoke this procedure */
    public CatalogMap<GroupRef> getAuthgroups() {
        return m_authGroups;
    }

    /** GETTER: Can the stored procedure modify data */
    public boolean getReadonly() {
        return m_readonly;
    }

    /** GETTER: Does the stored procedure need data on more than one partition? */
    public boolean getSinglepartition() {
        return m_singlepartition;
    }

    /** GETTER: Does the stored procedure as a single procedure txn at every site? */
    public boolean getEverysite() {
        return m_everysite;
    }

    /** GETTER: Is this procedure an internal system procedure? */
    public boolean getSystemproc() {
        return m_systemproc;
    }

    /** GETTER: Is this procedure a default auto-generated CRUD procedure? */
    public boolean getDefaultproc() {
        return m_defaultproc;
    }

    /** GETTER: Is this a full java stored procedure or is it just a single stmt? */
    public boolean getHasjava() {
        return m_hasjava;
    }

    /** GETTER: Do any of the proc statements use sequential scans? */
    public boolean getHasseqscans() {
        return m_hasseqscans;
    }

    /** GETTER: What language is the procedure implemented with */
    public String getLanguage() {
        return m_language;
    }

    /** GETTER: Which table contains the partition column for this procedure? */
    public Table getPartitiontable() {
        return m_partitiontable.get();
    }

    /** GETTER: Which column in the partitioned table is this procedure mapped on? */
    public Column getPartitioncolumn() {
        return m_partitioncolumn.get();
    }

    /** GETTER: Which parameter identifies the partition column? */
    public int getPartitionparameter() {
        return m_partitionparameter;
    }

    /** GETTER: The set of authorized programs for this procedure (users) */
    public CatalogMap<AuthProgram> getAuthprograms() {
        return m_authPrograms;
    }

    /** GETTER: The set of SQL statements this procedure may call */
    public CatalogMap<Statement> getStatements() {
        return m_statements;
    }

    /** GETTER: The set of parameters to this stored procedure */
    public CatalogMap<ProcParameter> getParameters() {
        return m_parameters;
    }

    /** SETTER: The full class name for the Java class for this procedure */
    public void setClassname(String value) {
        m_classname = value;
    }

    /** SETTER: Can the stored procedure modify data */
    public void setReadonly(boolean value) {
        m_readonly = value;
    }

    /** SETTER: Does the stored procedure need data on more than one partition? */
    public void setSinglepartition(boolean value) {
        m_singlepartition = value;
    }

    /** SETTER: Does the stored procedure as a single procedure txn at every site? */
    public void setEverysite(boolean value) {
        m_everysite = value;
    }

    /** SETTER: Is this procedure an internal system procedure? */
    public void setSystemproc(boolean value) {
        m_systemproc = value;
    }

    /** SETTER: Is this procedure a default auto-generated CRUD procedure? */
    public void setDefaultproc(boolean value) {
        m_defaultproc = value;
    }

    /** SETTER: Is this a full java stored procedure or is it just a single stmt? */
    public void setHasjava(boolean value) {
        m_hasjava = value;
    }

    /** SETTER: Do any of the proc statements use sequential scans? */
    public void setHasseqscans(boolean value) {
        m_hasseqscans = value;
    }

    /** SETTER: What language is the procedure implemented with */
    public void setLanguage(String value) {
        m_language = value;
    }

    /** SETTER: Which table contains the partition column for this procedure? */
    public void setPartitiontable(Table value) {
        m_partitiontable.set(value);
    }

    /** SETTER: Which column in the partitioned table is this procedure mapped on? */
    public void setPartitioncolumn(Column value) {
        m_partitioncolumn.set(value);
    }

    /** SETTER: Which parameter identifies the partition column? */
    public void setPartitionparameter(int value) {
        m_partitionparameter = value;
    }

    @Override
    void set(String field, String value) {
        if ((field == null) || (value == null)) {
            throw new CatalogException("Null value where it shouldn't be.");
        }

        switch (field) {
        case "classname":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_classname = value;
            break;
        case "readonly":
            assert(value != null);
            m_readonly = Boolean.parseBoolean(value);
            break;
        case "singlepartition":
            assert(value != null);
            m_singlepartition = Boolean.parseBoolean(value);
            break;
        case "everysite":
            assert(value != null);
            m_everysite = Boolean.parseBoolean(value);
            break;
        case "systemproc":
            assert(value != null);
            m_systemproc = Boolean.parseBoolean(value);
            break;
        case "defaultproc":
            assert(value != null);
            m_defaultproc = Boolean.parseBoolean(value);
            break;
        case "hasjava":
            assert(value != null);
            m_hasjava = Boolean.parseBoolean(value);
            break;
        case "hasseqscans":
            assert(value != null);
            m_hasseqscans = Boolean.parseBoolean(value);
            break;
        case "language":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            if (value != null) {
                assert(value.startsWith("\"") && value.endsWith("\""));
                value = value.substring(1, value.length() - 1);
            }
            m_language = value;
            break;
        case "partitiontable":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_partitiontable.setUnresolved(value);
            break;
        case "partitioncolumn":
            value = value.trim();
            if (value.startsWith("null")) value = null;
            assert((value == null) || value.startsWith("/"));
            m_partitioncolumn.setUnresolved(value);
            break;
        case "partitionparameter":
            assert(value != null);
            m_partitionparameter = Integer.parseInt(value);
            break;
        default:
            throw new CatalogException("Unknown field");
        }
    }

    @Override
    void copyFields(CatalogType obj) {
        // this is safe from the caller
        Procedure other = (Procedure) obj;

        other.m_classname = m_classname;
        other.m_authUsers.copyFrom(m_authUsers);
        other.m_authGroups.copyFrom(m_authGroups);
        other.m_readonly = m_readonly;
        other.m_singlepartition = m_singlepartition;
        other.m_everysite = m_everysite;
        other.m_systemproc = m_systemproc;
        other.m_defaultproc = m_defaultproc;
        other.m_hasjava = m_hasjava;
        other.m_hasseqscans = m_hasseqscans;
        other.m_language = m_language;
        other.m_partitiontable.setUnresolved(m_partitiontable.getPath());
        other.m_partitioncolumn.setUnresolved(m_partitioncolumn.getPath());
        other.m_partitionparameter = m_partitionparameter;
        other.m_authPrograms.copyFrom(m_authPrograms);
        other.m_statements.copyFrom(m_statements);
        other.m_parameters.copyFrom(m_parameters);
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
        Procedure other = (Procedure) obj;

        // are the fields / children the same? (deep compare)
        if ((m_classname == null) != (other.m_classname == null)) return false;
        if ((m_classname != null) && !m_classname.equals(other.m_classname)) return false;
        if ((m_authUsers == null) != (other.m_authUsers == null)) return false;
        if ((m_authUsers != null) && !m_authUsers.equals(other.m_authUsers)) return false;
        if ((m_authGroups == null) != (other.m_authGroups == null)) return false;
        if ((m_authGroups != null) && !m_authGroups.equals(other.m_authGroups)) return false;
        if (m_readonly != other.m_readonly) return false;
        if (m_singlepartition != other.m_singlepartition) return false;
        if (m_everysite != other.m_everysite) return false;
        if (m_systemproc != other.m_systemproc) return false;
        if (m_defaultproc != other.m_defaultproc) return false;
        if (m_hasjava != other.m_hasjava) return false;
        if (m_hasseqscans != other.m_hasseqscans) return false;
        if ((m_language == null) != (other.m_language == null)) return false;
        if ((m_language != null) && !m_language.equals(other.m_language)) return false;
        if ((m_partitiontable == null) != (other.m_partitiontable == null)) return false;
        if ((m_partitiontable != null) && !m_partitiontable.equals(other.m_partitiontable)) return false;
        if ((m_partitioncolumn == null) != (other.m_partitioncolumn == null)) return false;
        if ((m_partitioncolumn != null) && !m_partitioncolumn.equals(other.m_partitioncolumn)) return false;
        if (m_partitionparameter != other.m_partitionparameter) return false;
        if ((m_authPrograms == null) != (other.m_authPrograms == null)) return false;
        if ((m_authPrograms != null) && !m_authPrograms.equals(other.m_authPrograms)) return false;
        if ((m_statements == null) != (other.m_statements == null)) return false;
        if ((m_statements != null) && !m_statements.equals(other.m_statements)) return false;
        if ((m_parameters == null) != (other.m_parameters == null)) return false;
        if ((m_parameters != null) && !m_parameters.equals(other.m_parameters)) return false;

        return true;
    }

}
