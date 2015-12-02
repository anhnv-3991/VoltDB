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

#ifndef CATALOG_TABLE_H_
#define CATALOG_TABLE_H_

#include <string>
#include "catalogtype.h"
#include "catalogmap.h"

namespace catalog {

class Column;
class Index;
class Constraint;
class MaterializedViewInfo;
class Statement;
/**
 * A
 */
class Table : public CatalogType {
    friend class Catalog;
    friend class CatalogMap<Table>;

protected:
    Table(Catalog * catalog, CatalogType * parent, const std::string &path, const std::string &name);
    CatalogMap<Column> m_columns;
    CatalogMap<Index> m_indexes;
    CatalogMap<Constraint> m_constraints;
    bool m_isreplicated;
    CatalogType* m_partitioncolumn;
    int32_t m_estimatedtuplecount;
    CatalogMap<MaterializedViewInfo> m_views;
    CatalogType* m_materializer;
    std::string m_signature;
    int32_t m_tuplelimit;
    CatalogMap<Statement> m_tuplelimitDeleteStmt;

    virtual void update();

    virtual CatalogType * addChild(const std::string &collectionName, const std::string &name);
    virtual CatalogType * getChild(const std::string &collectionName, const std::string &childName) const;
    virtual bool removeChild(const std::string &collectionName, const std::string &childName);

public:
    ~Table();

    /** GETTER: The set of columns in the table */
    const CatalogMap<Column> & columns() const;
    /** GETTER: The set of indexes on the columns in the table */
    const CatalogMap<Index> & indexes() const;
    /** GETTER: The set of constraints on the table */
    const CatalogMap<Constraint> & constraints() const;
    /** GETTER: Is the table replicated? */
    bool isreplicated() const;
    /** GETTER: On which column is the table partitioned */
    const Column * partitioncolumn() const;
    /** GETTER: A rough estimate of the number of tuples in the table; used for planning */
    int32_t estimatedtuplecount() const;
    /** GETTER: Information about materialized views based on this table's content */
    const CatalogMap<MaterializedViewInfo> & views() const;
    /** GETTER: If this is a materialized view, this field stores the source table */
    const Table * materializer() const;
    /** GETTER: Catalog version independent signature of the table consisting of name and schema */
    const std::string & signature() const;
    /** GETTER: A maximum number of rows in a table */
    int32_t tuplelimit() const;
    /** GETTER: Delete statement to execute if tuple limit will be exceeded */
    const CatalogMap<Statement> & tuplelimitDeleteStmt() const;
};

} // namespace catalog

#endif //  CATALOG_TABLE_H_
