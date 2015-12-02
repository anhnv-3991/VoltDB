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

#ifndef CATALOG_DATABASE_H_
#define CATALOG_DATABASE_H_

#include <string>
#include "catalogtype.h"
#include "catalogmap.h"

namespace catalog {

class Table;
class Connector;
/**
 * A
 */
class Database : public CatalogType {
    friend class Catalog;
    friend class CatalogMap<Database>;

protected:
    Database(Catalog * catalog, CatalogType * parent, const std::string &path, const std::string &name);
    std::string m_schema;
    CatalogMap<Table> m_tables;
    CatalogMap<Connector> m_connectors;
    std::string m_securityprovider;

    virtual void update();

    virtual CatalogType * addChild(const std::string &collectionName, const std::string &name);
    virtual CatalogType * getChild(const std::string &collectionName, const std::string &childName) const;
    virtual bool removeChild(const std::string &collectionName, const std::string &childName);

public:
    ~Database();

    /** GETTER: Full SQL DDL for the database's schema */
    const std::string & schema() const;
    /** GETTER: The set of Tables for the database */
    const CatalogMap<Table> & tables() const;
    /** GETTER: Export connector configuration */
    const CatalogMap<Connector> & connectors() const;
    /** GETTER: The security provider used to authenticate users */
    const std::string & securityprovider() const;
};

} // namespace catalog

#endif //  CATALOG_DATABASE_H_
