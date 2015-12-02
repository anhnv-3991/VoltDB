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

#include <cassert>
#include "database.h"
#include "catalog.h"
#include "table.h"
#include "connector.h"

using namespace catalog;
using namespace std;

Database::Database(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_tables(catalog, this, path + "/" + "tables"), m_connectors(catalog, this, path + "/" + "connectors")
{
    CatalogValue value;
    m_fields["schema"] = value;
    m_childCollections["tables"] = &m_tables;
    m_childCollections["connectors"] = &m_connectors;
    m_fields["securityprovider"] = value;
}

Database::~Database() {
    std::map<std::string, Table*>::const_iterator table_iter = m_tables.begin();
    while (table_iter != m_tables.end()) {
        delete table_iter->second;
        table_iter++;
    }
    m_tables.clear();

    std::map<std::string, Connector*>::const_iterator connector_iter = m_connectors.begin();
    while (connector_iter != m_connectors.end()) {
        delete connector_iter->second;
        connector_iter++;
    }
    m_connectors.clear();

}

void Database::update() {
    m_schema = m_fields["schema"].strValue.c_str();
    m_securityprovider = m_fields["securityprovider"].strValue.c_str();
}

CatalogType * Database::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("tables") == 0) {
        CatalogType *exists = m_tables.get(childName);
        if (exists)
            return NULL;
        return m_tables.add(childName);
    }
    if (collectionName.compare("connectors") == 0) {
        CatalogType *exists = m_connectors.get(childName);
        if (exists)
            return NULL;
        return m_connectors.add(childName);
    }
    return NULL;
}

CatalogType * Database::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("tables") == 0)
        return m_tables.get(childName);
    if (collectionName.compare("connectors") == 0)
        return m_connectors.get(childName);
    return NULL;
}

bool Database::removeChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("tables") == 0) {
        return m_tables.remove(childName);
    }
    if (collectionName.compare("connectors") == 0) {
        return m_connectors.remove(childName);
    }
    return false;
}

const string & Database::schema() const {
    return m_schema;
}

const CatalogMap<Table> & Database::tables() const {
    return m_tables;
}

const CatalogMap<Connector> & Database::connectors() const {
    return m_connectors;
}

const string & Database::securityprovider() const {
    return m_securityprovider;
}

