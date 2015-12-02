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
#include "connector.h"
#include "catalog.h"
#include "connectorproperty.h"
#include "connectortableinfo.h"

using namespace catalog;
using namespace std;

Connector::Connector(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_tableInfo(catalog, this, path + "/" + "tableInfo"), m_config(catalog, this, path + "/" + "config")
{
    CatalogValue value;
    m_fields["loaderclass"] = value;
    m_fields["enabled"] = value;
    m_childCollections["tableInfo"] = &m_tableInfo;
    m_childCollections["config"] = &m_config;
}

Connector::~Connector() {
    std::map<std::string, ConnectorTableInfo*>::const_iterator connectortableinfo_iter = m_tableInfo.begin();
    while (connectortableinfo_iter != m_tableInfo.end()) {
        delete connectortableinfo_iter->second;
        connectortableinfo_iter++;
    }
    m_tableInfo.clear();

    std::map<std::string, ConnectorProperty*>::const_iterator connectorproperty_iter = m_config.begin();
    while (connectorproperty_iter != m_config.end()) {
        delete connectorproperty_iter->second;
        connectorproperty_iter++;
    }
    m_config.clear();

}

void Connector::update() {
    m_loaderclass = m_fields["loaderclass"].strValue.c_str();
    m_enabled = m_fields["enabled"].intValue;
}

CatalogType * Connector::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("tableInfo") == 0) {
        CatalogType *exists = m_tableInfo.get(childName);
        if (exists)
            return NULL;
        return m_tableInfo.add(childName);
    }
    if (collectionName.compare("config") == 0) {
        CatalogType *exists = m_config.get(childName);
        if (exists)
            return NULL;
        return m_config.add(childName);
    }
    return NULL;
}

CatalogType * Connector::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("tableInfo") == 0)
        return m_tableInfo.get(childName);
    if (collectionName.compare("config") == 0)
        return m_config.get(childName);
    return NULL;
}

bool Connector::removeChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("tableInfo") == 0) {
        return m_tableInfo.remove(childName);
    }
    if (collectionName.compare("config") == 0) {
        return m_config.remove(childName);
    }
    return false;
}

const string & Connector::loaderclass() const {
    return m_loaderclass;
}

bool Connector::enabled() const {
    return m_enabled;
}

const CatalogMap<ConnectorTableInfo> & Connector::tableInfo() const {
    return m_tableInfo;
}

const CatalogMap<ConnectorProperty> & Connector::config() const {
    return m_config;
}

