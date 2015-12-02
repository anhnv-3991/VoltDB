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
#include "materializedviewinfo.h"
#include "catalog.h"
#include "table.h"
#include "columnref.h"

using namespace catalog;
using namespace std;

MaterializedViewInfo::MaterializedViewInfo(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_groupbycols(catalog, this, path + "/" + "groupbycols")
{
    CatalogValue value;
    m_fields["dest"] = value;
    m_childCollections["groupbycols"] = &m_groupbycols;
    m_fields["predicate"] = value;
    m_fields["groupbyExpressionsJson"] = value;
    m_fields["aggregationExpressionsJson"] = value;
    m_fields["indexForMinMax"] = value;
}

MaterializedViewInfo::~MaterializedViewInfo() {
    std::map<std::string, ColumnRef*>::const_iterator columnref_iter = m_groupbycols.begin();
    while (columnref_iter != m_groupbycols.end()) {
        delete columnref_iter->second;
        columnref_iter++;
    }
    m_groupbycols.clear();

}

void MaterializedViewInfo::update() {
    m_dest = m_fields["dest"].typeValue;
    m_predicate = m_fields["predicate"].strValue.c_str();
    m_groupbyExpressionsJson = m_fields["groupbyExpressionsJson"].strValue.c_str();
    m_aggregationExpressionsJson = m_fields["aggregationExpressionsJson"].strValue.c_str();
    m_indexForMinMax = m_fields["indexForMinMax"].strValue.c_str();
}

CatalogType * MaterializedViewInfo::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("groupbycols") == 0) {
        CatalogType *exists = m_groupbycols.get(childName);
        if (exists)
            return NULL;
        return m_groupbycols.add(childName);
    }
    return NULL;
}

CatalogType * MaterializedViewInfo::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("groupbycols") == 0)
        return m_groupbycols.get(childName);
    return NULL;
}

bool MaterializedViewInfo::removeChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("groupbycols") == 0) {
        return m_groupbycols.remove(childName);
    }
    return false;
}

const Table * MaterializedViewInfo::dest() const {
    return dynamic_cast<Table*>(m_dest);
}

const CatalogMap<ColumnRef> & MaterializedViewInfo::groupbycols() const {
    return m_groupbycols;
}

const string & MaterializedViewInfo::predicate() const {
    return m_predicate;
}

const string & MaterializedViewInfo::groupbyExpressionsJson() const {
    return m_groupbyExpressionsJson;
}

const string & MaterializedViewInfo::aggregationExpressionsJson() const {
    return m_aggregationExpressionsJson;
}

const string & MaterializedViewInfo::indexForMinMax() const {
    return m_indexForMinMax;
}

