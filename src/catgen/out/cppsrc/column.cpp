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
#include "column.h"
#include "catalog.h"
#include "constraintref.h"
#include "column.h"
#include "materializedviewinfo.h"

using namespace catalog;
using namespace std;

Column::Column(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_constraints(catalog, this, path + "/" + "constraints")
{
    CatalogValue value;
    m_fields["index"] = value;
    m_fields["type"] = value;
    m_fields["size"] = value;
    m_fields["nullable"] = value;
    m_fields["name"] = value;
    m_fields["defaultvalue"] = value;
    m_fields["defaulttype"] = value;
    m_childCollections["constraints"] = &m_constraints;
    m_fields["matview"] = value;
    m_fields["aggregatetype"] = value;
    m_fields["matviewsource"] = value;
    m_fields["inbytes"] = value;
}

Column::~Column() {
    std::map<std::string, ConstraintRef*>::const_iterator constraintref_iter = m_constraints.begin();
    while (constraintref_iter != m_constraints.end()) {
        delete constraintref_iter->second;
        constraintref_iter++;
    }
    m_constraints.clear();

}

void Column::update() {
    m_index = m_fields["index"].intValue;
    m_type = m_fields["type"].intValue;
    m_size = m_fields["size"].intValue;
    m_nullable = m_fields["nullable"].intValue;
    m_name = m_fields["name"].strValue.c_str();
    m_defaultvalue = m_fields["defaultvalue"].strValue.c_str();
    m_defaulttype = m_fields["defaulttype"].intValue;
    m_matview = m_fields["matview"].typeValue;
    m_aggregatetype = m_fields["aggregatetype"].intValue;
    m_matviewsource = m_fields["matviewsource"].typeValue;
    m_inbytes = m_fields["inbytes"].intValue;
}

CatalogType * Column::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("constraints") == 0) {
        CatalogType *exists = m_constraints.get(childName);
        if (exists)
            return NULL;
        return m_constraints.add(childName);
    }
    return NULL;
}

CatalogType * Column::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("constraints") == 0)
        return m_constraints.get(childName);
    return NULL;
}

bool Column::removeChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("constraints") == 0) {
        return m_constraints.remove(childName);
    }
    return false;
}

int32_t Column::index() const {
    return m_index;
}

int32_t Column::type() const {
    return m_type;
}

int32_t Column::size() const {
    return m_size;
}

bool Column::nullable() const {
    return m_nullable;
}

const string & Column::name() const {
    return m_name;
}

const string & Column::defaultvalue() const {
    return m_defaultvalue;
}

int32_t Column::defaulttype() const {
    return m_defaulttype;
}

const CatalogMap<ConstraintRef> & Column::constraints() const {
    return m_constraints;
}

const MaterializedViewInfo * Column::matview() const {
    return dynamic_cast<MaterializedViewInfo*>(m_matview);
}

int32_t Column::aggregatetype() const {
    return m_aggregatetype;
}

const Column * Column::matviewsource() const {
    return dynamic_cast<Column*>(m_matviewsource);
}

bool Column::inbytes() const {
    return m_inbytes;
}

