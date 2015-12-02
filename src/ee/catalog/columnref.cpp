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
#include "columnref.h"
#include "catalog.h"
#include "column.h"

using namespace catalog;
using namespace std;

ColumnRef::ColumnRef(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name)
{
    CatalogValue value;
    m_fields["index"] = value;
    m_fields["column"] = value;
}

ColumnRef::~ColumnRef() {
}

void ColumnRef::update() {
    m_index = m_fields["index"].intValue;
    m_column = m_fields["column"].typeValue;
}

CatalogType * ColumnRef::addChild(const std::string &collectionName, const std::string &childName) {
    return NULL;
}

CatalogType * ColumnRef::getChild(const std::string &collectionName, const std::string &childName) const {
    return NULL;
}

bool ColumnRef::removeChild(const std::string &collectionName, const std::string &childName) {
    return false;
}

int32_t ColumnRef::index() const {
    return m_index;
}

const Column * ColumnRef::column() const {
    return dynamic_cast<Column*>(m_column);
}

