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
#include "statement.h"
#include "catalog.h"
#include "planfragment.h"

using namespace catalog;
using namespace std;

Statement::Statement(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_fragments(catalog, this, path + "/" + "fragments")
{
    CatalogValue value;
    m_fields["sqltext"] = value;
    m_fields["querytype"] = value;
    m_fields["readonly"] = value;
    m_fields["singlepartition"] = value;
    m_fields["replicatedtabledml"] = value;
    m_fields["iscontentdeterministic"] = value;
    m_fields["isorderdeterministic"] = value;
    m_fields["nondeterminismdetail"] = value;
    m_childCollections["fragments"] = &m_fragments;
    m_fields["cost"] = value;
    m_fields["seqscancount"] = value;
    m_fields["explainplan"] = value;
    m_fields["tablesread"] = value;
    m_fields["tablesupdated"] = value;
    m_fields["indexesused"] = value;
    m_fields["cachekeyprefix"] = value;
}

Statement::~Statement() {
    std::map<std::string, PlanFragment*>::const_iterator planfragment_iter = m_fragments.begin();
    while (planfragment_iter != m_fragments.end()) {
        delete planfragment_iter->second;
        planfragment_iter++;
    }
    m_fragments.clear();

}

void Statement::update() {
    m_sqltext = m_fields["sqltext"].strValue.c_str();
    m_querytype = m_fields["querytype"].intValue;
    m_readonly = m_fields["readonly"].intValue;
    m_singlepartition = m_fields["singlepartition"].intValue;
    m_replicatedtabledml = m_fields["replicatedtabledml"].intValue;
    m_iscontentdeterministic = m_fields["iscontentdeterministic"].intValue;
    m_isorderdeterministic = m_fields["isorderdeterministic"].intValue;
    m_nondeterminismdetail = m_fields["nondeterminismdetail"].strValue.c_str();
    m_cost = m_fields["cost"].intValue;
    m_seqscancount = m_fields["seqscancount"].intValue;
    m_explainplan = m_fields["explainplan"].strValue.c_str();
    m_tablesread = m_fields["tablesread"].strValue.c_str();
    m_tablesupdated = m_fields["tablesupdated"].strValue.c_str();
    m_indexesused = m_fields["indexesused"].strValue.c_str();
    m_cachekeyprefix = m_fields["cachekeyprefix"].strValue.c_str();
}

CatalogType * Statement::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("fragments") == 0) {
        CatalogType *exists = m_fragments.get(childName);
        if (exists)
            return NULL;
        return m_fragments.add(childName);
    }
    return NULL;
}

CatalogType * Statement::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("fragments") == 0)
        return m_fragments.get(childName);
    return NULL;
}

bool Statement::removeChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("fragments") == 0) {
        return m_fragments.remove(childName);
    }
    return false;
}

const string & Statement::sqltext() const {
    return m_sqltext;
}

int32_t Statement::querytype() const {
    return m_querytype;
}

bool Statement::readonly() const {
    return m_readonly;
}

bool Statement::singlepartition() const {
    return m_singlepartition;
}

bool Statement::replicatedtabledml() const {
    return m_replicatedtabledml;
}

bool Statement::iscontentdeterministic() const {
    return m_iscontentdeterministic;
}

bool Statement::isorderdeterministic() const {
    return m_isorderdeterministic;
}

const string & Statement::nondeterminismdetail() const {
    return m_nondeterminismdetail;
}

const CatalogMap<PlanFragment> & Statement::fragments() const {
    return m_fragments;
}

int32_t Statement::cost() const {
    return m_cost;
}

int32_t Statement::seqscancount() const {
    return m_seqscancount;
}

const string & Statement::explainplan() const {
    return m_explainplan;
}

const string & Statement::tablesread() const {
    return m_tablesread;
}

const string & Statement::tablesupdated() const {
    return m_tablesupdated;
}

const string & Statement::indexesused() const {
    return m_indexesused;
}

const string & Statement::cachekeyprefix() const {
    return m_cachekeyprefix;
}

