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

#ifndef CATALOG_PLANFRAGMENT_H_
#define CATALOG_PLANFRAGMENT_H_

#include <string>
#include "catalogtype.h"
#include "catalogmap.h"

namespace catalog {

/**
 * Instructions
 */
class PlanFragment : public CatalogType {
    friend class Catalog;
    friend class CatalogMap<PlanFragment>;

protected:
    PlanFragment(Catalog * catalog, CatalogType * parent, const std::string &path, const std::string &name);
    bool m_hasdependencies;
    bool m_multipartition;
    std::string m_plannodetree;
    bool m_nontransactional;
    std::string m_planhash;

    virtual void update();

    virtual CatalogType * addChild(const std::string &collectionName, const std::string &name);
    virtual CatalogType * getChild(const std::string &collectionName, const std::string &childName) const;
    virtual bool removeChild(const std::string &collectionName, const std::string &childName);

public:
    ~PlanFragment();

    /** GETTER: Dependencies must be received before this plan fragment can execute */
    bool hasdependencies() const;
    /** GETTER: Should this plan fragment be sent to all partitions */
    bool multipartition() const;
    /** GETTER: A serialized representation of the plan-graph/plan-pipeline */
    const std::string & plannodetree() const;
    /** GETTER: True if this fragment doesn't read from or write to any persistent tables */
    bool nontransactional() const;
    /** GETTER: SHA-1 Hash of the plan assumed to be unique */
    const std::string & planhash() const;
};

} // namespace catalog

#endif //  CATALOG_PLANFRAGMENT_H_
