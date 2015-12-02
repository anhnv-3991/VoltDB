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

#ifndef CATALOG_MATERIALIZEDVIEWINFO_H_
#define CATALOG_MATERIALIZEDVIEWINFO_H_

#include <string>
#include "catalogtype.h"
#include "catalogmap.h"

namespace catalog {

class Table;
class ColumnRef;
/**
 * Information
 */
class MaterializedViewInfo : public CatalogType {
    friend class Catalog;
    friend class CatalogMap<MaterializedViewInfo>;

protected:
    MaterializedViewInfo(Catalog * catalog, CatalogType * parent, const std::string &path, const std::string &name);
    CatalogType* m_dest;
    CatalogMap<ColumnRef> m_groupbycols;
    std::string m_predicate;
    std::string m_groupbyExpressionsJson;
    std::string m_aggregationExpressionsJson;
    std::string m_indexForMinMax;

    virtual void update();

    virtual CatalogType * addChild(const std::string &collectionName, const std::string &name);
    virtual CatalogType * getChild(const std::string &collectionName, const std::string &childName) const;
    virtual bool removeChild(const std::string &collectionName, const std::string &childName);

public:
    ~MaterializedViewInfo();

    /** GETTER: The table which will be updated when the source table is updated */
    const Table * dest() const;
    /** GETTER: The columns involved in the group by of the aggregation */
    const CatalogMap<ColumnRef> & groupbycols() const;
    /** GETTER: A filtering predicate */
    const std::string & predicate() const;
    /** GETTER: A serialized representation of the groupby expression trees */
    const std::string & groupbyExpressionsJson() const;
    /** GETTER: A serialized representation of the aggregation expression trees */
    const std::string & aggregationExpressionsJson() const;
    /** GETTER: The name of index on srcTable which can be used to maintain min()/max() */
    const std::string & indexForMinMax() const;
};

} // namespace catalog

#endif //  CATALOG_MATERIALIZEDVIEWINFO_H_
