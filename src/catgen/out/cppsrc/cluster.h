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

#ifndef CATALOG_CLUSTER_H_
#define CATALOG_CLUSTER_H_

#include <string>
#include "catalogtype.h"
#include "catalogmap.h"

namespace catalog {

class Database;
/**
 * A
 */
class Cluster : public CatalogType {
    friend class Catalog;
    friend class CatalogMap<Cluster>;

protected:
    Cluster(Catalog * catalog, CatalogType * parent, const std::string &path, const std::string &name);
    CatalogMap<Database> m_databases;
    int32_t m_localepoch;
    bool m_securityEnabled;
    int32_t m_httpdportno;
    bool m_jsonapi;
    bool m_networkpartition;
    std::string m_voltRoot;
    std::string m_exportOverflow;
    int32_t m_adminport;
    bool m_adminstartup;
    int32_t m_heartbeatTimeout;
    bool m_useddlschema;

    virtual void update();

    virtual CatalogType * addChild(const std::string &collectionName, const std::string &name);
    virtual CatalogType * getChild(const std::string &collectionName, const std::string &childName) const;
    virtual bool removeChild(const std::string &collectionName, const std::string &childName);

public:
    ~Cluster();

    /** GETTER: The set of databases the cluster is running */
    const CatalogMap<Database> & databases() const;
    /** GETTER: The number of seconds since the epoch that we're calling our local epoch */
    int32_t localepoch() const;
    /** GETTER: Whether security and authentication should be enabled/disabled */
    bool securityEnabled() const;
    /** GETTER: The port number httpd will listen on. A 0 value implies 8080. */
    int32_t httpdportno() const;
    /** GETTER: Is the http/json interface enabled? */
    bool jsonapi() const;
    /** GETTER: Is network partition detection enabled? */
    bool networkpartition() const;
    /** GETTER: Directory tree where snapshots, ppd snapshots, export data etc. will be output to */
    const std::string & voltRoot() const;
    /** GETTER: Directory where export data should overflow to */
    const std::string & exportOverflow() const;
    /** GETTER: The port number of the admin port */
    int32_t adminport() const;
    /** GETTER: Does the server start in admin mode? */
    bool adminstartup() const;
    /** GETTER: How long to wait, in seconds, between messages before deciding a host is dead */
    int32_t heartbeatTimeout() const;
    /** GETTER: Manage the database schemas via catalog updates or live DDL */
    bool useddlschema() const;
};

} // namespace catalog

#endif //  CATALOG_CLUSTER_H_
