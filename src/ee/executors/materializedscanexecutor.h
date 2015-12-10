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


#ifndef MATERIALIZEDSCANEXECUTOR_H
#define MATERIALIZEDSCANEXECUTOR_H

#include "common/common.h"
#include "common/valuevector.h"
#include "executors/abstractexecutor.h"

namespace voltdb
{

    /**
     * Used for SQL-IN that are accelerated with indexes.
     * A MaterializedScanExecutor fills a temp table with values
     * from the SQL-IN-LIST expression. It is inner-joined with NLIJ
     * to another table to make the SQL-IN fast.
     */
    class MaterializedScanExecutor : public AbstractExecutor {
    public:
        MaterializedScanExecutor(VoltDBEngine *engine, AbstractPlanNode* abstract_node)
        : AbstractExecutor(engine, abstract_node)
        {}
        ~MaterializedScanExecutor();
    protected:
        bool p_init(AbstractPlanNode* abstract_node,
                    TempTableLimits* limits);
        bool p_execute(const NValueArray& params);
    };
}

#endif // MATERIALIZEDSCANEXECUTOR_H