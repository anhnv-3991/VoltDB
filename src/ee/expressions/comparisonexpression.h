/* This file is part of VoltDB.
 * Copyright (C) 2008-2015 VoltDB Inc.
 *
 * This file contains original code and/or modifications of original code.
 * Any modifications made by VoltDB Inc. are licensed under the following
 * terms and conditions:
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
/* Copyright (C) 2008 by H-Store Project
 * Brown University
 * Massachusetts Institute of Technology
 * Yale University
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef HSTORECOMPARISONEXPRESSION_H
#define HSTORECOMPARISONEXPRESSION_H

#include "common/common.h"
#include "common/serializeio.h"
#include "common/valuevector.h"
#include "common/types.h"

#include "expressions/abstractexpression.h"
#include "expressions/parametervalueexpression.h"
#include "expressions/constantvalueexpression.h"
#include "expressions/tuplevalueexpression.h"
#include "expressions/operatorexpression.h"

#include <string>
#include <cassert>
#include <iostream>
#include <stdio.h>

namespace voltdb {

class CmpEq {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_equals_withoutNull(r);}
};
class CmpNe {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_notEquals_withoutNull(r);}
};
class CmpLt {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_lessThan_withoutNull(r);}
};
class CmpGt {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_greaterThan_withoutNull(r);}
};
class CmpLte {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_lessThanOrEqual_withoutNull(r);}
};
class CmpGte {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.op_greaterThanOrEqual_withoutNull(r);}
};
class CmpLike {
public:
    inline NValue cmp(NValue l, NValue r) const { return l.like(r);}
};
class CmpIn {
public:
    inline NValue cmp(NValue l, NValue r) const
    { return l.inList(r) ? NValue::getTrue() : NValue::getFalse(); }
};

template <typename C>
class ComparisonExpression : public AbstractExpression {
public:
    ComparisonExpression(ExpressionType type,
                                  AbstractExpression *left,
                                  AbstractExpression *right)
        : AbstractExpression(type, left, right)
    {
        m_left = left;
        m_right = right;
    };

    inline NValue eval(const TableTuple *tuple1, const TableTuple *tuple2) const {
        VOLT_TRACE("eval %s. left %s, right %s. ret=%s",
                   typeid(compare).name(), typeid(*(m_left)).name(),
                   typeid(*(m_right)).name(),
                   compare.cmp(m_left->eval(tuple1, tuple2),
                               m_right->eval(tuple1, tuple2)).isTrue()
                   ? "TRUE" : "FALSE");

        assert(m_left != NULL);
        assert(m_right != NULL);

        NValue lnv = m_left->eval(tuple1, tuple2);
        if (lnv.isNull()) {
            return NValue::getNullValue(VALUE_TYPE_BOOLEAN);
        }

        NValue rnv = m_right->eval(tuple1, tuple2);
        if (rnv.isNull()) {
            return NValue::getNullValue(VALUE_TYPE_BOOLEAN);
        }

        // comparisons with null or NaN are always false
        // [This code is commented out because doing the right thing breaks voltdb atm.
        // We need to re-enable after we can verify that all plans in all configs give the
        // same answer.]
        if (lnv.isNull() || lnv.isNaN() || rnv.isNull() || rnv.isNaN()) {
            return NValue::getFalse();
        }

        return compare.cmp(lnv, rnv);
    }

    std::string debugInfo(const std::string &spacer) const {
        return (spacer + "ComparisonExpression\n");
    }

    /**
       add for GPU join
     */
    inline NValue getLeftNV(const TableTuple *tuple1, const TableTuple *tuple2) const{
//        return static_cast<TupleValueExpression *>(m_left)->eval(tuple1,tuple2);
        return m_left->eval(tuple1,tuple2);
    }
    inline NValue getRightNV(const TableTuple *tuple1, const TableTuple *tuple2) const{
//        return static_cast<TupleValueExpression *>(m_right)->eval(tuple1,tuple2);
        return m_right->eval(tuple1,tuple2);
    }

    inline int getLeftTupleId(){
        TupleValueExpression* tve = dynamic_cast<TupleValueExpression*>(m_left);
        if(tve){
            return tve->getTupleId();
        }

        switch(m_left->getExpressionType()) {
        case (EXPRESSION_TYPE_OPERATOR_PLUS):
            return dynamic_cast<OperatorExpression<OpPlus> *>(m_left)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_MINUS):
            return dynamic_cast<OperatorExpression<OpMinus> *>(m_left)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_MULTIPLY):
            return dynamic_cast<OperatorExpression<OpMultiply> *>(m_left)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_DIVIDE):
            return dynamic_cast<OperatorExpression<OpDivide> *>(m_left)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_NOT):
//            OperatorNotExpression *oe = dynamic_cast<OperatorNotExpression *>(m_left);
//            break;
        case (EXPRESSION_TYPE_OPERATOR_IS_NULL):
//            OperatorIsNullExpression *oe = dynamic_cast<OperatorIsNullExpression *>(m_left);
//            break;
        default:
            return -1;
        }
    }

    inline int getRightTupleId(){
        TupleValueExpression* tve = dynamic_cast<TupleValueExpression*>(m_right);
        if(tve){
            return dynamic_cast<TupleValueExpression*>(m_right)->getTupleId();
        }

        int temp;
        switch(m_right->getExpressionType()) {
        case (EXPRESSION_TYPE_OPERATOR_PLUS):
            temp = dynamic_cast<OperatorExpression<OpPlus> *>(m_right)->getTupleIdInOperator();
            return temp;
        case (EXPRESSION_TYPE_OPERATOR_MINUS):
            return dynamic_cast<OperatorExpression<OpMinus> *>(m_right)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_MULTIPLY):
            return dynamic_cast<OperatorExpression<OpMultiply> *>(m_right)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_DIVIDE):
            return dynamic_cast<OperatorExpression<OpDivide> *>(m_right)->getTupleIdInOperator();
        case (EXPRESSION_TYPE_OPERATOR_NOT):
//            OperatorNotExpression *oe = dynamic_cast<OperatorNotExpression *>(m_right);
//            break;
        case (EXPRESSION_TYPE_OPERATOR_IS_NULL):
//            OperatorIsNullExpression *oe = dynamic_cast<OperatorIsNullExpression *>(m_right);
//            break;
        default:
            return -1;
        }

    }


private:
    AbstractExpression *m_left;
    AbstractExpression *m_right;
    C compare;
};

template <typename C, typename L, typename R>
class InlinedComparisonExpression : public ComparisonExpression<C> {
public:
    InlinedComparisonExpression(ExpressionType type,
                                         AbstractExpression *left,
                                         AbstractExpression *right)
        : ComparisonExpression<C>(type, left, right)
    {}
};

}
#endif
