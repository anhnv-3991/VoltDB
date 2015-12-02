/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
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

#ifndef GVALUE_DEFS_H_
#define GVALUE_DEFS_H_

// We use these values for NULL to make it compact and fast
// "-1" is a hack to shut up gcc warning

// null DECIMAL is private. Use VoltDecimal.isNull()

#define GINT8_NULL (-127 - 1)
#define GINT16_NULL (-32767 - 1)
#define GINT32_NULL (-2147483647L - 1)
#define GINT64_NULL (LONG_MIN)

//Minimum value user can represent that is not null
#define GVOLT_INT8_MIN GINT8_NULL + 1
#define GVOLT_INT16_MIN GINT16_NULL + 1
#define GVOLT_INT32_MIN GINT32_NULL + 1
#define GVOLT_INT64_MIN GINT64_NULL + 1
#define GVOLT_DECIMAL_MIN -9999999
#define GVOLT_DECIMAL_MAX 9999999

// float/double less than these values are null
#define GFLOAT_NULL -3.4e+38f
#define GDOUBLE_NULL -1.7E+308

// values to be substituted as null
#define GFLOAT_MIN -3.40282347e+38f
#define GDOUBLE_MIN -1.7976931348623157E+308

#endif
