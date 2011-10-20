/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of DistNumPy <https://github.com/distnumpy>.
 *
 * DistNumPy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DistNumPy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DistNumPy. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DISTNUMPY_H
#define DISTNUMPY_H
#ifdef __cplusplus
extern "C" {
#endif

//Only import when compiling distnumpymodule.c
#ifdef DISTNUMPY_MODULE
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#endif

//Flag indicating that it is a distributed array
#define DNPY_DIST 0x2000
//Flag indicating that it is a distributed array on one node
#define DNPY_DIST_ONENODE 0x4000

//Easy attribute retrievals.
#define PyDistArray_WANT_DIST(m) PyArray_CHKFLAGS(m,DNPY_DIST)
#define PyDistArray_WANT_ONENODE(m) PyArray_CHKFLAGS(m,DNPY_DIST_ONENODE)
#define PyDistArray_ARRAY(obj) (((PyArrayObject *)(obj))->distary)

//Import the API.
#include "distnumpy_api.h"

#ifdef __cplusplus
}
#endif

#endif /* !defined(DISTNUMPY_H) */
