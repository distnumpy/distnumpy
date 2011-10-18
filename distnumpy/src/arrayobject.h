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

#ifndef ARRAYOBJECT_H
#define ARRAYOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#include "distnumpy_types.h"

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * If 'one_node_dist_rank' is positive it specifies the rank of an
 * one-node-distribution.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewBaseArray(PyArrayObject *ary, npy_intp one_node_dist_rank);

/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return NULL and set exception on error.
 * Return a pointer to the new dndview on success.
 */
dndview *handle_NewBaseArray(dndarray *ary, dndview *view);

/*
 *===================================================================
 * Delete array view.
 * When it is the last view of the base array, the base array is de-
 * allocated.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_DelViewArray(PyArrayObject *array);

/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_DelViewArray(npy_intp uid);

/*
 *===================================================================
 * Assign the value to array at coordinate.
 * 'coord' size must be the same as view->ndims.
 * Steals all reference to item. (Item is lost).
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_PutItem(PyArrayObject *ary, npy_intp coord[NPY_MAXDIMS],
                    PyObject *item);

/*
 *===================================================================
 * Get a single value specified by coordinate from the array.
 * 'coord' size must be the same as view->ndims.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_GetItem(PyArrayObject *ary, char *retdata,
                    npy_intp coord[NPY_MAXDIMS]);

/*===================================================================
 *
 * Handler for PyDistArray_PutItem and PyDistArray_GetItem.
 * Direction: 0=Get, 1=Put.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_PutGetItem(int Direction, dndview *view, char* item,
                      npy_intp coord[NPY_MAXDIMS]);

/*===================================================================
 *
 * Un-distributes the array by transferring all data to the master
 * MPI-process.
 * Return -1 and set exception on error, 0 on success.
 */
int PyDistArray_UnDist(dndarray *ary);

/*===================================================================
 *
 * Handler for PyDistArray_UnDist.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_UnDist(npy_intp ary_uid);


#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYOBJECT_H) */
