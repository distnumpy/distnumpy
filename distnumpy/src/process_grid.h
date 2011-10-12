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

#ifndef PROCESS_GRID_H
#define PROCESS_GRID_H
#ifdef __cplusplus
extern "C" {
#endif

/*
 * ===================================================================
 * Setting the process grid.
 * Accepts NULL as a default value request.
 * When called no distributed array must be allocated.
 */
static PyObject *
PyDistArray_ProcGridSet(PyArrayObject *self, PyObject *args);

/*===================================================================
 *
 * Handler for PyDistArray_ProcGridSet.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_ProcGridSet(int pgrid[NPY_MAXDIMS*NPY_MAXDIMS]);

#ifdef __cplusplus
}
#endif

#endif /* !defined(PROCESS_GRID_H) */
