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

#ifndef DEPENDENCY_SYSTEM_H
#define DEPENDENCY_SYSTEM_H
#ifdef __cplusplus
extern "C" {
#endif

/*===================================================================
 *
 * Add the list of node to the dependency system.
 * Node that all nodes in the list must relate to the same operation.
 */
void dep_add(dndnode *nodes, int nnodes, int force_laziness);

/*===================================================================
 *
 * Removes a operation from the dependency system.
 * op2apply is the operations that should be applyed.
 * If op2apply is NULL then op2apply is ignored.
 * Returns number of operations in op2apply.
 */
npy_intp dep_remove(dndop *op, dndop *op2apply[]);

/*===================================================================
 *
 * Flush the dependency system.
 * Frees the work buffer when 'free_workbuf' is true.
 */
void dep_flush(int free_workbuf);



#ifdef __cplusplus
}
#endif

#endif /* !defined(DEPENDENCY_SYSTEM_H) */
