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

#ifndef HELPERS_H
#define HELPERS_H
#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>

/*===================================================================
 *
 * Computes the number of elements in a dimension of a distributed
 * array owned by the MPI-process indicated by proc_dim_rank.
 * From Fortran source: http://www.cs.umu.se/~dacke/ngssc/numroc.f
*/
npy_intp dnumroc(npy_intp nelem_in_dim, npy_intp block_size,
                 int proc_dim_rank, int nproc_in_dim,
                 int first_process);

/*===================================================================
 *
 * Process cartesian coords <-> MPI rank.
 */
int cart2rank(int ndims, const int coords[NPY_MAXDIMS]);
void rank2cart(int ndims, int rank, int coords[NPY_MAXDIMS]);

/*===================================================================
 *
 * Sends a message to all slaves.
 * msgsize is in bytes.
 */
#ifndef DNPY_SPMD
void msg2slaves(npy_intp *msg, int msgsize);
#endif

/*===================================================================
 *
 * Returns a string describing the operation type.
 */
char *optype2str(int optype);

/*===================================================================
 *  Returns a MPI data type that match the specified sub-view-block.
 */
static MPI_Datatype
calc_svb_MPIdatatype(const dndview *view, dndsvb *svb);

/*===================================================================
 *
 * Calculate the view block at the specified block-coordinate.
 * NB: vcoord is the visible coordinates and must therefore have
 * length view->ndims.
 */
void calc_vblock(const dndview *view, const npy_intp vcoord[NPY_MAXDIMS],
                 dndvb *vblock);

/*===================================================================
 *
 * Convert visible vblock dimension index to base vblock
 * dimension index.
 */
npy_intp idx_v2b(const dndview *view, npy_intp vindex);

/*===================================================================
 *
 * Convert visible vblock dimension index to slice dimension index.
 */
npy_intp idx_v2s(const dndview *view, npy_intp vindex);



#ifdef __cplusplus
}
#endif

#endif /* !defined(HELPERS_H) */
