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

/*===================================================================
 *
 * Computes the number of elements in a dimension of a distributed
 * array owned by the MPI-process indicated by proc_dim_rank.
 * From Fortran source: http://www.cs.umu.se/~dacke/ngssc/numroc.f
*/
npy_intp dnumroc(npy_intp nelem_in_dim, npy_intp block_size,
                 int proc_dim_rank, int nproc_in_dim,
                 int first_process)
{
    //Figure process's distance from source process.
    int mydist = (nproc_in_dim + proc_dim_rank - first_process) %
                  nproc_in_dim;

    //Figure the total number of whole NB blocks N is split up into.
    npy_intp nblocks = nelem_in_dim / block_size;

    //Figure the minimum number of elements a process can have.
    npy_intp numroc = nblocks / nproc_in_dim * block_size;

    //See if there are any extra blocks
    npy_intp extrablocks = nblocks % nproc_in_dim;

    //If I have an extra block.
    if(mydist < extrablocks)
        numroc += block_size;

    //If I have last block, it may be a partial block.
    else if(mydist == extrablocks)
        numroc += nelem_in_dim % block_size;

    return numroc;
} /* dnumroc */

/*===================================================================
 *
 * Process cartesian coords <-> MPI rank.
 */
int cart2rank(int ndims, const int coords[NPY_MAXDIMS])
{
    int *strides = cart_dim_strides[ndims-1];
    int rank = 0;
    int i;
    for(i=0; i<ndims; i++)
        rank += coords[i] * strides[i];
    assert(rank < worldsize);
    return rank;
}
void rank2cart(int ndims, int rank, int coords[NPY_MAXDIMS])
{
    int i;
    int *strides = cart_dim_strides[ndims-1];
    memset(coords, 0, ndims*sizeof(int));
    for(i=0; i<ndims; i++)
    {
        coords[i] = rank / strides[i];
        rank = rank % strides[i];
    }
} /* cart2rank & rank2cart */


/*===================================================================
 *
 * Sends a message to all slaves.
 * msgsize is in bytes.
 */
#ifndef DNPY_SPMD
void msg2slaves(npy_intp *msg, int msgsize)
{
    if(msgsize > DNPY_MAX_MSG_SIZE)
    {
        fprintf(stderr, "msg2slaves, the messages is greater "
                        "than DNPY_MAX_MSG_SIZE\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif

    MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.msg2slaves)
    #endif

} /* msg2slaves */
#endif
