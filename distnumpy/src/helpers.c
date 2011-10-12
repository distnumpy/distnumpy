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


/*===================================================================
 *
 * Returns a string describing the operation type.
 */
char *optype2str(int optype)
{
    switch(optype)
    {
        case DNPY_CREATE_ARRAY:
            return "DNPY_CREATE_ARRAY";
        case DNPY_DESTROY_ARRAY:
            return "del";
        case DNPY_CREATE_VIEW:
            return "DNPY_CREATE_VIEW";
        case DNPY_PUT_ITEM:
            return "DNPY_PUT_ITEM";
        case DNPY_GET_ITEM:
            return "DNPY_GET_ITEM";
        case DNPY_UFUNC:
            return "ufunc";
        case DNPY_RECV:
            return "recv";
        case DNPY_SEND:
            return "send";
        case DNPY_BUF_RECV:
            return "Brecv";
        case DNPY_BUF_SEND:
            return "Bsend";
        case DNPY_APPLY:
            return "apply";
        case DNPY_UFUNC_REDUCE:
            return "DNPY_UFUNC_REDUCE";
        case DNPY_ZEROFILL:
            return "DNPY_ZEROFILL";
        case DNPY_DATAFILL:
            return "DNPY_DATAFILL";
        case DNPY_DIAGONAL:
            return "DNPY_DIAGONAL";
        case DNPY_MATMUL:
            return "DNPY_MATMUL";
        case DNPY_REDUCE_SEND:
            return "reduce_send";
        case DNPY_REDUCE_RECV:
            return "DNPY_REDUCE_RECV";
        default:
            return "\"Unknown data type\"";
    }
} /* optype2str */


/*===================================================================
 *  Returns a MPI data type that match the specified sub-view-block.
 */
static MPI_Datatype
calc_svb_MPIdatatype(const dndview *view, dndsvb *svb)
{
    /*
    npy_intp i,j,stride;
    MPI_Datatype comm_viewOLD, comm_viewNEW;
    npy_intp start[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];

    //Convert vcoord to coord, which have length view->base->ndims.
    j=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == PseudoIndex)
        {
            continue;
        }
        if(view->slice[i].nsteps == SingleIndex)
        {
            nsteps[j] = 1;
            step[j] = 1;
        }
        else
        {
            nsteps[j] = view->slice[i].nsteps;
            step[j] = view->slice[i].step;
        }
        start[j++] = view->slice[i].start;
    }

    //Compute the MPI datatype for communication.
    MPI_Type_dup(view->base->mpi_dtype, &comm_viewOLD);
    for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
    {
        //Compute the MPI datatype for the view.
        stride = svb->stride[i] * step[i] * view->base->elsize;
        MPI_Type_create_hvector(svb->nsteps[i], 1, stride,
                                comm_viewOLD, &comm_viewNEW);

        //Cleanup and iterate comm types.
        MPI_Type_free(&comm_viewOLD);
        comm_viewOLD = comm_viewNEW;
    }
    MPI_Type_commit(&comm_viewNEW);
    return comm_viewNEW;
    */
    return 0;
}/* calc_svb_MPIdatatype */
