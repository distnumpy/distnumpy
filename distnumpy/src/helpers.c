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
}/* calc_svb_MPIdatatype */


/*===================================================================
 *
 * Calculate the view block at the specified block-coordinate.
 * NB: vcoord is the visible coordinates and must therefore have
 * length view->ndims.
 */
void calc_vblock(const dndview *view, const npy_intp vcoord[NPY_MAXDIMS],
                 dndvb *vblock)
{
    npy_intp i, j, B, item_idx, s, offset, goffset, voffset, boffset;
    npy_intp notfinished, stride, vitems, vvitems, vblocksize;
    npy_intp comm_offset, nelem;
    npy_intp coord[NPY_MAXDIMS];
    npy_intp scoord[NPY_MAXDIMS];
    npy_intp ncoord[NPY_MAXDIMS];
    int pcoord[NPY_MAXDIMS];
    int *cdims = cart_dim_sizes[view->base->ndims-1];
    npy_intp start[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];
    dndsvb *svb;

    //Convert vcoord to coord, which have length view->base->ndims.
    j=0;s=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == PseudoIndex)
        {
            assert(vcoord[s] == 0);
            s++;
            continue;
        }
        if(view->slice[i].nsteps == SingleIndex)
        {
            nsteps[j] = 1;
            step[j] = 1;
            coord[j] = 0;
        }
        else
        {
            coord[j] = vcoord[s];
            nsteps[j] = view->slice[i].nsteps;
            step[j] = view->slice[i].step;
            s++;
        }
        assert(nsteps[j] > 0);
        start[j++] = view->slice[i].start;
    }
    assert(j == view->base->ndims);

    vblock->sub = workbuf_nextfree;
    svb = vblock->sub;

    //Init number of sub-view-block in each dimension.
    memset(vblock->svbdims, 0, view->base->ndims * sizeof(npy_intp));

    //Sub-vblocks coordinate.
    memset(scoord, 0, view->base->ndims * sizeof(npy_intp));
    //Compute all sub-vblocks associated with the n'th vblock.
    notfinished=1; s=0;
    while(notfinished)
    {
        dndnode **rootnode = view->base->rootnodes;
        stride = 1;
        for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
        {
            //Non-block coordinates.
            ncoord[i] = coord[i] * blocksize;
            //View offset relative to array-view (non-block offset).
            voffset = ncoord[i] + scoord[i];
            //Global offset relative to array-base.
            goffset = voffset * step[i] + start[i];
            //Global block offset relative to array-base.
            B = goffset / blocksize;
            //Compute this sub-view-block's root node.
            rootnode += B * stride;
            stride *= view->base->blockdims[i];
            //Process rank of the owner in the i'th dimension.
            pcoord[i] = B % cdims[i];
            //Local block offset relative to array-base.
            boffset = B / cdims[i];
            //Item index local to the block.
            item_idx = goffset % blocksize;
            //Local offset relative to array-base.
            offset = boffset * blocksize + item_idx;
            //Save offset.
            svb[s].start[i] = offset;
            //Viewable items left in the block.
            vitems = MAX((blocksize - item_idx) / step[i], 1);
            //Size of current view block.
            vblocksize = MIN(blocksize, nsteps[i] - ncoord[i]);
            //Viewable items left in the view-block.
            vvitems = vblocksize - (voffset % blocksize);
            //Compute nsteps.
            svb[s].nsteps[i] = MIN(blocksize, MIN(vvitems, vitems));
            //Debug check.
            assert(svb[s].nsteps[i] > 0);
        }
        //Find rank.
        if(view->base->onerank < 0)
            svb[s].rank = cart2rank(view->base->ndims,pcoord);
        else
            svb[s].rank = view->base->onerank;

        assert(svb[s].rank >= 0);
        //Data has not been fetched.
        svb[s].data = NULL;
        //Communication has not been handled.
        svb[s].comm_received_by = -1;
        //Save rootnode.
        svb[s].rootnode = rootnode;

        //Compute the strides (we need the rank to do this).
        stride = 1;
        if(view->base->onerank < 0)
            for(i=view->base->ndims-1; i >= 0; i--)
            {
                svb[s].stride[i] = stride;
                stride *= dnumroc(view->base->dims[i], blocksize,
                                  pcoord[i], cdims[i], 0);
                assert(svb[s].stride[i] > 0);
            }
        else//All on one rank.
            for(i=view->base->ndims-1; i >= 0; i--)
            {
                svb[s].stride[i] = stride;
                stride = view->base->dims[i];
            }

        //Compute the MPI datatype for communication.
        comm_offset = 0;
        nelem = 1;
        for(i=view->base->ndims-1; i >= 0; i--)//Row-major.
        {
            //Compute offsets.
            comm_offset += svb[s].start[i] * svb[s].stride[i];
            //Computing total number of elements.
            nelem *= svb[s].nsteps[i];
        }
        //Save offsets.
        svb[s].comm_offset = comm_offset * view->base->elsize;

        //Save total number of elements.
        svb[s].nelem = nelem;

        //Save data pointer if local data.
        if(svb[s].rank == myrank)
        {
            delayed_array_allocation(view->base);
            vblock->sub[s].data = view->base->data + svb[s].comm_offset;
        }

        //Iterate Sub-vblocks coordinate (Row-major).
        for(j=view->base->ndims-1; j >= 0; j--)
        {
            //Count svbdims.
            vblock->svbdims[j]++;

            scoord[j] += svb[s].nsteps[j];
            if(scoord[j] >= MIN(blocksize, nsteps[j] - ncoord[j]))
            {
                //We a finished, if wrapping around.
                if(j == 0)
                {
                    notfinished = 0;
                    break;
                }
                scoord[j] = 0;
            }
            else
                break;
        }
        //Reset svbdims because we need the last iteration.
        if(notfinished)
            for(i=view->base->ndims-1; i > j; i--)
                vblock->svbdims[i] = 0;

        s++;
    }
    //Save number of sub-vblocks.
    vblock->nsub = s;
    assert(vblock->nsub > 0);
    //And the next free work buffer slot.
    WORKBUF_INC(s * sizeof(dndsvb));
} /* calc_vblock */

/*===================================================================
 *
 * Convert visible vblock dimension index to base vblock
 * dimension index.
 */
npy_intp idx_v2b(const dndview *view, npy_intp vindex)
{
    assert(vindex < view->ndims);
    npy_intp i, bindex=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
        {
            if(view->base->ndims > 1)
                bindex++;
            continue;
        }
        if(vindex == 0)
            break;
        if(view->slice[i].nsteps != PseudoIndex)
            bindex++;
        vindex--;
    }
    //We need the MIN since bindex is too high when PseudoIndex is
    //used at the end of the view.
    return MIN(bindex, view->base->ndims-1);
} /* idx_v2b */

/*===================================================================
 *
 * Convert visible vblock dimension index to slice dimension index.
 */
npy_intp idx_v2s(const dndview *view, npy_intp vindex)
{
    npy_intp i;
    assert(vindex < view->ndims);
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == SingleIndex)
            continue;
        if(vindex == 0)
            break;
        vindex--;
    }
    assert(i < view->nslice);
    return i;
} /* idx_v2s */
