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

#include <errno.h>
#include <sys/mman.h>
#include <signal.h>

/*
 *===================================================================
 * Check whether the array distributed or not.
 */
static int
PyDistArray_IsDist(PyArrayObject *ary)
{
    if(PyDistArray_ARRAY(ary) != NULL)
        return PyDistArray_ARRAY(ary)->base->isdist;
    return 0;//False
}/* PyDistArray_IsDist */

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * If 'one_node_dist_rank' is positive it specifies the rank of an
 * one-node-distribution.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewBaseArray(PyArrayObject *ary, npy_intp one_node_dist_rank)
{
    int i;

    //Make sure that the init message has been handled.
    if(initmsg_not_handled)
        PyDistArray_ProcGridSet(NULL,NULL);

    //Create dndarray.
    dndarray newarray;
    newarray.dtype = PyArray_TYPE(ary);
    newarray.elsize = PyArray_ITEMSIZE(ary);
    newarray.ndims = PyArray_NDIM(ary);
    newarray.nelem = PyArray_SIZE(ary);
    newarray.isdist = 1;
    newarray.refcount = 1;
    newarray.onerank = one_node_dist_rank;
    for(i=0; i<PyArray_NDIM(ary); i++)
        newarray.dims[i] = PyArray_DIM(ary, i);

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    dndview newview;
    newview.uid = ++uid_count;
    newview.nslice = PyArray_NDIM(ary);
    newview.ndims = PyArray_NDIM(ary);
    newview.alterations = 0;
    for(i=0; i<PyArray_NDIM(ary); i++)
    {
        //Default the view will span over the whole array.
        newview.slice[i].start = 0;
        newview.slice[i].step = 1;
        newview.slice[i].nsteps = PyArray_DIM(ary, i);
    }

#ifndef DNPY_SPMD
    //Tell slaves about the new array
    msg[0] = DNPY_CREATE_ARRAY;
    memcpy(&msg[1], &newarray, sizeof(dndarray));
    memcpy(((char *) &msg[1]) + sizeof(dndarray), &newview,
           sizeof(dndview));

    *(((char *) &msg[1])+sizeof(dndarray)+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg,2*sizeof(npy_intp)+sizeof(dndarray)+sizeof(dndview));
#endif

    dndview *ret = handle_NewBaseArray(&newarray, &newview);

    if(ret == NULL)
        return -1;

    PyDistArray_ARRAY(ary) = ret;
    ret->base->pyary = ary;

    //Protect the original NumPy data pointer.
    //This is only done by the Master MPI Process.
    return arydat_malloc(ary);
} /* PyDistArray_NewBaseArray */


/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return NULL and set exception on error.
 * Return a pointer to the new dndview on success.
 */
dndview *handle_NewBaseArray(dndarray *array, dndview *view)
{
    int ndims = array->ndims;
    int *cdims = cart_dim_sizes[ndims-1];
    npy_intp i;
    int cartcoord[NPY_MAXDIMS];

    ++ndndarrays;

    //Save array uid.
    array->uid = view->uid;

    //Save the new array-base.
    dndarray *ary = put_dndarray(array);
    view->base = ary;

    //Append the array to the linked list.
    ary->prev = NULL;
    ary->next = rootarray;
    rootarray = ary;
    if(ary->next != NULL)
    {
        assert(ary->next->prev == NULL);
        ary->next->prev = rootarray;
    }

    //Get cartesian coords.
    rank2cart(ndims, myrank, cartcoord);

    //Accumulate the total number of local sizes and save it.
    npy_intp localsize = 1;
    ary->nblocks = 1;
    for(i=0; i < ary->ndims; i++)
    {
        if(ary->onerank < 0)
            ary->localdims[i] = dnumroc(ary->dims[i], blocksize,
                                        cartcoord[i], cdims[i], 0);
        else
            ary->localdims[i] = (ary->onerank==myrank)?ary->dims[i]:0;

        localsize *= ary->localdims[i];
        ary->localblockdims[i] = ceil(ary->localdims[i] /
                                      (double) blocksize);
        ary->blockdims[i] = ceil(ary->dims[i] / (double) blocksize);
        ary->nblocks *= ary->blockdims[i];
    }
    ary->localsize = localsize;
    if(ary->localsize == 0)
    {
        memset(ary->localdims, 0, ary->ndims * sizeof(npy_intp));
        memset(ary->localblockdims, 0, ary->ndims * sizeof(npy_intp));
    }
    if(ary->nblocks == 0)
        memset(ary->blockdims, 0, ary->ndims * sizeof(npy_intp));

    //Allocate the root nodes array.
    ary->rootnodes = malloc(ary->nblocks * sizeof(dndnode*));
    if(ary->rootnodes == NULL)
    {
        PyErr_NoMemory();
        return NULL;
    }
    for(i=0; i<ary->nblocks; i++)
        ary->rootnodes[i] = NULL;

    //The memory allocation is delayed to the point where it is used.
    ary->data = NULL;

    //Create a MPI-datatype that correspond to an array element.
    MPI_Type_contiguous(ary->elsize, MPI_BYTE, &ary->mpi_dtype);
    MPI_Type_commit(&ary->mpi_dtype);

    //Compute number of blocks.
    view->nblocks = 1;
    for(i=0; i<ndims;i++)
    {
        view->blockdims[i] = ceil(ary->dims[i] / (double) blocksize);
        view->nblocks *= view->blockdims[i];
    }
    if(view->nblocks == 0)
        memset(view->blockdims, 0, ndims * sizeof(npy_intp));

    //Save and return the new view.
    return put_dndview(view);
} /* handle_NewBaseArray */


/*
 *===================================================================
 * Create a new view of an array and updates the PyArrayObject.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewViewArray(PyArrayObject *orig_ary, PyArrayObject *new_ary,
                         int nslice, dndslice slice[NPY_MAXDIMS])
{
    dndview *orgview = PyDistArray_ARRAY(orig_ary);

    //Create new view based on 'org_view' and the 'slice'.
    dndview newview;
    newview.uid = ++uid_count;
    newview.ndims = 0;
    newview.alterations = 0;
    newview.nblocks = 1;

    //Merging the two views.
    int si = 0; //slice index.
    int ni = 0; //new index.
    int oi = 0; //old index.
    int di = 0; //dim index.
    while(si < nslice || oi < orgview->nslice)
    {
        //If we come to the end of the slices, that happens
        //if not all dimensions is included in the 'slices', we will
        //use the whole dimension.
        int vs = (si < nslice)?1:0;//Valid slice.

        //If dimension is invisible we will just copy it to 'newview'.
        if(oi < orgview->nslice &&
           orgview->slice[oi].nsteps == SingleIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));
            ni++; oi++; di++;
            newview.alterations |= DNPY_NDIMS;
        }
        //A single index makes the dimension invisible.
        else if(vs && slice[si].nsteps == SingleIndex)
        {
            //If dimension is a Pseudo-dimension then just go to next
            //dimension.
            if(orgview->slice[oi].nsteps == PseudoIndex)
            {
                si++; oi++;
            }
            else
            {//Copy single index to 'newview'.
                newview.slice[ni].step = 0;
                newview.slice[ni].nsteps = SingleIndex;
                newview.slice[ni].start = orgview->slice[oi].start +
                                          (vs?slice[si].start:0) *
                                          orgview->slice[oi].step;
                si++; ni++; oi++; di++;
                newview.alterations |= DNPY_NDIMS;
            }
        }
        //If a extra pseudo index should be added we just copy the
        //slice to 'newview'.
        else if(vs && slice[si].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &slice[si], sizeof(dndslice));
            ni++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        else if(orgview->slice[oi].nsteps == PseudoIndex)
        {
            memcpy(&newview.slice[ni], &orgview->slice[oi],
                   sizeof(dndslice));

            if(slice[si].start > 0)
            {
                //This is a special case where the user indexes the
                //PseudoIndex, which is legal and will return [].
                newview.nblocks = 0;
            }

            ni++; oi++; si++;
            newview.ndims++;
            newview.alterations |= DNPY_NDIMS;
        }
        //If no special slices we just merge the two views.
        else
        {
            if(vs)
            {
                newview.slice[ni].start = orgview->slice[oi].start +
                                           slice[si].start *
                                           orgview->slice[oi].step;
                newview.slice[ni].step = slice[si].step *
                                          orgview->slice[oi].step;
                newview.slice[ni].nsteps = slice[si].nsteps;
            }
            else
                memcpy(&newview.slice[ni], &orgview->slice[oi],
                       sizeof(dndslice));

            if(newview.slice[ni].step > 1)
                newview.alterations |= DNPY_STEP | DNPY_NSTEPS;
            else if(newview.slice[ni].nsteps < orgview->base->dims[di])
            {
                newview.alterations |= DNPY_NSTEPS;
            }
            newview.ndims++;
            si++; ni++; oi++; di++;
        }
    }
    //Save the total number of sliceses for the new view.
    newview.nslice = ni;

    //Check if the view is not block alligned
    for(si=0; si<newview.nslice; ++si)
        if(newview.slice[si].start % blocksize != 0 ||
           newview.slice[si].step != 1)
        {
            newview.alterations |= DNPY_NONALIGNED;
            break;
        }

#ifndef DNPY_SPMD
    //Tell slaves about the new view.
    //NB: It is up to the slaves to add the newview.base adresse.
    msg[0] = DNPY_CREATE_VIEW;
    msg[1] = orgview->uid;
    memcpy(&msg[2], &newview, sizeof(dndview));
    *(((char *) &msg[2])+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + sizeof(dndview));
#endif

    PyDistArray_ARRAY(new_ary) = handle_NewViewArray(orgview, &newview);

    return 0;

}/* PyDistArray_NewViewArray */


/*===================================================================
 *
 * Handler for PyDistArray_NewViewArray.
 * Return NULL and set exception on error.
 * Return a pointer to the new dndview on success.
 */
dndview *handle_NewViewArray(dndview *orgview, dndview *newview)
{
    npy_intp n, i, j;

    //Add the base to the new view.
    assert(orgview->base->refcount > 0);
    newview->base = orgview->base;
    newview->base->refcount++;

    //Compute size of view-blocks.
    if(newview->nblocks == 0)//The view is empty
    {
        memset(newview->blockdims, 0, newview->ndims * sizeof(npy_intp));
    }
    else
    {
        newview->nblocks = 1;
        n=0;
        for(i=0; i < newview->nslice; i++)
        {
            if(newview->slice[i].nsteps != SingleIndex)
            {
                j = 1;//SingleIndex has length one.
                if(newview->slice[i].nsteps != PseudoIndex)
                    j = newview->slice[i].nsteps;

                newview->blockdims[n] = ceil(j / (double) blocksize);
                newview->nblocks *= newview->blockdims[n];
                n++;
            }
        }
    }
    //Save and return the view.
    return put_dndview(newview);
}/* handle_NewViewArray */

/*
 *===================================================================
 * Delete array view.
 * When it is the last view of the base array, the base array is de-
 * allocated.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_DelViewArray(PyArrayObject *array)
{
    //Get arrray structs.
    dndview *ary = PyDistArray_ARRAY(array);

#ifndef DNPY_SPMD
    //Tell slaves about the destruction
    msg[0] = DNPY_DESTROY_ARRAY;
    msg[1] = ary->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg,3 * sizeof(npy_intp));
#endif

    if(handle_DelViewArray(ary->uid) == -1)
        return -1;

    //We have to free the protected data pointer when the NumPy array
    //is not a view.
    if((array->flags & NPY_OWNDATA) && array->data != NULL)
        return arydat_free(array);

    return 0;

} /* PyDistArray_DelViewArray */

/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_DelViewArray(npy_intp uid)
{
    dndview *view = get_dndview(uid);

    dndop *op = workbuf_nextfree;
    WORKBUF_INC(sizeof(dndop));
    op->op = DNPY_DESTROY_ARRAY;
    op->optype = DNPY_NONCOMM;
    op->narys = 1;
    op->refcount = 0;
    op->views[0] = view;
    op->svbs[0] = NULL;//Whole array.
    op->accesstypes[0] = DNPY_WRITE;

    dndnode *node = workbuf_nextfree;
    WORKBUF_INC(sizeof(dndnode));
    node->op = op;
    node->op_ary_idx = 0;
    dep_add(node, 1, 0);

    return 0;
} /* handle_DelViewArray */

/*
 *===================================================================
 * Assign the value to array at coordinate.
 * 'coord' size must be the same as view->ndims.
 * Steals all reference to item. (Item is lost).
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_PutItem(PyArrayObject *ary, npy_intp coord[NPY_MAXDIMS],
                    PyObject *item)
{
    //Get arrray structs.
    dndview *view = PyDistArray_ARRAY(ary);

    //Convert item to a compatible type.
    PyObject *item2 = PyArray_FROM_O(item);
    PyObject *citem2 = PyArray_Cast((PyArrayObject*)item2,
                                    view->base->dtype);

    //Cleanup and return error if the cast failed.
    if(citem2 == NULL)
    {
        Py_DECREF(item2);
        return -1;
    }

#ifndef DNPY_SPMD
    int ndims = view->ndims;
    int elsize = view->base->elsize;
    //Tell slaves about the new item.
    msg[0] = DNPY_PUT_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], PyArray_DATA(citem2), elsize);
    memcpy(((char *) &msg[2]) + elsize, coord,
           sizeof(npy_intp) * ndims);
    *(((char *) &msg[2]) + elsize + sizeof(npy_intp) * ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3 * sizeof(npy_intp) + elsize +
                    ndims * sizeof(npy_intp));
#endif

    handle_PutGetItem(1, view, PyArray_DATA(citem2), coord);

    //Clean up.
    Py_DECREF(citem2);
    Py_DECREF(item2);

    return 0;//Succes
} /* PyDistArray_PutItem */

/*
 *===================================================================
 * Get a single value specified by coordinate from the array.
 * 'coord' size must be the same as view->ndims.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_GetItem(PyArrayObject *ary, char *retdata,
                    npy_intp coord[NPY_MAXDIMS])
{
    //Get arrray structs.
    dndview *view = PyDistArray_ARRAY(ary);

#ifndef DNPY_SPMD
    //Tell slaves to send item.
    msg[0] = DNPY_GET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], coord, sizeof(npy_intp)*view->ndims);
    *(((char *) &msg[2]) + sizeof(npy_intp)*view->ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3*sizeof(npy_intp) + view->ndims*sizeof(npy_intp));
#endif

    handle_PutGetItem(0, view, retdata, coord);

    return 0;
} /* PyDistArray_GetItem */

/*===================================================================
 *
 * Handler for PyDistArray_PutItem and PyDistArray_GetItem.
 * Direction: 0=Get, 1=Put.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_PutGetItem(int Direction, dndview *view, char* item,
                      npy_intp coord[NPY_MAXDIMS])
{
    npy_intp i,j,b,offset;
    npy_intp n, s;
    npy_intp tcoord[NPY_MAXDIMS];
    npy_intp rcoord[NPY_MAXDIMS];
    npy_intp nsteps[NPY_MAXDIMS];
    npy_intp step[NPY_MAXDIMS];
    char *data;
    dndvb vblock;

    dep_flush(1);

    //Convert to block coordinates.
    for(i=0; i<view->ndims; i++)
        tcoord[i] = coord[i] / blocksize;

    //Get view block info.
    calc_vblock(view, tcoord, &vblock);

    //Convert PseudoIndex and SingleIndex.
    j=0;n=0;
    for(i=0; i < view->nslice; i++)
    {
        if(view->slice[i].nsteps == PseudoIndex)
        {
            assert(view->slice[i].start == 0);
            n++;
        }
        else if(view->slice[i].nsteps == SingleIndex)
        {
            rcoord[j] = 0;//The offset is already incl. in the svb.
            nsteps[j] = 1;
            step[j] = 1;
            j++;
        }
        else
        {
            rcoord[j] = coord[n];
            nsteps[j] = view->slice[i].nsteps;
            step[j] = view->slice[i].step;
            j++; n++;
        }
    }

    //Convert global coordinate to index coordinates
    //relative to the view.
    for(i=0; i<view->base->ndims; i++)
        tcoord[i] = rcoord[i] % blocksize;

    //Find sub view block and convert icoord to coordinate
    //relative to the sub view block.
    s=1;b=0;
    for(i=view->base->ndims-1; i>=0; i--)//Row-major.
    {
        j = vblock.sub[b].nsteps[i];
        while(tcoord[i] >= vblock.sub[b].nsteps[i])
        {
            tcoord[i] -= vblock.sub[b].nsteps[i];
            j += vblock.sub[b].nsteps[i];
            b += s;
        }
        while(j < MIN(blocksize, nsteps[i] - rcoord[i]))
        {
            j += vblock.sub[b].nsteps[i];
        }
        s *= vblock.svbdims[i];
    }

    //Compute offset.
    offset = 0;
    for(i=view->base->ndims-1; i>=0; i--)//Row-major.
        offset += (vblock.sub[b].start[i] + tcoord[i] * step[i]) *
                   vblock.sub[b].stride[i];
    delayed_array_allocation(view->base);
    data = view->base->data + offset*view->base->elsize;

#ifndef DNPY_SPMD
    if(vblock.sub[b].rank == 0)//Local copying.
    {
        if(myrank == 0)
        {
            if(Direction)
                memcpy(data, item, view->base->elsize);
            else
                memcpy(item, data, view->base->elsize);
        }
    }
    else if(myrank == 0)
    {
        if(Direction)
            MPI_Ssend(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD);
        else
            MPI_Recv(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else if(myrank == vblock.sub[b].rank)
    {
        if(Direction)
            MPI_Recv(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Ssend(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD);
    }
#else
    if(Direction)
    {
        if(vblock.sub[b].rank == 0)//Local copying.
        {
            if(myrank == 0)
                memcpy(data, item, view->base->elsize);
        }
        else if(myrank == 0)
        {
            MPI_Ssend(item, 1, view->base->mpi_dtype, vblock.sub[b].rank,
                     0, MPI_COMM_WORLD);
        }
        else if(myrank == vblock.sub[b].rank)
        {
            MPI_Recv(data, 1, view->base->mpi_dtype, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        if(vblock.sub[b].rank == myrank)//Local copying.
            memcpy(item, data, view->base->elsize);

        MPI_Bcast(item, view->base->elsize, MPI_BYTE, vblock.sub[b].rank,
                  MPI_COMM_WORLD);
    }
#endif

    dep_flush(1);//Will cleanup the used sub-view-blocks.

    return 0;
} /* handle_PutGetItem */


/*===================================================================
 *
 * Un-distributes the array by transferring all data to the master
 * MPI-process.
 * Return -1 and set exception on error, 0 on success.
 */
int PyDistArray_UnDist(dndarray *ary)
{
    #ifndef DNPY_SPMD
        msg[0] = DNPY_UNDIST;
        msg[1] = ary->uid;
        msg[2] = DNPY_MSG_END;
        msg2slaves(msg, 3*sizeof(npy_intp));
    #endif

    if(ary->isdist)
    {
        //Un-protect the memory.
        if(mprotect(PyArray_DATA(ary->pyary), ary->nelem * ary->elsize,
                    PROT_READ|PROT_WRITE) == -1)
        {
            int errsv = errno;//mprotect() sets the errno.
            PyErr_Format(PyExc_RuntimeError, "PyDistArray_UnDist: "
                         "could not un-protect a data region. "
                         "Returned error code by mprotect: %s.",
                         strerror(errsv));
            return -1;
        }

        //Transfer all items to the pyary.
        npy_intp coord[NPY_MAXDIMS];
        memset(coord, 0, ary->ndims * sizeof(npy_intp));
        int notfinished = 1;
        char *data = PyArray_DATA(ary->pyary);
        while(notfinished)
        {
            PyDistArray_GetItem(ary->pyary, data, coord);
            data += ary->elsize;
            //Go to next coordinate.
            int i;
            for(i=ary->ndims-1; i >= 0; i--)
            {
                coord[i]++;
                if(coord[i] >= ary->dims[i])
                {
                    //We are finished, if wrapping around.
                    if(i == 0)
                    {
                        notfinished = 0;
                        break;
                    }
                    coord[i] = 0;//Start coord.
                }
                else
                    break;
            }
        }
    }
    ary->isdist = 0;//Not distributed anymore.
    return 0;
} /* PyDistArray_UnDist */


/*===================================================================
 *
 * Handler for PyDistArray_UnDist.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_UnDist(dndarray *ary)
{


    return 0;
} /* handle_UnDist */
