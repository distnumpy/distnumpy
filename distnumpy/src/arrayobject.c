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

/*
 *===================================================================
 * Create a new base array.
 * If 'one_node_dist_rank' is positive it specifies the rank of an
 * one-node-distribution.
 * Returns the uid of the view of the new base array.
 * Return -1 and set exception on error.
 */
static npy_intp
PyDistArray_NewBaseArray(PyArrayObject *ary, npy_intp one_node_dist_rank)
{
    int i;

    //Make sure that the init message has been handled.
//    if(initmsg_not_handled)
//        dnumpy_init_data_layout(NULL,NULL);

    //Create dndarray.
    dndarray *newarray = malloc(sizeof(dndarray));
    dndview *newview = malloc(sizeof(dndview));
    if(newarray == NULL || newview == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }
    newarray->dtype = PyArray_TYPE(ary);
    newarray->elsize = PyArray_ITEMSIZE(ary);
    newarray->ndims = PyArray_NDIM(ary);
    newarray->refcount = 1;
    newarray->onerank = one_node_dist_rank;
    for(i=0; i<PyArray_NDIM(ary); i++)
        newarray->dims[i] = PyArray_DIM(ary, i);

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    newview->uid = ++uid_count;
    newview->nslice = PyArray_NDIM(ary);
    newview->ndims = PyArray_NDIM(ary);
    newview->alterations = 0;
    for(i=0; i<PyArray_NDIM(ary); i++)
    {
        //Default the view will span over the whole array.
        newview->slice[i].start = 0;
        newview->slice[i].step = 1;
        newview->slice[i].nsteps = PyArray_DIM(ary, i);
    }

#ifndef DNPY_SPMD
    //Tell slaves about the new array
    msg[0] = DNPY_CREATE_ARRAY;
    memcpy(&msg[1], newarray, sizeof(dndarray));
    memcpy(((char *) &msg[1]) + sizeof(dndarray), newview,
           sizeof(dndview));

    *(((char *) &msg[1])+sizeof(dndarray)+sizeof(dndview)) = DNPY_MSG_END;

    msg2slaves(msg,2*sizeof(npy_intp)+sizeof(dndarray)+sizeof(dndview));
#endif

    if(handle_NewBaseArray(newarray, newview) < 0)
        return -1;

    //Freeup memory.
    free(newarray);
    free(newview);

    return uid_count;
} /* PyDistArray_NewBaseArray */


/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_NewBaseArray(dndarray *ary, dndview *view)
{
    int ndims = ary->ndims;
    int *cdims = cart_dim_sizes[ndims-1];
    npy_intp i;
    int cartcoord[NPY_MAXDIMS];

    ++ndndarrays;

    //Save array uid.
    ary->uid = view->uid;

    //Allocate and copy the array to the views's base.
    view->base = malloc(sizeof(dndarray));
    if(view->base == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(view->base, ary, sizeof(dndarray));
    ary = view->base;//Use the new pointer.

    //Append the array to the lisked list when statistic is defined.
    #ifdef DNPY_STATISTICS
        ary->prev = NULL;
        ary->next = rootarray;
        rootarray = ary;
        if(ary->next != NULL)
        {
            assert(ary->next->prev == NULL);
            ary->next->prev = rootarray;
        }
    #endif

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
        return -1;
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

    //Save the new view.
    put_dndview(view);
    return 0;
} /* handle_NewBaseArray */

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
    dndview *ary = get_dndview(array->dnduid);

#ifndef DNPY_SPMD
    //Tell slaves about the destruction
    msg[0] = DNPY_DESTROY_ARRAY;
    msg[1] = ary->uid;
    msg[2] = DNPY_MSG_END;
    msg2slaves(msg,3 * sizeof(npy_intp));
#endif

    return handle_DelViewArray(ary->uid);
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
    //dag_svb_add(node, 1, 0);

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
    dndview *view = get_dndview(ary->dnduid);

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

    //do_PUTGET_ITEM(1, view, PyArray_DATA(citem2), coord);

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
    dndview *view = get_dndview(ary->dnduid);

#ifndef DNPY_SPMD
    //Tell slaves to send item.
    msg[0] = DNPY_GET_ITEM;
    msg[1] = view->uid;
    memcpy(&msg[2], coord, sizeof(npy_intp)*view->ndims);
    *(((char *) &msg[2]) + sizeof(npy_intp)*view->ndims) = DNPY_MSG_END;

    msg2slaves(msg, 3*sizeof(npy_intp) + view->ndims*sizeof(npy_intp));
#endif

//    do_PUTGET_ITEM(0, view, retdata, coordinate);

    return 0;
} /* PyDistArray_GetItem */