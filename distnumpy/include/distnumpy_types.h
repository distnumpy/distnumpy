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

#ifndef DISTNUMPY_TYPES_H
#define DISTNUMPY_TYPES_H
#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>
#include <numpy/ndarraytypes.h>

//Datatype prototypes.
typedef struct dndnode_struct dndnode;
typedef struct dndarray_struct dndarray;

//Type describing a distributed array.
struct dndarray_struct
{
    //Unique identification.
    npy_intp uid;
    //Reference count.
    int refcount;
    //Number of dimensions.
    int ndims;
    //Size of dimensions.
    npy_intp dims[NPY_MAXDIMS];
    //Size of block-dimensions.
    npy_intp blockdims[NPY_MAXDIMS];
    //Number of blocks (global).
    npy_intp nblocks;
    //Data type of elements in array.
    int dtype;
    //Size of an element in bytes.
    int elsize;
    //Pointer to local data.
    char *data;
    //Number of elements (global).
    npy_intp nelem;
    //Number of local elements (local to the MPI-process).
    npy_intp localsize;
    //Size of local dimensions (local to the MPI-process).
    npy_intp localdims[NPY_MAXDIMS];
    //Size of local block-dimensions (local to the MPI-process).
    npy_intp localblockdims[NPY_MAXDIMS];
    //MPI-datatype that correspond to an array element.
    MPI_Datatype mpi_dtype;
    //Root nodes (one per block).
    dndnode **rootnodes;
    //Next and prev are used for traversing all arrays.
    dndarray *next;
    dndarray *prev;
    //When onerank is positiv this array is only distributed on that
    //MPI-process rank.
    npy_intp onerank;
    //Memory protected start address (incl.).
    npy_intp mprotected_start;
    //memory protected end address (excl.).
    npy_intp mprotected_end;
};

//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...] (Not used in DistNumPy)
#define SingleIndex -3//Dim not visible - 'A[1]'

//Type describing a slice of a dimension.
typedef struct
{
    //Start index.
    npy_intp start;
    //Elements between index.
    npy_intp step;
    //Number of steps (Length of the dimension).
    npy_intp nsteps;
} dndslice;

//View-alteration flags.
#define DNPY_NDIMS      0x001
#define DNPY_STEP       0x002
#define DNPY_NSTEPS     0x004
#define DNPY_NONALIGNED 0x008

//Type describing a view of a distributed array.
typedef struct
{
    //Unique identification.
    npy_intp uid;
    //The array this view is a view of.
    dndarray *base;
    //Number of viewable dimensions.
    int ndims;
    //Number of sliceses. NB: nslice >= base->ndims.
    int nslice;
    //Sliceses - the global view of the base-array.
    dndslice slice[NPY_MAXDIMS];
    //A bit mask specifying which alterations this view represents.
    //Possible flags:
    //Zero            - no alterations.
    //DNPY_NDIMS      - number of dimensions altered.
    //DNPY_STEP       - 'step' altered.
    //DNPY_NSTEPS     - 'nsteps' altered.
    //DNPY_NONALIGNED - 'start % blocksize != 0' or 'step != 1'.
    int alterations;
    //Number of view-blocks.
    npy_intp nblocks;
    //Number of view-blocks in each viewable dimension.
    npy_intp blockdims[NPY_MAXDIMS];
} dndview;


//Type describing a sub-section of a view block.
typedef struct
{
    //The rank of the MPI-process that owns this sub-block.
    int rank;
    //Start index (one per base-dimension).
    npy_intp start[NPY_MAXDIMS];
    //Number of elements (one per base-dimension).
    npy_intp nsteps[NPY_MAXDIMS];
    //Number of elements to next dimension (one per base-dimension).
    npy_intp stride[NPY_MAXDIMS];
    //The MPI communication offset (in bytes).
    npy_intp comm_offset;
    //Number of elements in this sub-view-block.
    npy_intp nelem;
    //This sub-view-block's root node.
    dndnode **rootnode;
    //Pointer to data. NULL if data needs to be fetched.
    char *data;
    //The rank of the MPI process that have received this svb.
    //A negative value means that nothing has been received.
    int comm_received_by;
} dndsvb;

//Type describing a view block.
typedef struct
{
    //The id of the view block.
    npy_intp uid;
    //All sub-view-blocks in this view block (Row-major).
    dndsvb *sub;
    //Number of sub-view-blocks.
    npy_intp nsub;
    //Number of sub-view-blocks in each dimension.
    npy_intp svbdims[NPY_MAXDIMS];
} dndvb;

//PyObject for the block iterator.
typedef struct
{
    PyObject_HEAD
    //The view that is iterated.
    dndview *view;
    //Current block coordinate.
    npy_intp curblock[NPY_MAXDIMS];
    //Slice of the blocks in the iterator.
    dndslice slice[NPY_MAXDIMS];
    //Strides for the Python array object.
    npy_intp strides[NPY_MAXDIMS];
    //Dimensions for the Python array object.
    npy_intp dims[NPY_MAXDIMS];
} dndblock_iter;

//Type describing the timing data.
typedef struct
{
    unsigned long long total;
    unsigned long long dag_svb_flush;
    unsigned long long dag_svb_rm;
    unsigned long long apply_ufunc;
    unsigned long long ufunc_comm;
    unsigned long long comm_init;
    unsigned long long arydata_free;
    unsigned long long reduce_1d;
    unsigned long long reduce_nd;
    unsigned long long reduce_nd_apply;
    unsigned long long zerofill;
    unsigned long long ufunc_svb;
    unsigned long long dag_svb_add;
    unsigned long long calc_vblock;
    unsigned long long arydata_malloc;
    unsigned long long msg2slaves;
    unsigned long long final_barrier;
    npy_intp mem_reused;
    npy_intp nconnect;
    npy_intp nconnect_max;
    npy_intp napply;
    npy_intp nflush;
} dndtime;

#ifdef __cplusplus
}
#endif

#endif /* !defined(DISTNUMPY_TYPES_H) */
