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

#include <mpi.h>

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
    #ifdef DNPY_STATISTICS
        dndarray *next;
        dndarray *prev;
    #endif
    //When onerank is positiv this array is only distributed on that
    //MPI-process rank.
    npy_intp onerank;
};
/*
//dndslice constants.
#define PseudoIndex -1//Adds a extra 1-dim - 'A[1,newaxis]'
#define RubberIndex -2//A[1,2,...] (Not used in distnumpy.inc)
#define SingleIndex -3//Dim not visible - 'A[1]'
*/
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

/*
 *===================================================================
 * Create a new base array.
 * If 'one_node_dist_rank' is positive it specifies the rank of an
 * one-node-distribution.
 * Returns the uid of the view of the new base array.
 * Return -1 and set exception on error.
 */
static npy_intp
PyDistArray_NewBaseArray(PyArrayObject *ary, npy_intp one_node_dist_rank);

/*===================================================================
 *
 * Handler for PyDistArray_NewBaseArray.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_NewBaseArray(dndarray *ary, dndview *view);

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



#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAYOBJECT_H) */
