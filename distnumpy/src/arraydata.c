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
 * The Array Data Protection handles the event when NumPy or external
 * libraries access the array data directly. Since DistNumPy distribute
 * this data, the result of such direct array data access is a
 * segmentation fault. The handle this access we allocate protected
 * memory and makes the local array data pointer points to this memory.
 */

#include <errno.h>
#include <sys/mman.h>

/*
 *===================================================================
 * Initialization of the Array Data Protection.
 */
int arydat_init(void)
{

    return 0;
} /* arydat_init */

/*
 *===================================================================
 * Finalization of the Array Data Protection.
 */
int arydat_finalize(void)
{

    return 0;
} /* arydat_finalize */

/*
 *===================================================================
 * Allocate data memory for the 'ary'.
 * Return -1 and set exception on error, 0 on success.
 */
int arydat_malloc(PyArrayObject *ary)
{
    dndview *view = PyDistArray_ARRAY(ary);
    npy_int size = view->base->nelem * view->base->elsize;

    //Allocate page-size alligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *addr = mmap(0, size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(addr == MAP_FAILED)
    {
        int errsv = errno;//mmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mmap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        return -1;
    }

    //Update the ary data pointer.
    PyArray_BYTES(ary) = addr;

    return 0;
}/* arydat_malloc */

/*
 *===================================================================
 * Free protected memory.
 */
int arydat_free(PyArrayObject *ary)
{
    void *addr = PyArray_DATA(ary);
    dndview *view = PyDistArray_ARRAY(ary);
    npy_int size = view->base->nelem * view->base->elsize;

    if(munmap(addr, size) == -1)
    {
        int errsv = errno;//munmmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mummap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        return -1;
    }
    return 0;
}
