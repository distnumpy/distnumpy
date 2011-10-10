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

#include <Python.h>
#define DISTNUMPY_MODULE
#include "distnumpy.h"
#include "arrayobject.h"
#include "distnumpy_priv.h"
#include <mpi.h>

/*
 * ===================================================================
 * Public
 * Initialization of distnumpy.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_Init(void)
{
    int provided;
    int flag;
    int i;

    //Make sure we only initialize once.
    MPI_Initialized(&flag);
    if (flag)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "DistNumPy error - multiple "
                        "initialization attempts.");
        return -1;
    }

    //We make use of MPI_Init_thread even though we only ask for
    //a MPI_THREAD_SINGLE level thread-safety because MPICH2 only
    //supports MPICH_ASYNC_PROGRESS when MPI_Init_thread is used.
    //Note that when MPICH_ASYNC_PROGRESS is defined the thread-safety
    //level will automatically be set to MPI_THREAD_MULTIPLE.
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    //Allocate buffers.
    workbuf = malloc(DNPY_WORK_BUFFER_MAXSIZE);
    workbuf_nextfree = workbuf;
    assert(workbuf != NULL);

    //We subtract one MB to avoid segmentation faults when the workbuf
    //is used before the call to WORKBUF_INC()
    workbuf_max = ((char*)workbuf) + DNPY_WORK_BUFFER_MAXSIZE - 1048576;

    //Lets make sure that the memory is aligned.
    WORKBUF_INC(1);

    //Allocate cart_dim_sizes and cart_dim_strides.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        cart_dim_sizes[i] = malloc((i+1)*sizeof(int));
        cart_dim_strides[i] = malloc((i+1)*sizeof(int));
    }

    //Set blocksize
    if(myrank == 0)
    {
        char *env;
        //Check for user-defined block size.
        env = getenv("DNPY_BLOCKSIZE");
        if(env == NULL)
            blocksize = DNPY_BLOCKSIZE;
        else
            blocksize = atoi(env);

        if(blocksize <= 0)
        {
            fprintf(stderr, "User-defined blocksize must be greater "
                            "than zero\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    MPI_Bcast(&blocksize, sizeof(npy_intp), MPI_BYTE, 0, MPI_COMM_WORLD);

    return 0;
} /* PyDistArray_Init */

/*
 * ===================================================================
 * Public
 * De-initialization of distnumpy.
 */
static void
PyDistArray_Exit(void)
{
    int i;

/*
#ifndef DNPY_SPMD
    if(myrank == 0)
    {
        //Shutdown slaves
        msg[0] = DNPY_SHUTDOWN;
        msg[1] = DNPY_MSG_END;
        msg2slaves(msg, 2 * sizeof(npy_intp));
        #ifdef DISTNUMPY_DEBUG
            printf("Rank 0 received msg: SHUTDOWN\n");
        #endif
    }
#endif
    //Make sure that the sub-view-block DAG is flushed.
    dag_svb_flush(1);
*/
    //Free buffers.
    free(workbuf);
    //Free Cartesian Information.
    for(i=0; i < NPY_MAXDIMS; i++)
    {
        free(cart_dim_strides[i]);
        free(cart_dim_sizes[i]);
    }
    int nleaks = 0;
    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] != 0)
            nleaks++;

    if(nleaks > 0)
        printf("DistNumPy - Warning %d distributed arrays didn't get "
               "deallocated.\n", nleaks);

    //Free the memory pool.
//    free_mem_pool(mem_pool);

    MPI_Finalize();
} /* PyDistArray_Exit */


static PyObject *
PySpam_System(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return Py_BuildValue("i", sts);
}


static PyMethodDef DistNumPyMethods[] = {
    {"system",  PySpam_System, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static npy_intp PyDistArray_NewBaseArray(PyArrayObject *pyary, npy_intp onerank)
{

    return 42;
}



PyMODINIT_FUNC
initdistnumpy(void)
{
    PyObject *m;
    static void *DistNumPy_API[DistNumPy_API_pointers];
    PyObject *c_api_object;

    m = Py_InitModule("distnumpy", DistNumPyMethods);
    if (m == NULL)
        return;

    /* Initialize the C API pointer array */
    DistNumPy_API[PyDistArray_Init_NUM] = (void *)PyDistArray_Init;
    DistNumPy_API[PyDistArray_Exit_NUM] = (void *)PyDistArray_Exit;
    DistNumPy_API[PyDistArray_NewBaseArray_NUM] = (void *)PyDistArray_NewBaseArray;


    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)DistNumPy_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();
}
