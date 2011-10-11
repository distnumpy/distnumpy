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
//Tells numpy that this file initiate the module.
#define PY_ARRAY_UNIQUE_SYMBOL DISTNUMPY_ARRAY_API
#include "arrayobject.h"
#include "distnumpy_priv.h"
#include <mpi.h>

//We include all .h and .c files.
//NumPy distutil complains when having multiple module files.
#include "helpers.h"
#include "array_database.h"
#include "memory.h"
#include "arrayobject.h"
#include "helpers.c"
#include "array_database.c"
#include "memory.c"
#include "arrayobject.c"

/*
 * ===================================================================
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
 * De-initialization of distnumpy.
 */
static void
PyDistArray_Exit(void)
{
    int i;


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
    //dag_svb_flush(1);

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

    //De-allocate the memory pool.
    mem_pool_finalize();

    MPI_Finalize();
} /* PyDistArray_Exit */


/*
 * ===================================================================
 * From this point on the master will continue with the pyton code
 * and the slaves will stay in C.
 * If returning False the Python must call sys.exit(0) immediately.
 */
static PyObject *
PyDistArray_MasterSlaveSplit(PyObject *self, PyObject *args)
{
    //Initiate timers to zero.
    memset(&dndt, 0, sizeof(dndtime));
    DNDTIME(totaldelta)

#ifdef DNPY_SPMD
    return Py_True;
#else

    if(myrank == 0)
        return Py_True;

    int shutdown = 0;
    while(shutdown == 0)//Work loop
    {
        char *t1, *t2, *t3;
        npy_intp d1, d2, d3, d4, d5;
        long l1;
        dndview *ary, *ary2, *ary3;
        //Receive message from master.
        MPI_Bcast(msg, DNPY_MAX_MSG_SIZE, MPI_BYTE, 0, MPI_COMM_WORLD);
        char *msg_data = (char *) &msg[1];
        #ifdef DISTNUMPY_DEBUG
            printf("Rank %d received msg: ", myrank);
        #endif
        switch(msg[0])
        {
            case DNPY_INIT_PGRID:
                //do_INIT_PGRID((int*)msg_data);
                break;
            case DNPY_INIT_BLOCKSIZE:
                //blocksize = *((npy_intp*)msg_data);
                break;
            case DNPY_CREATE_ARRAY:
                t1 = msg_data + sizeof(dndarray);
                handle_NewBaseArray((dndarray*) msg_data, (dndview*) t1);
                break;
            case DNPY_DESTROY_ARRAY:
                //do_DESTROY_ARRAY(*((npy_intp*)msg_data));
                break;
            case DNPY_CREATE_VIEW:
                d1 = *((npy_intp*)msg_data);
                t1 = msg_data+sizeof(npy_intp);
                //do_CREATE_VIEW(d1, (dndview*) t1);
                break;
            case DNPY_SHUTDOWN:
                #ifdef DISTNUMPY_DEBUG
                    printf("SHUTDOWN\n");
                #endif
                shutdown = 1;
                break;
            case DNPY_EVALFLUSH:
                #ifdef DISTNUMPY_DEBUG
                    printf("EVALFLUSH\n");
                #endif
                //dag_svb_flush(1);
                break;
            case DNPY_PUT_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                t2 = t1+ary->base->elsize;
                //do_PUTGET_ITEM(1, ary, t1, (npy_intp*) t2);
                break;
            case DNPY_GET_ITEM:
                ary = get_dndview(*((npy_intp*)msg_data));
                t1 = msg_data+sizeof(npy_intp);
                //do_PUTGET_ITEM(0, ary, NULL, (npy_intp*) t1);
                break;
            case DNPY_COPY_INTO:
                d1 = *((npy_intp*)msg_data);
                d2 = *(((npy_intp*)msg_data)+1);
                //do_COPY_INTO(d1,d2);
                break;
            case DNPY_UFUNC:
                d1 = *((npy_intp*)msg_data);
                d2 = *(((npy_intp*)msg_data)+1);
                d3 = *(((npy_intp*)msg_data)+2);
                d4 = *(((npy_intp*)msg_data)+3);
                d5 = *(((npy_intp*)msg_data)+4);
                t1 = msg_data+sizeof(npy_intp)*5;
                t2 = t1+d5;
                t3 = t2+d1*sizeof(npy_intp);
                //do_UFUNC((npy_intp *)t2,d1,d2,d3,d4,d5,t1,t3);
                break;
            case DNPY_UFUNC_REDUCE:
                d1 = *((npy_intp*)msg_data);
                d2 = *(((npy_intp*)msg_data)+1);
                d3 = *(((npy_intp*)msg_data)+2);
                d4 = *(((npy_intp*)msg_data)+3);
                d5 = *(((npy_intp*)msg_data)+4);
                t1 = msg_data+sizeof(npy_intp)*5;
                //do_UFUNC_REDUCE(d1, d2, d3, d4, NULL, d5, t1);
                break;
            case DNPY_ZEROFILL:
                //do_ZEROFILL(get_dndview(*((npy_intp*)msg_data)));
                break;
            case DNPY_DATAFILL:
                d1 = ((npy_intp*)msg_data)[0]; // view uid
                l1 = (long) ((npy_intp*)msg_data)[1]; // filepos
                t1 = msg_data+sizeof(npy_intp)+sizeof(long); // get filename
                //do_FILEIO(get_dndview(d1), t1, l1, DNPY_DATAFILL);
                break;
            case DNPY_DATADUMP:
                d1 = ((npy_intp*)msg_data)[0]; // view uid
                l1 = (long) ((npy_intp*)msg_data)[1]; // filepos
                t1 = msg_data+sizeof(npy_intp)+sizeof(long); // get filename
                //do_FILEIO(get_dndview(d1), t1, l1, DNPY_DATADUMP);
                break;
            case DNPY_DIAGONAL:
                ary  = get_dndview(((npy_intp*)msg_data)[0]);
                ary2 = get_dndview(((npy_intp*)msg_data)[1]);
                d1 = ((npy_intp*)msg_data)[2];
                d2 = ((npy_intp*)msg_data)[3];
                d3 = ((npy_intp*)msg_data)[4];
                //do_DIAGONAL(ary, ary2, d1, d2, d3);
                break;
            case DNPY_MATMUL:
                ary  = get_dndview(((npy_intp*)msg_data)[0]);
                ary2 = get_dndview(((npy_intp*)msg_data)[1]);
                ary3 = get_dndview(((npy_intp*)msg_data)[2]);
                //do_MATMUL(ary, ary2, ary3);
                break;
            case DNPY_TIME_RESET:
                //do_TIME_RESET();
                break;
            case DNPY_TIME_GETDICT:
                //do_TIME_GETDICT();
                break;
            default:
                fprintf(stderr, "Unknown msg: %ld\n", (long)msg[0]);
                MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    return Py_False;
#endif
} /* PyDistArray_MasterSlaveSplit */


static PyMethodDef DistNumPyMethods[] = {
    {"MasterSlaveSplit", PyDistArray_MasterSlaveSplit, METH_VARARGS,
     "From this point on the master will continue with the pyton code"\
     " and the slaves will stay in C"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

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
    DistNumPy_API[PyDistArray_MasterSlaveSplit_NUM] = (void *)PyDistArray_MasterSlaveSplit;
    DistNumPy_API[PyDistArray_NewBaseArray_NUM] = (void *)PyDistArray_NewBaseArray;
    DistNumPy_API[PyDistArray_GetItem_NUM] = (void *)PyDistArray_GetItem;
    DistNumPy_API[PyDistArray_PutItem_NUM] = (void *)PyDistArray_PutItem;


    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)DistNumPy_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();
}
