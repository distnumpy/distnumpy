#ifndef DISTNUMPY_PRIV_H
#define DISTNUMPY_PRIV_H
#include "mpi.h"
#include <sys/time.h>
#include "distnumpy_types.h"
#include "numpy/ufuncobject.h"

#ifdef __cplusplus
extern "C" {
#endif

//#define DISTNUMPY_DEBUG
//#define DNPY_STATISTICS
//#define DNDY_TIME
//#define DNPY_SPMD

//Minimum jobsize for an OpenMP thread. >blocksize means no OpenMP.
#define DNPY_MIN_THREAD_JOBSIZE 10

//Maximum message size (in bytes)
#define DNPY_MAX_MSG_SIZE 1024*4

//Maximum number of memory allocations in the memory pool.
#define DNPY_MAX_MEM_POOL 10

//Maximum number of view block operations in the sub-view-block DAG.
#define DNPY_MAX_VB_IN_SVB_DAG 100

//Disable Lazy Evaluation by definding this macro.
#undef DNPY_NO_LAZY_EVAL

//Maximum number of allocated arrays
#define DNPY_MAX_NARRAYS 1024

//Maximum number of operation merged together.
#define DNPY_MAX_OP_MERGES 10

//Default blocksize
#define DNPY_BLOCKSIZE 2

//Maximum number of nodes in the ready queue.
#define DNPY_RDY_QUEUE_MAXSIZE 1024*10

//Maximum MPI tag.
#define DNPY_MAX_MPI_TAG 1048576

//The maximum size of the work buffer in bytes (should be power of 2).
#define DNPY_WORK_BUFFER_MAXSIZE 536870912 //½GB

//The work buffer memory alignment.
#define DNPY_WORK_BUFFER_MEM_ALIGNMENT 32

//Operation types
enum opt {DNPY_MSG_END, DNPY_CREATE_ARRAY, DNPY_DESTROY_ARRAY,
          DNPY_CREATE_VIEW, DNPY_SHUTDOWN, DNPY_PUT_ITEM, DNPY_GET_ITEM,
          DNPY_UFUNC, DNPY_UFUNC_REDUCE, DNPY_ZEROFILL, DNPY_DATAFILL,
          DNPY_DATADUMP, DNPY_DIAGONAL, DNPY_MATMUL,
          DNPY_RECV, DNPY_SEND, DNPY_BUF_RECV, DNPY_BUF_SEND, DNPY_APPLY,
          DNPY_EVALFLUSH, DNPY_READ, DNPY_WRITE, DNPY_COMM, DNPY_NONCOMM,
          DNPY_REDUCE_SEND, DNPY_REDUCE_RECV, DNPY_INIT_BLOCKSIZE,
          DNPY_TIME_RESET, DNPY_TIME_GETDICT, DNPY_INIT_PGRID,
          DNPY_COPY_INTO, DNPY_UNDIST};


//Macro that increases the work buffer pointer.
#define WORKBUF_INC(bytes_taken)                                       \
{                                                                      \
    workbuf_nextfree += bytes_taken;                                   \
    workbuf_nextfree += DNPY_WORK_BUFFER_MEM_ALIGNMENT -               \
                        (((npy_intp)workbuf_nextfree)                  \
                        % DNPY_WORK_BUFFER_MEM_ALIGNMENT);             \
    if(workbuf_nextfree >= workbuf_max)                                \
    {                                                                  \
        fprintf(stderr, "Work buffer overflow - increase the maximum " \
                "work buffer size or decrease the maximum DAG size. "  \
                "The current values are %dMB and %d nodes,"            \
                "respectively.\n", DNPY_WORK_BUFFER_MAXSIZE / 1048576, \
                DNPY_MAX_VB_IN_SVB_DAG);                               \
        MPI_Abort(MPI_COMM_WORLD, -1);                                 \
    }                                                                  \
    assert(((npy_intp) workbuf_nextfree) %                             \
                       DNPY_WORK_BUFFER_MEM_ALIGNMENT == 0);           \
}


//Variables for statistics.
#ifdef DNPY_STATISTICS
    static int node_uid_count = 0;
    static int op_uid_count = 0;
#endif


//The Super-type of a operation.
//refcount         - number of dependency nodes in the svb DAG.
//op               - the operation, e.g. DNPY_RECV and DNPY_UFUNC.
//optype           - the operation type, e.g. DNPY_COMM/_NONCOMM.
//narys & views    - list of array views involved.
//svbs             - list of sub-view-blocks involved (one per array),
//                   NULL when whole arrays are involved.
//accesstype       - access type e.g. DNPY_READ (one per array)
//uid              - unique identification - only used for statistics.
#define DNDOP_HEAD_BASE                     \
    npy_intp refcount;                      \
    char op;                                \
    char optype;                            \
    char narys;                             \
    dndview *views[NPY_MAXARGS];            \
    dndsvb *svbs[NPY_MAXARGS];              \
    char accesstypes[NPY_MAXARGS];
#ifdef DNPY_STATISTICS
    #define DNDOP_HEAD DNDOP_HEAD_BASE npy_intp uid;
#else
    #define DNDOP_HEAD DNDOP_HEAD_BASE
#endif
typedef struct dndop_struct dndop;
struct dndop_struct {DNDOP_HEAD};

//Type describing a communication DAG node.
typedef struct
{
    DNDOP_HEAD
    //The MPI tag used for the communication.
    npy_intp mpi_tag;
    //The MPI rank of the process that is the remote communication peer.
    int remote_rank;
} dndop_comm;

//Type describing an apply-sub-view-block, which is a subsection of a
//sub-view-block that is used in apply.
typedef struct
{
    npy_intp dims[NPY_MAXDIMS];
    npy_intp stride[NPY_MAXDIMS];
    npy_intp offset;
} dndasvb;

//Type describing a universal function DAG node.
typedef struct
{
    DNDOP_HEAD
    //List of apply-sub-view-block.
    dndasvb asvb[NPY_MAXARGS];
    //Number of output array views.
    char nout;
    //The operation described as a function, a data and a Python pointer.
    PyUFuncGenericFunction func;
    void *funcdata;
    PyObject *PyOp;
} dndop_ufunc;

//Type describing a DAG node.
struct dndnode_struct
{
    //The operation associated with this dependency.
    dndop *op;
    //The index to use when accessing op->views[] and op->svbs[].
    int op_ary_idx;
    //Next node in the dependency list.
    dndnode *next;
    //Unique identification used for statistics.
    #ifdef DNPY_STATISTICS
        npy_intp uid;
    #endif
};

//MPI process variables.
static int myrank, worldsize;
static npy_intp blocksize;
#ifndef DNPY_SPMD
static npy_intp msg[DNPY_MAX_MSG_SIZE];
#endif
static npy_intp initmsg_not_handled=1;
//The work buffer and its next free slot.
static void *workbuf;
static void *workbuf_nextfree;
static void *workbuf_max;
//Unique identification counter
static npy_intp uid_count=0;
//Cartesian dimension information - one for every dimension-order.
static int *cart_dim_strides[NPY_MAXDIMS];
static int *cart_dim_sizes[NPY_MAXDIMS];
//Pointer to the python module who has the ufunc operators.
static PyObject *ufunc_module;
//The ready queue for operations and its current size.
static dndop *ready_queue[DNPY_RDY_QUEUE_MAXSIZE];
static npy_intp ready_queue_size=0;
//Unique MPI tag.
static int mpi_tag=0;
//Pointer to the PyUFunc_Reduce function in umath_ufunc_object.inc
typedef PyObject* (reduce_func_type)(PyUFuncObject *self,
                                     PyArrayObject *arr,
                                     PyArrayObject *out,
                                     int axis, int otype,
                                     void *threadlock);
static reduce_func_type *reduce_func = NULL;


//Variables for timing.
struct timeval tv;
struct timezone tz;
static dndtime dndt;
unsigned long long totaldelta;

#define DNDTIME(output)                                         \
    gettimeofday(&tv, &tz);                                     \
    output = (unsigned long long) tv.tv_usec +                  \
             (unsigned long long) tv.tv_sec * 1000000;
#define DNDTIME_SUM(in,sum)                                     \
    gettimeofday(&tv, &tz);                                     \
    sum += ((unsigned long long) tv.tv_usec +                   \
            (unsigned long long) tv.tv_sec * 1000000) - in;

#ifdef __cplusplus
}
#endif

#endif /* !defined(DISTNUMPY_PRIV_H) */
