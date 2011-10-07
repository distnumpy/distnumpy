#ifndef DISTNUMPY_H
#define DISTNUMPY_H
#ifdef __cplusplus
extern "C" {
#endif

/* Only import when compiling distnumpymodule.c */
#ifdef DISTNUMPY_MODULE
#include "ndarraytypes.h"
#include "arrayobject.h"
#endif

/* Public header file for disnumpy */

//Flag indicating that it is a distributed array
#define DNPY_DIST 0x2000
//Flag indicating that it is a distributed array on one node
#define DNPY_DIST_ONENODE 0x4000

//Easy attribute retrievals.
#define PyDistArray_ISDIST(m) PyArray_CHKFLAGS(m,DNPY_DIST)
#define PyDistArray_ISDIST_ONENODE(m) PyArray_CHKFLAGS(m,DNPY_DIST_ONENODE)
#define PyDistArray_DNDUID(obj) (((PyArrayObject *)(obj))->dnduid)


/* C API functions */

#define PyDistArray_NewBaseArray_NUM 0
#define PyDistArray_NewBaseArray_RETURN npy_intp
#define PyDistArray_NewBaseArray_PROTO (PyArrayObject *pyary, npy_intp onerank)

/* Total number of C API pointers */
#define DistNumPy_API_pointers 1


#ifdef DISTNUMPY_MODULE
/* This section is used when compiling distnumpymodule.c */

static PyDistArray_NewBaseArray_RETURN PyDistArray_NewBaseArray PyDistArray_NewBaseArray_PROTO;

#else
/* This section is used in modules that use distnumpy's API */

static void **DistNumPy_API;

#define PyDistArray_NewBaseArray \
 (*(PyDistArray_NewBaseArray_RETURN (*)PyDistArray_NewBaseArray_PROTO) DistNumPy_API[PyDistArray_NewBaseArray_NUM])


/* Return -1 and set exception on error, 0 on success. */
static int
import_distnumpy(void)
{
    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule("distnumpy");
    if (module == NULL)
        return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");
    if (c_api_object == NULL) {
        Py_DECREF(module);
        return -1;
    }
    if (PyCObject_Check(c_api_object))
        DistNumPy_API = (void **)PyCObject_AsVoidPtr(c_api_object);

    Py_DECREF(c_api_object);
    Py_DECREF(module);
    return 0;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(DISTNUMPY_H) */
