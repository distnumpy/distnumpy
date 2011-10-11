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

#ifndef DISTNUMPY_API_H
#define DISTNUMPY_API_H
#ifdef __cplusplus
extern "C" {
#endif

/* C API functions */

#define PyDistArray_Init_NUM 0
#define PyDistArray_Init_RETURN int
#define PyDistArray_Init_PROTO (void)

#define PyDistArray_Exit_NUM 1
#define PyDistArray_Exit_RETURN void
#define PyDistArray_Exit_PROTO (void)

#define PyDistArray_MasterSlaveSplit_NUM 2
#define PyDistArray_MasterSlaveSplit_RETURN PyObject *
#define PyDistArray_MasterSlaveSplit_PROTO (PyObject *self, PyObject *args)

#define PyDistArray_NewBaseArray_NUM 3
#define PyDistArray_NewBaseArray_RETURN npy_intp
#define PyDistArray_NewBaseArray_PROTO (PyArrayObject *pyary, npy_intp onerank)

/* Total number of C API pointers */
#define DistNumPy_API_pointers 4


#ifdef DISTNUMPY_MODULE
/* This section is used when compiling distnumpymodule.c */

static PyDistArray_Init_RETURN PyDistArray_Init PyDistArray_Init_PROTO;
static PyDistArray_Exit_RETURN PyDistArray_Exit PyDistArray_Exit_PROTO;
static PyDistArray_MasterSlaveSplit_RETURN PyDistArray_MasterSlaveSplit PyDistArray_MasterSlaveSplit_PROTO;
static PyDistArray_NewBaseArray_RETURN PyDistArray_NewBaseArray PyDistArray_NewBaseArray_PROTO;

#else
/* This section is used in modules that use distnumpy's API */

static void **DistNumPy_API;

#define PyDistArray_Init \
 (*(PyDistArray_Init_RETURN (*)PyDistArray_Init_PROTO) DistNumPy_API[PyDistArray_Init_NUM])

#define PyDistArray_Exit \
 (*(PyDistArray_Exit_RETURN (*)PyDistArray_Exit_PROTO) DistNumPy_API[PyDistArray_Exit_NUM])

#define PyDistArray_MasterSlaveSplit \
 (*(PyDistArray_MasterSlaveSplit_RETURN (*)PyDistArray_MasterSlaveSplit_PROTO) DistNumPy_API[PyDistArray_MasterSlaveSplit_NUM])

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

#endif /* !defined(DISTNUMPY_API_H) */
