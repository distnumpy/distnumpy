#include <Python.h>
#define DISTNUMPY_MODULE
#include "distnumpy.h"
#include "arrayobject.h"


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
    DistNumPy_API[PyDistArray_NewBaseArray_NUM] = (void *)PyDistArray_NewBaseArray;


    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)DistNumPy_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();

}
