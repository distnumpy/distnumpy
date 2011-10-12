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
 * ===================================================================
 * Setting the process grid.
 * Accepts NULL as a default value request.
 * When called no distributed array must be allocated.
 */
static PyObject *
PyDistArray_ProcGridSet(PyArrayObject *self, PyObject *args)
{
    PyObject *pgrid = Py_None;
    int i,j;
    int tmpsizes[NPY_MAXDIMS*NPY_MAXDIMS];

    if(args != NULL)
        if (!PyArg_ParseTuple(args, "O", &pgrid))
            return NULL;

    if(!initmsg_not_handled && ndndarrays > 0)
    {
        PyErr_Format(PyExc_RuntimeError, "numpy.datalayout must be "
                "called when no distributed array are allocated "
                "(%ld arrays are currently allocated).", ndndarrays);
        return NULL;
    }

    //Check for user-defined process grid.
    //The syntax used is: ndims:dim:size;
    //E.g. DNPY_PROC_SIZE="2:2:4;3:3:2" which means that array
    //with two dimensions should, at its second dimension, have
    //a size of four etc.
    memset(tmpsizes, 0, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
    char *env = getenv("DNPY_PROC_GRID");
    if(env != NULL)
    {
        char *res = strtok(env, ";");
        while(res != NULL)
        {
            char *size_ptr;
            int dsize = 0;
            int dim = 0;
            int ndims = strtol(res, &size_ptr, 10);
            if(size_ptr != '\0')
                dim = strtol(size_ptr+1, &size_ptr, 10);
            if(size_ptr != '\0')
                dsize = strtol(size_ptr+1, NULL, 10);
            //Make sure the input is valid.
            if(dsize <= 0 || dim <= 0 || dim > ndims ||
               ndims <= 0 || ndims > NPY_MAXDIMS)
            {
                PyErr_Format(PyExc_ValueError, "DNPY_PROC_GRID, invalid"
                             " syntax or value at \"%s\"\n", res);
                return NULL;
            }
            tmpsizes[(ndims-1)*NPY_MAXDIMS+(dim-1)] = dsize;
            //Go to next token.
            res = strtok(NULL, ";");
        }
    }
    else if(pgrid != Py_None)
    {//The environment variable supersedes the function call.
        for(i=0; i<PySequence_Size(pgrid); i++)
        {
            PyObject *tuple = PySequence_ITEM(pgrid, i);
            if(!PySequence_Check(tuple) || PySequence_Size(tuple) != 3)
            {
                PyErr_Format(PyExc_ValueError, "The datalayout "
                    "must use the following layout: [(#dimensions, "
                    "dimension, size),...] where each tuple impose a "
                    "dimension size.");
                return NULL;
            }
            PyObject *item = PySequence_ITEM(tuple, 0);
            int ndims = PyInt_AsLong(item);
            Py_DECREF(item);
            item = PySequence_ITEM(tuple, 1);
            int dim = PyInt_AsLong(item);
            Py_DECREF(item);
            item = PySequence_ITEM(tuple, 2);
            int dsize = PyInt_AsLong(item);
            Py_DECREF(item);
            Py_DECREF(tuple);

            //Make sure the input is valid.
            if(dsize <= 0 || dim <= 0 || dim > ndims ||
               ndims <= 0 || ndims > NPY_MAXDIMS)
            {
                PyErr_Format(PyExc_ValueError, "invalid values");
                return NULL;
            }
            tmpsizes[(ndims-1)*NPY_MAXDIMS+(dim-1)] = dsize;
        }
    }

    //Find a balanced distributioin of processes per direction
    //based on the restrictions specified by the user.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        int ndims = i+1;
        int t[NPY_MAXDIMS];
        int d = 0;
        //Need to reverse the order to match MPI_Dims_create
        for(j=i; j>=0; j--)
            t[d++] = tmpsizes[i*NPY_MAXDIMS+j];

        MPI_Dims_create(worldsize, ndims, t);
        d = ndims;
        for(j=0; j<ndims; j++)
            tmpsizes[i*NPY_MAXDIMS+j] = t[--d];
    }

    #ifndef DNPY_SPMD
        //Tell slaves
        msg[0] = DNPY_INIT_PGRID;
        msg[1] = DNPY_INIT_PGRID;
        memcpy(&msg[1], tmpsizes, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
        *(((int*)&msg[1])+NPY_MAXDIMS*NPY_MAXDIMS) = DNPY_MSG_END;
        msg2slaves(msg, NPY_MAXDIMS*NPY_MAXDIMS*sizeof(int));
    #endif

    handle_ProcGridSet(tmpsizes);

    Py_RETURN_NONE;
}/* PyDistArray_ProcGridSet */


/*===================================================================
 *
 * Handler for PyDistArray_ProcGridSet.
 * Return -1 and set exception on error, 0 on success.
 */
int handle_ProcGridSet(int pgrid[NPY_MAXDIMS*NPY_MAXDIMS])
{
    int j,i;
    initmsg_not_handled = 0;
    assert(ndndarrays == 0);

    //Save the cart_dim_sizes and compute the cart_dim_strides.
    for(i=0; i<NPY_MAXDIMS; i++)
    {
        int ndims = i+1;
        for(j=0; j<ndims; j++)
            cart_dim_sizes[i][j] = pgrid[i*NPY_MAXDIMS+j];

        //Set cartesian information.
        memset(cart_dim_strides[i], 0, ndims*sizeof(int));

        //Compute strides for all dims. Using row-major like MPI.
        //A 2x2 process grid looks like:
        //    coord (0,0): rank 0.
        //    coord (0,1): rank 1.
        //    coord (1,0): rank 2.
        //    coord (1,1): rank 3.
        for(j=0; j<ndims; j++)
        {
            int stride = 1, s;
            for(s=j+1; s<ndims; s++)
                stride *= cart_dim_sizes[i][s];
            cart_dim_strides[i][j] = stride;
        }
    }

    return 0;
}/* handle_ProcGridSet */
