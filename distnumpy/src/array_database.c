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

//Array-bases belonging to local MPI process
static dndarray dndarrays[DNPY_MAX_NARRAYS];
static npy_intp dndarrays_uid[DNPY_MAX_NARRAYS];

//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];

/*===================================================================
 *
 * Put, get & remove array from the local array database.
 */
dndarray *get_dndarray(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndarrays_uid[i] == uid)
                return &dndarrays[i];
    fprintf(stderr, "get_dndarray, uid %ld does not exist\n", (long) uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
dndarray *put_dndarray(dndarray *ary)
{
    npy_intp i;

    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndarrays_uid[i] == 0)
        {
            memcpy(&dndarrays[i], ary, sizeof(dndarray));
            dndarrays_uid[i] = ary->uid;
            return &dndarrays[i];
        }
    fprintf(stderr, "put_dndarray, MAX_NARRAYS is exceeded\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
void rm_dndarray(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndarrays_uid[i] == uid)
            {
                dndarray *ary = &dndarrays[i];
                //Cleanup base.
                dndarrays_uid[i] = 0;
                //Remove the array from to the linked list.
                if(ary->next != NULL)
                    ary->next->prev = ary->prev;
                if(ary->prev != NULL)
                    ary->prev->next = ary->next;
                else
                    rootarray = ary->next;

                MPI_Type_free(&ary->mpi_dtype);
                if(ary->data != NULL)
                {
                    mem_pool_put((dndmem*) (ary->data-sizeof(dndmem)));
                }
                free(ary->rootnodes);
                --ndndarrays;
                assert(ndndarrays >= 0);
                return;
            }
    fprintf(stderr, "rm_dndarray, uid %ld does not exist\n", (long)uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}/* Put, get & rm dndarray */

/*===================================================================
 *
 * Put, get & remove views from the local array database.
 */
dndview *get_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
                return &dndviews[i];
    fprintf(stderr, "get_dndview, uid %ld does not exist\n", (long) uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
dndview *put_dndview(dndview *view)
{
    npy_intp i;

    for(i=0; i < DNPY_MAX_NARRAYS; i++)
        if(dndviews_uid[i] == 0)
        {
            memcpy(&dndviews[i], view, sizeof(dndview));
            dndviews_uid[i] = view->uid;
            return &dndviews[i];
        }
    fprintf(stderr, "put_dndview, MAX_NARRAYS is exceeded\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
    return NULL;
}
//NB: rm_dndview will also free memory allocted for the dndarray
//if it is the last reference to the dndarray.
void rm_dndview(npy_intp uid)
{
    npy_intp i;
    if(uid)
        for(i=0; i < DNPY_MAX_NARRAYS; i++)
            if(dndviews_uid[i] == uid)
            {
                dndview *view = &dndviews[i];
                //Cleanup base.
                dndviews_uid[i] = 0;
                if(view->base->ndims == 0)//Dummy Scalar.
                {
                    //Remove the array from to the linked list.
                    if(view->base->next != NULL)
                        view->base->next->prev = view->base->prev;
                    if(view->base->prev != NULL)
                        view->base->prev->next = view->base->next;
                    else
                        rootarray = view->base->next;
                    free(view->base->rootnodes);
                    free(view->base->data);
                    free(view->base);
                }
                else if(--view->base->refcount == 0)
                {
                    //Remove the array.
                    rm_dndarray(view->base->uid);
                }
                return;
            }
    fprintf(stderr, "rm_dndview, uid %ld does not exist\n", (long)uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}/* Put, get & rm dndview */

