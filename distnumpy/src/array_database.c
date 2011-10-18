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

//Array-views belonging to local MPI process
static dndview dndviews[DNPY_MAX_NARRAYS];
static npy_intp dndviews_uid[DNPY_MAX_NARRAYS];

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
    fprintf(stderr, "put_dndarray, MAX_NARRAYS is exceeded\n");
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
                    //Remove the array from to the linked list.
                    if(view->base->next != NULL)
                        view->base->next->prev = view->base->prev;
                    if(view->base->prev != NULL)
                        view->base->prev->next = view->base->next;
                    else
                        rootarray = view->base->next;

                    MPI_Type_free(&view->base->mpi_dtype);
                    if(view->base->data != NULL)
                    {
                        mem_pool_put((dndmem*) (view->base->data -
                                                sizeof(dndmem)));
                    }
                    free(view->base->rootnodes);
                    free(view->base);
                    --ndndarrays;
                    assert(ndndarrays >= 0);
                }
                return;
            }
    fprintf(stderr, "rm_dndarray, uid %ld does not exist\n", (long)uid);
    MPI_Abort(MPI_COMM_WORLD, -1);
    return;
}/* Put, get & rm dndview */

