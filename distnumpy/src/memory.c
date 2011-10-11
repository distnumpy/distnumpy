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

//Memory pool.
static dndmem *mem_pool = NULL;

/*===================================================================
 *
 * Frees the memory pool from a given memory allocation.
 * Private
 */
static void mem_pool_free(dndmem *mem)
{
    while(mem != NULL)
    {
        dndmem *next = mem->next;
        MPI_Free_mem(mem);
        mem = next;
    }
} /* mem_pool_free */


/*===================================================================
 *
 * Put memory allocation into the memory pool.
 */
void mem_pool_put(dndmem *mem)
{
    //Put the allocated memory in front of the pool.
    mem->next = mem_pool;
    mem_pool = mem;
} /* mem_pool_put */


/*===================================================================
 *
 * Makes sure that the array's memory has been allocated.
 */
void delayed_array_allocation(dndarray *ary)
{
    #ifdef DNDY_TIME
        unsigned long long tdelta;
        DNDTIME(tdelta);
    #endif
    npy_intp size = ary->localsize * ary->elsize;
    npy_intp count = 0;
    dndmem *free = NULL;

    if(ary->data != NULL)//Already allocated.
        return;

    //Check if there is some free memory in the memory pool.
    if(mem_pool != NULL)
    {
        dndmem *prev = mem_pool;
        dndmem *next = mem_pool->next;

        //Handle first iteration as a special case.
        if(mem_pool->size == size)
        {
            ary->data = ((char*)mem_pool) + sizeof(dndmem);
            mem_pool = mem_pool->next;//Remove from pool.
            #ifdef DNDY_TIME
                ++dndt.mem_reused;
            #endif
        }
        else//Handle the rest.
        {
            while(next != NULL)
            {
                assert(prev->next == next);
                if(next->size == size)
                {
                    ary->data = ((char*)next) + sizeof(dndmem);
                    prev->next = next->next;//Remove from pool.
                    #ifdef DNDY_TIME
                        ++dndt.mem_reused;
                    #endif
                    break;
                }
                if(++count == DNPY_MAX_MEM_POOL)
                {
                    //Will remove all mem after this one.
                    prev->next = NULL;
                    free = next;
                }

                //Go to next memory allocation.
                prev = next;
                next = next->next;
            }
        }
    }

    if(ary->data == NULL)//Need to allocate new memory.
    {
        dndmem *mem;
        if(MPI_Alloc_mem(size + sizeof(dndmem), MPI_INFO_NULL,
                         &mem) != MPI_SUCCESS)
        {
            fprintf(stderr, "Out of memory!\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        mem->size = size;
        ary->data = (char*) (mem + 1);
    }

    //Reduce the pool size to DNPY_MAX_MEM_POOL.
    if(free != NULL)
        mem_pool_free(free);

    #ifdef DNDY_TIME
        DNDTIME_SUM(tdelta, dndt.arydata_malloc)
    #endif
}/* delayed_array_allocation */

/*===================================================================
 *
 * De-allocate the memory pool.
 */
void mem_pool_finalize(void)
{
    mem_pool_free(mem_pool);
} /* finalize_mem_pool */
