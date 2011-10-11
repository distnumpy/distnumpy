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
 * We use a memory pool to reduce the memory allocation overhead.
 */

#ifndef MEMORY_H
#define MEMORY_H
#ifdef __cplusplus
extern "C" {
#endif


//Type describing a memory allocation.
typedef struct dndmem_struct dndmem;
struct dndmem_struct
{
    //Size of allocated memory.
    npy_intp size;
    //Pointer to the next free memory allocation.
    dndmem *next;
};

/*===================================================================
 *
 * Put memory allocation into the memory pool.
 */
void mem_pool_put(dndmem *mem);

/*===================================================================
 *
 * Makes sure that the array's memory has been allocated.
 */
void delayed_array_allocation(dndarray *ary);

/*===================================================================
 *
 * De-allocate the memory pool.
 */
void mem_pool_finalize(void);


#ifdef __cplusplus
}
#endif

#endif /* !defined(MEMORY_H) */
