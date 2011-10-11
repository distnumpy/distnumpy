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
 * There is a local array database on each MPI-process.
 * The database consist of all array-views distributed.
 */


#ifndef ARRAY_DATABASE_H
#define ARRAY_DATABASE_H
#ifdef __cplusplus
extern "C" {
#endif


//Current number of dndviews allocated.
static npy_intp ndndarrays=0;

/*===================================================================
 *
 * Put, get & remove views from the local array database.
 */
dndview *get_dndview(npy_intp uid);
dndview *get_dndview(npy_intp uid);
dndview *put_dndview(dndview *view);
void rm_dndview(npy_intp uid);


#ifdef __cplusplus
}
#endif

#endif /* !defined(ARRAY_DATABASE_H) */
