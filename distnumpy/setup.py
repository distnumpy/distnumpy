"""
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
"""

from distutils.core import setup, Extension
from os.path import join

def build(build_path):
    print "build_path: ", build_path
    setup(name='DistNumPy',
          version='1.0',
          ext_modules=[Extension(name='distnumpymodule',
                                 sources=[join('distnumpy','src','distnumpymodule.c')],
                                 include_dirs=[join('distnumpy','include'),
                                               join('distnumpy','private'),
                                               join('numpy','core','include','numpy'),
                                               join(build_path, 'numpy','core','include','numpy')],
                                 extra_compile_args=[],
                                 extra_link_args=[],
                                 depends=[join('distnumpy','src','helpers.c'),
                                          join('distnumpy','src','helpers.h'),
                                          join('distnumpy','src','array_database.c'),
                                          join('distnumpy','src','array_database.h'),
                                          join('distnumpy','src','memory.c'),
                                          join('distnumpy','src','memory.h')]
                                 )],
          )

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
