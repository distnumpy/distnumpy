from distutils.core import setup, Extension
from os.path import join

def build(build_path):
    print "build_path: ", build_path
    setup(name='DistNumPy',
          version='1.0',
          ext_modules=[Extension(name='distnumpymodule',
                                 sources=[join('distnumpy','src','distnumpymodule.c')],
                                 include_dirs=[join('distnumpy','include'),join('distnumpy','private'),join('numpy','core','include','numpy'),join(build_path, 'numpy','core','include','numpy')],
                                 extra_compile_args=[],
                                 extra_link_args=[]
                                 )],
          )

if __name__ == '__main__':
    print('This is the wrong setup.py file to run')
