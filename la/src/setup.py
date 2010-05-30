"""
Use to convert cython files to C files.

This setup.py is NOT used to install the la package. The la setup.py file is
la/setup.py

The C files are distributed with larry, so this file is only useful if you
modify the cython (.pyx) files.

To convert from cython to C:

$ cd la/src
$ python setup.py build_ext --inplace

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("clistmap", ["clistmap.pyx"])]

setup(
  name = 'clistmap',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

