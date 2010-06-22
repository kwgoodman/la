"""
Use to convert cflabel.pyx to a C file.

This setup.py is NOT used to install the la package. The la setup.py file is
la/setup.py

The C files are distributed with larry, so this file is only useful if you
modify cflabel.pyx.

To convert from cython to C:

$ cd la/src
$ python setup.py build_ext --inplace

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("cflabel", ["cflabel.pyx"])]

setup(
  name = 'cflabel',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

