"""
Use to convert cflabel.pyx to a C file.

This setup.py is NOT used to install the la package. The la setup.py file is
la/setup.py

The C files are distributed with larry, so this file is only useful if you
modify cflabel.pyx.

To convert from cython to C:

$ cd la/src
$ python setup.py build_ext --inplace

Or use top-level Makefile
"""

import os
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

name = 'cflabel'
mod_dir = os.path.dirname(__file__)
ext_modules = [Extension(name, [os.path.join(mod_dir, name + ".pyx")])]

setup(
  name = name,
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

# Move compiled code one directory up
extension = '.so'
if sys.platform.startswith('win'):
    extension = '.pyd'
os.rename(name + extension, os.path.join(mod_dir, "../" + name + extension))
