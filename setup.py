#!/usr/bin/env python

import os
from distutils.core import setup
from distutils.extension import Extension

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "Label the rows, columns, any dimension, of your NumPy arrays."
fid = file('README.rst', 'r')
long_description = fid.read()
fid.close()
idx = max(0, long_description.find("The main class of the la package"))
long_description = long_description[idx:]

# Get la version
ver_file = os.path.join('la', 'version.py')
fid = file(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')

NAME                = 'la'
MAINTAINER          = "Keith Goodman"
MAINTAINER_EMAIL    = "labeled-array@googlegroups.com"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://berkeleyanalytics.com/la"
DOWNLOAD_URL        = "http://pypi.python.org/pypi/la"
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Archipel Asset Management AB"
AUTHOR_EMAIL        = "kwgoodman@gmail.com"
PLATFORMS           = "OS Independent"
MAJOR               = VER[0]
MINOR               = VER[1]
MICRO               = VER[2]
ISRELEASED          = False
VERSION             = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES            = ["la", "la/tests", "la/util", "la/util/tests",
                       "la/external", "la/farray", "la/farray/tests",
                       "la/data"]
PACKAGE_DATA        = {'la': ['LICENSE']}
REQUIRES            = ["numpy", "Bottleneck"]


try:
    # Try to compile clistmap.c
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          packages=PACKAGES,
          package_data=PACKAGE_DATA,
          requires=REQUIRES,
          ext_modules = [Extension("la.cflabel", ["la/src/cflabel.c"])]
         )
except SystemExit:
    # Probably clistmap.c failed to compile, so use slower python version
    msg = '\nLooks like cflabel.c failed to compile, so the slower python '
    msg += 'version will be used instead.\n'  
    print msg        
    setup(name=NAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          url=URL,
          download_url=DOWNLOAD_URL,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          platforms=PLATFORMS,
          version=VERSION,
          packages=PACKAGES,
          package_data=PACKAGE_DATA,
          requires=REQUIRES,
         )     
