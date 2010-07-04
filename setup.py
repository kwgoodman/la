#!/usr/bin/env python

import os
from distutils.core import setup
from distutils.extension import Extension

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "Label the rows, columns, any dimension, of your NumPy arrays."
long_description = """
Who's larry?
============

The main class of the la package is a labeled array, larry. A larry consists
of data and labels. The data is stored as a NumPy array and the labels as a
list of lists (one list per dimension).

Here's larry in schematic form:

::
    
                         date1    date2    date3
                'AAPL'   209.19   207.87   210.11
            y = 'IBM'    129.03   130.39   130.55
                'DELL'    14.82    15.11    14.94
                
The larry above is stored internally as a `Numpy <http://www.numpy.org>`_
array and a list of lists:

::
    
        y.label = [['AAPL', 'IBM', 'DELL'], [date1, date2, date3]]
        y.x = np.array([[209.19, 207.87, 210.11],
                        [129.03, 130.39, 130.55],
                        [ 14.82,  15.11,  14.94]])               
    
A larry can have any number of dimensions except zero. Here, for example, is
one way to create a one-dimensional larry:

::

    >>> import la
    >>> y = la.larry([1, 2, 3])
    
In the statement above the list is converted to a Numpy array and the labels
default to ``range(n)``, where *n* in this case is 3.
    
larry has built-in methods such as **movingsum, ranking, merge, shuffle,
zscore, demean, lag** as well as typical Numpy methods like **sum, max, std,
sign, clip**. NaNs are treated as missing data.
    
Alignment by label is automatic when you add (or subtract, multiply, divide)
two larrys.
    
You can archive larrys in HDF5 format using **save** and **load** or using a
dictionary-like interface:

::
    
    >>> io = la.IO('/tmp/dataset.hdf5')
    >>> io['y'] = y   # <--- save
    >>> z = io['y']   # <--- load
    >>> del io['y']   # <--- delete from archive
       
For the most part larry acts like a Numpy array. And, whenever you want,
you have direct access to the Numpy array that holds your data. For
example if you have a function, *myfunc*, that works on Numpy arrays and
doesn't change the shape or ordering of the array, then you can use it on a
larry, *y*, like this:

::
    
                           y.x = myfunc(y.x)
    
larry adds the convenience of labels, provides many built-in methods, and
let's you use your existing array functions.

===============   ========================================================
 code              http://github.com/kwgoodman/la
 docs              http://larry.sourceforge.net
 list 1            http://groups.google.ca/group/pystatsmodels
 list 2            http://groups.google.com/group/labeled-array
===============   ========================================================

"""

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
MAINTAINER_EMAIL    = "larry-discuss@lists.launchpad.net"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://larry.sourceforge.net"
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
PACKAGES            = ["la", "la/tests", "la/util", "la/util/tests"]
PACKAGE_DATA        = {'la': ['LICENSE']}
REQUIRES            = ["numpy"]


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
          zip_safe=False,
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
          zip_safe=False
         )     
