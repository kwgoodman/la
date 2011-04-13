Who's larry?
============

The main class of the la package is a labeled array, larry. A larry consists
of data and labels. The data is stored as a NumPy array and the labels as a
list of lists (one list per dimension).

Here's larry in schematic form::

                         date1    date2    date3
                'AAPL'   209.19   207.87   210.11
            y = 'IBM'    129.03   130.39   130.55
                'DELL'    14.82    15.11    14.94
                
The larry above is stored internally as a `Numpy <http://www.numpy.org>`_
array and a list of lists::

        y.label = [['AAPL', 'IBM', 'DELL'], [date1, date2, date3]]
        y.x = np.array([[209.19, 207.87, 210.11],
                        [129.03, 130.39, 130.55],
                        [ 14.82,  15.11,  14.94]])               
    
A larry can have any number of dimensions except zero. Here, for example, is
one way to create a one-dimensional larry::

    >>> import la
    >>> y = la.larry([1, 2, 3])
    
In the statement above the list is converted to a Numpy array and the labels
default to ``range(n)``, where *n* in this case is 3.
    
larry has built-in methods such as **ranking, merge, shuffle, move_sum,
zscore, demean, lag** as well as typical Numpy methods like **sum, max, std,
sign, clip**. NaNs are treated as missing data.
    
Alignment by label is automatic when you add (or subtract, multiply, divide)
two larrys. You can also specify the join method (inner, outer, left, right)
for binary operations on two larrys with unaligned labels.
    
You can archive larrys in HDF5 format using **save** and **load** or using a
dictionary-like interface::

    >>> io = la.IO('/tmp/dataset.hdf5')
    >>> io['y'] = y   # <--- save
    >>> z = io['y']   # <--- load
    >>> del io['y']   # <--- delete from archive
       
For the most part larry acts like a Numpy array. And, whenever you want,
you have direct access to the Numpy array that holds your data. For
example if you have a function, *myfunc*, that works on Numpy arrays and
doesn't change the shape or ordering of the array, then you can use it on a
larry, *y*, like this::

                           y.x = myfunc(y.x)
    
larry adds the convenience of labels, provides many built-in methods, and
let's you use your existing array functions.       

License
=======

The ``la`` package is distributed under a Simplified BSD license. Parts of
NumPy, Scipy, and numpydoc, which all have BSD licenses, are included in
``la``. Parts of matplotlib are also included. See the LICENSE file, which
is distributed with the ``la`` package, for details.

Install
=======

Requirements:

======================== ====================================================
la                       Python, NumPy 1.5.1, Bottleneck 0.4.3
Unit tests               nose
======================== ====================================================

Optional:

============================= ================================================
Archive larrys in HDF5        h5py, HDF 1.8
Compile for speed boost       gcc or MinGW
lar.ranking(norm='gaussian')  SciPy 0.8.0
============================= ================================================

You can download the `latest version of la <http://pypi.python.org/pypi/la>`_
and `Bottleneck <http://pypi.python.org/pypi/Bottleneck>`_ from the Python
Package Index.
            
**GNU/Linux, Mac OS X et al.**

To install ``la``::

    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where ``la`` is installed, for example inside
``/usr/local``::

    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local

**Windows**

In order to (optionally) compile the C code in the ``la`` package you need a
Windows version of the gcc compiler. MinGW (Minimalist GNU for Windows)
contains gcc and has been used to successfully compile ``la`` on Windows.

Install MinGW and add it to your system path. Then install ``la`` with the
commands::

    python setup.py build --compiler=mingw32
    python setup.py install

**Post install**

After you have installed ``la``, run the suite of unit tests::

    >>> import la
    >>> la.test()
    <snip>
    Ran 2975 tests in 2.017s
    OK
    <nose.result.TextTestResult run=2975 errors=0 failures=0> 
    
The ``la`` package contains C extensions that speed up common alignment
operations such as adding two unaligned larrys. If the C extensions don't
compile when you build ``la`` then there's an automatic fallback to python
versions of the functions. To see whether you are using the C functions or the
Python functions::

    >>> la.info()
    la version      0.5.0           
    la file         /usr/local/lib/python2.6/dist-packages/la/__init__.pyc
    NumPy           1.5.1  
    Bottleneck      0.4.2
    HDF5 archiving  Available (h5py 1.3.0)      
    listmap         Faster C version
    listmap_fill    Faster C version    
    
Since ``la`` can run in a pure python mode, you can use ``la`` by just saving
it and making sure that python can find it.    
    
URLs
====

===============   ========================================================
 download          http://pypi.python.org/pypi/la
 docs              http://berkeleyanalytics.com/la
 code              https://github.com/kwgoodman/la
 mailing list      http://groups.google.com/group/labeled-array
 issue tracker     https://github.com/kwgoodman/la/issues
===============   ========================================================
