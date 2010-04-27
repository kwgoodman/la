
============
Introduction
============

larry is a labeled Numpy array. In this part of the manual I'll try to give
you a sense of what larry can do and then talk about the license and
installation. 

Who's larry?
============

The main class of the la package is a labeled array, larry. A larry consists
of a data array and a label list. The data array is stored as a NumPy array
and the label list as a list of lists.

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
    
A larry can have any dimension. Here, for example, is one way to create a
one-dimensional larry:
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

License
=======

larry is distributed under a BSD license. Parts of Scipy and numpydoc, which
both have BSD licenses, are included in larry. See the LICENSE file, which
is distributed with the la package, for details.

Installation
============

The ``la`` package requires Python and Numpy. Numpy 1.4 or newer is
recommended for its improved NaN handling. Also some of the unit tests in the
``la`` package require Numpy 1.4 or newer and many require
`nose <http://somethingaboutorange.com/mrl/projects/nose>`_.

To save and load larrys in HDF5 format, you need
`h5py <http://h5py.alfven.org>`_ with HDF5 1.8.
        
The ``la`` package currently contains no extensions, just Python code, so
there is nothing to compile. You can just save the ``la`` package and make
sure Python can find it.
    
Atlernatively, you can install the traditional way:
::
    $ python setup.py build
    $ sudo python setup.py install
    
Or, if you wish to specify where ``la`` is installed, for example inside
``/usr/local``:
::        
    $ python setup.py build
    $ sudo python setup.py install --prefix=/usr/local
    
After you have installed ``la``, run the suite of unit tests:
::    
    >>> import la
    >>> la.test()
    <snip>
    Ran 906 tests in 0.771s
    OK
    <nose.result.TextTestResult run=906 errors=0 failures=0>       
    
URLs
====

===============   ========================================================
 code              https://launchpad.net/larry
 docs              http://larry.sourceforge.net
 list              http://groups.google.ca/group/pystatsmodels
 devel list        https://launchpad.net/~larry-discuss
 devel archive     https://lists.launchpad.net/larry-discuss/threads.html
===============   ========================================================

