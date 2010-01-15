
==========
Meet larry
==========

larry is a labeled Numpy array. Here's a two-dimensional larry, *y*, in
schematic form:
::    
                                 date1    date2    date3
                        'AAPL'   209.19   207.87   210.11
                    y = 'IBM'    129.03   130.39   130.55
                        'DELL'    14.82    15.11    14.94
                
larry stores the data as a `Numpy <http://www.numpy.org>`_ array and the
labels as a list of lists (one list for each dimension):
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
    
larry has many builtin methods such as **movingsum, ranking, merge, shuffle,
zscore, demean, lag** as well as typical Numpy methods like **sum, max, std,
sign, clip**. NaNs are treated as missing data.
    
Alignment is automatic when you add (or subtract, multiply, divide) two
larrys.
    
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
    
larry adds the convenience of labels, provides many builtin functions, and
let's you use your exisiting array functions.       

Requirements
============

To run larry you need Python and Numpy. Python 2.6 and Numpy 1.4 are
recommended. (There are many larry unit test failures with Numpy 1.3.)
To save and load larrys in HDF5 format, you need
`h5py <http://h5py.alfven.org>`_.
        
larry currently contains no extensions, just Python code, so there is
nothing to compile. Just save the la directory and make sure Python can
find it. Then run:
::    
    >>> import la
    >>> la.test()
    <snip>
    Ran 529 tests in 0.465s
    OK
    <nose.result.TextTestResult run=529 errors=0 failures=0>

Contents:

.. toctree::
   :maxdepth: 2
   
   archive

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

