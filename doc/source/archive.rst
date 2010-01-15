
===
I/O
===

The la package provides two ways to archive larrys. There are **save** and
**load** functions and there is a dictionary-like interface in the **IO**
class. Both I/O methods store larrys in
`HDF5 <http://www.hdfgroup.org/>`_ 1.8 format.


Save and Load
=============

The simplest way to archive larrys is with the **save** and **load**
functions. To demonstrate, let's start by creating a larry:
::

    >>> import la
    >>> y = la.larry([1,2,3])

Next let's save the larry, *y*, in an archive using the **save** function:
::
    >>> la.save('/tmp/data.hdf5', y, 'y')   
    
To load the larry we use the **load** function:
::
    >>> z = la.load('/tmp/data.hdf5', 'y')
    
The entire larry is loaded from the archive. The **load** function does not
have an option to load parts of a larry, such as a slice. (To load parts of
a larrys from the archive, use the IO class.)    

The name of the larry in **save** and **load** statements must be a string. 
But the string may contain on or more forward slashes ('/'), which means that
larrys can be archived in a hierarchical structure:
::
    >>> la.save('/tmp/data/hdf5', y, '/experiment/2/y')
    >>> z = la.load('/tmp/data/hdf5', '/experiment/2/y')
    
Instead of passing a filename to **save** and **load** you can optionally
pass a `h5py <http://h5py.alfven.org/>`_ File object:
::
    >>> import h5py
    >>> f = h5py.File('/tmp/data.hdf5')
    >>> z = la.load(f, 'y') 
    
    
IO class
========

The IO class provides a dictionary-like interface to the archive.

Let's start by creating two larrys, *y* and *z*:
::
    >>> import la
    >>> a = la.larry([1.0,2.0,3.0,4.0])
    >>> b = la.larry([[1,2],[3,4]])

Here's how to create an IO object:
::
    >>> io = la.IO('/tmp/data.hdf5')
    
Next, let's add the two larrys, *a* and *b*, to the archive and list the
contents of the archive:
::
    >>> io['a'] = a
    >>> io['b'] = b
    >>> io
   
    larry  dtype    shape 
    ----------------------
    a      float64  (4,)  
    b      int64    (2, 2)
    y      int64    (3,)   

We can get a list of the keys (larrys) in the archive:
::
    >>> io.keys()
        ['a', 'b', 'y']
        
When we load from the archive using an io object, we get a lara not a larry:
::
    >>> z = io['a']        
    >>> type(z)
        <class 'la.io.io.lara'>
        
A lara loads the larry label from the archive but does not load the data The
reason a lara is returned and not                


Limitations
===========

There are several limitations of the archiving method used by the la package.
In this section we will discuss two limitations:

* The freespace in the archive is not by default automatically reclaimed after
  deleting larrys.
* In order to archive a larry, its data and labels must be of a type supported
  by HDF5.   

**Freespace**

HDF5 does not keep track of the freespace in an archive across opening and
closing of the archive. Therefore, after opening, closing and deleting larrys
from the archive, the unused space in the archive may grow. The only way to
reclaim the freespace is to repack the archive.

You can use the utility provided by HDF5 to repack the archive or you can use
the repack method or function in the la package:
::
    >>> 
    
**Data types**  

A larry can have labels of mixed type, for example strings and numbers.
However, when archiving larrys in HDF5 format the labels are
converted to Numpy arrays and the elements of a Numpy array must be of the
same type. Therefore, to archive a larry the labels along any one dimension
must be of the same type and that type must be one that is recognized by
h5py and HDF5: strings and scalars. So, for example, if your labels are
datetime.date objects, then you must convert them (perhaps to integers using
the datetime.date.toordinal function) before archiving.


Archive format
==============

An HDF5 archive is contructed from two types of objects: Groups and Datasets.
Groups can contain Datasets and more Groups. Datasets can contain arrays.

larrys are stored in a HDF5 Group. The name of the group is the name of the
larry. The group is given an attribute called 'larry' and assigned the value
True. Inside the group are several HDF5 Datasets. For a 2d larry, for example,
there are three datasets: one to hold the data (named 'x') and two to hold the
labels (named '0' and '1'). In general, for a nd larry there are n+1
datasets.
