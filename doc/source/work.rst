
==================
Working with larry
==================

This section describes how to work with larrys.

.. contents::

All of the examples below assume that you have already imported larry:
::
    >>> from la import larry
    
More examples of what you can do with larrys are given in :ref:`reference`.    


.. _creation:

Creating a larry
----------------

Let's create a larry (LAbeled aRRaY):
::
    >>> y = larry([1, 2, 3])
    >>> y
    label_0
        0
        1
        2
    x
    array([1, 2, 3])

A larry consists of a data array and a label. In the statement above, larry
creates the data array by converting the list ``[1, 2, 3]`` to a Numpy array.
The label, since none was specified, defaults to ``range(n)``, where *n* in
this case is 3.

To use your own labels pass them in when you construct a larry:
::
    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['a', 'b'], [11, 13]])
    >>> y
    label_0
        a
        b
    label_1
        11
        13
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
In the example above, the first row is labeled 'a' and the second row is
labeled 'b'. The first and second columns are labeled 11 and 13, respectively.

Here is a more formal way to create a larry:
::
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])
    >>> label = [['north', 'south'], ['east', 'west']]
    
    >>> larry(x, label)
    label_0
        north
        south
    label_1
        east
        west
    x
    array([[1, 2],
           [3, 4]])

The labels, along any one axis, must be unique. Let's try to create a larry
with labels that are not unique:
::
    >>> larry([1, 2], [['a', 'a']])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 97, in __init__
        raise ValueError, msg % (i, value, key)
    ValueError: Elements of label not unique along axis 0. There are 2 labels named `a`.

The shape of the data array must agree with the shape of the label. Let's try
to create a larry whose data shape does not agree with the label shape:
::
    >>> larry([[1, 2], [3, 4]], [['a', 'b'], ['c']])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 86, in __init__
        raise ValueError, msg % i
    ValueError: Length mismatch in label and x along axis 1
    
The following functions can also be used to create larrys:

* **fromlist**
* **fromtuples**
* **fromdict**

Here's how to create a larry using **fromlist**:
::
    >>> data = [[1, 2, 3, 4], [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')]]
    >>> la.fromlist(data)
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
Note that **fromlist** changed the data type from integer to float. That
allows for the possibility of missing data. Let's throw out the last data
point in the example above (note the NaN):
::
    >>> data = [[1, 2, 3], [('a', 'c'), ('a', 'd'), ('b', 'c')]]
    >>> la.fromlist(data)
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[  1.,   2.],
           [  3.,  NaN]])
            
Here are examples of **fromtuples** and **fromdict**:
::
    >>> data = [('a', 'a', 1), ('a', 'b', 2), ('b', 'a', 3), ('b', 'b', 4)]
    >>> la.fromtuples(data)
    label_0
        a
        b
    label_1
        a
        b
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])

    >>> data = {('a', 'c'): 1, ('a', 'd'): 2, ('b', 'c'): 3, ('b', 'd'): 4}
    >>> la.fromdict(data)
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])         

See :ref:`conversion` for a discussion of the corresponding methods,
**tolist, totuples, todict**. 


Shape, size, type
-----------------

The shape, size, and type of a larry are the same as the underlying Numpy
array:
::
    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['r0', 'r1'], ['c0', 'c1']])
    >>> y.shape
    (2, 2)
    >>> y.size
    4
    >>> y.ndim
    2
    >>> y.dtype
    dtype('float64') 
    
larry does not have a reshape method. A reshape would scramble all the labels.
But larry does have a **flatten** method:
::
    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['r0', 'r1'], ['c0', 'c1']])
    >>> y.flatten()
    label_0
        ('r0', 'c0')
        ('r0', 'c1')
        ('r1', 'c0')
        ('r1', 'c1')
    x
    array([ 1.,  2.,  3.,  4.])
    >>> y.flatten(order='F')
    label_0
        ('r0', 'c0')
        ('r1', 'c0')
        ('r0', 'c1')
        ('r1', 'c1')
    x
    array([ 1.,  3.,  2.,  4.]) 
    
Flattened larrys can be unflattened:
::
    >>> yflat = y.flatten()
    >>> yflat.unflatten()
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
The transpose of a larry:
::
    >>> y.T
    label_0
        c0
        c1
    label_1
        r0
        r1
    x
    array([[ 1.,  3.],
           [ 2.,  4.]])
                
  
    
Missing values
--------------

NaNs in the data array (not the label) are treated as missing values:
::
    >>> import la
    >>> y = larry([1.0, la.nan, 3.0])
    >>> y.sum()
    4.0

Note that ``la.nan`` is the same as Numpy's NaN:
::
    >>> import numpy as np
    >>> la.nan is np.nan
    True
    
Missing values can be replaced:
::
    >>> from la import nan
    >>> y = larry([1.0, nan])
    >>> y.nan_replace(0.0) 
    label_0
        0
        1
    x
    array([ 1.,  0.])
    
There are more larry methods that deal with missing values. See
:ref:`missing` in :ref:`reference`.      

Indexing
--------

In most cases, indexing into a larry is similar to indexing into a Numpy
array:
::
    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['a', 'b'], [11, 13]])
    >>> y[:,0]
    label_0
        a
        b
    x
    array([ 1.,  3.])
    
    >>> z = larry([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> z[1:7:2]
    label_0
        1
        3
        5
    x
    array([2, 4, 6])
    
The following types of indexing are not currently supported by larry (but they
are supported when doing an assignment by indexing, see :ref:`assignment`):

* Fancy indexing
* Indexing with Ellipsis    
    
There are several ways to index by label name. For example, let's pull row
'a' (the first row) from the larry *y*. We can use **labelindex**:
::
    >>> idx = y.labelindex('a', axis=0)
    >>> y[idx,:]
    label_0
        11
        13
    x
    array([ 1.,  2.])

or **morph**:
::
    >>> y.morph(['a'], axis=0)
    label_0
        a
    label_1
        11
        13
    x
    array([[ 1.,  2.]])

or **pull**:    
::
    >>> y.pull('a', axis=0)
    label_0
        11
        13
    x
    array([ 1.,  2.])
    
As another example of indexing with labels, let's index into the rows using
the labels ['b', 'a']:
::
    >>> y.morph(['b', 'a'], axis=0)
    label_0
        b
        a
    label_1
        11
        13
    x
    array([[ 3.,  4.],
           [ 1.,  2.]]) 
           
or, resorting to hackery:
::                   
    >>> idx = map(y.labelindex, ['b', 'a'], [0]*2)
    >>> y[idx]
    label_0
        b
        a
    label_1
        11
        13
    x
    array([[ 3.,  4.],
           [ 1.,  2.]])


.. _assignment:

Assignment by indexing
----------------------

Assignment by indexing is the same as with Numpy arrays:
::
    >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
    
    >>> y[0,0] = 99
    >>> y
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[99,  2],
           [ 3,  4]])
           
    >>> y[:,0] = 99
    >>> y
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[99,  2],
           [99,  4]])
           
    >>> y[y > 10] = 0
    >>> y
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[0, 2],
           [0, 4]])
           
    >>> y[y==0] = [22, 33]
    >>> y
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[22,  2],
           [33,  4]])           

You can also assign values by updating them with the **merge** method. See
:ref:`merge` for details.

Alignment
---------

Alignment is automatic when you add (or subtract, multiply, divide, logical
and, logical or) two larrys. To demonstrate, let's create two larrys that are
not aligned:
::
    >>> y1 = larry([1, 2], [['a', 'z']])
    >>> y2 = larry([1, 2], [['z', 'a']])
    
What is ``y1 + y2``?
::
    >>> y1 + y2
    label_0
        a
        z
    x
    array([3, 3])

Let's look at a more complicated example:
::
    >>> z1 = larry([1, 2], [['a', 'b']])
    >>> z2 = larry([3, 4], [['c', 'd']])

    >>> z1 + z2
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 494, in __add__
        x, y, label = self.__align(other)
      File "la/la/deflarry.py", line 731, in __align
        raise IndexError, 'A dimension has no matching labels'
    IndexError: A dimension has no matching labels
    
Why did we get an index error when we tried to sum *z1* and *z2*? We got an
error because *z1* and *z2* have no overlap: there are no labels 'a' and 'b'
in *z2* to add to those in *z1*.

Let's make a third larry that can be added to *z1*:
::
    >>> z3 = larry([3, 4], [['b', 'c']])
    >>> z1 + z3
    label_0
        b
    x
    array([5])
    
Note that the only overlap between *z1* and *z3* is the second element of *z1*
(labeled 'b') with the first element of *z3* (also labeled 'b').

Although we cannot sum *z1* and *z2*, we can merge them:
::
    >>> z1.merge(z2)
    label_0
        a
        b
        c
        d
    x
    array([ 1.,  2.,  3.,  4.])
    
(See :ref:`merge` for more details.)    
       
It is often convenient to pre-align larrys. To align two larrys we use
**morph_like**:
::
    >>> y1 = larry([1, 2, 3], [['a', 'b', 'c']])
    >>> y2 = larry([6, 4, 5], [['c', 'a', 'b']])

    >>> y2.morph_like(y1)
    label_0
        a
        b
        c
    x
    array([ 4.,  5.,  6.])
    
Alternatively, when we only want to align the larry along one axis (the
example above only contain one axis):    
::    
    >>> y2.morph(y1.getlabel(axis=0), axis=0)
    label_0
        a
        b
        c
    x
    array([ 4.,  5.,  6.])
    
We can also morph an array with labels that do not yet exist ('d' and 'e' in
the following example):
::
    >>> lar.morph(['a', 'b', 'c', 'd', 'e'], axis=0)
    label_0
        a
        b
        c
        d
        e
    x
    array([  1.,   2.,   3.,  NaN,  NaN])
    

.. _merge:
    
Merging
-------    

Two larrys can be merged to form a single larry:
::
    >>> y1 = larry([1, 2], [['a', 'b']])
    >>> y2 = larry([3, 4], [['c', 'd']])

    >>> y1.merge(y2)
    label_0
        a
        b
        c
        d
    x
    array([ 1.,  2.,  3.,  4.])

In the example above there is no overlap between *y1* and *y2*: there are
no data in *y1* with labels 'c' or 'd' and there are no data in *y2* with
labels 'a' or 'b'.

Let's try to merge two larrys that have an overlap (label 'b' along axis 0):
::
    >>> y1 = larry([1, 2], [['a', 'b']])
    >>> y2 = larry([3, 4], [['b', 'c']])

    >>> y1.merge(y2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/deflarry.py", line 2381, in merge
        raise ValueError('overlapping values')
    ValueError: overlapping values
    
To merge larrys with overlaps you must set ``update`` to True:
::
    >>> y1.merge(y2, update=True)
    label_0
        a
        b
        c
    x
    array([ 1.,  3.,  4.])
    
When ``update`` is set to True, the data in *y1* that overlap with the data
in *y2* are updated with the data in *y2*. In the example above, the element
in *y1* with label 'b' is updated to 3 from 2.    


Groups
------

larry has several methods for calculating group statistics:

* **group_mean**
* **group_median**
* **group_ranking**

Let's start with an example where group1 contains labels 'a' and 'c' and
group2 contains labels 'b' and 'd':
::
    >>> y = larry([1, 2, 3, 4], [['a', 'b', 'c', 'd']])
    >>> group = larry(['group1', 'group2', 'group1', 'group2'], [['a', 'b', 'c', 'd']])

    >>> y.group_mean(group)
    label_0
        a
        b
        c
        d
    x
    array([ 2.,  3.,  2.,  3.])

The group statistics always work along axis 0 and ``group`` must be 1d. Let
find the group mean of a larry, *y*:
::
    >>> y = larry([[1, 2], [5, 6], [8, 9]])
    >>> group = larry(['g1', 'g2', 'g1'])

    >>> y.group_mean(group)
    label_0
        0
        1
        2
    label_1
        0
        1
    x
    array([[ 4.5,  5.5],
           [ 5. ,  6. ],
           [ 4.5,  5.5]]) 


Copying
-------

A larry consists of two parts: a data array and a label list. larry provides
methods that allow you to make a copy of the data array, a copy of the label
list, or a copy of the entire larry. Some examples:
::
    >>> y = larry([1, 2], [['a', 9]])
    >>> y.copyx()
    array([1, 2])
    >>> y.copylabel()
    [['a', 9]]
    >>> y.copy()
    label_0
        a
        9
    x
    array([1, 2])


.. _conversion:

Conversion
----------

A larry can be converted to various other formats using the following
conversion methods:

* **totuples**
* **tolist**
* **todict**

Some examples:
::
    >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])

    >>> y.totuples()
    [('r0', 'c0', 1), ('r0', 'c1', 2), ('r1', 'c0', 3), ('r1', 'c1', 4)]

    >>> y.tolist()
    [[1, 2, 3, 4], [('r0', 'c0'), ('r0', 'c1'), ('r1', 'c0'), ('r1', 'c1')]]

    >>> y.todict()
    {('r0', 'c1'): 2, ('r1', 'c1'): 4, ('r0', 'c0'): 1, ('r1', 'c0'): 3}

The corresponding methods **fromtuples, fromlist, and fromdict** are discused
in :ref:`creation`.

Archiving
---------

The archiving of larrys is described in :ref:`archive`.


Known issues
------------

**Complex numbers**

The are currently no unit tests for complex numbers in larry. Therefore, the
extent of support for complex numbers is unknown. Be aware that even if a
function or method runs with complex input, the output might be wrong.




    
    
               

  

        
