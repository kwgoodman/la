.. _reference:

======================
Larry method reference
======================

The larry methods can be divided into the following broad categories:

* :ref:`init`
* :ref:`Unary`
* :ref:`Binary`
* :ref:`Reduce`
* :ref:`Comparison`
* :ref:`Get`
* :ref:`Label`
* :ref:`Calculation`
* :ref:`Group`
* :ref:`Alignment`
* :ref:`Shuffle`
* :ref:`Missing`
* :ref:`Size`

Below you'll find the reference for each category along with an example.

.. _init:

__init__
---------

Here is an example of one way to initialize a larry, *y*:
::
    >>> import numpy as np
    >>> from la import larry

    >>> x = np.array([[1, 2], [3, 4]])
    >>> label = [['a', 'b'], [8, 10]]

    >>> y = larry(x, label)
    >>> y
    label_0
        a
        b
    label_1
        8
        10
    x
    array([[1, 2],
           [3, 4]])

.. autoclass:: la.larry
   :members: __init__


.. _Unary:

Unary
-----

The unary functions (such as **log, sqrt, sign**) operate on a single larry
and do not change its shape or ordering. For example:
::
    >>> from la import larry
    >>> y = larry([-1,2,-3,4])
    
    >>> y.sign()
    label_0
        0
        1
        2
        3
    x
    array([-1,  1, -1,  1])

.. autoclass:: la.larry
   :members: log, ext, isfinite, sqrt, sign, power, cumsum, clip, abs, isnan,
             isfinite, isinf 
             

.. _Binary:
             
Binary
------

The binary methods (such as +, -, / and *) combine a larry with a scalar,
Numpy array, or another larry. For example:
::
    >>> from la import larry
    >>> y1 = larry([1,2], [['a', 'b']])
    >>> y2 = larry([1,2], [['b', 'c']])

    >>> y1 + y2
    label_0
        b
    x
    array([3])

.. autoclass:: la.larry
   :members: __add__, __radd__, __sub__, __rsub__, __div__, __rdiv__, __mul__,
             __rmul__, __and__, __rand__, __or__, __ror__
             
             
.. _Reduce:

Reduce
------

The reduce methods (such as **sum** and **std**) aggregate along an axis or
axes and thereby reduce the dimension of the larry. For example:
::
    >>> from la import larry
    >>> y = larry([1, 2, 3])
    
    >>> y.sum()
    6
    

.. autoclass:: la.larry
   :members: sum, mean, median, std, var, max, min, any, all             


.. _Comparison:            
             
Comparison
-----------

The comparison methods, such as ==, >, and !=, perform an element-by-element
comparison and return a bool larry. For example:
::
    >>> from la import larry
    >>> y1 = larry([1,2], [['a', 'b']])
    >>> y2 = larry([1,2], [['b', 'c']])

    >>> y1 == y2
    label_0
        b
    x
    array([False], dtype=bool)

.. autoclass:: la.larry
   :members: __eq__, __ne__, __lt__, __gt__, __le__, __ne__  


.. _Get:
    
Get and set
-----------

The get methods return subsets of a larry through indexing and the set methods
assign values to a subset of a larry. For example, let's set all elements of
a larry with a value less 3 to zero:
::
    >>> from la import larry
    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])
    >>> label = [['a', 'b'], [8, 10]]
    >>> y = larry(x, label)

    >>> y[y < 3] = 0
    >>> y
    label_0
        a
        b
    label_1
        8
        10
    x
    array([[0, 0],
           [3, 4]])


.. autoclass:: la.larry
   :members: __getitem__, __setitem__, set, get, getx, fill, pull, 
             keep_label, keep_x
   

.. _Label:

Label
-----

The label methods work with the labels of a larry. For example, what column
number (starting from 0) of a 2d larry is labeled 'west':
::
    >>> from la import larry
    >>> y = larry([[1, 2], [3, 4]], [['north', 'south'], ['east', 'west']])
    
    >>> y.labelindex('west', axis=1)
    1

.. autoclass:: la.larry
   :members: maxlabel, minlabel, getlabel, labelindex, maplabel
   

.. _Calculation:

Calculation
----------- 

The calculation methods transform the larry. For example, here's how to
calculate the zscore of a larry:
::
    >>> from la import larry
    >>> y = larry([1, 2, 3])
    
    >>> y.zscore()
    label_0
        0
        1
        2
    x
    array([-1,  0,  1])  
    
.. autoclass:: la.larry
   :members: demean, demedian, zscore, movingsum, movingsum_forward,
             ranking, movingrank, quantile, cov, lastrank, lastrank_decay
             

.. _Group:

Group
-----

The group methods allow you to calculate the group mean (or median or ranking)
along axis=0 of a larry. For example, let's calculate the group mean of *y*
where group 1 is ('e', 'a'), group 2 is ('d', 'c'), and group 3 is ('b'):
::
    >>> from la import larry
    >>> y  = larry([[1], [2], [3], [4], [5]], [['a', 'b', 'c', 'd', 'e'], [0]])
    >>> group = larry([1, 1, 2, 2, 3], [['e', 'a', 'd', 'c', 'b']])

    >>> y.group_mean(group)
    label_0
        a
        b
        c
        d
        e
    label_1
        0
    x
    array([[ 3. ],
           [ 2. ],
           [ 3.5],
           [ 3.5],
           [ 3. ]])


.. autoclass:: la.larry
   :members: group_ranking, group_mean, group_median
   
  
.. _Alignment:

Alignment
---------

There are several alignment methods. Here are two examples. The first example
aligns *y1* to *y2*; the second example aligns *y2* to *y1*:
::
    >>> from la import larry
    >>> y1 = larry([1, 2], [['a', 'b']])
    >>> y2 = larry([3, 2, 1], [['c', 'b', 'a']])

    >>> y1.morph_like(y2)
    label_0
        c
        b
        a
    x
    array([ NaN,   2.,   1.])

    >>> y2.morph_like(y1)
    label_0
        a
        b
    x
    array([ 1.,  2.])

.. autoclass:: la.larry
   :members: morph, morph_like, merge, squeeze, lag 
   
   
.. _Shuffle:

Shuffle
-------

The data and the labels of larrys can be randomly shuffled in-place:
::
    >>> from la import larry
    >>> y = larry([[1, 2], [3,  4]], [['north', 'south'], ['east', 'west']])

    >>> y.shuffle()
    >>> y
    label_0
        north
        south
    label_1
        east
        west
    x
    array([[3, 4],
           [1, 2]])

    >>> y.shufflelabel()
    >>> y
    label_0
        south
        north
    label_1
        west
        east
    x
    array([[3, 4],
           [1, 2]])

.. autoclass:: la.larry
   :members: shuffle, shufflelabel


.. _Missing:

Missing data
------------

NaNs are treated as missing data in larry:
::
    >>> from la import larry
    >>> import numpy as np
    >>> y = larry([1.0, np.nan])
    >>> y.sum()
    1   

.. autoclass:: la.larry
   :members: cut_missing, push, vacuum, nan_replace 
      

.. _Size:
   
Size, shape, dtype
------------------

Here is an example of the shape and size methods:
::
    >>> from la import larry
    >>> y = larry([[1, 2], [3, 4]])
    
    >>> y.shape
    (2, 2)
    >>> y.size
    4
    >>> y.ndim
    2
    >>> y.dtype
    dtype('int64')

.. autoclass:: la.larry
   :members: nx, size, shape, ndim, dtype, T, A           


.. _Copy:

Copy
----

A larry, or just its label or data, can be copied:
::
    >>> from la import larry
    >>> y = larry([1, 2], [['a', 'b']])
    
    >>> z = y.copy()
    >>> z
    label_0
        a
        b
    x
    array([1, 2])

.. autoclass:: la.larry
   :members: copy, copylabel, copyx
  
 

