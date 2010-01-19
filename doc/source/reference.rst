
======================
Larry method reference
======================

The larry methods can be divided into the following broad categories:

* __init__
* Unary
* Binary
* Reduce
* Comparison
* Get, set
* Label
* Calculation
* Group
* Alignment
* Shuffle
* Size, shape


__init__
---------

Here is an example of one way to initialize a larry, *y*:
::
    >>> import numpy as np
    >>> from la import larry
    >>> 
    >>> x = np.array([[1, 2], [3, 4]])
    >>> label = [['a', 'b'], [8, 10]]
    >>>
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
   :members: log, ext, isfinite, sqrt, sign, power, cumsum, clip, nan_replace,
             abs, isnan, isfinite, isinf 
             
             
Binary
------

The binary methods (such as +, -, / and *) combine a larry with a scalar,
Numpy array, or another larry. For example:
::
    >>> from la import larry
    >>> y1 = larry([1,2], [['a', 'b']])
    >>> y2 = larry([1,2], [['b', 'c']])
    >>>
    >>> y1 + y2
    label_0
        b
    x
    array([3])

.. autoclass:: la.larry
   :members: __add__, __radd__, __sub__, __rsub__, __div__, __rdiv__, __mul__,
             __rmul__, __and__, __rand__, __or__, __ror__
             
             
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
   :members: sum, mean, median, std, var, max, min, lastrank, lastrank_decay,
             any, all             
             
             
Comparision
-----------

The comparison methods, such as ==, >, and !=, perform an element-by-element
comparison and return a bool larry. For example:
::
    >>> from la import larry
    >>> y1 = larry([1,2], [['a', 'b']])
    >>> y2 = larry([1,2], [['b', 'c']])
    >>>
    >>> y1 == y2
    label_0
        b
    x
    array([False], dtype=bool)

.. autoclass:: la.larry
   :members: __eq__, __ne__, __lt__, __gt__, __le__, __ne__  

    
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
    >>> 
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
   :members: __getitem__, __setitem__, set, get, getx, fill
   

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
   :members: maxlabel, minlabel, getlabel, labelindex
   

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
   :members: demean, demedian, zscore, push, movingsum, movingsum_forward,
             ranking, movingrank, quantile, cut_missing, cov, keep_label,
             keep_x
             

Group
-----

The group methods allow you to calculate the group mean (or median or ranking)
along axis=0 of a larry. For example, let's calculate the group mean of *y*
where group 1 is ('e', 'a'), group 2 is ('d', 'c'), and group 3 is ('b'):
::
    >>> from la import larry
    >>> y  = larry([[1], [2], [3], [4], [5]], [['a', 'b', 'c', 'd', 'e'], [0]])
    >>> group = larry([1, 1, 2, 2, 3], [['e', 'a', 'd', 'c', 'b']])
    >>>
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
   
  
Alignment
---------

.. autoclass:: la.larry
   :members: morph, morph_like, merge, vacuum, squeeze, pull, lag, 
   
   
Shuffle
-------

.. autoclass:: la.larry
   :members: shuffle, shufflelabel
   
   
Size, shape
-----------

.. autoclass:: la.larry
   :members: nx, size, shape, ndim, dtype, T, A           


Copy
----

.. autoclass:: la.larry
   :members: copy, copylabel
  
 

