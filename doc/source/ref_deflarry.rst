.. _reference:

=============
larry methods
=============

The larry methods can be divided into the following broad categories:

.. contents:: Methods and examples

Below you'll find the methods in each category along with examples. All
of the examples assume that you have already imported larry::

    >>> from la import larry
    
The reference guide for the larry functions, as opposed to methods, can be
found in :ref:`functions`.     
    

.. _creation:

Creation
--------

The creation methods allow you to create larrys.

------------

.. automethod:: la.larry.__init__

------------

.. automethod:: la.larry.fromtuples

------------

.. automethod:: la.larry.fromlist

------------

.. automethod:: la.larry.fromdict

------------

.. automethod:: la.larry.fromcsv


Unary
-----

The unary functions (such as **log, sqrt, sign**) operate on a single larry
and do not change its shape or ordering.


------------
             
.. automethod:: la.larry.log

------------

.. automethod:: la.larry.exp

------------

.. automethod:: la.larry.sqrt

------------

.. automethod:: la.larry.sign

------------

.. automethod:: la.larry.power

------------

.. automethod:: la.larry.cumsum

------------

.. automethod:: la.larry.cumprod

------------

.. automethod:: la.larry.clip

------------

.. automethod:: la.larry.abs

------------

.. automethod:: la.larry.isnan

------------

.. automethod:: la.larry.isfinite

------------

.. automethod:: la.larry.isinf                   

------------

.. automethod:: la.larry.invert

------------

.. automethod:: la.larry.__invert__

             
Binary methods
--------------

The binary methods (such as +, -, / and *) combine a larry with a scalar,
Numpy array, or another larry. More general binary functions, that give you
control of the join method and the fill method can be found in
:ref:`binaryfunc`.

------------ 
             
.. automethod:: la.larry.__add__
             
------------

.. automethod:: la.larry.__sub__
             
------------

.. automethod:: la.larry.__div__
             
------------

.. automethod:: la.larry.__mul__

------------

.. automethod:: la.larry.__and__
             
------------

.. automethod:: la.larry.__or__
             

Reduce
------

The reduce methods (such as **sum** and **std**) aggregate along an axis or
axes thereby reducing the dimension of the larry.

------------
             
.. automethod:: la.larry.sum
             
------------

.. automethod:: la.larry.prod

------------

.. automethod:: la.larry.mean
             
------------

.. automethod:: la.larry.geometric_mean
             
------------

.. automethod:: la.larry.median
             
------------

.. automethod:: la.larry.std

------------

.. automethod:: la.larry.var
             
------------

.. automethod:: la.larry.max

------------

.. automethod:: la.larry.min

------------

.. automethod:: la.larry.any

------------

.. automethod:: la.larry.all

------------

.. automethod:: la.larry.lastrank                      
           
             
Comparison
-----------

The comparison methods, such as ==, >, and !=, perform an element-by-element
comparison and return a bool larry. For example::

    >>> y1 = larry([1, 2, 3, 4])
    >>> y2 = larry([1, 9, 3, 9])
    >>> y1 == y2
    label_0
        0
        1
        2
        3
    x
    array([ True, False,  True, False], dtype=bool)

and ::

    >>> from la import larry
    >>> y1 = larry([1, 2], [['a', 'b']])
    >>> y2 = larry([1, 2], [['b', 'c']])
    >>> y1 == y2
    label_0
        b
    x
    array([False], dtype=bool)
    
A larry can be compared with a scalar, NumPy array, list, tuple, and another
larry.

.. warning::

    Do not compare a NumPy array on the left-hand side with a larry on the
    right-hand side. You will get unexpected results. To compare a larry to
    a NumPy array, put the array on the right-hand side. 

------------

.. automethod:: la.larry.__eq__

------------

.. automethod:: la.larry.__ne__

------------

.. automethod:: la.larry.__lt__

------------

.. automethod:: la.larry.__gt__

------------

.. automethod:: la.larry.__le__

------------

.. automethod:: la.larry.__ne__ 

    
Get and set
-----------

The get methods return subsets of a larry through indexing and the set methods
assign values to a subset of a larry.

------------

.. automethod:: la.larry.__getitem__

------------

.. automethod:: la.larry.take

------------

.. automethod:: la.larry.lix

------------

.. automethod:: la.larry.__setitem__

------------

.. automethod:: la.larry.get

------------

.. automethod:: la.larry.set

------------

.. automethod:: la.larry.getx

------------

.. automethod:: la.larry.A

------------

.. automethod:: la.larry.getlabel

------------

.. automethod:: la.larry.fill

------------

.. automethod:: la.larry.pull

------------

.. automethod:: la.larry.keep_label

------------

.. automethod:: la.larry.keep_x


Label
-----

The label methods allow you to get information (and change) the labels of a
larry.

------------

.. automethod:: la.larry.maxlabel

------------

.. automethod:: la.larry.minlabel

------------

.. automethod:: la.larry.labelindex

------------

.. automethod:: la.larry.maplabel


Moving window statistics
------------------------

Moving window statistics along the specified axis of a larry.

------------

.. automethod:: la.larry.move_sum

------------

.. automethod:: la.larry.move_mean

------------

.. automethod:: la.larry.move_std

------------

.. automethod:: la.larry.move_min

------------

.. automethod:: la.larry.move_max

------------

.. automethod:: la.larry.move_ranking

------------

.. automethod:: la.larry.move_median

------------

.. automethod:: la.larry.move_func

------------

.. automethod:: la.larry.movingsum_forward


Calculation
----------- 

The calculation methods transform the larry.

------------

.. automethod:: la.larry.demean

------------

.. automethod:: la.larry.demedian

------------

.. automethod:: la.larry.zscore

------------

.. automethod:: la.larry.ranking

------------

.. automethod:: la.larry.quantile


Group
-----

The group methods allow you to calculate the group mean (or median or ranking)
along axis=0 of a larry. For example, let's calculate the group mean of *y*
where group 1 is ('e', 'a'), group 2 is ('d', 'c'), and group 3 is ('b')::

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
   
------------

.. automethod:: la.larry.group_ranking

------------

.. automethod:: la.larry.group_mean

------------

.. automethod:: la.larry.group_median   


Alignment
---------

There are several alignment methods. See also the :func:`align <la.align>`
function.
    
------------

.. automethod:: la.larry.morph

------------

.. automethod:: la.larry.morph_like

------------

.. automethod:: la.larry.merge

------------

.. automethod:: la.larry.squeeze

------------

.. automethod:: la.larry.lag

------------

.. automethod:: la.larry.sortaxis

------------

.. automethod:: la.larry.flipaxis


Shuffle
-------

The data and the labels of larrys can be randomly shuffled in-place.

------------

.. automethod:: la.larry.shuffle

------------

.. automethod:: la.larry.shufflelabel


.. _missing:

Missing data
------------

NaNs are treated as missing data in larry::

    >>> import la
    >>> y = larry([1.0, la.nan])
    >>> y.sum()
    1.0
    
Missing value makers for various dtypes:

============== ===============
dtype           missing marker
============== ===============
float           NaN
object          None
str             ''
int, bool, etc  Not supported
============== ===============

------------

.. automethod:: la.larry.ismissing     
    
------------

.. automethod:: la.larry.cut_missing

------------

.. automethod:: la.larry.push

------------

.. automethod:: la.larry.vacuum

------------

.. automethod:: la.larry.nan_replace       


   
Size, shape, dtype
------------------

Here are the methods that tell you about the size, shape, and dtype of larry.
Some of the methods (**T, flatten, unflatten**) change the shape of the larry.
    
------------

.. automethod:: la.larry.nx
    
------------

.. automethod:: la.larry.size
    
------------

.. automethod:: la.larry.shape
    
------------

.. automethod:: la.larry.ndim
    
------------

.. automethod:: la.larry.dtype
    
------------

.. automethod:: la.larry.astype
    
------------

.. automethod:: la.larry.T

------------

.. automethod:: la.larry.swapaxes
    
------------

.. automethod:: la.larry.flatten
    
------------

.. automethod:: la.larry.unflatten 

------------

.. automethod:: la.larry.insertaxis


Conversion
----------

Methods to convert larrys to other formats. For the corresponding 'from'
methods, see :ref:`creation`.

------------

.. automethod:: la.larry.totuples
    
------------

.. automethod:: la.larry.tolist         

------------

.. automethod:: la.larry.todict 

------------

.. automethod:: la.larry.tocsv         

------------

.. automethod:: la.larry.tofile         


Copy
----

Here are the methods that copy a larry or its components.  
 
------------

.. automethod:: la.larry.copy

------------

.. automethod:: la.larry.copylabel

------------

.. automethod:: la.larry.copyx

