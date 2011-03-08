
==================
Working with larry
==================

This section describes how to work with larrys.

.. contents::

All of the examples below assume that you have already imported larry::

    >>> from la import larry
    
More examples of what you can do with larrys are given in :ref:`reference`.    


.. _creation:

Creating a larry
----------------

Let's create a larry (LAbeled aRRaY)::

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

To use your own labels pass them in when you construct a larry::

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

Here is a more formal way to create a larry::

    >>> import numpy as np
    >>> x = np.array([[1, 2], [3, 4]])
    >>> label = [['north', 'south'], ['east', 'west']]

    >>> larry(x, label, dtype=float)
    label_0
        north
        south
    label_1
        east
        west
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
.. warning::

    larry does not copy the data array if it is a Numpy array or if
    np.asarray() does not make a copy such as when the data array is a
    Numpy matrix. However, if you change the dtype of the data array (as in
    the example above), a copy is made. Similarly the label is not copied.
    
larry does not copy the data array or the label list::

    >>> x = np.array([1, 2, 3])
    >>> label = [['a', 'b', 'c']]
    >>> lar = larry(x, label)
    
    >>> lar[0] = -9
    >>> lar.label[0][0] = 'AA'
 
    >>> x    
    array([-9,  2,  3])
    >>> label
    [['AA', 'b', 'c']]

Here's one way to make copies of the data array and label::

    >>> x = np.array([1, 2, 3])
    >>> label = [['a', 'b', 'c']]
    >>> lar = larry(x.copy(), [list(l) for l in label])
     
    >>> lar[0] = -9
    >>> lar.label[0][0] = 'AA'
    
    >>> x
    array([1, 2, 3])
    >>> label
    [['a', 'b', 'c']]              

The labels, along any one axis, must be unique. Let's try to create a larry
with labels that are not unique::

    >>> larry([1, 2], [['a', 'a']])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 97, in __init__
        raise ValueError, msg % (i, value, key)
    ValueError: Elements of label not unique along axis 0. There are 2 labels named `a`.

The shape of the data array must agree with the shape of the label. Let's try
to create a larry whose data shape does not agree with the label shape::

    >>> larry([[1, 2], [3, 4]], [['a', 'b'], ['c']])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/la/deflarry.py", line 86, in __init__
        raise ValueError, msg % i
    ValueError: Length mismatch in label and x along axis 1
    
The integrity checks (labels are unique, dimensions of data and label agree,
etc.) take time. If you wish to skip the checks, then set integrity to False::

    >>> import numpy as np
    >>> x = np.random.rand(10, 10)
    >>> label = [range(10), range(10)]
    
    >>> timeit larry(x, label)
    100000 loops, best of 3: 4.58 us per loop
    >>> timeit larry(x, label, integrity=False)
    1000000 loops, best of 3: 1.22 us per loop

You can also create larrys, filled with random samples, using
:func:`la.rand` and :func:`la.randn`::

    >>> la.rand(2,2)
    label_0
        0
        1
    label_1
        0
        1
    x
    array([[ 0.09277439,  0.94194077],
           [ 0.72887997,  0.41124147]])
           
    >>> la.rand(label=[['row1', 'row2'], ['col1', 'col2']])
    label_0
        row1
        row2
    label_1
        col1
        col2
    x
    array([[ 0.3449072 ,  0.40397174],
           [ 0.7791279 ,  0.86084403]])

The following functions can also be used to create larrys:

* :meth:`fromtuples <la.larry.fromtuples>`
* :meth:`fromdict <la.larry.fromdict>`
* :meth:`fromlist <la.larry.fromlist>`
* :meth:`fromcsv <la.larry.fromcsv>`

Here's how to create a larry using :meth:`fromtuples <la.larry.fromtuples>`::

    >>> data = [('a', 'a', 1), ('a', 'b', 2), ('b', 'a', 3), ('b', 'b', 4)]
    >>> larry.fromtuples(data)
    label_0
        a
        b
    label_1
        a
        b
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
Note that :meth:`fromtuples <la.larry.fromtuples>` changed the data type from
integer to float. That allows for the possibility of missing data (because NaN
is represented as a float). Let's throw out the last data point in the example
above (note the NaN)::

    >>> data = [('a', 'a', 1), ('a', 'b', 2), ('b', 'a', 3)]
    >>> larry.fromtuples(data)
    label_0
        a
        b
    label_1
        a
        b
    x
    array([[  1.,   2.],
           [  3.,  NaN]])
            
Here are examples of :meth:`fromdict <la.larry.fromdict>` and
:meth:`fromlist <la.larry.fromlist>`::

    >>> data = {('a', 'c'): 1, ('a', 'd'): 2, ('b', 'c'): 3, ('b', 'd'): 4}
    >>> larry.fromdict(data)
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
           
    >>> data = [[1, 2, 3, 4], [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')]]
    >>> larry.fromlist(data)
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])           
           
And an example of creating a larry from a csv file::

    >>> y = larry([1, 2, 3], [['a', 'b', 'c']])
    >>> y.tocsv('/tmp/lar.csv')
    >>> larry.fromcsv('/tmp/lar.csv')
    label_0
        a
        b
        c
    x
    array([ 1.,  2.,  3.])                    

See :ref:`conversion` for a discussion of the corresponding methods:
:meth:`totuples <la.larry.totuples>`, :meth:`todict <la.larry.todict>`,
:meth:`tolist <la.larry.tolist>`, :meth:`tocsv <la.larry.tocsv>`.


Shape, size, type
-----------------

The shape, size, and type of a larry are the same as the underlying Numpy
array::

    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['r0', 'r1'], ['c0', 'c1']])
    >>> y.shape
    (2, 2)
    >>> y.size
    4
    >>> y.ndim
    2
    >>> y.dtype
    dtype('float64') 
    
You can change the dtype of a larry by using the
:meth:`astype <la.larry.astype>` method::

    >>> y = larry([1.0, 2.5])
    >>> y.astype(int)
    label_0
        0
        1
    x
    array([1, 2])    
    
larry does not have a reshape method. A reshape would scramble all the labels.
But larry does have a :meth:`flatten <la.larry.flatten>` method and an
:meth:`insertaxis <la.larry.insertaxis>` method.

Here's the :meth:`flatten <la.larry.flatten>` method::

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
    
Flattened larrys can be unflattened::

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
           
To insert a new axis use :meth:`insertaxis <la.larry.insertaxis>`::

    >>> y = larry([1, 2], [['a', 'b']])
    
    >>> y.insertaxis(axis=0, label='NEW')
    label_0
        NEW
    label_1
        a
        b
    x
    array([[1, 2]])

    >>> y.insertaxis(axis=1, label='NEW')
    label_0
        a
        b
    label_1
        NEW
    x
    array([[1],
           [2]])               
           
The transpose of a larry::

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
           
You can also swap any two axes of a larry::

    >>> y.swapaxes(1, 0)
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

NaNs in the data array (not the label) are treated as missing values::

    >>> import la
    >>> y = larry([1.0, la.nan, 3.0])
    >>> y.sum()
    4.0
    
Note that ``la.nan`` is the same as Numpy's NaN::

    >>> import numpy as np
    >>> la.nan is np.nan
    True    
    
Missing values can be found with the :meth:`ismissing <la.larry.ismissing>`
method::

    >>> y.ismissing()
    label_0
        0
        1
        2
    x
    array([False,  True, False], dtype=bool)
       
Missing value makers for various dtypes:

============== ===============
dtype           missing marker
============== ===============
float           NaN
object          None
str             ''
int, bool, etc  Not supported
============== ===============    
    
Missing values can be replaced::

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

There are several ways to access subsets of a larry:

* :ref:`regular_indexing`
* :ref:`label_indexing`
* :ref:`misc_indexing`

.. _regular_indexing:

Regular indexing
""""""""""""""""

Indexing into a larry is similar to indexing into a Numpy array::

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
    
Another similarity to Numpy arrays is the :meth:`take <la.larry.take>` method::

    >>> y = la.rand(2,10)
    >>> y.take([0, 2, 3], axis=1)
    label_0
        0
        1
    label_1
        0
        2
        3
    x
    array([[ 0.67481574,  0.89324137,  0.63388184],
           [ 0.71205393,  0.15894335,  0.01778499]])
 
The following types of indexing are not currently supported by larry (but they
are supported when doing an assignment by indexing, see :ref:`assignment`):

* Some forms of fancy indexing
* Indexing with Ellipsis 

.. _label_indexing:

Indexing by labels
""""""""""""""""""

You can also index into a larry using labels or index numbers or both.

Let's start by making a larry that we can use to demonstrate idexing
by label::

    >>> y = larry(range(6), [['a', 'b', 3, 4, 'e', 'f']])

We can select the first element of the larry using the index value, 0,
or the corresponding label, 'a'::

    >>> y.lix[0]
    0
    >>> y.lix[['a']]
    0
    
In order to distinguish between labels and indices, label elements
must be wrapped in a list while indices (integers) cannot be wrapped
in a list. If you wrap indices in a list they will be interpreted as
label elements.

Slicing can be done with labels or indices or a combination of the
two. A single element along an axis can be selected with a label or
the index value. Several elements along an axis can be selected with
a multi-element list of labels. Lists of indices are not allowed.    

We can slice with index values or with labels::

    >>> y.lix[0:]
    label_0
        a
        b
        3
        4
        e
        f
    x
    array([0, 1, 2, 3, 4, 5])

    >>> y.lix[['a']:]
    label_0
        a
        b
        3
        4
        e
        f
    x
    array([0, 1, 2, 3, 4, 5])
        
    >>> y.lix[['a']:['e']]
    label_0
        a
        b
        3
        4
    x
    array([0, 1, 2, 3])

    >>> y.lix[['a']:['e']:2]
    label_0
        a
        3
    x
    array([0, 2])   

Be careful of the difference between indexing with indices and
indexing with labels. In the first example below 4 is an index; in
the second example 4 is a label element::

    >>> y.lix[['a']:4]
    label_0
        a
        b
        3
        4
    x
    array([0, 1, 2, 3])

    >>> y.lix[['a']:[4]]
    label_0
        a
        b
        3
    x
    array([0, 1, 2])

.. warning::

    When indexing with multi-element lists of labels along more than one
    axes, rectangular indexing is used instead of fancy indexing. Note
    that the corresponding situation with NumPy arrays would produce
    fancy indexing.

Here's a demonstration of rectangular indexing::

    >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
    >>> y.lix[['a', 'b'], ['c', 'd']]
    label_0
        a
        b
    label_1
        c
        d
    x
    array([[1, 2],
           [3, 4]])
        
The rectangular indexing above is very different from how Numpy arrays
behave. The corresponding example with a NumyPy array::

    >>> x = np.array([[1, 2], [3, 4]])
    >>> x[[0, 1], [0, 1]]
    array([1, 4])
    
To get rectangular indexing with Numpy arrays::

    >>> x[np.ix_([0,1], [0,1])]
    array([[1, 2],
           [3, 4]])           

.. _misc_indexing:

Other indexing by labels
""""""""""""""""""""""""   
    
There are several other, miscellaneous ways to index by label name.

Let's look at several different ways to pull row 'a' (the first row) from a
larry *y*.

We can use :meth:`labelindex <la.larry.labelindex>`::

    >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['a', 'b'], [11, 13]])
    >>> idx = y.labelindex('a', axis=0)
    >>> y[idx,:]
    label_0
        11
        13
    x
    array([ 1.,  2.])

or :meth:`morph <la.larry.morph>`::

    >>> y.morph(['a'], axis=0)
    label_0
        a
    label_1
        11
        13
    x
    array([[ 1.,  2.]])

or :meth:`pull <la.larry.pull>`::

    >>> y.pull('a', axis=0)
    label_0
        11
        13
    x
    array([ 1.,  2.])
    
As another example of indexing with labels, let's index into the rows using
the labels ['b', 'a']::

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
           
or, resorting to hackery::

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

Assignment by indexing is the same as with Numpy arrays::

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

You can also assign values by updating them with the
:meth:`merge <la.larry.merge>` method. See :ref:`merge` for details.

Alignment
---------

Alignment is automatic when you add (or subtract, multiply, divide, logical
and, logical or) two larrys. To demonstrate, let's create two larrys that are
not aligned::

    >>> y1 = larry([1, 2], [['a', 'z']])
    >>> y2 = larry([1, 2], [['z', 'a']])
    
What is ``y1 + y2``? ::

    >>> y1 + y2
    label_0
        a
        z
    x
    array([3, 3])
    
By default, binary operations between two larrys use an inner join of the
labels (the intersection of the labels)::

    >>> lar1 = larry([1, 2])
    >>> lar2 = larry([1, 2, 3])
    >>> lar1 + lar2
    label_0
        0
        1
    x
    array([2, 4])

To control the join method (as well as the fill method) use the general
binary function :func:`la.binaryop`. Or use the convenience functions
:func:`la.add`, :func:`la.subtract`, :func:`la.multiply`, :func:`la.divide`.

The sum of two larrys using an outer join (union of the labels)::

    >>> la.add(lar1, lar2, join='outer')
    label_0
        0
        1
        2
    x
    array([  2.,   4.,  NaN])
    
The available join methods are inner, outer, left, right, and list. If the
join method is specified as a list then the first element in the list is the
join method for axis=0, the second element is the join method for axis=1, and
so on. 
    
The fill method can also be specified (see :func:`la.add` for details)::

    >>> la.add(lar1, lar2, join='outer', missone=0)
    label_0
        0
        1
        2
    x
    array([ 2.,  4.,  3.])
    
It is often useful to align two larrys. Once the labels are aligned then you
can use the underlying numpy arrays directly without worrying about alignment.
To align two larrys::

    >>> lar3, lar4 = la.align(lar1, lar2)
    >>> lar3
    label_0
        0
        1
    x
    array([1, 2])
    >>> lar4
    label_0
        0
        1
    x
    array([1, 2])

    >>> lar3, lar4 = la.align(lar1, lar2, join='outer')
    >>> lar3
    label_0
        0
        1
        2
    x
    array([  1.,   2.,  NaN])
    >>> lar4
    label_0
        0
        1
        2
    x
    array([1, 2, 3])
    
Sometimes you only want to align a larry along one axis. To align a larry
along one axis use :meth:`morph <la.larry.morph>`::

    >>> y = larry([[1, 2], [3, 4]], [['r1', 'r2'], ['c1', 'c2']])
    >>> y.morph(['r2', 'r1'], axis=0)
    label_0
        r2
        r1
    label_1
        c1
        c2
    x
    array([[3, 4],
           [1, 2]])

You may want to align with labels that don't exist in the larry::

    >>> y.morph(['r2', 'r1', 'r99'], axis=0)
    label_0
        r2
        r1
        r99
    label_1
        c1
        c2
    x
    array([[  3.,   4.],
           [  1.,   2.],
           [ NaN,  NaN]])
    
Binary operations such as ``+``, ``-``, ``*`` , and ``/`` may return a larry
whose label ordering is different from the two input larrys.

Along any axis where the two input larrys of a binary operation are not
aligned, the labels in the output larry will be sorted (in ascending order).
For those axes where the two input larrys are already aligned, the label
ordering will not change.

Let's look at an example where axis 0 is not aligned but axis 1 is aligned.
Note that the labels along axis 1 are in descending order::

    >>> y1 = larry([[1, 2], [3, 4]], [['a', 'z'], ['z', 'a']])
    >>> y2 = larry([[1, 2], [3, 4]], [['z', 'a'], ['z', 'a']])

    >>> y1 + y2
    label_0
        a
        z
    label_1
        z
        a
    x
    array([[4, 6],
           [4, 6]])
           
In the example above, axis 0 in ``y1`` and ``y2`` is not aligned, therefore
axis 0 in the output larry is aligned in ascending order. However, axis 1,
which is already aligned is left in descending order.

If you want to change the ordering of the labels, you can use
:meth:`sortaxis <la.larry.sortaxis>`::

    >>> y2.sortaxis()
    label_0
        a
        z
    label_1
        a
        z
    x
    array([[4, 3],
           [2, 1]])

    >>> y2.sortaxis(axis=1)
    label_0
        z
        a
    label_1
        a
        z
    x
    array([[2, 1],
           [4, 3]])
 
    >>> y2.sortaxis(reverse=True)
    label_0
        z
        a
    label_1
        z
        a
    x
    array([[1, 2],
           [3, 4]])

You can also change the ordering of the axis with
:meth:`flipaxis <la.larry.flipaxis>`::

    >>> y2.flipaxis(axis=0)
    label_0
        a
        z
    label_1
        z
        a
    x
    array([[3, 4],
           [1, 2]])
    

.. _merge:
    
Merging
-------    

Two larrys can be merged to form a single larry::

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

Let's try to :meth:`merge <la.larry.merge>` two larrys that have an overlap
(label 'b' along axis 0)::

    >>> y1 = larry([1, 2], [['a', 'b']])
    >>> y2 = larry([3, 4], [['b', 'c']])

    >>> y1.merge(y2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/deflarry.py", line 2381, in merge
        raise ValueError('overlapping values')
    ValueError: overlapping values
    
To merge larrys with overlaps you must set ``update`` to True::

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


Moving statistics
-----------------

larry has several methods for calculating moving window statistics:

* :meth:`move_sum <la.larry.move_sum>`
* :meth:`move_mean <la.larry.move_mean>`
* :meth:`move_std <la.larry.move_std>`
* :meth:`move_min <la.larry.move_min>`
* :meth:`move_max <la.larry.move_max>`
* :meth:`move_ranking <la.larry.move_ranking>`
* :meth:`move_median <la.larry.move_median>`
* :meth:`move_func <la.larry.move_func>`

Let's calculate the 3-element moving mean of a larry containing random samples
from a Gaussian distribution::

    >>> lar = la.randn(10)
    >>> mmean = lar.move_mean(window=3)
    >>> mmean
    label_0
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
    x
    array([        NaN,         NaN,  0.06219305, -0.30511785, -0.03330039,
            0.44019147,  0.38927096, -0.37702583, -0.07486695,  0.70283137])

larry also provides a generic moving window function that you can supply with
your own function. (The function you supply must be a reducing function and
take `axis` as input). To duplicate the moving mean above, let's use np.mean::

    >>> mmean2 = lar.move_func(np.mean, window=3) 
    >>> la.testing.assert_larry_equal(mmean, mmean2)
    >>>  


Groups
------

larry has several methods for calculating group statistics:

* :meth:`group_mean <la.larry.group_mean>`
* :meth:`group_median <la.larry.group_median>`
* :meth:`group_ranking <la.larry.group_ranking>`

Let's start with an example where group1 contains labels 'a' and 'c' and
group2 contains labels 'b' and 'd'::

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

The group statistics always work along axis 0 and ``group`` must be 1d. Let's
find the group mean of a larry, *y*::

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
list, or a copy of the entire larry. Some examples::

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

* :meth:`totuples <la.larry.totuples>`
* :meth:`todict <la.larry.todict>`
* :meth:`tolist <la.larry.tolist>`
* :meth:`tocsv <la.larry.tocsv>`
* :meth:`tofile <la.larry.tofile>`

Some examples::

    >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])

    >>> y.totuples()
    [('r0', 'c0', 1), ('r0', 'c1', 2), ('r1', 'c0', 3), ('r1', 'c1', 4)]

    >>> y.tolist()
    [[1, 2, 3, 4], [('r0', 'c0'), ('r0', 'c1'), ('r1', 'c0'), ('r1', 'c1')]]

    >>> y.todict()
    {('r0', 'c1'): 2, ('r1', 'c1'): 4, ('r0', 'c0'): 1, ('r1', 'c0'): 3}
    
    >>> y.tocsv('/tmp/lar.csv') 

    >>> import StringIO
    >>> f = StringIO.StringIO()
    >>> lar.tofile(f)
    >>> print f.getvalue()
    ,c0,c1
    r0,1,2
    r1,3,4

The corresponding methods
:meth:`fromtuples <la.larry.fromtuples>`,
:meth:`fromdict <la.larry.fromdict>`,
:meth:`fromlist <la.larry.fromlist>`,
:meth:`fromcsv <la.larry.fromcsv>`
are discused in :ref:`creation`.


Archiving
---------

The archiving of larrys is described in :ref:`archive`.


Known issues
------------

**Complex numbers**

The are currently no unit tests for complex numbers in larry. Therefore, the
extent of support for complex numbers is unknown. Be aware that even if a
function or method runs with complex input, the output might be wrong.

**Comparing with Numpy arrays**

Do not compare (==, !=, >, <, >=, <=, |, &) a NumPy array on the left-hand
side with a larry on the right-hand side. You will get unexpected results. To
compare a larry to a NumPy array, put the Numpy array on the right-hand side.

This works::

    >>> [2, 2, 4] == larry([1, 2, 3])
    label_0
        0
        1
        2
    x
    array([False,  True, False], dtype=bool)

    >>> larry([1, 2, 3]) == np.array([2, 2, 4])
    label_0
        0
        1
        2
    x
    array([False,  True, False], dtype=bool)
    
But this doesn't work::

    >>> np.array([2, 2, 4]) == larry([1, 2, 3])
    array([ True,  True,  True], dtype=bool)    
