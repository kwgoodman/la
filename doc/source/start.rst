
===========
Quick start
===========

Sometimes the best way to learn how to use a new package is to look at a few
examples. A line of code is worth a thousand words (or is that a thousand
bugs?).

Let's begin by creating a larry (Labeled ARRaY):
::
    >>> from la import larry
    >>> y = larry([1, 2, 3])
    >>> y
    label_0
        0
        1
        2
    x
    array([1, 2, 3])
    
In the statement above the list is converted to a Numpy array and the labels
default to ``range(n)``, where *n* in this case is 3.

To use our own labels we pass them in when we construct a larry:
::
    >>> y = larry([[1.0,2.0],[3.0,4.0]], [['a', 'b'], [11, 13]])
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

In most cases, indexing into a larry is similar to indexing into a Numpy
array:
::
    >>> y[:,0]
    label_0
        a
        b
    x
    array([ 1.,  3.])

    
Indexing by label name is only supported indirectly:
::
    >>> idx = y.labelindex('a', axis=0)
    >>> y[idx,:]
    label_0
        11
        13
    x
    array([ 1.,  2.])

Next let's look at some properties of the larry, *y*:
::
    >>> y.shape
    (2, 2)
    >>> y.dtype
    dtype('float64')
    >>> y.size
    4
    
And let's do some simple calculations:
::
    >>> y.mean()
    2.5
    >>> y.mean(axis=1)
    label_0
        a
        b
    x
    array([ 1.5,  3.5])
    >>> y.demean(axis=1)
    label_0
        a
        b
    label_1
        11
        13
    x
    array([[-0.5,  0.5],
           [-0.5,  0.5]])    
    >>> y.zscore(axis=1)
    label_0
        a
        b
    label_1
        11
        13
    x
    array([[-1.,  1.],
           [-1.,  1.]])
    >>> y.T
    label_0
        11
        13
    label_1
        a
        b
    x
    array([[ 1.,  3.],
           [ 2.,  4.]])

Let's looks at some operations that involve two larrys. First create two
larrys:
::
    >>> la1 = larry([1.0, 2.0], [['a', 'b']])
    >>> la2 = larry([3.0, 4.0], [['c', 'd']])
    
Let's try to sum la1 and la2:
::
    >>> la1 + la2
    IndexError: A dimension has no matching labels
    
Why did we get an index error when we tried to sum la1 and la2? We got an
error because la1 and la2 have no overlap: there are no elements 'a' and 'b'
in la2 to add to la1.

Let's make a third larry that can be added to la1:
::
    >>> la3 = larry([3.0, 4.0], [['b', 'c']])
    >>> la1 + la3
    label_0
        b
    x
    array([ 5.])
    
Note that the only overlap between la1 and la3 is the second element of la1
(labeled 'b') with the first element of la3 (also labeled 'b').

Although we cannot sum la1 and la2, we can merge them:
::
    >>> la1.merge(la2)
    label_0
        a
        b
        c
        d
    x
    array([ 1.,  2.,  3.,  4.])
    
Here is an example with two larrys that have full overlap but are not aligned.
In that case larry does the alignment for you:
::
    >>> x1 = larry([[1,2],[3,4]], [['north', 'south'],['east', 'west']])
    >>> x2 = larry([[1,2],[3,4]], [['south', 'north'],['west', 'east']])
    >>> x1 + x2
    label_0
        north
        south
    label_1
        east
        west
    x
    array([[5, 5],
           [5, 5]])
           
More examples of what you can do with larrys are given throughout the manual.           
    
    


    
    
               

  

        
