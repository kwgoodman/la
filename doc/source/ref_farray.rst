===============
array functions
===============

Most larry methods have an equivalent Numpy array function. For example, to
find the z-score along the last axis of a larry, `lar`, you would do::

    >>> lar.zscore(axis=-1)
    
Here's the corresponding operation on a Numpy array, `arr`::

    >>> la.farray.zscore(arr, axis=-1)

This section of the manual is a reference guide to most of the Numpy array
functions available in the ``la`` package.

.. contents:: Functions and examples


Moving window statistics
------------------------

This section contains Numpy array functions that calculate moving window
summary statistics.

.. note::

    The `Bottleneck package <http://pypi.python.org/pypi/Bottleneck>`_
    contains fast moving window functions.

Most of the moving window functions offer two algorithms for moving the
window:

            ==========  ==========================
            'strides'   strides tricks (ndim < 4)
            'loop'      brute force python loop
            ==========  ==========================

------------
             
.. autofunction:: la.farray.move_median

------------

.. autofunction:: la.farray.move_nanmedian

------------

.. autofunction:: la.farray.move_nanranking


Normalization
-------------

Normalization functions that take a Numpy array as input.

------------
             
.. autofunction:: la.farray.ranking

------------
             
.. autofunction:: la.farray.quantile

------------
             
.. autofunction:: la.farray.demean

------------

.. autofunction:: la.farray.demedian

------------
             
.. autofunction:: la.farray.zscore


Misc
----

Miscellaneous Numpy array functions.

------------
             
.. autofunction:: la.farray.correlation

------------
             
.. autofunction:: la.farray.shuffle

------------
             
.. autofunction:: la.farray.geometric_mean

------------

.. autofunction:: la.farray.covMissing
