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
summary statistics. Some of the functions remove NaNs from the moving window
(mov_nansum); some don't (mov_sum).

Most of the moving window functions offer three algorithms for moving the
window:

            ==========  ==========================
            'filter'    scipy.ndimage 
            'strides'   strides tricks (ndim < 4)
            'loop'      brute force python loop
            ==========  ==========================

------------
             
.. autofunction:: la.farray.mov_sum

------------
             
.. autofunction:: la.farray.mov_nansum

------------
             
.. autofunction:: la.farray.mov_mean

------------

.. autofunction:: la.farray.mov_nanmean

------------
             
.. autofunction:: la.farray.mov_var

------------

.. autofunction:: la.farray.mov_nanvar

------------
             
.. autofunction:: la.farray.mov_std

------------

.. autofunction:: la.farray.mov_nanstd

------------
             
.. autofunction:: la.farray.mov_min

------------

.. autofunction:: la.farray.mov_nanmin

------------
             
.. autofunction:: la.farray.mov_max

------------

.. autofunction:: la.farray.mov_nanmax

------------

.. autofunction:: la.farray.mov_nanranking

------------
             
.. autofunction:: la.farray.mov_count

------------
             
.. autofunction:: la.farray.mov_median

------------

.. autofunction:: la.farray.mov_nanmedian

------------
             
.. autofunction:: la.farray.mov_func


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

------------
             
.. autofunction:: la.farray.nanmean

------------
             
.. autofunction:: la.farray.nanmedian

------------
             
.. autofunction:: la.farray.nanvar

------------
             
.. autofunction:: la.farray.nanstd






