=====================
Numpy array functions
=====================

Most larry methods are available as Numpy array functions. For example, to
find the z-score along the last axis of a larry, `lar`, you would do::

    >>> lar.zscore(axis=-1)
    
Here's the corresponding operation on a Numpy array, `arr`::

    >>> la.farray.zscore(arr, axis=-1)

This section of the manual contains most of the Numpy array functions
available in the ``la`` package.

.. contents:: Functions and examples


Moving (rolling) statistics
---------------------------

Moving (rolling) summary statistics along the specified axis of a Numpy array.
Some of the functions remove NaNs from the rolling window (mov_nansum); some
don't (mov_sum).

Most of the moving window functions offer several algorithms for calculating
moving statistics.

            ==========  ==========================
            'filter'    scipy.ndimage.convolve1d 
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

.. autofunction:: la.farray.mov_std

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
             
.. autofunction:: la.farray.mov_func_strides

------------

.. autofunction:: la.farray.mov_func_loop
