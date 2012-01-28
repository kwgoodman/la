"Moving (rolling) statistics on numpy arrays."

import numpy as np
import bottleneck as bn

from la.missing import nans, ismissing
from la.farray import lastrank

__all__ = ['move_median', 'move_nanmedian', 'move_func', 'move_nanranking',
           'movingsum', 'movingsum_forward', 'movingrank'] #Last row deprecated


# MEDIAN --------------------------------------------------------------------

def move_median(arr, window, axis=-1, method='loop'):
    """
    Moving window median along the specified axis.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving median. By default the
        moving median is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'loop'      brute force python loop (default)
            'strides'   strides tricks (ndim < 4)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving median of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> la.farray.move_median(arr, window=2)
    array([ NaN,  1.5,  2.5,  3.5,  4.5])

    """
    if method == 'strides':
        y = move_func_strides(np.median, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.median, arr, window, axis=axis)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError, msg
    return y

def move_nanmedian(arr, window, axis=-1, method='loop'):
    """
    Moving window median along the specified axis, ignoring NaNs.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving median. By default the
        moving median is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'loop'      brute force python loop (default)
            'strides'   strides tricks (ndim < 4)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving median of the input array along the specified axis,
        ignoring NaNs. (A window with all NaNs returns NaN for the window
        maximum.) The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> la.farray.move_nanmedian(arr, window=2)
    array([ NaN,  1.5,  2. ,  4. ,  4.5])

    """
    if method == 'strides':
        y = move_func_strides(bn.nanmedian, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(bn.nanmedian, arr, window, axis=axis)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError, msg
    return y

# RANKING -------------------------------------------------------------------

def move_nanranking(arr, window, axis=-1, method='strides'):
    """
    Moving window ranking along the specified axis, ignoring NaNs.

    The output is normalized to be between -1 and 1. For example, with a window
    width of 3 (and with no ties), the possible output values are -1, 0, 1.
    
    Ties are broken by averaging the rankings. See the examples below. 

    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving ranking. By default the
        moving ranking is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'strides'   strides tricks (ndim < 4) (default)
            'loop'      brute force python loop
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving ranking of the input array along the specified axis,
        ignoring NaNs. (A window with all NaNs returns NaN for the window
        ranking; if all elements in a window are NaNs except the last element,
        this NaN is returned.) The output has the same shape as the input.

    Examples
    --------
    With window=3 and no ties, there are 3 possible output values, i.e.
    [-1., 0., 1.]:

    >>> arr = np.array([1, 2, 6, 4, 5, 3])
    >>> la.farray.move_nanranking(arr, window=3)
    array([ NaN,  NaN,   1.,   0.,   0.,  -1.])

    Ties are broken by averaging the rankings of the tied elements:

    >>> arr = np.array([1, 2, 1, 1, 1, 2])
    >>> la.farray.move_nanranking(arr, window=3)
    array([ NaN,  NaN, -0.5, -0.5,  0. ,  1. ])

    In a monotonically increasing sequence, the moving window ranking is always
    equal to 1:
    
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> la.farray.move_nanranking(arr, window=3)
    array([ NaN,  NaN,   1.,   1.,   1.])

    """
    if method == 'strides':
        y = move_func_strides(lastrank, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(lastrank, arr, window, axis=axis)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError, msg
    return y

# GENERAL --------------------------------------------------------------------

def move_func(func, arr, window, axis=-1, method='loop', **kwargs):
    """
    Generic moving window function along the specified axis.
    
    Parameters
    ----------
    func : function
        A reducing function such as np.sum, np.max, or np.median that takes
        a Numpy array and axis and, optionally, key word arguments as input.
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to evaluate `func`. By default the window moves
        along the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'loop'      brute force python loop (default)
            'strides'   strides tricks (ndim < 4)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        A moving window evaluation of `func` along the specified axis of the
        input array. The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.arange(4)
    >>> la.farray.move_func(np.sum, arr, window=2)
    array([ NaN,   1.,   3.,   5.])

    which give the same result as:

    >>> la.farray.move_sum(arr, window=2)
    array([ NaN,   1.,   3.,   5.])

    """
    if method == 'strides':
        y = move_func_strides(func, arr, window, axis=axis, **kwargs)
    elif method == 'loop':
        y = move_func_loop(func, arr, window, axis=axis, **kwargs)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError, msg
    return y

def move_func_loop(func, arr, window, axis=-1, **kwargs):
    "Generic moving window function implemented with a python loop."
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    y = nans(arr.shape)
    idx1 = [slice(None)] * arr.ndim
    idx2 = list(idx1)
    for i in range(window - 1, arr.shape[axis]):
        idx1[axis] = slice(i + 1 - window, i + 1)
        idx2[axis] = i
        y[idx2] = func(arr[idx1], axis=axis, **kwargs)
    return y    

def move_func_strides(func, arr, window, axis=-1, **kwargs):
    "Generic moving window function implemented with strides."
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    ndim = arr.ndim
    as_strided = np.lib.stride_tricks.as_strided
    idx = range(ndim)
    axis = idx[axis]
    arrshape0 = tuple(arr.shape)
    if axis >= ndim:
        raise IndexError, "`axis` is out of range."
    if ndim == 1:
        strides = arr.strides
        shape = (arr.size - window + 1, window)
        strides = 2 * strides
        z = as_strided(arr, shape=shape, strides=strides)
        y = func(z, axis=1, **kwargs)
    elif ndim == 2:
        if axis == 1:
            arr = arr.T
        strides = arr.strides
        shape = (arr.shape[0] - window + 1, window, arr.shape[1]) 
        strides = (strides[0],) + strides 
        z = as_strided(arr, shape=shape, strides=strides)
        y = func(z, axis=1, **kwargs)
        if axis == 1:
            y = y.T    
    elif ndim == 3:
        if axis > 0:
            arr = arr.swapaxes(0, axis)
        strides = arr.strides
        shape = (arr.shape[0]-window+1, window, arr.shape[1], arr.shape[2])
        strides = (strides[0],) + strides
        z = as_strided(arr, shape=shape, strides=strides)
        y = func(z, axis=1, **kwargs)
        if axis > 0:
            y = y.swapaxes(0, axis)
    else:
        raise ValueError, "Only 1d, 2d, and 3d input arrays are supported."
    ynan = nans(arrshape0)
    index = [slice(None)] * ndim 
    index[axis] = slice(window - 1, None)
    ynan[index] = y
    return ynan

# DEPRECATED ----------------------------------------------------------------

@np.deprecate(new_name='move_nansum')
def movingsum(arr, window, skip=0, axis=-1, norm=False):
    """
    Moving sum ignoring NaNs, optionally normalized for missing (NaN) data.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    skip : int, optional
        By default (skip=0) the movingsum at element *i* is the sum over the
        slice of elements from *i + 1 - window* to *i + 1* (so the last element
        in the sum is *i*). With nonzero `skip` the sum is over the slice from
        *i + 1 window - skip* to *i + 1 - skip*. `skip` cannot be negative.
    axis : int, optional
        The axis over which to perform the moving sum. By default the moving
        sum is taken over the last axis (-1).
    norm : bool, optional
        Whether or not to normalize the sum. The default is not to normalize.
        If there are 3 missing elements in a window, for example, then the
        normalization would be to multiply the sum in that window by
        *window / (window - 3)*.

    Returns
    -------
    y : ndarray
        The moving sum of the input array along the specified axis.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> movingsum(arr, 2)
    array([ NaN,   3.,   5.,   7.,   9.])

    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> movingsum(arr, 2)
    array([ NaN,   3.,   2.,   4.,   9.])
    >>> movingsum(arr, 2, norm=True)
    array([ NaN,   3.,   4.,   8.,   9.])    
    
    """

    # Check input
    if window < 1:  
        raise ValueError, 'window must be at least 1'
    if window > arr.shape[axis]:
        raise ValueError, 'Window is too big.'      
    if skip > arr.shape[axis]:
        raise IndexError, 'Your skip is too large.'
    
    # Set missing values to 0
    m = ismissing(arr) 
    arr = arr.astype(float)
    arr[m] = 0

    # Cumsum
    csx = arr.cumsum(axis)

    # Set up indexes
    index1 = [slice(None)] * arr.ndim 
    index2 = list(index1) 
    index3 = list(index1)
    index4 = list(index1)
    index1[axis] = slice(window - 1, -skip or None)
    index2[axis] = slice(None, -window-skip) 
    index3[axis] = slice(1, None)
    index4[axis] = slice(skip + window - 1, None)

    # Make moving sum
    msx = csx[index1]
    msx[index3] = msx[index3] - csx[index2] 
    csm = (~m).cumsum(axis)     
    msm = csm[index1]
    msm[index3] = msm[index3] - csm[index2]  
    
    # Normalize
    if norm:
        ms = 1.0 * window * msx / msm
    else:
        ms = msx
        ms[msm == 0] = np.nan
    
    # Pad to get back to original shape
    arr.fill(np.nan) 
    arr[index4] = ms

    return arr

def movingsum_forward(x, window, skip=0, axis=-1, norm=False):
    """Movingsum in the forward direction skipping skip dates."""
    flip_index = [slice(None)] * x.ndim 
    flip_index[axis] = slice(None, None, -1)
    msf = movingsum(x[flip_index], window, skip=skip, axis=axis, norm=norm)
    return msf[flip_index]

def movingrank(x, window, axis=-1):
    """Moving rank (normalized to -1 and 1) of a given window along axis.

    Normalized for missing (NaN) data.
    A data point with NaN data is returned as NaN
    If a window is all NaNs except last, this is returned as NaN
    """
    if window > x.shape[axis]:
        raise ValueError, 'Window is too big.'
    if window < 2:
        raise ValueError, 'Window is too small.'
    nt = x.shape[axis]
    mr = np.nan * np.zeros(x.shape)        
    for i in xrange(window-1, nt): 
        index1 = [slice(None)] * x.ndim 
        index1[axis] = i
        index2 = [slice(None)] * x.ndim 
        index2[axis] = slice(i-window+1, i+1, None)
        mr[index1] = np.squeeze(lastrank(x[index2], axis=axis))
    return mr
