"Functions that work with numpy arrays."

import numpy as np

from la.missing import nans, ismissing
from la.farray import lastrank

__all__ = ['movingsum', 'mov_sum', 'movingsum_forward', 'movingrank']


@np.deprecate(new_name='mov_sum')
def movingsum(x, window, skip=0, axis=-1, norm=False):
    """Moving sum optionally normalized for missing (NaN) data."""
    return mov_sum(x, window, skip=skip, axis=axis, norm=norm)

def mov_sum(arr, window, skip=0, axis=-1, norm=False):
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
        *i + 1 window - skip* to *i + 1 - skip*.
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
    >>> mov_sum(arr, 2)
    array([ NaN,   3.,   5.,   7.,   9.])

    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> mov_sum(arr, 2)
    array([ NaN,   3.,   2.,   4.,   9.])
    >>> mov_sum(arr, 2, norm=True)
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
    y = nans(arr.shape)
    y[index4] = ms

    return y

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
