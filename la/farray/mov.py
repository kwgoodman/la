"Moving (rolling) statistics on numpy arrays."

import numpy as np

from la.missing import nans, ismissing
from la.farray import nanmean, nanstd, lastrank
from scipy.ndimage import convolve1d, maximum_filter1d, minimum_filter1d

__all__ = ['mov_sum', 'mov_nansum', 'mov_mean', 'mov_nanmean',
           'mov_var', 'mov_nanvar', 'mov_std', 'mov_nanstd',
           'mov_min', 'mov_nanmin', 'mov_max', 'mov_nanmax',
           'mov_nanranking', 'mov_count',
           'mov_func_strides', 'mov_func_loop',
           'movingsum', 'movingsum_forward', 'movingrank'] #Last row deprecated

# Functions to add:
#
# mov_zscore, mov_median, mov_percentile

# SUM -----------------------------------------------------------------------

def mov_sum(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_sum_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.sum, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.sum, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_nansum(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nansum_filter(arr, window, axis=axis)
    elif method == 'cumsum':
        y = mov_nansum_cumsum(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.nansum, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.nansum, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'cumsum', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_sum_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    return arr

def mov_nansum_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = 0
    nrr = nrr.astype(int)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr[nrr == window] = np.nan
    return arr

def mov_nansum_cumsum(arr, window, axis=-1):
    """
    A moving window sum along the specified axis, ignoring NaNs.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving sum. By default the moving
        sum is taken over the last axis (-1).

    Returns
    -------
    y : ndarray
        The moving sum of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    TODO: examples  
    
    """

    # Check input
    if window < 1:  
        raise ValueError, 'window must be at least 1.'
    if window > arr.shape[axis]:
        raise ValueError, 'Window is too big.'      
    
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
    index1[axis] = slice(window - 1, None)
    index2[axis] = slice(None, -window) 
    index3[axis] = slice(1, None)

    # Make moving sum
    msx = csx[index1]
    msx[index3] = msx[index3] - csx[index2] 
    csm = (~m).cumsum(axis)     
    msm = csm[index1]
    msm[index3] = msm[index3] - csm[index2]  
    
    # Fill in windows with all missing values
    msx[msm == 0] = np.nan
    
    # Pad to get back to original shape
    arr.fill(np.nan) 
    arr[index1] = msx

    return arr

# MEAN ----------------------------------------------------------------------

def mov_mean(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_mean_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.mean, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.mean, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_nanmean(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nanmean_filter(arr, window, axis=axis)
    elif method == 'cumsum':
        y = mov_nanmean_cumsum(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(nanmean, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(nanmean, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'cumsum', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_mean_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    w = np.empty(window)
    w.fill(1.0 / window)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    return arr

def mov_nanmean_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = 0
    nrr = nrr.astype(int)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr /= (window - nrr)
    arr[nrr == window] = np.nan
    return arr

def mov_nanmean_cumsum(arr, window, axis=-1):
    """
    A moving window mean along the specified axis, ignoring NaNs.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving mean. By default the moving
        mean is taken over the last axis (-1).

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    TODO: examples  
    
    """

    # Check input
    if window < 1:  
        raise ValueError, 'window must be at least 1.'
    if window > arr.shape[axis]:
        raise ValueError, 'Window is too big.'      
    
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
    index1[axis] = slice(window - 1, None)
    index2[axis] = slice(None, -window) 
    index3[axis] = slice(1, None)

    # Make moving sum
    msx = csx[index1]
    msx[index3] = msx[index3] - csx[index2] 
    csm = (~m).cumsum(axis)     
    msm = csm[index1]
    msm[index3] = msm[index3] - csm[index2]  
    
    # Fill in windows with all missing values
    msx[msm == 0] = np.nan
    
    # Pad to get back to original shape
    arr.fill(np.nan) 
    arr[index1] = msx / msm

    return arr

# VAR -----------------------------------------------------------------------

def mov_var(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_var_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.var, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.var, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_nanvar(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nanvar_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(nanvar, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(nanvar, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_var_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    w = np.empty(window)
    w.fill(1.0 / window)
    x0 = (1 - window) // 2
    y = convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0)
    y *= y
    arr *= arr
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    arr -= y 
    return arr

def mov_nanvar_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = 0
    nrr = nrr.astype(int)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    y = convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0)
    y /= (window - nrr)
    y *= y
    arr *= arr
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    arr /= (window - nrr)
    arr -= y
    arr[nrr == window] = np.nan
    return arr

# STD -----------------------------------------------------------------------

def mov_std(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_std_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.std, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.std, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_nanstd(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nanstd_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(nanstd, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(nanstd, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

def mov_std_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    y = mov_var_filter(arr, window, axis=axis)
    np.sqrt(y, y)
    return y

def mov_nanstd_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    y = mov_nanvar_filter(arr, window, axis=axis)
    np.sqrt(y, y)
    return y

# MIN -----------------------------------------------------------------------

def mov_min(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_min_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.min, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.min, arr, window, axis=axis)
    else:
        raise ValueError, "`method` must be 'filter', 'strides', or 'loop'."
    return y

def mov_nanmin(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nanmin_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_nanmin_strides(arr, window, axis=axis)
    elif method == 'loop':
        y = mov_nanmin_loop(arr, window, axis=axis)
    else:
        raise ValueError, "`method` must be 'filter', 'strides', or 'loop'."
    return y

def mov_min_filter(arr, window, axis=-1):
    "Note: Nans handled differently from other methods. Give example."
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    y = arr.astype(float)
    x0 = (window - 1) // 2
    y = minimum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                         origin=x0)
    return y

def mov_nanmin_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = np.inf
    x0 = (window - 1) // 2
    minimum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=arr)
    w = np.ones(window, dtype=int)
    nrr = nrr.astype(int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr[nrr == window] = np.nan
    return arr

def mov_nanmin_loop(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = np.inf
    y = mov_func_loop(np.min, arr, window, axis=axis)
    m = mov_func_loop(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

def mov_nanmin_strides(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = np.inf
    y = mov_func_strides(np.min, arr, window, axis=axis)
    m = mov_func_strides(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

# MAX -----------------------------------------------------------------------

def mov_max(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_max_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.max, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.max, arr, window, axis=axis)
    else:
        raise ValueError, "`method` must be 'filter', 'strides', or 'loop'."
    return y

def mov_nanmax(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_nanmax_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = mov_nanmax_strides(arr, window, axis=axis)
    elif method == 'loop':
        y = mov_nanmax_loop(arr, window, axis=axis)
    else:
        raise ValueError, "`method` must be 'filter', 'strides', or 'loop'."
    return y

def mov_max_filter(arr, window, axis=-1):
    "Note: Nans handled differently from other methods. Give example."
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."  
    y = arr.astype(float)
    x0 = (window - 1) // 2
    y = maximum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                         origin=x0)
    return y

def mov_nanmax_filter(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    x0 = (window - 1) // 2
    maximum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=arr)
    w = np.ones(window, dtype=int)
    nrr = nrr.astype(int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr[nrr == window] = np.nan
    return arr

def mov_nanmax_loop(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    y = mov_func_loop(np.max, arr, window, axis=axis)
    m = mov_func_loop(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

def mov_nanmax_strides(arr, window, axis=-1):
    if axis == None:
        raise ValueError, "An `axis` value of None is not supported."
    if window < 1:  
        raise ValueError, "`window` must be at least 1."
    if window > arr.shape[axis]:
        raise ValueError, "`window` is too long."
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    y = mov_func_strides(np.max, arr, window, axis=axis)
    m = mov_func_strides(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

# RANKING -------------------------------------------------------------------

def mov_nanranking(arr, window, axis=-1, method='strides'):
    if method == 'strides':
        y = mov_func_strides(lastrank, arr, window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(lastrank, arr, window, axis=axis)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError, msg
    return y

# COUNT MISSING -------------------------------------------------------------

def mov_count(arr, window, axis=-1, method='filter'):
    if method == 'filter':
        y = mov_sum_filter(ismissing(arr), window, axis=axis)
    elif method == 'strides':
        y = mov_func_strides(np.sum, ismissing(arr), window, axis=axis)
    elif method == 'loop':
        y = mov_func_loop(np.sum, ismissing(arr), window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError, msg
    return y

# GENERAL --------------------------------------------------------------------

def mov_func_loop(func, arr, window, axis=-1, *args, **kwargs):
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
        y[idx2] = func(arr[idx1], axis, *args, **kwargs)
    return y    

def mov_func_strides(func, arr, window, axis=-1, *args, **kwargs):
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
        y = func(z, axis=1, *args, **kwargs)
    elif ndim == 2:
        if axis == 1:
            arr = arr.T
        strides = arr.strides
        shape = (arr.shape[0] - window + 1, window, arr.shape[1]) 
        strides = (strides[0],) + strides 
        z = as_strided(arr, shape=shape, strides=strides)
        y = func(z, axis=1, *args, **kwargs)
        if axis == 1:
            y = y.T    
    elif ndim == 3:
        if axis > 0:
            arr = arr.swapaxes(0, axis)
        strides = arr.strides
        shape = (arr.shape[0]-window+1, window, arr.shape[1], arr.shape[2])
        strides = (strides[0],) + strides
        z = as_strided(arr, shape=shape, strides=strides)
        y = func(z, axis=1, *args, **kwargs)
        if axis > 0:
            y = y.swapaxes(0, axis)
    else:
        raise ValueError, "Only 1d, 2d, and 3d input arrays are supported."
    ynan = nans(arrshape0)
    index = [slice(None)] * ndim 
    index[axis] = slice(window - 1, None)
    ynan[index] = y
    return ynan

# UTILITIES -----------------------------------------------------------------

def nanvar(x, axis):
    y = nanstd(x, axis)
    y *= y
    return y

# DEPRECATED ----------------------------------------------------------------

@np.deprecate(new_name='mov_nansum')
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


