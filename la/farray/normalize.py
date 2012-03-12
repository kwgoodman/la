"NaN-aware numpy array functions for normalization operations."

import numpy as np
ndtri = None
import bottleneck as bn
from la.missing import nans

__all__ = ['lastrank', 'ranking', 'push', 'quantile', 'demean',
           'demedian', 'zscore']

   
def lastrank(x, axis=-1, decay=0.0):
    """
    The ranking of the last element along the axis, ignoring NaNs.
    
    The ranking is normalized to be between -1 and 1 instead of the more
    common 1 and N. The results are adjusted for ties. Suitably slicing
    the output of the `ranking` method will give the same result as
    `lastrank`. The only difference is that `lastrank` is faster.       

    Parameters
    ----------
    x : numpy array
        The array to rank.
    axis : int, optional
        The axis over which to rank. By default (axis=-1) the ranking
        (and reducing) is performed over the last axis.          
    decay : scalar, optional
        Exponential decay strength. Cannot be negative. The default
        (decay=0) is no decay. In normal ranking (decay=0) all elements
        used to calculate the rank are equally weighted and so the
        ordering of all but the last element does not matter. In
        exponentially decayed ranking the ordering of the elements
        influences the ranking: elements nearer the last element get more
        weight.
        
    Returns
    -------
    d : array
        In the case of, for example, a 2d array of shape (n, m) and
        axis=1, the output will contain the rank (normalized to be between
        -1 and 1 and adjusted for ties) of the the last element of each row.
        The output in this example will have shape (n,). 
            
    Examples
    -------- 
    Create an array:
                    
    >>> y1 = larry([1, 2, 3])
    
    What is the rank of the last element (the value 3 in this example)?
    It is the largest element so the rank is 1.0:
    
    >>> import numpy as np
    >>> from la.afunc import lastrank
    >>> x1 = np.array([1, 2, 3])
    >>> lastrank(x1)
    1.0
    
    Now let's try an example where the last element has the smallest
    value:
    
    >>> x2 = np.array([3, 2, 1])
    >>> lastrank(x2)
    -1.0
    
    Here's an example where the last element is not the minimum or maximum
    value:
    
    >>> x3 = np.array([1, 3, 4, 5, 2])
    >>> lastrank(x3)
    -0.5
    
    Finally, let's add a large decay. The decay means that the elements
    closest to the last element receive the most weight. Because the
    decay is large, the first element (the value 1) doesn't get any weight
    and therefore the last element (2) becomes the smallest element:
    
    >>> lastrank(x3, decay=10)
    -1.0
    
    """
    if x.size == 0:
        # At least one dimension has length 0
        shape = list(x.shape)
        shape.pop(axis)
        r = nans(shape, dtype=x.dtype) 
        if (r.ndim == 0) and (r.size == 1):
            r = np.nan     
        return r 
    indlast = [slice(None)] * x.ndim 
    indlast[axis] = slice(-1, None)
    indlast2 = [slice(None)] * x.ndim 
    indlast2[axis] = -1  
    if decay > 0:
        # Exponential decay 
        nt = x.shape[axis]
        w = nt - np.ones(nt).cumsum()
        w = np.exp(-decay * w)
        w = nt * w / w.sum()
        matchdim = [None] * x.ndim 
        matchdim[axis] = slice(None)
        w = w[matchdim]
        g = ((x[indlast] > x) * w).sum(axis)
        e = ((x[indlast] == x) * w).sum(axis)
        n = (np.isfinite(x) * w).sum(axis)
        r = (g + g + e - w.flat[-1]) / 2.0
        r = r / (n - w.flat[-1])
    elif decay < 0:
        raise ValueError, 'decay must be greater than or equal to zero.'        
    else:
        # Special case the most common case, decay = 0, for speed
        g = (x[indlast] > x).sum(axis)
        e = (x[indlast] == x).sum(axis)
        n = np.isfinite(x).sum(axis)
        r = (g + g + e - 1.0) / 2.0
        r = r / (n - 1.0)      
    r = 2.0 * (r - 0.5)    
    if x.ndim == 1:
        if not np.isfinite(x[indlast2]):
            r = np.nan
    else:
        np.putmask(r, ~np.isfinite(x[indlast2]), np.nan)
    return r    

def ranking(x, axis=0, norm='-1,1'):
    """
    Normalized ranking treating NaN as missing and averaging ties.
    
    Parameters
    ----------
    x : ndarray
        Data to be ranked.
    axis : {int, None} optional
        Axis to rank over. Default axis is 0.
    norm: str, optional
        A string that specifies the normalization:
            ==========  ================================================
            '0,N-1'     Zero to N-1 ranking
            '-1,1'      Scale zero to N-1 ranking to be between -1 and 1
            'gaussian'  Rank data then scale to a Gaussian distribution
            ==========  ================================================
        The default ranking is '-1,1'.
        
    Returns
    -------
    idx : ndarray
        The ranked data.The dtype of the output is always np.float even if
        the dtype of the input is int.
    
    Notes
    -----
    If there is only one non-NaN value along the given axis, then that value
    is set to the midpoint of the specified normalization method. For example,
    if the input is array([1.0, nan]), then 1.0 is set to zero for the '-1,1'
    and 'gaussian' normalizations and is set to 0.5 (mean of 0 and 1) for the
    '0,N-1' normalization.
    
    For '0,N-1' normalization, note that N is x.shape[axis] even in there are
    NaNs. That ensures that when ranking along the columns of a 2d array, for
    example, the output will have the same min and max along all columns.
    
    """
    if axis is None:
        ranked_x = ranking(x.reshape(-1), norm=norm)
        return ranked_x.reshape(*x.shape)
    ax = axis
    if ax < 0:
        # This converts a negative axis to the equivalent positive axis
        ax = range(x.ndim)[ax]
    masknan = np.isnan(x)
    countnan = np.expand_dims(masknan.sum(ax), ax)
    countnotnan = x.shape[ax] - countnan
    idx = bn.nanrankdata(x, ax)
    idx -= 1
    if norm == '-1,1':
        idx /= (countnotnan - 1)
        idx *= 2
        idx -= 1
        middle = 0.0
    elif norm == '0,N-1':
        idx *= (1.0 * (x.shape[ax] - 1) / (countnotnan - 1))
        middle = (idx.shape[ax] + 1.0) / 2.0 - 1.0
    elif norm == 'gaussian':
        global ndtri
        if ndtri is None:
            try:
                from scipy.special import ndtri
            except ImportError:
                msg = "SciPy required for `gaussian` normalization"
                raise ImportError(msg)
        idx *= (1.0 * (x.shape[ax] - 1) / (countnotnan - 1))
        idx = ndtri((idx + 1.0) / (x.shape[ax] + 1.0))
        middle = 0.0
    else:
        msg = "norm must be '-1,1', '0,N-1', or 'gaussian'."
        raise ValueError(msg)
    np.putmask(idx, (countnotnan==1)*(~masknan), middle)
    return idx

def push(x, n, axis=-1):
    "Fill missing values (NaN) with most recent non-missing values if recent."
    if axis != -1 or axis != x.ndim-1:
        x = np.rollaxis(x, axis, x.ndim)
    y = np.array(x) 
    if y.ndim == 1:
        y = y[None, :]
    fidx = np.isfinite(y)
    recent = np.nan * np.ones(y.shape[:-1])  
    count = np.nan * np.ones(y.shape[:-1])          
    for i in xrange(y.shape[-1]):
        idx = (i - count) > n
        recent[idx] = np.nan
        idx = ~fidx[...,i]
        y[idx, i] = recent[idx]
        idx = fidx[...,i]
        count[idx] = i
        recent[idx] = y[idx, i]
    if axis != -1 or axis != x.ndim-1:
        y = np.rollaxis(y, x.ndim-1, axis)
    if x.ndim == 1:
        return y[0]
    return y

def _quantileraw1d(xi, q):
    y = np.nan * np.asarray(xi)
    idx = np.where(np.isfinite(xi))[0]
    xi = xi[idx,:]
    nx = idx.size
    if nx:
        jdx = xi.argsort(axis=0).argsort(axis=0)
        mdx = np.nan * jdx
        kdx = 1.0 * (nx - 1) / (q) * np.ones((q, 1))
        kdx = kdx.cumsum(axis=0)
        kdx = np.concatenate((-1 * np.ones((1, kdx.shape[1])), kdx), 0)
        kdx[-1, 0] = nx
        for j in xrange(1, q+1):
            mdx[(jdx > kdx[j-1]) & (jdx <= kdx[j]),:] = j
        y[idx] = mdx
    return y

def quantile(x, q, axis=0):
    """
    Convert elements in each column to integers between 1 and q then normalize.
    
    Result is normalized to -1, 1.
    
    Parameters
    ----------
    x : ndarray
        Input array.
    q : int
        The number of bins into which to quantize the data. Must be at
        least 1 but less than the number of elements along the specified
        axis.
    axis : {int, None}, optional
        The axis along which to quantize the elements. The default is
        axis 0.

    Returns
    -------
    y : ndarray
        A quantized copy of the array.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5, 6])
    >>> la.farray.quantile(arr, 3)
    array([-1., -1.,  0.,  0.,  1.,  1.])

    """
    if q < 1:
        raise ValueError, 'q must be one or greater.'
    elif q == 1:
        y = np.zeros(x.shape)
        np.putmask(y, np.isnan(x), np.nan)
        return y
    if axis == None:
        if q > x.size:
            msg = 'q must be less than or equal to the number of elements '
            msg += 'in x.'
            raise ValueError, msg
        y = np.apply_along_axis(_quantileraw1d, 0, x.flat, q)
        y = y.reshape(x.shape)
    else:        
        if q > x.shape[axis]:
            msg = 'q must be less than or equal to the number of rows in x.'
            raise ValueError, msg
        y = np.apply_along_axis(_quantileraw1d, axis, x, q)
    y = y - 1.0
    y = 1.0 * y / (q - 1.0)
    y = 2.0 * (y - 0.5)
    return y 

def demean(arr, axis=None):
    """
    Subtract the mean along the specified axis.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : {int, None}, optional
        The axis along which to remove the mean. The default (None) is
        to subtract the mean of the flattened array.

    Returns
    -------
    y : ndarray
        A copy with the mean along the specified axis removed.

    Examples
    --------
    >>> arr = np.array([1, np.nan, 2, 3])
    >>> demean(arr)
    array([ -1.,  NaN,   0.,   1.])
 
    """
    marr = bn.nanmean(arr, axis) 
    if (axis != 0) and (not axis is None) and (not np.isscalar(marr)):
        ind = [slice(None)] * arr.ndim
        ind[axis] = np.newaxis
        marr =  marr[ind]
    return arr - marr   

def demedian(arr, axis=None):
    """
    Subtract the median along the specified axis.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : {int, None}, optional
        The axis along which to remove the median. The default (None) is
        to subtract the median of the flattened array.
    
    Returns
    -------
    y : ndarray
        A copy with the median along the specified axis removed.
    
    Examples
    --------
    >>> arr = np.array([1, np.nan, 2, 10])
    >>> demedian(arr)
    array([ -1.,  NaN,   0.,   8.])        
    
    """
    marr = bn.nanmedian(arr, axis) 
    if (axis != 0) and (not axis is None) and (not np.isscalar(marr)):
        ind = [slice(None)] * arr.ndim
        ind[axis] = np.newaxis
        marr =  marr[ind]
    return arr - marr   
    
def zscore(arr, axis=None):
    """
    Z-score along the specified axis.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    axis : {int, None}, optional
        The axis along which to take the z-score. The default (None) is
        to find the z-score of the flattened array.
    
    Returns
    -------
    y : ndarray
        A copy normalized with the Z-score along the specified axis.
    
    Examples
    --------
    >>> arr = np.array([1, np.nan, 2, 3])
    >>> zscore(arr)
    array([-1.22474487,         NaN,  0.        ,  1.22474487])
        
    """
    arr = demean(arr, axis)
    norm = bn.nanstd(arr, axis) 
    if (axis != 0) and (not axis is None) and (not np.isscalar(norm)):
        ind = [slice(None)] * arr.ndim
        ind[axis] = np.newaxis
        norm = norm[ind]
    arr /= norm
    return arr
