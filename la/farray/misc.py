"NaN-aware numpy array functions for miscellaneous operations."

import numpy as np
import bottleneck as bn

__all__ = ['geometric_mean', 'correlation', 'covMissing', 'shuffle']


def geometric_mean(x, axis=-1, check_for_greater_than_zero=True):
    """
    Return the geometric mean of matrix x along axis, ignore NaNs.
    
    Raise an exception if any element of x is zero or less.
     
    """
    if (x <= 0).any() and check_for_greater_than_zero:
        msg = 'All elements of x (except NaNs) must be greater than zero.'
        raise ValueError, msg
    x = x.copy()
    m = np.isnan(x)
    np.putmask(x, m, 1.0)
    m = np.asarray(~m, np.float64)
    m = m.sum(axis)
    x = np.log(x).sum(axis)
    g = 1.0 / m
    x = np.multiply(g, x)
    x = np.exp(x)
    idx = np.ones(x.shape)
    if idx.ndim == 0:
        if m == 0:
            idx = np.nan
    else:
        np.putmask(idx, m==0, np.nan)
    x = np.multiply(x, idx)
    return x

def correlation(arr1, arr2, axis=None):
    """
    Correlation between two Numpy arrays along the specified axis.
    
    This is not a cross correlation function. If the two input arrays have
    shape (n, m), for example, then the output will have shape (m,) if axis
    is 0 and shape (n,) if axis is 1.
    
    Parameters
    ----------
    arr1 : Numpy ndarray
        Input array.
    arr2 : Numpy ndarray
        Input array.        
    axis : {int, None}, optional
        The axis along which to measure the correlation. The default, axis
        None, flattens the input arrays before finding the correlation and
        returning it as a scalar.
    
    Returns
    -------
    corr : Numpy ndarray, scalar
        The correlation between `arr1` and `arr2` along the specified axis.
        
    Examples
    -------- 
    Make two Numpy arrays:
       
    >>> a1 = np.array([[1, 2], [3, 4]])
    >>> a2 = np.array([[2, 1], [4, 3]])
    >>> a1
    array([[1, 2],
           [3, 4]])
    >>> a2
    array([[2, 1],
           [4, 3]])
           
    Find the correlation between the two arrays along various axes:       
    
    >>> correlation(a1, a2)
    0.59999999999999998
    >>> correlation(a1, a2, axis=0)
    array([ 1.,  1.])
    >>> correlation(a1, a2, axis=1)
    array([-1., -1.])

    """
    mask = np.logical_or(np.isnan(arr1), np.isnan(arr2))
    if mask.any():
        # arr1 and/or arr2 contain NaNs, so use slower NaN functions if needed
        if axis == None:
            x1 = arr1.flatten()
            x2 = arr2.flatten()
            idx = ~mask.flatten()
            x1 = x1[idx]
            x2 = x2[idx]
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()        
            num = (x1 * x2).sum()
            den = np.sqrt((x1 * x1).sum() * (x2 * x2).sum()) 
        else:
            x1 = arr1.copy()
            x2 = arr2.copy()
            np.putmask(x1, mask, np.nan)
            np.putmask(x2, mask, np.nan)
            if axis == 0:
                x1 = x1 - bn.nanmean(x1, axis)
                x2 = x2 - bn.nanmean(x2, axis)              
            else:
                idx = [slice(None)] * x1.ndim
                idx[axis] = None
                x1 = x1 - bn.nanmean(x1, axis)[idx]
                x2 = x2 - bn.nanmean(x2, axis)[idx]           
            num = np.nansum(x1 * x2, axis)
            den = np.sqrt(np.nansum(x1 * x1, axis) * np.nansum(x2 * x2, axis))
    else:
        # Neither arr1 or arr2 contains nans, so use faster non-nan functions
        if axis == None:
            x1 = arr1.flatten()
            x2 = arr2.flatten()
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()        
            num = (x1 * x2).sum()
            den = np.sqrt((x1 * x1).sum() * (x2 * x2).sum()) 
        else:
            x1 = arr1
            x2 = arr2
            if axis == 0:
                x1 = x1 - x1.mean(axis)
                x2 = x2 - x2.mean(axis)              
            else:
                idx = [slice(None)] * x1.ndim
                idx[axis] = None   
                x1 = x1 - x1.mean(axis)[idx]
                x2 = x2 - x2.mean(axis)[idx]           
            num = np.sum(x1 * x2, axis)
            den = np.sqrt(np.sum(x1 * x1, axis) * np.sum(x2 * x2, axis))                
    return num / den 

def covMissing(R):
    """
    Covariance matrix adjusted for missing returns.

    covMissing returns the covariance matrix adjusted for missing returns.
    R (NxT) is log stock returns; missing returns are NaN.

    Note the mean of each row of R is assumed to be zero. So returns are not
    demeaned and the covariance is normalized by T not T-1.
    
    """
    mask = np.isnan(R)
    np.putmask(R, mask, 0)
    mask = np.asarray(mask, np.float64)
    mask = 1 - mask # Change meaning of missing matrix to present matrix  

    normalization = np.dot(mask, mask.T)

    if np.any(normalization < 2):
        raise ValueError, 'covMissing: not enough observations'

    C = np.dot(R, R.T) / normalization

    return C   

def shuffle(x, axis=0):
    """
    Shuffle the data inplace along the specified axis.
    
    Unlike numpy's shuffle, this shuffle takes an axis argument. The
    ordering of the labels is not changed, only the data is shuffled.
    
    Parameters
    ----------
    x : ndarray
        Array to be shuffled.
    axis : int
        The axis to shuffle the data along. Default is axis 0.
        
    Returns
    -------
    out : None
        The data is shuffled inplace.        
    
    """
    np.random.shuffle(np.rollaxis(x, axis))
