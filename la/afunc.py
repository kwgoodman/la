"Functions that work with numpy arrays."

import numpy as np

from la.util.scipy import nanmedian, rankdata


# Group functions ----------------------------------------------------------

def group_ranking(x, groups, norm='-1,1', ties=True):
    """
    Ranking within groups along axis=0.
    
    Parameters
    ----------
    x : ndarray
        Data to be ranked.
    groups : list
        List of group membership of each element along axis=0.
    norm: str
        A string that specifies the normalization:
        '0,N-1'     Zero to N-1 ranking
        '-1,1'      Scale zero to N-1 ranking to be between -1 and 1
        'gaussian'  Rank data then scale to a Gaussian distribution
    ties: bool
        If two elements of `x` have the same value then they will be ranked
        by their order in the array (False). If `ties` is set to True
        (default), then the ranks are averaged.
        
    Returns
    -------
    idx : ndarray
        The ranked data. The dtype of the output is always np.float even if
        the dtype of the input is int.
    
    Notes
    ----
    If there is only one non-NaN value within a group along the axis=0, then
    that value is set to the midpoint of the specified normalization method.
    
    For '0,N-1' normalization, note that N is the number of element in the
    group even in there are NaNs.
    
    """
  
    # Find set of unique groups
    ugroups = unique_group(groups)
    
    # Convert groups to a numpy array
    groups = np.asarray(groups)
  
    # Loop through unique groups and normalize
    xnorm = np.nan * np.zeros(x.shape)
    for group in ugroups:
        idx = groups == group
        xnorm[idx,:] = ranking(x[idx,:], axis=0, norm=norm, ties=ties) 
           
    return xnorm

def group_mean(x, groups):
    """
    Mean with groups along axis=0.
    
    Parameters
    ----------
    x : ndarray
        Input data.
    groups : list
        List of group membership of each element along axis=0.
        
    Returns
    -------
    idx : ndarray
        The group mean of the data along axis 0.

    """

    # Find set of unique groups
    ugroups = unique_group(groups)
    
    # Convert groups to a numpy array
    groups = np.asarray(groups)    
  
    # Loop through unique groups and normalize
    xmean = np.nan * np.zeros(x.shape)
    for group in ugroups:
        idx = groups == group
        if idx.sum() > 0:
            norm = 1.0 * (~np.isnan(x[idx,...])).sum(0)
            xmean[idx,...] = np.nansum(x[idx,...], axis=0) / norm
            
    return xmean

def group_median(x, groups):
    """
    Median with groups along axis=0.
    
    Parameters
    ----------
    x : ndarray
        Input data.
    groups : list
        List of group membership of each element along axis=0.
        
    Returns
    -------
    idx : ndarray
        The group median of the data along axis 0.

    """

    # Find set of unique groups
    ugroups = unique_group(groups)
    
    # Convert groups to a numpy array
    groups = np.asarray(groups)    
  
    # Loop through unique groups and normalize
    xmedian = np.nan * np.zeros(x.shape)
    for group in ugroups:
        idx = groups == group
        if idx.sum() > 0:
            xmedian[idx,...] = nanmedian(x[idx,...])
            
    return xmedian
    
def unique_group(groups):
    """Find unique groups in list not including None."""    
    ugroups = set(groups)
    ugroups -= set((None,))
    ugroups = list(ugroups)
    ugroups.sort()
    return ugroups    
    
# Normalize functions -------------------------------------------------------

def geometric_mean(x, axis=1, check_for_greater_than_zero=True):
    """
    Return the geometric mean of matrix x along axis, ignore NaNs.
    
    Raise an exception if any element of x is zero or less.
    
    Notes
    -----
    The return array has the dimension so it can be broadcasted to 
    the original array and not the reduced dimension of np.mean
     
    """
    if (x <= 0).any() and check_for_greater_than_zero:
        msg = 'All elements of x (except NaNs) must be greater than zero.'
        raise ValueError, msg
    x = x.copy()
    m = np.isnan(x)
    x[m] = 1.0
    m = np.asarray(~m, np.float64)
    m = m.sum(axis)
    x = np.log(x).sum(axis)
    g = 1.0/m
    x = np.multiply(g, x)
    x = np.exp(x)
    idx = np.ones(x.shape)
    idx[m == 0] = np.nan
    x = np.multiply(x, idx)
    return np.expand_dims(x, axis) 

def movingsum(x, window, axis=-1, norm=False):
    """Moving sum optionally normalized for missing (NaN) data."""
    if window < 1:  
        raise ValueError, 'window must be at least 1'
    if window > x.shape[axis]:
        raise ValueError, 'Window is too big.'      
    m = np.isfinite(x) 
    x = 1.0 * x 
    x[m == 0] = 0
    csx = x.cumsum(axis)
    index1 = [slice(None)] * x.ndim 
    index1[axis] = slice(window - 1, None)
    index2 = [slice(None)] * x.ndim 
    index2[axis] = slice(None, -window) 
    msx = csx[index1]
    index3 = [slice(None)] * x.ndim
    index3[axis] = slice(1, None)
    msx[index3] = msx[index3] - csx[index2] 
    csm = m.cumsum(axis)     
    msm = csm[index1]
    msm[index3] = msm[index3] - csm[index2]  
    
    if norm:
        ms = 1.0 * window * msx / msm
    else:
        ms = msx
        ms[msm == 0] = np.nan

    initshape = list(x.shape)  
    initshape[axis] = window - 1
    nans = np.nan * np.zeros(initshape)
    ms = np.concatenate((nans, ms), axis) 
    return ms
  
def movingsum_forward(x, window, skip=0, axis=1, norm=False):
    """Movingsum in the forward direction skipping skip dates."""
    if axis == 0:
        x = x.T
    x = np.fliplr(x)
    nr, nc = x.shape
    if skip > nc:
        raise IndexError, 'Your skip is too large.'
    ms = movingsum(x, window, axis=1, norm=norm)
    ms = np.fliplr(ms)
    nans = np.nan * np.zeros((nr, skip))
    ms = np.concatenate((ms[:,skip:], nans), 1)  
    if axis == 0:
        ms = ms.T
    return ms

def movingrank(x, window, axis=1):
    """Moving rank (normalized to -1 and 1) of a given window along axis.

    Normalized for missing (NaN) data.
    A data point with NaN data is returned as NaN
    If a window is all NaNs except last, this is returned as NaN
    """
    if window > x.shape[axis]:
        raise ValueError, 'Window is too big.'
    if window < 2:
        raise ValueError, 'Window is too small.'
    if axis == 0:
        x = x.T
    nr, nt = x.shape
    mr = np.nan * np.zeros((nr,nt))        
    for i in xrange(window-1,nt): 
        mr[:,i] = np.squeeze(lastrank(x[:,(i-window+1):(i+1)]))  #check i:i+1      
    if axis == 0:
        mr = mr.T
    return mr
   
def lastrank(x):
    "Rank of last column only"
    g = (x[:,-1:] > x).sum(1)
    e = (x[:,-1:] == x).sum(1)
    n = np.isfinite(x).sum(1)
    r = (g + g + e - 1.0) / 2.0
    r = r / (n - 1.0)
    r = 2.0 * (r - 0.5)
    r[~np.isfinite(x[:,-1])] = np.nan
    return r[:,None]

def lastrank_decay(x, decay):
    "Exponential decay rank of last column only"
    assert decay >= 0, 'Min decay is 0.'
    x = np.atleast_2d(x) # so that indexing and axis work correctly
    nt = x.shape[1]
    w = nt - np.ones((1,nt)).cumsum(1)
    w = np.exp(-decay * w)
    w = nt * w / w.sum()
    # inner or dot ?
    g = np.inner((x[:,-1:] > x), w).sum(1)
    e = np.inner((x[:,-1:] == x), w).sum(1)
    n = np.inner(np.isfinite(x), w).sum(1)
    r = (g + g + e - w[0,-1]) / 2.0
    r = r / (n - w[0,-1])
    r = 2.0 * (r - 0.5)
    r[~np.isfinite(x[:,-1])] = np.nan
    return r[:,None]

def ranking(x, axis=0, norm='-1,1', ties=True):
    """
    Normalized ranking treating NaN as missing and average ties by default.
    
    Parameters
    ----------
    x : ndarray
        Data to be ranked.
    axis : int, optional
        Axis to rank over. Default axis is 0.
    norm: str
        A string that specifies the normalization:
        '0,N-1'     Zero to N-1 ranking
        '-1,1'      Scale zero to N-1 ranking to be between -1 and 1
        'gaussian'  Rank data then scale to a Gaussian distribution
    ties: bool
        If two elements of `x` have the same value then they will be ranked
        by their order in the array (False). If `ties` is set to True
        (default), then the ranks are averaged.
        
    Returns
    -------
    idx : ndarray
        The ranked data.The dtype of the output is always np.float even if
        the dtype of the input is int.
    
    Notes
    ----
    If there is only one non-NaN value along the given axis, then that value
    is set to the midpoint of the specified normalization method. For example,
    if the input is array([1.0, nan]), then 1.0 is set to zero for the '-1,1'
    and 'gaussian' normalizations and is set to 0.5 (mean of 0 and 1) for the
    '0,N-1' normalization.
    
    For '0,N-1' normalization, note that N is x.shape[axis] even in there are
    NaNs. That ensures that when ranking along the columns of a 2d array, for
    example, the output will have the same min and max along all columns.
    
    """
    ax = axis
    masknan = np.isnan(x)
    countnan = np.expand_dims(masknan.sum(ax), ax)
    countnotnan = x.shape[ax] - countnan
    if not ties:
        maskinf = np.isinf(x)
        adj = masknan.cumsum(ax)
        if masknan.any():
            x = x.copy()
            x[masknan] = np.inf
        idxraw = x.argsort(ax).argsort(ax)
        idx = idxraw.astype(float)
        idx[masknan] = np.nan
        idx[maskinf] -= adj[maskinf]
    else:
        rank1d = rankdata # Note: stats.rankdata starts ranks at 1
        idx = np.nan * np.ones(x.shape)
        itshape = list(x.shape)
        itshape.pop(ax)
        for ij in np.ndindex(*itshape):
            ijslice = list(ij[:ax]) + [slice(None)] + list(ij[ax:])
            x1d = x[ijslice].astype(float)
            mask1d = ~np.isnan(x1d)
            x1d[mask1d] = rank1d(x1d[mask1d]) - 1
            idx[ijslice] = x1d
    if norm == '-1,1':
        idx /= (countnotnan - 1)
        idx *= 2
        idx -= 1
        middle = 0.0
    elif norm == '0,N-1':
        idx *= (1.0 * (x.shape[ax] - 1) / (countnotnan - 1))
        middle = (idx.shape[ax] + 1.0) / 2.0 - 1.0
    elif norm == 'gaussian':
        try:
            from scipy.special import ndtri
        except ImportError:
            raise ImportError, 'SciPy required for gaussian normalization.'   
        idx *= (1.0 * (x.shape[ax] - 1) / (countnotnan - 1))
        idx = ndtri((idx + 1.0) / (x.shape[ax] + 1.0))
        middle = 0.0
    else:
        msg = "norm must be '-1,1', '0,N-1', or 'gaussian'."
        raise ValueError(msg)
    idx[(countnotnan==1)*(~masknan)] = middle
    return idx

def fillforward_partially(x, n):
    "Fill missing values (NaN) with most recent non-missing values if recent."
    y = np.asarray(x.copy())
    fidx = np.isfinite(y)
    recent = np.nan * np.ones(y.shape[0])  
    count = np.nan * np.ones(y.shape[0])          
    for i in xrange(y.shape[1]):
        idx = (i - count) > n
        recent[idx] = np.nan
        idx = ~fidx[:,i]
        y[idx, i] = recent[idx]
        idx = fidx[:,i]
        count[idx] = i
        recent[idx] = y[idx,i]
    return y

def quantile(x, q):
    """
    Convert elements in each column to integers between 1 and q then normalize.
    
    Result is normalized to -1, 1.
    
    Parameters
    ----------
    x : array_like, 1d or 2d
    q : int
        quantile between 2 and number of elements in first axis (x.shape[0])
    """
    assert q > 1, 'q must be greater than one.'
    assert q <= x.shape[0], 'q must be less than or equal to the number of rows in x.'
    y = np.nan * np.asarray(x)
    for i in xrange(x.shape[1]):
        xi = x[:,i]
        idx = np.where(np.isfinite(xi))[0]
        xi = xi[idx,:]
        nx = idx.size
        if nx:
            jdx = xi.argsort(axis=0).argsort(axis=0)
            mdx = np.nan * jdx
            kdx = 1.0 * (nx - 1) / (q) * np.ones((q,1))
            kdx = kdx.cumsum(axis=0)
            kdx = np.concatenate((-1*np.ones((1,kdx.shape[1])), kdx), 0)
            kdx[-1,0] = nx
            for j in xrange(1, q+1):
                mdx[(jdx > kdx[j-1]) & (jdx <= kdx[j]),:] = j
            y[idx,i] = mdx
    y = np.asarray(y)
    y = y - 1.0
    y = 1.0 * y / (q - 1.0)
    y = 2.0 * (y - 0.5)
    return y 
   
# Calc functions -----------------------------------------------------------

def covMissing(R):
    """
    Covariance matrix adjusted for missing returns.

    covMissing returns the covariance matrix adjusted for missing returns.
    R (NxT) is log stock returns; missing returns are NaN.

    Note the mean of each row of R is assumed to be zero. So returns are not
    demeaned and the covariance is normalized by T not T-1.
    
    
    Notes
    -----
    
    equivalence to using numpy masked array function
    l7.demean(axis=1).cov().x -np.ma.cov(np.ma.fix_invalid(x7), bias=1).data
    
    """
    mask = np.isnan(R)
    R[mask] = 0
    mask = np.asarray(mask, np.float64)
    mask = 1 - mask # Change meaning of missing matrix to present matrix  

    normalization = np.dot(mask, mask.T)

    if np.any(normalization < 2):
        raise ValueError, 'covMissing: not enough observations'

    C = np.dot(R, R.T) / normalization

    return C   

# Random functions ----------------------------------------------------------

def shuffle(x, axis=0, rs=None):
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

# NaN functions -------------------------------------------------------------

def nans(shape, dtype=float):
    "Works like ones and zeros except that the fill value is NaN"
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

