"Functions that work with numpy arrays."

import numpy as np


# Sector functions ----------------------------------------------------------

def sector_rank(x, sectors):
    """Rank normalize x within each sector to be between -1 and 1."""
  
    # Find set of unique sectors
    usectors = unique_sector(sectors)
    
    # Convert sectors to a numpy array
    sectors = np.asarray(sectors, dtype=object)
  
    # Loop through unique sectors and normalize
    xnorm = np.nan * np.zeros(x.shape)
    for sec in usectors:
        idx = sectors == sec
        xnorm[idx,:] = ranking(x[idx,:], axis=0)    
    return xnorm

def sector_mean(x, sectors):
    """Sector mean."""

    # Find set of unique sectors
    usectors = unique_sector(sectors)
    
    # Convert sectors to a numpy array
    sectors = np.asarray(sectors, dtype=object)    
  
    # Loop through unique sectors and normalize
    # this will be slow if there are many sectors
    xmean = np.nan * np.zeros(x.shape)
    for sec in usectors:
        idx = sectors == sec
        if idx.sum() > 0:
            norm = 1.0 * (~np.isnan(x[idx,...])).sum(0)
            xmean[idx,...] = np.nansum(x[idx,...], axis=0) / norm
    return xmean

def sector_median(x, sectors):
    """Sector median."""

    # Find set of unique sectors
    usectors = unique_sector(sectors)
    
    # Convert sectors to a numpy array
    sectors = np.asarray(sectors, dtype=object)    
  
    # Loop through unique sectors and normalize
    xmedian = np.nan * np.zeros(x.shape)
    for sec in usectors:
        idx = sectors == sec
        if idx.sum() > 0:
            xmedian[idx,...] = nanmedian(x[idx,...])
    return xmedian
    
def sector_dummy(sectors):
    """Create a NxS sector dummy matrix of N stocks and S unique sectors.
    
    Parameters
    ----------
    sectors : list
    
    Returns
    -------
    dummy : 2d array
    usesectors : list
        unique sectors
    
    Notes
    -----
    this always returns array
    """
    if type(sectors) is not list:
        raise TypeError, 'Sector input must be a list'
    usectors = unique_sector(sectors)
    sectors = np.asarray(sectors, dtype=object)
    dummy = (sectors[:,None] == usectors).astype(float)
    return dummy, usectors    
    
def unique_sector(sectors):
    """Find unique sector list not including None."""    
    usectors = set(sectors)
    usectors = [z for z in usectors if z is not None]
    usectors.sort()
    return usectors
    
# Normalize functions -------------------------------------------------------

def geometric_mean(x, axis=1, check_for_greater_than_zero=True):
    """Return the geometric mean of matrix x along axis, ignore NaNs.
    
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
 
def ranking_1N(x, axis=0):
    return ranking(x, axis=axis, norm='0,N-1', ties=False)

def ranking_norm(x, axis=0):
    return ranking(x, axis=axis, norm='-1,1', ties=False)

def ranking(x, axis=0, norm='-1,1', ties=True):
    ax = axis
    masknan = ~np.isfinite(x)
    countnan = np.expand_dims(masknan.sum(ax), ax)
    countnotnan = x.shape[ax] - countnan
    if not ties:
        if masknan.any():
            x = x.copy()
            x[masknan] = np.inf
        idxraw = x.argsort(ax).argsort(ax)
        idx = idxraw.astype(float)
        idx[masknan] = np.nan
    else:
        from scipy import stats
        rank1d = stats.rankdata #stats.rankdata starts ranks at 1
        idx = np.nan * np.ones(x.shape)
        itshape = list(x.shape)
        itshape.pop(ax)
        for ij in np.ndindex(*itshape):
            ijslice = list(ij[:ax]) + [slice(None)] + list(ij[ax:])
            x1d = x[ijslice]
            mask1d = np.isfinite(x1d)
            x1d[mask1d] = rank1d(x1d[mask1d])-1  #stats.rankdata starts at 1
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
        from scipy.special import ndtri
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

# NaN functions -------------------------------------------------------------

# These functions where taken from scipy. Should we just import them from
# scipy? I was trying to avoid the dependency.

def nans(shape, dtype=float):
    "Works like ones and zeros except that the fill value is NaN"
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def nanmean(x, axis=0):
    """Compute the mean over the given axis ignoring nans.

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the mean is computed.

    :Results:
        m : float
            the mean."""
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0-np.sum(np.isnan(x),axis)*1.0/Norig

    x[np.isnan(x)] = 0
    return np.mean(x,axis)/factor


def nanstd(x, axis=0, bias=True):
    """Compute the standard deviation over the given axis ignoring nans

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the standard deviation is computed.
        bias : boolean
            If true, the biased (normalized by N) definition is used. If false,
            the unbiased is used (the default).

    :Results:
        s : float
            the standard deviation."""
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]

    Nnan = np.sum(np.isnan(x),axis)*1.0
    n = Norig - Nnan

    x[np.isnan(x)] = 0.
    m1 = np.sum(x,axis)/n

    # Kludge to subtract m1 from the correct axis
    if axis!=0:
        shape = np.arange(x.ndim).tolist()
        shape.remove(axis)
        shape.insert(0,axis)
        x = x.transpose(tuple(shape))
        d = (x-m1)**2.0
        shape = tuple(np.array(shape).argsort())
        d = d.transpose(shape)
    else:
        d = (x-m1)**2.0
    m2 = np.sum(d,axis)-(m1*m1)*Nnan
    if bias:
        m2c = m2 / n
    else:
        m2c = m2 / (n - 1.)
    return np.sqrt(m2c)

def _nanmedian(arr1d):  # This only works on 1d arrays
    """Private function for rank a arrays. Compute the median ignoring Nan.

    :Parameters:
        arr1d : rank 1 ndarray
            input array

    :Results:
        m : float
            the median."""
    cond = 1-np.isnan(arr1d)
    x = np.sort(np.compress(cond,arr1d,axis=-1))
    if x.size == 0:
        return np.nan
    return median(x)


def nanmedian(x, axis=0):
    """ Compute the median along the given axis ignoring nan values

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the median is computed.

    :Results:
        m : float
            the median."""
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    return np.apply_along_axis(_nanmedian,axis,x)

def median(a, axis=0):
    # fixme: This would be redundant with numpy.median() except that the latter
    # does not deal with arbitrary axes.
    """Returns the median of the passed array along the given axis.

    If there is an even number of entries, the mean of the
    2 middle values is returned.

    Parameters
    ----------
    a : array
    axis=0 : int

    Returns
    -------
    The median of each remaining axis, or of all of the values in the array
    if axis is None.
    """
    a, axis = _chk_asarray(a, axis)
    if axis != 0:
        a = np.rollaxis(a, axis, 0)
    return np.median(a)

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis                
