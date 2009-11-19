"Functions that work with numpy arrays."

import numpy as np

import numpy.matlib as M
from decorator import decorator

M.seterr(divide='ignore')
M.seterr(invalid='ignore')

@decorator
def wraptomatrix1(func, *args, **kwds):
    '''wrapping function to convert first argument to matrix
    for use as a decorator
    '''
    #new = asarray(a)
    #print args
    a = args[0]
    new = np.asmatrix(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    if len(args)>1:
        res = func(new, *args[1:], **kwds)
    else:
        res = func(new, **kwds)
    if np.isscalar(res):
        return res
    elif type(res) is tuple:
        return map(wrap, res)
    else:
        return wrap(res)

@decorator
def wraptoarray1(func, *args, **kwds):
    '''wrapping function to convert first argument to matrix
    for use as a decorator
    '''
    #new = asarray(a)
    #print args
    a = args[0]
    new = np.asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    if len(args)>1:
        res = func(new, *args[1:], **kwds)
    else:
        res = func(new, **kwds)
    if np.isscalar(res):
        return res
    elif type(res) is tuple:
        return map(wrap, res)
    else:
        return wrap(res)

# Sector functions ----------------------------------------------------------

@wraptoarray1
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

@wraptoarray1
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
            norm = 1.0 * (~np.isnan(x[idx,:])).sum(0)
            xmean[idx,:] = np.nansum(x[idx,:], axis=0) / norm
    return xmean

@wraptomatrix1  
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
            xmedian[idx,:] = nanmedian(x[idx,:])
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
    this always returns array, not a matrix
    """
    if type(sectors) is not list:
        raise TypeError, 'Sector input must be a list'
    usectors = unique_sector(sectors)
#    dummy = np.zeros((len(sectors), len(usectors)))
    sectors = np.asarray(sectors, dtype=object).T
#    for i, sec in enumerate(usectors):
#        dummy[:,i] = sectors == sec
    dummy = (sectors[:,None] == usectors).astype(float)
    return dummy, usectors    
    
def unique_sector(sectors):
    """Find unique sector list not including None."""    
    usectors = set(sectors)
    usectors = [z for z in usectors if z is not None]
    usectors.sort()
    return usectors
    
# Normalize functions -------------------------------------------------------

@wraptoarray1
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
    m = np.asarray(~m, M.float64)
    m = m.sum(axis)
    x = np.log(x).sum(axis)
    g = 1.0/m
    x = np.multiply(g, x)
    x = np.exp(x)
    idx = np.ones(x.shape)
    idx[m == 0] = np.nan
    x = np.multiply(x, idx)
    return np.expand_dims(x, axis) 

@wraptoarray1
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
    index3[axis] = slice(1, None) # form slice(1,None)
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
    
def movingsum_old(x, window, axis=1, norm=False, q=1.0):
    """
    Moving sum optionally normalized for missing (NaN) data.
    
    This is the old matrix version. Unlike the new array version the output is
    the same shape as the input. I've kept for use in movingsum_forward until
    movingsum_forward is made to work on with the new movingsum function.
    """
    if norm is False and q != 1.0:
        raise ValueError, 'Since norm is False, q will be ignored.'
    if window > x.shape[axis]:
        raise ValueError, 'Window is too big.' 
    if axis == 0:
        x = x.T 
    x = 1.0 * x       
    nr = x.shape[0]  
    x = M.concatenate((M.zeros((nr,1)), x), 1)  
    m = M.asmatrix(~M.isnan(x), M.float64)
    x[m == 0] = 0
    csx = x.cumsum(1)  
    msx = csx[:,window:] - csx[:,:-window]
    csm = m.cumsum(1)     
    msm = csm[:,window:] - csm[:,:-window]    
    if norm:
        ms = M.multiply(M.power(window / msm, q), msx)
    else:
        ms = msx
        ms[msm == 0] = M.nan          
    x = x[:,1:]
    nans = M.nan * M.zeros((nr, window-1))
    ms = M.concatenate((nans, ms), 1) 
    if axis == 0:
        ms = ms.T  
    return ms    

@wraptoarray1    
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

@wraptoarray1  
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
        mr[:,i] = lastrank(x[:,(i-window+1):(i+1)])  #check i:i+1      
    if axis == 0:
        mr = mr.T
    return mr

@wraptoarray1    
def lastrank(x):
    "Rank of last column only"
    g = (x[:,-1:] > x).sum(1)
    e = (x[:,-1:] == x).sum(1)
    n = np.isfinite(x).sum(1)#[:,None]
    r = (g + g + e - 1.0) / 2.0
    r = r / (n - 1.0)
    r = 2.0 * (r - 0.5)
    r[~np.isfinite(x[:,-1])] = np.nan
    #raise ValueError
    return r

@wraptomatrix1    
def lastrank_decay(x, decay):
    "Exponential decay rank of last column only"
    assert decay >= 0, 'Min decay is 0.'
    nt = x.shape[1]
    w = nt - M.ones((1,nt)).cumsum(1)
    w = M.exp(-decay * w)
    w = nt * w / w.sum()
    g = M.multiply((x[:,-1] > x), w).sum(1)
    e = M.multiply((x[:,-1] == x), w).sum(1)
    n = M.multiply(M.isfinite(x), w).sum(1)
    r = (g + g + e - w[0,-1]) / 2.0
    r = r / (n - w[0,-1])
    r = 2.0 * (r - 0.5)
    r[~M.isfinite(x[:,-1])] = M.nan
    return r       

@wraptomatrix1
def ranking_1N(x, axis=0):
    """Rank elements of matrix x, ignore NaNs."""
    if axis not in (0,1):
        ValueError, 'axis(=%d) out of bounds'
    if (~M.isnan(x)).all():
        z = x.argsort(axis).argsort(axis)
    else:
        if axis == 1:
            x = x.T
        ax = 0      
        sannanidx = M.isnan(x).sum(ax) == 0
        nanidx = ~sannanidx
        sannanidx = M.where(sannanidx.A)[1]
        nanidx = M.where(nanidx.A)[1]
        z = M.nan * M.zeros(x.shape)
        z[:, sannanidx] = x[:, sannanidx].argsort(ax).argsort(ax)
        middle = (x.shape[ax] + 1.0)/2.0 - 1.0 
        nax = z.shape[ax]    
        for i in nanidx:
            idx = M.where(~M.isnan(x[:, i].A))[0]
            idx = M.asmatrix(idx).T
            if len(idx) == 0:
                z[idx, i] = M.nan
            elif len(idx) == 1:
                z[idx, i] = middle
            else:                    
                zi = x[idx, i].argsort(ax).argsort(ax)
                zmin = M.nanmin(zi, ax)         
                zmax = M.nanmax(zi, ax)
                zi = (nax - 1.0) * (zi - zmin) / (zmax - zmin)                 
                z[idx, i] = zi
        if axis == 1:
            z = z.T    
    return z

@wraptomatrix1    
def ranking_norm(x, axis=0):
    """Same as ranking_1N but normalize range to -1 to 1."""  
    xnanidx = M.isnan(x) 
    z = 1.0 * ranking_1N(x, axis)  
    zmin = M.nanmin(z, axis)
    zmax = M.nanmax(z, axis) 
    if type(zmin) == float:
        zmin = M.matrix(zmin)
    if type(zmin) == float:        
        zmax = M.matrix(zmax)
    zscaled = 2.0 * (z - zmin) / (zmax - zmin) - 1.0    
    idx = zmin == zmax   
    if axis == 0:
        idx = M.where(idx.A)[1]    
        zscaled[:, idx] = 0.0
    elif axis == 1:
        idx = M.where(idx.A)[0]
        idx = M.asmatrix(idx).T 
        zscaled[idx, :] = 0.0 
    else:
        raise ValueError, 'axis must be 0 or 1.'
    zscaled[xnanidx] = M.nan               
    return zscaled           

@wraptomatrix1
def ranking(x, axis=0):
    """Same as ranking_norm but break ties.
    
    Uses a brute force method---slow.
    """

    where = M.where
    if axis == 1:
        x = x.T
    y = ranking_norm(x, axis=0)
    sx = x.copy()
    sx[M.isnan(sx)] = M.inf
    sx.sort(axis=0)
    dsx = M.diff(sx, axis=0)
    idx = (dsx == 0).sum(axis=0)
    idx = M.where(idx.A)[1]
    for i in idx:
        yi = y[:,i]
        xi = x[:,i].A
        ux = M.unique(xi)
        for uxi in ux:
            jdx = where(xi == uxi)[0]
            y[jdx,i] = yi[jdx].mean()
    if axis == 1:
        y = y.T
    return y

@wraptoarray1  
def fillforward_partially(x, n):
    "Fill missing values (NaN) with most recent non-missing values if recent."
    y = np.asarray(x.copy())
    fidx = M.isfinite(y)
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
    return y #np.asmatrix(y) 

@wraptoarray1    
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
        #idx = np.asarray(idx[:,None])
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

@wraptoarray1
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
    #TODO: no test failures with array, but needs to be checked (returns matrix?) DONE

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

@wraptoarray1
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

@wraptoarray1
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

@wraptoarray1
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
