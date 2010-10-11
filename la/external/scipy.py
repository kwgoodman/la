"""
The functions in this module were copied from SciPy to avoid making la
depend on SciPy. See the la LICENSE file for the SciPy license.

In the SciPy project, these functions can be found in scipy/stats/stats.py

One change was made to the SciPy version of nanstd. The default for nanstd
was changed from bias=False (N-1 normalization) to bias=False (N
normalization). That makes it match the defaults for np.std and scipy.std.

nanmedian has been modifed. See http://projects.scipy.org/scipy/ticket/1098

"""

import numpy as np


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis

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

# Three changes were made to the SciPy version of nanstd:
# 1: The default for nanstd was changed from bias=False (N-1 normalization)
#    to bias=False (N normalization). That makes it match the defaults for
#    np.std and scipy.std.
# 2: array was changed to np.array.
# 3: Bug fix to allow negative axis
#    http://projects.scipy.org/scipy/ticket/1161
def nanstd(x, axis=0, bias=True):
    """Compute the standard deviation over the given axis ignoring nans

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the standard deviation is computed.
        bias : boolean
            If true, the biased (normalized by N, default) definition is used.
            If false, the unbiased (N-1) is used.

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

    if axis:
        d = (x - np.expand_dims(m1, axis))**2.0
    else:
        d = (x - m1)**2.0

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
    return np.median(x)

# A change was made from the scipy version to handle scalar input an to 
# return a scalar when a 1d array is passed in or when axis is None.
def nanmedian(x, axis=0):
    """ Compute the median along the given axis ignoring nan values
    
    Note: This function has been modified from the original scipy function.
    See http://projects.scipy.org/scipy/ticket/1098    

    :Parameters:
        x : ndarray
            input array
        axis : int
            axis along which the median is computed.

    :Results:
        m : float
            the median."""           
    x, axis = _chk_asarray(x,axis)
    if x.ndim == 0:
        return np.float(x)
    x = x.copy()
    x = np.apply_along_axis(_nanmedian,axis,x)
    if x.ndim == 0:
        x = np.float(x)
    return x

def rankdata(a):
    """Ranks the data in a, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Example
    -------
    In [15]: stats.rankdata([0, 2, 2, 3])
    Out[15]: array([ 1. ,  2.5,  2.5,  4. ])

    Parameters
    ----------
    a : array
        This array is first flattened.

    Returns
    -------
    An array of length equal to the size of a, containing rank scores.
    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray
    
def fastsort(a):
    # fixme: the wording in the docstring is nonsense.
    """Sort an array and provide the argsort.

    Parameters
    ----------
    a : array

    Returns
    -------
    (sorted array,
     indices into the original array,
    )
    """
    it = np.argsort(a)
    as_ = a[it]
    return as_, it
            
