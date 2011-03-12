"NaN-aware numpy array functions for group by operations."

import numpy as np
import bottleneck as bn
from la.farray import ranking

__all__ = ['group_ranking', 'group_mean', 'group_median', 'unique_group']


def group_ranking(x, groups, norm='-1,1', axis=0):
    """
    Ranking within groups along axis.
    
    Parameters
    ----------
    x : ndarray
        Data to be ranked.
    groups : list
        List of group membership of each element along axis=0.
    norm : str
        A string that specifies the normalization:
        '0,N-1'     Zero to N-1 ranking
        '-1,1'      Scale zero to N-1 ranking to be between -1 and 1
        'gaussian'  Rank data then scale to a Gaussian distribution
    axis : int, {default: 0}
        axis along which the ranking is calculated
        
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
        idxall = [slice(None)] * x.ndim
        idxall[axis] = idx
        xnorm[idxall] = ranking(x[idxall], axis=axis, norm=norm) 
           
    return xnorm

def group_mean(x, groups, axis=0):
    """
    Mean with groups along an axis.
    
    Parameters
    ----------
    x : ndarray
        Input data.
    groups : list
        List of group membership of each element along the axis.
    axis : int, {default: 0}
        axis along which the mean is calculated
        
    Returns
    -------
    idx : ndarray
        An array with the same shape as the input array where every element is
        replaced by the group mean along the given axis.

    """

    # Find set of unique groups
    ugroups = unique_group(groups)
    
    # Convert groups to a numpy array
    groups = np.asarray(groups)    
  
    # Loop through unique groups and normalize
    xmean = np.nan * np.zeros(x.shape)    
    for group in ugroups:
        idx = groups == group
        idxall = [slice(None)] * x.ndim
        idxall[axis] = idx
        if idx.sum() > 0:
            norm = 1.0 * (~np.isnan(x[idxall])).sum(axis)
            ns = np.nansum(x[idxall], axis=axis) / norm
            xmean[idxall] = np.expand_dims(ns, axis)
            
    return xmean

def group_median(x, groups, axis=0):
    """
    Median with groups along an axis.
    
    Parameters
    ----------
    x : ndarray
        Input data.
    groups : list
        List of group membership of each element along the given axis.
    axis : int, {default: 0}
        axis along which the ranking is calculated.
        
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
        idxall = [slice(None)] * x.ndim
        idxall[axis] = idx
        if idx.sum() > 0:
            ns = bn.nanmedian(x[idxall], axis=axis)
            xmedian[idxall] = np.expand_dims(ns, axis)
            
    return xmedian
    
def unique_group(groups):
    """Find unique groups in list not including None."""    
    ugroups = set(groups)
    ugroups -= set((None,))
    ugroups = list(ugroups)
    ugroups.sort()
    return ugroups    
