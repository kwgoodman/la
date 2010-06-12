"Missing value utilities"

import numpy as np


def nans(shape, dtype=float):
    """
    Works like ones and zeros except that the fill value is NaN (by default)
    
    The fill value is determined by dtype. The fill value is NaN for floats,
    '' for str, and None for object. A TypeError is raised for int dtype.
    
    Parameters
    ----------
    shape : {tuple, int}
        The desired shape pf the output
    dtype : various
        The desired dtype of output. Typically values are float, str, object.
        Integer dtype is not allowed.
        
    Returns
    -------
    a : ndarray
        An array of shape `shape` and dtype `dtype`. The fill value is NaN
        for floats, '' for str, and None for object.
        
    Examples
    --------            
    >>> nans(2)
    array([ NaN,  NaN])
    >>> nans((2,2))
    array([[ NaN,  NaN],
           [ NaN,  NaN]])
    >>> nans(2, str)
    array(['', ''], 
          dtype='|S1')
    >>> nans(2, object)
    array([None, None], dtype=object)
    >>> import datetime
    >>> nans(2, datetime.date)
    array([None, None], dtype=object)
    
    """
    a = np.empty(shape, dtype)
    # Check for str first since numpy considers str to be a scalar subtype.
    # No check for object dtype since empty already fills those with None
    if np.issubdtype(a.dtype, str):
        a.fill('')      
    elif np.issctype(a.dtype):
        if issubclass(a.dtype.type, np.inexact):
            a.fill(np.nan)
        else:
            # int dtype can't be filled with NaN
            msg = 'Inexact scalar dtype, such as float, needed for NaN '
            msg += 'fill value.'
            raise TypeError, msg     
    return a

