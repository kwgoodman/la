"Missing value utilities"

import numpy as np
import la


def nans(shape, dtype=float):
    """
    Works like ones and zeros except that the fill value is NaN (by default)
    
    The fill value is determined by dtype. The fill value is NaN for floats,
    '' for str, and None for object. A TypeError is raised for int dtype.
    
    Parameters
    ----------
    shape : {tuple, int}
        The desired shape pf the output
    dtype : {float-like, str, object}, optional
        The desired dtype of output. Typically values are float (default),
        str, object. Other dtypes, such as int and bool, raise a
        TypeErrorInteger.
        
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
    if issubclass(a.dtype.type, np.inexact):
        a.fill(np.nan)
        return a
    else:
        if np.issubdtype(a.dtype, str):
            a.fill('')
            return a
        elif a.dtype == np.bool_:
            msg = 'Missing values are not supported for bool dtype.'
            raise TypeError, msg            
        elif a.dtype == object:
            a.fill(None)
            return a       
    # int dtype can't be filled with NaN
    msg = 'Inexact scalar dtype, such as float, needed for NaN '
    msg += 'fill value.'
    raise TypeError, msg

def missing_marker(data):
    """
    Missing value marker, which is based on dtype, for the given data.
    
    Parameters
    ----------
    lar : {larry, ndarray}
        Input data can be a larry or a Numpy array.
        
    Returns
    -------
        The missing value marker used by larrys with the dtype of `lar`. If
        no missing values are used, such as with int larrys, then
        NotImplemented is returned
        
    Examples
    --------
    >>> missing_marker(larry([1]))
    NotImplemented
    >>> missing_marker(larry([1.0]))
    nan
    >>> missing_marker(larry([True]))
    NotImplemented
    >>> missing_marker(larry(['a']))
    ''
    >>> import datetime
    >>> missing_marker(la.larry([datetime.date(2011,1,1)])) # None is returned
    >>> missing_marker(la.larry([datetime.date(2011,1,1)])) == None
    True
            
    """
    dtype = data.dtype
    if issubclass(dtype.type, np.inexact):
        return np.nan
    else:
        if np.issubdtype(dtype, str):
            return ''
        elif dtype == np.bool_:
            return NotImplemented     
        elif dtype == object:
            return None
        else:              
            return NotImplemented

def ismissing(data):
    """
    A bool Numpy array with element-wise marking of missing values.
    
    Parameters
    ----------
    lar : {larry, ndarray}
        Input data can be a larry or a Numpy array.
        
    Returns
    -------
    arr : Numpy ndarray
        The result is a bool Numpy array that contains the value True if the
        corresponding element in `lar` is missing; otherwise False. The shape
        of `arr` is the same as `lar`.

    Examples
    --------         
    >>> ismissing(la.larry([1]))
    array([False], dtype=bool)
    >>> ismissing(la.larry([1.0]))
    array([False], dtype=bool)
    >>> ismissing(la.larry(['str']))
    array([False], dtype=bool)
    >>> ismissing(la.larry([None]))
    array([ True], dtype=bool)
    >>> import datetime
    >>> ismissing(la.larry([datetime.date(2011,1,1)]))
    array([False], dtype=bool)
    >>> ismissing(la.larry([la.nan]))
    array([ True], dtype=bool)
    >>> ismissing(la.larry([la.nan, 1.0]))
    array([ True, False], dtype=bool)
    
    """
    mm = missing_marker(data)
    if mm == NotImplemented:
        arr = np.empty(data.shape, dtype=bool)
        arr.fill(False)
        return arr
    else:
        if isinstance(data, la.larry):
            x = data.x
        else:
            x = data        
        if mm != mm:
            return np.isnan(x)
        else:
            return x == [mm]                    
