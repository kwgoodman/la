
import random
import string

import numpy as np

C = string.letters + string.digits

def randstring(n):
    "Random characters string selected from lower, upper letters and digits."
    s = []
    nc = len(C) - 1
    for i in range(n):
        s.append(C[random.randint(0, nc)])
    return ''.join(s) 
    
def flattenlabel(label, order='C'):
    """
    Flatten label in row-major order 'C' (default) or column-major order 'F'.
    
    Code taken (but modified) from http://code.activestate.com/recipes/496807
    
    """
    if order not in ('C', 'F'):
        raise ValueError, "order must be 'C' or 'F'"
    label = list(label)
    if order == 'C':
        label = label[::-1]
    idx = [[]]
    for x in label:
        t = []
        for y in x:
            for i in idx:
                t.append(i+[y])
        idx = t
    if order == 'C':
         idx = [i[::-1] for i in idx]
    idx = [tuple(i) for i in idx]     
    return [idx]
    
def isint(x):
    """Returns True if input is an integer; False otherwise.
    
    Parameters
    ----------
    x : any
        Input can be of any type.
        
    Returns
    -------
    y : bool
        True is `x` is an integer, False otherwise.
        
    Notes
    -----
    A table showing what isint returns for various types:
    
    ========== =======
       type     isint
    ========== =======
    int          True
    np.int32     True
    np.int64     True
    float        False
    np.float32   False
    np.float64   False
    complex      False
    str          False
    bool         False
    
    Examples
    --------
    >>> isint(1)
    True
    >>> isint(1.1)
    False
    >>> isint(True)
    False
    >>> isint(1j)
    False
    >>> isint('a')
    False     
                
    """
    return np.issubdtype(type(x), int)
    
def isfloat(x):
    """Returns True if input is a float; False otherwise.
    
    Parameters
    ----------
    x : any
        Input can be of any type.
        
    Returns
    -------
    y : bool
        True is `x` is a float, False otherwise.
        
    Notes
    -----
    A table showing what isfloat returns for various types:
    
    ========== =======
       type     isint
    ========== =======
    int          False
    np.int32     False
    np.int64     False
    float        True
    np.float32   True
    np.float64   True
    complex      False
    str          False
    bool         False
    
    Examples
    --------
    >>> isfloat(1)
    False
    >>> isfloat(1.1)
    True
    >>> isfloat(True)
    False
    >>> isfloat(1.1j)
    False
    >>> isfloat('a')
    False   
                
    """
    return np.issubdtype(type(x), float)
    
def isscalar(x):
    """Returns True if input is a scalar; False otherwise.
    
    Parameters
    ----------
    x : any
        Input can be of any type.
        
    Returns
    -------
    y : bool
        True is `x` is a scalar, False otherwise.
        
    Notes
    -----
    A table showing what isscalar returns for various types:
    
    ========== =======
       type     isint
    ========== =======
    int          True
    np.int32     True
    np.int64     True
    float        True
    np.float32   True
    np.float64   True
    complex      False
    str          False
    bool         False
    
    Examples
    --------
    >>> isscalar(1)
    True
    >>> isscalar(1.1)
    True
    >>> isscalar(True)
    False
    >>> isscalar(1j)
    False
    >>> isscalar('a')
    False 
                
    """
    return isfloat(x) or isint(x)                 
