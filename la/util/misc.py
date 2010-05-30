"Misc utility functions."

import random
import string
from itertools import izip

import numpy as np

try:
    # The c version is faster...
    from la.util.clistmap import listmap
except ImportError:
    # ...but perhaps it did not compile when you built the la package? So
    # we'll use the python version. If you are unsure which version you are
    # using, the doc string will tell you.
    def listmap(list1, list2):
        """
        list of indices idx such that [list1[i] for i in idx] is list2.
        
        This function is equivalent to idx = map(list1.index, list2) except
        that it is O(n) instead of O(n^2).
        
        All elements in list2 must be in list1 
        
        NOTE: This is the slower python version of the function; there is a 
        faster C version that setup.py will automatically try to compile at
        build (setup.py) time.   
        """
        list1map = dict(izip(list1, xrange(len(list1))))
        idx = [list1map[i] for i in list2]
        return idx
        
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
    
def isstring(s):
    "Return True if input is a str or np.string_."
    return issubclass(type(s), str)      
    
def list2index(L):
    "Convert a list to a unique list and the corresponding indices."
    uL = sorted(set(L))
    idx = dict((y, x) for x, y in enumerate(uL))
    return [idx[x] for x in L], uL  

def fromlists(xs, labels):
    """
    Convert list of values and list of label tuples to larry label and x.
    
    Parameters
    ----------
    xs : {tuple, list}
        A tuple or list of values that will be converted to a Numpy array in
        an ordering that is determined by `labels`.
    labels : list of tuples
        A list of tuples, one tuple per dimension of the output array, that
        give the label (coordinates) of the corresponding elements in `xs`.
        
    Returns
    -------
    x : Numpy ndarray
        A Numpy array with order and shape given by `labels`.
    label : list
        The label that corresponds to `x`.
        
    Examples
    --------
    >>> from la.util.misc import fromlists
    >>> xs = [1, 2, 3, 4]
    >>> labels = [('a', 'a', 'b', 'b'), ('a', 'b', 'a', 'b')]
    >>> x, label = fromlists(xs, labels)
    >>> x
    array([[ 1.,  2.],
           [ 3.,  4.]])
    >>> label
    [['a', 'b'], ['a', 'b']]
    >>> x
    array([[ 1.,  2.],
           [ 3.,  4.]])                    
    
    """
    if (len(xs) == 0) and (len(labels) == 0):
        x = np.array([])
        label = None
    else:
        shape = []
        index = []
        label = []
        for lab in labels:
            labelidx, label_unique = list2index(lab)
            shape.append(len(label_unique))
            index.append(labelidx)
            label.append(label_unique)
        x = np.empty(shape)
        x.fill(np.nan)
        x[index] = xs 
    return x, label 

