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
    def listmap(list1, list2, ignore_unmappable=False):
        """
        Indices that map one list onto another list.
        
        Parameters
        ----------
        list1 : list
            The list to map from.
        list2 : list
            The list to map to.
        ignore_unmappable : bool, optional
            If False (default) an element in `list2` that is not in `list1`
            will raise a KeyError. If True the unmappable elements will be
            ignored. The mapping is faster when `ignore_unmappable` is False.
        
        Returns
        -------
        idx : list
            If `ignore_unmappable` is False then returns a list of indices
            `idx` such that [list1[i] for i in idx] is `list2`. If
            `ignore_unmappable` is True then [list1[i] for i in idx] will not
            give `list2` if there are items in `list2` that are not in
            `list1`.

        Notes
        ----- 
        This is the slower python version of the function; there is a faster
        C version that setup.py will automatically try to compile at build
        (setup.py) time of the la package. You can also compile it by hand.
        
        Examples
        --------
        A simple mapping:

        >>> list1 = [1, 2, 3]
        >>> list2 = [3, 1, 2]
        >>> idx = listmap(list1, list2)
        >>> idx
        [2, 0, 1]
        >>> [list1[i] for i in idx] == list2
        True

        A KeyError is raised if an element in the second list is not in the
        first:

        >>> listmap(['a', 'b'], ['a', 'b', 'unmappable element'])
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "la/util/misc.py", line 55, in listmap
            idx = [list1map[i] for i in list2]        
        KeyError: 'unmappable element'
        
        If you wish to skip the unmappable element, then set
        `ignore_unmappable` to True:
        
        >>> listmap(['a', 'b'], ['a', 'b', 'unmappable element'], ignore_unmappable=True)
        [0, 1]
                  
        """
        list1map = dict(izip(list1, xrange(len(list1))))
        if ignore_unmappable:
            idx = [list1map[i] for i in list2 if i in list1map]
        else:
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

