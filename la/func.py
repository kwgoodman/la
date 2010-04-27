"Functions that operate on larrys."

import numpy as np

from la.deflarry import larry
from la.util.misc import flattenlabel
from la.afunc import covMissing


# Labels --------------------------------------------------------------------
    
def union(axis, *args):
    """
    Union of labels along specified axis.
    
    Parameters
    ----------
    axis : int
        The axis along which to take the union of the labels.
    args : larrys
        The larrys (separated by commas) over which the union is taken.
        
    Returns
    -------
    out : list
        A list containing the union of the labels.
        
    See Also
    --------
    la.intersection : Intersection of labels along specified axis.
    
    Examples
    --------            
    >>> import la
    >>> y1 = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
    >>> y2 = larry([[1, 2], [3, 4]], [['e', 'b'], ['f', 'd']])
    >>> la.union(0, y1, y2)
    ['a', 'b', 'e']
    >>> la.union(1, y1, y2)
    ['c', 'd', 'f']
    
    """
    rc = frozenset([])
    for arg in args:
        if isinstance(arg, larry):
            rc = frozenset(arg.label[axis]) | rc
        else:
            raise TypeError, 'One or more input is not a larry'
    rc = list(rc)
    rc.sort()
    return rc

def intersection(axis, *args):
    """
    Intersection of labels along specified axis.
    
    Parameters
    ----------
    axis : int
        The axis along which to take the intersection of the labels.
    args : larrys
        The larrys (separated by commas) over which the intersection is taken.
        
    Returns
    -------
    out : list
        A list containing the intersection of the labels.
        
    See Also
    --------
    la.union : Union of labels along specified axis.
    
    Examples
    --------            
    >>> import la
    >>> y1 = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
    >>> y2 = larry([[1, 2], [3, 4]], [['e', 'b'], ['f', 'd']])
    >>> la.intersection(0, y1, y2)
    ['b']
    >>> la.intersection(1, y1, y2)
    ['d']
    
    """
    rc = frozenset(args[0].label[axis])
    for i in xrange(1, len(args)):
        arg = args[i]
        if isinstance(arg, larry):
            rc = frozenset(arg.label[axis]) & rc
        else:
            raise TypeError, 'One or more input is not a larry'
    rc = list(rc)
    rc.sort()
    return rc

# Concatenating -------------------------------------------------------------
    
def stack(mode, **kwargs):
    """Stack 2d larrys to make a 3d larry.
    
    Parameters
    ----------
    mode : {'union', 'intersection'}
        Should the 3d larry be made from the union or intersection of all the
        rows and all the columns?
    kwargs : name=larry
        Variable length input listing the z axis name and larry. For example,
        stack('union', distance=x, temperature=y, pressure=z)
        
    Returns
    -------
    out : larry
        Returns a 3d larry.
        
    Raises
    ------
    ValueError
        If mode is not union or intersection or if any of the input larrys are
        not 2d.
        
    Examples
    --------
    >>> import la
    >>> y1 = la.larry([[1, 2], [3, 4]])
    >>> y2 = la.larry([[5, 6], [7, 8]])
    >>> la.stack('union', name1=y1, othername=y2)
    label_0
        othername
        name1
    label_1
        0
        1
    label_2
        0
        1
    x
    array([[[ 5.,  6.],
            [ 7.,  8.]],
    .
           [[ 1.,  2.],
            [ 3.,  4.]]])    
                        
    """
    if not np.all([kwargs[key].ndim == 2 for key in kwargs]):
        raise ValueError, 'All input larrys must be 2d'
    if mode == 'union':
        logic = union
    elif mode == 'intersection':
        logic = intersection
    else:    
        raise ValueError, 'mode must be union or intersection'   
    row = logic(0, *kwargs.values())
    col = logic(1, *kwargs.values())
    x = np.zeros((len(kwargs), len(row), len(col)))
    zlabel = []
    for i, key in enumerate(kwargs):
        y = kwargs[key]
        y = y.morph(row, 0)
        y = y.morph(col, 1)
        x[i] = y.x
        zlabel.append(key)
    label = [zlabel, row, col]
    return larry(x, label) 
    
def panel(lar):
    """
    Convert a 3d larry of shape (n, m, k) to a 2d larry of shape (m*k, n).
    
    Parameters
    ----------
    lar : 3d larry
        The input must be a 3d larry.
        
    Returns
    -------
    y : 2d larry
        If the input larry has shape (n, m, k) then a larry of shape (m*k, n)
        is returned.
        
    See Also
    --------
    la.larry.swapaxes : Swap the two specified axes.
    la.larry.flatten : Collapsing into one dimension.  
        
    Examples
    --------
    First make a 3d larry:
    
    >>> import numpy as np
    >>> y = larry(np.arange(24).reshape(2,3,4))
    >>> y
    label_0
        0
        1
    label_1
        0
        1
        2
    label_2
        0
        1
        2
        3
    x
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    .
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
            
    Then make a panel:        
            
    >>> la.func.panel(y)
    label_0
        (0, 0)
        (0, 1)
        (0, 2)
        ...
        (2, 1)
        (2, 2)
        (2, 3)
    label_1
        0
        1
    x
    array([[ 0, 12],
           [ 4, 16],
           [ 8, 20],
           [ 1, 13],
           [ 5, 17],
           [ 9, 21],
           [ 2, 14],
           [ 6, 18],
           [10, 22],
           [ 3, 15],
           [ 7, 19],
           [11, 23]])            
    
    """
    if lar.ndim != 3:
        raise ValueError, "lar must be 3d."
    y = lar.copy()
    y.label = [flattenlabel([y.label[1], y.label[2]])[0], y.label[0]]
    y.x = y.x.T.reshape(-1, y.shape[0])
    return y

# Calc -------------------------------------------------------------
        
def cov(lar):
    """
    Covariance matrix adjusted for missing (NaN) values.
    
    Note: Only works on 2d larrys.
    
    The mean of each row is assumed to be zero. So rows are not demeaned
    and therefore the covariance is normalized by the number of columns,
    not by the number of columns minus 1.        
    
    Parameters
    ----------
    lar : larry
        The larry you want to find the covariance of.
        
    Returns
    -------
    out : larry
        For 2d input of shape (N, T), for example, returns a NxN covariance
        matrix.
        
    Raises
    ------
    ValueError
        If input is not 2d    

    """
    if lar.ndim != 2:
        raise ValueError, 'This function only works on 2d larrys'      
    y = lar.copy()
    y.label[1] = list(y.label[0])
    y.x = covMissing(y.x)
    return y                
