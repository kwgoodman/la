"Functions that operate on larrys."

import numpy as np

from la.deflarry import larry
from la.util.misc import list2index, fromlists

   
def fromlist(data, check_input=True):
    """
    Convert a list of tuples to a larry.
    
    Parameters
    ----------
    data : list of tuples
        The input must be a list of tuples where each tuple represents one
        data point in the larry: (label0, label1, ..., labelN, value)
    check_input : {True, False}
        It takes time to check that the input is a list of all tuples and
        that each tuple has the same length. The default (True) is to check
        the input. If you already know your input is good, you can save time
        by skipping the check (by setting `check_input` to False).    
        
    Returns
    -------
    y : larry
        A larry constructed from `data` is returned.
        
    See also
    --------
    flatten : Return an unflattened copy of the larry.
    unflatten : Return a copy of the larry collapsed into one dimension.

    Examples
    --------
    Convert a list of label, value pairs to a larry:
    
    >>> data = [('r0', 'c0', 1), ('r0', 'c1', 2), ('r1', 'c0', 3), ('r1', 'c1', 4)]
    >>> la.fromlist(data)
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[ 1.,  2.],
           [ 3.,  4.]])
            
    What happens if we throw out the last data point? The missing value
    becomes NaN:       
            
    >>> data = data[:-1]
    >>> la.fromlist(data)
    label_0
        r0
        r1
    label_1
        c0
        c1
    x
    array([[  1.,   2.],
           [  3.,  NaN]])
            
    """

    # Check input
    if check_input:
        if type(data) != list:
            raise TypeError, 'data must be a list'
        if not all([type(t) == tuple for t in data]):
            raise TypeError, 'data must be a list of tuples'
        ndim = len(data[0])
        if not all([len(t) == ndim for t in data]):
            msg = 'All of the tuples in data must have the same length.'       
            raise ValueError, msg
        
    # Split data into label and x
    labels = zip(*data)
    xs = labels.pop(-1)   
    
    # Determine labels, shape, and index into array	
    x, label = fromlists(xs, labels)  
    
    return larry(x, label)         
    
def union(axis, *args):
    "Union of labels along specified axis."
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
    "Intersection of labels along specified axis."
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
    
def stack(mode, **kwargs):
    """Stack 2d larrys to make a 3d larry.
    
    Parameters
    ----------
    mode : {'union', 'intersection'}
        Should the 3d larry be made from the union or intersection of all the
        rows and all the columns?
    kwargs : name=larry
        Variable length input listing the z axis name and larry. For example,
        stack('union', momentum=x, volume=y, ep=z)
        
    Returns
    -------
    out : larry
        Returns a 3d larry.
        
    Raises
    ------
    ValueError
        If mode is not union or intersection or if any of the input larrys are
        not 2d.
                    
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
