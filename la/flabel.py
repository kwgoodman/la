"label (list of lists) functions"

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
    
def list2index(L):
    "Convert a list to a unique list and the corresponding indices."
    uL = sorted(set(L))
    idx = dict((y, x) for x, y in enumerate(uL))
    return [idx[x] for x in L], uL
                
