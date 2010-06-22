"Cython versions of la/flabel.py functions"

def listmap(list list1, list list2, bool ignore_unmappable=False):
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

    See Also
    --------
    la.flable.listmap_nofill: Indices that map one list onto another list

    Notes
    ----- 
    This is the C version of the function.
    
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
    cdef int i, n1 = len(list1), n2 = len(list2)
    cdef dict list1map = {}
    cdef list idx
    for i in xrange(n1):
        list1map[list1[i]] = i 
    if ignore_unmappable:
        idx = []
        for i in xrange(n2):
            try:
                idx.append(list1map[list2[i]]) 
            except KeyError:
                pass       
    else:
        idx = [0] * n2
        for i in xrange(n2):
            idx[i] = list1map[list2[i]]
    return idx
    
def listmap_fill(list list1, list list2, int fill=0):
    """
    Indices that map one list onto another and indices of unmappable elements.
    
    Similar to listmap() but additionaly returns a second index that gives the
    index values of the unmappable elements. The unmappable items are filled
    in the first index retuned with `fill`.
    
    Parameters
    ----------
    list1 : list
        The list to map from.
    list2 : list
        The list to map to.
    fill : fill value, optional
        Any element that cannot be mapped is given the index value `fill` in
        the frist of two index lists returned.
    
    Returns
    -------
    index : list
        An index such that [list1[i] for i in idx] is `list2`. If there are
        items in `list2` that are not in `list1` then the correponding index
        value is `fill`.
    index_missing : list
        The index values (relative to `list1`) of the elements in `list2` that
        are not in `list1`.
        
    See Also
    --------
    la.flable.listmap: Indices that map one list onto another list
    
    Notes
    ----- 
    This is the C version of the function.    

    Examples
    --------
    A simple mapping where all elements are mappable:

    >>> list1 = [1, 2, 3]
    >>> list2 = [3, 1, 2]
    >>> idx, idx_unmappable = listmap_fill(list1, list2)
    >>> idx
    [2, 0, 1]
    >>> idx_unmappable
    []
    >>> [list1[i] for i in idx] == list2
    True
    
    An example where list2 contains an element (4) that is not in list1:

    >>> list1 = [1, 2, 3]
    >>> list2 = [1, 2, 3, 4]
    >>> idx, idx_unmappable = listmap_fill(list1, list2)
    >>> idx
    [0, 1, 2, 0]
    >>> idx_unmappable
    [3]
    
    """
    cdef int i, n1 = len(list1), n2 = len(list2)
    cdef dict list1map = {}
    cdef list index = [fill] * n2, index_missing = []
    for i in xrange(n1):
        list1map[list1[i]] = i
    for i in xrange(n2):
        try:
            index[i] = list1map[list2[i]]
        except KeyError:
            index_missing.append(i)
    return index, index_missing
    
