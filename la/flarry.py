"Functions that operate on larrys."

import numpy as np

from la.deflarry import larry
from la.flabel import flattenlabel, listmap, listmap_fill
from la.farray import covMissing
from la.missing import missing_marker, ismissing


# Alignment -----------------------------------------------------------------

def align(lar1, lar2, join='inner', cast=True):    
    """
    Align two larrys using one of five join methods.
    
    Parameters
    ----------
    lar1 : larry
        One of the input larrys. Must have the same number of dimensions as
        `lar2`.
    lar2 : larry
        One of the input larrys. Must have the same number of dimensions as
        `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The join method used to align the two larrys. The default join method
        along each axis is 'inner', i.e., the intersection of the labels. If
        `join` is a list of strings then the length of the list should be the 
        same as the number of dimensions of the two larrys. The first element
        in the list is the join method for axis=0, the second element is the
        join method for axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.
        
    Returns
    -------
    lar3 : larry
        A copy of the aligned version of `lar1`.
    lar4 : larry
        A copy of the aligned version of `lar2`.                     
        
    Examples
    --------
    Create two larrys:
    
    >>> lar1 = larry([1, 2])
    >>> lar2 = larry([1, 2, 3])

    The default join method is an inner join:

    >>> lar3, lar4 = la.align(lar1, lar2)
    >>> lar3
    label_0
        0
        1
    x
    array([1, 2])
    >>> lar4
    label_0
        0
        1
    x
    array([1, 2])

    An outer join adds a missing value (NaN) to lar1, therefore the the dtype
    of lar1 is changed from int to float:

    >>> lar3, lar4 = la.align(lar1, lar2, join='outer')
    >>> lar3
    label_0
        0
        1
        2
    x
    array([  1.,   2.,  NaN])
    >>> lar4
    label_0
        0
        1
        2
    x
    array([1, 2, 3])                              

    """
    x1, x2, label, x1isview, x2isview = align_raw(lar1, lar2, join=join,
                                                   cast=cast)
    if x1isview:    
        x1 = x1.copy()
    lar3 = larry(x1, label, integrity=False)        
    label = [list(lab) for lab in label]
    if x2isview:    
        x2 = x2.copy()
    lar4 = larry(x2, label, integrity=False)    
    return lar3, lar4

def align_raw(lar1, lar2, join='inner', cast=True):    
    """
    Align two larrys but return Numpy arrays and label instead of larrys.
    
    This function is the same as la.align() except that instead of returning
    two larrys, the components of the two larrys are returned (two Numpy
    arrays, a label, and flags for whether the two Numpy arrays are views of
    the data arrays of the corresponding input larrys).
    
    Parameters
    ----------
    lar1 : larry
        One of the input larrys. Must have the same number of dimensions as
        `lar2`.
    lar2 : larry
        One of the input larrys. Must have the same number of dimensions as
        `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The join method used to align the two larrys. The default join method
        along each axis is 'inner', i.e., the intersection of the labels. If
        `join` is a list of strings then the length of the list should be the 
        same as the number of dimensions of the two larrys. The first element
        in the list is the join method for axis=0, the second element is the
        join method for axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.
        
    Returns
    -------
    x1 : ndarray
        The aligned version of `lar1`.
    x2 : ndarray
        The aligned version of `lar2`.
    label : list of lists
        The label of the joined larrys.
    x1isview : bool
        True if x1 is a view of lar1.x; False otherwise. A view of lar1.x is
        retuned if the labels of `lar1` and `lar2` are the same along all
        axes; otherwise a copy is returned.
    x2isview : bool           
        True if x2 is a view of lar2.x; False otherwise.  A view of lar2.x is
        retuned if the labels of `lar1` and `lar2` are the same along all
        axes; otherwise a copy is returned.
        
    See Also
    --------
    la.align: Align two larrys using one of five join methods.   
        
    Notes
    -----
    The returned Numpy arrays are views of the corresponding input larrys if
    the labels of the two input larrys are the same along all axes. If the
    labels are not the same along any axis then a copy is returned.     
       
    Examples
    --------
    Create two larrys:
    
    >>> y1 = larry([1, 2])
    >>> y2 = larry([1, 2, 3])

    The default join method is an inner join:

    >>> x1, x2, label, x1isview, x2isview = la.flarry._align_raw(lar1, lar2)
    >>> x1
    array([1, 2])
    >>> x2
    array([1, 2])
    >>> label
    [[0, 1]]
    >>> x1isview
    False
    >>> x2isview
    False

    An outer join adds a missing value (NaN) to lar1, therefore the the dtype
    of lar1 is changed from int to float:

    >>> x1, x2, label, x1isview, x2isview = la.flarry._align_raw(lar1, lar2, join='outer')
    >>> x1
    array([  1.,   2.,  NaN])
    >>> x2
    array([1, 2, 3])
    >>> label
    [[0, 1, 2]]
    >>> x1isview
    False
    >>> x2isview
    False
    
    If the labels are already aligned, then a view of the data array is
    returned:
    
    >>> lar1 = larry([1, 2])
    >>> lar2 = larry([3, 4])
    >>> x1, x2, label, x1isview, x2isview = la.flarry._align_raw(lar1, lar2)
    >>> x1isview
    True
    >>> x2isview
    True                                 

    """
    
    # Check number of dimensions
    ndim = lar2.ndim
    if lar1.ndim != ndim:
        msg = "'lar1' and 'lar2' must have the same number of dimensions."
        raise ValueError, msg
        
    # Check join type    
    typejoin = type(join)
    if typejoin is str:
        join = [join] * ndim
    elif typejoin is list:
        if len(join) != ndim:
            msg = "Length of `join` list equal number of dimension of `lar1`."
            raise ValueError, msg
    else:
        raise TypeError, "`join` must be a string or a list."
        
    # Initialize missing markers, set value later (in loop) only if needed.
    # The weird initialization value ensures a user would never pick the same 
    undefined = 'aB!@12#E~=-'
    miss1 = undefined
    miss2 = undefined
        
    # For loop initialization                         
    label = []
    x1 = lar1.x
    x2 = lar2.x
    label1 = lar1.label
    label2 = lar2.label
    x1isview = True
    x2isview = True
    
    # Loop: align one axis at a time 
    msg = "`fill` type not compatible with larry dtype"     
    for ax in range(ndim):    
        list1 = label1[ax]
        list2 = label2[ax]
        joinax = join[ax]        
        if joinax == 'inner':
            if list1 == list2:
                list3 = list(list1)
            else:
                list3 = list(set(list1) & (set(list2)))
                list3.sort()
                idx1 = listmap(list1, list3)
                idx2 = listmap(list2, list3)
                x1 = x1.take(idx1, ax)
                x2 = x2.take(idx2, ax)
                x1isview = False
                x2isview = False   
        elif joinax == 'outer':
            if list1 == list2:
                list3 = list(list1)
            else:                 
                list3 = list(set(list1) | (set(list2)))
                list3.sort()
                idx1, idx1_miss = listmap_fill(list1, list3, fill=0)
                idx2, idx2_miss = listmap_fill(list2, list3, fill=0)
                x1 = x1.take(idx1, ax)
                x2 = x2.take(idx2, ax) 
                if len(idx1_miss) > 0:
                    if miss1 == undefined:
                        miss1 = missing_marker(lar1)
                    if miss1 == NotImplemented:
                        if cast:
                            x1 = x1.astype(float)
                            miss1 = missing_marker(x1)   
                        else:                         
                            raise TypeError, msg
                    index1 = [slice(None)] * ndim
                    index1[ax] = idx1_miss      
                    x1[index1] = miss1                                        
                if len(idx2_miss) > 0:
                    if miss2 == undefined:
                        miss2 = missing_marker(lar2)
                    if miss2 == NotImplemented:
                        if cast:
                            x2 = x2.astype(float)
                            miss2 = missing_marker(x2)   
                        else:
                            raise TypeError, msg
                    index2 = [slice(None)] * ndim
                    index2[ax] = idx2_miss                             
                    x2[index2] = miss2
                x1isview = False
                x2isview = False                     
        elif joinax == 'left':
            list3 = list(list1)
            if list1 != list2:
                idx2, idx2_miss = listmap_fill(list2, list3, fill=0)
                x2 = x2.take(idx2, ax) 
                if len(idx2_miss) > 0:
                    if miss2 == undefined:
                        miss2 = missing_marker(lar2)
                    if miss2 == NotImplemented:
                        if miss2 is None:
                            miss2 = missing_marker(lar2)
                        if miss2 is None:
                            miss2 = missing_marker(lar2)
                        if cast:
                            x2 = x2.astype(float)
                            miss2 = missing_marker(x2)   
                        else:
                            raise TypeError, msg
                    index2 = [slice(None)] * ndim
                    index2[ax] = idx2_miss        
                    x2[index2] = miss2
                x2isview = False                    
        elif joinax == 'right':
            list3 = list(list2)
            if list1 != list2:            
                idx1, idx1_miss = listmap_fill(list1, list3, fill=0)
                x1 = x1.take(idx1, ax) 
                if len(idx1_miss) > 0:
                    if miss1 == undefined:
                        miss1 = missing_marker(lar1)
                    if miss1 == NotImplemented:
                        if cast:
                            x1 = x1.astype(float)
                            miss1 = missing_marker(x1)   
                        else:
                            raise TypeError, msg
                    index1 = [slice(None)] * ndim
                    index1[ax] = idx1_miss                            
                    x1[index1] = miss1 
                x1isview = False                                 
        else:
            raise ValueError, 'join type not recognized'  
        label.append(list3)
    
    return x1, x2, label, x1isview, x2isview
    
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
    Sorted list containing the intersection of labels along specified axis.
    
    Parameters
    ----------
    axis : int
        The axis along which to take the intersection of the labels.
    args : larrys
        The larrys (separated by commas) over which the intersection is taken.
        
    Returns
    -------
    out : list
        A sorted list containing the intersection of the labels.
        
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

# Binary-- -----------------------------------------------------------------

def binaryop(func, lar1, lar2, join='inner', cast=True, missone='ignore',
             misstwo='ignore', **kwargs):
    """
    Binary operation on two larrys using given function and join method.
    
    Parameters
    ----------
    func : function
        A function that takes two Numpy arrays as input and returns a Numpy
        array as output. For example: np.add. You can also pass keyword
        arguments to the function; see `**kwargs`.
    lar1 : larry
        The larry on the left-hand side of the binary operation. Must have
        the same number of dimensions as `lar2`.
    lar2 : larry
        The larry on the right-hand side of the binary operation. Must have
        the same number of dimensions as `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The method used to join the two larrys. The default join method along
        all axes is 'inner', i.e., the intersection of the labels. If `join`
        is a list of strings then the length of the list should be the number
        of dimensions of the two larrys. The first element in the list is the
        join method for axis=0, the second element is the join method for
        axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.   
    missone : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `missone` is set to something other than 'ignore', such
        as 0, then all elements that are missing in one larry but not missing
        in the other larry are replaced by `missone`. For example, if an
        element is in one larry but missing in the other larry then you may
        want to set the missing value to zero when summing two larrys.
    misstwo : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `misstwo` is set to something other than 'ignore', such
        as 0, then all elements that are missing in both larrys are replaced
        by `misstwo`.  
    **kwargs : Keyword arguments, optional
        Keyword arguments to pass to `func`. The keyword arguments passed to
        `func` cannot have the following keys: join, cast, missone, misstwo.
        
    Returns
    -------
    lar3 : larry
        The result of the binary operation.
        
    See Also
    --------
    la.align: Align two larrys using one of five join methods.  
        
    Examples
    --------
    Create two larrys:
    
    >>> from la import nan
    >>> lar1 = larry([1,   2, nan], [['a', 'b', 'c']])
    >>> lar2 = larry([1, nan, nan], [['a', 'b', 'dd']])
    
    The default is an inner join (note that lar1 and lar2 have two labels in
    common):
    
    >>> la.binaryop(np.add, lar1, lar2)
    label_0
        a
        b
    x
    array([  2.,  NaN])
        
    If one data element is missing in one larry but not in the other, then you
    can replace the missing value with `missone` (here 0):     
        
    >>> la.binaryop(np.add, lar1, lar2, missone=0)
    label_0
        a
        b
    x
    array([ 2.,  2.])
        
    An outer join: 
    
    >>> la.binaryop(np.add, lar1, lar2, join='outer')
    label_0
        a
        b
        c
        dd
    x
    array([  2.,  NaN,  NaN,  NaN])
    
    An outer join with single and double missing values replaced by zero:
        
    >>> la.binaryop(np.add, lar1, lar2, join='outer', missone=0, misstwo=0)
    label_0
        a
        b
        c
        dd
    x
    array([ 2.,  2.,  0.,  0.])                               

    """
    
    # Align
    x1, x2, label, ign1, ign2 = align_raw(lar1, lar2, join=join, cast=cast)
    
    # Replacing missing values is slow, so only do if requested
    if missone != 'ignore' or misstwo != 'ignore':
        miss1 = ismissing(x1)
        miss2 = ismissing(x2)
    if missone != 'ignore':    
        missone1 = miss1 & ~miss2
        if missone1.any():
            x1[missone1] = missone
        missone2 = miss2 & ~miss1    
        if missone2.any():
            x2[missone2] = missone
    if misstwo != 'ignore':            
        misstwo12 = miss1 & miss2    
        if misstwo12.any():
            x1[misstwo12] = misstwo
            x2[misstwo12] = misstwo           
            
    # Binary function
    x = func(x1, x2, **kwargs)
    
    return larry(x, label, integrity=False)
    
def add(lar1, lar2, join='inner', cast=True, missone='ignore',
        misstwo='ignore'):
    """
    Sum of two larrys using given join and fill methods. 
    
    Parameters
    ----------
    lar1 : larry
        The larry on the left-hand side of the sum. Must have the same number
        of dimensions as `lar2`.
    lar2 : larry
        The larry on the right-hand side of the sum. Must have the same number
        of dimensions as `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The method used to join the two larrys. The default join method along
        all axes is 'inner', i.e., the intersection of the labels. If `join`
        is a list of strings then the length of the list should be the number
        of dimensions of the two larrys. The first element in the list is the
        join method for axis=0, the second element is the join method for
        axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.   
    missone : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `missone` is set to something other than 'ignore', such
        as 0, then all elements that are missing in one larry but not missing
        in the other larry are replaced by `missone`. For example, if an
        element is in one larry but missing in the other larry then you may
        want to set the missing value to zero when summing two larrys.
    misstwo : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `misstwo` is set to something other than 'ignore', such
        as 0, then all elements that are missing in both larrys are replaced
        by `misstwo`.
               
    Returns
    -------
    y : larry
        The sum of the two larrys, `lar1` and `lar2`.
        
    See Also
    --------
    la.larry.__add__: Sum a larry with another larry, Numpy array, or scalar.
    la.binaryop: Binary operation on two larrys using given function.
    
    Notes
    -----
    This is a convenience function that calls la.binaryop() with `func` set
    to numpy.add.  
        
    Examples
    --------
    Create two larrys:
    
    >>> from la import nan
    >>> lar1 = larry([1,   2, nan], [['a', 'b', 'c']])
    >>> lar2 = larry([1, nan, nan], [['a', 'b', 'dd']])
    
    The default is an inner join (note that lar1 and lar2 have two labels in
    common):
    
    >>> la.add(lar1, lar2)
    label_0
        a
        b
    x
    array([  2.,  NaN])
    
    which is the same result you get with lar1 + lar2:

    >>> lar1 + lar2
    label_0
        a
        b
    x
    array([  2.,  NaN])    
        
    If one data element is missing in one larry but not in the other, then you
    can replace the missing value with `missone` (here 0):     
        
    >>> la.add(lar1, lar2, missone=0)
    label_0
        a
        b
    x
    array([ 2.,  2.])
        
    An outer join: 
    
    >>> la.add(lar1, lar2, join='outer')
    label_0
        a
        b
        c
        dd
    x
    array([  2.,  NaN,  NaN,  NaN])
    
    An outer join with single and double missing values replaced by zero:
        
    >>> la.add(lar1, lar2, join='outer', missone=0, misstwo=0)
    label_0
        a
        b
        c
        dd
    x
    array([ 2.,  2.,  0.,  0.])                               

    """    
    return binaryop(np.add, lar1, lar2, join=join, cast=cast, missone=missone,
                    misstwo=misstwo)

def subtract(lar1, lar2, join='inner', cast=True, missone='ignore',
             misstwo='ignore'):
    """
    Difference of two larrys using given join and fill methods. 
    
    Parameters
    ----------
    lar1 : larry
        The larry on the left-hand side of the difference. Must have the same
        number of dimensions as `lar2`.
    lar2 : larry
        The larry on the right-hand side of the difference. Must have the same
        number of dimensions as `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The method used to join the two larrys. The default join method along
        all axes is 'inner', i.e., the intersection of the labels. If `join`
        is a list of strings then the length of the list should be the number
        of dimensions of the two larrys. The first element in the list is the
        join method for axis=0, the second element is the join method for
        axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.   
    missone : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `missone` is set to something other than 'ignore', such
        as 0, then all elements that are missing in one larry but not missing
        in the other larry are replaced by `missone`. For example, if an
        element is in one larry but missing in the other larry then you may
        want to set the missing value to zero when subtracting two larrys.
    misstwo : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `misstwo` is set to something other than 'ignore', such
        as 0, then all elements that are missing in both larrys are replaced
        by `misstwo`.
               
    Returns
    -------
    y : larry
        The difference of the two larrys, `lar1` and `lar2`.
        
    See Also
    --------
    la.larry.__sub__: Subtract a larry from another larry, array, or scalar.
    la.binaryop: Binary operation on two larrys using given function.
    
    Notes
    -----
    This is a convenience function that calls la.binaryop() with `func` set
    to numpy.subtract.  
        
    Examples
    --------
    Create two larrys:
    
    >>> from la import nan
    >>> lar1 = larry([1,   2, nan], [['a', 'b', 'c']])
    >>> lar2 = larry([1, nan, nan], [['a', 'b', 'dd']])
    
    The default is an inner join (note that lar1 and lar2 have two labels in
    common):
    
    >>> la.subtract(lar1, lar2)
    label_0
        a
        b
    x
    array([  0.,  NaN])
    
    which is the same result you get with lar1 - lar2:

    >>> lar1 - lar2
    label_0
        a
        b
    x
    array([  0.,  NaN])
      
    If one data element is missing in one larry but not in the other, then you
    can replace the missing value with `missone` (here 0):     
        
    >>> la.subtract(lar1, lar2, missone=0)
    label_0
        a
        b
    x
    array([ 0.,  2.])
        
    An outer join: 
    
    >>> la.subtract(lar1, lar2, join='outer')
    label_0
        a
        b
        c
        dd
    x
    array([  0.,  NaN,  NaN,  NaN])
    
    An outer join with single and double missing values replaced by zero:
        
    >>> la.subtract(lar1, lar2, join='outer', missone=0, misstwo=0)
    label_0
        a
        b
        c
        dd
    x
    array([ 0.,  2.,  0.,  0.])                               

    """    
    return binaryop(np.subtract, lar1, lar2, join=join, cast=cast,
                    missone=missone, misstwo=misstwo)
                    
def multiply(lar1, lar2, join='inner', cast=True, missone='ignore',
             misstwo='ignore'):
    """
    Multiply two larrys element-wise using given join and fill methods.
    
    Parameters
    ----------
    lar1 : larry
        The larry on the left-hand side of the product. Must have the same
        number of dimensions as `lar2`.
    lar2 : larry
        The larry on the right-hand side of the product. Must have the same
        number of dimensions as `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The method used to join the two larrys. The default join method along
        all axes is 'inner', i.e., the intersection of the labels. If `join`
        is a list of strings then the length of the list should be the number
        of dimensions of the two larrys. The first element in the list is the
        join method for axis=0, the second element is the join method for
        axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.   
    missone : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `missone` is set to something other than 'ignore', such
        as 0, then all elements that are missing in one larry but not missing
        in the other larry are replaced by `missone`. For example, if an
        element is in one larry but missing in the other larry then you may
        want to set the missing value to one when multiplying two larrys.
    misstwo : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `misstwo` is set to something other than 'ignore', such
        as 0, then all elements that are missing in both larrys are replaced
        by `misstwo`.
               
    Returns
    -------
    y : larry
        The element-wise product of the two larrys, `lar1` and `lar2`.
        
    See Also
    --------
    la.larry.__mul__: Multiply a larry with another larry, array, or scalar.
    la.binaryop: Binary operation on two larrys using given function.
    
    Notes
    -----
    This is a convenience function that calls la.binaryop() with `func` set
    to numpy.multiply.  
        
    Examples
    --------
    Create two larrys:
    
    >>> from la import nan
    >>> lar1 = larry([1,   2, nan], [['a', 'b', 'c']])
    >>> lar2 = larry([1, nan, nan], [['a', 'b', 'dd']])
    
    The default is an inner join (note that lar1 and lar2 have two labels in
    common):
    
    >>> la.multiply(lar1, lar2)
    label_0
        a
        b
    x
    array([  1.,  NaN])
    
    which is the same result you get with lar1 * lar2:

    >>> lar1 * lar2
    label_0
        a
        b
    x
    array([  1.,  NaN])
      
    If one data element is missing in one larry but not in the other, then you
    can replace the missing value with `missone` (here 1):     
        
    >>> la.multiply(lar1, lar2, missone=1)
    label_0
        a
        b
    x
    array([ 1.,  2.])
        
    An outer join: 
    
    >>> la.multiply(lar1, lar2, join='outer')
    label_0
        a
        b
        c
        dd
    x
    array([  1.,  NaN,  NaN,  NaN])
    
    An outer join with single and double missing values replaced by one:       
        
    >>> la.multiply(lar1, lar2, join='outer', missone=1, misstwo=1)
    label_0
        a
        b
        c
        dd
    x
    array([ 1.,  2.,  1.,  1.])                               

    """    
    return binaryop(np.multiply, lar1, lar2, join=join, cast=cast,
                    missone=missone, misstwo=misstwo)                    

def divide(lar1, lar2, join='inner', cast=True, missone='ignore',
           misstwo='ignore'):
    """
    Divide two larrys element-wise using given join and fill methods.
    
    Parameters
    ----------
    lar1 : larry
        The larry on the left-hand side of the division. Must have the same
        number of dimensions as `lar2`.
    lar2 : larry
        The larry on the right-hand side of the division. Must have the same
        number of dimensions as `lar1`.
    join : {'inner', 'outer', 'left', 'right', list}, optional
        The method used to join the two larrys. The default join method along
        all axes is 'inner', i.e., the intersection of the labels. If `join`
        is a list of strings then the length of the list should be the number
        of dimensions of the two larrys. The first element in the list is the
        join method for axis=0, the second element is the join method for
        axis=1, and so on.
    cast : bool, optional
        Only float, str, and object dtypes have missing value markers (la.nan,
        '', and None, respectively). Other dtypes, such as int and bool, do
        not have missing value markers. If `cast` is set to True (default)
        then int and bool dtypes, for example, will be cast to float if any
        new rows, columns, etc are created. If cast is set to False, then a
        TypeError will be raised for int and bool dtype input if the join
        introduces new rows, columns, etc. An inner join will never introduce
        new rows, columns, etc.   
    missone : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `missone` is set to something other than 'ignore', such
        as 0, then all elements that are missing in one larry but not missing
        in the other larry are replaced by `missone`. For example, if an
        element is in one larry but missing in the other larry then you may
        want to set the missing value to one when dividing two larrys.
    misstwo : {scalar, 'ignore'}, optional
        By default ('ignore') no special treatment of missing values is made.
        If, however, `misstwo` is set to something other than 'ignore', such
        as 0, then all elements that are missing in both larrys are replaced
        by `misstwo`.
               
    Returns
    -------
    y : larry
        The element-wise quotient of the two larrys, `lar1` and `lar2`.
        
    See Also
    --------
    la.larry.__div__: Multiply a larry with another larry, array, or scalar.
    la.binaryop: Binary operation on two larrys using given function.
    
    Notes
    -----
    This is a convenience function that calls la.binaryop() with `func` set
    to numpy.divide.  
        
    Examples
    --------
    Create two larrys:
    
    >>> from la import nan
    >>> lar1 = larry([1,   2, nan], [['a', 'b', 'c']])
    >>> lar2 = larry([1, nan, nan], [['a', 'b', 'dd']])
    
    The default is an inner join (note that lar1 and lar2 have two labels in
    common):
    
    >>> la.divide(lar1, lar2)
    label_0
        a
        b
    x
    array([  1.,  NaN])
    
    which is the same result you get with lar1 / lar2:

    >>> lar1 / lar2
    label_0
        a
        b
    x
    array([  1.,  NaN])
      
    If one data element is missing in one larry but not in the other, then you
    can replace the missing value with `missone` (here 1):     
        
    >>> la.divide(lar1, lar2, missone=1)
    label_0
        a
        b
    x
    array([ 1.,  2.])
        
    An outer join: 
    
    >>> la.divide(lar1, lar2, join='outer')
    label_0
        a
        b
        c
        dd
    x
    array([  1.,  NaN,  NaN,  NaN])
    
    An outer join with single and double missing values replaced by one:       
        
    >>> la.divide(lar1, lar2, join='outer', missone=1, misstwo=1)
    label_0
        a
        b
        c
        dd
    x
    array([ 1.,  2.,  1.,  1.])                               

    """    
    return binaryop(np.divide, lar1, lar2, join=join, cast=cast,
                    missone=missone, misstwo=misstwo)

# Misc ----------------------------------------------------------------------

def unique(lar, return_index=False, return_inverse=False):
    """
    Find the unique elements of a larry.
    
    Returns the sorted unique elements of a larry as a Numpy array. There are
    two optional outputs in addition to the unique elements: the indices of the
    input larry that give the unique values, and the indices of the unique array
    that reconstruct the input array.
    
    Parameters
    ----------
    ar : larry
        Input larry. This will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `lar` that result in the unique
        larry.
    return_inverse : bool, optional
        If True, also return the indices of the unique larry that can be used
        to reconstruct `lar`.
    
    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the unique values in the (flattened) original larry.
        Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original larry from the
        unique array. Only provided if `return_inverse` is True.
    
    Examples
    --------
    >>> la.unique(larry([1, 1, 2, 2, 3, 3]))
    array([1, 2, 3])
    >>> lar = larry([[1, 1], [2, 3]])
    >>> la.unique(lar)
    array([1, 2, 3])
    
    Return the indices of the original larry that give the unique values:
    
    >>> lar = larry(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = la.unique(lar, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> lar[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')
    
    Reconstruct the input array (the data portion of the larry, not the label
    portion) from the unique values:
    
    >>> lar = larry([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = la.unique(lar, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    """
    return np.unique(lar.x, return_index, return_inverse)

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

# Random -----------------------------------------------------------    
    
def rand(*args, **kwargs):
    """
    Random samples from a uniform distribution in a given shape.
    
    The random samples are from a uniform distribution over ``[0, 1)``.
    
    Parameters
    ----------
    args : `n` ints, optional
        The dimensions of the returned larry, should be all positive. These
        may be omitted if you pass in a label as a keyword argument.
    kwargs : keyword arguments, optional
        Keyword arguments to use in the construction of the larry such as
        label and integrity. If a label is passed then its dimensions must
        match the `n` integers passed in or, optionally, you can pass in the
        label without the `n` shape integers. If rand is passed in then that
        will be used to generate the random numbers. In that way you can set
        the state of the random number generator outside of this function.  
    
    Returns
    -------
    Z : larry or float
        A ``(d1, ..., dn)``-shaped larry of floating-point samples from
        a uniform distribution, or a single such float if no parameters were
        supplied.
    
    See Also
    --------
    la.randn : Random samples from the "standard normal" distribution.
    
    Examples
    --------
    A single random sample:
    
    >>> la.rand()
    0.64323350463488804
    
    A shape (2, 2) random larry:
    
    >>> la.rand(2, 2)
    label_0
        0
        1
    label_1
        0
        1
    x
    array([[ 0.09277439,  0.94194077],
           [ 0.72887997,  0.41124147]])
           
    A shape (2, 2) random larry with given labels:
           
    >>> la.rand(label=[['row1', 'row2'], ['col1', 'col2']])
    label_0
        row1
        row2
    label_1
        col1
        col2
    x
    array([[ 0.3449072 ,  0.40397174],
           [ 0.7791279 ,  0.86084403]])
           
    Results are repeatable if you set the state of the random number generator
    outside of la.rand:
           
    >>> import numpy as np
    >>> rs = np.random.RandomState([1, 2, 3])
    >>> la.randn(randn=rs.randn)
    0.89858244820995015
    >>> la.randn(randn=rs.randn)
    0.25528876596298244
    >>> rs = np.random.RandomState([1, 2, 3])
    >>> la.randn(randn=rs.randn)
    0.89858244820995015
    >>> la.randn(randn=rs.randn)
    0.25528876596298244
        
    """
    if 'rand' in kwargs:
        randfunc = kwargs['rand']
        kwargs = dict(kwargs)
        del kwargs['rand']
    else:
        randfunc = np.random.rand   
    if len(args) > 0:
        return larry(randfunc(*args), **kwargs)
    elif 'label' in kwargs:
        n = [len(z) for z in kwargs['label']]
        return larry(randfunc(*n), **kwargs)     
    elif (len(args) == 0) and (len(kwargs) == 0):
        return randfunc()
    elif (len(args) == 0) and (len(kwargs) == 1) and ('rand' in kwargs):
        return randfunc()    
    else:
        raise ValueError, 'Input parameters not recognized'
    
def randn(*args, **kwargs):
    """
    Random samples from the "standard normal" distribution in a given shape.
    
    The random samples are from a "normal" (Gaussian) distribution of mean 0
    and variance 1.
    
    Parameters
    ----------
    args : `n` ints, optional
        The dimensions of the returned larry, should be all positive. These
        may be omitted if you pass in a label as a keyword argument.
    kwargs : keyword arguments, optional
        Keyword arguments to use in the construction of the larry such as
        label and integrity. If a label is passed then its dimensions must
        match the `n` integers passed in or, optionally, you can pass in the
        label without the `n` shape integers. If randn is passed in then that
        will be used to generate the random numbers. In that way you can set
        the state of the random number generator outside of this function.  
    
    Returns
    -------
    Z : larry or float
        A ``(d1, ..., dn)``-shaped larry of floating-point samples from
        the standard normal distribution, or a single such float if
        no parameters were supplied.
    
    See Also
    --------
    la.rand : Random values from a uniform distribution in a given shape.
    
    Examples
    --------
    A single random sample:    
    
    >>> la.randn()
    0.33086946957034052
    
    A shape (2, 2) random larry:    
    
    >>> la.randn(2, 2)
    label_0
        0
        1
    label_1
        0
        1
    x
    array([[-0.08182341,  0.79768108],
           [-0.23584547,  1.80118376]])
           
    A shape (2, 2) random larry with given labels:           
           
    >>> la.randn(label=[['row1', 'row2'], ['col1', 'col2']])
    label_0
        row1
        row2
    label_1
        col1
        col2
    x
    array([[ 0.10737701, -0.24947824],
           [ 1.51021208,  1.00280387]])

    Results are repeatable if you set the state of the random number generator
    outside of la.rand:

    >>> import numpy as np
    >>> rs = np.random.RandomState([1, 2, 3])
    >>> la.randn(randn=rs.randn)
    0.89858244820995015
    >>> la.randn(randn=rs.randn)
    0.25528876596298244
    >>> rs = np.random.RandomState([1, 2, 3])
    >>> la.randn(randn=rs.randn)
    0.89858244820995015
    >>> la.randn(randn=rs.randn)
    0.25528876596298244
    
    """
    if 'randn' in kwargs:
        randnfunc = kwargs['randn']
        kwargs = dict(kwargs)
        del kwargs['randn']        
    else:
        randnfunc = np.random.randn   
    if len(args) > 0:
        return larry(randnfunc(*args), **kwargs)
    elif 'label' in kwargs:
        n = [len(z) for z in kwargs['label']]
        return larry(randnfunc(*n), **kwargs)     
    elif (len(args) == 0) and (len(kwargs) == 0):
        return randnfunc()
    elif (len(args) == 0) and (len(kwargs) == 1) and ('randn' in kwargs):
        return randnfunc()         
    else:
        raise ValueError, 'Input parameters not recognized'
                            
