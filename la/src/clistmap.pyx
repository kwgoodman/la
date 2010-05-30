"Cython version of listmap in la/util/misc.py"

def listmap(list list1, list list2):
    """
    list of indices idx such that [list1[i] for i in idx] is list2.
    
    This function is equivalent to idx = map(list1.index, list2) except that
    it is O(n) instead of O(n^2).
    
    All elements in list2 must be in list1 .
    
    Note: This is the C version of the function.   
    """ 
    cdef int i, n1 = len(list1), n2 = len(list2)
    cdef dict list1map = {}
    cdef list idx
    for i in xrange(n1):
        list1map[list1[i]] = i 
    idx = [0] * n2
    for i in xrange(n2):
        idx[i] = list1map[list2[i]]
    return idx 
