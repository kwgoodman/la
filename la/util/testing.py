"LA unit test utility functions."

import numpy as np

from numpy.testing import assert_, assert_equal, assert_almost_equal

from la import larry


def assert_larry_equal(actual, desired, msg='', dtype=True, original=None,
                       iscopy=True):                      
    """
    Assert equality of two larries.
    
    If either `actual` or `desired` has a dtype that is inexact, such as
    float, then almost-equal is asserted; otherwise, equal is asserted.
    
    Parameters
    ----------
    actual : larry
        If you are testing a larry method, for example, then this is the larry
        returned by the method.    
    desired : larry
        This larry represents the expected result. If `actual` is not equal
        to `desired`, then an AssertionError will be raised.
    msg : str
        If `actual` is not equal to `desired`, then the string `msg` will
        be added to the top of the AssertionError message.
    dtype : {True, False}, optional
        The default (True) is to assert that the dtype of `actual` is the
        same as `desired`. If set to False, the dtype check is skipped.
    original : {None, larry}, optional
        If no `reference` or `nocopy` are True, then `original` must be a
        larry. Continuing the example discussed in `actual`, `original` would
        be the larry that was passed to the method.
    iscopy : {True, False}, optional
        Note: `iscopy` is ignored if `original` is None. If True (default) and
        if `original` is not None, then check that `actual` and `desired`
        share no references (i.e., are copies). If False and if `original` is
        not None, then check that `actual` and `desired` are views of each
        other (not copies).
            
    Returns
    -------
    None
    
    Raises
    ------
    AssertionError
        If the two larrys are not equal.
        
    Notes
    -----           
    If either `actual` or `desired` has a dtype that is inexact, such as
    float, then almost-equal is asserted; otherwise, equal is asserted.
            
    Examples
    --------    
    >>> from la.util.testing import assert_larry_equal
    >>> x = larry([1])
    >>> y = larry([1.1], [['a']])
    >>> assert_larry_equal(x, y, noreference=False, msg='Cumsum test')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "la/util/testing.py", line 94, in assert_larry_equal
        def heading(text):
    AssertionError: 

    -----------------
    TEST: Cumsum test
    -----------------

    	
    	-----
    	LABEL
    	-----
    	
    	Items are not equal:
    	item=0
    	item=0
    	
    	 ACTUAL: 0
    	 DESIRED: 'a'
    	
    	------------
    	X DATA ARRAY
    	------------
    	
    	Arrays are not equal
    	
    	(mismatch 100.0%)
    	 x: array([1])
    	 y: array([ 1.1])
    	
    	-----
    	DTYPE
    	-----
    	
    	Items are not equal:
    	 ACTUAL: dtype('int64')
    	 DESIRED: dtype('float64')
    	 
    """
    
    # Initialize
    fail = []
    
    # Function to make section headings
    def heading(text):
        line = '-' * len(text)
        return '\n\n' + line + '\n' + text + '\n' + line + '\n'
    
    # label
    try:         
        assert_equal(actual.label, desired.label)
    except AssertionError, err:
        fail.append(heading('LABEL') + str(err))       

    # Data array, x
    try:
        # Do both larrys have inexact dtype?
        if (issubclass(actual.x.dtype.type, np.inexact) or
            issubclass(desired.x.dtype.type, np.inexact)): 
            # Yes, so check for almost equal
            assert_almost_equal(actual.x, desired.x, decimal=13)
        else:
            # No, so check for exactly equal
           assert_equal(actual.x, desired.x)     
    except AssertionError, err:
        fail.append(heading('X DATA ARRAY') + str(err))
     
    # dtype
    if dtype: 
        try: 
            assert_equal(actual.dtype, desired.dtype)
        except AssertionError, err:
            fail.append(heading('DTYPE') + str(err))            

    # If original is not None, assert copies or views
    if not original is None:   
        if iscopy:
            # Check that the larrys are copies
            try:
                assert_iscopy(actual, original)
            except AssertionError, err:
                fail.append(heading('NOT A COPY') + str(err))               
        else:
            # Check that the larrys are views
            try:       
                assert_isview(actual, original)
            except AssertionError, err:
                text = heading('IS A COPY') + str(err)
                fail.append(text)              
    
    # Did the test pass?    
    if len(fail) > 0:
        # No
        err_msg = ''.join(fail)
        err_msg = err_msg.replace('\n', '\n\t')
        if len(msg):
            err_msg = heading("TEST: " + msg) + err_msg
        raise AssertionError, err_msg

# Utility functions ---------------------------------------------------------        
    
def printfail(theory, practice, header=None):
    x = []
    if header is not None:
        x.append('\n\n%s\n' % header)
    x.append('\ntheory\n')
    x.append(str(theory))
    x.append('\n')
    x.append('practice\n')
    x.append(str(practice))
    x.append('\n')    
    return ''.join(x) 
    
def noreference(larry1, larry2):
    "Return True if there are no shared references"
    if not isinstance(larry1, larry):
        raise TypeError, 'Input must be a larry'
    if not isinstance(larry2, larry):
        raise TypeError, 'Input must be a larry'
    if larry1.ndim != larry2.ndim:
        raise ValueError, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is not larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is not larry2.label[i])
    return out    

def nocopy(larry1, larry2):
    "Return True if there are only references"
    if not isinstance(larry1, larry):
        raise TypeError, 'Input must be a larry'
    if not isinstance(larry2, larry):
        raise TypeError, 'Input must be a larry'
    if larry1.ndim != larry2.ndim:
        raise ValueError, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is larry2.label[i])
    return out       
    
def assert_iscopy(larry1, larry2):
    "Return True if there are no shared references"
    if not isinstance(larry1, larry):
        raise TypeError, 'Input must be a larry'
    if not isinstance(larry2, larry):
        raise TypeError, 'Input must be a larry'
    msg = []    
    if np.may_share_memory(larry1.x, larry2.x):
        msg.append('The data arrays share a reference.')
    for i in xrange(min(larry1.ndim, larry2.ndim)):
        if larry1.label[i] is larry2.label[i]:
            msg.append('The labels along axis %d share a reference.' % i)
    if len(msg) > 0:
        msg.insert(0, '\n')
        msg = '\n'.join(msg)
        raise AssertionError, msg   

def assert_isview(larry1, larry2):
    "Return True if there are only references"
    if not isinstance(larry1, larry):
        raise TypeError, 'Input must be a larry'
    if not isinstance(larry2, larry):
        raise TypeError, 'Input must be a larry'
    msg = []    
    if not np.may_share_memory(larry1.x, larry2.x):
        msg.append('The data arrays do not share a reference.')
    for i in xrange(min(larry1.ndim, larry2.ndim)):
        if larry1.label[i] is not larry2.label[i]:
            text = 'The labels along axis %d does not share a reference.'
            msg.append(text % i)
    if len(msg) > 0:
        msg.insert(0, '\n')
        msg = '\n'.join(msg)
        raise AssertionError, msg  
                
