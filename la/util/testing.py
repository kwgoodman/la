"LA unit test utility functions."

from numpy.testing import assert_, assert_equal, assert_almost_equal

from la import larry

def assert_larry_equal(actual, desired, msg='', dtype=True, original=None,
                       noreference=True, nocopy=False, verbose=True):
    """Assert equality of attributes of two larries."""
    
    fail = []
    
    # label
    try:         
        assert_equal(actual.label, desired.label)
    except AssertionError, err:
        fail.append(heading('LABEL') + str(err))      

    # Data array
    try:  
        assert_equal(actual.x, desired.x)
    except AssertionError, err:
        fail.append(heading('X DATA ARRAY') + str(err))
     
    # dtype
    if dtype: 
        try: 
            assert_equal(actual.x.dtype, desired.x.dtype)
        except AssertionError, err:
            fail.append(heading('DTYPE') + str(err))            
    
    # Check for references or copies
    if noreference:
        if original is None:
            raise ValueError, 'original must be a larry to run noreference check.'
        try:
            assert_(assert_noreference(actual, original))
        except AssertionError, err:
            fail.append(heading('REFERENCE FOUND') + str(err))               
    elif nocopy:
        if original is None:
            raise ValueError, 'original must be a larry to run nocopy check.' 
        try:       
            assert_(assert_nocopy(actual, original))
        except AssertionError, err:
            fail.append(heading('COPY INSTEAD OF REFERENCE FOUND') + str(err))              
    else:   #FIXME check view for different dimensional larries
        pass
    
    # Did the test pass?    
    if len(fail) > 0:
        # No
        msg = ''.join(fail)
        raise AssertionError, msg

def heading(text):
    line = '-' * len(text)
    return '\n\n' + line + '\n' + text + '\n' + line + '\n'
    
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
    
def assert_noreference(larry1, larry2):
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
    if not out:
        raise AssertionError, 'The larrys share a reference.'    

def assert_nocopy(larry1, larry2):
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
    if not out:
        raise AssertionError, 'Parts of the larrys are not references.'   
                
