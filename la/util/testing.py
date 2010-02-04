"LA unit test utility functions."

from la import larry
    
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
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is not larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is not larry2.label[i])
    return out    

def nocopy(larry1, larry2):
    "Return True if there are only references"
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is larry2.label[i])
    return out       
