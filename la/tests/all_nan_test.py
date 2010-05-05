"Test larry methods for proper handling of all NaN input"

from numpy.testing import assert_equal

from la import larry, nan
from la.util.testing import assert_larry_equal


def lar():    
    return larry([nan, nan, nan])

def functions():
    # last_rank and lastrank_decay are not included because they do a partial
    # reduction: shape (n,m) imput become (n,1)
    f = {(): ['log', 'exp', 'sqrt', 'sign', 'abs', 'sum', 'prod', 'mean',
              'median', 'std', 'var', 'min', 'max', 'demean', 'demedian',
              'zscore', 'geometric_mean'],
         (0,): ['cumsum', 'cumprod', 'ranking'],
         (1,): ['power', 'movingsum', 'movingsum_forward'],
         (2,): ['movingrank', 'quantile']} 
    return f                   
                        
def test_all_nan(): 
    "Test larry methods for proper handling of all NaN input"      
    err_msg = "%s did not return NaN"
    for parameters, methods in functions().iteritems():
        for method in methods:
            actual = getattr(lar(), method)(*parameters)                    
            if type(actual) == larry:
                desired = lar()
                yield assert_larry_equal, actual, desired, err_msg % method
            else:  
                desired = nan  
                yield assert_equal, actual, desired, err_msg % method
             

