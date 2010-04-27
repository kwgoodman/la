# these are tests for use with nose using generators, not for unittest.

import numpy as np
from numpy.testing import assert_  # could look for nose
nan = np.nan

from la.util.testing import printfail
from la.util.scipy import nanstd
from la.afunc import (covMissing, push, geometric_mean, lastrank,
                      lastrank_decay, movingrank, movingsum,
                      movingsum_forward, nans, quantile, ranking,
                      group_mean, group_median, group_ranking, nanmedian)

# Functions to test
funcs_one = [geometric_mean, lastrank, nanstd, ranking, nanmedian]
funcs_oneint = [movingrank, movingsum, movingsum_forward, quantile, push]
funcs_onefrac = [lastrank_decay]
funcs_sect = [group_mean, group_median, group_ranking]

def check_return_array(func, args):
    "Check that function returns a numpy array or a scalar."
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(type(res1) is np.ndarray or np.isscalar(res1),
            repr(func) + 'does not return array or scalar')

def test_return_array():
    "Check that functions return a numpy array or a scalar."
    
    x = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                  [1.0, 1.0, 1.0, nan, nan, nan],
                  [2.0, 2.0, 9.0, nan, 1.0, nan],
                  [3.0, 9.0, 2.0, nan, nan, nan],
                  [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                  [5.0, 5.0, 4.0, 4.0, nan, nan]])          
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    
    for func in funcs_one:
        xc = x.copy()
        args = (xc,)
        yield check_return_array, func, args
        
    for func in funcs_oneint:
        xc = x.copy()
        args = (xc, 2)
        yield check_return_array, func, args
        
    for func in funcs_onefrac:
        xc = x.copy()
        args = (xc, 0.5)
        yield check_return_array, func, args
        
    for func in funcs_sect:
        xc = x.copy()
        args = (xc, sectors)
        yield check_return_array, func, args
    
def check_3d(func, args):
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(np.shape(res1)>0, repr(func)+'does not return array for 3d')
   
def test_3d():
    # many of these tests fail, skip to reduce noise during testing
    x = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                  [1.0, 1.0, 1.0, nan, nan, nan],
                  [2.0, 2.0, 0.1, nan, 1.0, nan],  # 0.0 kills geometric mean
                  [3.0, 9.0, 2.0, nan, nan, nan],
                  [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                  [5.0, 5.0, 4.0, 4.0, nan, nan]])
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    x = np.dstack((x,x))
    
    for func in funcs_one:
        xc = x.copy()
        args = (xc,)
        yield check_3d, func, args
        
    for func in funcs_oneint:
        xc = x.copy()
        args = (xc, 2)
        yield check_3d, func, args
        
    for func in funcs_onefrac:
        xc = x.copy()
        args = (xc, 0.5)
        yield check_3d, func, args
    
    for func in funcs_sect:
        xc = x.copy()
        args = (xc, sectors)
        yield check_3d, func, args
