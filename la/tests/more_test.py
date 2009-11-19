# these are tests for use with nose using generators, not for unittest.

import numpy as np
from numpy.testing import assert_  # could look for nose

np.seterr(divide='ignore')
np.seterr(invalid='ignore')
nan = np.nan

from test import printfail
from la.afunc import (covMissing, fillforward_partially, geometric_mean, lastrank, 
            lastrank_decay, median, movingrank, movingsum, movingsum_forward, 
            movingsum_old, nanmean, nanmedian, nans, nanstd, quantile, 
            ranking, ranking_1N, ranking_norm, sector_dummy, sector_mean, 
            sector_median, sector_rank, unique_sector)

funcs_tested = [sector_rank, sector_mean, sector_median,
            movingsum, ranking_1N, movingrank, ranking_norm,
                      movingsum_forward, geometric_mean, ranking, unique_sector,
                      sector_dummy]
funcs_all = [covMissing, fillforward_partially, geometric_mean, lastrank, 
            lastrank_decay, median, movingrank, movingsum, movingsum_forward, 
            movingsum_old, nanmean, nanmedian, nans, nanstd, quantile, 
            ranking, ranking_1N, ranking_norm, sector_dummy, sector_mean, 
            sector_median, sector_rank, unique_sector]

funcs_one = [covMissing, geometric_mean, lastrank, 
            median, nanmean, nanmedian, nanstd, 
            ranking, ranking_1N, ranking_norm]
#not: movingsum_old
funcs_oneint = [movingrank, movingsum, movingsum_forward,quantile, 
                fillforward_partially]
funcs_onefrac = [lastrank_decay]
funcs_sect = [sector_mean, sector_median, sector_rank]
funcs_sector = [unique_sector, sector_dummy]
funcs_special = [nans]


def check_return_array(func, args):
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(type(res1) is np.ndarray or np.isscalar(res1), \
            repr(func)+'does not return array')

def check_return_matrix(func, args):
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(type(res1) is np.matrix or np.isscalar(res1), \
            repr(func)+'does not return matrix')
    
def test_return_array():
    x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                   [1.0, 1.0, 1.0, nan, nan, nan],
                   [2.0, 2.0, 0.0, nan, 1.0, nan],
                   [3.0, 0.0, 2.0, nan, nan, nan],
                   [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                   [5.0, 5.0, 4.0, 4.0, nan, nan]])
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    x = x+1 #geometric requires >0
    
    for func in funcs_one:
        xc = x.copy()
        args = (xc,)
        yield check_return_array, func, args
        
    for func in funcs_oneint:
        xc = x.copy()
        args = (xc,2)
        yield check_return_array, func, args
        
    for func in funcs_onefrac:
        xc = x.copy()
        args = (xc, 0.5)
        yield check_return_array, func, args
        
    for func in funcs_sect:
        xc = x.copy()
        args = (xc, sectors)
        yield check_return_array, func, args
    
    yield check_return_array, sector_dummy, (sectors,)
    

def test_return_matrix():
    x = np.matrix([[0.0, 3.0, nan, nan, 0.0, nan],
                   [1.0, 1.0, 1.0, nan, nan, nan],
                   [2.0, 2.0, 0.0, nan, 1.0, nan],
                   [3.0, 0.0, 2.0, nan, nan, nan],
                   [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                   [5.0, 5.0, 4.0, 4.0, nan, nan]])
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    x = x+1 #geometric requires >0
    for func in funcs_one:
        xc = x.copy()
        args = (xc,)
        yield check_return_matrix, func, args
        
    for func in funcs_oneint:
        xc = x.copy()
        args = (xc,2)
        yield check_return_matrix, func, args
        
    for func in funcs_onefrac:
        xc = x.copy()
        args = (xc, 0.5)
        yield check_return_matrix, func, args
        
    for func in funcs_sect:
        xc = x.copy()
        args = (xc, sectors)
        yield check_return_matrix, func, args

def check_3d(func, args):
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(np.shape(res1)>0, repr(func)+'does not return array for 3d')
   
def _est_3d():
    x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                   [1.0, 1.0, 1.0, nan, nan, nan],
                   [2.0, 2.0, 0.0, nan, 1.0, nan],
                   [3.0, 0.0, 2.0, nan, nan, nan],
                   [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                   [5.0, 5.0, 4.0, 4.0, nan, nan]])
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    x = x+1 #geometric requires >0
    x = np.dstack((x,x))
    
    for func in funcs_one:
        xc = x.copy()
        args = (xc,)
        yield check_3d, func, args
        
    for func in funcs_oneint:
        xc = x.copy()
        args = (xc,2)
        yield check_3d, func, args
        
    for func in funcs_onefrac:
        xc = x.copy()
        args = (xc, 0.5)
        yield check_3d, func, args
    
    for func in funcs_sect:
        xc = x.copy()
        args = (xc, sectors)
        yield check_3d, func, args