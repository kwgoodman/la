"Test moving (rolling) statistics."

import numpy as np
from numpy.testing import assert_array_almost_equal
nan = np.nan

from la.farray import (mov_sum, mov_nansum, mov_mean, mov_nanmean,
                       mov_var, mov_nanvar, mov_std, mov_nanstd,
                       mov_min, mov_nanmin, mov_max, mov_nanmax,
                       mov_nanranking, mov_count, mov_median, mov_nanmedian,
                       mov_func, nanmean, nanmedian, nanstd, lastrank)
from la.farray import nanvar
from la.missing import ismissing


def mov_unit_maker(func, arrfunc, methods):
    "Test that different mov methods give the same results on 2d input."
    arr1 = np.array([1, 2, 3, 4, 5, 6, nan, nan, 7, 8, 9])
    arr2 = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                     [1.0, 1.0, 1.0, nan, nan, nan],
                     [2.0, 2.0, 0.1, nan, 1.0, nan],
                     [3.0, 9.0, 2.0, nan, nan, nan],
                     [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                     [5.0, 5.0, 4.0, 4.0, nan, nan]]) 
    arr3 = np.arange(60).reshape(3, 4, 5)
    arr4 = np.array([nan, nan, nan])
    arrs = [arr1, arr2, arr3, arr4]
    msg = '\nfunc %s | method %s | nd %d | window %d | axis %d\n'
    for arr in arrs:
        for axis in range(arr.ndim):
            for w in range(1, arr.shape[axis]):
                actual = func(arr, window=w, axis=axis, method='loop')
                for method in methods:
                    if method == 'func_loop':
                        d = mov_func(arrfunc, arr, window=w, axis=axis,
                                     method='loop')
                    elif method == 'func_strides':
                        d = mov_func(arrfunc, arr, window=w, axis=axis,
                                     method='strides')
                    else:
                        d = func(arr, window=w, axis=axis, method=method) 
                    err_msg = msg % (func.__name__, method, arr.ndim, w, axis)
                    assert_array_almost_equal(actual, d, 10, err_msg)

def test_mov_sum():
    "Test mov_sum."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_sum, np.sum, methods 

def test_mov_nansum():
    "Test mov_nansum."
    methods = ('cumsum', 'filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nansum, np.nansum, methods 

def test_mov_mean():
    "Test mov_mean."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_mean, np.mean, methods 

def test_mov_nanmean():
    "Test mov_nanmean."
    methods = ('cumsum', 'filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanmean, nanmean, methods 

def test_mov_var():
    "Test mov_var."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_var, np.var, methods 

def test_mov_nanvar():
    "Test mov_nanvar."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanvar, nanvar, methods 

def test_mov_std():
    "Test mov_std."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_std, np.std, methods 

def test_mov_nanstd():
    "Test mov_nanstd."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanstd, nanstd, methods 

def test_mov_max():
    "Test mov_max."
    # The 'filter' method is not tested since it doesn't give the same
    # output as the other methods when there are nans
    methods = ('strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_max, np.max, methods 

def test_mov_nanmax():
    "Test mov_nanmax."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanmax, np.nanmax, methods 

def test_mov_min():
    "Test mov_min."
    # The 'filter' method is not tested since it doesn't give the same
    # output as the other methods when there are nans
    methods = ('strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_min, np.min, methods 

def test_mov_nanmin():
    "Test mov_nanmin."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanmin, np.nanmin, methods 

def test_mov_nanranking():
    "Test mov_nanranking."
    methods = ('strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanranking, lastrank, methods 

def test_mov_count():
    "Test mov_count."
    methods = ('filter', 'strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_count, counter, methods

def test_mov_median():
    "Test mov_median."
    methods = ('strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_median, np.median, methods 

def test_mov_nanmedian():
    "Test mov_nanmedian."
    methods = ('strides', 'func_loop', 'func_strides') 
    yield mov_unit_maker, mov_nanmedian, nanmedian, methods 

# Utility -------------------------------------------------------------------

def counter(arr, axis):
    return (~ismissing(arr)).sum(axis=axis)

    
