"Test moving (rolling) statistics."

import numpy as np
from numpy.testing import assert_array_almost_equal
nan = np.nan

from la import larry
from la.farray import (mov_sum, mov_nansum, mov_mean, mov_nanmean,
                       mov_var, mov_nanvar, mov_std, mov_nanstd,
                       mov_min, mov_nanmin, mov_max, mov_nanmax,
                       mov_nanranking, mov_count, mov_median, mov_nanmedian,
                       mov_func, nanmean, nanmedian, nanstd, lastrank)
from la.farray import nanvar
from la.missing import ismissing


def mov_unit_maker(attr, func, methods):
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
                for method in methods:
                    a = func(arr, window=w, axis=axis, method=method)
                    d = larry(arr)
                    d = getattr(d, attr)
                    d = d(window=w, axis=axis, method=method)
                    err_msg = msg % (func.__name__, method, arr.ndim, w, axis)
                    assert_array_almost_equal(a, d.x, 10, err_msg)

def test_mov_sum():
    "Test mov_sum."
    methods = ('filter', 'strides', 'loop', 'cumsum') 
    yield mov_unit_maker, 'mov_sum', mov_nansum, methods 

def test_mov_mean():
    "Test mov_mean."
    methods = ('filter', 'strides', 'loop', 'cumsum') 
    yield mov_unit_maker, 'mov_mean', mov_nanmean, methods 

def test_mov_var():
    "Test mov_var."
    methods = ('filter', 'strides', 'loop') 
    yield mov_unit_maker, 'mov_var', mov_nanvar, methods 

def test_mov_std():
    "Test mov_std."
    methods = ('filter', 'strides', 'loop') 
    yield mov_unit_maker, 'mov_std', mov_nanstd, methods 

def test_mov_max():
    "Test mov_max."
    methods = ('filter', 'strides', 'loop') 
    yield mov_unit_maker, 'mov_max', mov_nanmax, methods 

def test_mov_min():
    "Test mov_min."
    methods = ('filter', 'strides', 'loop') 
    yield mov_unit_maker, 'mov_min', mov_nanmin, methods 

def test_mov_ranking():
    "Test mov_nanranking."
    methods = ('strides', 'loop') 
    yield mov_unit_maker, 'mov_ranking', mov_nanranking, methods 

def test_mov_count():
    "Test mov_count."
    methods = ('filter', 'strides', 'loop') 
    yield mov_unit_maker, 'mov_count', mov_count, methods

def test_mov_median():
    "Test mov_median."
    methods = ('strides', 'loop') 
    yield mov_unit_maker, 'mov_median', mov_nanmedian, methods 
