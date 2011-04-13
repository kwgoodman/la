"Test moving (rolling) statistics."

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import assert_array_almost_equal
nan = np.nan

import bottleneck as bn

from la import larry
from la.farray import move_nanranking, move_nanmedian 


def move_unit_maker(attr, func, methods):
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
                    if method is not None:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            a = func(arr, window=w, axis=axis, method=method)
                    else:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            a = func(arr, window=w, axis=axis)
                    d = larry(arr)
                    d = getattr(d, attr)
                    if method is not None:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            d = d(window=w, axis=axis, method=method)
                    else:    
                        with np.errstate(invalid='ignore', divide='ignore'):
                            d = d(window=w, axis=axis)
                    err_msg = msg % (func.__name__, method, arr.ndim, w, axis)
                    assert_array_almost_equal(a, d.x, 10, err_msg)

def test_move_sum():
    "Test move_sum."
    methods = (None,) 
    yield move_unit_maker, 'move_sum', bn.move_nansum, methods 

def test_move_mean():
    "Test move_mean."
    methods = (None,) 
    yield move_unit_maker, 'move_mean', bn.move_nanmean, methods 

def test_move_std():
    "Test move_std."
    methods = (None,) 
    yield move_unit_maker, 'move_std', bn.move_nanstd, methods 

def test_move_max():
    "Test move_max."
    methods = (None,) 
    yield move_unit_maker, 'move_max', bn.move_nanmax, methods 

def test_move_min():
    "Test move_min."
    methods = (None,) 
    yield move_unit_maker, 'move_min', bn.move_nanmin, methods 

def test_move_ranking():
    "Test move_nanranking."
    methods = ('strides', 'loop')
    yield move_unit_maker, 'move_ranking', move_nanranking, methods 

def test_move_median():
    "Test move_median."
    methods = ('strides', 'loop') 
    yield move_unit_maker, 'move_median', move_nanmedian, methods 
