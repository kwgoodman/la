"Test moving (rolling) statistics."

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import assert_array_almost_equal
nan = np.nan
import bottleneck as bn

from la.farray import move_median, move_nanmedian, move_func


def move_unit_maker(func, arrfunc, methods):
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
                        with np.errstate(invalid='ignore'):
                            d = move_func(arrfunc, arr, window=w, axis=axis,
                                          method='loop')
                    elif method == 'func_strides':
                        with np.errstate(invalid='ignore'):
                            d = move_func(arrfunc, arr, window=w, axis=axis,
                                          method='strides')
                    else:
                        d = func(arr, window=w, axis=axis, method=method) 
                    err_msg = msg % (func.__name__, method, arr.ndim, w, axis)
                    assert_array_almost_equal(actual, d, 10, err_msg)

def test_move_median():
    "Test move_median."
    methods = ('strides', 'func_loop', 'func_strides') 
    yield move_unit_maker, move_median, np.median, methods 

def test_move_nanmedian():
    "Test move_nanmedian."
    methods = ('strides', 'func_loop', 'func_strides') 
    yield move_unit_maker, move_nanmedian, bn.nanmedian, methods 
