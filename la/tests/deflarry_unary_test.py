
# For support of python 2.5
from __future__ import with_statement
from nose.tools import assert_raises
import numpy as np

import la
from la.util.testing import assert_larry_equal as ale

DTYPES = [np.int16, np.int32, np.int64, np.uint64, np.float32, np.float64,
          np.complex64]

def arrays(dtypes=DTYPES, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0,0), (2,0), (2,0,1)]}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3)]}
    ss[3] = {'size':  6, 'shapes': [(1,2,3)]}
    ss[4] = {'size': 24, 'shapes': [(1,2,3,4)]} 
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            yield np.zeros(size, dtype=dtype)
            yield np.ones(size, dtype=dtype)
            a = np.arange(size, dtype=dtype)
            for shape in shapes:
                a = a.reshape(shape)
                yield a
                yield -a
            if issubclass(a.dtype.type, np.inexact): 
                if nans:
                    for i in range(a.size):
                        a.flat[i] = np.nan
                        yield a
                        yield -a
                """
                # np.testing.assert_almost_equal fails on np.inf
                # for numpy 1.6.1
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                    yield -a
                """    

def unit_maker(name, npfunc, arg={'la':[], 'np':[]}, imag=True):
    "Test that larry.xxx gives the same output as np.xxx."
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays(nans=True)):
        if not imag and (arr.dtype == np.complex64):
            continue
        original = la.larry(arr.copy())
        method = getattr(original, name)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            desired = la.larry(npfunc(arr.copy(), *arg['np']))
            actual = method(*arg['la'])
        ale(actual, desired, msg=name, original=original)

def test_log():
    yield unit_maker, "log", np.log

def test_exp():
    yield unit_maker, "exp", np.exp

def test_sqrt():
    yield unit_maker, "sqrt", np.sqrt

def test_sign():
    yield unit_maker, "sign", np.sign

def test_power():
    yield unit_maker, "power", np.power, {'la':[1.3], 'np':[1.3]}

def test___pow__():
    yield unit_maker, "__pow__", np.power, {'la':[1.3], 'np':[1.3]}

def test_cumsum():
    yield unit_maker, "cumsum", np.cumsum, {'la':[0], 'np':[0]}
    yield unit_maker, "cumsum", np.cumsum, {'la':[-1], 'np':[-1]}

def test_cumprod():
    yield unit_maker, "cumprod", np.cumprod, {'la':[0], 'np':[0]}
    yield unit_maker, "cumprod", np.cumprod, {'la':[-1], 'np':[-1]}

def test_clip():
    yield unit_maker, "clip", np.clip, {'la':[-1,1], 'np':[-1,1]}

def test___neg__():
    yield unit_maker, "__neg__", np.negative

def test___pos__():
    def positive(a):
        return a.copy()
    yield unit_maker, "__pos__", positive

def test_abs():
    yield unit_maker, "abs", np.absolute, {'la':[], 'np':[]}, False

def test___abs__():
    yield unit_maker, "__abs__", np.absolute, {'la':[], 'np':[]}, False

def test_isnan():
    yield unit_maker, "isnan", np.isnan

def test_isfinite():
    yield unit_maker, "isfinite", np.isfinite

def test_isinf():
    yield unit_maker, "isinf", np.isinf

def test_nan_replace():
    def nanreplace(a, w=0):
        a = a.copy()
        a[np.isnan(a)] = w
        return a
    yield unit_maker, "nan_replace", nanreplace

def test_invert():
    o = la.larry([True, False])
    d = la.larry([False, True])
    ale(o.invert(), d, 'invert', original=o)                  

def test___invert__():
    o = la.larry([True, False])
    d = la.larry([False, True])
    ale(~o, d, '__invert__', original=o)

# assert raises -------------------------------------------------------------

def test_cumsum_raises():
    "cumsum raises"
    y = la.larry([0, 1])
    assert_raises(ValueError, y.cumsum, None)      

def test_cumprod_raises():
    "cumprod raises"
    y = la.larry([0, 1])
    assert_raises(ValueError, y.cumprod, None)      

def test_clip_raises():
    "clip raises"
    y = la.larry([0, 1])
    assert_raises(ValueError, y.clip, 3, 2)      

def test_invert_raises():
    "invert raises"
    y = la.larry([0, 1])
    assert_raises(TypeError, y.invert)

def test___invert__raises():
    "invert raises"
    y = la.larry([0, 1])
    assert_raises(TypeError, y.invert)

