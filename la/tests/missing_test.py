"Unit tests of array functions."

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
nan = np.nan

from la.missing import nans
                                       

class Test_nans(unittest.TestCase):
    "Test nans"                 

    def test_nans_1(self):
        "afunc.nans_1"
        shape = (2,)
        actual = nans(shape)    
        desired = np.array([nan, nan])
        assert_almost_equal(actual, desired) 
    
    def test_nans_2(self):
        "afunc.nans_2"
        shape = (2,)
        dtype = float
        actual = nans(shape, dtype)    
        desired = np.array([nan, nan]) 
        assert_almost_equal(actual, desired)       

    def test_nans_3(self):
        "afunc.nans_3"
        shape = (2,)
        dtype = str
        actual = nans(shape, dtype)    
        desired = np.array(['', ''])
        assert_equal(actual, desired) 

    def test_nans_4(self):
        "afunc.nans_4"
        shape = (2,)
        dtype = object
        actual = nans(shape, dtype)
        desired = np.array([None, None])
        assert_equal(actual, desired)

    def test_nans_5(self):
        "afunc.nans_5"
        shape = (2, 4, 3)
        dtype = object
        actual = nans(shape, dtype)    
        desired = np.zeros(shape, dtype=dtype)
        desired[:] = None
        assert_equal(actual, desired)

    def test_nans_6(self):
        "afunc.nans_6"
        shape = 0
        dtype = str
        actual = nans(shape, dtype)    
        desired = np.zeros(shape, dtype=dtype)
        assert_equal(actual, desired) 

    def test_nans_7(self):
        "afunc.nans_7"
        shape = 0
        dtype = int
        assert_raises(TypeError, nans, shape, dtype)
        

# Unit tests ----------------------------------------------------------------        
    
def suite():

    unit = unittest.TestLoader().loadTestsFromTestCase
    s = []
    s.append(unit(Test_nans))                 
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)

