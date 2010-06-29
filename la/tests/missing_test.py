"Unit tests of array functions."

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
nan = np.nan

from la import larry
from la.missing import nans, missing_marker, ismissing
                                       

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
        
    def test_nans_8(self):
        "afunc.nans_8"
        shape = 0
        dtype = bool
        assert_raises(TypeError, nans, shape, dtype)        

class Test_missing_marker(unittest.TestCase):
    "Test missing_marker"                 

    def test_missing_marker_1(self):
        "afunc.missing_marker_1"
        assert_equal(missing_marker(larry([1])), NotImplemented) 

    def test_missing_marker_2(self):
        "afunc.missing_marker_2"
        assert_equal(missing_marker(larry([1.0])), nan)
        
    def test_missing_marker_3(self):
        "afunc.missing_marker_3"
        assert_equal(missing_marker(larry([True])), NotImplemented)

    def test_missing_marker_4(self):
        "afunc.missing_marker_4"
        assert_equal(missing_marker(larry(['a'])), '')                

    def test_missing_marker_5(self):
        "afunc.missing_marker_5"
        import datetime
        d = datetime.date(2011, 1, 1)
        assert_equal(missing_marker(larry([d])), None)
        
    def test_missing_marker_1a(self):
        "afunc.missing_marker_1a"
        assert_equal(missing_marker(np.array([1])), NotImplemented) 

    def test_missing_marker_2a(self):
        "afunc.missing_marker_2a"
        assert_equal(missing_marker(np.array([1.0])), nan)
        
    def test_missing_marker_3a(self):
        "afunc.missing_marker_3a"
        assert_equal(missing_marker(np.array([True])), NotImplemented)

    def test_missing_marker_4a(self):
        "afunc.missing_marker_4a"
        assert_equal(missing_marker(np.array(['a'])), '')                

    def test_missing_marker_5a(self):
        "afunc.missing_marker_5a"
        import datetime
        d = datetime.date(2011, 1, 1)
        assert_equal(missing_marker(np.array([d])), None)         
        
class Test_ismissing(unittest.TestCase):
    "Test ismissing"                 

    def test_ismissing_1(self):
        "afunc.ismissing_1"
        assert_equal(ismissing(larry([1])), np.array([False])) 

    def test_ismissing_2(self):
        "afunc.ismissing_2"
        assert_equal(ismissing(larry([1.0])), np.array([False]))
        
    def test_ismissing_3(self):
        "afunc.ismissing_3"
        assert_equal(ismissing(larry(['str'])), np.array([False]))

    def test_ismissing_4(self):
        "afunc.ismissing_4"
        assert_equal(ismissing(larry([None])), np.array([True]))               

    def test_ismissing_5(self):
        "afunc.ismissing_5"
        import datetime
        d = datetime.date(2011, 1, 1)
        assert_equal(ismissing(larry([d])), np.array([False])) 

    def test_ismissing_6(self):
        "afunc.ismissing_6"
        assert_equal(ismissing(larry([nan])), np.array([True]))
        
    def test_ismissing_7(self):
        "afunc.ismissing_7"
        assert_equal(ismissing(larry([nan, 1])), np.array([True, False])) 
        
    def test_ismissing_8(self):
        "afunc.ismissing_8"
        assert_equal(ismissing(larry([''])), np.array([True])) 
        
    def test_ismissing_9(self):
        "afunc.ismissing_9"
        assert_equal(ismissing(larry([True])), np.array([False]))                   

    def test_ismissing_1a(self):
        "afunc.ismissing_1a"
        assert_equal(ismissing(np.array([1])), np.array([False])) 

    def test_ismissing_2a(self):
        "afunc.ismissing_2a"
        assert_equal(ismissing(np.array([1.0])), np.array([False]))
        
    def test_ismissing_3a(self):
        "afunc.ismissing_3a"
        assert_equal(ismissing(np.array(['str'])), np.array([False]))

    def test_ismissing_4a(self):
        "afunc.ismissing_4a"
        assert_equal(ismissing(np.array([None])), np.array([True]))               

    def test_ismissing_5a(self):
        "afunc.ismissing_5a"
        import datetime
        d = datetime.date(2011, 1, 1)
        assert_equal(ismissing(np.array([d])), np.array([False])) 

    def test_ismissing_6a(self):
        "afunc.ismissing_6a"
        assert_equal(ismissing(np.array([nan])), np.array([True]))
        
    def test_ismissing_7a(self):
        "afunc.ismissing_7a"
        assert_equal(ismissing(np.array([nan, 1])), np.array([True, False])) 
        
    def test_ismissing_8a(self):
        "afunc.ismissing_8a"
        assert_equal(ismissing(np.array([''])), np.array([True])) 
        
    def test_ismissing_9a(self):
        "afunc.ismissing_9a"
        assert_equal(ismissing(np.array([True])), np.array([False])) 

# Unit tests ----------------------------------------------------------------        
    
def suite():

    unit = unittest.TestLoader().loadTestsFromTestCase
    s = []
    s.append(unit(Test_nans)) 
    s.append(unit(Test_missing_marker))
    s.append(unit(Test_ismissing))             
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)

