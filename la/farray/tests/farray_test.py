"Unit tests of array functions."

# For support of python 2.5
from __future__ import with_statement

import unittest

import numpy as np
from numpy.testing import assert_almost_equal
aae = assert_almost_equal
nan = np.nan

from la.util.testing import printfail
from la.farray import group_ranking, group_mean, group_median
from la.farray import (movingsum, movingrank, movingsum_forward, ranking, 
                       geometric_mean, unique_group, correlation, lastrank)

# Sector functions ----------------------------------------------------------

class Test_group_ranking(unittest.TestCase):
    "Test farray.group_ranking"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_ranking_1(self):
        "farray.group_ranking #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = np.array([[-1.0, 0.0,  nan, nan, -1.0, nan],
                            [-1.0, 1.0, -1.0, nan,  nan, nan],
                            [ 0.0,-1.0, -1.0, nan,  0.0, nan],
                            [ 1.0,-1.0,  1.0, nan,  nan, nan],
                            [ 1.0, 1.0,  1.0, 0.0,  1.0, nan],
                            [ 0.0, 0.0,  0.0, 0.0,  nan, nan]])
        with np.errstate(invalid='ignore'):
            actual = group_ranking(self.x, sectors)
        assert_almost_equal(actual, desired)
        
    def test_group_ranking_2(self):
        "farray.group_ranking #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        desired = np.array([[-1.0,  0.0,  nan, nan,-1.0, nan],
                            [-1.0,  1.0, -1.0, nan, nan, nan],
                            [ 0.0, -1.0, -1.0, nan, 0.0, nan],
                            [ 1.0, -1.0,  1.0, nan, nan, nan],
                            [ 1.0,  1.0,  1.0, 0.0, 1.0, nan],
                            [ nan,  nan,  nan, nan, nan, nan]])
        with np.errstate(invalid='ignore'):
            actual = group_ranking(self.x, sectors)
        assert_almost_equal(actual, desired)


class Test_group_mean(unittest.TestCase):
    "Test farray.group_mean"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_mean_1(self):
        "farray.group_mean #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
        actual = group_mean(self.x, sectors)
        assert_almost_equal(actual, desired)
        
    def test_group_mean_2(self):
        "farray.group_mean #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        desired = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ nan, nan,  nan, nan,  nan, nan]])
        actual = group_mean(self.x, sectors)
        assert_almost_equal(actual, desired)

    def test_group_mean_3(self):
        "farray.group_mean #3"
        sectors = ['a', 'b', 'a', 'b', 'a']
        x = np.array([[1,2],
                      [3,4],
                      [6,7],
                      [0,0],
                      [8,-1]])
        desired = np.array([[5.0, 8/3.0],
                            [1.5, 2.0],
                            [5.0, 8/3.0],
                            [1.5, 2.0],
                            [5.0, 8/3.0]])
        actual = group_mean(x, sectors)
        assert_almost_equal(actual, desired)           


class Test_group_median(unittest.TestCase):
    "Test farray.group_median"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_median_1(self):
        "farray.group_median #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 2.0, 0.5,  1.5, nan,  nan, nan],
                            [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                            [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
        actual = group_median(self.x, sectors)
        assert_almost_equal(actual, desired)
        
    def test_group_median_2(self):
        "farray.group_median #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        desired = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ nan, nan,  nan, nan,  nan, nan]])
        actual = group_median(self.x, sectors)
        assert_almost_equal(actual, desired)

    def test_group_median_3(self):
        "farray.group_median #3"
        sectors = ['a', 'b', 'a', 'b', 'a']
        x = np.array([[1,2],
                      [3,4],
                      [6,7],
                      [0,0],
                      [8,-1]])
        desired = np.array([[6.0,2.0],
                            [1.5,2.0],
                            [6.0,2.0],
                            [1.5,2.0],
                            [6.0,2.0]])
        actual = group_median(x, sectors)
        assert_almost_equal(actual, desired)


class Test_sector_oth(unittest.TestCase):
    "Test farray.group_mean"
        
    def test_sector_unique_1(self):
        "farray.unique_group #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = ['a', 'b', 'c']
        actual = unique_group(sectors)
        msg = printfail(desired, actual)   
        self.assert_(desired == actual, msg)      

    
# Normalize functions -------------------------------------------------------

class Test_ranking(unittest.TestCase):
    "Test farray.ranking"

    def test_ranking_1(self):
        "farray.ranking_1"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                            [ 0.0,  -1.0,   nan,   nan,   nan],
                            [ 1.0,   1.0,   1.0,   0.0,   nan]])   
        with np.errstate(invalid='ignore'):
            actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired) 

    def test_ranking_2(self):
        "farray.ranking_2"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                            [ 0.0,  -1.0,   nan,   nan,   nan],
                            [ 1.0,   1.0,   1.0,   0.0,   nan]])   
        with np.errstate(invalid='ignore'):
            actual = ranking(x)
        assert_almost_equal(actual, desired) 

    def test_ranking_3(self):
        "farray.ranking_3"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan],
                      [ 4.0,   2.0,   3.0, 1.0  , 0.0  ]])   
        desired = np.array([[-1.0,   nan,   1.0,   nan,   nan],
                            [ 0.0,   0.0,   nan,   nan,   nan],
                            [ 0.0,   0.0,   0.0,   0.0,   nan],
                            [ 1.0,   0.0,   0.5,  -0.5,  -1.0]])    
        actual = ranking(x, axis=1)
        assert_almost_equal(actual, desired)
        
    def test_ranking_4(self):
        "farray.ranking_4"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([1.0,-1.0, 0.0])[:,None]
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired) 

    def test_ranking_5(self):
        "farray.ranking_5"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([0.0, 0.0, 0.0])[:,None]
        with np.errstate(invalid='ignore'):
            actual = ranking(x, axis=1)
        assert_almost_equal(actual, desired)
        
    def test_ranking_6(self):
        "farray.ranking_6"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        desired = np.array([[ 0.0,   nan,   0.5,  nan,   nan],
                            [ 0.0,  -1.0,   nan,  nan,   nan],
                            [ 0.0,   0.0,  -1.0,  1.0,   nan],
                            [ 0.0,   1.0,   0.5, -1.0,   0.0]])     
        with np.errstate(invalid='ignore'):
            actual = ranking(x)
        assert_almost_equal(actual, desired)       

    def test_ranking_7(self):
        "farray.ranking_7"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        desired = np.array([[ 0.0,   nan ,   0.0,  nan  ,   nan],
                            [ 0.0,   0.0 ,   nan,  nan  ,   nan],
                            [-1.0/3, 2.0/3, -1.0,  2.0/3,   nan],
                            [ 0.0,   1.0 ,   0.0,  0.0  ,  -1.0]])  
        actual = ranking(x, 1)
        assert_almost_equal(actual, desired)

    def test_ranking_8(self):
        "farray.ranking_8"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]])   
        desired = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                            [-2.0/3, -2.0/3, -1.0/3,   0.0],
                            [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                            [ 2.0/3,    1.0,  2.0/3,   1.0]])
        actual = ranking(x, 0)
        assert_almost_equal(actual, desired)
        
    def test_ranking_9(self):
        "farray.ranking_9"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]]) 
        x = x.T  
        desired = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                            [-2.0/3, -2.0/3, -1.0/3,   0.0],
                            [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                            [ 2.0/3,    1.0,  2.0/3,   1.0]])
        desired = desired.T                                       
        actual = ranking(x, 1)
        assert_almost_equal(actual, desired)              

    def test_ranking_10(self):
        "farray.ranking_10"
        x = np.array([[ nan],
                      [ nan],
                      [ nan]])  
        desired = np.array([[ nan],
                            [ nan],
                            [ nan]])
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_11(self):
        "farray.ranking_11"
        x = np.array([[ nan, nan],
                      [ nan, nan],
                      [ nan, nan]])  
        desired = np.array([[ nan, nan],
                            [ nan, nan],
                            [ nan, nan]])
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_12(self):
        "farray.ranking_12"
        x = np.array([[ nan, nan, nan]])  
        desired = np.array([[ nan, nan, nan]])      
        actual = ranking(x, axis=1)
        assert_almost_equal(actual, desired)
        
    def test_ranking_13(self):
        "farray.ranking_13"
        x = np.array([ 1.0, np.inf, 2.0])  
        desired = np.array([-1.0, 1.0, 0.0])     
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)        

    def test_ranking_14(self):
        "farray.ranking_14"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        desired = np.array([-1.0, nan, 0.0, 1.0])    
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_15(self):
        "farray.ranking_15"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        desired = np.array([-1.0, nan, 0.0, 1.0])    
        actual = ranking(x, axis=None)
        assert_almost_equal(actual, desired)

    def test_ranking_16(self):
        "farray.ranking_16"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   1.0,   2.0],
                      [ 2.0,   2.0,   1.0,   2.0]]) 
        desired = np.array([[-1.0, -1.0,   -1.0,  -1.0],
                            [-1.0, -1.0,    1.0,   1.0],
                            [ 1.0,  1.0,   -1.0,   1.0],
                            [ 1.0,  1.0,   -1.0,   1.0]])
        desired *= 8.0 / 15
        actual = ranking(x, axis=None)
        assert_almost_equal(actual, desired)

    def test_ranking_17(self):
        "farray.ranking_17"
        x = np.array([[ nan,   1.0,   1.0,   1.0],
                      [ 1.0,   1.5,   2.0,   2.0],
                      [ 2.0,   nan,   1.0,   2.0],
                      [ 2.0,   1.5,   1.0,   2.0]]) 
        desired = np.array([[ nan, -1.0,   -1.0,  -1.0],
                            [-1.0,  0.0,    1.0,   1.0],
                            [ 1.0,  nan,   -1.0,   1.0],
                            [ 1.0,  0.0,   -1.0,   1.0]])
        desired *= 8.0 / 13
        actual = ranking(x, axis=None)
        assert_almost_equal(actual, desired)
        
class Test_geometric_mean(unittest.TestCase):
    "Test farray.geometric_mean"

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 2.0, 8.0],
                           [2.0, 4.0, 8.0, 2.0, 1.0]])
        self.xnan = np.array([[nan, nan, nan, nan, nan],
                              [nan, nan, nan, nan, nan]])
        self.x2 = np.array([[ 2.0,  2.0],
                            [ 1.0,  3.0],
                            [ 3.0,  1.0]])                    

    def test_geometric_mean_1(self):
        "farray.geometric_mean #1"
        desired = np.array([ 2.0, 1.73205081, 1.73205081])
        actual =  geometric_mean(self.x2, 1)
        assert_almost_equal(actual, desired)

    def test_geometric_mean_2(self):
        "farray.geometric_mean #2"
        desired = np.array([ 2.0, 1.73205081, 1.73205081])
        actual = geometric_mean(self.x2)
        assert_almost_equal(actual, desired)   

    def test_geometric_mean_3(self):
        "farray.geometric_mean #3"
        desired = np.array([ 1.81712059, 1.81712059])
        actual = geometric_mean(self.x2, 0)
        assert_almost_equal(actual, desired) 

    def test_geometric_mean_4(self):
        "farray.geometric_mean #4"
        desired = np.array([nan, nan])
        with np.errstate(invalid='ignore', divide='ignore'):
            actual = geometric_mean(self.xnan)
        assert_almost_equal(actual, desired)
        
    def test_geometric_mean_5(self):
        "farray.geometric_mean #5"
        desired = np.array([ 3.1301691601465746, 2.6390158215457888])
        actual = geometric_mean(self.x, 1)
        assert_almost_equal(actual, desired)

    def test_geometric_mean_6(self):
        "farray.geometric_mean #6"
        desired = np.array([ 1.4142135623730951, 4.0, 6.9282032302755088, 2.0,
                                                         2.8284271247461903])
        actual = geometric_mean(self.x, 0)
        assert_almost_equal(actual, desired)
        
    def test_geometric_mean_7(self):
        "farray.geometric_mean #7"
        x = np.array([[1e200, 1e200]])
        desired = 1e200
        actual = geometric_mean(x)
        msg = printfail(desired, actual)
        self.assert_((abs(desired - actual) < 1e187).all(), msg)         
        
class Test_movingsum(unittest.TestCase):
    "Test farray.movingsum"       

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[  nan,  nan,  nan,  nan,  nan],
                              [  nan,  nan,  nan,  nan,  nan]])
        self.window = 2
        self.x2 = np.array([[ 2.0,  2.0],
                            [ 1.0,  3.0],
                            [ 3.0,  1.0]]) 

    def test_movingsum_1(self):
        "farray.movingsum #1"  
        desired = self.xnan 
        with np.errstate(invalid='ignore'):
            actual = movingsum(self.xnan, self.window, norm=True)
        assert_almost_equal(actual, desired)             

    def test_movingsum_2(self):
        "farray.movingsum #2"    
        desired = self.xnan
        actual = movingsum(self.xnan, self.window, norm=False)
        assert_almost_equal(actual, desired)   

    def test_movingsum_3(self):
        "farray.movingsum #3"    
        desired = np.array([[  nan, 2.0, 12.0, 6.0, 8.0],
                            [  nan, 6.0, 12.0, 8.0,-1.0]])   
        actual = movingsum(self.x, self.window, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_4(self):
        "farray.movingsum #4"   
        desired = np.array([[  nan, 1.0,  6.0, 6.0, 8.0],
                            [  nan, 6.0, 12.0, 8.0,-1.0]])
        actual = movingsum(self.x, self.window, norm=False)
        assert_almost_equal(actual, desired) 

    def test_movingsum_5(self):
        "farray.movingsum #5"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                            [3.0,  8.0,  14.0, 0.0,  7.0]])
        actual = movingsum(self.x, self.window, axis=0, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_6(self):
        "farray.movingsum #6"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                            [3.0,  4.0,  14.0, 0.0,  7.0]])
        actual = movingsum(self.x, self.window, axis=0, norm=False)
        assert_almost_equal(actual, desired) 
        
    def test_movingsum_7(self):
        "farray.movingsum #7"  
        desired = np.array([[nan, 4.0],
                            [nan, 4.0],
                            [nan, 4.0]])
        actual = movingsum(self.x2, self.window)
        assert_almost_equal(actual, desired) 

class Test_movingsum_forward(unittest.TestCase):
    "Test farray.movingsum_forward"
 
    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[  nan,  nan,  nan,  nan,  nan],
                              [  nan,  nan,  nan,  nan,  nan]])
        self.window = 2

    def test_movingsum_forward_1(self):
        "farray.movingsum_forward #1"
        desired = np.array([[2.0, 12.0, 6.0, 8.0, nan],
                            [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0            
        actual = movingsum_forward(self.x, self.window, skip, norm=True)
        assert_almost_equal(actual, desired) 
        
    def test_movingsum_forward_2(self):
        "farray.movingsum_forward #2"    
        desired = np.array([[1.0,  6.0, 6.0, 8.0, nan],
                            [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0                     
        actual = movingsum_forward(self.x, self.window, skip, norm=False)
        assert_almost_equal(actual, desired)        

    def test_movingsum_forward_3(self):
        "farray.movingsum_forward #3"    
        desired = np.array([[12.0, 6.0, 8.0, nan, nan],
                            [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                   
        actual = movingsum_forward(self.x, self.window, skip, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_forward_4(self):
        "farray.movingsum_forward #4"    
        desired = np.array([[ 6.0, 6.0, 8.0, nan, nan],
                            [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                     
        actual = movingsum_forward(self.x, self.window, skip, norm=False)
        assert_almost_equal(actual, desired) 

    def test_movingsum_forward_5(self):
        "farray.movingsum_forward #5"  
        desired = np.array([[2.0, 4.0, 8.0, 0.0,-1.0],
                            [nan, nan, nan, nan, nan]])
        skip = 1
        window = 1                    
        actual = movingsum_forward(self.x, window, skip, axis=0)
        assert_almost_equal(actual, desired)          

class Test_movingrank(unittest.TestCase):
    "Test movingrank"

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[nan, nan, nan, nan, nan],
                              [nan, nan, nan, nan, nan]])
        self.window = 2
        self.x2 = np.array([[nan, 2.0],
                            [1.0, 3.0],
                            [3.0, 1.0]])                  
    
    def test_movingrank_1(self):
        "farray.movingrank #1"    
        desired = self.xnan 
        with np.errstate(invalid='ignore'):
            actual = movingrank(self.xnan, self.window)
        assert_almost_equal(actual, desired) 
    
    def test_movingrank_2(self):
        "farray.movingrank #2"    
        desired = np.array([[  nan,  nan,  nan,-1.0,1.0],
                           [  nan,1.0,1.0,-1.0,-1.0]]) 
        with np.errstate(invalid='ignore', divide='ignore'):
            actual = movingrank(self.x, self.window)
        assert_almost_equal(actual, desired)          

    def test_movingrank_3(self):
        "farray.movingrank #3"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                           [1.0,  nan,  1.0,  0.0,  -1.0]])
        with np.errstate(invalid='ignore'):
            actual = movingrank(self.x, self.window, axis=0)
        assert_almost_equal(actual, desired) 
        
    def test_movingrank_4(self):
        "farray.movingrank #4"    
        desired = np.array([[nan,  nan],
                           [nan,  1.0],
                           [nan, -1.0]])
        with np.errstate(invalid='ignore', divide='ignore'):
            actual = movingrank(self.x2, self.window)
        assert_almost_equal(actual, desired)
        
class Test_correlation(unittest.TestCase):
    "Test farray.correlation"
    
    def setUp(self):
        self.a1 = np.array([[1, 1, 1],
                            [1, 2, 3],
                            [1, 2, 3],
                            [2, 2, 1]])
        self.a2 = np.array([[1, 1, 1],
                            [2, 3, 4],
                            [4, 3, 2],
                            [1, 2, 2]])
        self.b1 = np.array([[nan, 1,    1,   1],
                            [1,   nan,  2,   3],
                            [1,   2,    nan, 3],
                            [2,   2,    1,   nan]])
        self.b2 = np.array([[nan, 1,    1,   1],
                            [2,   nan,  3,   4],
                            [4,   3,    nan, 2],
                            [1,   2,    2,   nan]])
                                                          
    def test_correlation_1(self):
        "farray.correlation_1"
        x = np.array([])
        y = np.array([])
        corr = correlation(x, y) 
        aae(corr, np.nan, err_msg="Empty correlation should be NaN")
        
    def test_correlation_2(self):
        "farray.correlation_2"        
        x = np.array([nan, nan])
        y = np.array([nan, nan])
        corr = correlation(x, y) 
        aae(corr, np.nan, err_msg="All NaN correlation should be NaN")
        
    def test_correlation_3(self):
        "farray.correlation_3"
        x = self.a1[0,:]
        y = self.a2[0,:]
        corr = correlation(x, y) 
        aae(corr, np.nan, err_msg="Correlation undefined")
        x = self.b1[0,:]
        y = self.b2[0,:]
        corr = correlation(x, y) 
        aae(corr, np.nan, err_msg="Correlation undefined")        

    def test_correlation_4(self):
        "farray.correlation_4"
        x = self.a1[1,:]
        y = self.a2[1,:]
        corr = correlation(x, y) 
        aae(corr, 1, err_msg="Perfect +1 correation")
        x = self.b1[1,:]
        y = self.b2[1,:]
        corr = correlation(x, y) 
        aae(corr, 1, err_msg="Perfect +1 correation")

    def test_correlation_5(self):
        "farray.correlation_5"
        x = self.a1[2,:]
        y = self.a2[2,:]
        corr = correlation(x, y) 
        aae(corr, -1, err_msg="Perfect -1 correation")
        x = self.b1[2,:]
        y = self.b2[2,:]
        corr = correlation(x, y) 
        aae(corr, -1, err_msg="Perfect -1 correation")

    def test_correlation_6(self):
        "farray.correlation_6"
        x = self.a1[3,:]
        y = self.a2[3,:]
        corr = correlation(x, y) 
        aae(corr, -0.5, err_msg="-0.5 correation")
        x = self.b1[3,:]
        y = self.b2[3,:]
        corr = correlation(x, y) 
        aae(corr, -0.5, err_msg="-0.5 correation")

    def test_correlation_7(self):
        "farray.correlation_7"
        x = self.a1
        y = self.a2
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        x = self.b1
        y = self.b2
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")

    def test_correlation_8(self):
        "farray.correlation_8"
        x = self.a1.T
        y = self.a2.T
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=0)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        x = self.b1.T
        y = self.b2.T
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=0)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")

    def test_correlation_9(self):
        "farray.correlation_9"
        x = self.a1
        y = self.a2
        x2 = np.empty((2, x.shape[0], x.shape[1]))
        x2[0] = x
        x2[1] = x
        y2 = np.empty((2, y.shape[0], y.shape[1]))
        y2[0] = y
        y2[1] = y        
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=-1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        x = self.b1
        y = self.b2
        x2 = np.empty((2, x.shape[0], x.shape[1]))
        x2[0] = x
        x2[1] = x
        y2 = np.empty((2, y.shape[0], y.shape[1]))
        y2[0] = y
        y2[1] = y        
        with np.errstate(invalid='ignore'):
            corr = correlation(x, y, axis=-1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")

class Test_lastrank(unittest.TestCase):
    "Test farray.lastrank"
    
    def test_lastrank_1(self):
        "farray.lastrank_1"
        s = lastrank(np.array([1, 3, 2]))
        aae(s, 0.0, err_msg="1d lastrank fail")

    def test_lastrank_2(self):
        "farray.lastrank_2"
        s = lastrank(np.array([[1, 3, 2], [1, 2, 3]]))
        aae(s, np.array([0.0, 1.0]), err_msg="2d lastrank fail")

    def test_lastrank_3(self):
        "farray.lastrank_3"
        s = lastrank(np.array([]))
        aae(s, np.nan, err_msg="size 0 lastrank fail")

# Unit tests ---------------------------------------------------------------- 
    
def suite():

    unit = unittest.TestLoader().loadTestsFromTestCase
    s = []
    
    # Sector functions
    s.append(unit(Test_group_ranking))
    s.append(unit(Test_group_mean)) 
    s.append(unit(Test_group_median)) 
    
    # Normalize functions
    s.append(unit(Test_ranking))    
    s.append(unit(Test_geometric_mean))
    s.append(unit(Test_movingsum))
    s.append(unit(Test_movingsum_forward))
    s.append(unit(Test_movingrank))
    s.append(unit(Test_lastrank))
    
    # Calc function
    s.append(unit(Test_correlation))              
         
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)

