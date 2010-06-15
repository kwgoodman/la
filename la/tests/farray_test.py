"Unit tests of array functions."

import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
aae = assert_almost_equal
nan = np.nan

from la.util.testing import printfail
from la.farray import group_ranking, group_mean, group_median
from la.farray import (movingsum, movingrank, movingsum_forward, ranking, 
                       geometric_mean, unique_group, correlation)

# Sector functions ----------------------------------------------------------

class Test_group_ranking(unittest.TestCase):
    "Test afunc.group_ranking"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_ranking_1(self):
        "afunc.group_ranking #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = np.array([[-1.0, 0.0,  nan, nan, -1.0, nan],
                            [-1.0, 1.0, -1.0, nan,  nan, nan],
                            [ 0.0,-1.0, -1.0, nan,  0.0, nan],
                            [ 1.0,-1.0,  1.0, nan,  nan, nan],
                            [ 1.0, 1.0,  1.0, 0.0,  1.0, nan],
                            [ 0.0, 0.0,  0.0, 0.0,  nan, nan]])
        actual = group_ranking(self.x, sectors)
        assert_almost_equal(actual, desired)
        
    def test_group_ranking_2(self):
        "afunc.group_ranking #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        desired = np.array([[-1.0,  0.0,  nan, nan,-1.0, nan],
                            [-1.0,  1.0, -1.0, nan, nan, nan],
                            [ 0.0, -1.0, -1.0, nan, 0.0, nan],
                            [ 1.0, -1.0,  1.0, nan, nan, nan],
                            [ 1.0,  1.0,  1.0, 0.0, 1.0, nan],
                            [ nan,  nan,  nan, nan, nan, nan]])
        actual = group_ranking(self.x, sectors)
        assert_almost_equal(actual, desired)


class Test_group_mean(unittest.TestCase):
    "Test afunc.group_mean"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_mean_1(self):
        "afunc.group_mean #1"
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
        "afunc.group_mean #2"
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
        "afunc.group_mean #3"
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
    "Test afunc.group_median"
    
    def setUp(self):
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_median_1(self):
        "afunc.group_median #1"
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
        "afunc.group_median #2"
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
        "afunc.group_median #3"
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
    "Test afunc.group_mean"
        
    def test_sector_unique_1(self):
        "afunc.unique_group #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        desired = ['a', 'b', 'c']
        actual = unique_group(sectors)
        msg = printfail(desired, actual)   
        self.assert_(desired == actual, msg)      

    
# Normalize functions -------------------------------------------------------

class Test_ranking(unittest.TestCase):
    "Test afunc.ranking"

    def test_ranking_1(self):
        "afunc.ranking #1"
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])
        desired = np.array([[ 0.0, nan, 0.0, nan, nan],
                            [ 1.0, 0.0, nan, nan, nan],
                            [ 2.0, 2.0, 2.0, 1.0, nan]])                     
        actual = ranking(x, axis=0, norm='0,N-1', ties=False)
        assert_almost_equal(actual, desired) 

    def test_ranking_2(self):
        "afunc.ranking #2"
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])
        desired = np.array([[ 0.0, nan, 0.0, nan, nan],
                            [ 1.0, 0.0, nan, nan, nan],
                            [ 2.0, 2.0, 2.0, 1.0, nan]])                     
        actual = ranking(x, norm='0,N-1', ties=False)
        assert_almost_equal(actual, desired) 

    def test_ranking_3(self):
        "afunc.ranking #3"
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan],
                      [ 4.0, 2.0, 3.0, 1.0, 0.0]])   
        desired = np.array([[ 0.0,   nan,   4.0, nan, nan],
                            [ 0.0,   4.0,   nan, nan, nan],
                            [ 0.0, 4/3.0, 8/3.0, 4.0, nan],
                            [ 4.0,   2.0,   3.0, 1.0, 0.0]])                     
        actual = ranking(x, axis=1, norm='0,N-1', ties=False)
        assert_almost_equal(actual, desired)
        
    def test_ranking_4(self):
        "afunc.ranking #4"
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([2, 0, 1])[:,None]
        actual = ranking(x, axis=0, norm='0,N-1', ties=False)
        assert_almost_equal(actual, desired)

    def test_ranking_5(self):
        "afunc.ranking #5"
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([0, 0, 0])[:,None]
        actual = ranking(x, axis=1, norm='0,N-1', ties=False)
        assert_almost_equal(actual, desired)    

    def test_ranking_6(self):
        "afunc.ranking #6"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                            [ 0.0,  -1.0,   nan,   nan,   nan],
                            [ 1.0,   1.0,   1.0,   0.0,   nan]])                     
        actual = ranking(x, axis=0, norm='-1,1', ties=False)
        assert_almost_equal(actual, desired) 

    def test_ranking_7(self):
        "afunc.ranking #7"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                           [ 0.0,  -1.0,   nan,   nan,   nan],
                           [ 1.0,   1.0,   1.0,   0.0,   nan]])                    
        actual = ranking(x, norm='-1,1', ties=False)
        assert_almost_equal(actual, desired) 

    def test_ranking_8(self):
        "afunc.ranking #8"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan],
                      [ 4.0,   2.0,   3.0, 1.0  , 0.0  ]])   
        desired = np.array([[-1.0,   nan,   1.0,   nan,   nan],
                            [-1.0,   1.0,   nan,   nan,   nan],
                            [-1.0,-1/3.0, 1/3.0,   1.0,   nan],
                            [ 1.0,   0.0,   0.5,  -0.5,  -1.0]])                     
        actual = ranking(x, axis=1, norm='-1,1', ties=False)
        assert_almost_equal(actual, desired)
        
    def test_ranking_9(self):
        "afunc.ranking #9" 
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([1.0,-1.0, 0.0])[:,None]
        actual = ranking(x, axis=0, norm='-1,1', ties=False)
        assert_almost_equal(actual, desired)

    def test_ranking_10(self):
        "afunc.ranking #10"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([0.0, 0.0, 0.0])[:,None]
        actual = ranking(x, axis=1, norm='-1,1', ties=False)
        assert_almost_equal(actual, desired)  

    def test_ranking_11(self):
        "afunc.ranking_11"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                            [ 0.0,  -1.0,   nan,   nan,   nan],
                            [ 1.0,   1.0,   1.0,   0.0,   nan]])                     
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired) 

    def test_ranking_12(self):
        "afunc.ranking_12"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        desired = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                            [ 0.0,  -1.0,   nan,   nan,   nan],
                            [ 1.0,   1.0,   1.0,   0.0,   nan]])                    
        actual = ranking(x)
        assert_almost_equal(actual, desired) 

    def test_ranking_13(self):
        "afunc.ranking_13"
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
        
    def test_ranking_14(self):
        "afunc.ranking_14"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([1.0,-1.0, 0.0])[:,None]
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired) 

    def test_ranking_15(self):
        "afunc.ranking_15"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        desired = np.array([0.0, 0.0, 0.0])[:,None]
        actual = ranking(x, axis=1)
        assert_almost_equal(actual, desired)
        
    def test_ranking_16(self):
        "afunc.ranking_16"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        desired = np.array([[ 0.0,   nan,   0.5,  nan,   nan],
                            [ 0.0,  -1.0,   nan,  nan,   nan],
                            [ 0.0,   0.0,  -1.0,  1.0,   nan],
                            [ 0.0,   1.0,   0.5, -1.0,   0.0]])                    
        actual = ranking(x)
        assert_almost_equal(actual, desired)       

    def test_ranking_17(self):
        "afunc.ranking_17"
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

    def test_ranking_18(self):
        "afunc.ranking_18"
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
        
    def test_ranking_19(self):
        "afunc.ranking_19"
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

    def test_ranking_20(self):
        "afunc.ranking_20"
        x = np.array([[ nan],
                      [ nan],
                      [ nan]])  
        desired = np.array([[ nan],
                            [ nan],
                            [ nan]])
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_21(self):
        "afunc.ranking_21"
        x = np.array([[ nan, nan],
                      [ nan, nan],
                      [ nan, nan]])  
        desired = np.array([[ nan, nan],
                            [ nan, nan],
                            [ nan, nan]])
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_22(self):
        "afunc.ranking_22"
        x = np.array([[ nan, nan, nan]])  
        desired = np.array([[ nan, nan, nan]])                                     
        actual = ranking(x, axis=1)
        assert_almost_equal(actual, desired)
        
    def test_ranking_23(self):
        "afunc.ranking_23"
        x = np.array([ 1.0, np.inf, 2.0])  
        desired = np.array([-1.0, 1.0, 0.0])                                      
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)        

    def test_ranking_24(self):
        "afunc.ranking_24"
        x = np.array([ 1.0, np.inf, 2.0])  
        desired = np.array([-1.0, 1.0, 0.0])                                      
        actual = ranking(x, axis=0, ties=False)
        assert_almost_equal(actual, desired) 

    def test_ranking_25(self):
        "afunc.ranking_25"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        desired = np.array([-1.0, nan, 0.0, 1.0])                                      
        actual = ranking(x, axis=0)
        assert_almost_equal(actual, desired)

    def test_ranking_26(self):
        "afunc.ranking_26"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        desired = np.array([-1.0, nan, 0.0, 1.0])                                      
        actual = ranking(x, axis=0, ties=False)
        assert_almost_equal(actual, desired)
        
        
class Test_geometric_mean(unittest.TestCase):
    "Test afunc.geometric_mean"

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 2.0, 8.0],
                           [2.0, 4.0, 8.0, 2.0, 1.0]])
        self.xnan = np.array([[nan, nan, nan, nan, nan],
                              [nan, nan, nan, nan, nan]])
        self.x2 = np.array([[ 2.0,  2.0],
                            [ 1.0,  3.0],
                            [ 3.0,  1.0]])                    

    def test_geometric_mean_1(self):
        "afunc.geometric_mean #1"
        desired = np.array([ 2.0, 1.73205081, 1.73205081])
        actual =  geometric_mean(self.x2, 1)
        assert_almost_equal(actual, desired)

    def test_geometric_mean_2(self):
        "afunc.geometric_mean #2"
        desired = np.array([ 2.0, 1.73205081, 1.73205081])
        actual = geometric_mean(self.x2)
        assert_almost_equal(actual, desired)   

    def test_geometric_mean_3(self):
        "afunc.geometric_mean #3"
        desired = np.array([ 1.81712059, 1.81712059])
        actual = geometric_mean(self.x2, 0)
        assert_almost_equal(actual, desired) 

    def test_geometric_mean_4(self):
        "afunc.geometric_mean #4"
        desired = np.array([nan, nan])
        actual = geometric_mean(self.xnan)
        assert_almost_equal(actual, desired)
        
    def test_geometric_mean_5(self):
        "afunc.geometric_mean #5"
        desired = np.array([ 3.1301691601465746, 2.6390158215457888])
        actual = geometric_mean(self.x, 1)
        assert_almost_equal(actual, desired)

    def test_geometric_mean_6(self):
        "afunc.geometric_mean #6"
        desired = np.array([ 1.4142135623730951, 4.0, 6.9282032302755088, 2.0,
                                                         2.8284271247461903])
        actual = geometric_mean(self.x, 0)
        assert_almost_equal(actual, desired)
        
    def test_geometric_mean_7(self):
        "afunc.geometric_mean #7"
        x = np.array([[1e200, 1e200]])
        desired = 1e200
        actual = geometric_mean(x)
        msg = printfail(desired, actual)
        self.assert_((abs(desired - actual) < 1e187).all(), msg)         
        
class Test_movingsum(unittest.TestCase):
    "Test afunc.movingsum"       

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
        "afunc.movingsum #1"  
        desired = self.xnan 
        actual = movingsum(self.xnan, self.window, norm=True)
        assert_almost_equal(actual, desired)             

    def test_movingsum_2(self):
        "afunc.movingsum #2"    
        desired = self.xnan
        actual = movingsum(self.xnan, self.window, norm=False)
        assert_almost_equal(actual, desired)   

    def test_movingsum_3(self):
        "afunc.movingsum #3"    
        desired = np.array([[  nan, 2.0, 12.0, 6.0, 8.0],
                            [  nan, 6.0, 12.0, 8.0,-1.0]])   
        actual = movingsum(self.x, self.window, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_4(self):
        "afunc.movingsum #4"   
        desired = np.array([[  nan, 1.0,  6.0, 6.0, 8.0],
                            [  nan, 6.0, 12.0, 8.0,-1.0]])
        actual = movingsum(self.x, self.window, norm=False)
        assert_almost_equal(actual, desired) 

    def test_movingsum_5(self):
        "afunc.movingsum #5"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                            [3.0,  8.0,  14.0, 0.0,  7.0]])
        actual = movingsum(self.x, self.window, axis=0, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_6(self):
        "afunc.movingsum #6"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                            [3.0,  4.0,  14.0, 0.0,  7.0]])
        actual = movingsum(self.x, self.window, axis=0, norm=False)
        assert_almost_equal(actual, desired) 
        
    def test_movingsum_7(self):
        "afunc.movingsum #7"  
        desired = np.array([[nan, 4.0],
                            [nan, 4.0],
                            [nan, 4.0]])
        actual = movingsum(self.x2, self.window)
        assert_almost_equal(actual, desired) 

class Test_movingsum_forward(unittest.TestCase):
    "Test afunc.movingsum_forward"
 
    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[  nan,  nan,  nan,  nan,  nan],
                              [  nan,  nan,  nan,  nan,  nan]])
        self.window = 2

    def test_movingsum_forward_1(self):
        "afunc.movingsum_forward #1"
        desired = np.array([[2.0, 12.0, 6.0, 8.0, nan],
                            [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0            
        actual = movingsum_forward(self.x, self.window, skip, norm=True)
        assert_almost_equal(actual, desired) 
        
    def test_movingsum_forward_2(self):
        "afunc.movingsum_forward #2"    
        desired = np.array([[1.0,  6.0, 6.0, 8.0, nan],
                            [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0                     
        actual = movingsum_forward(self.x, self.window, skip, norm=False)
        assert_almost_equal(actual, desired)        

    def test_movingsum_forward_3(self):
        "afunc.movingsum_forward #3"    
        desired = np.array([[12.0, 6.0, 8.0, nan, nan],
                            [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                   
        actual = movingsum_forward(self.x, self.window, skip, norm=True)
        assert_almost_equal(actual, desired) 

    def test_movingsum_forward_4(self):
        "afunc.movingsum_forward #4"    
        desired = np.array([[ 6.0, 6.0, 8.0, nan, nan],
                            [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                     
        actual = movingsum_forward(self.x, self.window, skip, norm=False)
        assert_almost_equal(actual, desired) 

    def test_movingsum_forward_5(self):
        "afunc.movingsum_forward #5"  
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
        "afunc.movingrank #1"    
        desired = self.xnan 
        actual = movingrank(self.xnan, self.window)
        assert_almost_equal(actual, desired) 
    
    def test_movingrank_2(self):
        "afunc.movingrank #2"    
        desired = np.array([[  nan,  nan,  nan,-1.0,1.0],
                           [  nan,1.0,1.0,-1.0,-1.0]]) 
        actual = movingrank(self.x, self.window)
        assert_almost_equal(actual, desired)          

    def test_movingrank_3(self):
        "afunc.movingrank #3"    
        desired = np.array([[nan,  nan,  nan,  nan,  nan],
                           [1.0,  nan,  1.0,  0.0,  -1.0]])
        actual = movingrank(self.x, self.window, axis=0)
        assert_almost_equal(actual, desired) 
        
    def test_movingrank_4(self):
        "afunc.movingrank #4"    
        desired = np.array([[nan,  nan],
                           [nan,  1.0],
                           [nan, -1.0]])
        actual = movingrank(self.x2, self.window)
        assert_almost_equal(actual, desired)
        
class Test_correlation(unittest.TestCase):
    "Test afunc.correlation"
    
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
        corr = correlation(x, y, axis=1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        x = self.b1
        y = self.b2
        corr = correlation(x, y, axis=1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")

    def test_correlation_8(self):
        "farray.correlation_8"
        x = self.a1.T
        y = self.a2.T
        corr = correlation(x, y, axis=0)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        x = self.b1.T
        y = self.b2.T
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
        corr = correlation(x, y, axis=-1)
        desired = np.array([nan, 1, -1, -0.5]) 
        aae(corr, desired, err_msg="aggregate of 1d tests")
        
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
    
    # Calc function
    s.append(unit(Test_correlation))              
         
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)

