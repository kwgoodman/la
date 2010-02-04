"""Unit tests of array functions."""

import unittest

import numpy as np
nan = np.nan

from la.util.testing import printfail
from la.afunc import group_ranking, group_mean, group_median
from la.afunc import (movingsum, movingrank, movingsum_forward, ranking, 
                      geometric_mean, unique_group)

# Sector functions ----------------------------------------------------------

class Test_group_ranking(unittest.TestCase):
    """Test afunc.group_ranking"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_ranking_1(self):
        "afunc.group_ranking #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        theory = np.array([[-1.0, 0.0,  nan, nan, -1.0, nan],
                           [-1.0, 1.0, -1.0, nan,  nan, nan],
                           [ 0.0,-1.0, -1.0, nan,  0.0, nan],
                           [ 1.0,-1.0,  1.0, nan,  nan, nan],
                           [ 1.0, 1.0,  1.0, 0.0,  1.0, nan],
                           [ 0.0, 0.0,  0.0, 0.0,  nan, nan]])
        practice = group_ranking(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_group_ranking_2(self):
        "afunc.group_ranking #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        theory = np.array([[-1.0,  0.0,  nan, nan,-1.0, nan],
                           [-1.0,  1.0, -1.0, nan, nan, nan],
                           [ 0.0, -1.0, -1.0, nan, 0.0, nan],
                           [ 1.0, -1.0,  1.0, nan, nan, nan],
                           [ 1.0,  1.0,  1.0, 0.0, 1.0, nan],
                           [ nan,  nan,  nan, nan, nan, nan]])
        practice = group_ranking(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)                                          

class Test_group_mean(unittest.TestCase):
    """Test afunc.group_mean"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_group_mean_1(self):
        "afunc.group_mean #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        theory = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
        practice = group_mean(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_group_mean_2(self):
        "afunc.group_mean #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        theory = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ nan, nan,  nan, nan,  nan, nan]])
        practice = group_mean(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_group_mean_3(self):
        "afunc.group_mean #3"
        sectors = ['a', 'b', 'a', 'b', 'a']
        x = np.array([[1,2],
                      [3,4],
                      [6,7],
                      [0,0],
                      [8,-1]])
        theory = np.array([[5.0,8/3.0],
                           [1.5,2.0],
                           [5.0,8/3.0],
                           [1.5,2.0],
                           [5.0,8/3.0]])
        practice = group_mean(x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)            

class Test_group_median(unittest.TestCase):
    """Test afunc.group_median"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        
    def test_median_1(self):
        "afunc.group_median #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        theory = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
        practice = group_median(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_group_median_2(self):
        "afunc.group_median #2"
        sectors = ['a', 'b', 'a', 'b', 'a', None]
        theory = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ 2.0, 0.5,  1.5, nan,  nan, nan],
                           [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                           [ nan, nan,  nan, nan,  nan, nan]])
        practice = group_median(self.x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_group_median_3(self):
        "afunc.group_median #3"
        sectors = ['a', 'b', 'a', 'b', 'a']
        x = np.array([[1,2],
                      [3,4],
                      [6,7],
                      [0,0],
                      [8,-1]])
        theory = np.array([[6.0,2.0],
                           [1.5,2.0],
                           [6.0,2.0],
                           [1.5,2.0],
                           [6.0,2.0]])
        practice = group_median(x, sectors)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)


class Test_sector_oth(unittest.TestCase):
    """Test afunc.group_mean"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8
        
    def test_sector_unique_1(self):
        "afunc.unique_group #1"
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        theory = ['a', 'b', 'c']
        practice = unique_group(sectors)
        msg = printfail(theory, practice)
        #theory[np.isnan(theory)] = self.nancode
        #practice[np.isnan(practice)] = self.nancode        
        self.assert_(theory == practice, msg)      

    
# Normalize functions -------------------------------------------------------

class Test_ranking_1N(unittest.TestCase):
    """Test afunc.ranking_1N"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8

    def test_ranking_1N_1(self):
        """afunc.ranking_1N #1"""
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])
        theory = np.array([[ 0.0, nan, 0.0, nan, nan],
                           [ 1.0, 0.0, nan, nan, nan],
                           [ 2.0, 2.0, 2.0, 1.0, nan]])                     
        practice = ranking(x, axis=0, norm='0,N-1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_1N_2(self):
        """afunc.ranking_1N #2"""
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])
        theory = np.array([[ 0.0, nan, 0.0, nan, nan],
                           [ 1.0, 0.0, nan, nan, nan],
                           [ 2.0, 2.0, 2.0, 1.0, nan]])                     
        practice = ranking(x, norm='0,N-1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_1N_3(self):
        """afunc.ranking_1N #3"""
        x = np.array([[ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan],
                      [ 4.0, 2.0, 3.0, 1.0, 0.0]])   
        theory = np.array([[ 0.0,   nan,   4.0, nan, nan],
                           [ 0.0,   4.0,   nan, nan, nan],
                           [ 0.0, 4/3.0, 8/3.0, 4.0, nan],
                           [ 4.0,   2.0,   3.0, 1.0, 0.0]])                     
        practice = ranking(x, axis=1, norm='0,N-1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_ranking_1N_4(self):
        """afunc.ranking_1N #4""" 
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([2, 0, 1])[:,None]
        practice = ranking(x, axis=0, norm='0,N-1', ties=False)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg) 

    def test_ranking_1N_5(self):
        """afunc.ranking_1N #5"""  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([0, 0, 0])[:,None]
        practice = ranking(x, axis=1, norm='0,N-1', ties=False)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg)        

class Test_ranking_norm(unittest.TestCase):
    """Test afunc.ranking_norm"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8    

    def test_ranking_norm_1(self):
        """afunc.ranking_norm #1"""
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        theory = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                           [ 0.0,  -1.0,   nan,   nan,   nan],
                           [ 1.0,   1.0,   1.0,   0.0,   nan]])                     
        practice = ranking(x, axis=0, norm='-1,1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_norm_2(self):
        """afunc.ranking_norm #2"""
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        theory = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                           [ 0.0,  -1.0,   nan,   nan,   nan],
                           [ 1.0,   1.0,   1.0,   0.0,   nan]])                    
        practice = ranking(x, norm='-1,1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_norm_3(self):
        """afunc.ranking_norm #3"""
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan],
                      [ 4.0,   2.0,   3.0, 1.0  , 0.0  ]])   
        theory = np.array([[-1.0,   nan,   1.0,   nan,   nan],
                           [-1.0,   1.0,   nan,   nan,   nan],
                           [-1.0,-1/3.0, 1/3.0,   1.0,   nan],
                           [ 1.0,   0.0,   0.5,  -0.5,  -1.0]])                     
        practice = ranking(x, axis=1, norm='-1,1', ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_ranking_norm_4(self):
        """afunc.ranking_norm #4"""  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([1.0,-1.0, 0.0])[:,None]
        practice = ranking(x, axis=0, norm='-1,1', ties=False)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg) 

    def test_ranking_norm_5(self):
        """afunc.ranking_norm #5"""  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([0.0, 0.0, 0.0])[:,None]
        practice = ranking(x, axis=1, norm='-1,1', ties=False)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg)

class Test_ranking(unittest.TestCase):
    """Test afunc.ranking"""
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8    

    def test_ranking_1(self):
        "afunc.ranking_1"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan]])
        theory = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                           [ 0.0,  -1.0,   nan,   nan,   nan],
                           [ 1.0,   1.0,   1.0,   0.0,   nan]])                     
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_2(self):
        "afunc.ranking_2"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        theory = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                           [ 0.0,  -1.0,   nan,   nan,   nan],
                           [ 1.0,   1.0,   1.0,   0.0,   nan]])                    
        practice = ranking(x)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_ranking_3(self):
        "afunc.ranking_3"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan],
                      [ 4.0,   2.0,   3.0, 1.0  , 0.0  ]])   
        theory = np.array([[-1.0,   nan,   1.0,   nan,   nan],
                           [ 0.0,   0.0,   nan,   nan,   nan],
                           [ 0.0,   0.0,   0.0,   0.0,   nan],
                           [ 1.0,   0.0,   0.5,  -0.5,  -1.0]])                    
        practice = ranking(x, axis=1)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_ranking_4(self):
        "afunc.ranking_4"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([1.0,-1.0, 0.0])[:,None]
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg) 

    def test_ranking_5(self):
        "afunc.ranking_5"  
        x = np.array([3.0, 1.0, 2.0])[:,None]
        theory = np.array([0.0, 0.0, 0.0])[:,None]
        practice = ranking(x, axis=1)
        msg = printfail(theory, practice)    
        self.assert_((theory == practice).all(), msg)
        
    def test_ranking_6(self):
        "afunc.ranking_6"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        theory = np.array([[ 0.0,   nan,   0.5,  nan,   nan],
                           [ 0.0,  -1.0,   nan,  nan,   nan],
                           [ 0.0,   0.0,  -1.0,  1.0,   nan],
                           [ 0.0,   1.0,   0.5, -1.0,   0.0]])                    
        practice = ranking(x)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)        

    def test_ranking_7(self):
        "afunc.ranking_7"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        theory = np.array([[ 0.0,   nan ,   0.0,  nan  ,   nan],
                           [ 0.0,   0.0 ,   nan,  nan  ,   nan],
                           [-1.0/3, 2.0/3, -1.0,  2.0/3,   nan],
                           [ 0.0,   1.0 ,   0.0,  0.0  ,  -1.0]])                    
        practice = ranking(x, 1)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_ranking_8(self):
        "afunc.ranking_8"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]])   
        theory = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                           [-2.0/3, -2.0/3, -1.0/3,   0.0],
                           [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                           [ 2.0/3,    1.0,  2.0/3,   1.0]])                    
        practice = ranking(x, 0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg) 
        
    def test_ranking_9(self):
        "afunc.ranking_9"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]]) 
        x = x.T  
        theory = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                           [-2.0/3, -2.0/3, -1.0/3,   0.0],
                           [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                           [ 2.0/3,    1.0,  2.0/3,   1.0]])
        theory = theory.T                                       
        practice = ranking(x, 1)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)              

    def test_ranking_10(self):
        "afunc.ranking_10"
        x = np.array([[ nan],
                      [ nan],
                      [ nan]])  
        theory = np.array([[ nan],
                           [ nan],
                           [ nan]])                                     
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_ranking_11(self):
        "afunc.ranking_11"
        x = np.array([[ nan, nan],
                      [ nan, nan],
                      [ nan, nan]])  
        theory = np.array([[ nan, nan],
                           [ nan, nan],
                           [ nan, nan]])                                     
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_ranking_12(self):
        "afunc.ranking_12"
        x = np.array([[ nan, nan, nan]])  
        theory = np.array([[ nan, nan, nan]])                                     
        practice = ranking(x, axis=1)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_ranking_13(self):
        "afunc.ranking_13"
        x = np.array([ 1.0, np.inf, 2.0])  
        theory = np.array([-1.0, 1.0, 0.0])                                      
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)        

    def test_ranking_14(self):
        "afunc.ranking_14"
        x = np.array([ 1.0, np.inf, 2.0])  
        theory = np.array([-1.0, 1.0, 0.0])                                      
        practice = ranking(x, axis=0, ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg) 

    def test_ranking_15(self):
        "afunc.ranking_15"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        theory = np.array([-1.0, nan, 0.0, 1.0])                                      
        practice = ranking(x, axis=0)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)

    def test_ranking_16(self):
        "afunc.ranking_16"
        x = np.array([ -np.inf, nan, 1.0, np.inf])  
        theory = np.array([-1.0, nan, 0.0, 1.0])                                      
        practice = ranking(x, axis=0, ties=False)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode        
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
class Test_geometric_mean(unittest.TestCase):
    """Test afunc.geometric_mean"""

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 2.0, 8.0],
                           [2.0, 4.0, 8.0, 2.0, 1.0]])
        self.xnan = np.array([[nan, nan, nan, nan, nan],
                              [nan, nan, nan, nan, nan]])
        self.x2 = np.array([[ 2.0,  2.0],
                            [ 1.0,  3.0],
                            [ 3.0,  1.0]])
        self.tol = 1e-8
        self.nancode = -9999                    

    def test_geometric_mean_1(self):
        """afunc.geometric_mean #1"""
        theory = np.array([[ 2.        ],
                           [ 1.73205081],
                           [ 1.73205081]])
        practice =  geometric_mean(self.x2, 1)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < self.tol).all(), msg)                  

    def test_geometric_mean_2(self):
        """afunc.geometric_mean #2"""
        theory = np.array([[ 2.        ],
                           [ 1.73205081],
                           [ 1.73205081]])
        practice =  geometric_mean(self.x2)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < self.tol).all(), msg)     

    def test_geometric_mean_3(self):
        """afunc.geometric_mean #3"""
        theory = np.array([[ 1.81712059,  1.81712059]])
        practice =  geometric_mean(self.x2, 0)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < self.tol).all(), msg)  

    def test_geometric_mean_4(self):
        """afunc.geometric_mean #4"""
        theory = np.array([[   nan],
                           [   nan]])
        practice =  geometric_mean(self.xnan)
        msg = printfail(theory, practice)
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((abs(theory - practice) < self.tol).all(), msg) 
        
    def test_geometric_mean_5(self):
        """afunc.geometric_mean #5"""
        theory = np.array([[ 3.1301691601465746],
                           [ 2.6390158215457888]])
        practice =  geometric_mean(self.x, 1)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < self.tol).all(), msg)                   

    def test_geometric_mean_6(self):
        """afunc.geometric_mean #6"""
        theory = np.array([[ 1.4142135623730951, 4.0, 6.9282032302755088, 2.0,
                                                         2.8284271247461903]])
        practice =  geometric_mean(self.x, 0)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < self.tol).all(), msg)
        
    def test_geometric_mean_7(self):
        """afunc.geometric_mean #7"""
        x = np.array([[1e200, 1e200]])
        theory = np.array([[1e200]])
        practice =  geometric_mean(x)
        msg = printfail(theory, practice)
        self.assert_((abs(theory - practice) < 1e187).all(), msg)         
        
class Test_movingsum(unittest.TestCase):
    """Test afunc.movingsum"""        

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[  nan,  nan,  nan,  nan,  nan],
                              [  nan,  nan,  nan,  nan,  nan]])
        self.window = 2
        self.x2 = np.array([[ 2.0,  2.0],
                            [ 1.0,  3.0],
                            [ 3.0,  1.0]])
        self.nancode = -9999                    
        #convert to array
        self.x = np.asarray(self.x)
        self.xnan2 = np.asarray(self.xnan[:,:-1])
        self.x2 = np.asarray(self.x2)

    def test_movingsum_1(self):
        """afunc.movingsum #1"""    
        theory = self.xnan 
        practice = movingsum(self.xnan, self.window, norm=True)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)              

    def test_movingsum_2(self):
        """afunc.movingsum #2"""    
        theory = self.xnan
        practice = movingsum(self.xnan, self.window, norm=False)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)    

    def test_movingsum_3(self):
        """afunc.movingsum #3"""    
        theory = np.array([[  nan, 2.0, 12.0, 6.0, 8.0],
                           [  nan, 6.0, 12.0, 8.0,-1.0]])   
        practice = movingsum(self.x, self.window, norm=True)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)

    def test_movingsum_4(self):
        """afunc.movingsum #4"""    
        theory = np.array([[  nan, 1.0,  6.0, 6.0, 8.0],
                           [  nan, 6.0, 12.0, 8.0,-1.0]])
        practice = movingsum(self.x, self.window, norm=False)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)

    def test_movingsum_5(self):
        """afunc.movingsum #5"""    
        theory = np.array([[nan,  nan,  nan,  nan,  nan],
                           [3.0,  8.0,  14.0, 0.0,  7.0]])
        practice = movingsum(self.x, self.window, axis=0, norm=True)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)

    def test_movingsum_6(self):
        """afunc.movingsum #6"""    
        theory = np.array([[nan,  nan,  nan,  nan,  nan],
                           [3.0,  4.0,  14.0, 0.0,  7.0]])
        practice = movingsum(self.x, self.window, axis=0, norm=False)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)
        
    def test_movingsum_7(self):
        """afunc.movingsum #7"""   
        theory = np.array([[nan, 4.0],
                           [nan, 4.0],
                           [nan, 4.0]])
        practice = movingsum(self.x2, self.window)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_(np.all(theory == practice), msg)

class Test_movingsum_forward(unittest.TestCase):
    """Test afunc.movingsum_forward""" 
 
    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[  nan,  nan,  nan,  nan,  nan],
                              [  nan,  nan,  nan,  nan,  nan]])
        self.window = 2
        self.nancode = -9999           

    def test_movingsum_forward_1(self):
        """afunc.movingsum_forward #1"""
        theory = np.array([[2.0, 12.0, 6.0, 8.0, nan],
                           [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0            
        practice = movingsum_forward(self.x, self.window, skip, norm=True)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)
        
    def test_movingsum_forward_2(self):
        """afunc.movingsum_forward #2"""    
        theory = np.array([[1.0,  6.0, 6.0, 8.0, nan],
                           [6.0, 12.0, 8.0,-1.0, nan]]) 
        skip = 0                     
        practice = movingsum_forward(self.x, self.window, skip, norm=False)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)        

    def test_movingsum_forward_3(self):
        """afunc.movingsum_forward #3"""    
        theory = np.array([[12.0, 6.0, 8.0, nan, nan],
                           [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                   
        practice = movingsum_forward(self.x, self.window, skip, norm=True)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)

    def test_movingsum_forward_4(self):
        """afunc.movingsum_forward #4"""    
        theory = np.array([[ 6.0, 6.0, 8.0, nan, nan],
                           [12.0, 8.0,-1.0, nan, nan]]) 
        skip = 1                     
        practice = movingsum_forward(self.x, self.window, skip, norm=False)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)

    def test_movingsum_forward_5(self):
        """afunc.movingsum_forward #5"""    
        theory = np.array([[2.0, 4.0, 8.0, 0.0,-1.0],
                           [nan, nan, nan, nan, nan]])
        skip = 1
        window = 1                    
        practice = movingsum_forward(self.x, window, skip, axis=0)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)          

class Test_movingrank(unittest.TestCase):
    """Test movingrank"""

    def setUp(self):
        self.x = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                           [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.xnan = np.array([[nan, nan, nan, nan, nan],
                              [nan, nan, nan, nan, nan]])
        self.window = 2
        self.x2 = np.array([[nan, 2.0],
                            [1.0, 3.0],
                            [3.0, 1.0]])
        self.nancode = -9999                    
    
    def test_movingrank_1(self):
        """afunc.movingrank #1"""    
        theory = self.xnan 
        practice = movingrank(self.xnan, self.window)
        msg = printfail(theory, practice)    
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg) 
    
    def test_movingrank_2(self):
        """afunc.movingrank #2"""    
        theory = np.array([[  nan,  nan,  nan,-1.0,1.0],
                           [  nan,1.0,1.0,-1.0,-1.0]]) 
        practice = movingrank(self.x, self.window)
        msg = printfail(theory, practice)        
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)           

    def test_movingrank_3(self):
        """afunc.movingrank #3"""    
        theory = np.array([[nan,  nan,  nan,  nan,  nan],
                           [1.0,  nan,  1.0,  0.0,  -1.0]])
        practice = movingrank(self.x, self.window, axis=0)
        msg = printfail(theory, practice)        
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode  
        self.assert_((theory == practice).all(), msg) 
        
    def test_movingrank_4(self):
        """afunc.movingrank #4"""    
        theory = np.array([[nan,  nan],
                           [nan,  1.0],
                           [nan, -1.0]])
        practice = movingrank(self.x2, self.window)
        msg = printfail(theory, practice)        
        theory[np.isnan(theory)] = self.nancode
        practice[np.isnan(practice)] = self.nancode
        self.assert_((theory == practice).all(), msg)           

# Unit tests ----------------------------------------------------------------        
    
def suite():

    unit = unittest.TestLoader().loadTestsFromTestCase
    s = []
    
    # Sector functions
    s.append(unit(Test_group_ranking))
    s.append(unit(Test_group_mean)) 
    s.append(unit(Test_group_median)) 
    
    # Normalize functions
    s.append(unit(Test_ranking_1N))
    s.append(unit(Test_ranking_norm))
    s.append(unit(Test_ranking))    
    s.append(unit(Test_geometric_mean))
    s.append(unit(Test_movingsum))
    s.append(unit(Test_movingsum_forward))
    s.append(unit(Test_movingrank))      
         
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
