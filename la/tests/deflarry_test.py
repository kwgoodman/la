"Unit tests of larry."

import datetime
import unittest

import numpy as np
nan = np.nan
from numpy.testing import (assert_, assert_almost_equal, assert_raises,
                           assert_equal)

from la import larry
from la.util.testing import printfail, noreference, nocopy
from la.util.testing import assert_larry_equal as ale


class Test_init(unittest.TestCase):
    "Test init of larry class"
    
    def setUp(self):
        self.list = [[1,2],[3,4]]
        self.tuple = ((1,2),(3,4))
        self.matrix = np.matrix([[1,2],[3,4]])
        self.array = np.array([[1,2],[3,4]])                
        self.label = [[0,1],[0,1]]                                                      

    def test_init_list(self):
        "larry.__init__list"
        p = larry(self.list)
        t = self.array
        msg = printfail(t, p.x, 'x')      
        self.assert_((p.x == t).all(), msg) 
        self.assert_(self.label == p.label,
                     printfail(self.label, p.label, 'label'))

    def test_init_tuple(self):
        "larry.__init__tuple"
        p = larry(self.tuple)
        t = self.array
        msg = printfail(t, p.x, 'x')      
        self.assert_((p.x == t).all(), msg) 
        self.assert_(self.label == p.label,
                     printfail(self.label, p.label, 'label'))

    def test_init_matrix(self):
        "larry.__init__matrix"
        p = larry(self.matrix)
        t = self.array
        msg = printfail(t, p.x, 'x')      
        self.assert_((p.x == t).all(), msg) 
        self.assert_(self.label == p.label,
                     printfail(self.label, p.label, 'label'))

    def test_init_array(self):
        "larry.__init__array"
        p = larry(self.array)
        t = self.array
        msg = printfail(t, p.x, 'x')      
        self.assert_((p.x == t).all(), msg) 
        self.assert_(self.label == p.label,
                     printfail(self.label, p.label, 'label'))

    
class Test_unary(unittest.TestCase):
    "Test unary functions of larry class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999
        self.x = np.array([[ 1.0, 1.0],
                           [ 1.0, 1.0],
                           [ 1.0, 1.0]])                  
        self.l = larry(self.x)
        self.x2 = np.array([1.0, 1.0])  
        self.l2 = larry(self.x2)        
        self.x3 = np.random.rand(2,3,4)                
        self.l3 = larry(self.x3)
        self.l4 = larry([ nan, 0.0, np.inf, -10.0, -np.inf])                                                       

    def test_log_1(self):
        "larry.log_1"
        d = larry([[ 0.0, 0.0],
                   [ 0.0, 0.0],
                   [ 0.0, 0.0]])
        ale(self.l.log(), d, 'log_1', original=self.l)

    def test_log_2(self):
        "larry.log_2"
        t = np.log(self.x3)
        t = larry(t)
        p = self.l3.log()
        msg = printfail(t, p, 'larry')
        t[np.isnan(t.x)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        self.assert_(noreference(p, self.l3), 'Reference found')
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_log_3(self):
        "larry.log_3"
        t = np.array([ 0.0, 0.0])
        p = self.l2.log()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')        
        
    def test_exp(self):
        "larry.exp"      
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        t = np.e * t               
        p = self.l.exp()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
       
    def test_sqrt(self):
        "larry.sqrt"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])              
        p = self.l.sqrt()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_sign(self):
        "larry.sign"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])              
        p = self.l.sign()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')        

    def test_power(self):
        "larry.power"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])              
        p = self.l.power(2)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')

    def test___pow__(self):
        "larry.__pow__"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])              
        p = self.l**2
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_cumsum_1(self):
        "larry.cumsum_1"       
        t = np.array([[ 1.0, 1.0],
                      [ 2.0, 2.0],
                      [ 3.0, 3.0]])               
        p = self.l.cumsum(0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')   

    def test_cumsum_2(self):
        "larry.cumsum_2"       
        t = np.array([[ 1.0, 2.0],
                      [ 1.0, 2.0],
                      [ 1.0, 2.0]])               
        p = self.l.cumsum(1)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_cumsum_3(self):
        "larry.cumsum_3"
        t = np.cumsum(self.x3, 0)
        t = larry(t)
        p = self.l3.cumsum(0)
        msg = printfail(t, p, 'larry')
        t[np.isnan(t.x)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        self.assert_(noreference(p, self.l3), 'Reference found')
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_cumsum_4(self):
        "larry.cumsum_4"       
        t = np.array([ 1.0, 2.0])               
        p = self.l2.cumsum(0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')                      

    def test_cumprod_1(self):
        "larry.cumprod_1"       
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])               
        p = self.l.cumprod(0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')   

    def test_cumprod_2(self):
        "larry.cumprod_2"       
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])               
        p = self.l.cumprod(1)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_cumprod_3(self):
        "larry.cumprod_3"
        t = np.cumprod(self.x3, 0)
        t = larry(t)
        p = self.l3.cumprod(0)
        msg = printfail(t, p, 'larry')
        t[np.isnan(t.x)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        self.assert_(noreference(p, self.l3), 'Reference found')
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_cumprod_4(self):
        "larry.cumprod_4"       
        t = np.array([ 1.0, 1.0])               
        p = self.l2.cumprod(0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found') 
        
    def test_clip_1(self):
        "larry.clip_1"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])               
        p = self.l.clip(0, 2)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')

    def test_clip_2(self):
        "larry.clip_2"       
        t = np.array([[ 2.0, 2.0],
                      [ 2.0, 2.0],
                      [ 2.0, 2.0]])               
        p = self.l.clip(2, 3)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')

    def test_clip_3(self):
        "larry.clip_3"       
        t = np.array([[ 0.5, 0.5],
                      [ 0.5, 0.5],
                      [ 0.5, 0.5]])               
        p = self.l.clip(0.5, 0.5)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_clip_4(self):
        "larry.clip_4"
        self.failUnlessRaises(ValueError, self.l.clip, 3, 2)
        
    def test_nan_replace_1(self):
        "larry.nan_replace_1"        
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])               
        p = self.l.nan_replace(0.0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')      
        
    def test_nan_replace_2(self):
        "larry.nan_replace_2"        
        t = np.array([[ 1.0, 0.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])               
        p = self.l
        p[0,1] = np.nan
        p = p.nan_replace(0.0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_nan_replace_3(self):
        "larry.nan_replace_3"       
        self.x3[0,0] = 999        
        t = larry(self.x3.copy())
        p = self.l3
        p[0,0] = 999       
        msg = printfail(t, p, 'larry')
        t[np.isnan(t.x)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg)
        self.assert_(noreference(p, t), 'Reference found')        
                
    def test___neg__(self):
        "larry.__neg__"
        t = np.array([[-1.0,-1.0],
                      [-1.0,-1.0],
                      [-1.0,-1.0]])
        p = -self.l
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')

    def test___pos__(self):
        "larry.__pos__"     
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        p = +self.l
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_abs_1(self):
        "larry.abs_1"     
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        p = self.l.abs()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found') 
        
    def test_abs_2(self):
        "larry.abs_2"     
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        p = -self.l
        p = p.abs()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')              

    def test___abs___1(self):
        "larry.__abs___1"     
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        p = abs(self.l)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test___abs___2(self):
        "larry.__abs___2"     
        t = np.array([[ 1.0, 1.0],
                      [ 1.0, 1.0],
                      [ 1.0, 1.0]])
        p = -self.l
        p = abs(p)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found') 
        
    def test_isnan_1(self):
        "larry.isnan_1"
        t = np.array([True, False, False, False, False])
        label = [[0, 1, 2, 3, 4]]
        p = self.l4.isnan()
        msg = printfail(t, p.x, 'x')      
        self.assert_((t == p.x).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')                 
        
    def test_isfinite_1(self):
        "larry.isfinite_1"
        t = np.array([False, True, False, True, False])
        label = [[0, 1, 2, 3, 4]]
        p = self.l4.isfinite()
        msg = printfail(t, p.x, 'x')      
        self.assert_((t == p.x).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')         
        
    def test_isinf_1(self):
        "larry.isinf_1"
        t = np.array([False, False, True, False, True])
        label = [[0, 1, 2, 3, 4]]
        p = self.l4.isinf()
        msg = printfail(t, p.x, 'x')      
        self.assert_((t == p.x).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')
        
    def test_invert_1(self):
        "larry.invert_1"
        o = larry([True, False])
        d = larry([False, True])
        ale(o.invert(), d, 'invert_1', bool, original=o)                  

    def test_invert_2(self):
        "larry.invert_2"
        y = larry([0, 1])
        self.failUnlessRaises(TypeError, y.invert)      

    def test___invert___1(self):
        "larry.__invert___1"
        o = larry([True, False])
        d = larry([False, True])
        ale(~o, d, 'invert_1', bool, original=o)

    def test___invert___2(self):
        "larry.__invert___2"
        y = larry([0, 1])
        self.failUnlessRaises(TypeError, y.invert)  
        
        
class Test_binary(unittest.TestCase):
    "Test binary functions of Data class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999
        self.x1 = np.array([[ 0.0, 0.0, 1.0, 1.0],
                            [ 0.0, 0.0, 1.0, 1.0],
                            [ 1.0, 1.0, 1.0, 1.0]])
        self.l1 = larry(self.x1)                     
        self.x2 = np.array([[ 1.0, 2.0],
                            [ 3.0, 4.0]])                                                                        
        self.l2 = larry(self.x2)
        self.x3 = np.array([[ 2.0, 2.0, 3.0, 1.0],
                            [ 3.0, 2.0, 2.0, 1.0],
                            [ 1.0, 1.0, 1.0, 1.0]])
        self.l3 = larry(self.x3) 
        self.x4 = np.array([1.0, 1.0])  
        self.l4 = larry(self.x4)                              
        
    def test___add___1(self):
        "larry.__add___1"
        
        # larry + larry                                 
        p = self.l1 + self.l2
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i]) & set(self.l2.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = np.array([[ 1.0, 2.0],
                      [ 3.0, 4.0]]) 
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test___add___2(self):
        "larry.__add___2"
        
        # larry + array                                
        p = self.l1 + self.x1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = 2 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')           
        
        
    def test___add___3(self):
        "larry.__add___3"
        
        # array + larry                               
        p = self.x1 + self.l1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = 2 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___add___4(self):
        "larry.__add___4"
        
        # array + larry                               
        p = self.l1 + 1.0
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = self.x1 + 1.0
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___add___5(self):
        "larry.__add___5"
        
        # array + larry                               
        p = 1.0 + self.l1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = self.x1 + 1.0
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___add___6(self):
        "larry.__add___6"
        p = self.l1
        self.failUnlessRaises(TypeError, p.__add__, datetime.date(2007, 1, 1))
        
    def test___add___7(self):
        "larry.__add___7"
       
        # larry + larry                                 
        p = self.l2 + self.l2
        
        # References
        self.assert_(noreference(p, self.l2), 'Reference found') 
        
    def test___add___8(self):
        "larry.__add___8"
        p = self.l1
        self.failUnlessRaises(IndexError, p.__add__, self.l4)                                                 

    def test___add___9(self):
        "larry.__add___9"
        
        # larry + larry                                 
        p = self.l1 + self.l1
        
        # label
        label = [range(self.l1.shape[0]), range(self.l1.shape[1])]  
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = 2 * self.x1 
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')


    def test___sub___1(self):
        "larry.__sub___1"

        # larry + larry                                 
        p = self.l2 - self.l1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i]) & set(self.l2.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = np.array([[ 1.0, 2.0],
                      [ 3.0, 4.0]]) 
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')          

    def test___sub___2(self):
        "larry.__sub___2"
        
        # larry - array                                
        p = self.l1 - self.x1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = 0 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')         
         
    def test___sub___3(self):
        "larry.__sub___3"
        
        # larry - array                                
        p = self.x1 - self.l1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = 0 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___sub___4(self):
        "larry.__sub___4"
        
        # larry - scalar                              
        p = self.l1 - 1.0
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = self.x1 - 1.0
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___sub___5(self):
        "larry.__sub___5"
        
        # scalar - larry                              
        p = 1.0 - self.l1
        
        # label 
        msg = printfail(self.l1.label, p.label, 'label')  
        self.assert_(p.label == self.l1.label, msg)        
        
        # x
        t = 1.0 - self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')
        
    def test___sub___6(self):
        "larry.__sub___6"
        p = self.l1
        self.failUnlessRaises(TypeError, p.__add__, datetime.date(2007, 1, 1))
        
    def test___sub___7(self):
        "larry.__sub___7"
       
        # larry - larry                                 
        p = self.l2 - self.l2
        
        # label 
        msg = printfail(self.l2.label, p.label, 'label')  
        self.assert_(p.label == self.l2.label, msg)        
        
        # x
        t = self.x2
        t.fill(0.0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)        
        
        # References
        self.assert_(noreference(p, self.l2), 'Reference found')         

    def test___mul___1(self):
        "larry.__mul___1"
        
        # larry * larry                                 
        p = self.l3 * self.l2
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i]) & set(self.l2.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)                  
        
        # x
        t = np.array([[ 2.0, 4.0],
                      [ 9.0, 8.0]]) 
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l3), 'Reference found')          
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test___mul___2(self):
        "larry.__mul___2"
        
        # larry * matrix                                
        p = self.l1 * self.x1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)      
        
        # x
        t = self.x1 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')           
        
    def test___mul___3(self):
        "larry.__mul___3"
        
        # array * larry                                                               
        p = self.l1 * self.x1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)      
        
        # x
        t = self.x1 * self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___mul___4(self):
        "larry.__mul___4"
        
        # larry * scalar                                 
        p = self.l1 * 1.0
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)      
        
        # x
        t = self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___mul___5(self):
        "larry.__mul___5"
        
        # larry * scalar                                 
        p = 1.0 * self.l1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)      
        
        # x
        t = self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found') 
        
    def test___mul___6(self):
        "larry.__mul___6"
        p = self.l2
        self.failUnlessRaises(TypeError, p.__mul__, datetime.date(2007, 1, 1))
        
    def test___mul___7(self):
        "larry.__mul___7"
       
        # larry * larry                                 
        p = self.l2 * self.l2
        
        # label 
        msg = printfail(self.l2.label, p.label, 'label')  
        self.assert_(p.label == self.l2.label, msg)        
        
        # x
        t = self.x2 * self.x2
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)            
        
        # References
        self.assert_(noreference(p, self.l2), 'Reference found')         

    def test___div___1(self):
        "larry.__div___1"
       
        # larry / larry                                 
        p = self.l3 / self.l2
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i]) & set(self.l2.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = np.array([[ 2.0, 1.0],
                      [ 1.0, 0.5]]) 
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l3), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')        
        
    def test___div___2(self):
        "larry.__div___2"
        
        # larry / array                                 
        p = self.l1 / self.x1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = self.l1.x / self.x1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')        
    
    def test___div___3(self):
        "larry.__div___3"
        
        # array / larry                                 
        p = self.x1 / self.l1
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = self.x1 / self.l1.x
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found') 
        
    def test___div___4(self):
        "larry.__div___4"
        
        # larry / scalar                               
        p = self.l1 / 2.0
        
        # label
        label = []
        for i in xrange(self.l1.ndim):
            lab = set(self.l1.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = self.l1.x / 2.0
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')
        
    def test___div___5(self):
        "larry.__div___5"
        
        # larry / scalar                               
        p = 2.0 / self.l2
        
        # label
        label = []
        for i in xrange(self.l2.ndim):
            lab = set(self.l2.label[i])
            lab = sorted(list(lab))
            label.append(lab)   
        msg = printfail(label, p.label, 'label')    
        self.assert_(label == p.label, msg)        
        
        # x
        t = 2.0 / self.l2.x
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        
        # References
        self.assert_(noreference(p, self.l1), 'Reference found')   
        self.assert_(noreference(p, self.l2), 'Reference found')
        
    def test___div___6(self):
        "larry.__div___6"
        p = self.l2
        self.failUnlessRaises(TypeError, p.__div__, datetime.date(2007, 1, 1))
        
    def test___div___7(self):
        "larry.__div___7"
       
        # larry / larry                                 
        p = self.l2 / self.l2
        
        # label 
        msg = printfail(self.l2.label, p.label, 'label')  
        self.assert_(p.label == self.l2.label, msg)        
        
        # x
        t = self.x2 / self.x2
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)            
        
        # References
        self.assert_(noreference(p, self.l2), 'Reference found')              

    def test___div___8(self):
        "larry.__div___8"
       
        # larry / larry                                 
        p = self.l2 / self.l3
        
        # References
        self.assert_(noreference(p, self.l2), 'Reference found')
        self.assert_(noreference(p, self.l3), 'Reference found')


class Test_reduce(unittest.TestCase):
    "Test reducing functions of the larry class"
    
    def setUp(self):
        self.nancode = -9999
        self.tol = 1e-8
        self.x = np.array([[ 1.0, nan],
                           [ 1.0, 1.0],
                           [ 1.0, 1.0]])                    
        self.l = larry(self.x) 
        self.x2 = np.array([1.0, 2.0, 3.0])  
        self.l2 = larry(self.x2)                                                         
        self.x3 = np.array([[ 1.0, nan, nan],
                            [ 0.0, 1.0, 3.0],
                            [ 1.0, 1.0, 0.0]])                    
        self.l3 = larry(self.x3) 
                
    def test_sum_1(self):
        "larry.sum_1"
        t = 5.0
        p = self.l.sum()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_sum_2(self):
        "larry.sum_2"
        x = self.x.copy()
        label = [range(x.shape[1])]
        x[np.isnan(x)] = 0
        x = x.sum(0)
        t = larry(x, label)
        p = self.l.sum(0)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)
        self.assert_(noreference(p, t), 'Reference found')        

    def test_sum_3(self):
        "larry.sum_3"
        x = self.x.copy()
        label = [range(x.shape[0])]        
        x[np.isnan(x)] = 0
        x = x.sum(1)
        t = larry(x, label)
        p = self.l.sum(1)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_sum_4(self):
        "larry.sum_4"
        self.failUnlessRaises(ValueError, self.l.sum, 3)
        
    def test_sum_5(self):
        "larry.sum_5"
        t = 6.0
        p = self.l2.sum()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)                         

    def test_prod_1(self):
        "larry.prod_1"
        t = 1.0
        p = self.l.prod()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_prod_2(self):
        "larry.prod_2"
        x = self.x.copy()
        label = [range(x.shape[1])]
        x[np.isnan(x)] = 1
        x = x.prod(0)
        t = larry(x, label)
        p = self.l.prod(0)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)
        self.assert_(noreference(p, t), 'Reference found')        

    def test_prod_3(self):
        "larry.prod_3"
        x = self.x.copy()
        label = [range(x.shape[0])]        
        x[np.isnan(x)] = 1
        x = x.prod(1)
        t = larry(x, label)
        p = self.l.prod(1)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_prod_4(self):
        "larry.prod_4"
        self.failUnlessRaises(ValueError, self.l.prod, 3)
        
    def test_prod_5(self):
        "larry.prod_5"
        t = 6.0
        p = self.l2.prod()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)

    def test_mean_1(self):
        "larry.mean_1"
        t = 1.0
        p = self.l.mean()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_mean_2(self):
        "larry.mean_2"
        x = np.array([1.0, 1.0])
        label = [range(x.shape[0])]
        t = larry(x, label)
        p = self.l.mean(0)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_mean_3(self):
        "larry.mean_3"
        x = np.array([1.0, 1.0, 1.0])
        label = [range(x.size)]
        t = larry(x, label)
        p = self.l.mean(1)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_mean_4(self):
        "larry.mean_4"
        self.failUnlessRaises(IndexError, self.l.mean, 3)

    def test_mean_5(self):
        "larry.mean_5"
        t = 2.0
        p = self.l2.mean()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)

    def test_median_1(self):
        "larry.median_1"
        t = 1.0
        p = self.l.median()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_median_2(self):
        "larry.median_2"
        x = np.array([1.0, 1.0])
        label = [range(x.shape[0])]
        t = larry(x, label)
        p = self.l.median(0)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')        

    def test_median_3(self):
        "larry.median_3"
        x = np.array([1.0, 1.0, 1.0])
        label = [range(x.size)]
        t = larry(x, label)
        p = self.l.median(1)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_median_4(self):
        "larry.median_4"
        self.failUnlessRaises(ValueError, self.l.median, 3)

    def test_median_5(self):
        "larry.median_5"
        t = 2.0
        p = self.l2.median()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)

    def test_std_1(self):
        "larry.std_1"
        t = 0.0
        p = self.l.std()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_std_2(self):
        "larry.std_2"
        x = np.array([0.0, 0.0])
        label = [range(x.shape[0])]
        t = larry(x, label)
        p = self.l.std(0)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_std_3(self):
        "larry.std_3"
        x = np.array([0.0, 0.0, 0.0])
        label = [range(x.size)]
        t = larry(x, label)
        p = self.l.std(1)
        t[np.isnan(t.x)] = self.nancode
        p[p.isnan()] = self.nancode          
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_std_4(self):
        "larry.std_4"
        self.failUnlessRaises(IndexError, self.l.std, 3)

    def test_std_5(self):
        "larry.std_5"
        t = np.sqrt(2./3)
        p = self.l2.std()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_std_6(self):
        "larry.std_6"        
        s = larry([1, 1, 1]).std(axis=-1)
        self.assert_(s == 0, 'Not equal')

    def test_var_1(self):
        "larry.var_1"
        t = 0.0
        p = self.l.var()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_var_2(self):
        "larry.var_2"
        x = np.array([0.0, 0.0])
        label = [range(x.shape[0])]
        t = larry(x, label)
        p = self.l.var(0)
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_var_3(self):
        "larry.var_3"
        x = np.array([0.0, 0.0, 0.0])
        label = [range(x.size)]
        t = larry(x, label)
        p = self.l.var(1)
        t[np.isnan(t.x)] = self.nancode
        p[p.isnan()] = self.nancode          
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')
        
    def test_var_4(self):
        "larry.var_4"
        self.failUnlessRaises(IndexError, self.l.var, 3)

    def test_var_5(self):
        "larry.var_5"
        t = 2./3
        p = self.l2.var()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg) 
        
    def test_var_6(self):
        "larry.var_6"        
        s = larry([1, 1, 1]).var(axis=-1)
        self.assert_(s == 0, 'Not equal')             
        
    def test_max_1(self):
        "larry.max_1"
        t = 1.0
        p = self.l.max()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_max_2(self):
        "larry.max_2"
        x = np.array([1.0, 1.0])
        t = larry(x)
        p = self.l.max(0)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)        

    def test_max_3(self):
        "larry.max_3"
        x = np.array([1.0, 1.0])
        t = larry(x)
        p = self.l.max(1)
        msg = printfail(t, p, '')
        self.assert_((p.x == t).all(), msg) 
        
    def test_max_4(self):
        "larry.max_4"
        self.failUnlessRaises(ValueError, self.l.max, 3) 
        
    def test_max_5(self):
        "larry.max_5"
        t = 3.0
        p = self.l2.max()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)                 

    def test_min_1(self):
        "larry.min_1"
        t = 1.0
        p = self.l.min()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_min_2(self):
        "larry.min_2"
        x = np.array([1.0, 1.0])
        t = larry(x)
        p = self.l.min(0)
        msg = printfail(t, p, '')
        self.assert_((p == t).all(), msg)        

    def test_min_3(self):
        "larry.min_3"
        x = np.array([1.0, 1.0])
        t = larry(x)
        p = self.l.min(1)
        msg = printfail(t, p, '')
        self.assert_((p.x == t).all(), msg) 
        
    def test_min_4(self):
        "larry.min_4"
        self.failUnlessRaises(ValueError, self.l.min, 3)

    def test_min_5(self):
        "larry.min_5"
        t = 1.0
        p = self.l2.min()
        msg = printfail(t, p, '')
        self.assert_(p == t, msg)
        
    def test_lastrank_1(self):
        "larry.lastrank_1"
        t = np.array([[ nan],
                      [ 1.0],
                      [-1.0]])            
        label = [[0, 1, 2], [2]] 
        t = larry(t, label)              
        p = self.l3.lastrank()
        t[np.isnan(t.x)] = self.nancode
        p[p.isnan()] = self.nancode          
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_lastrank_decay_1(self):
        "larry.lastrank_decay_1"
        t = np.array([[ nan],
                      [ 1.0],
                      [-1.0]])           
        label = [[0, 1, 2], [2]] 
        t = larry(t, label)              
        p = self.l3.lastrank_decay(0)
        t[np.isnan(t.x)] = self.nancode
        p[p.isnan()] = self.nancode          
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found') 
        
    def test_lastrank_decay_2(self):
        "larry.lastrank_decay_2"
        t = np.array([[ nan],
                      [ 1.0],
                      [-1.0]])           
        label = [[0, 1, 2], [2]] 
        t = larry(t, label)              
        p = self.l3.lastrank_decay(10)
        t[np.isnan(t.x)] = self.nancode
        p[p.isnan()] = self.nancode          
        msg = printfail(t.x, p.x, 'x')
        self.assert_((abs(p.x - t.x) < self.tol).all(), msg)     
        msg = printfail(t.label, p.label, 'label')
        self.assert_(p.label == t.label, msg)
        self.assert_(noreference(p, t), 'Reference found')                        

        
class Test_comparison(unittest.TestCase):
    "Test comparison functions of the larry class"
    
    def setUp(self):
        self.x = np.array([[ 1.0, nan],
                           [ 1.0, 1.0],
                           [ 1.0, 3.0]])
        self.y = np.array([[ 0.0, 1.0],
                           [ 1.0, 1.0],
                           [ 1.0, 1.0]])                                               
        self.l = larry(self.x)                                               
        self.x2 = np.array([ 1.0, 2.0, nan])
        self.y2 = np.array([ 1.0, 0.0, 1.0])                                              
        self.l2 = larry(self.x2)
                
    def test_eq_1(self):
        "larry.__eq___1"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True, False]])
        p = self.l == 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_eq_2(self):
        "larry.__eq___2"
        t = np.array([[False, False],
                      [ True,  True],
                      [ True, False]])
        p = self.l == self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_eq_3(self):
        "larry.__eq___3"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True,  True]])
        p = self.l == self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_eq_4(self):
        "larry.__eq___4"
        date = datetime.date(2004, 1, 1)
        self.failUnlessRaises(TypeError, self.l.__eq__, date) 
        
    def test_eq_5(self):
        "larry.__eq___5"
        t = np.array([True, False, False])
        p = self.l2 == self.y2
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))                                

    def test_ne_1(self):
        "larry.__ne___1"
        t = np.array([[False,  True],
                      [False, False],
                      [False,  True]])
        p = self.l != 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_ne_2(self):
        "larry.__ne___2"
        t = np.array([[ True,  True],
                      [False, False],
                      [False,  True]])
        p = self.l != self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_ne_3(self):
        "larry.__ne___3"
        t = np.array([[False,  True],
                      [False, False],
                      [False, False]])
        p = self.l != self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_ne_4(self):
        "larry.__ne___4"
        date = datetime.date(2004, 1, 1)
        self.failUnlessRaises(TypeError, self.l.__ne__, date)
        
    def test_lt_1(self):
        "larry.__lt___1"
        t = np.array([[False, False],
                      [False, False],
                      [False, False]])
        p = self.l < 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_lt_2(self):
        "larry.__lt___2"
        t = np.array([[False, False],
                      [False, False],
                      [False, False]])
        p = self.l < self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_lt_3(self):
        "larry.__lt___3"
        t = np.array([[False, False],
                      [False, False],
                      [False, False]])
        p = self.l < self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_lt_4(self):
        "larry.__lt___4"
        date = datetime.date(2004, 1, 1)
        self.failUnlessRaises(TypeError, self.l.__lt__, date)                        

    def test_gt_1(self):
        "larry.__gt___1"
        t = np.array([[False, False],
                      [False, False],
                      [False,  True]])
        p = self.l > 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_gt_2(self):
        "larry.__gt___2"
        t = np.array([[ True, False],
                      [False, False],
                      [False,  True]])
        p = self.l > self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))  
        
    def test_gt_3(self):
        "larry.__gt___3"
        t = np.array([[False, False],
                      [False, False],
                      [False, False]])
        p = self.l > self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_gt_4(self):
        "larry.__gt___4"
        date = datetime.date(2004, 1, 1)
        self.failUnlessRaises(TypeError, self.l.__gt__, date) 
        
    def test_le_1(self):
        "larry.__le___1"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True, False]])
        p = self.l <= 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_le_2(self):
        "larry.__le___2"
        t = np.array([[False, False],
                      [ True,  True],
                      [ True, False]])
        p = self.l <= self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_le_3(self):
        "larry.__le___3"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True,  True]])
        p = self.l <= self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_le_4(self):
        "larry.__le___4"
        date = datetime.date(2004, 1, 1)
        self.failUnlessRaises(TypeError, self.l.__le__, date)                        

    def test_ge_1(self):
        "larry.__ge___1"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True,  True]])
        p = self.l >= 1.0
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_ge_2(self):
        "larry.__ge___2"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True,  True]])
        p = self.l >= self.y
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_ge_3(self):
        "larry.__ge___3"
        t = np.array([[ True, False],
                      [ True,  True],
                      [ True,  True]])
        p = self.l >= self.l
        msg = printfail(t, p.x, 'x')       
        self.assert_((t == p.x).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_ge_4(self):
        "larry.__ge___4"
        x = [1, 2]
        self.failUnlessRaises(TypeError, self.l.__ge__, x)

        
class Test_anyall(unittest.TestCase):
    "Test any and all of the larry class"
    
    def setUp(self):
        self.x1 = np.array([[ 1.0, nan]])
        self.x2 = np.array([ 0.0, 1.0])
        self.x3 = np.array([[ 3.0, 1.0]])
        self.x4 = np.array([[True, True]])
        self.x5 = np.array([[True, False]])
                                                                                    
        label = [range(self.x1.shape[0]), range(self.x1.shape[1])]                    
        self.d1 = larry(self.x1, label)                                               
        self.d2 = larry(self.x2)
        self.d3 = larry(self.x3, label)
        self.d4 = larry(self.x4, label)
        self.d5 = larry(self.x5, label)
                        
    def test_any_1(self):
        "larry.any_1"
        t = True
        p = self.d1.any()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)                                 

    def test_any_2(self):
        "larry.any_2"
        t = True
        p = self.d2.any()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)
        
    def test_any_3(self):
        "larry.any_3"
        t = True
        p = self.d3.any()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)
        
    def test_any_4(self):
        "larry.any_4"
        t = True
        p = self.d4.any()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)                

    def test_any_5(self):
        "larry.any_5"
        t = True
        p = self.d5.any()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg) 

    def test_all_1(self):
        "larry.all_1"
        t = True
        p = self.d1.all()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)                                 

    def test_all_2(self):
        "larry.all_2"
        t = False
        p = self.d2.all()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)
        
    def test_all_3(self):
        "larry.all_3"
        t = True
        p = self.d3.all()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)
        
    def test_all_4(self):
        "larry.all_4"
        t = True
        p = self.d4.all()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)                

    def test_all_5(self):
        "larry.all_5"
        t = False
        p = self.d5.all()
        msg = printfail(t, p, 'Truth')       
        self.assert_(t == p, msg)


class Test_getset(unittest.TestCase):
    "Test get and set functions of the larry class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999    
        self.x = np.array([[ 1.0, nan],
                           [ 3.0, 4.0],
                           [ 5.0, 6.0]])                                              
        self.l = larry(self.x) 
        self.x2 = np.array([ 0, 1, 2, 3])                                              
        self.l2 = larry(self.x2) 
                                
    def test_getitem_1(self):
        "larry.__getitem___1"
        t = np.array([[3.0, 4.0]])
        p = self.l[1]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))        

    def test_getitem_2(self):
        "larry.__getitem___2"
        t = np.matrix([[3.0, 4.0]])
        p = self.l[1,:]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_getitem_3(self):
        "larry.__getitem___3"
        t = np.array([[3.0, 4.0]])
        p = self.l[1,0:2]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_getitem_4(self):
        "larry.__getitem___4"
        t = 4.0
        p = self.l[1,1]
        msg = printfail(t, p, 'x')      
        self.assert_((abs(t - p) < self.tol).all(), msg)
        
    def test_getitem_5(self):
        "larry.__getitem___5"
        t = np.array([[ 3.0, 4.0],
                      [ 5.0, 6.0]])  
        idx = self.l.x.sum(1) > 2
        idx = np.where(idx)[0]
        p = self.l[idx,:]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_getitem_6(self):
        "larry.__getitem___6"
        t = np.array([[ 1.0, nan],
                      [ 3.0, 4.0]])
        p = self.l[0:2,0:2]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))       
        
    def test_getitem_7(self):
        "larry.__getitem___7"
        t = np.array([[ 3.0, 4.0],
                      [ 5.0, 6.0]])  
        idx = self.l.x.sum(1) > 2
        idx = np.where(idx)[0]
        self.failUnlessRaises(IndexError, self.l.__getitem__, idx)
        
    def test_getitem_8(self):
        "larry.__getitem___8"
        self.failUnlessRaises(IndexError, self.l.__getitem__, 100)  
        
    def test_getitem_9(self):
        "larry.__getitem___9"
        self.failUnlessRaises(IndexError, self.l.__getitem__, 'a')

    def test_getitem_10(self):
        "larry.__getitem___10"
        t = 1
        p = self.l2[1]
        msg = printfail(t, p, 'x')       
        self.assert_((abs(t - p) < self.tol).all(), msg)       

    def test_getitem_11(self):
        "larry.__getitem___11"
        t = np.array([0, 1])
        p = self.l2[:2]
        msg = printfail(t, p.x, 'x')      
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_getitem_12(self):
        "larry.__getitem___12"
        t = np.array([[ 1.0, nan],
                      [ 5.0, 6.0],
                      [ 3.0, 4.0]]) 
        p = self.l[[0,2,1]]
        msg = printfail(t, p.x, 'x')   
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode           
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 2, 1], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))     

    def test_getitem_13(self):
        "larry.__getitem___13"
        t = np.array([[ 1.0, nan],
                      [ 5.0, 6.0],
                      [ 3.0, 4.0]]) 
        p = self.l[[0.99,2.6,1.78]]
        msg = printfail(t, p.x, 'x')   
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode           
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 2, 1], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_getitem_14(self):
        "larry.__getitem___14"
        t = np.array([[ 3.0, 4.0],
                      [ 1.0, nan]]) 
        p = self.l[[True, False]]
        msg = printfail(t, p.x, 'x')   
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode           
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[1, 0], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_getitem_15(self):
        "larry.__getitem___15"
        self.failUnlessRaises(ValueError, self.l.__getitem__, [0,1,0])
        
    def test_getitem_16(self):
        "larry.__getitem___16"
        t = np.array([[ 1.0, nan]]) 
        p = self.l[np.array([True, False]),:]
        msg = printfail(t, p.x, 'x')   
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode           
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))        

    def test_getitem_17(self):
        "larry.__getitem___17"
        t = np.array([[ 1.0],
                      [ 3.0],
                      [ 5.0]]) 
        p = self.l[:, np.array([True, False])]
        msg = printfail(t, p.x, 'x')   
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode           
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_getitem_18(self):
        "larry.__getitem___18"
        t = np.array([[3.0, 4.0]])
        p = self.l[1.9]
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_getitem_19(self):
        "larry.__getitem___19"
        t = 4.0
        p = self.l[1.1,1.1]
        msg = printfail(t, p, 'x')      
        self.assert_((abs(t - p) < self.tol).all(), msg)                
                
    def test_setitem_1(self):
        "larry.__setitem___1"
        t = np.array([[ 1.0, nan],
                      [ 3.0, 4.0],
                      [ 5.0,-1.0]]) 
        p = self.l
        p[-1,-1] = -1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))         

    def test_setitem_2(self):
        "larry.__setitem___2"
        t = np.array([[ 1.0,-1.0],
                      [ 3.0,-1.0],
                      [ 5.0,-1.0]]) 
        p = self.l
        p[:,-1] = -1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 

    def test_setitem_3(self):
        "larry.__setitem___3"
        t = np.array([[ 1.0, nan],
                      [-1.0,-1.0],
                      [-1.0,-1.0]])  
        idx = self.l.x.sum(1) > 2
        idx = np.where(idx)[0]
        p = self.l
        p[idx,:] = -1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_setitem_4(self):
        "larry.__setitem___4"
        t = np.array([[-1.0, nan],
                      [-1.0,-1.0],
                      [-1.0,-1.0]]) 
        p = self.l
        p[p == p] = -1
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 

    def test_setitem_5(self):
        "larry.__setitem___5"
        t = np.array([0, 1, 2, 9]) 
        p = self.l2
        p[-1] = 9
        msg = printfail(t, p.x, 'x')        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2, 3]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 
        
    def test_setitem_6(self):
        "larry.__setitem___6"
        t = np.array([0, 1, 2, 3]) 
        p = self.l2
        p[:] = [0, 1, 2, 3]
        msg = printfail(t, p.x, 'x')        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2, 3]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))         
        
    def test_setitem_7(self):
        "larry.__setitem___7"
        t = np.array([0, 9, 8, 7]) 
        p = self.l2
        p[1:] = [9, 8, 7]
        msg = printfail(t, p.x, 'x')        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2, 3]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 

    def test_setitem_8(self):
        "larry.__setitem___8"
        t = np.array([9, 8, 7, 3]) 
        p = self.l2
        p[:-1] = larry([9, 8, 7])
        msg = printfail(t, p.x, 'x')        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2, 3]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 

    def test_setitem_9(self):
        "larry.__setitem___9"
        t = np.array([[ 9.0, 8.0],
                      [ 7.0, 6.0],
                      [ 5.0, 6.0]]) 
        p = self.l
        p[:2,:2] = larry([[9.0, 8.0], [7.0, 6.0]])
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_set_1(self):
        "larry.set_1"
        t = np.array([[ 1.0, nan],
                      [ 3.0, 4.0],
                      [ 5.0,-1.0]]) 
        p = self.l
        p.set([2,1], -1)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label')) 

    def test_get_1(self):
        "larry.get_1"
        t = 3.0 
        p = self.l.get([1,0])
        msg = printfail(t, p, 'x')       
        self.assert_((abs(t - p) < self.tol).all(), msg)

    def test_getx_1(self):
        "larry.getx_1"
        t = np.array([[ 1.0, nan],
                      [ 3.0, 4.0],
                      [ 5.0, 6.0]]) 
        p = self.l.getx(copy=True)
        msg = printfail(t, p, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg)
        self.assert_(p is not self.x, 'reference but should be copy')                

    def test_getx_2(self):
        "larry.getx_2"
        t = np.array([[ 1.0, nan],
                      [ 3.0, 4.0],
                      [ 5.0, 6.0]]) 
        p = self.l.getx(copy=False)
        msg = printfail(t, p, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg)
        self.assert_(p is self.x, 'copy but should be reference')

    def test_getx_3(self):
        "larry.getx_3"
        t = np.array([0, 1, 2, 3]) 
        p = self.l2.getx(copy=True)
        msg = printfail(t, p, 'x')     
        self.assert_((abs(t - p) < self.tol).all(), msg)
        self.assert_(p is not self.x, 'reference but should be copy')     
        
    def test_fill_1(self):
        "larry.fill_1"
        t = np.array([[-1.0,-1.0],
                      [-1.0,-1.0],
                      [-1.0,-1.0]]) 
        p = self.l
        p.fill(-1)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        
    def test_pull_1(self):
        "larry.pull_1"
        original = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        actual = original.pull('r0', axis=0)
        desired = larry([1, 2], [['c0', 'c1']])
        ale(actual, desired, 'pull_1')
        
    def test_pull_2(self):
        "larry.pull_2"        
        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        label = [['experiment1', 'experient2'], ['r0', 'r1'], ['c0', 'c1']]
        original = larry(x, label)
        actual = original.pull('experiment1', axis=0)
        desired = larry([[1, 2],
                         [3, 4]],
                        [['r0', 'r1'], ['c0', 'c1']])
        ale(actual, desired, 'pull_2')
       

class Test_label(unittest.TestCase):
    "Test label functions of the larry class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999    
        self.x = np.array([[ 1.0, nan],
                           [ 3.0, 4.0],
                           [ 5.0, 6.0]])                                            
        self.l = larry(self.x)
        self.x2 = np.array([0, 1, 2, 3])
        self.l2 = larry(self.x2) 
                        
    def test_maxlabel_1(self):
        "larry.maxlabel_1"
        t = 2
        p = self.l.maxlabel()
        self.assert_(t == p, printfail(t, p, 'label'))  
        
    def test_maxlabel_2(self):
        "larry.maxlabel_2"
        t = 2
        p = self.l.maxlabel(0)
        self.assert_(t == p, printfail(t, p, 'label'))               

    def test_maxlabel_3(self):
        "larry.maxlabel_3"
        t = 1
        p = self.l.maxlabel(1)
        self.assert_(t == p, printfail(t, p, 'label'))
        
    def test_maxlabel_4(self):
        "larry.maxlabel_4"
        t = 3
        p = self.l2.maxlabel()
        self.assert_(t == p, printfail(t, p, 'label'))         

    def test_maxlabel_5(self):
        "larry.maxlabel_5"
        self.failUnlessRaises(IndexError, self.l2.maxlabel, 1)

    def test_minlabel_1(self):
        "larry.minlabel_1"
        t = 0
        p = self.l.minlabel()
        self.assert_(t == p, printfail(t, p, 'label'))  
        
    def test_minlabel_2(self):
        "larry.minlabel_2"
        t = 0
        p = self.l.minlabel(0)
        self.assert_(t == p, printfail(t, p, 'label'))               

    def test_minlabel_3(self):
        "larry.minlabel_3"
        t = 0
        p = self.l.minlabel(1)
        self.assert_(t == p, printfail(t, p, 'label'))
        
    def test_minlabel_4(self):
        "larry.minlabel_4"
        t = 0
        p = self.l2.minlabel()
        self.assert_(t == p, printfail(t, p, 'label'))         

    def test_minlabel_5(self):
        "larry.minlabel_5"
        self.failUnlessRaises(IndexError, self.l2.minlabel, 1)        
        
    def test_getlabel_1(self):
        "larry.getlabel_1"
        t = [0, 1, 2]
        axis = 0
        p = self.l.getlabel(axis, copy=True)
        self.assert_(t == p, printfail(t, p, 'label0'))
        self.assert_(p is not self.l.label[axis], 'reference but should be copy')                  

    def test_getlabel_2(self):
        "larry.getlabel_2"
        t = [0, 1, 2]
        axis = 0
        p = self.l.getlabel(axis, copy=False)
        self.assert_(t == p, printfail(t, p, 'label0'))
        self.assert_(p is self.l.label[axis], 'copy but should be reference')

    def test_getlabel_3(self):
        "larry.getlabel_3"
        t = [0, 1]
        axis = 1
        p = self.l.getlabel(axis, copy=True)
        self.assert_(t == p, printfail(t, p, 'label0'))
        self.assert_(p is not self.l.label[axis], 'reference but should be copy')                  

    def test_getlabel_4(self):
        "larry.getlabel_4"
        t = [0, 1]
        axis = 1
        p = self.l.getlabel(axis, copy=False)
        self.assert_(t == p, printfail(t, p, 'label0'))
        self.assert_(p is self.l.label[axis], 'copy but should be reference')  

    def test_getlabel_5(self):
        "larry.getlabel_5"
        t = [0, 1, 2, 3]
        axis = 0
        p = self.l2.getlabel(axis, copy=True)
        self.assert_(t == p, printfail(t, p, 'label0'))
        self.assert_(p is not self.l2.label[axis], 'reference but should be copy') 
      
    def test_labelindex_1(self):
        "larry.labelindex_1"
        t = 2
        axis = 0
        p = self.l.labelindex(2, axis)
        self.assert_(t == p, printfail(t, p, 'label'))                 

    def test_labelindex_2(self):
        "larry.labelindex_2"
        t = 1
        axis = 1
        p = self.l.labelindex(1, axis)
        self.assert_(t == p, printfail(t, p, 'label0'))

    def test_labelindex_3(self):
        "larry.labelindex_3"
        self.failUnlessRaises(ValueError, self.l.labelindex, 3, None)
        
    def test_labelindex_4(self):
        "larry.labelindex_4"
        t = 3
        axis = 0
        p = self.l2.labelindex(3, axis)
        self.assert_(t == p, printfail(t, p, 'label'))
        
    def test_maplabel_1(self):
        "label.maplabel_1"
        d = datetime.date
        y1 = larry([1, 2], [[d(2010,1,1), d(2010,1,2)]])
        y2 = y1.maplabel(datetime.date.toordinal)
        self.assert_(y2.label == [[733773, 733774]], 'Did not map correctly')
        self.assert_((y2.x == np.array([1, 2])).all(), 'x values changed')
        def func(x):
            return x + 1       
        y3 = y2.maplabel(func)
        self.assert_(y3.label == [[733774, 733775]], 'Did not map correctly')
        self.assert_((y2.x == np.array([1, 2])).all(), 'x values changed')

class Test_calc(unittest.TestCase):
    "Test calc functions of larry class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999
        self.x1 = np.array([[ 2.0, 2.0, 3.0, 1.0],
                            [ 3.0, 2.0, 2.0, 1.0],
                            [ 1.0, 1.0, 1.0, 1.0]])
        self.l1 = larry(self.x1)         
        self.x2 = np.array([[ 2.0, 2.0, nan, 1.0],
                            [ nan, nan, nan, 1.0],
                            [ 1.0, 1.0, nan, 1.0]])
        self.l2 = larry(self.x2)
        self.x3 = np.array([1, 2, 3, 4, 5])  
        self.l3 = larry(self.x3)
        self.x4 = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                            [ 1.0, nan, 2.0, nan, nan],
                            [ 2.0, 2.0, nan, nan, nan],
                            [ 3.0, 3.0, 3.0, 3.0, nan]])
        self.l4 = larry(self.x4)       
        self.x5 = np.array([[1.0, nan, 6.0, 0.0, 8.0],
                            [2.0, 4.0, 8.0, 0.0,-1.0]])
        self.l5 = larry(self.x5)                    
        self.x6 = np.array([[  nan,  nan,  nan,  nan,  nan],
                            [  nan,  nan,  nan,  nan,  nan]])                                                                                    
        self.l6 = larry(self.x6)
        self.x7 = np.array([[nan, 2.0],
                            [1.0, 3.0],
                            [3.0, 1.0]])  
        self.l7 = larry(self.x7)                                  
    
    def test_demean_1(self):
        "larry.demean_1"
        t = np.array([[ 0.5, 0.5, nan, 0.0],
                      [ nan, nan, nan, 0.0],
                      [-0.5,-0.5, nan, 0.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.demean(0)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test_demean_2(self):
        "larry.demean_2"
        t = np.array([[ 2.0-5.0/3, 2.0-5.0/3, nan, 1.0-5.0/3],
                      [       nan,       nan, nan,       0.0],
                      [ 0.0,  0.0,       nan,            0.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.demean(1)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test_demean_3(self):
        "larry.demean_3"
        t = np.array([-2, -1, 0, 1, 2])
        label = [[0, 1, 2, 3, 4]]
        p = self.l3.demean(0)
        msg = printfail(t, p.x, 'x')            
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l3), 'Reference found')

    def test_demean_4(self):
        "larry.demean_4"
        t = np.array([-0.5, 0.5])
        label = [[0, 1]]
        p = larry([1, 2]).demean(0)
        msg = printfail(t, p.x, 'x')            
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_demedian_1(self):
        "larry.demedian_1"
        t = np.array([[ 0.5, 0.5, nan, 0.0],
                      [ nan, nan, nan, 0.0],
                      [-0.5,-0.5, nan, 0.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.demedian(0)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test_demedian_2(self):
        "larry.demedian_2"
        t = np.array([[ 0.0, 0.0, nan,-1.0],
                      [ nan, nan, nan, 0.0],
                      [ 0.0, 0.0, nan, 0.0]])                    
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.demedian(1)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')

    def test_demedian_3(self):
        "larry.demedian_3"
        t = np.array([-2, -1, 0, 1, 2])
        label = [[0, 1, 2, 3, 4]]
        p = self.l3.demedian(0)
        msg = printfail(t, p.x, 'x')            
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l3), 'Reference found')

    def test_demedian_4(self):
        "larry.demedian_4"
        t = np.array([-0.5, 0.5])
        label = [[0, 1]]
        p = larry([1, 2]).demedian(0)
        msg = printfail(t, p.x, 'x')            
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))

    def test_zscore_1(self):
        "larry.zscore_1"
        t = self.x1.copy()
        t = t - t.mean(0)
        t = t / t.std(0)                                          
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.zscore(0)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode                        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')

    def test_zscore_2(self):
        "larry.zscore_2"
        t = self.x1.copy()
        t = t - t.mean(1).reshape(-1,1)
        t = t / t.std(1).reshape(-1,1)                                         
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.zscore(1)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode                        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')
        
    def test_zscore_3(self):
        "larry.zscore_3"
        t = self.x3.copy()
        t = t - t.mean()
        t = t / t.std()                                          
        label = [[0, 1, 2, 3, 4]]
        p = self.l3.zscore()
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode                        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l3), 'Reference found')

    def test_push_1(self):
        "larry.push_1"
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 1.0, 1.0, 2.0, 2.0, 2.0],
                      [ 2.0, 2.0, 2.0, 2.0, nan],
                      [ 3.0, 3.0, 3.0, 3.0, 3.0]]) 
        label = [[0, 1, 2, 3], [0, 1, 2, 3, 4]]                                  
        p = self.l4.push(2)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode    
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')
                
    def test_push_2(self):
        "larry.push_2"
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])                   
        label = [[0, 1, 2, 3], [0, 1, 2, 3, 4]]                                  
        p = self.l4.push(0)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode    
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')

    def test_push_3(self):
        "larry.push_3"
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 1.0, 1.0, 2.0, 2.0, nan],
                      [ 2.0, 2.0, 2.0, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, 3.0]])                   
        label = [[0, 1, 2, 3], [0, 1, 2, 3, 4]]                                  
        p = self.l4.push(1)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode    
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')


    def test_movingsum_1(self):
        "larry.movingsum_1"
        t = np.array([[ 4.0, 5.0, 4.0],
                      [ 5.0, 4.0, 3.0],
                      [ 2.0, 2.0, 2.0]])
        t = np.array([[ nan,   4.0,   5.0,   4.0],
                      [ nan,   5.0,   4.0,   3.0],
                      [ nan,   2.0,   2.0,   2.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.movingsum(2)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')
        
    def test_movingsum_2(self):
        "larry.movingsum_2"
        t = np.array([[ 4.0, 5.0, 4.0],
                      [ 5.0, 4.0, 3.0],
                      [ 2.0, 2.0, 2.0]])
        t = np.array([[ nan,   4.,   5.,   4.],
                      [ nan,   5.,   4.,   3.],
                      [ nan,   2.,   2.,   2.]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')            

    def test_movingsum_3(self):
        "larry.movingsum_3"
        t = np.array([[ 5.0, 4.0, 5.0, 2.0],
                      [ 4.0, 3.0, 3.0, 2.0]])
        t = np.array([[ nan,  nan,  nan,  nan],
                       [  5.,   4.,   5.,   2.],
                       [  4.,   3.,   3.,   2.]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.movingsum(2, axis=0)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')
        
    def test_movingsum_4(self):
        "larry.movingsum_4"
        t = np.array([[ 2.0, 2.0, 3.0, 1.0],
                      [ 3.0, 2.0, 2.0, 1.0],
                      [ 1.0, 1.0, 1.0, 1.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l1.movingsum(1, axis=0)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode   
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')

    def test_movingsum_5(self):
        "larry.movingsum_5"
        t = np.array([[ 4.0, 2.0, 1.0],
                      [ nan, nan, 1.0],
                      [ 2.0, 1.0, 1.0]])  
        t = np.array([[ nan, 4.0, 2.0, 1.0],
                      [ nan, nan, nan, 1.0],
                      [ nan, 2.0, 1.0, 1.0]])                       
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.movingsum(2)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode       
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')
        
    def test_movingsum_6(self):
        "larry.movingsum_6"
        t = np.array([[ nan, 4.0, 4.0, 2.0],
                      [ nan, nan, nan, 2.0],
                      [ nan, 2.0, 2.0, 2.0]]) 
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')            

    def test_movingsum_7(self):
        "larry.movingsum_7"
        t = np.array([[ nan, nan, nan, nan],
                      [ 2.0, 2.0, nan, 2.0],
                      [ 1.0, 1.0, nan, 2.0]])                                            
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.movingsum(2, axis=0)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode                  
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')
   
    def test_movingsum_8(self):
        "larry.movingsum_8"
        t = np.array([[ 2.0, 2.0, nan, 1.0],
                      [ nan, nan, nan, 1.0],
                      [ 1.0, 1.0, nan, 1.0]])
        label = [[0, 1, 2], [0, 1, 2, 3]]
        p = self.l2.movingsum(1, axis=0)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')
        
    def test_movingsum_9(self):
        "larry.movingsum_9"
        t = np.array([nan, 3.0, 5.0, 7.0, 9.0])
        label = [[0, 1, 2, 3, 4]]
        p = self.l3.movingsum(2, axis=0)
        msg = printfail(t, p.x, 'x')    
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode          
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l3), 'Reference found')        

    def test_movingsum_forward_1(self):
        "larry.movingsum_forward_1"    
        t = np.array([[2.0, 12.0, 6.0, 8.0, nan],
                      [6.0, 12.0, 8.0,-1.0, nan]]) 
        label = [[0, 1], [0, 1, 2, 3, 4]]              
        skip = 0                     
        p = self.l5.movingsum_forward(2, skip, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found')
        
    def test_movingsum_forward_2(self):
        "larry.movingsum_forward_2"    
        t = np.array([[1.0,  6.0, 6.0, 8.0, nan],
                      [6.0, 12.0, 8.0,-1.0, nan]]) 
        label = [[0, 1], [0, 1, 2, 3, 4]]              
        skip = 0                     
        p = self.l5.movingsum_forward(2, skip, norm=False)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found')       

    def test_movingsum_forward_3(self):
        "larry.movingsum_forward_3"    
        t = np.array([[12.0, 6.0, 8.0, nan, nan],
                      [12.0, 8.0,-1.0, nan, nan]]) 
        label = [[0, 1], [0, 1, 2, 3, 4]]              
        skip = 1                     
        p = self.l5.movingsum_forward(2, skip, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found') 

    def test_movingsum_forward_4(self):
        "larry.movingsum_forward_4"    
        t = np.array([[ 6.0, 6.0, 8.0, nan, nan],
                      [12.0, 8.0,-1.0, nan, nan]]) 
        label = [[0, 1], [0, 1, 2, 3, 4]]                
        skip = 1                     
        p = self.l5.movingsum_forward(2, skip, norm=False)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found') 

    def test_movingsum_forward_5(self):
        "larry.movingsum_forward_5"    
        t = np.array([[2.0, 4.0, 8.0, 0.0,-1.0],
                      [nan, nan, nan, nan, nan]])
        label = [[0, 1], [0, 1, 2, 3, 4]]              
        skip = 1                     
        p = self.l5.movingsum_forward(1, skip, axis=0)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found')
 
    def test_ranking_1(self):
        "larry.ranking_1"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan]])
        lx = larry(x)
        t = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                      [ 0.0,  -1.0,   nan,   nan,   nan],
                      [ 1.0,   1.0,   1.0,   0.0,   nan]])                     
        p = lx.ranking(axis=0)
        label = [range(3), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found') 

    def test_ranking_2(self):
        "larry.ranking_2"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0,   3.0,   nan]])
        lx = larry(x)
        t = np.array([[-1.0,   nan,  -1.0,   nan,   nan],
                      [ 0.0,  -1.0,   nan,   nan,   nan],
                      [ 1.0,   1.0,   1.0,   0.0,   nan]])                    
        p = lx.ranking()
        label = [range(3), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found') 

    def test_ranking_3(self):
        "larry.ranking_3"
        x = np.array([[ 1.0,   nan,   2.0,   nan,   nan],
                      [ 2.0,   2.0,   nan,   nan,   nan],
                      [ 3.0,   3.0,   3.0, 3.0  ,   nan],
                      [ 4.0,   2.0,   3.0, 1.0  , 0.0  ]])   
        lx = larry(x)
        t = np.array([[-1.0,   nan,   1.0,   nan,   nan],
                      [ 0.0,   0.0,   nan,   nan,   nan],
                      [ 0.0,   0.0,   0.0,   0.0,   nan],
                      [ 1.0,   0.0,   0.5,  -0.5,  -1.0]])                    
        p = lx.ranking(axis=1)
        label = [range(4), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')
        
    def test_ranking_4(self):
        "larry.ranking_4"  
        x = np.array([[3.0], [1.0], [2.0]])
        lx = larry(x)
        t = np.array([[1.0],[-1.0], [0.0]])
        p = lx.ranking(axis=0)
        label = [range(3), range(1)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')

    def test_ranking_5(self):
        "larry.ranking_5"  
        x = np.array([[3.0], [1.0], [2.0]])
        lx = larry(x)
        t = np.array([[0.0], [0.0], [0.0]])
        p = lx.ranking(axis=1)
        label = [range(3), range(1)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')
        
    def test_ranking_6(self):
        "larry.ranking_6"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        lx = larry(x)
        t = np.array([[ 0.0,   nan,   0.5,  nan,   nan],
                      [ 0.0,  -1.0,   nan,  nan,   nan],
                      [ 0.0,   0.0,  -1.0,  1.0,   nan],
                      [ 0.0,   1.0,   0.5, -1.0,   0.0]])                    
        p = lx.ranking()
        label = [range(4), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')      

    def test_ranking_7(self):
        "larry.ranking_7"
        x = np.array([[ 1.0,   nan,   1.0,   nan,   nan],
                      [ 1.0,   1.0,   nan,   nan,   nan],
                      [ 1.0,   2.0,   0.0,   2.0,   nan],
                      [ 1.0,   3.0,   1.0,   1.0,   0.0]])   
        lx = larry(x)
        t = np.array([[ 0.0,   nan ,   0.0,  nan  ,   nan],
                      [ 0.0,   0.0 ,   nan,  nan  ,   nan],
                      [-1.0/3, 2.0/3, -1.0,  2.0/3,   nan],
                      [ 0.0,   1.0 ,   0.0,  0.0  ,  -1.0]])                    
        p = lx.ranking(1)
        label = [range(4), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')

    def test_ranking_8(self):
        "larry.ranking_8"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]]) 
        lx = larry(x)                
        t = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                      [-2.0/3, -2.0/3, -1.0/3,   0.0],
                      [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                      [ 2.0/3,    1.0,  2.0/3,   1.0]])                    
        p = lx.ranking(0)
        label = [range(4), range(4)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found')
        
    def test_ranking_9(self):
        "larry.ranking_9"
        x = np.array([[ 1.0,   1.0,   1.0,   1.0],
                      [ 1.0,   1.0,   2.0,   2.0],
                      [ 2.0,   2.0,   3.0,   2.0],
                      [ 2.0,   3.0,   3.0,   3.0]]) 
        x = x.T  
        lx = larry(x)
        t = np.array([[-2.0/3, -2.0/3,   -1.0,  -1.0],
                      [-2.0/3, -2.0/3, -1.0/3,   0.0],
                      [ 2.0/3,  1.0/3,  2.0/3,   0.0],
                      [ 2.0/3,    1.0,  2.0/3,   1.0]])
        t = t.T                                       
        p = lx.ranking(1)
        label = [range(4), range(4)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, lx), 'Reference found') 

    def test_movingrank_1(self):
        "larry.movingrank_1"    
        t = self.x6 
        p = self.l6.movingrank(2)
        label = [range(2), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l6), 'Reference found') 
    
    def test_movingrank_2(self):
        "larry.movingrank_2"    
        t = np.array([[  nan,  nan,  nan,-1.0,1.0],
                      [  nan,1.0,1.0,-1.0,-1.0]]) 
        p = self.l5.movingrank(2)
        label = [range(2), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found')          

    def test_movingrank_3(self):
        "larry.movingrank_3"    
        t = np.array([[nan,  nan,  nan,  nan,  nan],
                      [1.0,  nan,  1.0,  0.0,  -1.0]])
        p = self.l5.movingrank(2, axis=0)
        label = [range(2), range(5)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l5), 'Reference found')
        
    def test_movingrank_4(self):
        "larry.movingrank_4"    
        t = np.array([[nan,  nan],
                      [nan,  1.0],
                      [nan, -1.0]])
        p = self.l7.movingrank(2)
        label = [range(3), range(2)]
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l7), 'Reference found') 
        
    def test_quantile_1(self):
        "larry.quantile_1"    
        t = np.array([[-1., -1.,  1., -1.],
                      [ 1.,  1., -1., -1.],
                      [-1., -1., -1.,  1.]])
        label = [range(3), range(4)]                                    
        p = self.l1.quantile(2)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')                 

    def test_quantile_2(self):
        "larry.quantile_2"    
        t = np.array([[ 0.,  0.,  1., -1.],
                      [ 1.,  1.,  0.,  0.],
                      [-1., -1., -1.,  1.]])
        label = [range(3), range(4)]                                    
        p = self.l1.quantile(3)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found') 

    def test_cut_missing_1(self):
        "larry.cut_missing_1" 
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])        
        label = [range(4), range(5)]                                    
        p = self.l4.cut_missing(0.9, axis=0)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found') 

    def test_cut_missing_2(self):
        "larry.cut_missing_2" 
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 1.0, nan, 2.0, nan, nan],
                      [ 2.0, 2.0, nan, nan, nan],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])        
        label = [range(4), range(5)]                                    
        p = self.l4.cut_missing(0.9, axis=1)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found') 

    def test_cut_missing_3(self):
        "larry.cut_missing_3" 
        t = np.array([[ nan, 1.0, 2.0],
                      [ 1.0, nan, 2.0],
                      [ 2.0, 2.0, nan],
                      [ 3.0, 3.0, 3.0]])        
        label = [range(4), range(3)]                                    
        p = self.l4.cut_missing(0.5, axis=0)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found') 

    def test_cut_missing_4(self):
        "larry.cut_missing_4" 
        t = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
                      [ 3.0, 3.0, 3.0, 3.0, nan]])        
        label = [[0, 3], range(5)]                                    
        p = self.l4.cut_missing(0.5, axis=1)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l4), 'Reference found')             


class Test_alignment(unittest.TestCase):
    "Test alignment functions of larry class"
    
    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999
        self.x = np.array([[ nan, nan],
                           [ 1.0, 2.0],
                           [ 3.0, 4.0]])                  
        self.l = larry(self.x)
        self.l2 = self.l.copy()
        self.l2.label[0] = [2,1,0]
        self.l2.label[1] = [1,0]
        self.x3 = np.random.rand(2,3,4)                
        self.l3 = larry(self.x3)                                                      

    def test_morph_1(self):
        "larry.morph_1"
        t = np.array([[ 1.0, 2.0],
                      [ nan, nan],
                      [ nan, nan],
                      [ 3.0, 4.0]]) 
        label = [1, 0, 3, 2]
        axis = 0
        p = self.l.morph(label, axis)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [label, [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')         

    def test_morph_2(self):
        "larry.morph_2"
        t = np.array([[ nan, nan, nan, nan],
                      [ 2.0, 1.0, nan, nan],
                      [ 4.0, 3.0, nan, nan]]) 
        label = [1, 0, 3, 2]
        axis = 1
        p = self.l.morph(label, axis)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], label]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_morph_3(self):
        "larry.morph_3"
        t = np.array([[ nan, nan],
                      [ 1.0, 2.0],
                      [ 3.0, 4.0]])  
        label = [0, 1, 2]
        axis = 0
        p = self.l.morph(label, axis)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [label, [0, 1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')         

    def test_morph_4(self):
        "larry.morph_4"
        l = larry(['a', 'b', 'cc'])
        t = np.array(['cc', 'b', 'a'])  
        label = [2, 1, 0]
        axis = 0
        p = l.morph(label, axis)
        msg = printfail(t, p.x, 'x')   
        self.assert_((t == p.x).all(), msg)
        label = [label]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, l), 'Reference found')

    def test_morph_5(self):
        "larry.morph_5"
        l = larry(['a', 'b', 'cc'])
        t = np.array(['cc', 'b', 'a', ''])  
        label = [2, 1, 0, 3]
        axis = 0
        p = l.morph(label, axis)
        msg = printfail(t, p.x, 'x')   
        self.assert_((t == p.x).all(), msg)
        label = [label]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, l), 'Reference found')

    def test_morph_6(self):
        "larry.morph_6"
        d = datetime.date
        l = larry([d(2011,1,1), d(2011,1,2)])
        t = np.array([d(2011,1,2), d(2011,1,1)])  
        label = [1, 0]
        axis = 0
        p = l.morph(label, axis)
        msg = printfail(t, p.x, 'x')   
        self.assert_((t == p.x).all(), msg)
        label = [label]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, l), 'Reference found')
        
    def test_morph_like_1(self):
        "larry.morph_like_1"
        t = np.array([[ 4.0, 3.0],
                      [ 2.0, 1.0],
                      [ nan, nan]]) 
        p = self.l.morph_like(self.l2)
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[2,1,0], [1, 0]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        self.assert_(noreference(p, self.l2), 'Reference found')         
        
    def test_lag_1(self):
        "larry.lag_1"
        t = np.array([[nan], [2], [4.0/3.0]])
        p = self.l / self.l.lag(1) 
        msg = printfail(t, p.x, 'x')       
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p.x) < self.tol).all(), msg) 
        label = [[0, 1, 2], [1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')
        
    def test_flatten_1(self):
        "larry.flatten_1"
        y = larry([1, 2, 3])
        order = 'C'
        f = y.flatten(order)
        label = [[(0,), (1,), (2,)]]        
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x.flatten(order)).all(), 'data are wrong')      

    def test_flatten_2(self):
        "larry.flatten_2"
        y = larry([1, 2, 3])
        order = 'F'
        f = y.flatten(order)
        label = [[(0,), (1,), (2,)]]        
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x.flatten(order)).all(), 'data are wrong')        
        
    def test_flatten_3(self):
        "larry.flatten_3"
        y = larry([[1, 2], [3, 4]])
        f = y.flatten()
        label = [[(0,0), (0,1), (1,0), (1,1)]]
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x.flatten()).all(), 'data are wrong')

    def test_flatten_4(self):
        "larry.flatten_4"
        y = larry([[1, 2], [3, 4]])
        order = 'F'
        f = y.flatten(order)
        label = [[(0,0), (1,0), (0,1), (1,1)]]
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x.flatten(order)).all(), 'data are wrong')

    def test_unflatten_1(self):
        "larry.unflatten_1"
        y = larry([1, 2, 3])
        order = 'C'
        f = y.flatten(order).unflatten()
        label = y.label     
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x).all(), 'data are wrong')  

    def test_unflatten_2(self):
        "larry.unflatten_2"
        y = larry([1, 2, 3])
        order = 'F'
        f = y.flatten(order).unflatten()
        label = y.label        
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x).all(), 'data are wrong') 

    def test_unflatten_3(self):
        "larry.unflatten_3"
        y = larry([[1, 2], [3, 4]])
        f = y.flatten().unflatten()
        label = y.label 
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x).all(), 'data are wrong')

    def test_unflatten_4(self):
        "larry.unflatten_4"
        y = larry([[1, 2], [3, 4]])
        order = 'F'
        f = y.flatten(order).unflatten()
        label = y.label 
        self.assert_(f.label == label, 'labels are wrong')
        self.assert_((f.x == y.x).all(), 'data are wrong')
        
    def test_sortaxis_1(self):
        "larry.sortaxis_1"
        original = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        actual = original.sortaxis()
        desired = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        ale(actual, desired, 'sortaxis_1', original=original)
             
    def test_sortaxis_2(self):
        "larry.sortaxis_2"
        original = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        actual = original.sortaxis(reverse=True)
        desired = original.copy()
        ale(actual, desired, 'sortaxis_1', original=original)

    def test_sortaxis_3(self):
        "larry.sortaxis_3"
        original = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        actual = original.sortaxis(axis=0)
        desired = larry([[2, 1], [4, 3]], [['a', 'b'], ['d', 'c']])
        ale(actual, desired, 'sortaxis_1', original=original)

    def test_sortaxis_4(self):
        "larry.sortaxis_4"
        original = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        actual = original.sortaxis(axis=1)
        desired = larry([[3, 4], [1, 2]], [['b', 'a'], ['c', 'd']]) 
        ale(actual, desired, 'sortaxis_1', original=original)
        
    def test_flipaxis_1(self):
        "larry.flipaxis_1"
        original = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]) 
        actual = original.flipaxis()
        desired = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        ale(actual, desired, 'flipaxis_1', original=original)
        
    def test_flipaxis_2(self):
        "larry.flipaxis_2"
        original = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]) 
        actual = original.flipaxis(copy=False)
        desired = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        ale(actual, desired, 'flipaxis_2', original=original, iscopy=False)

    def test_flipaxis_3(self):
        "larry.flipaxis_3"
        original = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]) 
        actual = original.flipaxis(0).flipaxis(1)
        desired = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']]) 
        ale(actual, desired, 'flipaxis_3', original=original)

    def test_flipaxis_4(self):
        "larry.flipaxis_4"
        original = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]) 
        actual = original.flipaxis(0)
        desired = larry([[3, 4], [1, 2]], [['b', 'a'], ['c', 'd']]) 
        ale(actual, desired, 'flipaxis_4', original=original)

    def test_insertaxis_1(self):
        "larry.insertaxis_1"
        original = larry([1, 2]) 
        actual = original.insertaxis(0, 'new')
        desired = larry([[1, 2]], [['new'], [0, 1]]) 
        ale(actual, desired, 'insertaxis_1', original=original)

    def test_insertaxis_2(self):
        "larry.insertaxis_2"
        original = larry([1, 2]) 
        actual = original.insertaxis(1, 'new')
        desired = larry([[1], [2]], [[0, 1], ['new']]) 
        ale(actual, desired, 'insertaxis_2', original=original)

    def test_insertaxis_3(self):
        "larry.insertaxis_3"
        original = larry([1, 2]) 
        actual = original.insertaxis(-2, 'new')
        desired = larry([[1, 2]], [['new'], [0, 1]]) 
        ale(actual, desired, 'insertaxis_1', original=original)

    def test_insertaxis_4(self):
        "larry.insertaxis_4"
        original = larry([1, 2]) 
        actual = original.insertaxis(-1, 'new')
        desired = larry([[1], [2]], [[0, 1], ['new']]) 
        ale(actual, desired, 'insertaxis_2', original=original)
        
                        
class Test_random(unittest.TestCase):
    "Test randomizing functions of the larry class"

    def setUp(self):
        self.lar = larry(np.random.randint(0, 1000000, (200, 200)))

    def test_shuffle_1(self):
        "larry.shuffle_1"
        y = self.lar.copy()
        y.shuffle(axis=None)
        self.assert_(y.shape == self.lar.shape, 'shape changed')
        self.assert_((y.x != self.lar.x).any(), 'No shuffling took place')
        self.assert_(y.label == self.lar.label, 'labels changed')

    def test_shuffle_2(self):
        "larry.shuffle_2"
        y = self.lar.copy()
        y.shuffle()
        self.assert_(y.shape == self.lar.shape, 'shape changed')
        self.assert_((y.x != self.lar.x).any(), 'No shuffling took place')
        self.assert_(y.label == self.lar.label, 'labels changed')

    def test_shufflelabel_1(self):
        "larry.shufflelabel_1"
        y = self.lar.copy()
        y.shufflelabel()
        self.assert_(y.shape == self.lar.shape, 'shape changed')
        self.assert_((y.x == self.lar.x).all(), 'Values shuffled')
        self.assert_(y.label[1] == self.lar.label[1], 'labels shuffled')
        self.assert_(y.label[0] != self.lar.label[0], 'No shuffling')      

    def test_shufflelabel_2(self):
        "larry.shufflelabel_2"
        y = self.lar.copy()
        y.shufflelabel(axis=None)
        self.assert_(y.shape == self.lar.shape, 'shape changed')
        self.assert_((y.x == self.lar.x).all(), 'Values shuffled')
        self.assert_(y.label[1] != self.lar.label[1], 'labels shuffled')
        self.assert_(y.label[0] != self.lar.label[0], 'No shuffling') 
        

class Test_properties_01(unittest.TestCase):
    "Test properties larry class"

    def setUp(self):
        self.tol = 1e-8
        self.nancode = -9999
        self.x = np.array([[ 1.0, nan],
                           [ 1.0, 1.0],
                           [ 1.0, 1.0]])
        self.l = larry(self.x)
        self.x1 = np.array([ 1.0, nan])
        self.l1 = larry(self.x1)
        self.x3 = np.array([[[ 1.0, nan],
                             [ 1.0, 1.0],
                             [ 1.0, 1.0]],

                            [[ 2.0, nan],
                             [ 3.0, 6.0],
                             [ 4.0, 7.0]]])
        self.l3 = larry(self.x3)

    def test_01(self):
        "larry.shape"
        t = self.x.shape
        p = self.l.shape
        self.assert_(t == p, printfail(t, p, 'shape'))

    def test_02(self):
        "larry.ndim"
        t = self.x.ndim
        p = self.l.ndim
        self.assert_(t == p, printfail(t, p, 'ndim'))

    def test_05(self):
        "larry.dtype"
        t = self.x.dtype
        p = self.l.dtype
        self.assert_(t == p, printfail(t, p, 'dtype'))

    def test_06(self):
        "larry.nx"
        t = np.isfinite(self.x).sum()
        p = self.l.nx
        self.assert_(t == p, printfail(t, p, 'nx'))

    def test_07(self):
        "larry.T_2d"
        x = self.x.copy()
        x = x.T
        t = larry(x)
        p = self.l.T
        msg = printfail(t, p, 'T')
        t.x[np.isnan(t.x)] = self.nancode
        p.x[np.isnan(p.x)] = self.nancode
        self.assert_(t == p, msg)

    def test_08(self):
        "larry.T_1d"
        x = self.x1.copy()
        x = x.T
        t = larry(x)
        p = self.l1.T
        msg = printfail(t, p, 'T')
        t.x[np.isnan(t.x)] = self.nancode
        p.x[np.isnan(p.x)] = self.nancode
        self.assert_(t == p, msg)

    def test_09(self):
        "larry.T_"
        x = self.x3.copy()
        x = x.T
        t = larry(x)
        p = self.l3.T
        msg = printfail(t, p, 'T')
        t.x[np.isnan(t.x)] = self.nancode
        p.x[np.isnan(p.x)] = self.nancode
        self.assert_(t == p, msg)

    def test_10(self):
        "larry.T_4d"
        x = np.random.randint(0,9,(2,3,4,5))
        x = x.T
        t = larry(x)
        p = larry(x).T
        msg = printfail(t, p, 'T')
        t.x[np.isnan(t.x)] = self.nancode
        p.x[np.isnan(p.x)] = self.nancode
        self.assert_(t == p, msg)
        
    def test_11(self):
        t = self.l.x
        p = self.l.A
        msg = printfail(t, p, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p)] = self.nancode                  
        self.assert_((abs(t - p) < self.tol).all(), msg)     
        self.assert_(t is p, 'no reference found')


class Test_merge(unittest.TestCase):
    "Test merge functions of the larry class"
    
    def test_merge1(self):
        "larry.merge_1"
        lar1 = larry(
               np.array([[ 2.,  2.,  3.,  1.],
                         [ 3.,  2.,  2.,  1.],
                         [ 1.,  1.,  1.,  1.]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[ 2.,  2.,  3.,  1.],
                         [ 3.,  2.,  2.,  1.],
                         [ 1.,  1.,  1.,  1.]]), 
               [[1, 2, 3], ['A', 2, 3, 4]])
        larr = larry(
               np.array([[ 2.,  2.,  3.,  1.,  2.],
                         [ 3.,  2.,  2.,  1.,  3.],
                         [ 1.,  1.,  1.,  1.,  1.]]), 
               [[1, 2, 3], [1, 2, 3, 4, 'A']])
        larm = lar1.merge(lar2, update=True)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)
        assert_raises(ValueError, lar1.merge, lar2, update=False)
       
    def test_merge2(self):
        "larry.merge_2"
        lar1 = larry(
               np.array([[  2.,   2.,  nan,  nan],
                         [  3.,   2.,  nan,  nan],
                         [  1.,   1.,  nan,  nan]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[  2.,  nan,   3.,   1.],
                         [  3.,  nan,   2.,   1.],
                         [  1.,  nan,   1.,   6.]]), 
               [[1, 2, 3], ['A', 2, 3, 4]])
        larr = larry(
               np.array([[ 2.,  2.,  3.,  1.,  2.],
                         [ 3.,  2.,  2.,  1.,  3.],
                         [ 1.,  1.,  1.,  6.,  1.]]), 
               [[1, 2, 3], [1, 2, 3, 4, 'A']])
        larm = lar1.merge(lar2, update=False)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)
        
    def test_merge3(self):
        "larry.merge_3"
        lar1 = larry(
               np.array([[  2.,   2.,   3.,   1.],
                         [  3.,   2.,   2.,   1.],
                         [ nan,  nan,  nan,  nan]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[  2.,  nan,   3.,   1.],
                         [  3.,  nan,   2.,   1.],
                         [  1.,  nan,   1.,   6.]]), 
               [[1, 2, 3], ['A', 2, 3, 4]])
        larr = larry(
               np.array([[  2.,   2.,   3.,   1.,   2.],
                         [  3.,   2.,   2.,   1.,   3.],
                         [ nan,  nan,   1.,   6.,   1.]]), 
               [[1, 2, 3], [1, 2, 3, 4, 'A']])
        larm = lar1.merge(lar2, update=True)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)
        
    def test_merge4(self):
        "larry.merge_4"
        lar1 = larry(
               np.array([[  2.,   2.,   3.,   1.],
                         [  3.,   2.,   2.,   1.],
                         [ nan,  nan,  nan,  nan]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[ 1.,  1.,  1.,  6.]]), 
               [[3], [1, 2, 3, 4]])
        larr = larry(
               np.array([[ 2.,  2.,  3.,  1.],
                         [ 3.,  2.,  2.,  1.],
                         [ 1.,  1.,  1.,  6.]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        larm = lar1.merge(lar2, update=True)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)
        
    def test_merge5(self):
        "larry.merge_5"
        lar1 = larry(
               np.array([[ 2.,  2.,  3.,  1.],
                         [ 3.,  2.,  2.,  1.],
                         [ 1.,  1.,  1.,  6.]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[ 10.,  10.,  10.,  60.]]), 
               [[4], [1, 2, 3, 4]])
        larr = larry(
               np.array([[  2.,   2.,   3.,   1.],
                         [  3.,   2.,   2.,   1.],
                         [  1.,   1.,   1.,   6.],
                         [ 10.,  10.,  10.,  60.]]), 
               [[1, 2, 3, 4], [1, 2, 3, 4]])
        larm = lar1.merge(lar2, update=True)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)
       
    def test_merge6(self):
        "larry.merge_6"
        lar1 = larry(
               np.array([[ 2.,  2.,  3.,  1.],
                         [ 3.,  2.,  2.,  1.],
                         [ 1.,  1.,  1.,  6.]]), 
               [[1, 2, 3], [1, 2, 3, 4]])
        lar2 = larry(
               np.array([[ 20.,  10.],
               [ 10.,  60.]]), 
               [[1, 2], [5, 6]])
        larr = larry(
               np.array([[  2.,   2.,   3.,   1.,  20.,  10.],
                         [  3.,   2.,   2.,   1.,  10.,  60.],
                         [  1.,   1.,   1.,   6.,  nan,  nan]]), 
               [[1, 2, 3], [1, 2, 3, 4, 5, 6]])
        larm = lar1.merge(lar2, update=False)
        assert_almost_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)

    def test_merge7(self):
        "larry.merge_7"
        lar1 = larry(['a', 'b'], [[0, 1]])
        lar2 = larry(['a', 'z'], [[0, 1]])
        larr = larry(['a', 'z'], [[0, 1]])
        larm = lar1.merge(lar2, update=True)
        assert_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)

    def test_merge8(self):
        "larry.merge_8"
        d = datetime.date
        lar1 = larry([d(2010,2,3), d(2010,2,4)], [[0, 1]])
        lar2 = larry([d(2010,2,3), d(2010,2,5)], [[0, 1]])
        larr = larry([d(2010,2,3), d(2010,2,5)], [[0, 1]])
        larm = lar1.merge(lar2, update=True)
        assert_equal(larr.x, larm.x)
        assert_(larr.label == larm.label)

    def test_merge9(self):
        "larry.merge_9"
        lar1 = larry(np.array(['a'], dtype=object))
        larr = lar1.merge(lar1, update=True)
        assert_equal(larr.x, lar1.x)
        assert_(larr.label == lar1.label)
        
class Test_vacuum(unittest.TestCase):
    "Test vacuum functions of the larry class"

    def setUp(self):
        x1 = np.array([[ 2.0, nan, 3.0, 1.0],
                       [ 3.0, nan, 2.0, 1.0],
                       [ 1.0, nan, 1.0, 1.0]])

        labels = [[0, 1, 2], ['A', 'B', 'C', 'D']]
        labels1d = [['A', 'B', 'C', 'D']]
        labels3dt = [['A', 'B', 'C', 'D'], [0, 1, 2], [0, 1]]
        labels3d = [[0, 1], [0, 1, 2], ['A', 'B', 'C', 'D']]
        self.la1_1d = larry(x1[0], labels1d)
        self.la1_2d0 = larry(x1, labels)
        la1_2d1 = larry(x1, labels)
        x1_3d = np.rollaxis(np.dstack([x1,2*x1]),2)
        self.la1_3d = larry(x1_3d, labels3d)
        la1_3dt = larry(self.la1_3d.x.T.copy(), labels3dt)
                        
    def test_vacuum1a(self):
        "larry.vacuum_1a"
        larr = larry(
               np.array([[ 2.,  3.,  1.],
                         [ 3.,  2.,  1.],
                         [ 1.,  1.,  1.]]), 
               [[0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_2d0.vacuum(axis=None)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)

    def test_vacuum1b(self):
        "larry.vacuum_1b"    
        larr = larry(
               np.array([[ 2.,  3.,  1.],
                         [ 3.,  2.,  1.],
                         [ 1.,  1.,  1.]]), 
               [[0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_2d0.vacuum(axis=0)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)

    def test_vacuum1c(self):
        "larry.vacuum_1c"    
        #no vacuum of rows
        larr = self.la1_2d0
        larv = self.la1_2d0.vacuum(axis=1)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)    
    
    def test_vacuum2(self):
        "larry.vacuum_2"    
        larr = larry(
               np.array([[[ 2.,  3.,  1.],
                          [ 3.,  2.,  1.],
                          [ 1.,  1.,  1.]],
                
                         [[ 4.,  6.,  2.],
                          [ 6.,  4.,  2.],
                          [ 2.,  2.,  2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=None)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum3(self):
        "larry.vacuum_3"    
        larr = larry(
               np.array([ 2.,  3.,  1.]), 
               [['A', 'C', 'D']])
        larv = self.la1_1d.vacuum(axis=None)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum4(self):
        "larry.vacuum_4" 
        larr = larry(
               np.array([[[ 2.,  3.,  1.],
                          [ 3.,  2.,  1.],
                          [ 1.,  1.,  1.]],
                
                         [[ 4.,  6.,  2.],
                          [ 6.,  4.,  2.],
                          [ 2.,  2.,  2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=1)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum5(self):
        "larry.vacuum_5" 
        larr = larry(
               np.array([[[  2.,  nan,   3.,   1.],
                          [  3.,  nan,   2.,   1.],
                          [  1.,  nan,   1.,   1.]],
                
                         [[  4.,  nan,   6.,   2.],
                          [  6.,  nan,   4.,   2.],
                          [  2.,  nan,   2.,   2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'B', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=(1,2))
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum6(self):
        "larry.vacuum_6" 
        larr = larry(
               np.array([[[  2.,  nan,   3.,   1.],
                          [  3.,  nan,   2.,   1.],
                          [  1.,  nan,   1.,   1.]],
                
                         [[  4.,  nan,   6.,   2.],
                          [  6.,  nan,   4.,   2.],
                          [  2.,  nan,   2.,   2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'B', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=2)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum7(self):
        "larry.vacuum_7" 
        larr = larry(
               np.array([[[ 2.,  3.,  1.],
                          [ 3.,  2.,  1.],
                          [ 1.,  1.,  1.]],
                
                         [[ 4.,  6.,  2.],
                          [ 6.,  4.,  2.],
                          [ 2.,  2.,  2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=0)
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
        
    def test_vacuum8(self):
        "larry.vacuum_8" 
        larr = larry(
               np.array([[[ 2.,  3.,  1.],
                          [ 3.,  2.,  1.],
                          [ 1.,  1.,  1.]],
            
                         [[ 4.,  6.,  2.],
                          [ 6.,  4.,  2.],
                          [ 2.,  2.,  2.]]]), 
               [[0, 1], [0, 1, 2], ['A', 'C', 'D']])
        larv = self.la1_3d.vacuum(axis=(0,1))
        assert_almost_equal(larv.x, larr.x)
        assert_(larv.label == larr.label)
 
           
def suite():
    s = []
    u = unittest.TestLoader().loadTestsFromTestCase
    s.append(u(Test_init))
    s.append(u(Test_unary))
    s.append(u(Test_binary))
    s.append(u(Test_reduce))
    s.append(u(Test_comparison)) 
    s.append(u(Test_anyall))
    s.append(u(Test_getset))
    s.append(u(Test_label))
    s.append(u(Test_calc))
    s.append(u(Test_alignment))
    s.append(u(Test_properties_01))      
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()   
     
