
import datetime
import unittest

import numpy as np
nan = np.nan

from la import larry


def noreference(larry1, larry2):
    "Return True if there are no shared references"
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is not larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is not larry2.label[i])
    return out    

def nocopy(larry1, larry2):
    "Return True if there are only references"
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is larry2.label[i])
    return out
    
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

    def test_log_1(self):
        "larry.log_1"
        t = np.array([[ 0.0, 0.0],
                      [ 0.0, 0.0],
                      [ 0.0, 0.0]])
        p = self.l.log()
        msg = printfail(t, p.x, 'x')
        t[np.isnan(t)] = self.nancode
        p[np.isnan(p.x)] = self.nancode        
        self.assert_((abs(t - p) < self.tol).all(), msg) 
        label = [[0,1,2], [0,1]]
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l), 'Reference found')

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
        self.failUnlessRaises(AssertionError, self.l.clip, 3, 2)
        
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

    def test_zscore_1(self):
        "larry.zscore_1"
        t = self.x1.copy()
        t -= t.mean(0)
        t /= t.std(0)                                          
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
        t -= t.mean(1).reshape(-1,1)
        t /= t.std(1).reshape(-1,1)                                         
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
        t -= t.mean()
        t /= t.std()                                          
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

    def test_fillforward_partially_3(self):
        "larry.fillforward_partially_3"
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
        label = [[0, 1, 2], [1, 2, 3]]
        p = self.l1.movingsum(2)
        msg = printfail(t, p.x, 'x')       
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')
        
    def test_movingsum_2(self):
        "larry.movingsum_2"
        t = np.array([[ 4.0, 5.0, 4.0],
                      [ 5.0, 4.0, 3.0],
                      [ 2.0, 2.0, 2.0]])
        label = [[0, 1, 2], [1, 2, 3]]
        p = self.l1.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')       
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')            

    def test_movingsum_3(self):
        "larry.movingsum_3"
        t = np.array([[ 5.0, 4.0, 5.0, 2.0],
                      [ 4.0, 3.0, 3.0, 2.0]])
        label = [[1, 2], [0, 1, 2, 3]]
        p = self.l1.movingsum(2, axis=0)
        msg = printfail(t, p.x, 'x')       
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
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')

    def test_movingsum_5(self):
        "larry.movingsum_5"
        t = np.array([[ 4.0, 2.0, 1.0],
                      [ nan, nan, 1.0],
                      [ 2.0, 1.0, 1.0]])                      
        label = [[0, 1, 2], [1, 2, 3]]
        p = self.l2.movingsum(2)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode       
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')
        
    def test_movingsum_6(self):
        "larry.movingsum_6"
        t = np.array([[ 4.0, 4.0, 2.0],
                      [ nan, nan, 2.0],
                      [ 2.0, 2.0, 2.0]]) 
        label = [[0, 1, 2], [1, 2, 3]]
        p = self.l2.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l2), 'Reference found')            

    def test_movingsum_7(self):
        "larry.movingsum_7"
        t = np.array([[ 2.0, 2.0, nan, 2.0],
                      [ 1.0, 1.0, nan, 2.0]])                                            
        label = [[1, 2], [0, 1, 2, 3]]
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
        t = np.array([ 3.0, 5.0, 7.0, 9.0])
        label = [[1, 2, 3, 4]]
        p = self.l3.movingsum(2, axis=0)
        msg = printfail(t, p.x, 'x')            
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
        
    def test_cov_1(self):
        "larry.cov_1" 
        self.l1dm = self.l1.copy()
        #demean by rows
        self.l1dm.x = self.x1 - self.x1.mean(1)[:,None]
        #TODO: what's the right answer for zero variance
        t = np.array([[ 1. ,  0.5,  nan],
                   [ 0.5,  1. ,  nan],
                   [ nan,  nan,  nan]])
        
        t = np.array([[ 0.5,   0.25,  0.  ],
                     [ 0.25,  0.5,   0.  ],
                     [ 0.,    0.,    0.  ]])
          
        label = [range(3), range(3)]                                    
        p = self.l1dm.cov()
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode               
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.l1), 'Reference found')

# Here's where I left off with my unit test review: cov                  

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
        
class Test_properties_01(unittest.TestCase):
    "Test properties larry class"
    
    def setUp(self):
        self.nancode = -9999
        self.x = np.array([[ 1.0, nan],
                           [ 1.0, 1.0],
                           [ 1.0, 1.0]])                 
        self.l = larry(self.x)                                               
        
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
        "larry.T"     
        x = self.x.copy()
        x = x.T
        t = larry(x)
        p = self.l.T
        msg = printfail(t, p, 'T')
        t.x[np.isnan(t.x)] = self.nancode
        p.x[np.isnan(p.x)] = self.nancode 
        self.assert_(t == p, msg)             
        
def printfail(theory, practice, header):
    x = []
    x.append('\n\n%s\n' % header)
    x.append('\ntheory\n')
    x.append(str(theory))
    x.append('\n')
    x.append('practice\n')
    x.append(str(practice))
    x.append('\n')    
    return ''.join(x)
           
def testsuite():
    s = []
    u  =unittest.TestLoader().loadTestsFromTestCase
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
     
