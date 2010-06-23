"Unit tests of larry functions."

import unittest

import numpy as np
nan = np.nan

from la import larry
from la import union, intersection, panel, stack, cov, align
from la.util.testing import assert_larry_equal as ale


class Test_func(unittest.TestCase):
    "Test larry functions in flarry"                         
        
    def test_union_1(self):
        "func.union_1" 
        y1 = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        y2 = larry([[1, 2], [3, 4]], [['e', 'b'], ['f', 'd']])
        actual = union(0, y1, y2)
        desired = ['a', 'b', 'e']
        self.assert_(actual == desired, 'union axis=0')
        actual = union(1, y1, y2)
        desired = ['c', 'd', 'f']
        self.assert_(actual == desired, 'union axis=1') 
        
    def test_intersection_1(self):
        "func.intersection_1" 
        y1 = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        y2 = larry([[1, 2], [3, 4]], [['e', 'b'], ['f', 'd']])
        actual = intersection(0, y1, y2)
        desired = ['b']
        self.assert_(actual == desired, 'intersection axis=0')
        actual = intersection(1, y1, y2)
        desired = ['d']
        self.assert_(actual == desired, 'intersection axis=1') 
        
    def test_panel_1(self):
        "func.panel_1"         
        original = larry(np.arange(24).reshape(2,3,4))                   
        actual = panel(original)
        x = np.array([[ 0, 12],
                      [ 4, 16],
                      [ 8, 20],
                      [ 1, 13],
                      [ 5, 17],
                      [ 9, 21],
                      [ 2, 14],
                      [ 6, 18],
                      [10, 22],
                      [ 3, 15],
                      [ 7, 19],
                      [11, 23]])
        label = [[(0, 0),
                  (0, 1),
                  (0, 2),
                  (0, 3),
                  (1, 0),
                  (1, 1),
                  (1, 2),
                  (1, 3),
                  (2, 0),
                  (2, 1),
                  (2, 2),
                  (2, 3)], [0, 1]]
        desired = larry(x, label)         
        ale(actual, desired, msg='panel test #1', original=original) 
        
    def test_stack_1(self):
        "func.stack_1"           
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[5, 6], [7, 8]])
        actual = stack('union', name1=y1, othername=y2)
        x = np.array([[[ 5.,  6.],
                       [ 7.,  8.]],
                      [[ 1.,  2.],
                       [ 3.,  4.]]]) 
        label = [['othername', 'name1'], [0, 1], [0, 1]]
        desired = larry(x, label)
        ale(actual, desired, msg='stack test #1')                                           
        
    def test_cov_1(self):
        "func.cov_1" 
        original = larry([[ 2.0, 2.0, 3.0, 1.0],
                          [ 3.0, 2.0, 2.0, 1.0],
                          [ 1.0, 1.0, 1.0, 1.0]])
        original = original.demean(axis=1)      
        desired = larry([[ 0.5,   0.25,  0.  ],
                         [ 0.25,  0.5,   0.  ],
                         [ 0.,    0.,    0.  ]])                                   
        actual = cov(original)
        ale(actual, desired, msg='cov test #1', original=original)       

    def test_cov_2(self):
        "func.cov_2" 
        original = larry([[nan, 2.0, 1.0],
                          [2.0, 3.0, 1.0],
                          [4.0, 1.0, 1.0]])  
        original = original.demean(1)        
        desired = larry(np.ma.cov(np.ma.fix_invalid(original.x), bias=1).data)                                            
        actual = cov(original)
        ale(actual, desired, msg='cov test #2', original=original) 


class Test_align_1d(unittest.TestCase):
    "Test 1d alignment of larrys"   

    def test_1d1(self):
        "align 1d test #1"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2)
        d1 = larry([1, 2], dtype=float)
        d2 = larry([1, 2], dtype=float)
        msg = "align 1d #1 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)            

    def test_1d2(self):
        "align 1d test #2"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='inner')
        d1 = larry([1, 2], dtype=float)
        d2 = larry([1, 2], dtype=float)
        msg = "align 1d #2 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_1d3(self):
        "align 1d test #3"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='inner', fill=0)
        d1 = larry([1, 2])
        d2 = larry([1, 2])
        msg = "align 1d #3 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_1d4(self):
        "align 1d test #4"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='outer')
        d1 = larry([1, 2, nan], dtype=float)
        d2 = larry([1, 2, 3], dtype=float)
        msg = "align 1d #4 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)
        
    def test_1d5(self):
        "align 1d test #5"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='outer', fill=0)
        d1 = larry([1, 2, 0])
        d2 = larry([1, 2, 3])
        msg = "align 1d #5 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)        

    def test_1d6(self):
        "align 1d test #6"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='left')
        d1 = larry([1, 2], dtype=float)
        d2 = larry([1, 2], dtype=float)
        msg = "align 1d #6 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

    def test_1d7(self):
        "align 1d test #7"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join='right')
        d1 = larry([1, 2, nan], dtype=float)
        d2 = larry([1, 2, 3], dtype=float)
        msg = "align 1d #7 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)
        
    def test_1d8(self):
        "align 1d test #8"
        y1 = larry([1, 2])
        y2 = larry([1, 2, 3])
        a1, a2 = align(y1, y2, join=['right'])
        d1 = larry([1, 2, nan], dtype=float)
        d2 = larry([1, 2, 3], dtype=float)
        msg = "align 1d #8 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)        

    def test_1d9(self):
        "align 1d test #9"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2)
        d1 = larry([])
        d2 = larry([])
        msg = "align 1d #9 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)            

    def test_1d10(self):
        "align 1d test #10"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='inner')
        d1 = larry([])
        d2 = larry([])
        msg = "align 1d #10 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_1d11(self):
        "align 1d test #11"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='inner', fill=0)
        d1 = larry([], dtype=int)
        d2 = larry([], dtype=int)
        msg = "align 1d #11 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_1d12(self):
        "align 1d test #12"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='outer')
        d1 = larry([1,   2,   nan, nan], [[0, 1, 'a', 'b']], dtype=float)
        d2 = larry([nan, nan, 1,   2],   [[0, 1, 'a', 'b']], dtype=float)
        msg = "align 1d #12 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)
        
    def test_1d13(self):
        "align 1d test #13"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='outer', fill=0)
        d1 = larry([1, 2, 0, 0], [[0, 1, 'a', 'b']])
        d2 = larry([0, 0, 1, 2], [[0, 1, 'a', 'b']])
        msg = "align 1d #13 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

    def test_1d14(self):
        "align 1d test #14"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='left')
        d1 = larry([1, 2], dtype=float)
        d2 = larry([nan, nan])
        msg = "align 1d #14 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_1d15(self):
        "align 1d test #15"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join='right')
        d1 = larry([nan, nan], [['a', 'b']], dtype=float)
        d2 = larry([1,   2], [['a', 'b']], dtype=float)
        msg = "align 1d #15 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)
        
    def test_1d16(self):
        "align 1d test #16"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])
        a1, a2 = align(y1, y2, join=['left'])
        d1 = larry([1, 2], dtype=float)
        d2 = larry([nan, nan])
        msg = "align 1d #16 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)
        
    def test_1d17(self):
        "align 1d test #17"
        y1 = larry([1, 2])
        y2 = larry([1, 2], [['a', 'b']])        
        self.failUnlessRaises(TypeError, align, y1, y2, 'outer', 'default',
                                                                       False)          

    def test_1d18(self):
        "align 1d test #18"
        y1 = larry([1, 2])
        y2 = larry([1, 2])
        a1, a2 = align(y1, y2, cast=False)
        d1 = larry([1, 2])
        d2 = larry([1, 2])
        msg = "align 1d #18 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

class Test_align_2d(unittest.TestCase):
    "Test 2d alignment of larrys"   

    def test_2d1(self):
        "align 2d test #1"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2)
        d1 = larry([[1, 2], [3, 4]], dtype=float)
        d2 = larry([[1, 2], [3, 4]], dtype=float)
        msg = "align 2d #1 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)            

    def test_2d2(self):
        "align 2d test #2"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2, join='inner')
        d1 = larry([[1, 2], [3, 4]], dtype=float)
        d2 = larry([[1, 2], [3, 4]], dtype=float)
        msg = "align 2d #2 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2) 

    def test_2d3(self):
        "align 2d test #3"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2, join='inner', fill=0)
        d1 = larry([[1, 2], [3, 4]])
        d2 = larry([[1, 2], [3, 4]])
        msg = "align 2d #3 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

    def test_2d4(self):
        "align 2d test #4"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2, join='outer')
        d1 = larry([[1, 2, nan], [3, 4, nan]], dtype=float)
        d2 = larry([[1, 2, 5],   [3, 4,   6]], dtype=float)
        msg = "align 2d #4 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

    def test_2d5(self):
        "align 2d test #5"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2, join='outer', fill=0)
        d1 = larry([[1, 2, 0], [3, 4, 0]])
        d2 = larry([[1, 2, 5], [3, 4, 6]])
        msg = "align 2d #5 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)       

    def test_2d6(self):
        "align 2d test #6"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        a1, a2 = align(y1, y2, join=['inner', 'outer'], fill=0)
        d1 = larry([[1, 2, 0], [3, 4, 0]])
        d2 = larry([[1, 2, 5], [3, 4, 6]])
        msg = "align 2d #6 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)

    def test_2d7(self):
        "align 2d test #7"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2, 5], [3, 4, 6]])
        j = ['outer', 'inner', 'left']
        self.failUnlessRaises(ValueError, align, y1, y2, j, 'default', False)

    def test_2d8(self):
        "align 2d test #8"
        y1 = larry([[1, 2], [3, 4]])
        y2 = larry([[1, 2], [3, 4]])
        a1, a2 = align(y1, y2, cast=False)
        d1 = larry([[1, 2], [3, 4]])
        d2 = larry([[1, 2], [3, 4]])
        msg = "align 2d #8 fail on %s larry"
        ale(a1, d1, msg % 'left', original=y1)
        ale(a2, d2, msg % 'right', original=y2)        
               
def suite():
    s = []
    u = unittest.TestLoader().loadTestsFromTestCase
    s.append(u(Test_func)) 
    s.append(u(Test_align_1d)) 
    s.append(u(Test_align_2d))  
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()   
             
