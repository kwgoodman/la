"Unit tests of larry functions."

import unittest

import numpy as np
nan = np.nan

from la import larry
from la.func import union, intersection, panel, stack, cov
from la.util.testing import assert_larry_equal as ale


class Test_func(unittest.TestCase):
    "Test larry functions in func"                         
        
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
    
        
def suite():
    s = []
    u = unittest.TestLoader().loadTestsFromTestCase
    s.append(u(Test_func))   
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()   
             
