"Unit tests of larry functions."

import unittest

import numpy as np
nan = np.nan

from la import larry
from la.func import cov
from la.util.testing import assert_larry_equal as ale


class Test_cov(unittest.TestCase):
    "Test cov function"                         
        
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

    def test_cov_b(self):
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
    s.append(u(Test_cov))     
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()   
             
