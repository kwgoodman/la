"util.misc unit tests."

import unittest

import numpy as np

from la.util.misc import randstring, isint, isfloat, isscalar


class Test_misc(unittest.TestCase):
    "Test util.misc."
        
    def test_randstring_1(self):
        "util.misc.randstring_1"
        rs = randstring(4)
        self.assert_(len(rs) == 4, 'Wrong length string.')
        
class Test_isa(unittest.TestCase):

    def test_isa_1(self):
        "util.misc.isa_1"
        t = {}
        #                       int    float
        t[ 1]                = (True,  False)
        t[1.1]               = (False, True)
        t['a']               = (False, False)
        t[True]              = (False, False)
        t[False]             = (False, False)
        t[np.array(1)[()]]   = (True,  False)
        t[np.array(1.1)[()]] = (False, True)
        t[1j]                = (False, False)
        for key, value in t.iteritems():
            self.assert_(isint(key) == value[0], str(key) + ' ' + str(value[0]))
            self.assert_(isfloat(key) == value[1])
            self.assert_(isscalar(key) == (value[0] or value[1]))
                        
def suite():
    s = []
    u = unittest.TestLoader().loadTestsFromTestCase
    s.append(u(Test_misc))
    return unittest.TestSuite(s)

def run():   
    suite = suite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()           
        
    
          
