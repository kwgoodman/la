"util.misc unit tests."

import unittest

from la.util.misc import randstring


class Test_misc(unittest.TestCase):
    "Test util.misc."
        
    def test_randstring_1(self):
        "util.misc.randstring_1"
        rs = randstring(4)
        self.assert_(len(rs) == 4, 'Wrong length string.')

        
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
        
    
          
