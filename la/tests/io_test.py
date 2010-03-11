"io unit tests."

import unittest
import tempfile
import os
import datetime

import numpy as np
nan = np.nan

from la import larry
from la import (IO, archive_directory)
from la.util.testing import assert_larry_equal


class Test_io(unittest.TestCase):
    "Test io."
    
    def setUp(self):
        suffix = '.hdf5'
        prefix = 'la_io_unittest'
        self.filename = tempfile.mktemp(suffix=suffix, prefix=prefix)
        
    def tearDown(self):
        os.unlink(self.filename)
        
    def test_io_1(self):
        "io_general"
        io = IO(self.filename)
        x = larry([1,2,3]) 
        io['x'] = x
        self.assert_('x' in io, 'key missing')
        self.assert_((x == io['x'][:]).all(), 'save and load difference')
        self.assert_(['x'] == io.keys(), 'keys are different')
        self.assert_(x.dtype == io['x'].dtype, 'dtype changed')
        del io['x']
        self.assert_(io.keys() == [], 'key still present')
        
    def test_io_2(self):
        "io_repack"
        io = IO(self.filename)                
        io['larry'] = larry(np.random.rand(100,100))
        fs1 = io.freespace
        sp1 = io.space
        del io['larry']
        io.repack()
        fs2 = io.freespace
        sp2 = io.space
        self.assert_(fs2 < fs1, 'repack did not reduce freespace')
        self.assert_(sp2 < sp1, 'repack did not reduce space')
        
    def test_io_3(self):
        "io_keys"
        io = IO(self.filename)                
        io['1'] = larry([1,2,3])
        io['2'] = larry([1,2,3])
        io.f['3'] = [1,2,3]
        io['1/2/3/4'] = larry([1,2,3])
        keys = io.keys()
        keys.sort()
        theory = ['1', '1/2/3/4', '2']
        self.assert_(keys == theory, 'keys do not match')
        
    def test_io_4(self):
        "io_dates"
        io = IO(self.filename)
        x = [1, 2]
        label = [[datetime.date(2010,3,1), datetime.date(2010,3,2)]]
        desired = larry(x, label) 
        io['desired'] = desired
        actual = io['desired']
        assert_larry_equal(actual, desired)      
     
        
def testsuite():
    s = []
    u = unittest.TestLoader().loadTestsFromTestCase
    s.append(u(Test_io))
    return unittest.TestSuite(s)

def run():   
    suite = testsuite()
    unittest.TextTestRunner(verbosity=2).run(suite)
    
if __name__ == '__main__':
    run()           
        
    
          
