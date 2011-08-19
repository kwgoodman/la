"io unit tests."

import unittest
import tempfile
import os
import datetime

import numpy as np
nan = np.nan

import la
from la import larry
from la import IO
from la.io import datetime2tuple, tuple2datetime, time2tuple, tuple2time
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
        io['larry'] = la.rand(100, 100)
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
        actual = io['desired'][:]
        assert_larry_equal(actual, desired)      
     
    def test_io_5(self):
        "io_datetimes"
        io = IO(self.filename)
        x = [1, 2]
        label = [[datetime.datetime(2010,3,1,13,15,59,9998),
                  datetime.datetime(2010,3,2,11,23)]]
        desired = larry(x, label) 
        io['desired'] = desired
        actual = io['desired'][:]
        assert_larry_equal(actual, desired)
        
    def test_io_6(self):
        "io_datetimes"
        io = IO(self.filename)
        x = [1, 2]
        label = [[datetime.time(13,15,59,9998),
                  datetime.time(11,23)]]
        desired = larry(x, label) 
        io['desired'] = desired
        actual = io['desired'][:]
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
        
# nose tests ----------------------------------------------------------------

def datetime_test():
    "Test datetime.datetime conversion"
    dd = datetime.datetime
    dates = [dd(1950, 12, 31),
             dd(2010,  1,  6, 12),
             dd(2010,  1,  6, 22, 34),
             dd(1945,  9,  1,  2,  5,  6),
             dd(2034,  4, 10, 15,  1, 23, 500),
             dd(2011,  8, 12, 12,  0,  0,   1),
             dd(1988, 12, 31, 23, 59, 59, 999999)]
    for date in dates:
        i = datetime2tuple(date)
        d = tuple2datetime(i)
        msg = "datetime.datetime to tuple roundtrip failed."
        np.testing.assert_equal(d, date, msg)
        
def time_test():
    "Test datetime.datetime conversion"
    dt = datetime.time
    times = [dt(1,30),
             dt(23,59,59),
             dt(0,0,0,1),
             dt(15,  1, 23, 500),
             dt(23, 59, 59, 999999)]
    for time in times:
        i = time2tuple(time)
        t = tuple2time(i)
        msg = "datetime.datetime to tuple roundtrip failed."
        np.testing.assert_equal(t, time, msg)
