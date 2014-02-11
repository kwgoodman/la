"io unit tests."

import unittest
import tempfile
import os
import datetime

import numpy as np
nan = np.nan
import h5py

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
        self.assertTrue('x' in io, 'key missing')
        self.assertTrue((x == io['x'][:]).all(), 'save and load difference')
        self.assertTrue(['x'] == list(io.keys()), 'keys are different')
        self.assertTrue(x.dtype == io['x'].dtype, 'dtype changed')
        del io['x']
        self.assertTrue(list(io.keys()) == [], 'key still present')

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
        self.assertTrue(fs2 < fs1, 'repack did not reduce freespace')
        self.assertTrue(sp2 < sp1, 'repack did not reduce space')

    def test_io_3(self):
        "io_keys"
        io = IO(self.filename)
        io['1'] = larry([1,2,3])
        io['2'] = larry([1,2,3])
        f = h5py.File(self.filename)
        f['3'] = [1,2,3]
        f.close()
        io['1/2/3/4'] = larry([1,2,3])
        keys = list(io.keys())
        keys.sort()
        theory = ['1', '1/2/3/4', '2']
        self.assertTrue(keys == theory, 'keys do not match')

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

    def test_io_7(self):
        "io_empty (gh #68)"
        io = IO(self.filename)
        desired = larry([])
        io['desired'] = desired
        actual = io['desired']
        if actual.size == 0:
            actual = la.larry([])
        assert_larry_equal(actual, desired)

    def test_io_values(self):
        "io_values"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        values = io.values()
        values = [v[:] for v in values]
        assert_larry_equal(values[0], x)
        assert_larry_equal(values[1], y)

    def test_io_items(self):
        "io_items"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        items = io.items()
        values = [v[1][:] for v in items]
        assert_larry_equal(values[0], x)
        assert_larry_equal(values[1], y)
        keys = [v[0] for v in items]
        self.assertTrue(keys == ['x', 'y'], 'keys do not match')

    def test_io_iterkeys(self):
        "io_iterkeys"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        itk = io.iterkeys()
        keys = [k for k in itk]
        self.assertTrue(keys == ['x', 'y'], 'keys do not match')

    def test_io_itervalues(self):
        "io_itervalues"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        itv = io.itervalues()
        values = [v[:] for v in itv]
        assert_larry_equal(values[0], x)
        assert_larry_equal(values[1], y)

    def test_io_iteritems(self):
        "io_iteritems"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        iti = io.iteritems()
        items = [i for i in iti]
        values = [v[1][:] for v in items]
        assert_larry_equal(values[0], x)
        assert_larry_equal(values[1], y)
        keys = [v[0] for v in items]
        self.assertTrue(keys == ['x', 'y'], 'keys do not match')

    def test_io_haskey(self):
        "io_haskey"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        self.assertTrue(io.has_key('x'), 'keys do not match')
        self.assertTrue(io.has_key('y'), 'keys do not match')
        self.assertTrue(~io.has_key('z'), 'keys do not match')

    def test_io_len(self):
        "io_len"
        io = IO(self.filename)
        x = larry([1,2,3])
        y = larry([7,8,9])
        io['x'] = x
        io['y'] = y
        self.assertTrue(len(io) == 2, 'number of keys is wrong')

    def test_io_load_1(self):
        "io.load_1"
        d = larry([1,2,3])
        la.save(self.filename, d, 'd')
        a = la.load(self.filename, 'd')
        assert_larry_equal(a, d)
        la.io.delete(self.filename, 'd')

    def test_io_load_2(self):
        "io.load_2"
        f = h5py.File(self.filename)
        d = larry([1,2,3])
        la.save(f, d, 'd')
        a = la.load(f, 'd')
        assert_larry_equal(a, d)
        la.io.delete(f, 'd')
        f.close()

    def test_io_lara_1(self):
        "lara indexing bug #40"
        io = IO(self.filename)
        io['a'] = la.lrange(3)
        b = io['a']
        # b[[0, 2]] raised:
        # AttributeError: 'Dataset' object has no attribute 'take'
        b[[0,2]]

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
