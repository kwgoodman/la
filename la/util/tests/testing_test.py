'''

'''

import numpy as np
from numpy.testing import assert_equal, assert_raises

from la import larry
from la.util.testing import assert_larry_equal



class Test_Testing_Assert_Larry(object):
    #tests for assert_larry_equal
    
    def setup(self):
        self.x = larry([1,2,3])
        self.y = larry([2.0,2.0,3.0], [['a', 'b', 'c']])
        self.yc = self.y.copy()
    
    def test_assert_larry_0(self):
        #TODO: these need to be split up into separate functions
        y,x,yc = self.y, self.x, self.yc
        assert_larry_equal(y, y, 'identity', noreference=False)
        assert_raises(AssertionError, assert_larry_equal, x, y, 'failall', noreference=False)
        assert_larry_equal(y, y, 'failall', original=y, noreference=False, nocopy=True)
        assert_raises(AssertionError, assert_larry_equal, y, y, 'failall', original=y, noreference=True)
        assert_larry_equal(y, y, 'identity', original=y, noreference=False)
        assert_raises(AssertionError, assert_larry_equal, y, y+1, 'identity', original=y, noreference=False)
        assert_larry_equal(y, y, 'identity', original=y+1, noreference=False)
        assert_raises(AssertionError, assert_larry_equal, y, y, 'identity', original=y, noreference=True)
        
        assert_larry_equal(y, y, 'identity', original=yc, noreference=True) #error raises ??
        assert_raises(AssertionError, assert_larry_equal, y, y, 'identity', original=yc, noreference=False, nocopy=True) #raises ??
        assert_larry_equal(y, y, 'identity', original=yc, noreference=False, nocopy=False)



