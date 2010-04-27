
import numpy as np
from numpy.testing import assert_equal, assert_raises

from la import larry
from la.util.testing import assert_larry_equal

nan = np.nan


class Test_Testing_Assert_Larry(object):
    # Tests for assert_larry_equal
    # No yields because of double list in assert_raises
    
    def setup(self):
        self.y1 = larry([2.0,2.0,3.0], [['a', 'b', 'c']])
        self.y1dt = larry([2,2,3], [['a', 'b', 'c']])
        self.x = larry([[nan, 2.0],
                        [2.0, 3.0],
                        [3.0, 0.0]],
                       [['a', 'b', 'c'], [1, 2]])
        self.x2 = larry([[nan, 2.0],
                        [2.0, 3.0],
                        [3.0, 0.0]],
                       [['a', 'b', 'd'], [1, 2]])
        self.y = larry([[nan, 2.0],
                        [1.0, 3.0],
                        [3.0, 1.0]],
                       [['a', 'b', 'c'], [1, 2]])
        self.yc = self.y.copy()
        self.ycl = self.y.copy()
        self.ycl.label[0] = self.y.label[0]
    
    def test_assert_larry_identical(self):
        y,x,yc = self.y, self.x, self.yc
        assert_larry_equal(y, y, 'identity')

    def test_assert_larry_xr(self):
        y,x = self.y, self.x
        assert_raises(AssertionError, assert_larry_equal, 
                      y, x, 'different x')

    def test_assert_larry_labelr(self):
        y,x = self.y, self.x2
        assert_raises(AssertionError, assert_larry_equal, 
                      y, x, 'different labels')
    
    def test_assert_larry_norefr(self):
        y,x,yc = self.y, self.x, self.yc
        assert_larry_equal(y, y, 'identity', original=yc, iscopy=True)
    
        
    def test_assert_larry_norefr(self):
        y,x,yc = self.y, self.x, self.yc
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y, 'raise noreference in c', original=y, iscopy=True)

    def test_assert_larry_noreflr(self):
        y, yc = self.y, self.ycl
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y, 'raise noreference in labels', original=yc,
                      iscopy=False)
        
    def test_assert_larry_nocopy(self):
        y, yc = self.y, self.yc
        assert_larry_equal(y, y, 'nocopy', original=y, iscopy=False)
        
    def test_assert_larry_nocopyr(self):
        y, yc = self.y, self.yc
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y, 'raise nocopy', original=yc,
                      iscopy=False)

    def test_assert_larry_nocopylr(self):
        y, yc = self.y, self.ycl
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y, 'raise nocopy labels', original=yc,
                      iscopy=False)

    def test_assert_larry_shaper(self):
        y,y1 = self.y, self.y1
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y1, 'different shape')

    def test_assert_larry_dtyper(self):
        y,y1 = self.y1, self.y1dt
        assert_raises(AssertionError, assert_larry_equal, 
                      y, y1, 'different dtype')

    def test_assert_larry_dtype(self):
        y,y1 = self.y1, self.y1dt
        assert_larry_equal(y, y1, 'different dtype', dtype=False)
            


