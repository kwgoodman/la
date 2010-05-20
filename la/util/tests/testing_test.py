
import numpy as np
from numpy.testing import assert_equal, assert_raises

from la import larry
from la.util.testing import assert_larry_equal as ale

nan = np.nan


class Test_Testing_Assert_Larry(object):
    "Test assert_larry_equal"
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
        y, x, yc = self.y, self.x, self.yc
        ale(y, y, 'identity')

    def test_assert_larry_xr(self):
        y, x = self.y, self.x
        assert_raises(AssertionError, ale, y, x, 'different x')

    def test_assert_larry_labelr(self):
        y, x = self.y, self.x2
        assert_raises(AssertionError, ale, y, x, 'different labels')
    
    def test_assert_larry_norefr(self):
        y, x, yc = self.y, self.x, self.yc
        ale(y, y, 'identity', original=yc, iscopy=True)
    
        
    def test_assert_larry_norefr(self):
        y, x, yc = self.y, self.x, self.yc
        assert_raises(AssertionError, ale, 
                      y, y, 'raise noreference in c', original=y, iscopy=True)

    def test_assert_larry_noreflr(self):
        y, yc = self.y, self.ycl
        assert_raises(AssertionError, ale, 
                      y, y, 'raise noreference in labels', original=yc,
                      iscopy=False)
        
    def test_assert_larry_nocopy(self):
        y, yc = self.y, self.yc
        ale(y, y, 'nocopy', original=y, iscopy=False)
        
    def test_assert_larry_nocopyr(self):
        y, yc = self.y, self.yc
        assert_raises(AssertionError, ale, 
                      y, y, 'raise nocopy', original=yc, iscopy=False)

    def test_assert_larry_nocopylr(self):
        y, yc = self.y, self.ycl
        assert_raises(AssertionError, ale, y, y, 'raise nocopy labels',
                      original=yc, iscopy=False)

    def test_assert_larry_shaper(self):
        y, y1 = self.y, self.y1
        assert_raises(AssertionError, ale, y, y1, 'different shape')

    def test_assert_larry_dtyper(self):
        y, y1 = self.y1, self.y1dt
        assert_raises(AssertionError, ale, y, y1, 'different dtype')

    def test_assert_larry_dtype(self):
        y, y1 = self.y1, self.y1dt
        ale(y, y1, 'different dtype', dtype=False)
        
    def test_input_types(self):
        ar = assert_raises
        x, y = self.x, self.y
        msg, dtype, iscopy = True, True, True
        ar(TypeError, ale, x, y, msg, dtype, x, iscopy)
        x, y = self.x, self.y
        msg, dtype, iscopy = '', '', True
        ar(TypeError, ale, x, y, msg, dtype, x, iscopy)                              
        x, y = self.x, self.y
        msg, dtype, iscopy = '', True, ''
        ar(TypeError, ale, x, y, msg, dtype, x, iscopy) 
        
    def test_assert_larry_scalar(self):
        assert_raises(AssertionError, ale, 1, 1.0, 'scalar #1')                     
        assert_raises(AssertionError, ale, 1.0, 1, 'scalar #2')
        assert_raises(AssertionError, ale, 2, nan, 'scalar #3')
        ale(1, 1, 'scalar #4')
        ale(1.0, 1.0, 'scalar #5')
        ale(nan, nan, 'scalar #6') 

    def test_assert_larry_nonlarry_type(self):
        assert_raises(AssertionError, ale, larry([1]), 1, 'nonlarry type #1')
        
    def test_more_dtypes(self):
        y1 = larry([1.0, 2.0, 3.0])
        y2 = larry(['1', '2', '3'])        
        msg = 'float, str should fail but there should be no error'
        assert_raises(AssertionError, ale, y1, y2, msg) 
        y1 = larry([ 1,   2,   3])
        y2 = larry(['1', '2', '3'])        
        msg = 'int, str should fail but there should be no error'
        assert_raises(AssertionError, ale, y1, y2, msg)         
        y1 = larry(['1', '2', '3'])
        y2 = larry(['1', '2', '3'])        
        msg = 'str, str failed'
        ale(y1, y2)                                                      
