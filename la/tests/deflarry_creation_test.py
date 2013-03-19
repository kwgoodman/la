import os
import tempfile
from nose.tools import assert_raises
import numpy as np
import la
from la import larry
from la.util.testing import assert_larry_equal as ale


def larry_init_list_test():
    "larry.__init__list"
    desired = larry([[0, 1], [2, 3]])
    actual = la.lrange((2, 2))
    ale(actual, desired, msg='create larry from list')

def larry_init_tuple_test():
    "larry.__init__tuple"
    desired = larry(((0, 1), (2, 3)))
    actual = la.lrange((2, 2))
    ale(actual, desired, msg='create larry from tuple')

def larry_init_matrix_test():
    "larry.__init__matrix"
    desired = larry(np.matrix([[0, 1], [2, 3]]))
    actual = la.lrange((2, 2))
    ale(actual, desired, msg='create larry from matrix')

def larry_init_array_test():
    "larry.__init__array"
    desired = larry(np.array([[0, 1], [2, 3]]))
    actual = la.lrange((2, 2))
    ale(actual, desired, msg='create larry from matrix')

def larry_init_raises_test():
    "larry.__init__raises"
    def make_bad_larry(x, label=None, dtype=None):
        return larry(x, label, dtype)
    assert_raises(ValueError, make_bad_larry, [1], [['a']], 1)
    assert_raises(ValueError, make_bad_larry, [[1, 2]], [['a', 'b']])
    assert_raises(ValueError, make_bad_larry, 0)
    assert_raises(ValueError, make_bad_larry, [1, 2], [['a']])
    assert_raises(ValueError, make_bad_larry, [1, 2], [['a', 'a']])

# --------------------------------------------------------------------------

# larry dtype test
#
# Make sure the optional dtype input works as expected

def test_dtype():
    "larry dtype test"
    dtypes = [float, int, str, bool, complex, object, None]
    data = [0, 1, 2]
    msg = 'larry creation failed with dtype %s using a %s as input'
    for dtype in dtypes:
        lar1 = larry(list(data), dtype=dtype)           # larry does dtype
        lar2 = larry(np.array(list(data), dtype=dtype)) # array does dtype
        yield ale, lar1, lar2, msg % (dtype, 'list')
        if dtype != str:
            # Bug in numpy 1.7.0 makes the following block fail on dtype=str
            # See https://github.com/numpy/numpy/issues/3159
            lar1 = larry(np.array(list(data)), dtype=dtype) # larry does dtype
            lar2 = larry(np.array(list(data), dtype=dtype)) # array does dtype
            yield ale, lar1, lar2, msg % (dtype, 'array')

# --------------------------------------------------------------------------

# Conversion tests
#
# The larry conversion methods are:
#
#             fromtuples, totuples
#             fromlist,   tolist
#             fromdict,   todict 
#
# Make sure that larrys don't change after a round trip:

def test_conversion():
    "Make sure that larrys don't change after a conversion round trip."
    shapes = [(1,), (1,1), (3,), (3,1), (1,1,1), (1,1,2), (1,2,2), (2,2,2),
              (3,2,1), (5,4,3,2,1), (0,)]
    msg = 'Round trip %s conversion failed on shape %s'          
    for shape in shapes:
        y1 = larry(np.arange(np.prod(shape)).reshape(shape))
        y2 = larry.fromtuples(y1.copy().totuples())
        yield ale, y1, y2, msg % ('tuples', str(shape)), False
        y2 = larry.fromlist(y1.copy().tolist())
        yield ale, y1, y2, msg % ('list', str(shape)), False        
        y2 = larry.fromdict(y1.copy().todict())
        yield ale, y1, y2, msg % ('dict', str(shape)), False
        suffix = '.csv'
        prefix = 'la_csv_unittest'
        filename = tempfile.mktemp(suffix=suffix, prefix=prefix)
        y1.copy().tocsv(filename)
        y2 = larry.fromcsv(filename)
        y2 = y2.maplabel(int) # labels loaded as strings; convert to int
        os.unlink(filename)
        yield ale, y1, y2, msg % ('csv', str(shape)), False                
