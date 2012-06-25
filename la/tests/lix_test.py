"Unit tests of larry's indexing by label method, lix"

from nose.tools import assert_raises

import numpy as np
from numpy.testing import assert_equal

import la
from la import larry
from la.util.misc import isscalar
from la.util.testing import assert_larry_equal as ale

S = slice
N = None

si = [
      ((2,3,4)  , ([0],           [0],          [0])),

      ((3,4)    , ([0],           [0])),
      ((3,)     , ([0],)),
            
      ((2,3,4)  , ([0, 1],        [0, 1, 2],    [0, 1, 2, 3])),
      ((3,4)    , ([0, 1, 2],     [0, 1, 2, 3])),
      ((3,)     , ([0, 1, 2],)),
            
      ((2,3,4),   ([0],           [0],          S(N))),
      ((2,3,4),   ([0],           S(N),         S(N))),
      ((2,3,4),   (S(N),          S(N),         S(N))),

      ((2,3,4),   ([0],           [0],          S([1],N))),
      ((2,3,4),   ([0],           S([1],N),     S(N))),      
      ((2,3,4),   (S([1],N),      S(N),         S(N))),

      ((2,3,4),   ([0],           [0],          S(N,[1]))),
      ((2,3,4),   ([0],           S(N,[1]),     S(N))),      
      ((2,3,4),   (S(N,[1]),      S(N),         S(N))),

      ((2,3,4),   ([0],           [0],          S(N,[1],2))),
      ((2,3,4),   ([0],           S(N,[1],2),   S(N))),      
      ((2,3,4),   (S(N,[1],2),    S(N),         S(N))),

      ((2,3,4),   ([0,1],         [0,2],        S(N,[1]))),
      ((2,3,4),   ([1,0],         S(N,[1]),     S(N))),      
      ((2,3,4),   (S(N,[1]),      [0],          S(N))),
      
      ((2,3,4),   (S(N),)),      
      ((5,5,5,5), ([0, 1],        [0, 1, 2],    [0, 1, 2, 3])),            
      ((2,3,4),   (S(N),          S(N),         [1])),
      
      # Repeat the tests above but change some of the labels to integers
      # to test integer support
      
      ((2,3,4)  , (0,             0,            0)),

      ((3,4)    , ([0],           0)),
      ((3,)     , (0,)),
            
      ((2,3,4),   (0,             0,            S(N))),
      ((2,3,4),   (0,             S(N),         S(N))),

      ((2,3,4),   ([0],           [0],          S(1,N))),
      ((2,3,4),   (0,             S(1,N),       S(N))),      
      ((2,3,4),   (S(1,N),        S(N),         S(N))),

      ((2,3,4),   (0,             [0],          S(N,[1]))),
      ((2,3,4),   (0,             S(N,1),       S(N))),      
      ((2,3,4),   (S(0,[1]),      S(N),         S(N))),

      ((2,3,4),   (0,             [0],          S(N,1,2))),
      ((2,3,4),   (0,             S(N,1,2),     S(N))),      
      ((2,3,4),   (S(N,[1],2),    S(N),         S(N))),

      ((2,3,4),   ([0,1],         [0,2],        S(N,1))),
      ((2,3,4),   ([1,0],         S(N,1),       S(N))),      
      ((2,3,4),   (S(N,1),        0,            S(N))),
                  
      ((2,3,4),   (S(N),          S(N),         1)),            
     ]

def test_lix(si=si):
    "Unit tests of larry's indexing by label method, lix"           
    for shape, index in si:
        original = larry(np.arange(np.prod(shape)).reshape(shape))
        aindex = []
        for ax, idx in enumerate(index):
            if type(idx) == slice:
                start = idx.start
                if type(start) == list:
                    start = start[0]
                stop = idx.stop
                if type(stop) == list:
                    stop = stop[0] 
                idx = slice(start, stop, idx.step)                       
                aindex.append(np.arange(*idx.indices(original.shape[ax])))
            elif isscalar(idx):
                aindex.append([idx])
            else:
                aindex.append([np.array(i) for i in idx])
        desired = np.squeeze(original.x[np.ix_(*aindex)])
        if len(index) == 1:
            actual = original.lix[index[0]]
        elif len(index) == 2:
            actual = original.lix[index[0], index[1]]       
        elif len(index) == 3:
            actual = original.lix[index[0], index[1], index[2]]
        elif len(index) == 4:
            actual = original.lix[index[0], index[1], index[2], index[3]]
        if isinstance(actual, larry):
            actual = actual.x
        msg = '\nlix fail on shape %s and index %s\n'           
        yield assert_equal, actual, desired, msg  % (str(shape), str(index))                           
        
# ---------------------------------------------------------------------------
# Test lix setitem

def lix_setitem_test_01():
    actual = la.lrange(label=[['a', 'b']])
    actual.lix[['a']] = 9
    desired = la.lrange(label=[['a', 'b']])
    desired[0] = 9
    ale(actual, desired)

def lix_setitem_test_02():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[['a']] = 9
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[0] = 9
    ale(actual, desired)

def lix_setitem_test_03():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[0] = 9
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[0] = 9
    ale(actual, desired)

def lix_setitem_test_04():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[['a']:['b']] = 9
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[0:1] = 9
    ale(actual, desired)

def lix_setitem_test_05():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[['b'], ['d']] = 9
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[1, 1] = 9
    ale(actual, desired)

def lix_setitem_test_06():
    actual = la.lrange(label=[['a', 'b', 'c'], ['1', '2', '3']])
    actual.lix[['a', 'b'], ['1', '2']] = 9
    desired = la.lrange(label=[['a', 'b', 'c'], ['1', '2', '3']])
    desired[:2, :2] = 9
    ale(actual, desired)

def lix_setitem_test_07():
    actual = la.lrange(label=[['a', 'b', 'c'], ['1', '2', '3']])
    actual.lix[['a', 'b'], ['1', '2']] = np.arange(4).reshape(2, 2)
    desired = la.lrange(label=[['a', 'b', 'c'], ['1', '2', '3']])
    desired[:2, :2] = np.arange(4).reshape(2, 2)
    ale(actual, desired)

def lix_setitem_test_08():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[['a']] = la.larry([9, 10], [['c', 'd']])
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[0] = [9, 10]
    ale(actual, desired)

def lix_setitem_test_09():
    def lixit(lar, index, value):
        lar.lix[index] = value
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    index = ['a']
    value = la.larry([10, 9], [['d', 'c']])
    assert_raises(IndexError, lixit, actual, index, value)

def lix_setitem_test_10():
    actual = la.lrange(label=[['a', 'b'], ['c', 'd']])
    actual.lix[:,['d']] = la.larry([[9], [10]], [['a', 'b'],['d']])
    desired = la.lrange(label=[['a', 'b'], ['c', 'd']])
    desired[:,-1] = [9, 10]
    ale(actual, desired)
