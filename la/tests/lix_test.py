"Unit tests of larry's indexing by label method, lix"

import numpy as np
from numpy.testing import assert_equal

from la import larry
from la.util.misc import isscalar

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
        
        
        
        
        
        
        
        
        
