"larry string indexing; uses nose"

import numpy as np
from numpy.testing import assert_equal as ass

from la import larry
from la.util.testing import assert_larry_equal as ale


def la1():
    "Return a 1d larry with integer label elements"
    return larry(np.arange(24))

def la3():
    "Return a 3d larry in integer label elements"
    return larry(np.arange(24).reshape(2,3,4))
    
def la4():
    "Return a 3d larry with string label elements"
    y = la3()
    return y.maplabel(str, None)    
                  
def test_string_indexing():          
    "larry string indexing tests using nose"
    msg = 'Indexing with string failed'
    yield ass, la1()['0']           , la1()[0]                 , msg    
    yield ale, la1()['0':]          , la1()[0:]                , msg  
    yield ale, la3()['0']           , la3()[0]                 , msg
    yield ale, la3()[0,'0']         , la3()[0,0]               , msg
    yield ale, la3()[1:,'0']        , la3()[1:,0]              , msg
    yield ale, la3()['1':,'0']      , la3()[1:,0]              , msg
    yield ale, la3()['1']['0']      , la3()[1][0]              , msg    
    yield ale, la3()[1]['0']        , la3()[1][0]              , msg     
    yield ale, la3()['0':]          , la3()[0:]                , msg
    yield ale, la4()['0']           , la4()[0]                 , msg
    yield ale, la4()[0,'0']         , la4()[0,0]               , msg
    yield ale, la4()[1:,'0']        , la4()[1:,0]              , msg
    yield ale, la4()['1':,'0']      , la4()[1:,0]              , msg
    yield ale, la4()['1']['0']      , la4()[1][0]              , msg    
    yield ale, la4()[1]['0']        , la4()[1][0]              , msg     
    yield ale, la4()['0':]          , la4()[0:]                , msg     
