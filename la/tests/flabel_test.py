"flabel (list of lists) unit tests."
 
import numpy as np
from numpy.testing import assert_equal

from la.flabel import listmap, listmap_fill         

# ---------------------------------------------------------------------------

# listmap
#
# test to make sure listmap returns the same output as
#
#                     idx = map(list1.index, list2)        

def listmap_test():
    "listmap test"
    list1 = range(6)
    list2 = range(5)
    msg = "listmap failed on list1=%s and list2=%s and ignore_unmappable=%s"
    for i in range(100):
        np.random.shuffle(list2)
        idx1 = map(list1.index, list2)
        idx2 = listmap(list1, list2)
        ignore_unmappable = False
        yield assert_equal, idx1, idx2, msg % (list1, list2, ignore_unmappable)
        ignore_unmappable = True
        yield assert_equal, idx1, idx2, msg % (list1, list2, ignore_unmappable)              

def listmap_unmappable_test():
    "listmap unmappable test"
    msg = "listmap failed on list1=%s and list2=%s and ignore_unmappable=%s"
    for i in range(100):
        list1 = range(6)
        list2 = range(5)
        np.random.shuffle(list2)
        idx1 = map(list1.index, list2)
        list2 = ['unmappable #1'] + list2 + ['unmappable #2']
        ignore_unmappable = True
        idx2 = listmap(list1, list2, ignore_unmappable=ignore_unmappable)
        yield assert_equal, idx1, idx2, msg % (list1, list2, ignore_unmappable) 

# ---------------------------------------------------------------------------

# listmap_fill unit tests    

def listmap_fill_test():
    "listmap_fill test"
    # test to make sure listmap_nofill returns the same output as
    #
    #                     idx = map(list1.index, list2)
    #
    # when there are no items in list2 that are not in list1 
    list1 = range(6)
    list2 = range(5)
    msg = "listmap_fill failed on list1=%s and list2=%s"
    for i in range(100):
        np.random.shuffle(list2)
        idx1 = map(list1.index, list2)
        idx2, ignore = listmap_fill(list1, list2)
        yield assert_equal, idx1, idx2, msg % (list1, list2)            
        
def listmap_fill_unmappable_test():
    "listmap_fill unmappable test"
    list1 = ['a', 2, 3]
    list2 = ['a', 2, 3, 4]
    idx, idx_unmappable = listmap_fill(list1, list2)
    idx2 = [0, 1, 2, 0]
    idx2_unmappable = [3]
    msg = "listmap_fill failed on list1=%s and list2=%s"
    yield assert_equal, idx, idx2, msg % (list1, list2)
    yield assert_equal, idx_unmappable, idx2_unmappable, msg % (list1, list2)
