"flabel (list of lists) unit tests."
 
import numpy as np
from numpy.testing import assert_equal

from la.flabel import listmap          

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
