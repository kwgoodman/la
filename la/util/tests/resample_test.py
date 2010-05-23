"Unit tests of resample index iterators"

import numpy as np        
from numpy.testing import assert_equal, assert_raises, assert_

from la.util.resample import cross_validation, bootstrap
      
        
def cv_count_test():
    "cv index iterator test for proper count and uniqueness"
    msg1 = "Each element was not exactly once in test when n=%d and kfold=%d"
    msg2 = "Test elements are not unique when n=%d and kfold=%d"
    msg3 = "Train elements are not unique when n=%d and kfold=%d"
    msg4 = "Each element was not in train the expected number of times "
    msg4 += "when n=%d and kfold=%d"
    msg5 = "Same element in train and test when n=%d and kfold=%d"
    msg6 = "Min index value is not 0 when n=%d and kfold=%d"
    msg7 = "Max index value is not n-1 when n=%d and kfold=%d"
    msg8 = "test index is empty when n=%d and kfold=%d"    
    for n in range(2, 8):
        for k in range(2, n+1):
            trains = []
            tests = []
            for train, test in cross_validation(n, k):
                trains += train
                tests += test
                yield assert_, len(test) > 0, msg8 % (n, k)
                ncommon = len(set(train) & set(test))                               
                yield assert_equal, ncommon, 0, msg5 % (n, k)
                unique = set(train) | set(test)
                yield assert_equal, min(unique), 0, msg6 % (n, k) 
                yield assert_equal, max(unique), n-1, msg7 % (n, k)    
            yield assert_equal, len(tests), n, msg1 % (n, k)
            yield assert_equal, len(set(tests)), n, msg2 % (n, k)
            yield assert_equal, len(set(trains)), n, msg3 % (n, k)
            yield assert_equal, len(trains), n*k - n, msg4 % (n, k)

def cv_repeatability_test():
    "cv index iterator test for repeatability"
    msg = "%s indices were not repeatable when n=%d and kfold=%d"
    n = 5
    k = 3
    train1 = []
    test1 = []
    shuffle = np.random.RandomState([1, 2, 3]).shuffle
    for train, test in cross_validation(n, k, shuffle):
        train1 += train
        test1 += test
    train2 = []
    test2 = []
    shuffle = np.random.RandomState([1, 2, 3]).shuffle    
    for train, test in cross_validation(n, k, shuffle):
        train2 += train
        test2 += test         
    yield assert_equal, train1, train2, msg % ('train', n, k)

def boot_count_test():
    "boot index iterator test for proper count and uniqueness"
    msg1 = "Did not get number of training elements expected when "
    msg1 += "n=%d and nboot=%d"
    msg2 = "Same element in train and test when n=%d and nboot=%d"
    msg3 = "test is empty when n=%d and nboot=%d"
    msg4 = "There are not n unique elements in train and test "
    msg4 += "n=%d and nboot=%d"
    msg5 = "Min index value is not 0 when n=%d and nboot=%d"
    msg6 = "Max index value is not 0 when n=%d and nboot=%d"
    for n in range(2, 8):
        for nboot in range(2, n+1):
            for train, test in bootstrap(n, nboot):
                yield assert_equal, len(train), n, msg1 % (n, nboot) 
                ncommon = len(set(train) & set(test))                               
                yield assert_equal, ncommon, 0, msg2 % (n, nboot)
                yield assert_, len(test) > 0, msg3 % (n, nboot)
                unique = set(train) | set(test)
                yield assert_equal, len(unique), n, msg4 % (n, nboot)
                yield assert_equal, min(unique), 0, msg5 % (n, nboot)
                yield assert_equal, max(unique), n-1, msg6 % (n, nboot)

def boot_repeatability_test():
    "boot index iterator test for repeatability"   
    msg = "%s indices were not repeatable when n=%d and nboot=%d"
    n = 4
    nboot = 3
    train1 = []
    test1 = []
    randint = np.random.RandomState([1, 2, 3]).randint
    for train, test in bootstrap(n, nboot, randint):
        train1 += train
        test1 += test
    train2 = []
    test2 = []
    randint = np.random.RandomState([1, 2, 3]).randint    
    for train, test in bootstrap(n, nboot, randint):
        train2 += train
        test2 += test   
    yield assert_equal, train1, train2, msg % ('train', n, nboot)
                                      
