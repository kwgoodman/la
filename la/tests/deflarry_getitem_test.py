"Unit tests of larry.__getitem__"

from nose.tools import assert_raises, assert_equal
import numpy as np
from numpy.testing import assert_array_equal
nan = np.nan

from la import larry
from la.util.testing import assert_larry_equal as ale


def make_larrys():
    a1 = np.array([[ 1.0, nan],
                   [ 3.0, 4.0],
                   [ 5.0, 6.0]])                                              
    lar1 = larry(a1) 
    a2 = np.array([ 0, 1, 2, 3])                                              
    lar2 = larry(a2)
    return lar1, lar2

def test_getitem_01():
    "larry.getitem #01"
    desired = larry([3.0, 4.0])
    lar1, lar2 = make_larrys()
    actual = lar1[1]
    ale(actual, desired)

def test_getitem_02():
    "larry.getitem #02"
    desired = larry([3.0, 4.0])
    lar1, lar2 = make_larrys()
    actual = lar1[1,:]
    ale(actual, desired)

def test_getitem_03():
    "larry.getitem #03"
    desired = larry([3.0, 4.0])
    lar1, lar2 = make_larrys()
    actual = lar1[1,0:2]
    ale(actual, desired)

def test_getitem_04():
    "larry.getitem #04"
    desired = np.array([4.0])[0]
    lar1, lar2 = make_larrys()
    actual = lar1[1,1]
    ale(actual, desired)

def test_getitem05():
    "larry.getitem #05"
    label = [[1, 2], [0, 1]]
    desired = larry([[3.0, 4.0],
                     [5.0, 6.0]],
                     label)
    lar1, lar2 = make_larrys()
    idx = np.array([False, True, True])
    actual = lar1[idx,:]
    ale(actual, desired, original=lar1)

def test_getitem_06():
    "larry.getitem #06"
    desired = larry([[1.0, nan],
                     [3.0, 4.0]])
    lar1, lar2 = make_larrys()
    actual = lar1[0:2,0:2]
    ale(actual, desired)

def test_getitem_07():
    "larry.getitem #07"
    desired = larry([[ 3.0, 4.0],
                     [ 5.0, 6.0]], [[1, 2], [0, 1]])
    lar1, lar2 = make_larrys()
    actual = lar1[np.array([1, 2])]
    ale(actual, desired, original=lar1)
 
def test_getitem_08():
    "larry.getitem #08"
    lar1, lar2 = make_larrys()
    assert_raises(IndexError, lar1.__getitem__, 100)  

def test_getitem_09():
    "larry.getitem #09"
    lar1, lar2 = make_larrys()
    assert_raises(IndexError, lar1.__getitem__, 'a')

def test_getitem_10():
    "larry.getitem #10"
    desired = np.array([1])[0]
    lar1, lar2 = make_larrys()
    actual = lar2[1]
    ale(actual, desired)

def test_getitem_11():
    "larry.getitem #11"
    desired = larry([0, 1])
    lar1, lar2 = make_larrys()
    actual = lar2[:2]
    ale(actual, desired)
 
def test_getitem_12():
    "larry.getitem #12"
    label = [[0, 2, 1], [0, 1]]
    desired = larry([[ 1.0, nan],
                     [ 5.0, 6.0],
                     [ 3.0, 4.0]],
                     label) 
    lar1, lar2 = make_larrys()
    actual = lar1[[0, 2, 1]]
    #ale(actual, desired, original=lar1) fails: axis 1 label is not a copy
    ale(actual, desired)

def test_getitem_13():
    "larry.getitem #13"
    label = [[0, 2, 1], [0, 1]]
    desired = larry([[ 1.0, nan],
                     [ 5.0, 6.0],
                     [ 3.0, 4.0]],
                     label) 
    lar1, lar2 = make_larrys()
    actual = lar1[[0.99, 2.6, 1.78]]
    #ale(actual, desired, original=lar1) fails: axis 1 label is not a copy
    ale(actual, desired)

def test_getitem_14():
    "larry.getitem #14"
    label = [[1, 0], [0, 1]]
    desired = larry([[ 3.0, 4.0],
                     [ 1.0, nan]],
                     label) 
    lar1, lar2 = make_larrys()
    idx = [True, False]
    actual = lar1[idx]
    #ale(actual, desired, original=lar1) fails: axis 1 label is not a copy
    ale(actual, desired)

def test_getitem_15():
    "larry.getitem #15"
    lar1, lar2 = make_larrys()
    assert_raises(IndexError, lar1.__getitem__, [0,1,0])

def test_getitem_16():
    "larry.getitem #16"
    desired = larry([[ 1.0, nan]]) 
    lar1, lar2 = make_larrys()
    idx = np.array([True, False])
    actual = lar1[idx,:]
    ale(actual, desired, original=lar1)

def test_getitem_17():
    "larry.getitem #17"
    desired = larry([[ 1.0],
                     [ 3.0],
                     [ 5.0]])
    lar1, lar2 = make_larrys()
    idx = np.array([True, False])
    actual = lar1[:, idx]
    ale(actual, desired, original=lar1)

def test_getitem_18():
    "larry.getitem #18"
    desired = larry([3.0, 4.0])
    lar1, lar2 = make_larrys()
    actual = lar1[1.9]
    ale(actual, desired)

def test_getitem_19():
    "larry.getitem #19"
    desired = np.array([4.0])[0]
    lar1, lar2 = make_larrys()
    actual = lar1[1.1, 1.1]
    ale(actual, desired)

def test_getitem_20():
    "larry.getitem #20"
    desired = larry([[ 3.0, 4.0],
                     [ 5.0, 6.0]], [[1, 2], [0, 1]])
    lar1, lar2 = make_larrys()
    actual = lar1[np.array([1, 2]),:]
    ale(actual, desired, original=lar1)                       

def test_getitem_21():
    "larry.getitem #21"
    desired = larry([ 3.0, 5.0], [[1, 2]])
    lar1, lar2 = make_larrys()
    actual = lar1[np.array([1, 2]), 0]
    ale(actual, desired, original=lar1) 

def test_getitem_22():
    "larry.getitem #22"
    lar = larry([None, None])
    desired = None
    actual = lar[0]
    assert_equal(actual, desired, "Indexing object dtype failed.")

def test_getitem_23():
    "larry.getitem #23"
    a = np.empty(2, dtype=object)
    a[0] = np.array([1, 2, 3])
    a[1] = np.array([4, 5, 6])
    lar = larry(a)
    desired = np.array([4, 5, 6])
    actual = lar[1]
    err_msg = "Indexing 1d object dtype (array of arrays) failed."
    assert_array_equal(actual, desired, err_msg=err_msg)

def test_getitem_24():
    "larry.getitem #24"
    a = np.empty((2,1), dtype=object)
    a[0,0] = np.array([1, 2, 3])
    a[1,0] = np.array([4, 5, 6])
    lar = larry(a)
    desired = np.array([4, 5, 6])
    actual = lar[1,0]
    err_msg = "Indexing 2d object dtype (array of arrays) failed."
    assert_array_equal(actual, desired, err_msg=err_msg)

def test_getitem_25():
    "larry.getitem #25"
    desired = larry(np.ones((3,0)))
    lar1, lar2 = make_larrys()
    actual = lar1[:,1:1]
    ale(actual, desired, original=lar1) 
