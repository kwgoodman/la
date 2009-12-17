"""Notes on slicing
Created on Thu Dec 17 09:50:31 2009

Author: josef-pktd
"""

''' indexing with nd boolean array fails
>>> lar[lar>0]
Traceback (most recent call last):
  File "c:\josef\eclipsegworkspace\larry\trunk\la\deflarry.py", line 722, in __getitem__
    raise IndexError, msg
IndexError: Only slice, integer, and seq (list, tuple, 1d array) indexing supported
lar.x[lar.x>0]  # this works but creates 1d array, only useful for changes
'''

'''indexing with 1d boolean array fails
>>> la1_3d[lar.x[:,0,0]>0,:,:]
Traceback (most recent call last):
  File "c:\josef\eclipsegworkspace\larry\trunk\la\deflarry.py", line 725, in __getitem__
    return type(self)(x, label)
  File "c:\josef\eclipsegworkspace\larry\trunk\la\deflarry.py", line 60, in __init__
    raise ValueError, msg2 % (i, value, key)
ValueError: Elements of label not unique along dimension 0. There are 2 labels named `1`.
>>> lar.x[lar.x[:,0,0]>0,:,:] # this works
'''
import numpy as np
import la

x1 = np.array([[ 2.0, 2.0, 3.0, 1.0],
                [ 3.0, 2.0, 2.0, 1.0],
                [ 1.0, 1.0, 1.0, 1.0]])

# slice tests for 3d

la1_2d0 = la.larry(x1)
la1_2d1 = la.larry(x1)
x1_3d = np.rollaxis(np.dstack([x1,2*x1]),2)
la1_3d = la.larry(x1_3d)
n0,n1,n2 = la1_3d.shape
idx = np.mgrid[0:n0,0:n1,0:n2]

def getnplabel3d(idx, slices, reduced=True):
    ndim = idx[0].ndim
    labs = [np.unique(idx[ii][slices]).tolist() for ii in range(ndim)]
    if reduced:
        labs = [ii for ii in labs if len(ii) > 1]
    return labs
    
    

slices = (slice(None), slice(None), slice(None,2))
larli = [la1_3d, la1_2d0]
sliceli = [(slice(None), slice(None), slice(None,2)),
           (slice(None), slice(None), slice(2)),
           (slice(None), slice(2))]

for lar in larli:
    for ss in sliceli:
        ndim = lar.ndim
        # create arrays corresponding to labels for slicing
        idxslice = [slice(0,nn) for nn in lar.shape]
        idx = np.mgrid[idxslice]
        slices = slices[:ndim]  # to run same loop on 2d and 3d
        lasliced = lar[slices]
        print np.all(lasliced.x == lar.x[slices])
        print lasliced.label == getnplabel3d(idx, slices)




