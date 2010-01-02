"la init"

# Classes
from la.deflarry import larry
del deflarry  # Remove deflarry from namespace

from la.io.npz import save_npz, load_npz
try:
    from la.io.hdf5 import save_hdf5, load_hdf5
except:
    # I assume you haven't installed h5py and hdf5
    pass    

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except ImportError:
    print "No nose testing available"
