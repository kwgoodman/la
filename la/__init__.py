"la init"

# Classes
from la.deflarry import larry
del deflarry  # Remove deflarry from namespace

from la.io.npz import save_npz, load_npz

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except ImportError:
    print "No nose testing available"
