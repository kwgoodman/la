"la init"

# Classes
from la.deflarry import larry
del deflarry  # Remove deflarry from namespace

try:
    from numpy.testing import Tester
    test = Tester().test
except ImportError:
    print "No nose testing available"
