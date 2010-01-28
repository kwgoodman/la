"la init"

# Classes
from la.deflarry import larry

try:
    from la.io import (IO, save, load, repack, is_archived_larry,
                       archive_directory)
except ImportError:
    print "Cannot import h5py; no archiving available."        

from numpy import nan, inf

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No la unit testing available."
