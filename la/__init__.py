"la init"

# Classes
from la.deflarry import larry
del deflarry  # Remove deflarry from namespace

try:
    from la.io import (IO, save, load, repack, is_archived_larry,
                       archive_directory)
except (ImportError, ValueError):
    print "Cannot import h5py; no archiving available."        

from numpy import nan, inf

from la.func import union, intersection
from la.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No la unit testing available."
