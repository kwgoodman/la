"la init"

# Classes
from la.deflarry import larry

try:
    from la.io import (IO, save, load, repack, is_archived_larry,
                       archive_directory)
except:
    # Cannot import h5py; no archiving available.
    pass        

from numpy import nan, inf

from la.flarry import (union, intersection, stack, panel, cov, rand, randn,
                       align, align_raw, binaryop, add, subtract, multiply,
                       divide, unique)
from la.util.report import info
from la.version import __version__
from la.util import testing

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No la unit testing available."
    
try:
    # Namespace cleaning
    del deflarry, flabel, func, io, missing, testing, util, version
except:
    pass     
