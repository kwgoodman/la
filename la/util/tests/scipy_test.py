"""
The functions in this module were copied from SciPy to avoid making la
depend on SciPy. See the la LICENSE file for the SciPy license.

In the SciPy project, these functions can be found in
scipy/stats/tests/test_stats.py

Some modifications were made.

"""

import numpy as np
from numpy.testing import *

from la.util.scipy import (nanmean, nanmedian, nanstd)


X = np.array([1,2,3,4,5,6,7,8,9],float)

class TestNanFunc(TestCase):
    def __init__(self, *args, **kw):
        TestCase.__init__(self, *args, **kw)
        self.X = X.copy()

        self.Xall = X.copy()
        self.Xall[:] = np.nan

        self.Xsome = X.copy()
        self.Xsomet = X.copy()
        self.Xsome[0] = np.nan
        self.Xsomet = self.Xsomet[1:]

    def test_nanmean_none(self):
        """Check nanmean when no values are nan."""
        m = nanmean(X)
        assert_approx_equal(m, X[4])

    def test_nanmean_some(self):
        """Check nanmean when some values only are nan."""
        m = nanmean(self.Xsome)
        assert_approx_equal(m, 5.5)

    def test_nanmean_all(self):
        """Check nanmean when all values are nan."""
        m = nanmean(self.Xall)
        assert np.isnan(m)

    def test_nanstd_none(self):
        """Check nanstd when no values are nan."""
        s = nanstd(self.X)
        assert_approx_equal(s, np.std(self.X, ddof=0))

    def test_nanstd_some(self):
        """Check nanstd when some values only are nan."""
        s = nanstd(self.Xsome)
        assert_approx_equal(s, np.std(self.Xsomet, ddof=0))

    def test_nanstd_all(self):
        """Check nanstd when all values are nan."""
        s = nanstd(self.Xall)
        assert np.isnan(s)

    def test_nanmedian_none(self):
        """Check nanmedian when no values are nan."""
        m = nanmedian(self.X)
        assert_approx_equal(m, np.median(self.X))

    def test_nanmedian_some(self):
        """Check nanmedian when some values only are nan."""
        m = nanmedian(self.Xsome)
        assert_approx_equal(m, np.median(self.Xsomet))

    def test_nanmedian_all(self):
        """Check nanmedian when all values are nan."""
        m = nanmedian(self.Xall)
        assert np.isnan(m)


if __name__ == "__main__":
    run_module_suite()
