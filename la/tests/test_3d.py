"Test ability to handle 3d arrays"

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import assert_almost_equal
nan = np.nan

from la.util.testing import assert_larry_equal
from la import farray, larry

def getfuncs(argint, argfrac, argsector):
    funcs = [('geometric_mean'        , (), ()),
             ('lastrank'              , (), ()),
             ('ranking'               , (), ()),
             ('movingrank'            , (argint,), ()),
             ('movingsum_forward'     , (argint,), ()),
             ('quantile'              , (argint,), ()),
             ('push'                  , (argint,), ()),
             ('group_mean'            , (argsector,), ()),
             ('group_median'          , (argsector,), ()),
             ('group_ranking'         , (argsector,), ())]
    return funcs

def test_3d():
    "Test ability to handle 3d arrays"
    x2d = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                    [1.0, 1.0, 1.0, nan, nan, nan],
                    [2.0, 2.0, 0.1, nan, 1.0, nan],
                    [3.0, 9.0, 2.0, nan, nan, nan],
                    [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                    [5.0, 5.0, 4.0, 4.0, nan, nan]])
    sectors = ['a', 'b', 'a', 'b', 'a', 'c']
    lasectors = larry(np.array(sectors, dtype=object))
    x3 = np.dstack((x2d,x2d))
    x = np.rollaxis(x3, 2).copy()
    lar2d = larry(x2d)
    lar3 = larry(x3)
    lar = larry(x)
    
    argint = 3
    argfrac = 0.75
    argsectors = sectors
    funcs = getfuncs(argint, argfrac, argsectors)
    for funcname, funcargs, axisargs in funcs:
        func = getattr(farray, funcname)
        xc = x.copy()
        x2dc = x2d.copy()
        lar2dc = lar2d.copy()

        if not axisargs:
            kwds2d = dict(axis=1)
            kwds3d = dict(axis=2)
            xc = x.copy()
            lar3dc = lar.copy()
            resind = 0
        elif axisargs == (0,0):
            xc = x3.copy()
            lar3dc = lar3.copy()
            kwds2d = {}
            kwds3d = {}
            resind = (slice(None), slice(None), 0)
        elif axisargs == (-1,-1):
            xc = x3.copy()
            kwds2d = {}
            kwds3d = {}           

        args2d = (x2dc,) + funcargs 
        with np.errstate(invalid='ignore', divide='ignore'):
            res2d = func(*args2d, **kwds2d)
        args3d = (xc,) + funcargs 
        with np.errstate(invalid='ignore', divide='ignore'):
            res3d = func(*args3d, **kwds3d)
        
        msg = funcname + str(funcargs)
        yield assert_almost_equal, res3d[resind], res2d, 14, msg
        
        # Test the corresponding larry methods
        if funcname == 'nanmedian': funcname = 'median'
        if funcname == 'nanmean': funcname = 'mean'
        if funcname == 'nanstd': funcname = 'std'
        if 'group' in funcname:
            funcargs = (lasectors,)
        meth2d = getattr(lar2dc, funcname)
        meth3d = getattr(lar3dc, funcname)
        msg = "method '" + funcname + "' " + repr((kwds3d, resind))
        with np.errstate(invalid='ignore', divide='ignore'):
            m2 = meth2d(*funcargs, **kwds2d)
            m3 = meth3d(*funcargs, **kwds3d)[resind] 
        yield assert_larry_equal, m2, m3, msg
