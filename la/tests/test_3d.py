'''testing afunc functions for 3d arrays'''

import numpy as np
from numpy.testing import assert_, assert_almost_equal
nan = np.nan

#from la.util.testing import printfail
from la.util.testing import assert_larry_equal
from la.util.scipy import nanstd, nanmean
from la import afunc, larry

def getfuncs(argint, argfrac, argsector):
    funcs = [#(('covMissing'            , ()),  #2d only
             ('geometric_mean'        , (), ()),
             ('lastrank'              , (), ()),
             #('nanstd'                , ()), #in scipy
             #('nanmean'               , ()), #in scipy
             ('ranking'               , (), ()),
             ('nanmedian'             , (), ()),
             ('movingrank'            , (argint,), ()),
             ('movingsum'             , (argint,), ()),
             ('movingsum_forward'     , (argint,), ()),
             ('quantile'              , (argint,), ()),
             ('fillforward_partially' , (argint,), ()),
             ('lastrank_decay'        , (argfrac,), ()),
             ('group_mean'            , (argsector,), ()),#(0,0)),
             ('group_median'          , (argsector,), (0,0)),
             ('group_ranking'         , (argsector,), (0,0))]
    return funcs
         
def check_3d(func, args):
    res = func(*args)
    if type(res) is tuple:
        res1 = res[0]
    else:
        res1 = res
    assert_(np.shape(res1)>0, repr(func)+'does not return array for 3d')

def check_3d(func, args):
    res3d = func(*args)
    res2d = func(*args)


def test_3d():
    # many of these tests fail, skip to reduce noise during testing
    x2d = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                  [1.0, 1.0, 1.0, nan, nan, nan],
                  [2.0, 2.0, 0.1, nan, 1.0, nan],  # 0.0 kills geometric mean
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
    for funcname, funcargs, axisargs in funcs:#[:1]:
        func = getattr(afunc, funcname)
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
        res2d = func(*args2d, **kwds2d)
        args3d = (xc,) + funcargs 
        res3d = func(*args3d, **kwds3d)
        
        msg = funcname + str(funcargs)
        yield assert_almost_equal, res3d[resind], res2d, 14, msg
        #print funcname, 'is ok'
        
        #test the corresponding larry methods
        if funcname in ['geometric_mean', 'fillforward_partially']:
                        #'lastrank', 
                        #'ranking','nanmedian',
                        #'movingrank', 'lastrank_decay']:
            continue
        if funcname == 'nanmedian': funcname = 'median'
        if 'group' in funcname:
            funcargs = (lasectors,)
            
        #print funcname, 'here', kwds3d, resind
        meth2d = getattr(lar2dc, funcname)
        meth3d = getattr(lar3dc, funcname)
        msg = "method '" + funcname + "' " + repr((kwds3d, resind))
        yield assert_larry_equal, meth2d(*funcargs, **kwds2d), \
                           meth3d(*funcargs, **kwds3d)[resind], msg
        #print funcname, '(larry) is ok'
        












