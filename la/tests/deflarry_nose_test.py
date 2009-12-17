
import unittest
from copy import deepcopy
import numpy as np
nan = np.nan

from la import larry

######### copies from deflarry_test.py
def noreference(larry1, larry2):
    "Return True if there are no shared references"
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is not larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is not larry2.label[i])
    return out    

def nocopy(larry1, larry2):
    "Return True if there are only references"
    assert isinstance(larry1, larry), 'Input must be a larry'
    assert isinstance(larry2, larry), 'Input must be a larry'
    assert larry1.ndim == larry2.ndim, 'larrys must have the same dimensions'
    out = True
    out = out & (larry1.x is larry2.x)
    for i in xrange(larry1.ndim):
        out = out & (larry1.label[i] is larry2.label[i])
    return out

def printfail(theory, practice, header):
    x = []
    x.append('\n\n%s\n' % header)
    x.append('\ntheory\n')
    x.append(str(theory))
    x.append('\n')
    x.append('practice\n')
    x.append(str(practice))
    x.append('\n')    
    return ''.join(x)
##############

def dup23_(x):
    '''convert 2d to 3d array to compare axis=0 with axis=2 for reduce
       operations
       
    Notes
    -----
    similar for axis=1 and axis=2 equal results
    np.all(larry(xa).median(1).x == larry(np.rollaxis(np.dstack([xa,xa]),2)).median(2).x)
    
    Example
    -------
    >>> xa = np.arange(12).reshape(3,4)
    >>> np.sum(xa,0)
    array([12, 15, 18, 21])
    >>> np.sum(dup23(xa),2)
    array([[12, 15, 18, 21],
           [12, 15, 18, 21]])
    >>> np.all(np.sum(dup23(xa),2) == np.sum(xa,0))
    True
    >>> np.all(larry(xa).mean(0).x == larry(dup23(xa)).mean(2).x)
    True
    
    for non reducing operations another swap of axes is required
    
    >>> np.all(np.cumsum(dup23(xa),2).swapaxes(1,2) == np.cumsum(xa,0))
    True
    
    '''
    return np.swapaxes(np.dstack([x,x]),0,2)


def dup23(x):
    '''convert 2d to 3d array to compare axis=0 with axis=2 for reduce
       operations
       
    see dup23_
    
    '''
    return np.rollaxis(np.dstack([x,x]),2)

from numpy.testing import assert_

tol = 1e-8
nancode = -9999  

meth_unary1 = ['log', 'exp', 'sqrt', 'abs']
 #all reduce operations except lastrank
meth_reduce1 = ['sum', 'mean', 'var', 'std', 'max', 'min', 'median', 'any', 'all']
meth_nonan = [('cumsum')]
            #'clip']


def assert_larry(opname, la, t, lab, msgstr):
    msg = printfail(la.label, lab, 'label'+msgstr)  
    assert_(la.label == lab, msg)        
    msg = printfail(t, la.x, 'x'+msgstr)
    t[np.isnan(t)] = nancode
    la.x[np.isnan(la.x)] = nancode        
    assert_((abs(t - la.x) < tol).all(), msg)
    
    # References
    #self.assert_(noreference(p, self.l1), 'Reference found') 


x1 = np.array([[ 2.0, 2.0, 3.0, 1.0],
                [ 3.0, 2.0, 2.0, 1.0],
                [ 1.0, 1.0, 1.0, 1.0]])
la1 = larry(x1)
x3 = np.array([[ 2.0, 2.0, 3.0, 1.0],
                [ 3.0, 2.0, 2.0, nan],
                [ 1.0, nan, 1.0, 1.0]])
la3 = larry(x3)

x2 = np.array([[ 2.0, nan, 3.0, 1.0],
                [ 3.0, nan, 2.0, nan],
                [ 1.0, nan, 1.0, 1.0]])
la2 = larry(x2)

la3_3d = larry(np.dstack((x3,x1)))
la1_3d = larry(np.dstack((x1,2*x1)))

las = [(la1,'la1'), (la2,'la2'), (la1_3d, 'la1_3d'), (la3,'la3')]
lasnonan = [(la1,'la1'), (la1_3d, 'la1_3d')]
#las_valid = [la1, la1_3d]  #las without nans

def test_methods_unary():
    # simple unary elementwise operations
    for la, laname in las:
        for opname in meth_unary1:
            npop = getattr(np,opname)
            t = npop(la.x)   #+1  to see whether tests fail
            p = getattr(la, opname)()
            
            yield assert_larry, opname, p, t, la.label, laname
            yield assert_, noreference(p, la), opname + ' - noreference'
            
def test_methods_reduce():
    # simple unary elementwise operations
    for la, laname in las:
        for opname in meth_reduce1:
            if np.isnan(la.x).any(): 
                npmaop = getattr(np.ma,opname)
                npop = lambda *args: np.ma.filled(npmaop(np.ma.fix_invalid(args[0]),args[1]),np.nan)
                #print 'using ma', npop
            else:
                npop = getattr(np,opname)
                #print 'not using ma', npop
                
            for axis in range(la.x.ndim):
                t = npop(la.x, axis)   #+1  to see whether tests fail
                #print t 
                #print npop
                p = getattr(la, opname)(axis)
                tlab = deepcopy(la.label)
                tlab.pop(axis)
                
                yield assert_larry, opname, p, t, tlab, laname+' axis='+str(axis)
                #yield assert_, noreference(p, la), opname + ' - noreference'

def test_methods_nonan():
    # simple unary elementwise operations
    for la, laname in lasnonan:
        for opname in meth_nonan:
            npop = getattr(np,opname)
                
            for axis in range(la.x.ndim):
                t = npop(la.x, axis)   #+1  to see whether tests fail
                p = getattr(la, opname)(axis)
                tlab = deepcopy(la.label)
                #tlab.pop(axis)
                yield assert_larry, opname, p, t, tlab, laname+' axis='+str(axis)

class est_calc(object):
    "Test calc functions of larry class"
    # still too much boiler plate
    
    def setUp(self):
        self.assert_ = np.testing.assert_
        self.tol = 1e-8
        self.nancode = -9999
#        self.x1 = np.array([[ 2.0, 2.0, 3.0, 1.0],
#                            [ 3.0, 2.0, 2.0, 1.0],
#                            [ 1.0, 1.0, 1.0, 1.0]])
#        self.l1 = larry(self.x1)         
#        self.x2 = np.array([[ 2.0, 2.0, nan, 1.0],
#                            [ nan, nan, nan, 1.0],
#                            [ 1.0, 1.0, nan, 1.0]])
#        self.larry = larry(self.x2)
#        self.x3 = np.array([1, 2, 3, 4, 5])  
#        self.l3 = larry(self.x3)
#        self.x4 = np.array([[ nan, 1.0, 2.0, 3.0, 4.0],
#                            [ 1.0, nan, 2.0, nan, nan],
#                            [ 2.0, 2.0, nan, nan, nan],
#                            [ 3.0, 3.0, 3.0, 3.0, nan]])
#        self.l4 = larry(self.x4)       
#        self.x5 = np.array([[1.0, nan, 6.0, 0.0, 8.0],
#                            [2.0, 4.0, 8.0, 0.0,-1.0]])
#        self.l5 = larry(self.x5)                    
#        self.x6 = np.array([[  nan,  nan,  nan,  nan,  nan],
#                            [  nan,  nan,  nan,  nan,  nan]])                                                                                    
#        self.l6 = larry(self.x6)
#        self.x7 = np.array([[nan, 2.0],
#                            [1.0, 3.0],
#                            [3.0, 1.0]])  
#        self.l7 = larry(self.x7)     
#        self.x8 = np.array([[nan, 2.0, 1.0],
#                            [2.0, 3.0, 1.0],
#                            [4.0, 1.0, 1.0]])  
#        self.l8 = larry(self.x8)                                    
#    

    def check_function(self, t, label, p, orig, view=False):
        "check a method of larry - comparison helper function"
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        if not view:
            self.assert_(noreference(p, orig), 'Reference found')   
        elif view == 'nocopy' :
            self.assert_(nocopy(p, orig), 'copy instead of reference found') 
        else:   #FIXME check view for different dimensional larries
            pass  

    def test_demedian_2(self):
        "larry.demedian_2"
        t = self.tmedian
        label = self.label
        p = self.lar.demedian(1)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.lar), 'Reference found')

    def test_demedian_3(self):
        "larry.demedian_2"
        t = self.tmedian
        label = self.label3
        p = self.lar3.demedian(2)
        msg = printfail(t, p.x, 'x') 
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.lar3), 'Reference found')
    
    def test_movingsum_6(self):
        "larry.movingsum_6"
        t = self.tmovingsum 
        label = self.label
        p = self.lar.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.lar), 'Reference found')   

    def est_movingsum_6_3(self):
        "larry.movingsum_6"
        t = self.tmovingsum
        label = self.label3
        p = self.lar3.movingsum(2, norm=True)
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        self.assert_(noreference(p, self.lar3), 'Reference found')  
        
    def test_movingsum_6_3(self):
        "duplicate larry.movingsum_6"
        t = self.tmovingsum
        label = self.label3
        p = self.lar3.movingsum(2, norm=True)
        self.check_function(t, label, p, self.lar3)

    def test_ranking_(self):
        "larry.ranking"  #not in deflarry_test
        t = self.tranking
        label = self.label
        p = self.lar.ranking(1)
        self.check_function(t, label, p, self.lar)
        
    @np.testing.dec.knownfailureif(True)
    def test_ranking_3(self):
        "larry.ranking 3d" 
        t = self.tranking
        label = self.label3
        p = self.lar3.ranking(1)
        self.check_function(t, label, p, self.lar3)

    def test_lag(self):
        "larry.ranking 3d" 
        t = self.tlag
        label = self.labellag
        p = self.lar.lag(1)
        self.check_function(t, label, p, self.lar)
                
    def test_lag_3(self):
        "larry.ranking 3d" 
        t = self.tlag
        label = self.label3lag
        p = self.lar3.lag(1)
        self.check_function(t, label, p, self.lar3)

    def test_pull(self):
        "larry.ranking 3d" 
        t = self.tpull
        label = self.labelpull
        p = self.lar.pull(1,1)
        self.check_function(t, label, p, self.lar, view='skip')
                
    def test_pull_3(self):
        "larry.ranking 3d" 
        t = self.tpull
        label = self.label3pull
        p = self.lar3.pull(1,2)
        self.check_function(t, label, p, self.lar3, view='skip')

    def test_squeeze(self):
        "larry.squeeze 3d" 
        t = self.lar.x
        label = self.lar.label
        x = self.lar.x[None,:]
        lar = larry(x)
        p = larry(x).squeeze()
        self.check_function(t, label, p, lar, view='skip') #should be nocopy
   
    def test_squeeze_3(self):
        "larry.squeeze 3d" 
        t = self.lar3.x
        label = self.lar3.label
        x = self.lar3.x[None,:]
        lar = larry(x)
        p = lar.squeeze()
        self.check_function(t, label, p, lar, view='skip') #should be nocopy

    def test_morph_3(self):
        "larry.morph 3d" 
        t = self.tmorph
        label = self.labelmorph
        label[1] = [0,3,2]
        p = self.lar.morph([0,3,2],axis=1)
        self.check_function(t, label, p, self.lar, view='copy')

    def test_morph_3(self):
        "larry.morph 3d" 
        t = self.tmorph
        label = self.label3
        label[2] = [0,3,2]
        p = self.lar3.morph([0,3,2],axis=2)
        self.check_function(t, label, p, self.lar3, view='copy')


class Test_calc_la2(est_calc):
    "Test calc functions of larry class"
    #this runs all methods with `test_` of the super class as tests
    #check by introducing a failure in original data self.x2
    
    def setUp(self):
        self.assert_ = np.testing.assert_
        self.tol = 1e-8
        self.nancode = -9999
#        self.x1 = np.array([[ 2.0, 2.0, 3.0, 1.0],
#                            [ 3.0, 2.0, 2.0, 1.0],
#                            [ 1.0, 1.0, 1.0, 1.0]])
#        self.l1 = larry(self.x1)         
        self.x2 = np.array([[ 2.0, 2.0, nan, 1.0],
                            [ nan, nan, nan, 1.0],
                            [ 1.0, 1.0, nan, 1.0]])
        self.lar = larry(self.x2)
        self.lar3 = larry(dup23(self.x2))
        self.tmedian = np.array([[ 0.0, 0.0, nan,-1.0],
                          [ nan, nan, nan, 0.0],
                          [ 0.0, 0.0, nan, 0.0]])                    
        self.label = [[0, 1, 2], [0, 1, 2, 3]]
        self.label3 = [[0,1], [0, 1, 2], [0, 1, 2, 3]]
        self.labelmedian = self.label
        self.label3median = self.label3
        
        self.tmovingsum = np.array([[ nan, 4.0, 4.0, 2.0],
                                    [ nan, nan, nan, 2.0],
                                    [ nan, 2.0, 2.0, 2.0]]) 
        
        self.tranking = np.array([[ 0.5,  0.5,  nan, -1. ],
                                 [ nan,  nan,  nan,  0. ],
                                 [ 0. ,  0. ,  nan,  0. ]])
        
        self.tlag = np.array([[  2.,   2.,  nan],
                              [ nan,  nan,  nan],
                              [  1.,   1.,  nan]])
        self.labellag = [[0, 1, 2], [1, 2, 3]]
        self.label3lag = [[0,1], [0, 1, 2], [1, 2, 3]]
        
        self.tpull = np.array([  2.,  nan,   1.])
        self.labelpull = [[0, 1, 2]]
        self.label3pull = [[0,1], [0, 1, 2]]
        
        self.tmorph = np.array([[  2.,   1.,  nan],
                               [ nan,   1.,  nan],
                               [  1.,   1.,  nan]])

#junk:  trying to subclass a second time, not needed now
class eest_calc_la23(Test_calc_la2):
    "Test calc functions of larry class"
    
#    def __init__(self):
#        super(self.__class__, self).__init__()
    def setUp(self):
        super(self.__class__, self).setUp()
        #overwrite 2d by 3d larry
        self.lar = larry(dup23(self.x2))
        
        self.tmedian = np.array([[ 0.0, 0.0, nan,-1.0],
                          [ nan, nan, nan, 0.0],
                          [ 0.0, 0.0, nan, 0.0]])   
                         
        self.labelmedian = [[0,1], [0, 1, 2], [0, 1, 2, 3]]
        
        
class est_groups(object):
    "Test calc functions of larry class"
    # still too much boiler plate
    
    def setUp(self):
        self.assert_ = np.testing.assert_
        self.tol = 1e-8
        self.nancode = -9999


    def check_function(self, t, label, p, orig, view=False):
        "check a method of larry - comparison helper function"
        msg = printfail(t, p.x, 'x')  
        t[np.isnan(t)] = self.nancode
        p[p.isnan()] = self.nancode             
        self.assert_((abs(t - p.x) < self.tol).all(), msg)
        self.assert_(label == p.label, printfail(label, p.label, 'label'))
        if not view:
            self.assert_(noreference(p, orig), 'Reference found')   
        elif view == 'nocopy' :
            self.assert_(nocopy(p, orig), 'copy instead of reference found') 
        else:   #FIXME check view for different dimensional larries
            pass  

    def test_grouprank(self):
        "larry.grouprank"  #not in deflarry_test
        t = self.trank1
        label = self.label
        p = self.lar.group_ranking(self.sectors)
        self.check_function(t, label, p, self.lar)
        
    def test_groupmean(self):
        "larry.grouprank"  #not in deflarry_test
        t = self.tmean1
        label = self.label
        p = self.lar.group_mean(self.sectors)
        self.check_function(t, label, p, self.lar)

    def test_groupmedian(self):
        "larry.grouprank"  #not in deflarry_test
        t = self.tmedian1
        label = self.label
        p = self.lar.group_median(self.sectors)
        self.check_function(t, label, p, self.lar)


class Test_group(est_groups):
    "Test calc functions of larry class"
    #this runs all methods with `test_` of the super class as tests
    #check by introducing a failure in original data self.x2
    
    def setUp(self):
        self.assert_ = np.testing.assert_
        self.tol = 1e-8
        self.nancode = -9999

        self.label3 = [[0,1], [0, 1, 2], [0, 1, 2, 3]]
        self.x = np.array([[0.0, 3.0, nan, nan, 0.0, nan],
                           [1.0, 1.0, 1.0, nan, nan, nan],
                           [2.0, 2.0, 0.0, nan, 1.0, nan],
                           [3.0, 0.0, 2.0, nan, nan, nan],
                           [4.0, 4.0, 3.0, 0.0, 2.0, nan],
                           [5.0, 5.0, 4.0, 4.0, nan, nan]])
        sectors = ['a', 'b', 'a', 'b', 'a', 'c']
        #labels = [[0, 1, 2, 3, 4, 5], sectors]
        self.lar = larry(self.x)
        self.sectors = larry(np.array(sectors, dtype=object))
        self.lar3 = larry(dup23(self.x))
        
        self.label = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
        
        #test_sector_rank_1(self):
        "afunc.sector_rank #1"
        
        self.trank1 = np.array([[-1.0, 0.0,  nan, nan, -1.0, nan],
                               [-1.0, 1.0, -1.0, nan,  nan, nan],
                               [ 0.0,-1.0, -1.0, nan,  0.0, nan],
                               [ 1.0,-1.0,  1.0, nan,  nan, nan],
                               [ 1.0, 1.0,  1.0, 0.0,  1.0, nan],
                               [ 0.0, 0.0,  0.0, 0.0,  nan, nan]])
        
        self.tmean1 = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 2.0, 0.5,  1.5, nan,  nan, nan],
                               [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 2.0, 0.5,  1.5, nan,  nan, nan],
                               [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
        
        self.tmedian1 = np.array([[ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 2.0, 0.5,  1.5, nan,  nan, nan],
                               [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 2.0, 0.5,  1.5, nan,  nan, nan],
                               [ 2.0, 3.0,  1.5, 0.0,  1.0, nan],
                               [ 5.0, 5.0,  4.0, 4.0,  nan, nan]])
             