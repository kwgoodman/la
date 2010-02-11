"Test larry methods for proper handling of empty larrys"

import numpy as np
from numpy.testing import assert_, assert_equal

from la import larry, nan
from la.util.testing import assert_larry_equal as ale

def lar():
    return larry([])

def arr():
    return np.array([])

#               Method        Parameters     Returns
mts = [('A'          ,  'property' ,    arr()),
       ('T'          ,  'property' ,    'use_numpy'),
       ('__abs__'    ,  None       ,    'use_numpy'),
       ('__add__'    ,  [lar()]    ,    lar()),
       ('__and__'    ,  [lar()]    ,    lar()),
       ('__div__'    ,  [lar()]    ,    lar()),
       ('__eq__'     ,  [lar()]    ,    lar()),
       ('__ge__'     ,  [lar()]    ,    lar()),
       ('__gt__'     ,  [lar()]    ,    lar()),
       ('__lt__'     ,  [lar()]    ,    lar()),
       ('__mul__'    ,  [lar()]    ,    lar()),
       ('__ne__'     ,  [lar()]    ,    lar()),               
       ('__neg__'    ,  None       ,    'use_numpy'),               
       ('__or__'     ,  [lar()]    ,    lar()),               
       ('__pos__'    ,  None       ,    'use_numpy'),               
       ('__pow__'    ,  [2]        ,    lar()),
       ('__radd__'   ,  [lar()]    ,    lar()),
       ('__rand__'   ,  [lar()]    ,    lar()),
       ('__rdiv__'   ,  [lar()]    ,    'skip'),  # Hmm...
       ('__rmul__'   ,  [lar()]    ,    lar()),
       ('__ror__'    ,  [lar()]    ,    lar()),  
       ('__rsub__'   ,  [lar()]    ,    lar()),
       ('__sub__'    ,  [lar()]    ,    lar()),
       ('abs'        ,  None       ,    lar()),                                                                           
       ('all'        ,  None       ,    'use_numpy'),
       ('any'        ,  None       ,    'use_numpy'),
       ('clip'       ,  [-1, 1]    ,    'use_numpy'),
       ('copy'       ,  None       ,    'use_numpy'), 
       ('copylabel'  ,  None       ,    [[]]),                              
       ('copyx'      ,  None       ,    arr()),
      #('cov'        ,  None       ,    lar()),       # 2d only
       ('cumprod'    ,  [0]        ,    'use_numpy'),
       ('cumsum'     ,  [0]        ,    'use_numpy'), 
       ('cut_missing',  [0]        ,    lar()),                                            
       ('demean'     ,  None       ,    lar()),
       ('dtype'      ,  'property' ,    'use_numpy'),
       ('exp'        ,  None       ,    lar()),
       ('fill'       ,  [0]        ,    None),
       ('flatten'    ,  None       ,    lar()),
       ('flipaxis'   ,  None       ,    lar()),
       ('fromdict'   ,  [{}]       ,    lar()), 
       ('fromlist'   ,  [([],[])]  ,    lar()),
       ('fromlist'   ,  [[]]       ,    lar()),                               
       ('fromtuples' ,  [[]]       ,    lar()),                
       ('fromtuples' ,  [tuple()]  ,    lar()),
       ('get'        ,  None       ,    'skip'),  # Not expected to work                                                                                           
       ('getlabel'   ,  [0]        ,    []),
       ('getx'       ,  [0]        ,    arr()),
       ('group_mean' ,  [lar()]    ,    lar()),               
       ('group_median', [lar()]    ,    lar()),
      #('group_ranking',[lar()]    ,    lar()),       # 2d only
      #('insertaxis')                             # Not expected to pass
       ('isfinite'   ,  None       ,    lar()),                                                           
       ('isinf'      ,  None       ,    lar()), 
       ('isnan'      ,  None       ,    lar()), 
       ('keep_label' ,  ['==', 99, 0],  lar()),
       ('label'      ,  'property' ,    [[]]),               
      #('labelindex',[0, 0],IndexError),# Unit test doesn't handle Errors
       ('lag'        ,  [0]        ,    lar()),
      #('last_rank'  ,  None       ,    lar()),       # 2d only
      #('last_rank_decay', None    ,    lar()),       # 2d only
       ('log'        ,  None       ,    lar()), 
       ('maplabel'   ,  [None, 0]  ,    lar()),
      #('max',[0],IndexError),          # Unit test doesn't handle Errors               
      #('maxlabel'   ,[0],IndexError),  # Unit test doesn't handle Errors
      #('mean'       ,None    ,nan),    # NaN not supported by unit test                                                                                            
      #('median'     ,None    ,nan),    # NaN not supported by unit test 
       ('merge'      , [lar()]     ,    lar()),
      #('min',[0],IndexError)      ,    # Unit test doesn't handle Errors               
      #('minlabel',[0],IndexError) ,    # Unit test doesn't handle Errors
       ('morph'      ,  [[], 0]    ,    lar()),              
       ('morph_like' ,  [lar()]    ,    lar()), 
      #('movingrank' ,  None       ,    lar()),       # 2d only
      #('movingsum'  , [0] ,    ValueError),          # 2d only
      #('movingsum_forward',[0],ValueError),          # 2d only
       ('nan_replace',  [0]        ,    lar()),
       ('ndim'       ,  'property' ,    'use_numpy'),               
       ('nx'         ,  'property' ,    np.array([0])[0]),                
       ('power'      ,  [2]        ,    lar()),                                                            
       ('prod'       ,  None       ,    'use_numpy'), 
      #('pull',[[], 0],IndexError) ,    # Unit test doesn't handle Errors
      #('push'       ,  [0]        ,    lar()),       # 2d only              
      #('quantile'   ,  [4]        ,    lar()),       # 2d only               
       ('ranking'    ,  None       ,    lar()), 
       ('set'        ,  None       ,    'skip'),  # Not expected to work                
       ('shape'      ,  'property' ,    'use_numpy'),               
       ('shuffle'    ,  None       ,    None),                 
       ('shufflelabel', None       ,    None),
       ('sign'       ,  None       ,    lar()),                                            
       ('size'       ,  'property' ,    'use_numpy'),
       ('sortaxis'   ,  None       ,    lar()),
       ('sqrt'       ,  None       ,    lar()), 
       ('squeeze'    ,  None       ,    'use_numpy'),
       ('std'        ,  None       ,    'use_numpy'),
       ('sum'        ,  None       ,    'use_numpy'), 
       ('swapaxes'   ,  [0, 0]     ,    lar()), 
       ('todict'     ,  None       ,    {}), 
       ('tolist'     ,  None       ,    [[],[]]),
       ('totuples'   ,  None       ,    []),
       ('unflatten'  ,  None       ,    lar()),                                       
       ('vacuum'     ,  None       ,    lar()),
       ('var'        ,  None       ,    'use_numpy'),       
       ('std'        ,  None       ,    'use_numpy'),
       ('x'          ,  'property' ,    arr()),
       ('zscore'     ,  None       ,    lar()),
      ] 
                          
def test_empty(): 
    "Test larry methods for proper handling of empty larrys"      
    for attr, parameters, desired in mts:
        if str(desired) != 'skip':
            method = getattr(lar(), attr)
            if parameters == 'property':
                actual = method
            elif parameters is None:
                actual = method()
            else:
                actual = method(*parameters)
            if str(desired) == 'use_numpy':
                method = getattr(arr(), attr)
                if parameters == 'property':
                    desired = method
                elif parameters is None:
                    desired = method()
                else:
                    desired = method(*parameters)
                if type(desired) == np.ndarray:
                    desired = larry(desired)
            if type(actual) != type(desired):
                msg = "Type of 'actual' and 'desired' do not match for '%s'"
                yield assert_, False, msg % attr
            else:                                         
                result = actual == desired
                if type(result) == larry:
                    result = result.all()
                else:
                    result = np.all(result)    
                result &= type(actual) == type(desired) 
                msg = "method '" + attr + "' failed empty larry test"     
                yield assert_, result, msg      

# ---------------------------------------------------------------------------

# Above we tested larrys of shape (0,). What about larrys of shape, say,
# (2,0) or (2,0,3)?

def lar2():
    return larry(arr2())

def arr2():
    return np.ones((2,0,3))

def test_empty2():
    "Test larry methods for proper handling of empty larrys with ndim > 1."
    msg = "%s failed emtpy ndim > 1 test."
    yield assert_equal, lar2().sum(), arr2().sum(), msg % "sum"
    yield assert_equal, lar2().std(), arr2().std(), msg % "std"
    yield assert_equal, lar2().var(), arr2().var(), msg % "var"
            
