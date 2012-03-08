"Test larry methods for proper handling of empty larrys"

import numpy as np
from numpy.testing import assert_

from la import larry, nan
from la.util.testing import assert_larry_equal as ale

def lar():
    return larry([])

def arr():
    return np.array([])   

#       Method          Parameters      Returns
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
       ('lastrank'   ,  None       ,    nan),
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
      #('move_sum'   , [0] ,    ValueError),          # 2d only
      #('movingsum_forward',[0],ValueError),          # 2d only
       ('nan_replace',  [0]        ,    lar()),
       ('ndim'       ,  'property' ,    'use_numpy'),               
       ('nx'         ,  'property' ,    np.array([0])[0]),                
       ('power'      ,  [2]        ,    lar()),                                                            
       ('prod'       ,  None       ,    nan), 
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
       ('swapaxes'   ,  [0, 0]     ,    lar()), 
       ('todict'     ,  None       ,    {}), 
       ('tolist'     ,  None       ,    [[],[]]),
       ('totuples'   ,  None       ,    []),
       ('unflatten'  ,  None       ,    lar()),                                       
       ('vacuum'     ,  None       ,    lar()),     
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
                if np.isscalar(actual) and np.isscalar(desired):
                    if np.isnan(actual).all() and np.isnan(desired).all():
                        result = True
                if type(result) == larry:
                    result = result.all()
                else:
                    result = np.all(result)    
                result &= type(actual) == type(desired) 
                msg = "method '" + attr + "' failed empty larry test"     
                yield assert_, result, msg
    
# ---------------------------------------------------------------------------

# Check that the right shape and value are returned by the reducing methods
# when the input has a shape tuple that contains 0

False = np.False_
True = ~False

def reduce_methods():
    r = [{'la': 'sum',      'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'prod',     'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'mean',     'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'geometric_mean', 'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'median',   'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'max',      'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'min',      'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'std',      'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'var',      'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'any',      'np': 'sum', 'dflt': False, 'kw': {}},
         {'la': 'all',      'np': 'sum', 'dflt': True,  'kw': {}},
         {'la': 'lastrank', 'np': 'sum', 'dflt': nan,   'kw': {}},
         {'la': 'lastrank', 'np': 'sum', 'dflt': nan,   'kw': {'decay': 10}}]
    return r
        
def get_shapes():
    s = [(0,), (0,1), (0,2), (2,0), (1,2,0), (1,0,2), (0,1,2), (2,3,4,0),
         (0,2,3,4), (1,0,0,2), (0,0,1), (0,0)] 
    return s        
    
def test_reduce_shape():
    "Empty larry test"
    msg = 'larry.%s failed for shape %s and axis %s'
    for method in reduce_methods():
        for shape in get_shapes():
            axeslist = [None] + range(len(shape))
            for axis in axeslist:
                arr = np.zeros(shape)
                npmethod = getattr(arr, method['np'])
                arr = npmethod(axis=axis)
                default = method['dflt']
                if np.isscalar(arr):
                    arr = default 
                else:    
                    arr.fill(default)
                    arr = larry(arr)
                y = larry(np.zeros(shape))
                ymethod = getattr(y, method['la'])
                lar = ymethod(axis=axis, **method['kw'])               
                yield ale, lar, arr, msg % (method['la'], shape, axis)             
def test_50():
    "Regression #50"
    actual = larry([], dtype=np.int).sum(0)
    desired = np.nan
    ale(actual, desired, "Regression #50")
