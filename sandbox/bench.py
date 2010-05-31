
import numpy as np

import la

from autotimeit import autotimeit

def bench(verbose=True):
    statements, setups = suite()
    results = []
    for key in statements:
        if verbose:
            print
            print key
        for stmt in statements[key]:
            for shortname in setups:
                t = autotimeit(stmt, setups[shortname])
                results.append((stmt, shortname, t))
                if verbose:
                    print
                    print '\t' + stmt
                    print '\t' + shortname         
                    print '\t' + str(t)
    return la.larry.fromtuples(results)                

def fx(shape):
    x = np.random.randn(*shape)
    lar = la.larry(x)
    lar.shufflelabel()
    return lar
    
def suite():

    statements = {}
    setups = {}
    
    setups['(1000,)'] = "from bench import fx; N = 1000; x = fx((N,)); y = fx((N,)); idx = range(N)[::-1]"
    setups['(500,500)'] = "from bench import fx; N = 500; x = fx((N, N)); y = fx((N, N)); idx = range(N)[::-1]"

    # Unary
    s = ['x.log()',
         'x.exp()',
         'x.sqrt()',
         'x.power(q=2)']
    statements['unary'] = s
    
    # Binary
    s = ['x + x',
         'x + y']
    statements['binary'] = s 
    
    # Alignment      
    s = ['x.morph(idx, axis=0)',
         'x.merge(y, update=True)']
    statements['alignment'] = s 
    
    return statements, setups

