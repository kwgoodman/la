"Functions for saving and loading larrys."
     
import cPickle
import numpy as np
from la import larry


# npz ----------------------------------------------------------------------

def save_npz(file, **kwargs):
    """
    Save larry(s) to a numpy npz file.
    
    Parameters
    ----------
    file : {str, file}
        File in which to save the larrys. If `file` is a str and it does not
        end in '.npz' then '.npz' will be appended to the file name.
    kwargs : Keyword arguments
        Keyword, value pairs where keyword is the name of the larry and value
        is the larry.
        
    Returns
    -------
    None
    
    Example
    -------    
    >> from la import larry, save_npz, load_npz
    >> from numpy import array
    >> 
    >> x = larry(array([[1,2,3], [3,4,5]]))
    >> y = larry(array([1,2,3]))
    >> z = larry(array([[9,2,3], [3,4,9]]))
    >> 
    >> save_npz('/tmp/larrys.npz', x=x, y=y, z=z)
    >> d = load_npz('/tmp/larrys.npz')
    >> 
    >> (d['x'] == x).all()
       True
    >> (d['y'] == y).all()
       True
    >> (d['z'] == z).all()
       True
            
    """
    kwdict = {}
    for kw in kwargs:
        kwdict[kw + '.x'] = kwargs[kw].x
        kwdict[kw + '.label'] = (cPickle.dumps(kwargs[kw].label),)
    np.savez(file, **kwdict)
    
def load_npz(file):
    """
    Load larrys from a numpy npz file into a dictionary of larrys.
    
    The file can contain data other than larrys, a plain Numpy array, for
    example. The non larry data will be ignored. 
    
    Parameters
    ----------
    file : {str, file}
        File from which to load the larrys.
        
    Returns
    -------
    larrydict : dict
        Returns a dictionary of all the larrys in the file.
    
    Example
    -------    
    >> from la import larry, save_npz, load_npz
    >> from numpy import array
    >> 
    >> x = larry(array([[1,2,3], [3,4,5]]))
    >> y = larry(array([1,2,3]))
    >> z = larry(array([[9,2,3], [3,4,9]]))
    >> 
    >> save_npz('/tmp/larrys.npz', x=x, y=y, z=z)
    >> d = load_npz('/tmp/larrys.npz')
    >> 
    >> (d['x'] == x).all()
       True
    >> (d['y'] == y).all()
       True
    >> (d['z'] == z).all()
       True
            
    """
    
    f = np.load(file)
    
    # There may be non larry objects in the file (plain numpy arrays, for
    # example). If a name has a .x and .label component, such as lar.x and
    # lar.label, then assume lar is a larry. 
    fkeys = f.keys()
    names = [z.split('.')[0] for z in fkeys]
    names = list(set(names))
    larrydict = {}
    for name in names:
        if ((name + '.x') in fkeys) and ((name + '.label') in fkeys):
            x = f[name + '.x']
            label = f[name + '.label']
            label = cPickle.loads(label[0])
            larrydict[name] = larry(x, label)
                
    return larrydict
        
