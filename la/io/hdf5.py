"Functions for saving and loading larrys in Numpy npz files."
     
import cPickle

import numpy as np
import h5py

from la import larry


def save_hdf5(filename, **kwargs):
    """
    Save larry(s) to a h5py HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to file in which to save the larrys.
    kwargs : Keyword arguments
        Keyword, value pairs where keyword is the name of the larry and value
        is the larry.
        
    Returns
    -------
    None
            
    """
    f = h5py.File(filename)
    for kw in kwargs:
        f[kw + '.x'] = kwargs[kw].x
        f[kw + '.label'] = np.asarray([cPickle.dumps(kwargs[kw].label)])
    f.close()
    
def load_hdf5(filename, larry_name):
    """
    Load one larry from a h5py HDF5 file.
    
    Parameters
    ----------
    filename : str
        Path to file from which to load the larry.
    larry_name : str
        The name of the larry to load.    
        
    Returns
    -------
    out : larry
        Returns a larry.
            
    """    
    f = h5py.File(filename)
    x = f[larry_name + '.x'].value
    label = f[larry_name + '.label'].value[0]
    label = cPickle.loads(label)   
    f.close()                
    return larry(x, label)
    
