
import cPickle, zipfile

import numpy as np
from prettytable import indent

from la import larry


def IO(filename, mode=None):
    """
    Save and load larrys.
    
    Parameters
    ----------
    filename : str, file object (in npz mode only)
        The `filename` is the path to the archive. In mode 'npz' the filename
        can be a file object.
    mode : {'hdf5', 'npz'}, optional
        Archive mode. To use the hdf5 mode you must have h5py installed. To
        use the npz mode your Python distribution must have gzip. If mode is
        None (default) then an attempt will be made to determine the mode
        from the filename extension (.npz for npz mode; .hdf5 for hdf5 mode).
        
    Returns
    -------
        An IO object.        
    
    """
    if mode is None:
        if type(filename) is str:
            if '.' in filename:
                mode = filename.split('.')[-1]
    if mode == 'hdf5':
        return IO_hdf5(filename)
    elif mode == 'npz':
        return IO_npz(filename)    
    else:
        raise ValueError, "mode must be 'hdf5' or 'npz'."

class IO_base(object):
        
    def __iter__(self):
        return iter(self.keys())        
        
    def __contains__(self, key):
        return key in self.keys()
        
    def __len__(self):
        return len(self.keys())
        
    def __repr__(self):
        table = [['larry', 'dtype', 'shape']]
        for key in self.keys():
            shape = str(self[key].shape)
            dtype = str(self[key].dtype)            
            table.append([key, dtype, shape])         
        return indent(table, hasHeader=True, delim='  ')  
        
class IO_hdf5(IO_base):

    def __init__(self, filename):
        try:
            # Lazy import in case user hasn't installed this great package
            import h5py
        except:
            raise ImportError, 'Cannot import h5py.'    
        self.file = filename
        self.fid = h5py.File(self.file)
        
    def keys(self):
        return list2keys(self.fid.keys())        
        
    def __getitem__(self, key):
        if key in self:
            x = self.fid[key + '.x'].value
            label = self.fid[key + '.label'].value[0]
            label = cPickle.loads(label)   
            return larry(x, label)
        else:
            raise KeyError, "A larry named %s is not in the file." % key   
        
    def __setitem__(self, key, value):
        
        # Make sure the data looks OK before saving since there is no rewind
        if type(key) != str:
            raise TypeError, 'key must be a string of type str.'        
        if not isinstance(value, larry):
            raise TypeError, 'value must be a larry.'
        x = value.x
        label = value.label
        
        # If you've made it this far the data looks OK so save it
        self.fid[key + '.x'] = x
        self.fid[key + '.label'] = np.asarray([cPickle.dumps(label)]) 
        self.fid.flush()
        
class IO_npz(IO_base):

    def __init__(self, filename):   
        self.file = filename
        
    def keys(self):
        try:
            fid = np.load(self.file)
        except IOError:
            return []    
        return list2keys(fid.keys())        
        
    def __getitem__(self, key):
        try:
            fid = np.load(self.file)
        except IOError:
            raise KeyError, "File is not yet opened; does it exist?"  
        if key in self:
            x = fid[key + '.x']
            label = fid[key + '.label']
            label = cPickle.loads(label[0])   
            return larry(x, label)
        else:
            raise KeyError, "A larry named %s is not in the file." % key   
        
    def __setitem__(self, key, value):
        
        # Make sure the data looks OK before saving since there is no rewind
        if type(key) != str:
            raise TypeError, 'key must be a string of type str.'        
        if not isinstance(value, larry):
            raise TypeError, 'value must be a larry.'
        x = value.x
        label = value.label
        
        # If you've made it this far the data looks OK so save it
        kwdict = {}
        kwdict[key + '.x'] = x
        kwdict[key + '.label'] = (cPickle.dumps(label),)
        np.savez(self.file, **kwdict) 
        
def list2keys(x):
    names = [z.split('.')[0] for z in x]
    names = set(names)
    keys = []
    for name in names:
        if ((name + '.x') in x) and ((name + '.label') in x):
            keys.append(name)
    return keys              

