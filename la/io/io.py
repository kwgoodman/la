
import cPickle
import os

import numpy as np
import h5py
from prettytable import indent

from la import larry

        
class IO(object):
    
    def __init__(self, filename, max_freespace=np.inf):
        """
        Save and load larrys in HDF5 format using a dictionary-like interface.
        
        Dictionaries are made up of (key, value) pairs. In an IO object, a key
        is the name (string) of a larry and a value is a larry object.        
        
        Parameters
        ----------
        filename : str
            The `filename` is the path to the archive. If the file does not
            exists, it will be created.
        max_freespace : scalar
            If the size of the freespace (unused archive space) exceeds
            `max_freespace` bytes after a larry is deleted from the archive,
            then the archive is repacked. The default (np.inf) is to never
            repack. Repack means to transfer all the larrys to a new archive
            (with the same name) and delete the old archive.
            
        Returns
        -------
            A dictionary-like IO object.
            
        Notes
        -----
        - Each larry is stored as two files in HDF5: the data part of the
          larry is stored as a Numpy array and the label part is first pickled
          and then placed in a one-element 1d Numpy array.
        - Because the archive interface is dictionary-like, data will be
          overwritten when assigning a (key, value) pair if the key already
          exists in the archive.
        - Deleting a larry from the archive only unlinks it. You won't be able
          to reuse the unlinked space if you close the connection. This is
          a limitation of the HDF5 format, not a limitation of the IO class
          or h5py. You can repack the archive with the repack method or have
          it done automatically for you: see `freespace` above.
          
        Examples
        --------       
        >>> import la
        >>> io = la.IO('/tmp/dataset.hdf5')
        >>> io['x'] = la.larry([1,2,3])  # <-- Save
        >>> io
           
        larry  dtype  shape
        -------------------
        x      int64  (3,) 

        >>> io['x'] = la.larry([4.0])  # <-- Overwrite
        >>> io
           
        larry  dtype    shape
        ---------------------
        x      float64  (1,) 

        >>> y = io['x']  # <-- Load
        >>> 'x' in io
            True    
        >>> del io['x']  # <-- Delete (unlink)
        >>> 'x' in io
            False             
            
        """   
        self.file = filename
        self.fid = h5py.File(self.file)
        self.max_freespace = max_freespace
        
    def keys(self):
        "Return a list of larry names (keys) in archive."
        return list2keys(self.fid.keys())
        
    def values(self):
        "Return a list of larry objects (values) in archive."
        v = []
        for key in self:
            v.append(self[key])
        return v
        
    def items(self):
        "Return a list of all (key, value) pairs."
        i = []
        for key in self:
            i.append((key, self[key]))
        return i            

    def iterkeys(self):
        "An iterator over the keys."
        for key in self:
            yield key

    def itervalues(self):
        "An iterator over the values."
        for key in self:
            yield self[key]
        
    def iteritems(self):
        "An iterator over (key, value) items."
        for key in self:
            yield (key, self[key])                
            
    def has_key(self, key):
        "True if key is in archive, False otherwise."
        return key in self
        
    def clear(self):
        """
        Warning: this will delete (unlink) all larrys from the archive!
        """
        for key in self:
            self.__delitem__(key)
        self._repack_conditional()              

    def __iter__(self):
        return iter(self.keys())
        
    def __len__(self):
        return len(self.keys())
        
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
        label = np.asarray([cPickle.dumps(label)])
        
        # Does a larry with given key already exist? If so delete
        if key in self:
            self.__delitem__(key)    
        
        # If you've made it this far the data looks OK so save it
        self.fid[key + '.x'] = x
        self.fid[key + '.label'] = label 
        self.fid.flush()
        
    def __delitem__(self, key):
        del self.fid[key + '.x']
        del self.fid[key + '.label']
        self._repack_conditional()  
        
    def __repr__(self):
        table = [['larry', 'dtype', 'shape']]
        keys = self.keys()
        keys.sort()  # Display in alphabetical order
        for key in keys:
            # Code would be neater if I wrote shape = str(self[key].shape)
            # but I don't want to load the array, I just want the shape
            shape = str(self.fid[key + '.x'].shape)
            dtype = str(self.fid[key + '.x'].dtype)    
            table.append([key, dtype, shape])         
        return indent(table, hasHeader=True, delim='  ')

    # Disk space issues ----------------------------------------------------
        
    def space(self):
        "How many bytes does the archive use?"
        return self.fid.fid.get_filesize()
         
    def freespace(self):
        "How many bytes of freespace are in the archive?"
        used = [self.fid[z].id.get_storage_size() for z in self.fid.keys()] 
        return self.space() - sum(used)
        
    def repack(self):
        "Repack the archive to remove freespace."
        filenew = self.file + '_la_repack_tmp'
        fidnew = h5py.File(filenew)
        for key in self.fid.keys():
            fidnew[key] = self.fid[key].value
        fileold = self.file + '_la_rename_tmp'
        os.rename(self.file, fileold)
        os.rename(filenew, self.file)
        self.fid = h5py.File(self.file)
        os.remove(fileold)
        
    def _repack_conditional(self):
        "Repack if `max_freespace` is exceeded."
        if np.isfinite(self.max_freespace):
            if self.freespace() > self.max_freespace:
                self.repack()        
        
def list2keys(x):
    names = [z.split('.')[0] for z in x]
    names = set(names)
    keys = []
    for name in names:
        if ((name + '.x') in x) and ((name + '.label') in x):
            keys.append(name)
    return keys              

