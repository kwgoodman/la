
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
        
        Dictionaries are made up of (key, value) pairs. In an IO object, a
        key is the name of a larry. The value part of the dictionary is a
        larry when saving data and is a lara, a larry-like archive object,
        when loading data.
        
        (h5py has the same duality. When saving, the values are Numpy arrays;
        when loading the values are h5py Dataset objects.)
        
        To convert a lara into a larry just index into the lara.
        
        The reason why loading does not return a larry is that you may not
        want to load the entire larry which might, for example, be very large.
        
        A lara loads the labels but does not load the array data until you
        index into it.     
        
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
        Save a larry in the archive:
             
        >>> import la
        >>> io = la.IO('/tmp/dataset.hdf5')
        >>> io['x'] = la.larry([1,2,3])  # <-- Save
        
        Look at what is in the archive:
        
        >>> io
           
        larry  dtype  shape
        -------------------
        x      int64  (3,)
        
        Overwrite the contents of x in the archive: 

        >>> io['x'] = la.larry([4.0])  # <-- Overwrite

        Load from the archive:

        >>> y = io['x']  # <-- Load
        >>> type(y)
            <class 'la.io.io.lara'>
        >>> type(y[:])
            <class 'la.deflarry.larry'>
        >>> type(y[2:])
            <class 'la.deflarry.larry'> 
            
        Test if x is in the archive:           
        
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
            x = self.fid[key + '.x']
            label = self.fid[key + '.label'].value[0]
            label = cPickle.loads(label)   
            return lara(x, label)
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
        self.fid.flush()
        return self.fid.fid.get_filesize()
         
    def freespace(self):
        "How many bytes of freespace are in the archive?"
        self.fid.flush()
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
                
class lara(object):
    """
    Meet lara, a larry-like archive object.
    
    larry stores its data in a numpy array and a list (labels). lara stores
    its data in a h5py Dataset object and a list (labels).
    
    The reason for this class is that you may want to extract only part of the
    data from a larry in your archive. If you index into a lara you will get
    a larry back and only the data needed will be loaded from the archive.
    
    The values in the dictionary-like archive object, IO, are laras. You
    would not generally create your own lara; IO does that for you.
    
    """

    def __init__(self, x_h5py_Dataset, label):
        """
        Meet lara, she's a larry-like archive object.
        
        Parameters
        ----------
        x_h5py_Dataset : h5py Dataset object
            An object that knows how to extract all or parts of a larry in the
            archive.
        label : list of lists
            A list with labels for each dimension of x. If x is 2d, for
            example, then label should be a list that contains two lists, one
            for the row labels and one for the column labels. If x is 1d label
            should be a list that contain one list of names.
            
        Example
        -------
        First let's make an archive and save a larry in it:
        
        >>> import la
        >>> import numpy as np
        >>> io = la.IO('/tmp/data.hdf5')
        >>> io['x'] = la.larry([1,2,3,4])

        Next load the data from the archive:

        >>> y = io['x']
        
        Actually, the data is not loaded. Instead y is a lara object.
        
        >>> type(y)
            <class 'la.io.io.lara'>
        >>> type(y.x)
            <class 'h5py.highlevel.Dataset'>
            
        To convert to a larry you need to index into y:
            
        >>> type(y[:])
            <class 'la.deflarry.larry'>
        >>> type(y[2:])
            <class 'la.deflarry.larry'>
            
        Only the data you index into is loaded from the archive.    
        
        """
        self.x = x_h5py_Dataset
        self.label = label
    
    # Grab these methods from larry    
    __getitem__ = larry.__getitem__.im_func    
    maxlabel = larry.maxlabel.im_func
    minlabel = larry.minlabel.im_func
    getlabel = larry.getlabel.im_func 
    labelindex = larry.labelindex.im_func
    shape = larry.shape
    dtype = larry.dtype
       
    def __setitem__(self, index, value):
        raise NotImplementedError, 'I will code this after I get HDF5 1.8.'         
        
    @property
    def ndim(self):
        return len(self.shape)                       

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)
        
def list2keys(x):
    names = [z.split('.')[0] for z in x]
    names = set(names)
    keys = []
    for name in names:
        if ((name + '.x') in x) and ((name + '.label') in x):
            keys.append(name)
    return keys              

