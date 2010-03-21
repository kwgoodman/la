
import os
import datetime

import numpy as np
import h5py
from la.util.prettytable import indent
from la.util.misc import randstring

from la import larry

        
class IO(object):
    "Save and load larrys in HDF5 format using a dictionary-like interface."
    
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
        want to load the entire larry which could, for example, be very large.
        
        A lara loads the labels but does not load the array data until you
        index into it.
        
        Each larry is stored in a HDF5 group. The group is assigned an
        attribute named 'larry' which is set to True. Inside the group is a
        HDF5 dataset containing the data (named 'x') and one dataset for each
        dimension of the label (named str(dimension)). For example, a 2d larry
        named 'price' is stored in a group called 'price' that contains a
        dataset called 'x' (the price) and two datasets called '0' and '1'
        (the labels).
        
        Before saving, the labels are converted to Numpy arrays, one array for
        each dimension. Therefore, to save a larry in HDF5 format, the
        elements of a label along any one dimension must be of the same type
        and that type must be supported by HDF5.     
        
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
            (with the same name) and delete the old archive. HDF5 does not
            reuse the freespace across openening and closing of the archive.
            
        Returns
        -------
            A dictionary-like IO object.
            
        See Also
        --------
        la.io.save : Save larrys without a dictionary-like interface.
        la.io.load : Load larrys without a dictionary-like interface.  
            
        Notes
        -----
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
        
        Examine the contents of the archive:
        
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
        self.f = h5py.File(filename)
        self.max_freespace = max_freespace
        
    def keys(self):
        "Return a list of larry names (keys) in archive."
        return archive_directory(self.f)
        
    def values(self):
        "Return a list of larry objects (values) in archive."
        return [self[key] for key in self]
        
    def items(self):
        "Return a list of all (key, value) pairs."
        return [(key, self[key]) for key in self]          

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
        
    def merge(self, key, lar, update=False):
        """
        Merge, or optionally update, a larry with a second larry.
        
        See larry.merge for details.
        
        Note: the entire larry is loaded from the archive, merged with `lar`
        and then the merged larry is saved back to the archive. The resize
        function of h5py is not used. In other words, this function might not
        be practical for very large larrys.
        
        """
        lar1 = self[key][:]
        lar2 = lar1.merge(lar, update=update)
        del self.f[key]
        self[key] = lar2 

    def __iter__(self):
        return iter(self.keys())
        
    def __len__(self):
        return len(self.keys())
        
    def __getitem__(self, key):
        if key in self.f:
            if _is_archived_larry(self.f[key]): 
                return lara(self.f[key])
            else:
                msg = "%s is in the archive but it is not a larry." 
                raise KeyError, msg % key   
        else:
            raise KeyError, "A larry named %s is not in the archive." % key   
        
    def __setitem__(self, key, value):
        
        # Make sure the data looks OK before saving
        if type(key) != str:
            raise TypeError, 'key must be a string of type str.'        
        if not isinstance(value, larry):
            raise TypeError, 'value must be a larry.'
        
        # Does an item (larry or otherwise) with given key already exist? If
        # so delete. Note that self.f.keys() [all keys] is used instead of
        # self.keys() [keys that are larrys].
        if key in self.f.keys():
            self.__delitem__(key)              
        
        # If you've made it this far the data looks OK so save it
        save(self.f, value, key)
        
    def __delitem__(self, key):
        delete(self.f, key)        
        self._repack_conditional()          
        
    def __repr__(self):
        table = [['larry', 'dtype', 'shape']]
        keys = self.keys()
        keys.sort()  # Display in alphabetical order
        for key in keys:
            # Code would be neater if I wrote shape = str(self[key].shape)
            # but I don't want to load the array, I just want the shape
            shape = str(self.f[key]['x'].shape)
            dtype = str(self.f[key]['x'].dtype)    
            table.append([key, dtype, shape])         
        return indent(table, hasHeader=True, delim='  ')  
    
    @property    
    def space(self):
        "The number of bytes used by the archive."
        self.f.flush()
        return self.f.fid.get_filesize()

    @property         
    def freespace(self):
        "The number of bytes of freespace in the archive."
        self.f.flush()
        global size
        size = 0
        def sizefinder(key, value):
            global size
            if isinstance(value, h5py.Dataset):
                size += value.id.get_storage_size()
        self.f.visititems(sizefinder)
        return self.space - size
        
    def repack(self):
        "Repack archive to remove freespace."
        self.f = repack(self.f)
        
    def _repack_conditional(self):
        "Repack if `max_freespace` is exceeded."
        if np.isfinite(self.max_freespace):
            if self.freespace() > self.max_freespace:
                self.f = self.repack() 
                
    @property    
    def filename(self):
        return self.f.filename                
                              
class lara(object):
    """
    Meet lara, she's a larry-like archive object.
    
    larry stores its data in a numpy array and a list (labels). lara stores
    its data in a h5py Dataset object and a list (labels).
    
    The reason for this class is that you may want to extract only part of the
    data from a larry in your archive. If you index into a lara you will get
    a larry back and only the data needed will be loaded from the archive.
    
    The values in the dictionary-like archive object, IO, are laras. You
    would not generally create your own lara; IO does that for you.
    
    """

    def __init__(self, group):
        """
        Meet lara, she's a larry-like archive object.
        
        Parameters
        ----------
        group : h5py.Group
            An instance of the h5py Group object that contains a larry.
            
        Example
        -------
        First let's make an archive and save a larry in it:
        
        >>> import la
        >>> io = la.IO('/tmp/data.hdf5')
        >>> io['x'] = la.larry([1,2,3,4])

        Next load the data from the archive:

        >>> y = io['x']
        
        Actually, only the labels are loaded. y is a lara object:
        
        >>> type(y)
            <class 'la.io.io.lara'>
        >>> type(y.x)
            <class 'h5py.highlevel.Dataset'>
        >>> type(y.label)
            <type 'list'>
      
        To convert y into a larry just index into y:
            
        >>> type(y[:])
            <class 'la.deflarry.larry'>
        >>> type(y[2:])
            <class 'la.deflarry.larry'>  
        
        """
        self.x = group['x']
        self.label = _load_label(group, len(self.x.shape))
    
    # Grab these methods from larry    
    __getitem__ = larry.__getitem__.im_func
    __setitem__ = larry.__setitem__.im_func    
    maxlabel = larry.maxlabel.im_func
    minlabel = larry.minlabel.im_func
    getlabel = larry.getlabel.im_func 
    labelindex = larry.labelindex.im_func
    shape = larry.shape
    dtype = larry.dtype            
        
    @property
    def ndim(self):
        return len(self.shape)                       

    @property
    def size(self):
        return np.prod(self.shape, dtype=int)
        
# Archive functions ---------------------------------------------------------       

def save(file, lar, key):
    """
    Save a larry in HDF5 format.

    Each larry is stored in a HDF5 group. The group is assigned an
    attribute named 'larry' which is set to True. Inside the group is a
    HDF5 dataset containing the data (named 'x') and one dataset for each
    dimension of the label (named str(dimension)). For example, a 2d larry
    named 'price' is stored in a group called 'price' that contains a
    dataset called 'x' (the price) and two datasets called '0' and '1'
    (the labels).
    
    Before saving, the labels are converted to Numpy arrays, one array for
    each dimension. Therefore, to save a larry in HDF5 format, the
    elements of a label along any one dimension must be of the same type
    and that type must be supported by HDF5.
    
    If all labels along an axis are dates of type datetime.date, then the
    dates are converted to integers before saving and the HDF5 Dataset used
    to store that label is assigned an attribute name 'isdate' which is set
    to True. When loading the larry, the dates will automatically be converted
    back to datetime.date dates.
    
    Parameters
    ----------
    file : str or h5py.File
        Filename or h5py.File object of the archive.
    lar : larry
        Data to save.
    key : str
        Name of larry.
        
    See Also
    --------
    la.io.load : Load larrys without a dictionary-like interface.
    la.IO : A dictionary-like interface to the archive.    
        
    Examples
    --------
    Create a larry:
    
    >>> la.save(x, 'x')

    Save the larry:

    >>> la.save('/tmp/x.hdf5', x, 'x')        
 
    """

    # Check input
    if type(lar) != larry:
        raise TypeError, 'lar must be a larry.'
    if type(key) != str:
        raise TypeError, 'key must be a string.'    
    
    # Get a h5py.File instance
    f, opened = _openfile(file)
    
    # Do we need to create any intermediate groups?
    _create_nested_groups(f, key)  
        
    # Save larry
    fkey = f[key]
    fkey.attrs['larry'] = True
    fkey['x'] = lar.x
    for i in range(lar.ndim):
        fkey[str(i)], isdate = _list2array(lar.label[i])
        fkey[str(i)].attrs['isdate'] = isdate
    
    # Close if file is a filename   
    if opened:
        f.close()
    else:
        f.flush()    
        
def load(file, key):
    """
    Load a larry from a HDF5 archive.

    Each larry is stored in a HDF5 group. The group is assigned an
    attribute named 'larry' which is set to True. Inside the group is a
    HDF5 dataset containing the data (named 'x') and one dataset for each
    dimension of the label (named str(dimension)). For example, a 2d larry
    named 'price' is stored in a group called 'price' that contains a
    dataset called 'x' (the price) and two datasets called '0' and '1'
    (the labels).
    
    Parameters
    ----------
    file : str or h5py.File
        Filename or h5py.File object of the archive.
    key : str
        Name of larry.
        
    Returns
    ------- 
    out : larry
        Returns the larry from the archive.   
        
    See Also
    --------
    la.io.save : Save larrys without a dictionary-like interface.
    la.IO : A dictionary-like interface to the archive.  
        
    Examples
    --------
    Create a larry:
    
    >>> la.save(x, 'x')

    Save the larry:

    >>> la.save('/tmp/x.hdf5', x, 'x')
    
    Now load it:
    
    >>> y = la.load('/tmp/x.hdf5', 'x')            
 
    """
    
    # Check input
    if type(key) != str:
        raise TypeError, 'key must be a string.'    
    f, opened = _openfile(file)
    if key not in f:
        raise KeyError, "A larry named '%s' is not in archive." % key
    if not _is_archived_larry(f[key]):
        raise KeyError, 'key (%s) is not a larry.' % key
        
    # Load larry    
    group = f[key]
    x = group['x'][:]
    label = _load_label(group, x.ndim)                 
                     
    # Close if file is a filename   
    if opened:
        f.close()
        
    return larry(x, label)
    
def delete(file, key):
    """
    Delete a larry from a HDF5 archive.
    
    Parameters
    ----------
    file : str or h5py.File
        Filename or h5py.File object of the archive.
    key : str
        Name of larry.
        
    Returns
    ------- 
    out : None
        Nothing is returned, just None.   
        
    See Also
    --------
    la.io.save : Save larrys without a dictionary-like interface.
    la.io.load : Load larrys without a dictionary-like interface.
    la.IO : A dictionary-like interface to the archive.  
        
    Examples
    --------
    Create a larry:
    
    >>> la.save(x, 'x')

    Save the larry:

    >>> la.save('/tmp/x.hdf5', x, 'x')
    
    Now delete it:
    
    >>> y = la.delete('/tmp/x.hdf5', 'x')            
 
    """
    
    # Check input
    if type(key) != str:
        raise TypeError, 'key must be a string.'    
    f, opened = _openfile(file)
    if key not in f:
        raise KeyError, "A larry named '%s' is not in archive." % key
    if not _is_archived_larry(f[key]):
        raise KeyError, 'key (%s) is not a larry.' % key    
    
    # Delete
    del f[key]               
                     
    # Close if file is a filename   
    if opened:
        f.close()        
    
def repack(file):
    """
    Repack archive to remove freespace.
    
    Parameters
    ----------
    file : h5py File or str
        A h5py File instance of an archive such as h5py.File('/tmp/data.hdf5')
        or a filename.
        
    Returns
    -------
    file : h5py File or None
        If the input is a h5py.File then a h5py File instance of the
        repacked archive is returned. The input File instance will no longer
        be useable. If the input was a filename, then None is returned. 

    """
    f1, opened = _openfile(file) 
    filename1 = f1.filename
    filename2 = filename1 + '_repack_tmp_' + randstring(4)
    f2 = h5py.File(filename2)
    for key in f1.keys():
        f1.copy(key, f2)
    f1.close()
    f2.close()
    filename_tmp = filename1 + '_repack_rename_tmp_' + randstring(4)
    os.rename(filename1, filename_tmp)
    os.rename(filename2, filename1) 
    if opened:
        f = None  
    else:
        f = h5py.File(filename1)
    os.remove(filename_tmp)
    return f   
    
def is_archived_larry(file, key):
    "True if the key (larry name) is in the archive, False otherwise."
    f, opened = _openfile(file)
    if key in f:
        answer = _is_archived_larry(f[key])
    else:
        raise ValueError, 'key (%s) is not in archive.' % str(key)    
    if opened:
        f.close()                
    return answer
    
def archive_directory(file):
    "Return a list of the keys (larry names) in the archive."
    f, opened = _openfile(file) 
    keys = []
    def append_larrys(name, obj):
        if _is_archived_larry(obj):
            keys.append(name)
    f.visititems(append_larrys)
    if opened:
        f.close()
    return keys            
    
# Utility functions for internal use ----------------------------------------                  

def _load_label(group, ndim):
    "Load larry labels from archive given the hpy5.Group object of the larry."
    label = []
    for i in range(ndim):
        labellist = group[str(i)][:].tolist()
        if group[str(i)].attrs['isdate']:
            labellist = map(datetime.date.fromordinal, labellist)
        label.append(labellist)
    return label                     

def _list2array(x):
    "Convert list to array if elements are of the same type, raise otherwise."
    if type(x) != list:
        raise TypeError, 'x must be a list'
    type0 = type(x[0])
    if not all([type(i)==type0 for i in x]):
        msg = 'Elements of a label along any one dimension must be of the '
        msg += 'same type.'  
        raise TypeError, msg
    isdate = False    
    if type0 == datetime.date:
        x = map(datetime.date.toordinal, x)
        isdate = True    
    return np.asarray(x), isdate                 
    
def _openfile(file):
    """
    Open an archive if input is a path.
    
    Parameters
    ----------
    file : str or h5py.File
        Filename or h5py.File instance of the archive.
        
    Returns
    ------- 
    f : h5py.File
        Returns a h5py.File instance.
    opened : bool
        True is `file` is a path; False if `file` is a h5py.File object.   
    
    """
    if isinstance(file, h5py.File):
        f = file
        opened = False
    elif type(file) == str:
        f = h5py.File(file)
        opened = True
    else:
        msg = "file must be a h5py.File object or a string (path)."
        raise TypeError, msg    
    return f, opened                 

def _is_archived_larry(obj):
    "True if obj is an archived larry, False otherwise."
    if isinstance(obj, h5py.Group):
        if 'larry' in obj.attrs:
            if obj.attrs['larry'] is True:
                if 'x' in obj:
                    ndim = len(obj['x'].shape)
                    labels = map(str, range(ndim))
                    if all([label in obj for label in labels]):
                        return True               
    return False
    
def _create_nested_groups(f, path):
    "Create a nested set of groups."
    groups = path.split('/')
    groups = [group for group in groups if group != '']
    for i in range(len(groups)):
        group = '/'.join(groups[:i+1])
        if group not in f:
            f.create_group(group)
        else:
            if not isinstance(f[group], h5py.Group):
                msg = '%s already exists and is not a h5.py.Group object.'
                raise ValueError, msg % group   
                    
