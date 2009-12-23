"Labeled array class"

import datetime
from copy import deepcopy

import numpy as np   

from la.afunc import (sector_rank, sector_mean, sector_median, covMissing,
                      fillforward_partially, quantile, ranking, lastrank,
                      movingsum_forward, lastrank_decay, movingrank,
                      movingsum, nans, nanmean, nanmedian, nanstd)


class larry(object):
    "Labeled array"

    def __init__(self, x, label=None):
        """Meet larry, he's a labeled array.
        
        Parameters
        ----------
        x : numpy array
            Data, NaN are treated as missing data.
        label : {list of lists, None}, optional
            A list with labels for each dimension of x. If x is 2d, for
            example, then label should be a list that contains two lists, one
            for the row labels and one for the column labels. If x is 1d label
            should be a list that contain one list of names. If label is None
            (default) integers will be used to label the the row, columns, etc.
            
        Raises
        ------
        ValueError
            If x is not a numpy array, or if number of elements in label does
            not match the dimensions of x, or if the elements in label are not
            unique along each dimension, or if the elements of label are not
            lists. 

        """
        if type(x) is not np.ndarray:
            raise ValueError, 'x must be a numpy array'
        if label is None:
            label = [range(z) for z in x.shape]
        if x.ndim != len(label):
            ValueError, 'Exactly one label per dimension needed'
        msg1 = 'Length mismatch in label and x along dimension %d'
        msg2 = "Elements of label not unique along dimension %d. There are %d labels named `%s`."
        for i, l in enumerate(label):
            nlabel = len(l)
            if x.shape[i] != nlabel:
                raise ValueError, msg1 % i
            if len(frozenset(l)) != nlabel:
                # We have duplicates in the label, give an example
                count = {}
                for li in l:
                    count[li] = count.get(li, 0) + 1
                for key, value in count.iteritems():
                    if value > 1:
                        break   
                raise ValueError, msg2 % (i, value, key)
            if type(l) is not list:
                raise ValueError, 'label must be a list of lists'
        self.x = x
        self.label = label

    # Unary functions --------------------------------------------------------  

    def log(self):
        """Element by element base e logarithm.
        
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : larry
            Returns a copy with log of x values.
        
        """
        y = self.copy()
        np.log(y.x, y.x)
        return y  

    def exp(self):
        """Element by element exponential.
                
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : larry
            Returns a copy with exp of x values.
                
        """
        y = self.copy()
        np.exp(y.x, y.x)
        return y
        
    def sqrt(self):
        """Element by element square root.
                
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : larry
            Returns a copy with square root of x values.
                
        """
        y = self.copy()        
        np.sqrt(y.x, y.x)
        return y 
        
    def power(self, q):               
        """Element by element x**q.
                
        Parameters
        ----------
        q : scalar
            The power to raise to.
        
        Returns
        -------
        out : larry
            Returns a copy with x values to the qth power.
                
        """
        y = self.copy()
        np.power(y.x, q, y.x)
        return y
        
    def __pow__(self, q):
        return self.power(q)           
        
    def cumsum(self, axis): 
        """Cumsum, treating NaNs as zero.
        
        Parameters
        ----------
        axis : integer, optional
            axis to cumsum along, no default. None is not allowed.
            
        Returns
        -------
        out : larry
            Returns a copy with cumsum along axis.  
            
        Raises
        ------
        AssertionError
            If axis is None.    
                
        """
        assert axis is not None, 'axis cannot be None'
        y = self.copy()
        y[np.isnan(y.x)] = 0
        y.x.cumsum(axis, out=y.x)
        return y        

    def clip(self, lo, hi):
        """Clip x values.

        Parameters
        ----------
        lo : scalar
            All data values less than `lo` are set to `lo`.
        hi : scalar    
            All data values greater than `hi` are set to `hi`.
                        
        Returns
        -------
        out : larry
            Returns a copy with x values clipped.       
            
        Raises
        ------
        AssertionError
            If `lo` is greater than `hi`.
        
        """
        assert lo <= hi, 'lo should be less than or equal to hi'
        y = self.copy()
        y.x.clip(lo, hi, y.x)
        return y

    def nan_replace(self, replace_with=0):
        """Replace NaNs.
        
        Parameters
        ----------
        replace_with : scalar
            Value to replace NaNs with.
                        
        Returns
        -------
        out : larry
            Returns a copy with NaNs replaced.     
                    
        """
        y = self.copy()
        y.x[np.isnan(y.x)] = replace_with
        return y
        
    def __neg__(self):
        y = self.copy()
        y.x *= -1
        return y
    
    def __pos__(self):
        return self.copy()
        
    def abs(self):
        """Absolute value of x.
                        
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : larry
            Returns a copy with the absolute values of the x data.
       
        """
        y = self.copy()
        np.absolute(y.x, y.x)
        return y
        
    def __abs__(self):
        return self.abs()        
        
    # Binary Functions ------------------------------------------------------- 
    
    # We need this to take care of radd and rsub when a matrix is on the left-
    # hand side. Without it, the matrix object will use broadcasting, treating
    # larry objects as scalars.
    __array_priority__ = 10                      
        
    def __add__(self, other):
        if isinstance(other, larry):       
            x, y, label = self.__align(other)
            x += y
            return type(self)(x, label)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            y = self.copy()
            y.x += other
            return y                    
        raise TypeError, 'Input must be scalar, array, or larry.' 
    
    __radd__ = __add__
    
    def __sub__(self, other):
        if isinstance(other, larry):
            x, y, label = self.__align(other)        
            x -= y
            return type(self)(x, label)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            y = self.copy()
            y.x -= other
            return y          
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __rsub__(self, other):
        return -self.__sub__(other)       

    def __div__(self, other):
        if isinstance(other, larry):
            x, y, label = self.__align(other)        
            x /= y
            return type(self)(x, label)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            y = self.copy()
            y.x /= other
            return y           
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __rdiv__(self, other):                
        if isinstance(other, larry):
            x, y, label = self.__align(other)        
            x = x / y
            return type(self)(x, label)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            y = self.copy()
            y.x = other / y.x
            return y           
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __mul__(self, other):    
        if isinstance(other, larry):
            x, y, label = self.__align(other)
            np.multiply(x, y, x)
            return type(self)(x, label)
        if np.isscalar(other):
            y = self.copy()
            y.x *= other
            return y
        if isinstance(other, np.ndarray):
            y = self.copy()
            y.x *= other
            return y 
        raise TypeError, 'Input must be scalar, array, or larry.'

    __rmul__ = __mul__

    def __align(self, other):
        if self.ndim != other.ndim:
            msg = 'Binary operation on two larrys with different dimension'
            raise IndexError, msg
        idxs = []
        idxo = []
        label = []
        shape = []  
        for ls, lo  in zip(self.copylabel(), other.label):
            if ls == lo:
                lab = ls
                ids = range(len(lab))
                ido = ids 
            else:    
                lab = list(frozenset(ls) & frozenset(lo))
                if len(lab) == 0:
                    raise IndexError, 'A dimension has no matching labels'
                lab.sort()
                ids = [ls.index(i) for i in lab]
                ido = [lo.index(i) for i in lab]
            label.append(lab)    
            idxs.append(ids)
            idxo.append(ido)
            shape.append(len(lab))
        shape = tuple(shape)            
        x = np.zeros(shape)               
        x += self.x[np.ix_(*idxs)]
        y = other.x[np.ix_(*idxo)]
        return x, y, label                     

    # Reduce functions -------------------------------------------------------   
        
    def sum(self, axis=None):
        """Sum of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to sum along or sum over all (None, default).
            
        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        return self.__reduce(axis, np.nansum)    

    def mean(self, axis=None):
        """Mean of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the mean along (integer) or the global mean (None,
            default).

        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        return self.__reduce(axis, nanmean) 

    def median(self, axis=None):
        """Median of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the median along (0 or 1) or the global median (None,
            default).
            
        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        return self.__reduce(axis, nanmedian) 
            
    def std(self, axis=None):
        """Standard deviation of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the standard deviation along (integer) or the global
            standard deviation (None, default).

        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        return self.__reduce(axis, nanstd)  
        
    def var(self, axis=None):
        """Variance of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the variance along (integer) or the global variance
            (None, default).

        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        y = self.__reduce(axis, nanstd)
        if np.isscalar(y):
            y *= y 
        else:       
            np.multiply(y.x, y.x, y.x)
        return y                 
                            
    def max(self, axis=None):
        """Maximum values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the max along (integer) or the global max (None,
            default).
            
        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """            
        return self.__reduce(axis, np.nanmax)             
           
    def min(self, axis=None):
        """Minimum values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the min along (integer) or the global min (None,
            default).
            
        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.
                    
        """
        return self.__reduce(axis, np.nanmin)  

    def __reduce(self, axis, op):
        if np.isscalar(axis):
            x = op(self.x, axis)
            if np.isscalar(x):
                return x
            else:    
                label = self.copylabel()
                label.pop(axis)
                return type(self)(x, label)      
        elif axis is None:
            return op(self.x, axis)
        else:
            raise ValueError, 'axis should be an integer or None'
                        
    def lastrank(self):
        """Rank of elements in last column, ignoring NaNs.
        
        Note: Only works on 2d larrys.
            
        Returns
        -------
        d : larry
            A 2d larry is returned.
            
        Raises
        ------
        ValueError
            If larry is not 2d.    
                    
        """
        self._2donly()
        label = self.copylabel()
        label[1] = [label[1][-1]]
        x = lastrank(self.x)
        return type(self)(x, label)
        
    def lastrank_decay(self, decay):
        """Exponentially decayed rank of elements in last column, ignoring NaNs.
        
        Note: Only works on 2d larrys.        

        Parameters
        ----------
        decay : scalar
            Exponential decay strength. Should not be negative.
            
        Returns
        -------
        d : larry
            A 2d larry is returned.
            
        Raises
        ------
        ValueError
            If larry is not 2d. 
        AssertionError
            If decay is less than zero.            
                    
        """
        self._2donly()
        label = self.copylabel()
        label[1] = [label[1][-1]]
        x = lastrank_decay(self.x, decay)
        return type(self)(x, label)                                 
        
    # Comparision ------------------------------------------------------------                                              
        
    def __eq__(self, other):
        return self.__compare(other, '==')                   

    def __ne__(self, other):
        return self.__compare(other, '!=')                       

    def __lt__(self, other):
        return self.__compare(other, '<')  

    def __gt__(self, other):
        return self.__compare(other, '>') 

    def __le__(self, other):
        return self.__compare(other, '<=') 

    def __ge__(self, other):
        return self.__compare(other, '>=') 

    def __compare(self, other, op):         
        if np.isscalar(other) or isinstance(other, np.ndarray):
            y = self.copy()
            if op == '==':
                y.x = y.x == other
            elif op == '!=':
                y.x = y.x != other
            elif op == '<':
                y.x = y.x < other           
            elif op == '>':
                y.x = y.x > other           
            elif op == '<=':
                y.x = y.x <= other           
            elif op == '>=':
                y.x = y.x >= other           
            else:
                raise ValueError, 'Unknown comparison operator'
            return y                                                             
        elif isinstance(other, larry):
            x, y, label = self.__align(other)
            if op == '==':
                x = x == y
            elif op == '!=':
                x = x != y
            elif op == '<':
                x = x < y
            elif op == '>':
                x = x > y            
            elif op == '<=':
                x = x <= y            
            elif op == '>=':
                x = x >= y                              
            else:
                raise ValueError, 'Unknown comparison operator'              
            return type(self)(x, label)
        else:
            raise TypeError, 'Input must be scalar, numpy array, or larry.'

    # Any, all ---------------------------------------------------------------           
                
    def any(self, axis=None):
        """Return true if any elements of x is true.
        
        Note: NaN is True since it is not equal to 0.
        
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : {True, False}
        
        """
        if axis is None:
            return self.x.any()
        else:
            return self.__reduce(axis, np.any)

        
    def all(self, axis=None):
        """Return true if all elements of x are true.
        
        Note: NaN is True since it is not equal to 0.        
        
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : {True, False}
        
        """
        if axis is None:
            return self.x.all()
        else:
            return self.__reduce(axis, np.all)

    # Get and set ------------------------------------------------------------
    
    def __getitem__(self, index):
        if type(index) in (int, np.int, np.int16, np.int32, np.int64):      
            if index >= self.shape[0]:
                raise IndexError, 'index out of range'
            label = self.label[1:]
            x = self.x[index]                     
        elif type(index) is tuple:
            label = []
            for i in xrange(self.ndim):
                if i < len(index):
                    idx = index[i]
                    typ = type(idx)
                    if typ is int:
                        if idx >= self.shape[i]:
                            raise IndexError, 'index out of range'
                        lab = None
                    elif typ is list or typ is tuple:
                        try:
                            lab = [self.label[i][z] for z in idx]
                        except IndexError:
                            raise IndexError, 'index out of range' 
                        lab = list(lab)                              
                    elif typ is np.ndarray:
                        try:
                            lab = [self.label[i][z] for z in idx]
                        except IndexError:
                            raise IndexError, 'index out of range' 
                        lab = list(lab)                   
                    elif typ is np.matrix:
                        msg = 'matrix indexing not supported, '
                        msg = msg + 'use 1d array instead'    
                        raise IndexError, msg                           
                    elif typ is slice:
                        lab = self.label[i][idx]
                    else:
                        msg = 'I do not recognize the way you are indexing'
                        raise IndexError, msg                       
                else:
                    lab = self.label[i]
                if lab:     
                    label.append(lab)              
            x = self.x[index]
        elif type(index) is slice:
            label = list(self.label)
            label[0] = label[0][index]
            x = self.x[index]                  
        else:        
            msg = 'Only slice, integer, and seq (list, tuple, 1d array)'
            msg = msg + ' indexing supported'
            raise IndexError, msg        
        if np.isscalar(x):
            return x                                
        return type(self)(x, label)
        
    def __setitem__(self, index, value):
        if isinstance(index, larry):
            if self.label == index.label:
                self.x[index.x] = value
            else:
                # Could use morph to do this, if every row and column of self
                # is in index, but I think it is better to raise an IndexError
                msg = 'Indexing with a larry that is not aligned'
                raise IndexError, msg    
        else:
            self.x[index] = value
            
    def set(self, label, value):
        """Set one x element given a list of label names.
        
        Give one label name (not label index) for each dimension.
        
        Parameters
        ----------
        label : {list, tuple}
            List or tuple of one label name for each dimension. For example,
            for stock ID and date: (411, datetime.date(2004, 1, 15)).
        value : Float, string, etc.
            Value to place in the single cell specified by label.
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the length of label is not equal to the number of dimensions of
            larry.        
        
        """
        if len(label) != self.ndim:
            raise ValueError, 'Must have exactly one label per dimension'
        index = []
        for i in xrange(self.ndim):
            index.append(self.labelindex(label[i], axis=i))    
        self.x[tuple(index)] = value        

    def get(self, label):
        """Get one x element given a list of label names.
        
        Give one label name (not label index) for each dimension.
        
        Parameters
        ----------
        label : {list, tuple}
            List or tuple of one label name for each dimension. For example,
            for stock ID and date: (411, datetime.date(2004, 1, 15)).
            
        Returns
        -------
        out : Float, string, etc.
            Value of the single cell specified by label.
        
        Raises
        ------
        ValueError
            If the length of label is not equal to the number of dimensions of
            larry.        
        
        """    
        if len(label) != self.ndim:
            raise ValueError, 'Must have exactly one label per dimension'
        index = []
        for i in xrange(self.ndim):
            index.append(self.labelindex(label[i], axis=i))    
        return self.x[tuple(index)]
                
    def getx(self, copy=True):
        """Return a copy of the x data or a reference to it.
        
        Parameters
        ----------
        copy : {True, False}
            Return a copy (True) of the x values or a reference (False) to x.
            
        Returns
        -------
        out : array
            Copy or reference of x array.
                
        """
        if copy:
            return self.x.copy()
        else:
            return self.x
        
    def fill(self, fill_value):
        """Inplace filling of data array with specified value.
        
        Parameters
        ----------
        fill_value : {scalar, string, etc}
            Value to replace every element of the data array.
            
        Returns
        -------
        out : None
                
        """
        self.x.fill(fill_value)
                    
    # label operations -------------------------------------------------------
        
    def maxlabel(self, axis=None):
        "Max label value"
        if axis is None:
            return max([max(z) for z in self.label])
        else:
            return max([z for z in self.label[axis]])

    def minlabel(self, axis=None):
        "Min label value"
        if axis is None:
            return min([min(z) for z in self.label])
        else:
            return min([z for z in self.label[axis]])
        
    def getlabel(self, axis, copy=True):
        "Return label for specificed dimension." 
        if axis >= self.ndim:
            raise IndexError, 'axis out of range'
        label = self.label[axis]    
        if copy:
            label =  list(label)
        return label
        
    def labelindex(self, name, axis, exact=True):
        """Return index of given label element along specified axis.
        
        Parameters
        ----------
        name : str, datetime.date, int, etc.
            Name of label element to index.
        axis : int
            Axis to index along. Cannot be None.
        exact : bool, optional
            If an exact match is specfied (default) then an IndexError is
            raised if an exact match cannot be found. If exact match is False
            and if a perfect match is not found then the index of the nearest
            label is returned. Nearest is defined as the closest that is equal
            or smaller.
            
        Returns
        -------
        idx : int
            Index of given label element.
                        
        """
        if axis >= self.ndim:
            raise IndexError, 'axis out of range'
        if axis is None:
            raise ValueError, 'axis cannot be None'            
        try:
            index = self.label[axis].index(name)
        except ValueError:
            if exact:
                raise IndexError, 'name not in label along axis %d' % axis
            else:
                idx = [i for i, z in enumerate(self.label[axis]) if z <= name]
                if len(idx) == 0:
                    raise IndexError, 'name not in label along axis %d' % axis
                index = max(idx)                        
        return index            
            
    # Calc -------------------------------------------------------------------                                            

    def demean(self, axis=None):
        "Demean values along specified axis."
        # Adapted from pylab.demean
        y = self.copy()        
        if axis:
            ind = [slice(None)] * y.ndim
            ind[axis] = np.newaxis
            y.x -= nanmean(y.x, axis)[ind]
        else:
            y.x -= nanmean(y.x, axis)   
        return y

    def demedian(self, axis=None):
        "Demean values (using median) along specified axis."
        # Adapted from pylab.demean
        y = self.copy()
        if axis:
            ind = [slice(None)] * axis
            ind.append(np.newaxis)
            y.x -= nanmedian(y.x, axis)[ind]
        else:
            y.x -= nanmedian(y.x, axis)   
        return y
        
    def zscore(self, axis=None):
        "Zscore along specified axis."
        y = self.demean(axis)
        if axis:
            ind = [slice(None)] * axis
            ind.append(np.newaxis)
            y.x /= nanstd(y.x, axis)[ind]
        else:
            y.x /= nanstd(y.x, axis)   
        return y              
        
    def push(self, window):
        """Fill missing values (NaNs) with most recent non-missing values if
        recent, where recent is defined by the window. The filling proceeds
        from left to right along each row.
        """
        self._2donly()
        y = self.copy()
        y.x = fillforward_partially(y.x, window)
        return y
            
    def movingsum(self, window, axis=-1, norm=False):
        """Moving sum, NaNs treated as 0, optionally normalized for NaNs."""
        y = self.copy()
        #y.label[axis] = y.label[axis][window-1:]  #add dropped init obs back in
        y.x = movingsum(y.x, window, axis, norm)
        return y 
        
    def movingsum_forward(self, window, skip=0, axis=1, norm=False):    
        """Movingsum in the forward direction skipping skip dates"""
        self._2donly()        
        y = self.copy()
        y.x = movingsum_forward(y.x, window, skip, axis, norm)
        return y
                         
    def ranking(self, axis=0):
        """Ranking along axis.
        
        Output is between -1 and 1, ties are broken, and NaNs are handled.
        """
        #XXX: why is default axis=0 and not 1 as usual
        self._2donly()
        y = self.copy()
        y.x = ranking(y.x, axis)
        return y
                            
    def movingrank(self, window, axis=1):
        """Moving rank (normalized to -1 and 1) of a given window along axis.

        Normalized for missing (NaN) data.
        A data point with NaN data is returned as NaN
        If a window is all NaNs except last, this is returned as NaN
        """
        self._2donly()
        y = self.copy()
        y.x = movingrank(y.x, window, axis)
        return y
        
    def quantile(self, q):
        """Convert elements in each column to integers between 1 and q; then
        normalize to to -1, 1
        """
        self._2donly()
        y = self.copy()
        y.x = quantile(y.x, q)       
        return y
               
    def cut_missing(self, fraction, axis):
        """Cut rows and columns that contain too many NaNs.
        
        Note: Only works on 2d larrys. 
        
        Parameters
        ----------
        fraction : scalar
            Usually a float that give the minimum allowable fraction of missing
            data before the row or column is cut.
        axis : {0, 1}
            Look for missing data along this axis. So for axis=0, the missing
            data along columns are checked and columns are cut. For axis=1, the
            missing data along rows are checked and rows are cut.
            
        Returns
        -------
        out : larry
            Returns a copy with rows or columns with lots of missing data cut.                
        
        Raises
        ------
        ValueError
            If larry is not 2d.        
        IndexError
            If axis is not 0 or 1.
            
        """
        self._2donly()
        if axis not in (0,1):
            raise IndexError, 'axis is out of range'
        threshold = (1.0 - fraction) * self.shape[axis]
        idx = np.isfinite(self.x).sum(axis) > threshold
        if idx.all():
            return self.copy()
        else: 
            idx = np.where(idx)[0]   
            index = [idx, idx]
            index[axis] = slice(None) 
            y = self.copy()
            if axis == 0:
                y.label[1] = [y.label[1][j] for j in idx]
            else:
                y.label[0] = [y.label[0][j] for j in idx]
            y.x = y.x[index]
            return y   
     
    def cov(self):
        """Covariance matrix adjusted for missing (NaN) values.
        
        Note: Only works on 2d larrys.
        
        The mean of each row is assumed to be zero. So rows are not demeaned
        and therefore the covariance is normalized by the number of columns,
        not by the number of columns minus 1.        
        
        Parameters
        ----------
        No input.
        
        Returns
        -------
        out : larry
            Returns NxN covariance matrix where N is the number of rows.

        """
        self._2donly()       
        y = self.copy()
        y.label[1] = list(y.label[0])
        y.x = covMissing(y.x)
        return y         
        
    def keep_label(self, op, value, axis):
        """Keep labels (and corresponding values) that satisfy conditon.
        
        Keep labels that satify:
        
                            label[axis] op value,
                       
        where op can be '==', '>', '<', '>=', '<=', '!=', 'in', 'not in'.               
        
        Parameters
        ----------
        op : string
            Operation to perform. op can be '==', '>', '<', '>=', '<=', '!=',
            'in', 'not in'.
        value : anything that can be compared to labels
            Usually the same type as the labels. So if the labels are integers
            then value is an integer.
        axis : integer
            axis over which to test condiction.
        
        Returns
        -------
        out : larry
            Returns a copy with only the labels and corresponding values that
            satisfy the specified condition.
            
        Raises
        ------
        ValueError
            If op is unknown or if axis is None.
        IndexError
            If axis is out of range.        

        """
        ops = ('==', '>', '<', '>=', '<=', '!=', 'in', 'not in')
        if op not in ops:
            raise ValueError, 'Unknown op'
        if axis is None:
            raise ValueError, 'axis cannot be None'    
        if axis >= self.ndim:
            raise IndexError, 'axis is out of range'   
        y = self.copy()
        cmd = '[(idx, z) for idx, z in enumerate(y.label[axis]) if z '
        cmd = cmd + op + ' value]'  
        idxlabel = eval(cmd)
        idx, label = zip(*idxlabel)
        y.label[axis] = list(label)
        index = [slice(None,None,None)] * self.ndim
        index[axis] = list(idx)
        y.x = y.x[index]
        return y
        
    def keep_x(self, op, value, vacuum=True):
        """Keep labels (and corresponding values) that satisfy conditon.
        
        Set x values that do not satify the condition to NaN. Then, if
        vacuum is True, rows and columns with all NaNs will be removed. If
        vacuum is True, larry must be 2d. 
        
        Note that when vacuum is True, all rows and columns with all NaNs
        (even if they already had all NaNs in the row or column before this
        function was called) will be removed.
        
        The op can be '==', '>', '<', '>=', '<=', '!=', 'in', 'not in'.
        
        Parameters
        ----------
        op : string
            Operation to perform. op can be '==', '>', '<', '>=', '<=', '!=',
            'in', 'not in'.
        value : anything that can be compared to labels
            Usually the same type as the labels. So if the labels are integers
            then value is an integer.
        vacuum : {True, False}, optional
            Vacuum larry after conditionally setting data values to NaN. False
            is the default.
        
        Returns
        -------
        out : larry
            Returns a copy with only the labels and corresponding values that
            satisfy the specified condition.
            
        Raises
        ------
        ValueError
            If op is unknown or if axis is None.
        IndexError
            If axis is out of range.        

        """
        if (vacuum == True) and (self.ndim != 2):
            raise ValueError, 'When vacuum is True, larry must be 2d'
        ops = ('==', '>', '<', '>=', '<=', '!=', 'in', 'not in')
        if op not in ops:
            raise ValueError, 'Unknown op'   
        y = self.copy()
        idx = eval('y.x ' + op + 'value')
        y.x[~idx] = np.nan
        if vacuum:
            y = y.vacuum()
        return y                                                                
        
    # Group calc -------------------------------------------------------------  
                 
    def group_ranking(self, group):
        """Group (e.g. sector) ranking along columns.
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """ 
        self._2donly()
        y = self.copy()
        aligned_group_list = y._group_align(group)
        y.x = sector_rank(y.x, aligned_group_list)
        return y
            
    def group_mean(self, group):
        """Group (e.g. sector) mean along columns.
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """ 
        self._2donly()  
        y = self.copy() 
        aligned_group_list = y._group_align(group)
        y.x = sector_mean(y.x, aligned_group_list)                                         
        return y
        
    def group_median(self, group):
        """Group (e.g. sector) median along columns.
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """ 
        self._2donly()
        y = self.copy()
        aligned_group_list = y._group_align(group)   
        y.x = sector_median(y.x, aligned_group_list)
        return y
                
    def _group_align(self, group):
        """Return a row aligned group list (e.g. sector list) of values.
        
        group must have exactly one column. The row labels of the object must
        be a subset of the row labels of the group.
        """
        if not isinstance(group, larry):
            raise TypeError, 'group must be a larry'
        if group.ndim != 1:
            raise ValueError, 'group must be a 1d larry'
        if len(frozenset(self.label[0]) - frozenset(group.label[0])):
            raise IndexError, 'label is not a subset of group label'
        g = group.morph(self.label[0], 0)
        g = g.x.tolist()
        return g                                                             

    # Alignment --------------------------------------------------------------

    def morph(self, label, axis):
        """Change ordering of larry along specified axis.
        
        Supply a list with the ordering you would like. If some elements in the
        list do not exist in the larry, NaNs will be used.
        
        Since NaNs are used to mark missing data, if the input larry uses
        integers then the output will uses floats.
        
        Parameters
        ----------
        label : list
            Desired ordering of elements along specified axis.
        axis : {None, integer}, optional
            axis along which to perform the reordering.
        
        Returns
        -------
        out : larry
            A reordered copy.       
        
        Raises
        ------
        IndexError
            axis is out of range.
            
        """
        if axis >= self.ndim:
            raise IndexError, 'axis out of range'
        label = list(label)    
        shape = list(self.shape)
        shape[axis] = len(label)
        if self.dtype == object:
            x = np.empty(shape, dtype=object)
            x.fill(None)
        else:
            x = nans(shape)       
        idx0 = [z for z in self.label[axis] if z in label]
        idx1 = [label.index(i) for i in idx0]
        idx2 = [self.label[axis].index(i) for i in idx0]
        index1 = [slice(None)] * self.ndim
        index1[axis] = idx1
        index2 = [slice(None)] * self.ndim
        index2[axis] = idx2        
        x[index1] = self.x[index2]
        lab = self.copylabel()
        lab[axis] = label
        return type(self)(x, lab)
        
    def morph_like(self, lar):
        """Morph to line up with the specified larry.

        If some elements in the list do not exist in the larry, NaNs will be
        used.
        
        Since NaNs are used to mark missing data, if the input larry uses
        integers then the output will uses floats.
        
        Parameters
        ---------- 
        lar : larry
            The target larry to align to.
            
        Returns
        -------
        lar : larry
            A morphed larry that is aligned with the input larry.
            

        Raises
        ------
        IndexError
            If the larrys are not of the same dimension.                              
        
        """
        if self.ndim != lar.ndim:
            raise IndexError, 'larrys must be of the same dimension.'
        y = self.copy()      
        for i in range(self.ndim):
            y = y.morph(lar.getlabel(i), axis=i)     
        return y        

    def merge(self, other, update=False):
        '''
        Merge, or optionally update, a larry with a second larry.
     
        Parameters
        ----------
        other : larry
            The larry to merge or to use to update the values. It must have the
            same number of dimensions as as the existing larry.
        update : bool
            Raise a ValueError (default) if there is any overlap in the two
            larrys. An overlap is defined as a common label in both larrys that
            contains a finite value in both larrys. If `update` is True then the
            overlapped values in the current larry will be overwritten with the
            values in `other`.            
     
        Returns
        -------
        lar1 : larry
            The merged larry.
     
        '''

        ndim = self.ndim
        if ndim != other.ndim:
            raise IndexError, 'larrys must be of the same dimension.'
        lar1 = self
        lar2 = other
        for ax in range(ndim):
            mergelabel = sorted(set(lar1.label[ax]).union(set(lar2.label[ax])))
            lar1 = lar1.morph(mergelabel, ax)
            lar2 = lar2.morph(mergelabel, ax)
     
        # Check for overlap if requested
        if (not update) and (np.isfinite(lar1.x)*np.isfinite(lar2.x)).any():
            raise ValueError('overlapping values')
        else:
            mask = np.isfinite(lar2.x)
            lar1.x[mask] = lar2.x[mask]
     
        return lar1
     
    def vacuum(self, axis=None):
        """Remove all rows and/or columns that contain all NaNs.
              
        
        Parameters
        ----------
        axis : None or int or tuple of int
            Remove columns (0) or rows (1) or both (None, default) that contain
            no finite values, for nd arrays see Notes.
            
        Returns
        -------
        out : larry
            Return a copy with rows and/or columns removed that contain all
            NaNs.  
            
        Notes
        -----
        
        For nd arrays, axis can also be a tuple. In this case, all other 
        axes are checked for nans. If the corresponding slice of the array
        contains only nans then the slice is removed.
                
        """        

        y = self.copy()
        ndim = y.ndim
        
        if axis is None:
            axes = range(ndim)
        elif not hasattr(axis, '__iter__'):
            axes = [axis]
        else:
            axes = axis
        
        idxsl = []
        labsnew = []
        for ax in range(ndim):
            sl = [None]*ndim
            if ax not in axes:
                labsnew.append(y.label[ax])
                sl[ax] = slice(None)
                idxsl.append(np.arange(y.shape[ax])[sl])
                continue
            
            # find all nans over all other axes
            xtmp = np.rollaxis(np.isfinite(y.x), ax, 0)
            for _ in range(ndim-1):
                xtmp = xtmp.any(-1)
    
            labsnew.append([y.label[ax][ii] for ii in np.nonzero(xtmp)[0]])
            sl[ax] = slice(None)
            idxsl.append(np.nonzero(xtmp)[0][sl])
        
        y.x = y.x[idxsl]
        y.label = labsnew
        return y


    def vacuum_old(self, axis=None):
        """Remove all rows and/or columns that contain all NaNs.
        
        Note: Only works on 2d larrys.        
        
        Parameters
        ----------
        axis : {None, 0, 1}
            Remove columns (0) or rows (1) or both (None, default) that contain
            no finite values.
            
        Returns
        -------
        out : larry
            Return a copy with rows and/or columns removed that contain all
            NaNs.  
            
        Raises
        ------
        AssertionError
            If axis is not None, 0, or 1.
                
        """
        self._2donly()
        assert axis in (None, 0, 1)
        y = self.copy()
        if axis in (None, 1):
            idx = np.isfinite(y.x).sum(1) != 0
            idx = np.where(idx)[0]
            y.x = y.x[idx,:]
            y.label[0] = [y.label[0][i] for i in idx]
        if axis in (None, 0):
            idx = np.isfinite(y.x).sum(0) != 0
            idx = np.where(idx)[0]
            y.x = y.x[:,idx]
            y.label[1] = [y.label[1][i] for i in idx]
        return y
        
    def squeeze(self):
        """Eliminate all length-1 dimensions and corresponding labels.
        
        Note that a view (reference) is returned, not a copy.
        
        Parameters
        ----------
        No input
        
        Returns
        -------
        out : larry
            Returns a view with all length-1 dimensions and corresponding
            labels removed. 
        
        """
        idx = [i for i, z in enumerate(self.shape) if z != 1]
        label = []
        for i in idx:
            label.append(self.label[i])    
        x = self.x.squeeze()
        return type(self)(x, label)
        
    def pull(self, name, axis):
        """Pull out the values for a given label name along a specified axis.
        
        A view is returned and the dimension is reduced by one.
        
        Parameters
        ----------
        name : scalar, string, etc.
            Label name.
        axis : integer
            The axis the label name is in.
            
        Returns
        -------
        out : {view of larry, scalar}
            A view of the larry with the dimension reduced by one is returned
            unless the larry is alread 1d, then a scalar is returned.
            
        Raises
        ------
        ValueError
            If the axis is None.
            
        Example
        -------
        Say you have a 3d larry called indicator with indicators along axis 0.
        To get a 2d view of the indicator momentum:
        
                        mom = momentum.pull('indicator', 0)    
                        
        """
        if axis is None:
            raise ValueError, 'axis cannot be None'
        label = [z for i, z in enumerate(self.label) if i != axis]    
        idx = self.labelindex(name, axis)
        index = [slice(None)] * self.ndim 
        index[axis] = idx
        x = self.x[index]
        if x.shape == (1,):
            return x[0]
        return type(self)(x, label)

    def lag(self, nlag, axis=-1):
        """Lag the values along the specified axis.
        
        Parameters
        ----------
        nlag : int
            Number of periods (rows, columns, etc) to lag.
        axis : int
            The axis to lag along. The default is -1.
            
        Returns
        -------
        out : larry
            A lagged larry is returned.
            
        Raises
        ------
        ValueError
            If nlag < 0.        
        IndexError
            If the axis is None.   
                        
        """
        if axis is None:
            raise IndexError, 'axis cannot be None.'
        if nlag < 0:
            raise ValueError, 'nlag cannot be negative'
        y = self.copy()
        y.label[axis] = y.label[axis][nlag:]
        index = [slice(None)] * self.ndim
        index[axis] = slice(0,-nlag)            
        y.x = y.x[index]
        return y                         

    # Size -------------------------------------------------------------------

    @property
    def nx(self):
        return np.isfinite(self.x).sum()

    @property
    def size(self):
        return self.x.size

    @property
    def shape(self):
        return self.x.shape
        
    @property
    def ndim(self):
        return self.x.ndim
        
    @property
    def dtype(self):
        return self.x.dtype
            
    # Report -----------------------------------------------------------------            
        
    def hist(self, bins=10, align='center', orientation='vertical'):
        "Display histogram. See pylab.hist for description."
        import pylab
        pylab.hist(self.x.reshape(-1), bins=bins, align=align,
                   orientation=orientation)
        pylab.show()
        
    def plot(self, name, axis):
        "Plot"        
        import pylab
        if self.ndim == 1:
            pylab.plot(self.x)
            pylab.hold(True)
            pylab.plot(self.x, 'b.')
        elif self.ndim == 2:
            if axis == 0:      
                pylab.plot(self.x[:,self.labelindex(name, axis)].T)
            elif axis == 1:
                pylab.plot(self.x[self.labelindex(name, axis),:].T)              
        else:
            raise ValueError, 'Only 1d and 2d larrys can currently be plotted'
        pylab.ylabel(str(name))
        pylab.show()
        
    def stat(self):
        "Print some stats"  
        print '%10.4f missing' % (1.0 * (self.size - self.nx) / self.size)
        print '%10.4f min' % self.min()
        print '%10.4f mean' % self.mean()
        print '%10.4f max' % self.max() 
        print '%10.4f positive' % (1.0 * (self.x > 0).sum() / self.nx) 
        print '    ' + str(self.shape)       

    def __repr__(self):

        x = []

        # Labels
        pad = '    '
        for i, label in enumerate(self.label):
            x.append('label_%d\n' % i)
            if len(label) > 10:
                x.append(pad + str(self.label[i][0]) + '\n')
                x.append(pad + str(self.label[i][1]) + '\n')
                x.append(pad + str(self.label[i][2]) + '\n')
                x.append(pad + '...\n')
                x.append(pad + str(self.label[i][-3]) + '\n')
                x.append(pad + str(self.label[i][-2]) + '\n')
                x.append(pad + str(self.label[i][-1]) + '\n')
            else:
                for l in label:
                    x.append(pad + str(l) + '\n')
            x.append('\n')        
        
        # x
        x.append('x\n')
        x.append(repr(self.x))
        return ''.join(x)                        
        
    # IO ---------------------------------------------------------------------               
            
    def save(self, name, commit=True):
        """Save data to database
        
        Here we assume that row = ID and col = date.
        """
        pass
        # I deleted this function (and some others) since it is specific to
        # our project.
               
    # Misc -------------------------------------------------------------------        
          
    def copy(self):
        label = deepcopy(self.label)
        x = self.x.copy()
        return type(self)(x, label)
        
    def copylabel(self):
        return deepcopy(self.label)        

    @property
    def T(self):
        self._2donly()
        y = self.copy()
        y.x = y.x.T
        y.label = [y.label[1], y.label[0]]
        return y
        
    def _2donly(self):
        "Only works on 2d arrays"
        if self.ndim != 2:
            raise ValueError, 'This function only works on 2d larrys'
            
    def isnan(self):
        label = self.copylabel()
        x = np.isnan(self.x)
        return type(self)(x, label)                             

    def isfinite(self):
        label = self.copylabel()
        x = np.isfinite(self.x)
        return type(self)(x, label)
        
