"Labeled array class"

import csv

import numpy as np
import bottleneck as bn

from la.missing import ismissing, missing_marker, nans  
from la.flabel import listmap, listmap_fill, flattenlabel
from la.util.misc import isscalar, fromlists
from la.farray import (group_ranking, group_mean, group_median, shuffle,
                       push, quantile, ranking, lastrank, movingsum_forward,
                       movingsum, geometric_mean, demean, demedian, zscore)
from la.farray import (move_nanmedian, move_nanranking, move_func)


class larry(object):
    "Labeled array"

    def __init__(self, x, label=None, dtype=None, validate=True):
        """
        Meet larry, he's a labeled array.
        
        Parameters
        ----------
        x : numpy array_like
            Data, NaN are treated as missing data.
        label : {list of lists, None}, optional
            A list with labels for each dimension of x. If x is 2d, for
            example, then label should be a list that contains two lists, one
            for the row labels and one for the column labels. If x is 1d label
            should be a list that contain one list of names. If label is None
            (default) integers will be used to label the the row, columns,
            etc.
        dtype : data-type, optional
            The desired data type of the larry.         
        validate : bool, optional
            Check the integrity of the larry by checking that the dimensions
            of the data match the dimension of the label, that the labels are
            unique along each axis, and so on. This check adds time to the
            creation of a larry. The default is the check the integrity.
                        
        Raises
        ------
        ValueError
            If x cannot be converted to a numpy array, or if the number of
            elements in label does not match the dimensions of x, or if the
            elements in label are not unique along each dimension, or if the
            elements of label are not lists, or if the number of dimensions
            is zero.
            
        Notes
        -----
        larry does not copy the data array if it is a Numpy array or if
        np.asarray() does not make a copy such as when the data array is a
        Numpy matrix. However, if you change the dtype of the data array, a
        copy is made. Similarly the label is not copied.            
            
        Examples
        --------
        The labels default to range(n):
        
        >>> larry([6, 7, 8])
        label_0
            0
            1
            2
        x
        array([6, 7, 8])

        A more formal way to make a larry:

        >>> import numpy as np
        >>> x = np.array([1, 2, 3])
        >>> label = [['one', 'two', 'three']]
        >>> larry(x, label)
        label_0
            one
            two
            three
        x
        array([1, 2, 3])
        
        The dtype can be specified:
        
        >>> larry([0, 1, 2], dtype=bool)
        label_0
            0
            1
            2
        x
        array([False,  True,  True], dtype=bool)            

        """
        if type(x) is not np.ndarray:
            # The if statement above is faster than asarray. When you add two
            # larrys, for example, it will use __init__ to create the result,
            # so any improvement in speed is good. That's why this entire
            # block of code is not replaced with x = np.asarray(x)
            try:
                x = np.asarray(x, dtype=dtype)
            except:
                raise ValueError, "x must be array_like."
        elif dtype != None:
            x = x.astype(dtype)            
        if label is None:
            label = [range(z) for z in x.shape]
        if validate: 
            ndim = x.ndim
            if ndim != len(label):
                raise ValueError, 'Exactly one label per dimension needed'
            if ndim == 0:
                # A 0d larry can be created if you comment out this ndim == 0
                # test. The reason that 0d is not allowed is that not all
                # methods support 0d larrys
                raise ValueError, '0d larrys are not supported'
            for i, l in enumerate(label):
                nlabel = len(l)
                if x.shape[i] != nlabel:
                    msg = 'Length mismatch in label and x along axis %d'
                    raise ValueError, msg % i
                if len(frozenset(l)) != nlabel:
                    # We have duplicates in the label, give an example
                    count = {}
                    for li in l:
                        count[li] = count.get(li, 0) + 1
                    for key, value in count.iteritems():
                        if value > 1:
                            break 
                    msg = "Elements of label not unique along axis %d. "
                    msg += "There are %d labels named `%s`."          
                    raise ValueError, msg % (i, value, key)
                if type(l) is not list:
                    raise ValueError, 'label must be a list of lists'          
        self.x = x
        self.label = label

    # Unary functions --------------------------------------------------------  

    def log(self):
        """
        Element by element base e logarithm.
        
        Returns
        -------
        out : larry
            Returns a copy with log of x values.
        
        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.log()
        label_0
            0
            1
            2
        x
        array([ 0.        ,  0.69314718,  1.09861229])
        >>>
        
        """
        x = np.log(self.x)
        label = self.copylabel()
        return larry(x, label, validate=False) 

    def exp(self):
        """
        Element by element exponential.
        
        Returns
        -------
        out : larry
            Returns a copy with exp of x values.

        Examples
        --------            
        >>> y = larry([1, 2, 3])
        >>> y.exp()
        label_0
            0
            1
            2
        x
        array([  2.71828183,   7.3890561 ,  20.08553692])            
                
        """
        x = np.exp(self.x)
        label = self.copylabel()
        return larry(x, label, validate=False)
        
    def sqrt(self):
        """
        Element by element square root.
        
        Returns
        -------
        out : larry
            Returns a copy with square root of x values.

        Examples
        --------            
        >>> y = larry([1, 4, 9])
        >>> y.sqrt()
        label_0
            0
            1
            2
        x
        array([ 1.,  2.,  3.])
                
        """
        x = np.sqrt(self.x)
        label = self.copylabel()
        return larry(x, label, validate=False)

    def sign(self):
        """
        Element by element sign of the element.
        
        Returns -1 if x < 0; 0 if x == 0, and 1 if x > 0.
        
        Returns
        -------
        out : larry
            Returns a copy with the sign of the values.
            
        Examples
        --------
        >>> y = larry([-1, 2, -3, 4])
        >>> y.sign()
        label_0
            0
            1
            2
            3
        x
        array([-1,  1, -1,  1])
                
        """
        x = np.sign(self.x)
        label = self.copylabel()
        return larry(x, label, validate=False)
        
    def power(self, q):               
        """
        Element by element x**q.
                
        Parameters
        ----------
        q : scalar
            The power to raise to.
        
        Returns
        -------
        out : larry
            Returns a copy with x values to the qth power.

        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.power(2)
        label_0
            0
            1
            2
        x
        array([1, 4, 9])
                
        """
        x = np.power(self.x, q)
        label = self.copylabel()
        return larry(x, label, validate=False)
        
    def __pow__(self, q):
        """
        Element by element x**q.
                
        Parameters
        ----------
        q : scalar
            The power to raise to.
        
        Returns
        -------
        out : larry
            Returns a copy with x values to the qth power.
                
        """
        return self.power(q)           
        
    def cumsum(self, axis): 
        """
        Cumulative sum, ignoring NaNs.
        
        Parameters
        ----------
        axis : int
            axis to cumsum along, no default. None is not allowed.
            
        Returns
        -------
        out : larry
            Returns a copy with cumsum along axis.  
            
        Raises
        ------
        ValueError
            If axis is None.
            
        See Also
        --------
        la.larry.sum: Sum of values along axis, ignoring NaNs.
        
        Notes
        -----
        NaNs are ignored except for when all elements in a cumsum are NaN.
        In that case, a NaN is returned.
            
        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.cumsum(axis=0)
        label_0
            0
            1
            2
        x
        array([1, 3, 6])
        
        If all elements are NaN along the specified axis then NaN is returned:
        
        >>> from la import nan
        >>> y = larry([[nan, 2], [nan,  4]])
        >>> y.cumsum(axis=0)
        label_0
            0
            1
        label_1
            0
            1
        x
        array([[ NaN,   2.],
               [ NaN,   6.]])
                        
        """
        if axis == None:
            raise ValueError, 'axis cannot be None'
        y = self.copy()
        idx = np.isnan(y.x)
        np.putmask(y.x, idx, 0)
        y.x.cumsum(axis, out=y.x)
        if idx.any():
            np.putmask(y.x, idx, np.nan)
        return y        

    def cumprod(self, axis): 
        """
        Cumulative product, ignoring NaNs.
        
        Parameters
        ----------
        axis : int
            axis to find the cumulative product along, no default. None is
            not allowed.
            
        Returns
        -------
        out : larry
            Returns a copy with cumprod along axis.  
            
        Raises
        ------
        ValueError
            If axis is None.
            
        See Also
        --------
        la.larry.prod: Product of values along axis, ignoring NaNs.
        
        Notes
        -----
        NaNs are ignored except for when all elements in a cumprod are NaN.
        In that case, a NaN is returned.            
            
        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.cumprod(axis=0)
        label_0
            0
            1
            2
        x
        array([1, 2, 6])
        
        If all elements are NaN along the specified axis then NaN is returned:
        
        >>> from la import nan
        >>> y = larry([[nan, 2], [nan,  4]])
        >>> y.cumprod(axis=0)
        label_0
            0
            1
        label_1
            0
            1
        x
        array([[ NaN,   2.],
               [ NaN,   8.]])                               
                        
        """
        if axis == None:
            raise ValueError, 'axis cannot be None'
        y = self.copy()
        idx = np.isnan(y.x)
        np.putmask(y.x, idx, 1)
        y.x.cumprod(axis, out=y.x)
        if idx.any():
            np.putmask(y.x, idx, np.nan)
        return y

    def clip(self, lo, hi):
        """
        Clip x values.

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
        ValueError
            If `lo` is greater than `hi`.
            
        Examples
        --------
        >>> y = larry([1, 2, 3, 4])
        >>> y.clip(2, 3)
        label_0
            0
            1
            2
            3
        x
        array([2, 2, 3, 3])                    
        
        """
        if lo > hi:
            raise ValueError, 'lo should be less than or equal to hi'
        y = self.copy()
        y.x.clip(lo, hi, y.x)
        return y
        
    def __neg__(self):
        "Return a copy with each element switched with its negative."
        label = self.copylabel()
        x = self.x.__neg__()
        return larry(x, label, validate=False)
    
    def __pos__(self):
        "Return a copy."
        return self.copy()
        
    def abs(self):
        """
        Absolute value of x.
        
        Returns
        -------
        out : larry
            Returns a copy with the absolute values of the x data.

        Examples
        --------
        >>> y = larry([-1, 2, -3, 4])
        >>> y.abs()
        label_0
            0
            1
            2
            3
        x
        array([1, 2, 3, 4])
       
        """
        y = self.copy()
        np.absolute(y.x, y.x)
        return y
        
    def __abs__(self):
        """
        Absolute value of x.
        
        Returns
        -------
        out : larry
            Returns a copy with the absolute values of the x data.
       
        """    
        return self.abs()
        
    def isnan(self):
        """
        Returns a bool larry with NaNs replaced by True, non-NaNs False.

        Returns
        -------
        out : larry
            Returns a copy with NaNs replaced by True, non-NaNs False.
        
        Examples
        --------
        >>> import la
        >>> y = larry([-la.inf, 1.0, la.nan, la.inf])
        >>> y.isnan()
        label_0
            0
            1
            2
            3
        x
        array([False, False,  True, False], dtype=bool)

        """
        label = self.copylabel()
        x = np.isnan(self.x)
        return larry(x, label, validate=False)                             

    def isfinite(self):
        """
        Returns a bool larry with NaNs and Inf replaced by True, others False.

        Returns
        -------
        out : larry
            Returns a copy with NaNs and Inf replaced by True, others False.

        Examples
        --------        
        >>> import la
        >>> y = larry([-la.inf, 1.0, la.nan, la.inf])
        >>> y.isfinite()
        label_0
            0
            1
            2
            3
        x
        array([False,  True, False, False], dtype=bool)
        
        """    
        label = self.copylabel()
        x = np.isfinite(self.x)
        return larry(x, label, validate=False)
        
    def isinf(self):
        """
        Returns a bool larry with -Inf and Inf replaced by True, others False.

        Returns
        -------
        out : larry
            Returns a copy with -Inf and Inf replaced by True, others False.
        
        Examples
        --------
        >>> import la
        >>> y = larry([-la.inf, 1.0, la.nan, la.inf])
        >>> y.isinf()
        label_0
            0
            1
            2
            3
        x
        array([ True, False, False,  True], dtype=bool)
        
        """    
        label = self.copylabel()
        x = np.isinf(self.x)
        return larry(x, label, validate=False) 
        
    def __invert__(self):
        """
        Element by element inverting of True to False and False to True.
        
        Raises
        ------
        TypeError
            If larry does not have bool dtype.
            
        Examples
        --------
        >>> y = larry([True, False])
        >>> ~y
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
            
        """
        return self.invert()
        
    def invert(self):
        """
        Element by element inverting of True to False and False to True.
        
        Raises
        ------
        TypeError
            If larry does not have bool dtype.
            
        Examples
        --------
        >>> y = larry([True, False])
        >>> y.invert()
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
            
        """
        if self.dtype != bool:
            raise TypeError, 'Only larrys with bool dtype can be inverted.'
        return larry(~self.x, self.copylabel(), validate=False)
        
    # Binary Functions ------------------------------------------------------- 
    
    # We need this to take care of radd and rsub when a matrix is on the left-
    # hand side. Without it, the matrix object will use broadcasting, treating
    # larry objects as scalars.
    __array_priority__ = 10                      
        
    def __add__(self, other):
        """
        Sum a larry with another larry, Numpy array, or scalar.
        
        If two larrys are added then the larrys are joined with an inner join
        (i.e., the intersection of the labels).
        
        See Also
        --------
        la.add: Sum of two larrys using given join and fill methods.
        
        Examples
        --------
        >>> larry([1.0, 2.0]) + larry([2.0, 3.0])
        label_0
            0
            1
        x
        array([ 3.,  5.])        
        
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> y1 + y2
        label_0
            b
        x
        array([3])        
        
        """
        if isinstance(other, larry):
            if self.label == other.label:
                x = self.x + other.x
                label = self.copylabel()
                return larry(x, label, validate=False)                        
            else:       
                x, y, label = self.__align(other)
                x = x + y
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            x = self.x + other
            label = self.copylabel()
            return larry(x, label, validate=False)                 
        raise TypeError, 'Input must be scalar, array, or larry.' 
    
    __radd__ = __add__
    
    def __sub__(self, other):
        """
        Subtract a larry from another larry, Numpy array, or scalar.
        
        If two larrys are subtracted then the larrys are joined with an inner
        join (i.e., the intersection of the labels).
        
        See Also
        --------
        la.subtract: Difference of two larrys using given join and fill methods.
        
        Examples
        --------
        >>> larry([1.0, 2.0]) - larry([2.0, 3.0])
        label_0
            0
            1
        x
        array([-1., -1.])        
        
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> y1 - y2
        label_0
            b
        x
        array([1])        
        """   
        if isinstance(other, larry):
            if self.label == other.label:
                x = self.x - other.x
                label = self.copylabel()
                return larry(x, label, validate=False)                          
            else:          
                x, y, label = self.__align(other)        
                x = x - y
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            x = self.x - other
            label = self.copylabel()
            return larry(x, label, validate=False)       
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __rsub__(self, other):
        "Right subtract a larry with a another larry, Numpy array, or scalar."
        return -self.__sub__(other)       

    def __div__(self, other):
        """
        Divide a larry with another larry, Numpy array, or scalar.
        
        If two larrys are divided then the larrys are joined with an inner
        join (i.e., the intersection of the labels).
        
        See Also
        --------
        la.divide: divide two larrys element-wise using given join method.        
        
        Examples
        -------- 
        
        >>> larry([1.0, 2.0]) / larry([2.0, 3.0])
        label_0
            0
            1
        x
        array([ 0.5       ,  0.66666667])        
        
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> y1 / y2
        label_0
            b
        x
        array([2])        
               
        """    
        if isinstance(other, larry):
            if self.label == other.label:
                x = self.x / other.x
                label = self.copylabel()
                return larry(x, label, validate=False)                          
            else:          
                x, y, label = self.__align(other)        
                x = x / y
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            x = self.x / other
            label = self.copylabel()
            return larry(x, label, validate=False)        
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __rdiv__(self, other):
        "Right divide a larry with a another larry, Numpy array, or scalar."
        if isinstance(other, larry):
            msg = 'I could not come up with a problem that used this code '
            msg += 'so I removed it. Send me your example and I will fix.'
            raise RuntimeError, msg                   
        if np.isscalar(other) or isinstance(other, np.ndarray):
            label = self.copylabel()
            x = other / self.x
            return larry(x, label, validate=False)
        raise TypeError, 'Input must be scalar, array, or larry.'
        
    def __mul__(self, other): 
        """
        Multiply a larry with another larry, Numpy array, or scalar.

        If two larrys are multiplied then the larrys are joined with an inner
        join (i.e., the intersection of the labels).
        
        See Also
        --------
        la.multiply: Multiply two larrys element-wise using given join method.
        
        Examples
        --------
        >>> larry([1.0, 2.0]) * larry([2.0, 3.0])
        label_0
            0
            1
        x
        array([ 2.,  6.])        
         
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> y1 * y2
        label_0
            b
        x
        array([2])        
                
        """      
        if isinstance(other, larry):
            if self.label == other.label:
                x = self.x * other.x
                label = self.copylabel()
                return larry(x, label, validate=False)                          
            else:           
                x, y, label = self.__align(other)
                x = x * y
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):
            x = self.x * other
            label = self.copylabel()
            return larry(x, label, validate=False)   
        raise TypeError, 'Input must be scalar, array, or larry.'

    __rmul__ = __mul__

    def __and__(self, other):
        """
        Logical and a larry with another larry, Numpy array, or scalar.
        
        Notes
        -----
        Numpy defines & as bitwise_and; here & is defined as
        numpy.logical_and.
        
        Examples
        --------
        >>> (larry([1.0, 2.0]) > 1) & (larry([2.0, 3.0]) > 1)
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)        
        
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> (y1 > 0) & (y2 > 0)
        label_0
            b
        x
        array([ True], dtype=bool)                 
        
        """    
        if isinstance(other, larry):
            if self.label == other.label:
                x = np.logical_and(self.x, other.x)
                label = self.copylabel()
                return larry(x, label, validate=False)                      
            else:          
                x, y, label = self.__align(other)
                x = np.logical_and(x, y)
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):            
            x = np.logical_and(self.x, other)
            label = self.copylabel()
            return larry(x, label, validate=False)
        raise TypeError, 'Input must be scalar, array, or larry.'

    __rand__ = __and__

    def __or__(self, other):
        """
        Logical or a larry with another larry, Numpy array, or scalar.
        
        Notes
        -----
        Numpy defines | as bitwise_or; here & is defined as
        numpy.logical_or.

        Examples
        --------
        >>> (larry([1.0, 2.0]) > 1) | (larry([2.0, 3.0]) > 1)
        label_0
            0
            1
        x
        array([ True,  True], dtype=bool)        
        
        >>> y1 = larry([1,2], [['a', 'b']])
        >>> y2 = larry([1,2], [['b', 'c']])
        >>> (y1 > 0) | (y2 > 0)
        label_0
            b
        x
        array([ True], dtype=bool)
                        
        """     
        if isinstance(other, larry):
            if self.label == other.label:
                x = np.logical_or(self.x, other.x)
                label = self.copylabel()
                return larry(x, label, validate=False)                      
            else:          
                x, y, label = self.__align(other)
                x = np.logical_or(x, y)
                return larry(x, label, validate=False)
        if np.isscalar(other) or isinstance(other, np.ndarray):            
            x = np.logical_or(self.x, other)
            label = self.copylabel()
            return larry(x, label, validate=False)
        raise TypeError, 'Input must be scalar, array, or larry.'

    __ror__ = __or__

    def __align(self, other):
        "Align larrys for binary operations."
        if self.label == other.label:
            # Labels are already aligned
            x = self.x
            y = other.x
            label = self.copylabel()
        else:  
            # Labels are not aligned.  
            if self.ndim != other.ndim:
                msg = 'Binary operation on two larrys with different dimension'
                raise IndexError, msg
            label = []
            x = self.x
            y = other.x 
            ax = -1           
            for ls, lo in zip(self.copylabel(), other.label):
                ax += 1
                if ls == lo:
                    lab = ls
                else:
                    lab = list(frozenset(ls) & frozenset(lo))
                    lab.sort()
                    ids = listmap(ls, lab)
                    ido = listmap(lo, lab)
                    x = x.take(ids, ax)
                    y = y.take(ido, ax)    
                label.append(lab)
        return x, y, label
                  
    # Reduce functions -------------------------------------------------------
        
    def sum(self, axis=None):
        """
        Sum of values along axis, ignoring NaNs.

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
            
        Examples
        --------
        >>> from la import nan 
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.sum()
        9.0
        >>> y.sum(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  6.])
                    
        """   
        return self.__reduce(bn.nansum, axis=axis)    

    def prod(self, axis=None):
        """
        Product of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis to find the product along or find the product over a
            all axes (None, default).
            
        Returns
        -------
        d : {larry, scalar}
            Returns a larry or a scalar. When axis is None (default) the larry
            is flattened and a scalar, the product of all elements, is
            returned; when larry is 1d and axis=0 a scalar is returned.
            
        See Also
        --------
        la.larry.cumprod: Cumulative product, ignoring NaNs.  

        Notes
        -----
        NaNs are ignored except for when all elements in a product are NaN.
        In that case, a NaN is returned. Also the product of an empty
        larry is NaN.
            
        Examples
        --------
        >>> from la import nan        
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.prod()
        24.0
        >>> y.prod(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  8.])
        
        If all elements are NaN along the specified axis then NaN is returned:
        
        >>> from la import nan 
        >>> y = larry([[nan, 2], [nan,  4]])
        >>> y.prod(axis=0)
        label_0
            0
            1
        x
        array([ NaN,   8.])
                    
        """
        y = self.copy()
        idx = np.isnan(y.x)
        np.putmask(y.x, idx, 1)
        y = y.__reduce(np.prod, axis=axis)
        idx = idx.all(axis)
        if idx.ndim == 0:
            if idx:
                y = np.nan
        else:
            np.putmask(y.x, idx, np.nan)
        return y 

    def mean(self, axis=None):
        """
        Mean of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the mean along (integer) or the global mean (None,
            default).

        Returns
        -------
        d : {larry, scalar}
            The mean.

        Examples
        --------
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.mean()
        3.0
        >>> y.mean(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  3.])       
                    
        """
        return self.__reduce(bn.nanmean, axis=axis)
        
    def geometric_mean(self, axis=None, check_for_greater_than_zero=True):
        """
        Geometric mean of values along axis, ignoring NaNs.

        Parameters
        ----------
        axis : {None, integer}, optional
            Axis to find the geometric mean along (integer) or the global
            geometric mean (None, default).
        check_for_greater_than_zero : bool
            If True (default) an exception is raised if any element is zero or
            less.    

        Returns
        -------
        d : {larry, scalar}
            The geometric mean.

        Examples
        --------
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.geometric_mean()
        2.8844991406148162
        >>> y.geometric_mean(axis=0)
        label_0
            0
            1
        x
        array([ 3.        ,  2.82842712])      
                    
        """
        return self.__reduce(geometric_mean, axis=axis,
                      check_for_greater_than_zero=check_for_greater_than_zero)

    def median(self, axis=None):
        """
        Median of values along axis, ignoring NaNs.

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

        Examples
        --------
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.median()
        3.0
        >>> y.median(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  3.])
                    
        """
        return self.__reduce(bn.nanmedian, axis=axis) 
            
    def std(self, axis=None, ddof=0):
        """
        Standard deviation of values along axis, ignoring NaNs.

        `float64` intermediate and return values are used for integer inputs.

        Instead of a faster one-pass algorithm, a more stable two-pass
        algorithm is used.

        An example of a one-pass algorithm:

            >>> np.sqrt((arr*arr).mean() - arr.mean()**2)
        
        An example of a two-pass algorithm:    
        
            >>> np.sqrt(((arr - arr.mean())**2).mean())

        Note in the two-pass algorithm the mean must be found (first pass)
        before the squared deviation (second pass) can be found.

        Parameters
        ----------
        axis : {int, None}, optional
            Axis along which the standard deviation is computed. The default
            (axis=None) is to compute the standard deviation of the flattened
            array.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.

        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.

        Examples
        -------- 
        >>> from la import nan 
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.std()
        0.81649658092772603
        >>> y.std(axis=0)
        label_0
            0
            1
        x
        array([ 0.,  1.])  
                         
        """      
        return self.__reduce(bn.nanstd, axis=axis, ddof=ddof)  
        
    def var(self, axis=None, ddof=0):
        """
        Variance of values along axis, ignoring NaNs.
        
        `float64` intermediate and return values are used for integer inputs.

        Instead of a faster one-pass algorithm, a more stable two-pass
        algorithm is used.

        An example of a one-pass algorithm:

            >>> np.sqrt((arr*arr).mean() - arr.mean()**2)
        
        An example of a two-pass algorithm:    
        
            >>> np.sqrt(((arr - arr.mean())**2).mean())

        Note in the two-pass algorithm the mean must be found (first pass)
        before the squared deviation (second pass) can be found.

        Parameters
        ----------
        axis : {int, None}, optional
            Axis along which the variance is computed. The default (axis=None)
            is to compute the variance of the flattened array.
        ddof : int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            By default `ddof` is zero.

        Returns
        -------
        d : {larry, scalar}
            When axis is an integer a larry is returned. When axis is None
            (default) a scalar is returned (assuming larry contains scalars).
            
        Raises
        ------
        ValueError
            If axis is not an integer or None.

        Examples
        -------- 
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.var()
        0.66666666666666663
        >>> y.var(axis=0)
        label_0
            0
            1
        x
        array([ 0.,  1.])
                    
        """         
        return self.__reduce(bn.nanvar, axis=axis, ddof=ddof)  
                            
    def max(self, axis=None):
        """
        Maximum values along axis, ignoring NaNs.

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

        Examples
        -------- 
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.max()
        4.0
        >>> y.max(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  4.])
                    
        """            
        return self.__reduce(bn.nanmax, axis=axis)             
           
    def min(self, axis=None):
        """
        Minimum values along axis, ignoring NaNs.

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

        Examples
        -------- 
        >>> from la import nan
        >>> y = larry([[nan, 2], [3,  4]])
        >>> y.min()
        2.0
        >>> y.min(axis=0)
        label_0
            0
            1
        x
        array([ 3.,  2.])
                    
        """
        return self.__reduce(bn.nanmin, axis=axis)  

    def __reduce(self, op, default=np.nan, **kwargs):
        axis = kwargs['axis']
        if self.size == 0:
            # At least one dimension has length 0
            if axis == None:
                return default
            shape = list(self.shape)
            shape.pop(axis)
            if len(shape) == 0:
                return default
            else:
                x = np.empty(shape, dtype=self.dtype)
                x.fill(default)
                label = self.copylabel()
                label.pop(axis)
                return larry(x, label, validate=False)
        if np.isscalar(axis):
            x = op(self.x, **kwargs)
            if np.isscalar(x):
                return x
            else:    
                label = self.copylabel()
                label.pop(axis)
                return larry(x, label, validate=False)      
        elif axis is None:
            return op(self.x, **kwargs)
        else:
            raise ValueError, 'axis should be an integer or None'
        
    def any(self, axis=None):
        """
        True if any element along specified axis is True; False otherwise.

        Parameters
        ----------
        axis : {int, None}, optional
            The axis over which to reduce the truth. By default (axis=None)
            the larry is flattened before the truth is found. 
        
        Returns
        -------
        out : {larry, True, False}
            If `axis` is None then returns True if any data element of the
            larry (not including the label) is True; False otherwise. If
            `axis` is an integer then a bool larry is returned.
            
        Notes
        -----
        As in Numpy, NaN is True since it is not equal to 0.    

        Examples
        -------- 
        >>> y = larry([[1, 2], [3,  4]]) < 2
        >>> y
        label_0
            0
            1
        label_1
            0
            1
        x
        array([[ True, False],
               [False, False]], dtype=bool)
        >>> y.any()
        True
        >>> y.any(axis=1)
        label_0
            0
            1
        x
        array([ True, False], dtype=bool)
        
        """
        return self.__reduce(np.any, default=np.False_, axis=axis)
        
    def all(self, axis=None):
        """
        True if all elements along specified axis are True; False otherwise.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis over which to reduce the truth. By default (axis=None)
            the larry is flattened before the truth is found.      
        
        Returns
        -------
        out : {larry, True, False}
            If `axis` is None then returns True if all data elements of the
            larry (not including the label) are True; False otherwise. If
            `axis` is an integer then a bool larry is returned.
            
        Notes
        -----
        As in Numpy, NaN is True since it is not equal to 0.                    

        Examples
        -------- 
        >>> y = larry([[1, 2], [3,  4]]) > 1
        >>> y
        label_0
            0
            1
        label_1
            0
            1
        x
        array([[False,  True],
               [ True,  True]], dtype=bool)
        >>> y.all()
        False
        >>> y.all(axis=0)
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
        
        """
        return self.__reduce(np.all, default=~np.False_, axis=axis) 
            
    def lastrank(self, axis=-1, decay=0):
        """
        The ranking of the last element along the axis, ignoring NaNs.
        
        The ranking is normalized to be between -1 and 1 instead of the more
        common 1 and N. The results are adjusted for ties. Suitably slicing
        the output of the `ranking` method will give the same result as
        `lastrank`. The only difference is that `lastrank` is faster.   

        Parameters
        ----------
        axis : int, optional
            The axis over which to rank. By default (axis=-1) the ranking
            (and reducing) is performed over the last axis.          
        decay : scalar, optional
            Exponential decay strength. Cannot be negative. The default
            (decay=0) is no decay. In normal ranking (decay=0) all elements
            used to calculate the rank are equally weighted and so the
            ordering of all but the last element does not matter. In
            exponentially decayed ranking the ordering of the elements
            influences the ranking: elements nearer the last element get more
            weight.
            
        Returns
        -------
        d : larry
            In the case of, for example, a 2d larry of shape (n, m) and
            axis=1, the output will contain the rank (normalized to be between
            -1 and 1 and adjusted for ties) of the the last element of each
            row. The output in this example will have shape (n,). 
            
        See Also
        --------
        la.larry.ranking: Rank elements treating NaN as missing. 
        la.larry.movingrank: Moving rank in a given window along axis.
                
        Examples
        -------- 
        Create a larry:
                     
        >>> y1 = larry([1, 2, 3])
        
        What is the rank of the last element (the value 3 in this example)?
        It is the largest element so the rank is 1.0:
        
        >>> y1.lastrank()
        1.0
        
        Now let's try an example where the last element has the smallest
        value:
        
        >>> y2 = larry([3, 2, 1])
        >>> y2.lastrank()
        -1.0
        
        Here's an example where the last element is not the minimum or maximum
        value:
        
        >>> y3 = larry([1, 3, 4, 5, 2])
        >>> y3.lastrank()
        -0.5
        
        Finally, let's add a large decay. The decay means that the elements
        closest to the last element receive the most weight. Because the
        decay is large, the first element (the value 1) doesn't get any weight
        and therefore the last element (2) becomes the smallest element:
        
        >>> y3.lastrank(decay=10)
        -1.0
                    
        """
        return self.__reduce(lastrank, axis=axis, decay=decay)
        
    # Comparision ------------------------------------------------------------
        
    def __eq__(self, other):
        "Element by element equality (==) comparison."
        return self.__compare(other, '==')                   

    def __ne__(self, other):
        "Element by element inequality (!=) comparison."
        return self.__compare(other, '!=')                       

    def __lt__(self, other):
        "Element by element 'less than' (<) comparison."    
        return self.__compare(other, '<')  

    def __gt__(self, other):
        "Element by element 'greater than' (>) comparison."    
        return self.__compare(other, '>') 

    def __le__(self, other):
        "Element by element 'less than or equal to' (<=) comparison."    
        return self.__compare(other, '<=') 

    def __ge__(self, other):
        "Element by element 'greater than or equal to' (>=) comparison."     
        return self.__compare(other, '>=') 
  
    def __compare(self, other, op):         
        if (np.isscalar(other) or isinstance(other, np.ndarray) or
            (type(other) == list) or (type(other) == tuple)):
            if op == '==':
                x = self.x == other
            elif op == '!=':
                x = self.x != other
            elif op == '<':
                x = self.x < other           
            elif op == '>':
                x = self.x > other           
            elif op == '<=':
                x = self.x <= other           
            elif op == '>=':
                x = self.x >= other           
            else:
                raise ValueError, 'Unknown comparison operator'
            if isinstance(x, np.ndarray):
                y = larry(x, self.copylabel(), validate=False)
            else:
                y = x
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
            return larry(x, label, validate=False)
        else:
            raise TypeError, 'Input must be scalar, numpy array, or larry.'

    # Get and set ------------------------------------------------------------
    
    def __getitem__(self, index):
        """Index into a larry.
        
        Examples
        --------
        >>> y = larry([[1, 2], [3,  4]])
        >>> y[0,0]
        1
        >>> y[0,:]
        label_0
            0
            1
        x
        array([1, 2])
        >>> y[:,1:]
        label_0
            0
            1
        label_1
            1
        x
        array([[2],
               [4]])
        
        """
        typidx = type(index)
        if isscalar(index):
            index = int(index)                
            if index >= self.shape[0]:
                raise IndexError, 'index out of range'
            x = self.x[index]
            if self.ndim <= 1:
                return x
            label = self.label[1:]
        elif typidx is tuple:
            label = []
            allscalar = True
            for ax in xrange(self.ndim):
                if ax < len(index):
                    idx = index[ax]
                    typ = type(idx)
                    if isscalar(idx):
                        if idx >= self.shape[ax]:
                            raise IndexError, 'index out of range'
                        lab = None
                    elif typ is list or typ is tuple:
                        try:
                            lab = [self.label[ax][z] for z in idx]
                        except IndexError:
                            raise IndexError, 'index out of range'
                        allscalar = False
                    elif typ is np.ndarray:
                        if idx.ndim != 1:
                            msg = 'You can use a Numpy array for indexing, '
                            msg += 'but it must be 1d.'
                            raise IndexError, msg
                        if idx.dtype.type == np.bool_:
                            try:
                                lab = [self.label[ax][j] for j, z in
                                                          enumerate(idx) if z]
                            except IndexError:
                                raise IndexError, 'index out of range'
                        else:
                            try:
                                lab = [self.label[ax][z] for z in idx]
                            except IndexError:
                                raise IndexError, 'index out of range'                       
                        allscalar = False
                    elif typ is slice:
                        lab = self.label[ax][idx] 
                        allscalar = False
                    else:
                        msg = 'I do not recognize the way you are indexing'
                        raise IndexError, msg                       
                else:
                    lab = self.label[ax]
                if lab is not None:     
                    label.append(lab)
            x = self.x[index]
            if allscalar:
                return x
        elif typidx is slice:
            label = list(self.label)
            label[0] = label[0][index]
            x = self.x[index]
        elif typidx is list:
            label = list(self.label)
            label[0] = [label[0][int(i)] for i in index]
            x = self.x.take(index, axis=0) 
        elif typidx is np.ndarray:    
            if index.ndim != 1:
                msg = 'You can use a Numpy array for indexing, '
                msg += 'but it must be 1d.'
                raise IndexError, msg
            if index.dtype.type == np.bool_:
                try:
                    lab = [self.label[0][j] for j, z in
                                                enumerate(index) if z]
                except IndexError:
                    raise IndexError, 'index out of range'
            else:
                try:
                    lab = [self.label[0][z] for z in index]
                except IndexError:
                    raise IndexError, 'index out of range' 
            label = self.copylabel()
            label[0] = lab        
            x = self.x[index]                                 
        else:        
            msg = 'Only slice, integer, and seq (list, tuple, 1d array)'
            msg = msg + ' indexing supported'
            raise IndexError, msg        
        return larry(x, label)

    def take(self, indices, axis):
        """
        A copy of the specified elements of a larry along an axis.
        
        This method does the same thing as "fancy" indexing (indexing larrys
        using lists or Numpy arrays); however, it can be easier to use if you
        need elements along a given axis. It is also faster in many
        situations.
        
        Parameters
        ----------
        indices : sequence
            The indices of the values to extract. You can use a list, tuple,
            1d Numpy array or any other iterable object.
        axis : int
            The axis over which to select values.
        
        Returns
        -------
        lar : larry
            A copy of the specified elements of a larry along the specifed
            axis.
        
        Examples
        --------
        Take elements at index 0 and 2 from a 1d larry:
        
        >>> y = larry([1, 2, 3])
        >>> y.take([0, 2], axis=0)
        label_0
            0
            2
        x
        array([1, 3])
        
        Take columns 0 and 2 from a 2d larry:

        >>> y = la.rand(2, 3)
        >>> y.take([0, 2], axis=1)
        label_0
            0
            1
        label_1
            0
            2
        x
        array([[ 0.07887698,  0.44490303],
               [ 0.75024392,  0.92896999]])

        The above is equivalent to (but faster than):
 
        >>> y[:, [0, 2]]
        label_0
            0
            1
        label_1
            0
            2
        x
        array([[ 0.07887698,  0.44490303],
               [ 0.75024392,  0.92896999]])
       
        """
        label = self.copylabel()
        labelaxis = label[axis]
        label[axis] = [labelaxis[idx] for idx in indices]
        x = self.x.take(indices, axis)
        return larry(x, label)

    @property    
    def lix(self):
        """
        Index into a larry using labels or index numbers or both.
        
        In order to distinguish between labels and indices, label elements
        must be wrapped in a list while indices (integers) cannot be wrapped
        in a list. If you wrap indices in a list they will be interpreted as
        label elements.
        
        When indexing with multi-element lists of labels along more than one
        axes, rectangular indexing is used instead of fancy indexing. Note
        that the corresponding situation with NumPy arrays would produce
        fancy indexing.
        
        Slicing can be done with labels or indices or a combination of the
        two. A single element along an axis can be selected with a label or
        the index value. Several elements along an axis can be selected with
        a multi-element list of labels. Lists of indices are not allowed.
        
        Examples
        --------
        
        Let's start by making a larry that we can use to demonstrate idexing
        by label:
        
        >>> y = larry(range(6), [['a', 'b', 3, 4, 'e', 'f']])

        We can select the first element of the larry using the index value, 0,
        or the corresponding label, 'a':

        >>> y.lix[0]
        0
        >>> y.lix[['a']]
        0
        
        We can slice with index values or with labels:
        
        >>> y.lix[0:]
        label_0
            a
            b
            3
            4
            e
            f
        x
        array([0, 1, 2, 3, 4, 5])
        
        >>> y.lix[['a']:]
        label_0
            a
            b
            3
            4
            e
            f
        x
        array([0, 1, 2, 3, 4, 5])
         
        >>> y.lix[['a']:['e']]
        label_0
            a
            b
            3
            4
        x
        array([0, 1, 2, 3])
        
        >>> y.lix[['a']:['e']:2]
        label_0
            a
            3
        x
        array([0, 2])

        Be careful of the difference between indexing with indices and
        indexing with labels. In the first example below 4 is an index; in
        the second example 4 is a label element:

        >>> y.lix[['a']:4]
        label_0
            a
            b
            3
            4
        x
        array([0, 1, 2, 3])
        
        >>> y.lix[['a']:[4]]
        label_0
            a
            b
            3
        x
        array([0, 1, 2])
        
        Here's a demonstration of rectangular indexing:
        
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y.lix[['a', 'b'], ['c', 'd']]
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[1, 2],
               [3, 4]])
               
        The rectangular indexing above is very different from how Numpy arrays
        behave. The corresponding example with a NumyPy array:       

        >>> x = np.array([[1, 2], [3, 4]])
        >>> x[[0, 1], [0, 1]]
        array([1, 4])       
        
        """                       
        return Getitemlabel(self)                               
        
    def __setitem__(self, index, value):
        """
        Assign values to a subset of a larry using indexing to select subset.
        
        Examples
        --------
        Let's set all elements of a larry with values less then 3 to zero:
                
        >>> import numpy as np
        >>> x = np.array([[1, 2], [3, 4]])
        >>> label = [['a', 'b'], [8, 10]]
        >>> y = larry(x, label)
        >>> y[y < 3] = 0
        >>> y
        label_0
            a
            b
        label_1
            8
            10
        x
        array([[0, 0],
               [3, 4]])
        
        """
        if isinstance(index, larry):
            if self.label == index.label:
                self.x[index.x] = value
            else:
                # Could use morph to do this, if every row and column of self
                # is in index, but I think it is better to raise an IndexError
                msg = 'Indexing with a larry that is not aligned'
                raise IndexError, msg    
        else:
            if isinstance(value, larry):
                # TODO The line below (self[index].label) is slow. Need a
                # function that indexes into labels without indexing into x.
                # Then use that function in getitem
                if self[index].label == value.label:
                    self.x[index] = value.x
                else:    
                    raise IndexError, 'larrys are not aligned.'    
            else:
                self.x[index] = value
            
    def set(self, label, value):
        """
        Set one x element given a list of label names.
        
        Give one label name (not label index) for each dimension.
        
        Parameters
        ----------
        label : {list, tuple}
            List or tuple of one label name for each dimension. For example,
            for row label 'a' and column label 7: ('a', 7).
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
        
        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        >>> y.set(['r0', 'c1'], 99)
        >>> y
        label_0
            r0
            r1
        label_1
            c0
            c1
        x
        array([[ 1, 99],
               [ 3,  4]])
        
        """
        if len(label) != self.ndim:
            raise ValueError, 'Must have exactly one label per dimension'
        index = []
        for i in xrange(self.ndim):
            index.append(self.labelindex(label[i], axis=i))    
        self.x[tuple(index)] = value        

    def get(self, label):
        """
        Get one x element given a list of label names.
        
        Give one label name (not label index) for each dimension.
        
        Parameters
        ----------
        label : {list, tuple}
            List or tuple of one label name for each dimension. For example,
            for row label 'a' and column label 7: ('a', 7).
            
        Returns
        -------
        out : scalar, string, etc.
            Value of the single cell specified by label.
        
        Raises
        ------
        ValueError
            If the length of label is not equal to the number of dimensions of
            larry.        

        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        >>> y.get(['r0', 'c1'])
        2
        
        """    
        if len(label) != self.ndim:
            raise ValueError, 'Must have exactly one label per dimension'
        index = []
        for i in xrange(self.ndim):
            index.append(self.labelindex(label[i], axis=i))    
        return self.x[tuple(index)]
                
    def getx(self, copy=True):
        """
        Return a copy of the x data or a reference to it.
        
        Parameters
        ----------
        copy : {True, False}, optional
            Return a copy (True, default) of the x values or a reference
            (False) to it.
            
        Returns
        -------
        out : array
            Copy or reference of x array.

        Examples
        --------
        >>> y = larry([0, 1, 2])
        >>> x = y.getx()
        >>> (x == y.x).all()
        True
        >>> x is y.x
        False
        >>> x = y.getx(copy=False)
        >>> x is y.x
        True
               
        """
        if copy:
            return self.x.copy()
        else:
            return self.x
            
    @property
    def A(self):
        """
        Return a reference to the underlying Numpy array.
        
        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.A
        array([1, 2, 3])
        >>> type(y.A)
        <type 'numpy.ndarray'>        
        
        """
        return self.x            
                    
    def getlabel(self, axis, copy=True):
        """
        Return a copy of the label or a reference to it.
        
        Parameters
        ----------
        axis : int
            The `axis` identifies the label you wish to get.         
        copy : {True, False}, optional
            Return a copy (True, default) of the label or a reference (False)
            to it.
            
        Returns
        -------
        out : list
            Copy or reference of the label.

        Examples
        --------
        Get a copy of the label:
        
        >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        >>> y.getlabel(axis=0)
        ['r0', 'r1']
        >>> y.getlabel(axis=1)
        ['c0', 'c1']
        
        The difference between a copy and a reference to the label:
        
        >>> label = y.getlabel(0)
        >>> label == y.label[0]
        True
        >>> label is y.label[0]
        False
        >>> label = y.getlabel(0, copy=False)
        >>> label is y.label[0]
        True        
               
        """
        if axis >= self.ndim:
            raise IndexError, 'axis out of range'
        label = self.label[axis]    
        if copy:
            label =  list(label)
        return label            
            
    def pull(self, name, axis):
        """
        Pull out the values for a given label name along a specified axis.
        
        A view of the data (but a copy of the label) is returned and the
        dimension is reduced by one.
        
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
            
        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        >>> y.pull('r0', axis=0)
        label_0
            c0
            c1
        x
        array([1, 2])

        >>> import numpy as np
        >>> x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> label = [['experiment1', 'experient2'], ['r0', 'r1'], ['c0', 'c1']]
        >>> y = larry(x, label)
        >>> y.pull('experiment1', axis=0)
        label_0
            r0
            r1
        label_1
            c0
            c1
        x
        array([[1, 2],
               [3, 4]])
                        
        """
        if axis is None:
            raise ValueError, 'axis cannot be None'        
        label = list(self.label)
        label.pop(axis)  
        idx = self.labelindex(name, axis)
        index = [slice(None)] * self.ndim 
        index[axis] = idx
        x = self.x[index]
        if x.shape == (1,):
            return x[0]
        return larry(x, label)            
        
    def fill(self, fill_value):
        """
        Inplace filling of data array with specified value.
        
        Parameters
        ----------
        fill_value : {scalar, string, etc}
            Value to replace every element of the data array.
            
        Returns
        -------
        out : None
        
        Examples
        --------
        >>> y = larry([0, 1])
        >>> y.fill(9)
        >>> y
        label_0
            0
            1
        x
        array([9, 9])        
                
        """
        self.x.fill(fill_value)
        
    def keep_label(self, op, value, axis):
        """
        Keep labels (and corresponding values) that satisfy conditon.
        
        Keep labels that satify:
        
                            label[`axis`] `op` `value`,
                       
        where `op` can be '==', '>', '<', '>=', '<=', '!=', 'in', 'not in'.               
        
        Parameters
        ----------
        op : string
            Operation to perform. `op` can be '==', '>', '<', '>=', '<=',
            '!=', 'in', 'not in'.
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
            If `op` is unknown or if `axis` is None.
        IndexError
            If `axis` is out of range.
            
        Examples
        --------
        >>> y = larry([1, 2, 3, 4], [['a', 'b', 'c', 'd']])
        >>> y.keep_label('<', 'c', axis=0)
        label_0
            a
            b
        x
        array([1, 2])                    

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
        if len(idxlabel) == 0:
            return larry([])
        else:
            idx, label = zip(*idxlabel)
            y.label[axis] = list(label)
            index = [slice(None, None, None)] * self.ndim
            index[axis] = list(idx)
            y.x = y.x[index]
            return y
        
    def keep_x(self, op, value, vacuum=True):
        """
        Keep labels (and corresponding values) that satisfy conditon.
        
        Set x values that do not satify the condition to NaN. Then, if
        vacuum is True, rows and columns with all NaNs will be removed. If
        vacuum is True, larry must be 2d. 
        
        Note that when vacuum is True, all rows and columns with all NaNs
        (even if they already had all NaNs in the row or column before this
        function was called) will be removed.
        
        The op can be '==', '>', '<', '>=', '<=', '!='.
        
        Parameters
        ----------
        op : string
            Operation to perform. op can be '==', '>', '<', '>=', '<=', '!='.
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
        ops = ('==', '>', '<', '>=', '<=', '!=')
        if op not in ops:
            raise ValueError, 'Unknown op'   
        y = self.copy()
        idx = eval('y.x ' + op + ' value')
        y.x[~idx] = np.nan
        if vacuum:
            y = y.vacuum()
        return y         
                    
    # label operations -------------------------------------------------------
        
    def maxlabel(self, axis=None):
        """
        Maximum label value along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis over which to find the maximum label. By default (None)
            the search for the maximum label element is performed along all
            axes.
            
        Returns
        -------
        out : scalar, string, etc.
            The maximum label element along the specified axis.
            
        Examples
        --------
        What is the maximum label value in the following larry?
        
        >>> y = larry([1, 2, 3], [['a', 'z', 'w']])
        >>> y.maxlabel()
        'z'               
        
        """
        if axis is None:
            return max([max(z) for z in self.label])
        else:
            return max([z for z in self.label[axis]])

    def minlabel(self, axis=None):
        """
        Minimum label value along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis over which to find the minimum label. By default (None)
            the search for the minimum label element is performed along all
            axes.
            
        Returns
        -------
        out : scalar, string, etc.
            The minimum label element along the specified axis.
            
        Examples
        --------
        What is the minimum label value in the following larry?
        
        >>> y = larry([1, 2, 3], [['a', 'z', 'w']])
        >>> y.minlabel()
        'a'               
        
        """
        if axis is None:
            return min([min(z) for z in self.label])
        else:
            return min([z for z in self.label[axis]])
        
    def labelindex(self, name, axis, exact=True):
        """
        Return index of given label element along specified axis.
        
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

        Examples
        --------
        What column number (starting from 0) of the following 2d larry is
        labeled 'west'?
        
        >>> from la import larry
        >>> y = larry([[1, 2], [3, 4]], [['north', 'south'], ['east', 'west']])        
        >>> y.labelindex('west', axis=1)
        1        
                        
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
        
    def maplabel(self, func, axis=None, copy=True):
        """
        Apply given function to each element of label along specified axis.
        
        Parameters
        ----------
        func : function
            Function to apply to each element of label.
        axis : {int, None}, optional
            Axis along which to apply the function.
        copy : bool
            Whether to return a copy (True, default) or to return a reference.
            
        Returns
        -------
        y : larry
            A copy or a reference (dending on the value of `copy`) of the
            larry with the given function applied to the specified labels.
                    
        Examples
        -------- 
        Create a larry with dates in the label:        
        
        >>> import datetime
        >>> d = datetime.date
        >>> y = larry([1, 2], [[d(2010,1,1), d(2010,1,2)]])
        
        Convert the dates in the label to integers:
        
        >>> y.maplabel(datetime.date.toordinal)
        label_0
            733773
            733774
        x
        array([1, 2])
        
        Convert the dates in the label to strings:
               
        >>> y.maplabel(str)
        label_0
            2010-01-01
            2010-01-02
        x
        array([1, 2])
                                  
        """
        if copy:
            y = self.copy()
        else:
            y = self    
        if axis is None:
            for ax in range(y.ndim):
                y.label[ax] = map(func, y.label[ax])
        else:
            y.label[axis] = map(func, y.label[axis])
        return y                    
    
    # Moving window statistics ----------------------------------------------

    def move_sum(self, window, axis=-1):
        """
        Moving window sum along the specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving sum. By default the
            moving sum is taken over the last axis (-1).

        Returns
        -------
        y : larry
            The moving sum along the specified axis, ignoring NaNs. (A window
            with all NaNs returns NaN for the window sum.) The output has the
            same shape as the input.
            
        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4])
        >>> lar.move_sum(window=2) 
        label_0
            0
            1
            2
            3
        x
        array([ NaN,   3.,   2.,   4.])

        """
        x = bn.move_nansum(self.x, window, axis=axis)
        return larry(x, self.copylabel(), validate=False)

    def move_mean(self, window, axis=-1):
        """
        Moving window mean along the specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving mean. By default the
            moving mean is taken over the last axis (-1).

        Returns
        -------
        y : larry
            The moving mean along the specified axis, ignoring NaNs. (A window
            with all NaNs returns NaN for the window mean.) The output has the
            same shape as the input.
        
        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4])
        >>> lar.move_mean(window=2)  
        label_0
            0
            1
            2
            3
        x
        array([ NaN,  1.5,  2. ,  4. ])
            
        """
        x = bn.move_nanmean(self.x, window, axis=axis)
        return larry(x, self.copylabel(), validate=False)

    def move_std(self, window, axis=-1):
        """
        Moving window standard deviation along specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving standard deviation.
            By default the moving standard deviation is taken over the last
            axis (-1).
       
        Returns
        -------
        y : larry
            The moving standard deviation along the specified axis, ignoring
            NaNs. (A window with all NaNs returns NaN for the window standard
            deviation.) The output has the same shape as the input.

        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4, 5])
        >>> lar.move_std(window=3) 
        label_0
            0
            1
            2
            3
            4
        x
        array([ NaN,  NaN,  0.5,  1. ,  0.5])
        
        """
        x = bn.move_nanstd(self.x, window, axis=axis)
        return larry(x, self.copylabel(), validate=False)

    def move_min(self, window, axis=-1):
        """
        Moving window minimum along the specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving minimum. By default the
            moving minimum is taken over the last axis (-1).

        Returns
        -------
        y : larry
            The moving minimum of the input array along the specified axis,
            ignoring NaNs. (A window with all NaNs returns NaN for the window
            minimum.) The output has the same shape as the input.

        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4])
        >>> lar.move_min(window=2) 
        label_0
            0
            1
            2
            3
        x
        array([ NaN,   1.,   2.,   4.])

        """
        x = bn.move_nanmin(self.x, window, axis=axis)
        return larry(x, self.copylabel(), validate=False)

    def move_max(self, window, axis=-1):
        """
        Moving window maximum along the specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving maximum. By default the
            moving maximum is taken over the last axis (-1).

        Returns
        -------
        y : larry
            The moving maximum of the input array along the specified axis,
            ignoring NaNs. (A window with all NaNs returns NaN for the window
            maximum.) The output has the same shape as the input.

        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4])
        >>> lar.move_max(window=2) 
        label_0
            0
            1
            2
            3
        x
        array([ NaN,   2.,   2.,   4.])

        """
        x = bn.move_nanmax(self.x, window, axis=axis)
        return larry(x, self.copylabel(), validate=False)

    def move_ranking(self, window, axis=-1, method='strides'):
        """
        Moving window ranking along the specified axis, ignoring NaNs.

        The output is normalized to be between -1 and 1. For example, with a
        window width of 3 (and with no ties), the possible output values are
        -1, 0, 1.
        
        Ties are broken by averaging the rankings. See the examples below. 

        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving ranking. By default the
            moving ranking is taken over the last axis (-1).
        method : str, optional
            The following moving window methods are available:
                ==========  =====================================
                'strides'   strides tricks (ndim < 4) (default)
                'loop'      brute force python loop
                ==========  =====================================

        Returns
        -------
        y : larry
            The moving ranking along the specified axis, ignoring NaNs. (A
            window with all NaNs returns NaN for the window ranking; if all
            elements in a window are NaNs except the last element, a NaN is
            returned.) The output has the same shape as the input.

        Examples
        --------
        With window=3 and no ties, there are 3 possible output values, i.e.
        [-1., 0., 1.]:

        >>> lar = larry([1, 2, 6, 4, 5, 3])
        >>> lar.move_ranking(window=3) 
        label_0
            0
            1
            2
            3
            4
            5
        x
        array([ NaN,  NaN,   1.,   0.,   0.,  -1.])

        Ties are broken by averaging the rankings of the tied elements:

        >>> lar = larry([1, 2, 1, 1, 1, 2])
        >>> lar.move_ranking(window=3) 
        label_0
            0
            1
            2
            3
            4
            5
        x
        array([ NaN,  NaN, -0.5, -0.5,  0. ,  1. ])

        In a monotonically increasing sequence, the moving window ranking is
        always equal to 1:
        
        >>> lar = larry([1, 2, 3, 4, 5])
        >>> lar.move_ranking(window=3) 
        label_0
            0
            1
            2
            3
            4
        x
        array([ NaN,  NaN,   1.,   1.,   1.])

        """
        x = move_nanranking(self.x, window, axis=axis, method=method)
        return larry(x, self.copylabel(), validate=False)

    def move_median(self, window, axis=-1, method='loop'):
        """
        Moving window median along the specified axis, ignoring NaNs.
        
        Parameters
        ----------
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to perform the moving median. By default the
            moving median is taken over the last axis (-1).
        method : str, optional
            The following moving window methods are available:
                ==========  =====================================
                'loop'      brute force python loop (default)
                'strides'   strides tricks (ndim < 4)
                ==========  =====================================

        Returns
        -------
        y : larry
            The moving median along the specified axis, ignoring NaNs. (A
            window with all NaNs returns NaN for the window maximum.) The
            output has the same shape as the input.

        Examples
        --------
        >>> lar = larry([1, 2, la.nan, 4, 5])
        >>> lar.move_median(window=2)
        label_0
            0
            1
            2
            3
            4
        x
        array([ NaN,  1.5,  2. ,  4. ,  4.5])
        
        """
        x = move_nanmedian(self.x, window, axis=axis, method=method)
        return larry(x, self.copylabel(), validate=False)

    def move_func(self, func, window, axis=-1, method='loop', **kwargs):
        """
        Generic moving window function along the specified axis.
        
        Parameters
        ----------
        func : function
            A reducing function such as np.sum, np.max, or np.median that takes
            a Numpy array and axis and, optionally, key word arguments as
            input.
        window : int
            The number of elements in the moving window.
        axis : int, optional
            The axis over which to evaluate `func`. By default the window moves
            along the last axis (-1).
        method : str, optional
            The following moving window methods are available:
                ==========  =====================================
                'loop'      brute force python loop (default)
                'strides'   strides tricks (ndim < 4)
                ==========  =====================================

        Returns
        -------
        y : larry
            A moving window evaluation of `func` along the specified axis.
            The output has the same shape as the input.

        Examples
        --------
        >>> lar = larry([1, 2, 3, 4])
        >>> lar.move_func(np.sum, window=2)
        label_0
            0
            1
            2
            3
        x
        array([ NaN,   3.,   5.,   7.])

        which give the same result as:

        >>> lar.move_sum(window=2)
        label_0
            0
            1
            2
            3
        x
        array([ NaN,   3.,   5.,   7.])

        """
        x = move_func(func, self.x, window, axis=axis, method=method)
        return larry(x, self.copylabel(), validate=False)

    @np.deprecate(new_name='move_sum')
    def movingsum(self, window, axis=-1, norm=False):
        y = self.copy()
        y.x = movingsum(y.x, window, axis=axis, norm=norm)
        return y
        
    def movingsum_forward(self, window, skip=0, axis=-1, norm=False):    
        """Movingsum in the forward direction skipping skip dates"""      
        y = self.copy()
        y.x = movingsum_forward(y.x, window, skip=skip, axis=axis, norm=norm)
        return y
    
    @np.deprecate(new_name='move_ranking')                        
    def movingrank(self, window, axis=-1):
        """Moving rank (normalized to -1 and 1) of a given window along axis.

        Normalized for missing (NaN) data.
        A data point with NaN data is returned as NaN
        If a window is all NaNs except last, this is returned as NaN
        """
        return self.move_ranking(window, axis=axis)
        
    # Calc ------------------------------------------------------------------

    def demean(self, axis=None):
        """
        Subtract the mean along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis along which to remove the mean. The default (None) is
            to subtract the mean of the flattened larry.

        Returns
        -------
        y : larry
            A copy with the mean along the specified axis removed.

        Examples
        --------
        >>> y = larry([1, 2, 3, 4])
        >>> y.demean()
        label_0
            0
            1
            2
            3
        x
        array([-1.5, -0.5,  0.5,  1.5])
            
        """
        return larry(demean(self.x, axis), self.copylabel(), validate=False)

    def demedian(self, axis=None):
        """
        Subtract the median along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis along which to remove the median. The default (None) is
            to subtract the median of the flattened larry.
        
        Returns
        -------
        y : larry
            A copy with the median along the specified axis removed.
        
        Examples
        --------
        >>> y = larry([1, 2, 3, 4])
        >>> y.demedian()
        label_0
            0
            1
            2
            3
        x
        array([-1.5, -0.5,  0.5,  1.5])
            
        """
        return larry(demedian(self.x, axis), self.copylabel(), validate=False)
        
    def zscore(self, axis=None):
        """
        Z-score along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis along which to take the z-score. The default (None) is
            to find the z-score of the flattened larry.
        
        Returns
        -------
        y : larry
            A copy normalized with the Z-score along the specified axis.
        
        Examples
        --------
        >>> y = larry([1, 2, 3])
        >>> y.zscore()
        label_0
            0
            1
            2
        x
        array([-1.22474487,  0.        ,  1.22474487])
            
        """
        return larry(zscore(self.x, axis), self.copylabel(), validate=False)
      
    def ranking(self, axis=0, norm='-1,1'):
        """
        Rank elements treating NaN as missing and averaging ties.

        Parameters
        ----------
        axis : {int, None} optional
            Axis to rank over. Default axis is 0.
        norm: str, optional
            A string that specifies the normalization:
                ==========  ================================================
                '0,N-1'     Zero to N-1 ranking
                '-1,1'      Scale zero to N-1 ranking to be between -1 and 1
                'gaussian'  Rank data then scale to a Gaussian distribution
                ==========  ================================================
            The default ranking is '-1,1'.    
            
        Returns
        -------
        y : larry
            The ranked data. The dtype of the output is always np.float even
            if the dtype of the input is int.
            
        Notes
        -----
        If there is only one non-NaN value along the given axis, then that
        value is set to the midpoint of the specified normalization method.
        For example, if the input is array([1.0, nan]), then 1.0 is set to
        zero for the '-1,1' and 'gaussian' normalizations and is set to 0.5
        (mean of 0 and 1) for the '0,N-1' normalization.
        
        For '0,N-1' normalization, note that N is x.shape[axis] even if there
        are NaNs. That ensures that when ranking along the columns of a 2d
        array, for example, the output will have the same min and max along
        all columns.

        """
        return larry(ranking(self.x, axis, norm=norm), self.copylabel(),
                     validate=False)

    def quantile(self, q, axis=0):
        """
        Assign elements along specified axis into q bins, where smallest
        elements are in bin 1, next smallest in bin 2, ..., largest elements
        are in bin q; then normalize output be between [-1, 1].

        Parameters
        ----------
        q : int
            The number of bins into which to quantize the data. Must be at
            least 1 but less than the number of elements along the specified
            axis.
        axis : {int, None}, optional
            The axis along which to quantize the elements. The default is
            axis 0.

        Returns
        -------
        lar : larry
           A quantized copy of the larry.

        Examples
        --------
        >>> lar = larry([1, 2, 3, 4, 5, 6])
        >>> lar.quantile(3)
        label_0
            0
            1
            2
            3
            4
            5
        x
        array([-1., -1.,  0.,  0.,  1.,  1.])
            
        """
        label = self.copylabel()
        x = quantile(self.x, q, axis=axis)       
        return larry(x, label, validate=False)
        
    # Group calc -------------------------------------------------------------  
                 
    def group_ranking(self, group, axis=0):
        """Group (e.g. sector) ranking along columns.
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """
        y = self.copy()
        aligned_group_list = y._group_align(group, axis=axis)
        y.x = group_ranking(y.x, aligned_group_list, axis=axis)
        return y
            
    def group_mean(self, group, axis=0):
        """Group (e.g. sector) mean along columns (zero axis).
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """        
        y = self.copy() 
        aligned_group_list = y._group_align(group, axis=axis)
        y.x = group_mean(y.x, aligned_group_list, axis=axis)
        return y
        
    def group_median(self, group, axis=0):
        """Group (e.g. sector) median along columns (zero axis).
        
        The row labels of the object must be a subset of the row labels of the
        group.
        """ 
        y = self.copy()
        aligned_group_list = y._group_align(group, axis=axis)   
        y.x = group_median(y.x, aligned_group_list, axis=axis)
        return y
    
    def _group_align(self, group, axis=0):
        """Return a row aligned group list (e.g. sector list) of values.
        
        group must have exactly one column. The row labels of the object must
        be a subset of the row labels of the group.
        """
        if not isinstance(group, larry):
            raise TypeError, 'group must be a larry'
        if group.ndim != 1:
            raise ValueError, 'group must be a 1d larry'
        if len(frozenset(self.label[axis]) - frozenset(group.label[0])):
            raise IndexError, 'label is not a subset of group label'
        g = group.morph(self.label[axis], 0)
        g = g.x.tolist()
        return g 
                                                                 
    # Alignment --------------------------------------------------------------     
    def morph(self, label, axis):
        """
        Reorder the elements along the specified axis.
        
        If an element in `label` does not exist in the larry, NaNs will be
        used for float dtype, None will be used for object dtype, and ''
        will be used for string (np.string_) dtype. All other dtype, such as
        int and bool, will be cast to float if there are any elements in
        `label` does not exist in the larry.
        
        Parameters
        ----------
        label : list
            Desired ordering of elements along specified axis.
        axis : int
            axis along which to perform the reordering.
        
        Returns
        -------
        out : larry
            A reordered copy.       
        
        See Also
        --------
        la.larry.morph_like: Morph along all axes to align with given larry.
        la.larry.merge: Merge, or optionally update, a larry with a second
                        larry.
        la.align: Align two larrys using one of five join methods.
            
        Examples
        --------
        >>> y = larry([1, 2, 3], [['a', 'b', 'c']])
        >>> y.morph(['b', 'ee', 'a', 'c'], axis=0)
        label_0
            b
            ee
            a
            c
        x
        array([  2.,  NaN,   1.,   3.])                    
            
        """
        if axis >= self.ndim:
            raise IndexError, 'axis out of range'
        if self.label[axis] == label:
            return self.copy()
        else:    
            idx, idx_miss = listmap_fill(self.label[axis], label)           
            if len(idx) == len(idx_miss):
                # None of the elements we want are in the input larry
                shape = list(self.x.shape)
                shape[axis] = len(idx)
                miss = missing_marker(self.x)
                if miss == NotImplemented:
                    x = self.x.astype(float)
                else:
                    x = self.x
                x = nans(shape, dtype=x.dtype)
            else:    
                x = self.x.take(idx, axis)
                if len(idx_miss) > 0:
                    index = [slice(None)] * self.ndim
                    index[axis] = idx_miss
                    miss = missing_marker(x)
                    if miss == NotImplemented:
                        x = x.astype(float)
                        miss = missing_marker(x)
                    x[index] = miss      
            lab = self.copylabel()
            lab[axis] = list(label)
        return larry(x, lab)
        
    def morph_like(self, lar):
        """
        Morph along all axes to align with given larry.

        If label elements in `lar` do not exist in the larry, then a fill
        value is used to mark the missing values. See `morph` for details.
        
        Parameters
        ---------- 
        lar : larry
            The target larry to align to.
            
        Returns
        -------
        lar : larry
            A morphed larry that is aligned with the input larry, `lar`.
            
        See Also
        --------
        la.larry.morph: Reorder the elements along the specified axis.
        la.larry.merge: Merge, or optionally update, a larry with a second
                        larry.
        la.align: Align two larrys using one of five join methods.                            

        Examples
        --------
        Align y1 to y2:
        
        >>> y1 = larry([1, 2], [['a', 'b']])
        >>> y2 = larry([3, 2, 1], [['c', 'b', 'a']])

        >>> y1.morph_like(y2)
        label_0
            c
            b
            a
        x
        array([ NaN,   2.,   1.])

        Align y2 to y1:
       
        >>> y2.morph_like(y1)
        label_0
            a
            b
        x
        array([ 1.,  2.])
        
        """
        if self.ndim != lar.ndim:
            raise IndexError, 'larrys must be of the same dimension.'
        y = self.copy()      
        for i in range(self.ndim):
            y = y.morph(lar.getlabel(i), axis=i)     
        return y        

    def merge(self, other, update=False):
        """
        Merge, or optionally update, a larry with a second larry.
     
        Parameters
        ----------
        other : larry
            The larry to merge or to use to update the values. It must have
            the same number of dimensions as the existing larry.
        update : bool
            Raise a ValueError (default) if there is any overlap in the two
            larrys. An overlap is defined as a common label in both larrys
            that contains a finite value in both larrys. If `update` is True
            then the overlapped values in the current larry will be
            overwritten with the values in `other`.            
     
        Returns
        -------
        lar1 : larry
            The merged larry.
            
        Notes
        -----
        If either larry has dtype of object or np.string_ then both larrys
        must have the same dtype, otherwise a TypeError is raised.   

        Examples
        --------
        >>> y1 = larry([1, 2], [['a', 'b']])
        >>> y2 = larry([3, 4], [['c', 'd']])
        >>> y1.merge(y2)
        label_0
            a
            b
            c
            d
        x
        array([ 1.,  2.,  3.,  4.])
     
        """

        ndim = self.ndim
        if ndim != other.ndim:
            raise IndexError, 'larrys must be of the same dimension.'
        lar1 = self
        lar2 = other
        for ax in range(ndim):
            if lar1.label[ax] != lar2.label[ax]:
                mergelabel = set(lar1.label[ax]) | set(lar2.label[ax])
                mergelabel = sorted(mergelabel)
                lar1 = lar1.morph(mergelabel, ax)
                lar2 = lar2.morph(mergelabel, ax)
     
        # Mask       
        dtype1 = self.dtype       
        if dtype1 == object:
            mask1 = lar1.x != [None]
        elif self.dtype.type == np.string_:
            mask1 = lar1.x != ''  
        else:
            mask1 = np.isfinite(lar1.x)
        dtype2 = other.dtype       
        if dtype2 == object:
            mask2 = lar2.x != [None]
        elif self.dtype.type == np.string_:
            mask2 = lar2.x != ''  
        else:
            mask2 = np.isfinite(lar2.x)
            
        # Trap cases that merge cannot handle
        if dtype1 in (np.string_, object) or dtype2 in (np.string_, object):
            if dtype1 != dtype2:
                raise TypeError, 'Incompatible dtypes'             

        # Check for overlap if requested             
        if (not update) and np.logical_and(mask1, mask2).any():
            raise ValueError('Overlapping values')
        else:
            if mask2.any():
                lar1.x[mask2] = lar2.x[mask2]
     
        return lar1
        
    def squeeze(self):
        """
        Remove all length-1 dimensions and corresponding labels.
        
        Note that a view (reference) is returned, not a copy.
        
        Returns
        -------
        out : larry
            Returns a view with all length-1 dimensions and corresponding
            labels removed.
            
        Examples
        --------
        >>> y = larry([[1, 2]], [['row'], ['c1', 'c2']])
        >>> y
        label_0
            row
        label_1
            c1
            c2
        x
        array([[1, 2]])
        >>> y.squeeze()
        label_0
            c1
            c2
        x
        array([1, 2])             
        
        """
        idx = [i for i, z in enumerate(self.shape) if z != 1]
        label = []
        for i in idx:
            label.append(self.label[i])    
        x = self.x.squeeze()
        return larry(x, label, validate=False)

    def lag(self, nlag, axis=-1):
        """
        Lag the values along the specified axis.
        
        Parameters
        ----------
        nlag : int
            Number of periods (rows, columns, etc) to lag. The lag can be
            positive (delay), zero (copy of input), or negative (push
            forward).
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
            
        Examples
        --------
        Create a larry:

        >>> y = larry([1, 2, 3, 4], [['a', 'b', 'c', 'd']])

        A positive lag:

        >>> y.lag(2)
        label_0
            c
            d
        x
        array([1, 2])

        A negative lag:

        >>> y.lag(-2)
        label_0
            a
            b
        x
        array([3, 4])
                        
        """
        if axis is None:
            raise IndexError, 'axis cannot be None.'
        if nlag > 0:
            y = self.copy()
            y.label[axis] = y.label[axis][nlag:]
            index = [slice(None)] * self.ndim
            index[axis] = slice(0, -nlag)            
            y.x = y.x[index]
        elif nlag < 0:
            y = self.copy()
            y.label[axis] = y.label[axis][:nlag]
            index = [slice(None)] * self.ndim
            index[axis] = slice(-nlag, None)            
            y.x = y.x[index]
        elif nlag == 0:
            y = self.copy()
        else:
            raise RuntimeError("Unrecognized value of `nlag`.")
        return y

    def sortaxis(self, axis=None, reverse=False):
        """
        Sort data (and label) according to label along specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis to sort along. The default (None) is to sort all axes.
        reverse : {True, False}, optional
            Sort in descending order (True) or ascending order (False). The
            default is to sort in ascending order. 
            
        Returns
        -------
        y : larry
            A sorted copy of the larry.
            
        Examples
        -------- 
        Let's make a larry that we can use to demonstrate the `sortaxis`
        method: 
                 
        >>> y = larry([[4, 3], [2, 1]], [['b', 'a'], ['d', 'c']])
        >>> y
        label_0
            b
            a
        label_1
            d
            c
        x
        array([[4, 3],
               [2, 1]])

        By default all axes are sorted:

        >>> y.sortaxis()
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[1, 2],
               [3, 4]])
        
        You can also sort in reverse order (although in this particular
        example the larry is already in reverse order):
        
        >>> y.sortaxis(reverse=True)
        label_0
            b
            a
        label_1
            d
            c
        x
        array([[4, 3],
               [2, 1]])

        And you can sort along a single axis:

        >>> y.sortaxis(axis=0)
        label_0
            a
            b
        label_1
            d
            c
        x
        array([[2, 1],
               [4, 3]])
               
        """
        if axis is None:
            axes = range(self.ndim)
        else:
            axes = [axis]
        y = self.copy()    
        shape = self.shape    
        for ax in axes:
            if shape[ax] > 1:        
                index = sorted(self.label[ax], reverse=reverse)
                y = y.morph(index, ax)        
        return y
        
    def flipaxis(self, axis=None, copy=True):
        """
        Reverse the order of the elements along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis to flip. The default (None) is to flip all axes.
        copy : {True, False}, optional
            If True (default) return a copy; if False return a view.
            
        Returns
        -------
        y : larry
            A copy or view (depending on the value of `copy`) of the larry
            that has been flipped.
            
        Examples
        --------
        Create a larry:
        
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[1, 2],
               [3, 4]])
               
        Flip all axes:       

        >>> y.flipaxis()
        label_0
            b
            a
        label_1
            d
            c
        x
        array([[4, 3],
               [2, 1]])
               
        Flip axis 0 only:       

        >>> y.flipaxis(axis=0)
        label_0
            b
            a
        label_1
            c
            d
        x
        array([[3, 4],
               [1, 2]])            
        
        """
        if copy:
            y = self.copy()
        else:
            y = self
        if axis is None:
            axes = range(self.ndim)
        else:
            axes = [axis]
        flip = slice(None, None, -1)    
        for ax in axes:                
            y.label[ax] = y.label[ax][flip]    
            index = [slice(None)] * y.ndim
            index[ax] = flip
            y.x = y.x[index] 
        return y               
        
    # Shuffle ----------------------------------------------------------------
    
    def shuffle(self, axis=0):
        """
        Shuffle the data inplace along the specified axis.
        
        Unlike numpy's shuffle, this shuffle takes an axis argument. The
        ordering of the labels is not changed, only the data is shuffled.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis to shuffle the data along. Default is axis 0. If None,
            then the data will be shuffled along all axes.
            
        Returns
        -------
        out : None
            The data are shuffled inplace.
            
        See Also
        --------
        la.larry.shufflelabel : Shuffle the label inplace along the specified
                                axis.             
            
        Examples
        --------
        >>> y = larry([[1, 2], [3,  4]], [['north', 'south'], ['east', 'west']])
        >>> y.shuffle()
        >>> y
        label_0
            north
            south
        label_1
            east
            west
        x
        array([[3, 4],
               [1, 2]])
        
        """
        if axis is None:
            for ax in range(self.ndim):
                shuffle(self.x, ax) 
        else:
            shuffle(self.x, axis)
        
    def shufflelabel(self, axis=0):
        """
        Shuffle the label inplace along the specified axis.
        
        Parameters
        ----------
        axis : {int, None}, optional
            The axis to shuffle the data along. Default is axis 0. If None,
            then the labels will be shuffled along all axes, where each
            label axis will still contain the same set of labels (labels
            from one axis will not be shuffle to another axis).
            
        Returns
        -------
        out : None
            The labels are shuffled inplace.
            
        See Also
        --------
        la.larry.shuffle : Shuffle the data inplace along the specified axis.    
            
        Examples
        --------
        >>> y = larry([[1, 2], [3,  4]], [['north', 'south'], ['east', 'west']])
        >>> y.shufflelabel()
        >>> y
        label_0
            south
            north
        label_1
            west
            east
        x
        array([[3, 4],
               [1, 2]])                    
        
        """
        if axis is None:
            for ax in range(self.ndim):
                np.random.shuffle(self.label[ax]) 
        else:
            np.random.shuffle(self.label[axis])
            
    # Missing ----------------------------------------------------------------

    def ismissing(self):
        """
        A bool larry with element-wise marking of missing values.
            
        Returns
        -------
        y : larry
            Returns a bool larry that contains the value True if the
            corresponding element of the larry is missing; otherwise False.

        Examples
        --------  
        Floats:
               
        >>> larry([1.0, la.nan]).ismissing()
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
        
        Strings:
        
        >>> larry(['string', '']).ismissing()
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
        
        Objects:
        
        >>> larry(['', None], dtype=object).ismissing()
        label_0
            0
            1
        x
        array([False,  True], dtype=bool)
        
        bool and int dtype do not support missing values so always return
        False:
        
        >>> larry([0, 1]).ismissing()
        label_0
            0
            1
        x
        array([False, False], dtype=bool)
        .
        >>> larry([True, False]).ismissing()
        label_0
            0
            1
        x
        array([False, False], dtype=bool)        
        
        """   
        label = self.copylabel()
        x = ismissing(self)
        return larry(x, label, validate=False)

    def cut_missing(self, fraction, axis=None):
        """
        Cut rows and columns that contain too many NaNs.
        
        Parameters
        ----------
        fraction : scalar
            Usually a float that give the minimum allowable fraction of
            missing data before the row or column is cut.
        axis : {int, None}
            Look for missing data along this axis. So for axis=0, the missing
            data along columns are checked and columns are cut. For axis=1,
            the missing data along rows are checked and rows are cut.
            
        Returns
        -------
        out : larry
            Returns a copy with rows or columns with lots of missing data cut.                
            
        """    
        
        y = self.copy()
        ndim = y.ndim
        
        if axis is None:
            axes = range(ndim)
        else:
            axes = [range(ndim)[axis]]

        idxsl = []
        labsnew = []
        for ax in range(ndim):
            sl = [None] * ndim
            if ax in axes:
                labsnew.append(y.label[ax])
                sl[ax] = slice(None)
                idxsl.append(np.arange(y.shape[ax])[sl])
                continue
            
            # Find all nans over all other axes
            xtmp = np.rollaxis(np.isfinite(y.x), ax, 0)
            count = np.ones(xtmp.shape)
            for _ in range(ndim-1):
                xtmp = xtmp.sum(-1)
                count = count.sum(-1)
    
            xtmp = xtmp > (1.0 - fraction) * count
            labsnew.append([y.label[ax][ii] for ii in np.nonzero(xtmp)[0]])
            sl[ax] = slice(None)
            idxsl.append(np.nonzero(xtmp)[0][sl])
        
        y.x = y.x[idxsl]
        y.label = labsnew
        
        if y.x.size == 0: 
            # Empty larry left over
            return larry(np.array([]))
        else:
            return y
            
    def push(self, window, axis=-1):
        """Fill missing values (NaNs) with most recent non-missing values if
        recent, where recent is defined by the window. The filling proceeds
        from left to right along each row.
        """
        label = self.copylabel()
        x = push(self.x, window, axis=axis)
        return larry(x, label, validate=False)
        
    def vacuum(self, axis=None):
        """
        Remove all rows and/or columns that contain all NaNs.
                     
        Parameters
        ----------
        axis : None or int or tuple of int
            Remove columns (0) or rows (1) or both (None, default) that
            contain no finite values, for nd arrays see Notes.
            
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
            # Change meaning of axes to axes not in original axes
            axes = [a for a in range(ndim) if a not in axes]
        else:
            axes = axis
            # Change meaning of axes to axes not in original axes
            axes = [a for a in range(ndim) if a not in axes]
        
        idxsl = []
        labsnew = []
        for ax in range(ndim):
            sl = [None]*ndim
            if ax not in axes:
                labsnew.append(y.label[ax])
                sl[ax] = slice(None)
                idxsl.append(np.arange(y.shape[ax])[sl])
                continue
            
            # Find all nans over all other axes
            xtmp = np.rollaxis(np.isfinite(y.x), ax, 0)
            for _ in range(ndim-1):
                xtmp = xtmp.any(-1)
    
            labsnew.append([y.label[ax][ii] for ii in np.nonzero(xtmp)[0]])
            sl[ax] = slice(None)
            idxsl.append(np.nonzero(xtmp)[0][sl])
        
        y.x = y.x[idxsl]
        y.label = labsnew
        return y        
        
    def nan_replace(self, replace_with=0):
        """
        Replace NaNs.
        
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
        np.putmask(y.x, np.isnan(y.x), replace_with)
        return y                                     

    # Size, shape, type ------------------------------------------------------

    @property
    def nx(self):
        """
        Number of finite values (not NaN, -Inf, or Inf) in the larry.
        
        Examples
        --------        
        >>> from la import nan
        >>> y = larry([1, 2, nan])
        >>> y.nx
        2
        
        """
        return np.isfinite(self.x).sum()

    @property
    def size(self):
        """
        Number of elements in the larry.
        
        Examples
        --------   
        >>> from la import nan
        >>> y = larry([1, 2, nan])
        >>> y.size
        3
        
        """
        return self.x.size

    @property
    def shape(self):
        """
        Shape of the larry as a tuple.
        
        Examples
        --------   
        >>> from la import nan
        >>> y = larry([1, 2, nan])
        >>> y.shape
        (3,)       
        """
        return self.x.shape
        
    @property
    def ndim(self):
        """
        Number of dimensions in the larry.
        
        Examples
        --------   
        >>> from la import nan
        >>> y = larry([1, 2, nan])
        >>> y.ndim
        1          
        """
        return self.x.ndim
        
    @property
    def dtype(self):
        """
        The dtype of the elements (not the labels) in the larry.
        
        Examples
        --------   
        >>> from la import nan
        >>> y = larry([1, 2, nan])
        >>> y.dtype
        dtype('float64')         
        """
        return self.x.dtype
        
    def astype(self, dtype):
        """
        Copy of larry cast to specified type.
        
        Parameters
        ----------
        dtype: string or data-type
            Typecode or data-type to which the larry is cast.
            
        Returns
        -------
        y : larry
            A copy of the larry, cast to the specified type.
            
        Examples
        --------
        Create a larry with float dtype:
        
        >>> y = la.larry([1, 2, 2.5])
        >>> y
        label_0
            0
            1
            2
        x
        array([ 1. ,  2. ,  2.5])
        
        Cast the float larry to int:
        
        >>> y.astype(int)
        label_0
            0
            1
            2
        x
        array([1, 2, 2])
                
        """
        label = self.copylabel()
        x = self.x.astype(dtype)
        return larry(x, label, validate=False)
        
    @property
    def T(self):
        """
        Returns a transposed copy of the larry.

        Examples
        --------        
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[1, 2],
               [3, 4]])
        >>> y.T
        label_0
            c
            d
        label_1
            a
            b
        x
        array([[1, 3],
               [2, 4]])
               
        """
        y = self.copy()
        y.x = y.x.T
        y.label = y.label[::-1]
        return y 
        
    def swapaxes(self, axis1, axis2):
        """
        Swap the two specified axes.
        
        Parameters
        ----------
        axis1 : int
            First axis. This axis will become the `axis2`.
        axis2 : int    
            Second axis. This axis will become the `axis1`.
            
        Returns
        -------
        y : larry
            A larry with the specified axes swapped.
            
        Examples
        --------
        First create a (3,2) larry:
        
        >>> y = larry([[0, 1], [2, 3], [4, 5]])
        >>> y
        label_0
            0
            1
            2
        label_1
            0
            1
        x
        array([[0, 1],
               [2, 3],
               [4, 5]])
        
        Then swap axes 0 and 1 (i.e., take the transpose):
        
        >>> y.swapaxes(1,0)
        label_0
            0
            1
        label_1
            0
            1
            2
        x
        array([[0, 2, 4],
               [1, 3, 5]])
                        
        """
        y = self.copy()
        y.label[axis1], y.label[axis2] =  y.label[axis2], y.label[axis1]
        y.x = np.swapaxes(y.x, axis1, axis2)
        return y
            
    def flatten(self, order='C'):
        """
        Return a copy of the larry after collapsing into one dimension.
        
        The elements of the label become tuples.
        
        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to flatten in row-major order ('C', default) or
            column-major order ('F').

        Returns
        -------
        y : larry
            A copy of the input larry, collapsed to one dimension where the
            labels are tuples.
            
        See also
        --------
        la.larry.unflatten : Return an unflattened copy of larry.            
            
        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[1, 2],
               [3, 4]])
               
        >>> y.flatten()
        label_0
            ('a', 'c')
            ('a', 'd')
            ('b', 'c')
            ('b', 'd')
        x
        array([1, 2, 3, 4])
   
        """
        label = flattenlabel(self.label, order)
        x = self.x.flatten(order)
        return larry(x, label, validate=False)
        
    def unflatten(self):
        """
        Return an unflattened copy of larry.
        
        The larry to be unflattened must be in flattened form: 1d and label
        elements must be tuples containing the label elements of the
        corresponding data array element. Refer to the example below to see
        what a flattened array looks like.
        
        Returns
        -------
        y : larry
                    
        See also
        --------
        la.larry.flatten : Return a copy of the larry collapsed into one
                           dimension.
        
        Examples
        --------
        First create a flattened larry:
        
        >>> y = larry([[1, 2], [3, 4]], [['r0', 'r1'], ['c0', 'c1']])
        >>> yf = y.flatten()
        >>> yf
        label_0
            ('r0', 'c0')
            ('r0', 'c1')
            ('r1', 'c0')
            ('r1', 'c1')
        x
        array([1, 2, 3, 4])
        
        Then unflatten it:
        
        >>> yf.unflatten()
        label_0
            r0
            r1
        label_1
            c0
            c1
        x
        array([[ 1.,  2.],
               [ 3.,  4.]])
        
        """
        
        # Check input
        if self.ndim != 1:
            raise ValueError, 'Only 1d larrys can be unflattened.'
            
        if self.shape == (0,):            
            return larry([])
        else:
            # Determine labels, shape, and index into array
            if not isscalar(self.x.flat[0]):
                msg = 'Only scalar dtype is currently supported.'
                raise NotImplementedError, msg 
            labels = zip(*self.label[0])
            x, label = fromlists(self.x, labels)     
            return larry(x, label)
                        
    def insertaxis(self, axis, label):
        """
        Insert a new axis at the specified position.
        
        Parameters
        ----------
        axis : int
            The position to insert the new axis into.
        label : str, scalar, object, etc
            The label element of the new axis. The length of the new axis is
            always 1, so only one label element is needed.
            
        Returns
        -------
        y : larry
            A copy of the larry with a new axis inserted in the specified
            position.
            
        Examples
        --------
        Create a 1d larry and then insert a new axis in position 0:
                  
        >>> y = larry([1, 2, 3])
        >>> y.insertaxis(0, 'NEW')
        label_0
            NEW
        label_1
            0
            1
            2
        x
        array([[1, 2, 3]])

        Try inserting a new axis in position 1:
         
        >>> y.insertaxis(1, 'NEW')
        label_0
            0
            1
            2
        label_1
            NEW
        x
        array([[1],
               [2],
               [3]])
                   
        """
        if axis is None:
            raise ValueError, "`axis` cannot be None."
        x = self.getx(copy=True)
        x = np.expand_dims(x, axis)
        lab = self.copylabel()
        if int(axis) == -1:
            ax = len(lab)
        elif int(axis) < -1:
            ax = axis + 1
        else:
            ax = axis        
        lab.insert(ax, [label])
        return larry(x, lab)            
        
    # Conversion -------------------------------------------------------------

    def totuples(self):
        """
        Convert to a flattened list of tuples.
        
        See Also
        --------
        la.larry.fromtuples : Convert a list of tuples to a larry.
        la.larry.tolist : Convert to a flattened list.
        la.larry.todict : Convert to a dictionary.
        la.larry.tocsv : Save larry to a csv file.
        la.larry.tofile : Save 1d or 2d larry to text file.
        
        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y.totuples()
        [('a', 'c', 1), ('a', 'd', 2), ('b', 'c', 3), ('b', 'd', 4)]       
        
        """
        yf = self.flatten()
        z = zip(*yf.label[0])
        z.append(yf.x.tolist())
        return zip(*z)

    @staticmethod
    def fromtuples(data):
        """
        Convert a list of tuples to a larry.
        
        The input data, if there are N dimensions and M data points, should have
        this form::
        
         [(label0_1, label1_1, ..., labelN_1, value_1),
         (label0_2, label1_2, ..., labelN_2, value_2),
         ...
         (label0_M, label1_M, ..., labelN_M, value_M)]
        
        Parameters
        ----------
        data : list of tuples
            The input must be a list of tuples where each tuple represents one
            data point in the larry: (label0, label1, ..., labelN, value)   
            
        Returns
        -------
        y : larry
            A larry constructed from `data` is returned.
            
        See Also
        --------
        la.larry.totuples : Convert to a flattened list of tuples.
        la.larry.fromlist : Convert a list of tuples to a larry.
        la.larry.fromdict : Convert a dictionary to a larry.
        la.larry.fromcsv : Load a larry from a csv file. 

        Examples
        --------
        Convert a list of label, value pairs to a larry:
        
        >>> data = [('r0', 'c0', 1), ('r0', 'c1', 2), ('r1', 'c0', 3), ('r1', 'c1', 4)]
        >>> larry.fromtuples(data)
        label_0
            r0
            r1
        label_1
            c0
            c1
        x
        array([[ 1.,  2.],
               [ 3.,  4.]])
                
        What happens if we throw out the last data point? The missing value
        becomes NaN:       
                
        >>> data = data[:-1]
        >>> larry.fromtuples(data)
        label_0
            r0
            r1
        label_1
            c0
            c1
        x
        array([[  1.,   2.],
               [  3.,  NaN]])
                
        """        
        if len(data) == 0:        
            return larry([])                   
        else:            
            # Split data into label and x
            labels = zip(*data)
            xs = labels.pop(-1)               
            # Determine labels, shape, and index into array	
            x, label = fromlists(xs, labels)          
            return larry(x, label) 
        
    def tolist(self):
        """
        Convert to a flattened list.
        
        See Also
        --------
        la.larry.fromlist : Convert a flattened list to a larry.
        la.larry.totuples : Convert to a flattened list of tuples.
        la.larry.todict : Convert to a dictionary.
        la.larry.tocsv : Save larry to a csv file.
        la.larry.tofile : Save 1d or 2d larry to text file.
        
        Examples
        --------
        >>> y = larry([[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']])
        >>> y.tolist()
        [[1, 2, 3, 4], [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')]]       
        
        """
        yf = self.flatten()
        return [yf.x.tolist(), yf.label[0]]                                   

    @staticmethod
    def fromlist(data):
        """
        Convert a flattened list to a larry.

        The input data, if there are N dimensions and M data points, should have
        this form::

            [[value_1,  value_2,  ..., value_M],
            [(label0_1, label1_1, ..., labelN_1),
            (label0_2, label1_2, ..., labelN_2),
            ...
            (label0_M, label1_M, ..., labelN_M)]]    
        
        Parameters
        ----------
        data : list
            The input must be a list such as that returned by larry.tolist. See
            the example below.  
            
        Returns
        -------
        y : larry
            A larry constructed from `data` is returned.
            
        See Also
        --------
        la.larry.tolist : Convert to a flattened list.
        la.larry.fromtuples : Convert a list of tuples to a larry.
        la.larry.fromdict : Convert a dictionary to a larry.
        la.larry.fromcsv : Load a larry from a csv file.   

        Examples
        --------
        >>> data = [[1, 2, 3, 4], [('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd')]]
        >>> larry.fromlist(data)
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[ 1.,  2.],
               [ 3.,  4.]])
               
        """
        if len(data) == 0:            
            return larry([])           
        else:    
            x, label = fromlists(data[0], zip(*data[1]))      
            return larry(x, label)             

    def todict(self):
        """
        Convert to a dictionary.
        
        See Also
        --------
        la.larry.totuples : Convert to a flattened list of tuples.
        la.larry.tolist : Convert to a flattened list.
        la.larry.tocsv : Save larry to a csv file.
        la.larry.tofile : Save 1d or 2d larry to text file.
        
        Examples
        --------
        >>> y = larry([[1.0, 2.0], [3.0, 4.0]], [['a', 'b'], ['c', 'd']])
        >>> y.todict()
        {('b', 'c'): 3.0, ('a', 'd'): 2.0, ('a', 'c'): 1.0, ('b', 'd'): 4.0}     
        
        """
        ylist = self.tolist()
        return dict(zip(ylist[1], ylist[0]))

    @staticmethod    
    def fromdict(data):
        """
        Convert a dictionary to a larry.

        The input data, if there are N dimensions and M data points, should
        have this form::

            {(label0_1, label1_1, ..., labelN_1): value_1,
            (label0_2, label1_2, ..., labelN_2): value_2,
            ...
            (label0_M, label1_M, ..., labelN_M): value_M}   
        
        Parameters
        ----------
        data : dict
            The input must be a dictionary such as that returned by
            larry.todict See the example below. 
            
        Returns
        -------
        y : larry
            A larry constructed from `data` is returned.
            
        See Also
        --------
        la.larry.todict : Convert to a dictionary. 
        la.larry.fromtuples : Convert a list of tuples to a larry.
        la.larry.fromlist : Convert a list of tuples to a larry.
        la.larry.fromcsv : Load a larry from a csv file. 

        Examples
        --------
        >>> data = {('a', 'c'): 1, ('a', 'd'): 2, ('b', 'c'): 3, ('b', 'd'): 4}
        >>> larry.fromdict(data)
        label_0
            a
            b
        label_1
            c
            d
        x
        array([[ 1.,  2.],
               [ 3.,  4.]])
               
        """ 
        return larry.fromlist([data.values(), data.keys()])
        
    def tocsv(self, filename, delimiter=','):
        """
        Save larry to a csv file.
        
        The type information of the labels will be lost. So if a label element
        is, for example, an integer, a round trip (`tocsv` followed by
        `fromcsv`) will convert it to an integer. You can use the `maplabel`
        method to convert it back to an integer.
        
        As you can see from above, the tocsv and fromcvs methods are fragile.
        A more robust archiving solution is given by the IO class.        
        
        The format of the csv file is::

            label0, label1, ..., labelN, value
            label0, label1, ..., labelN, value
            label0, label1, ..., labelN, value
        
        Parameters
        ----------
        filname : str
            The filename of the csv file.
        delimiter : str
            The delimiter used to separate the labels elements from eachother
            and from the values.

        See Also
        --------
        la.larry.fromcsv : Load a larry from a csv file.
        la.larry.tofile : Save 1d or 2d larry to text file.
        la.IO: Save and load larrys in HDF5 format using a dictionary-like
               interface.         
        la.larry.totuples : Convert to a flattened list of tuples.
        la.larry.tolist : Convert to a flattened list.
        la.larry.todict : Convert to a dictionary.
            
        Examples
        --------        
        >>> y = larry([1, 2, 3], [['a', 'b', 'c']])
        >>> y.tocsv('/tmp/lar.csv')
        >>> larry.fromcsv('/tmp/lar.csv')
        label_0
            a
            b
            c
        x
        array([ 1.,  2.,  3.])
        
        """
        fid = open(filename, 'w')
        writer = csv.writer(fid, delimiter=delimiter)
        writer.writerows(self.totuples())
        fid.close()                   

    @staticmethod
    def fromcsv(filename, delimiter=',', skiprows=0):
        """
        Load a larry from a csv file.
        
        The type information of the labels is not contained in a csv file.
        Therefore, a label element that was, for example, an integer, will
        be converted to a string after a round trip (`tocsv` followed by
        `fromcsv`). You can use the `maplabel` methods to convert it back to
        an integer.
        
        Integer data values will be converted to floats.
        
        If a data value is missing, a ValueError will be raised. One label
        element per axis can be missing; the missing label element will be
        replace with the empty string ''.
        
        As you can see from above, the tocsv and fromcvs methods are fragile.
        A more robust archiving solution is given by the IO class.
        
        The format of the csv file is::

            label0, label1, ..., labelN, value
            label0, label1, ..., labelN, value
            label0, label1, ..., labelN, value
        
        Parameters
        ----------
        filname : str
            The filename of the csv file.
        delimiter : str
            The delimiter used to separate the labels elements from eachother
            and from the values.
        skiprows : int, optional
            Skip the first `skiprows` lines. No rows are skipped by default.          
            
        Raises
        ------
        ValueError
            If a data value is missing in the csv file.
            
        See Also
        --------
        la.larry.tocsv : Save larry to a csv file.
        la.IO: Save and load larrys in HDF5 format using a dictionary-like
               interface.
        la.larry.fromtuples : Convert a list of tuples to a larry.
        la.larry.fromlist : Convert a flattened list to a larry.
        la.larry.fromdict : Convert a dictionary to a larry.      
            
        Examples
        --------        
        >>> y = larry([1, 2, 3], [['a', 'b', 'c']])
        >>> y.tocsv('/tmp/lar.csv')
        >>> larry.fromcsv('/tmp/lar.csv')
        label_0
            a
            b
            c
        x
        array([ 1.,  2.,  3.])
        
        """
        fid = open(filename, 'r')
        reader = csv.reader(fid, delimiter=delimiter)
        [reader.next() for i in range(skiprows)]
        data = [row for row in reader]
        fid.close()
        return larry.fromtuples(data) 
    
    def tofile(self, file, delimiter=','):
        """
        Save 1d or 2d larry to text file (overwrites file if already exists).
        
        Parameters
        ----------
        file : {str, file object}
            A file name (str) or file object. If file object, then it will
            not be closed.
        delimiter : str
            The delimiter used to separate the elements in the file.

        See Also
        --------
        la.IO: Save and load larrys in HDF5 format using a dictionary-like
               interface.         
        la.larry.tocsv : Save larry to a csv file.
        la.larry.totuples : Convert to a flattened list of tuples.
        la.larry.tolist : Convert to a flattened list.
        la.larry.todict : Convert to a dictionary.
            
        Examples
        --------
        Create a larry:

        >>> lar = larry([[1, 2], [3, 4]], [['r1', 'r2'], ['c1', 'c2']])

        Pick a file name or file object or string buffer. Let's use a string
        buffer:

        >>> import StringIO
        >>> f = StringIO.StringIO()

        Write to file:

        >>> lar.tofile(f)

        Display file (or in this case, string buffer):

        >>> print f.getvalue()
        ,c1,c2
        r1,1,2
        r2,3,4

        """
        
        # Check number of dimensions in larry
        ndim = self.ndim
        if ndim not in (1, 2):
            msg = "Only 1d and 2d larrys supported; "
            msg +="try the IO function or tocsv method."
            raise ValueError, msg

        # Open file if needed
        if type(file) == str:
            f = open(file, 'w')
            opened = True
        else:
            f = file
            opened = False

        # Write data
        if ndim == 1:

            for i in range(self.size):
                line = [str(self.label[0][i]), str(self.x[i]) + '\n']
                line = delimiter.join(line)
                f.write(line)

        elif ndim == 2:
            
            # Column labels
            f.write(delimiter)
            line = []
            for i in range(self.shape[1]):
                line.append(str(self.label[1][i]))
            line = delimiter.join(line)
            line += '\n'
            f.write(line)
            
            # Row labels and array data
            for i in range(self.shape[0]):
                line = [str(self.label[0][i])]
                for j in range(self.shape[1]):
                    line.append(str(self.x[i,j]))
                line = delimiter.join(line)
                line += '\n'
                f.write(line)
        
        else:

            if opened:
                f.close()
            msg = "Please report this bug; the code should never reach here."
            raise RuntimeError, msg

        # Close file if opened (i.e., if file was a str)
        if opened:
            f.close()

               
    # Copy -------------------------------------------------------------------
          
    def copy(self):
        """
        Return a copy of a larry.

        Examples
        --------
        >>> y = larry([1, 2], [['a', 'b']])        
        >>> z = y.copy()
        >>> z
        label_0
            a
            b
        x
        array([1, 2])
            
        """
        label = [list(z) for z in self.label]
        x = self.x.copy()
        return larry(x, label, validate=False)
        
    def copylabel(self):
        """
        Return a copy of a larry's label.
        
        Examples
        --------        
        >>> y = larry([1, 2], [['a', 'b']])        
        >>> label = y.copylabel()
        >>> label
        [['a', 'b']]
        
        """
        return [list(z) for z in self.label]
        
    def copyx(self):
        """Return a copy of a larry's data as a Numpy array.

        Examples
        --------        
        >>> y = larry([1, 2], [['a', 'b']])  
        >>> x = y.copyx()
        >>> x
        array([1, 2])
        
        """
        return self.x.copy()    
        
    # Print ------------------------------------------------------------------
                
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
        
        # x
        x.append('x\n')
        x.append(repr(self.x))
        return ''.join(x)        


# Label indexing support functions for the lix method ------------------------

class Getitemlabel(object):
    "Utility class for the lix method."
    
    def __init__(self2, self):
        self2.lar = self
        
    def __getitem__(self2, index):
        y = self2.lar
        if y.shape == (0,):
            return self2.lar
        elif 0 in y.shape:
            msg = 'lix does not support shapes that contain 0 '
            msg += 'such as (0,) and (2, 0 ,3).'
            raise ValueError, msg
        typ = type(index)
        if typ == list:
            # Example: lar.lix[['a', 'b', 'c']]
            index2 = labels2indices(y.label[0], index)
            if len(index) == 1:       
                index2 = index2[0]
            return y[index2]               
        elif typ == slice:
            # Examples: lar.lix[['a']:], lar.lix[['a']:['b']],
            #           lar.lix[['a']:['b']:2], lar.lix[2:['b']] 
            return y[slicemaker(index, y.labelindex, 0)]
        elif typ == tuple:
            index2 = []
            label = []
            if len(index) == y.ndim:
                index3 = index
            elif len(index) < y.ndim:
                index3 = list(index) + [slice(None)] * (y.ndim - len(index))
            else:
                raise IndexError, 'Invalid index'        
            for ax, idx in enumerate(index3):
                typ = type(idx)
                if typ == list:
                    idx2 = labels2indices(y.label[ax], idx)
                    if len(idx) > 1:      
                        label.append(idx)
                    index2.append(idx2)
                elif typ == slice: 
                    s = slicemaker(idx, y.labelindex, ax)
                    slar = range(*s.indices(y.shape[ax]))
                    lab = y.label[ax][s]
                    if len(lab) > 1:
                        label.append(lab)        
                    index2.append(slar)
                elif isscalar(idx):
                    index2.append([idx])    
                else:
                    raise IndexError, 'Unsupported indexing operation.'
            x = np.squeeze(y.x[np.ix_(*index2)])
            if x.ndim == 0:
                return x[()]
            else:    
                return larry(x, label)                       
        elif isscalar(index):
            # Example: lar.lix[0]
            return y[index]             
        else:
            raise IndexError, 'Unsupported indexing operation.'           
    
def slicemaker(index, labelindex, axis): 
    "Convert a slice that may contain labels to a slice with indices."
    msg1 = 'The %s element of a slice must be a list or a scalar.'
    msg2 = 'The %s element of the slice contains more than one item.'
    if index.start is None:
        start = None
    elif type(index.start) is list:
        if len(index.start) > 1:
            raise ValueError, msg2 % 'start'    
        start = labelindex(index.start[0], axis=axis)
    elif isscalar(index.start):
        start = index.start
    else:
        raise ValueError, msg1 % 'start'    
    if index.stop is None:
        stop = None
    elif type(index.stop) is list:
        if len(index.stop) > 1:
            raise ValueError, msg2 % 'start'    
        stop = labelindex(index.stop[0], axis=axis)
    elif isscalar(index.stop):
        stop = index.stop                                                  
    else:
        raise ValueError, msg1 % 'stop'
    return slice(start, stop, index.step)        

def labels2indices(label, labels):
    "Convert list of labels to indices"
    try:
        indices = map(label.index, labels)
    except ValueError:
        raise ValueError, 'Could not map label to index value.'
    return indices  
        
