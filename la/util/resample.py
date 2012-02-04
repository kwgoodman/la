"Resample index iterators"

import numpy as np
        

def cross_validation(n, kfold, shuffle=None):
    """
    K-fold cross validation iterator of training and testing indices.
    
    The division of the possible indices into k folds is random.
    
    Parameters
    ----------
    n : int
        Number of elements in total. Must be two or greater.
    kfold : int
        Number of folds. Must be two or greater.
    shuffle : function, optional
        A shuffle method that shuffles lists in place. If no shuffle function
        is given (default) then the following shuffle function will used:
        np.random.shuffle 
        
    Returns
    -------
    idx_train : list
        The training index. Each of the `kfold` iterations returns training
        and testing indices.
    idx_test : list     
        The testing index. Each of the `kfold` iterations returns training
        and testing indices.
        
    Examples
    --------
    K-fold cross validation indices for 5 elements and 2 folds:
     
    >>> from la.util.resample import cross_validation
    >>> for train, test in cross_validation(5,2):
    ...     print
    ...     print 'train: ', train
    ...     print 'test:  ', test
    ... 

    train:  [4, 3, 1]
    test:   [0, 2]

    train:  [0, 2]
    test:   [4, 3, 1]
            
    """
    if n < 2:
        raise ValueError, "`n` must be two or greater."
    if kfold < 2:
        raise ValueError, "`kfold` must be two or greater."
    if kfold > n:
        raise ValueError, "`kfold` cannot be greater than `n`"
    index = range(n)
    if shuffle == None:
        np.random.shuffle(index)
    else:
        shuffle(index)
    nperk = int(1.0 * n / kfold)   
    remainder = n - nperk * kfold
    for k in range(kfold):
        idx1 = nperk * k
        idx2 = idx1 + nperk
        if k == kfold - 1:
            idx2 = idx2 + remainder
        idx_test = index[idx1:idx2]
        idx_train = index[:idx1] + index[idx2:]
        yield idx_train, idx_test
        
def bootstrap(n, nboot, randint=None):
    """
    Bootstrap iterator for training and testing indices.
    
    Parameters
    ----------
    n : int
        Number of elements in total. Must be two or greater.
    nboot : int
        Number of bootstrap samples. Must be one or greater.
    randint : function, optional
        A randint function that behaves like numpy.random.randint. If no
        randint function is given (default) then the following will be used:
        np.random.randint
        
    Returns
    -------
    idx_train : list
        The training index. Each of the `nboot` iterations returns training
        and testing indices.
    idx_test : list     
        The testing index. Each of the `nboot` iterations returns training
        and testing indices.
        
    Examples
    --------
    Three bootstrap samples taken with replacement from four elements:
    
    >>> from la.util.resample import bootstrap 
    >>> for train, test in bootstrap(4, 3):
    ...     print
    ...     print 'train: ', train
    ...     print 'test:  ', test
    ... 

    train:  [2 1 3 1]
    test:   [0]

    train:  [1 1 2 1]
    test:   [0, 3]

    train:  [1 3 0 0]
    test:   [2]
            
    """
    if n < 2:
        raise ValueError, "`n` must be at least 2"
    if nboot < 1:
        raise ValueError, "`nboot` must be at least 1" 
    if randint == None:
        randint2 = np.random.randint
    else:
        randint2 = randint                  
    nrange = set(range(n))
    count = 0
    while 1:
        idx_train = randint2(0, n, n).tolist()
        idx_test = list(nrange - set(idx_train))
        if len(idx_test) > 0:
            yield idx_train, idx_test
            count += 1
        if count >= nboot:
            break            

def split(data, idx_train, idx_test, axis):
    """
    Split into train and test data along give axis.
    
    Parameters
    ----------
    data : larry, Numpy array
        The data to split into train and test data. `data` can be any object
        that has a take method similar to numpy.ndarray.take().
    idx_train : array_like
        The indices of the training data.
    idx_test : array_like
        The indices of the testing data.
    axis : int
        The axis along which the data is split.
        
    Returns
    -------
    train : larry, Numpy array
        The training data.
    test : larry, Numpy array
        The testing data.                        
    
    Examples
    --------
    Create train and test larrys for leave-one-out cross validation from a
    1d larry of length 4:
    
    >>> from la.util.resample import cross_validation, split
    >>> y = la.larry([1, 2, 3, 4])
    >>> n = 4
    >>> kfold = 4
    >>> cv = cross_validation(n, kfold)
    >>> for idx_train, idx_test in cv:
    ...     ytrain, ytest = split(y, idx_train, idx_test, axis=0)
    ...     print '-' * 10
    ...     print 'ytrain:'
    ...     print ytrain
    ...     print 'ytest:'
    ...     print ytest
    ... 
    ----------
    ytrain:
    label_0
        0
        3
        2
    x
    array([1, 4, 3])
    ytest:
    label_0
        1
    x
    array([2])
    ----------
    ytrain:
    label_0
        0
        3
        2
    x
    array([1, 4, 3])
    ytest:
    label_0
        1
    x
    array([2])
    ----------
    ytrain:
    label_0
        0
        3
        2
    x
    array([1, 4, 3])
    ytest:
    label_0
        1
    x
    array([2])
    ----------
    ytrain:
    label_0
        0
        3
        2
    x
    array([1, 4, 3])
    ytest:
    label_0
        1
    x
    array([2])
    
    """
    return data.take(idx_train, axis), data.take(idx_test, axis)
    
