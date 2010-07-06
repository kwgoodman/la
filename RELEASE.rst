
=============
Release Notes
=============

These are the major changes made in each release. For details of the changes
see http://github.com/kwgoodman/la

la 0.4 (celery)
===============

*Release date: 2010-07-06*

The focus of this release was binary operations between unaligned larrys with
user control of the join method (five available) and the fill method. A
general binary function, la.binaryop(), was added as were the convenience
functions add, subtract, multiply, divide. Supporting functions such as
la.align(), which aligns two larrys, were also added.

**New larry methods**

- ismissing: A bool larry with element-wise marking of missing values
- take: A copy of the specified elements of a larry along an axis

**New functions**

- rand: Random samples from a uniform distribution
- randn: Random samples from a Gaussian distribution
- missing_marker: Return missing value marker for the given larry
- ismissing: A bool Numpy array with element-wise marking of missing values
- correlation: Correlation of two Numpy arrays along the specified axis
- split: Split into train and test data along given axis
- listmap_fill: Index map a list onto another and index of unmappable elements
- listmap_fill: Cython version of listmap_fill
- align: Align two larrys using one of five join methods
- info: la package information such as version number and HDF5 availability
- binaryop: Binary operation on two larrys with given function and join method
- add: Sum of two larrys using given join and fill methods
- subtract: Difference of two larrys using given join and fill methods
- multiply: Multiply two larrys element-wise using given join and fill methods
- divide: Divide two larrys element-wise using given join and fill methods

**Enhancements**

- listmap now has option to ignore unmappable elements instead of KeyError
- listmap.pyx now has option to ignore unmappable elements instead of KeyError
- larry.morph() is much faster as are methods, such as merge, that use it

**Breakage from la 0.3**

- Development moved from launchpad to github
- func.py and afunc.py renamed flarry.py and farray.py to match new flabel.py.
  Broke: "from la.func import stack"; Did not break: "from la import stack"
- Default binary operators (+, -, ...) no longer raise an error when no labels
  overlap 

**Bug fixes**

- #590270 Index with 1d array bug: lar[1darray,:] worked; lar[1darray] crashed


la 0.3 (banana)
===============

*Release date: 2010-06-04*

**New larry methods**

- astype: Copy of larry cast to specified type
- geometric_mean: new method based on existing array function

**New functions**

- la.util.resample.cross_validation: k-fold cross validation index iterator
- la.util.resample.bootstrap: bootstrap index iterator
- la.util.misc.listmap: O(n) version of map(list1.index, list2)
- la/src/clistmap.pyx: Cython version of listmap with python fallback

**Enhancements**

- Major performance boost in most larry methods!
- You can now use an optional dtype when creating larrys
- You can now optionally skip the integrity test when creating a new larry
- Add ability to compare (==, >, !=, etc) larrys with lists and tuples
- Documentation and unit tests

**Breakage from la 0.2**

- lastrank and lastrank_decay methods combined into one method: lastrank
- Given shape (n,m) input, lastrank now returns shape (n,) instead of (n,1)
- geometric_mean now reduces input in the same way as lastrank (see above)

**Bug fixes**

- #571813 Three larry methods crashed on 1d input
- #571737 skiprows missing from parameters section of the fromcsv doc string
- #571899 label indexing fails when larry is 3d and index is a tuple of len 2
- #571830 prod, cumprod, and cumsum did not return NaN for all-NaN input
- #572638 lastrank chokes on input with a shape tuple that contains zero
- #573240 Reduce methods give wrong output with shapes that contain zero
- #582579 la.afunc.nans: wrong output for str and object dtype
- #583596 assert_larry_equal crashed when comparing float larry to str larry
- #585694 cumsum and cumprod crashed on dtype=int


la 0.2 (avocado)
================

*Release date: 2010-04-27*

**New larry methods**

- lix : Index into a larry using labels or index numbers or both
- swapaxes : Swap the two specified axes
- sortaxis : Sort data (and label) according to label along specified axis
- flipaxis : Reverse the order of the elements along the specified axis
- tocsv : Save larry to a csv file
- fromcsv : Load a larry from a csv file
- insertaxis : Insert a new axis at the specified position
- invert : Element by element inverting of True to False and False to True

**Enhancements**

- All larry methods can now take nd input arrays (some previously 2d only)
- Added ability to save larrys with datetime.date labels to HDF5
- New function (panel) to convert larry of shape (n, m, k) to shape (m*k, n)
- Expanded documentation
- Over 280 new unit tests; testing easier with new assert_larry_equal function

**Bug fixes**

- #517912: larry([]) == larry([]) raised IndexError
- #518096: larry.fromdict failed due to missing import
- #518106: la.larry.fromdict({}) failed
- #518114: fromlist([]) and fromtuples([]) failed
- #518135: keep_label crashed when there was nothing to keep
- #518210: sum, std, var returned NaN for empty larrys; now return 0.0 
- #518215: unflatten crashed on an empty larry
- #518442: sum, std, var returned NaN for shapes that contain zero: (2, 0, 3)
- #568175: larry.std(axis=-1) and var crashed on negative axis input
- #569622: Negative axis input gave wrong output for several larry methods


la 0.1 (first release)
======================

*Release date: 2010-02-03*

This is the first release of the la package.
