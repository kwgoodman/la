.. _functions:

===============
larry functions
===============

The functions that operate on larry can be divided into the following broad
categories:

.. contents:: Functions and examples

Below you'll find the functions in each category along with examples. All
of the examples assume that you have already imported larry:
::
    >>> from la import larry
    
The reference guide for the larry methods, as opposed to functions, can be
found in :ref:`reference`.


Alignment
---------

The alignment functions help you align one of more larrys.

------------
             
.. autofunction:: la.align

------------
             
.. autofunction:: la.align_raw

------------
             
.. autofunction:: la.union

------------

.. autofunction:: la.intersection


.. _binaryfunc:

Binary functions
----------------

The binary functions combine two larrys into one.

------------
             
.. autofunction:: la.binaryop

------------
             
.. autofunction:: la.add

------------
             
.. autofunction:: la.subtract

------------
             
.. autofunction:: la.multiply

------------
             
.. autofunction:: la.divide


Random
------

Functions that return larrys containing random samples.

------------

.. autofunction:: la.rand

------------

.. autofunction:: la.randn


Misc
----

Miscellaneous functions that operate on larrys. 

------------

.. autofunction:: la.unique

------------

.. autofunction:: la.stack

------------

.. autofunction:: la.panel

------------

.. autofunction:: la.cov


