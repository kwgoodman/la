
===============
Unary functions
===============

The unary functions (such as log, sqrt, sign) operate on a single larry and
do not change its shape or ordering. For example:
::
    >>> from la import larry
    >>> y = larry([-1,2,-3,4])
    >>> y.sign()
    label_0
        0
        1
        2
        3
    x
    array([-1,  1, -1,  1])
    
Method summary
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   larry.sqrt
   larry.clip
   larry.sign
    




