"Information about your particular installation of the la package."

from la.external.prettytable import indent
    
def info():
    "la package information such as version number and HDF5 availability."

    # la version and file
    import la

    # Are you using the C or Python version of functions
    from la.flabel import listmap, listmap_fill
    version = ('Slower Python version', 'Faster C version')
    listmap = listmap.__module__.split('.')[-1] == 'cflabel'
    listmap_fill = listmap_fill.__module__.split('.')[-1] == 'cflabel'
    listmap = version[listmap]
    listmap_fill = version[listmap_fill]
    
    # IO
    try:
        from la import IO
        io = "Available"
        try:
            import h5py
            version = h5py.version.version
            io = io + " (h5py %s)" % str(version)
        except:
            pass
    except ImportError:
        io = "Not available"

    # Others
    import numpy as np
    import bottleneck as bn
    
    # Make and print report
    table = []
    table.append(['la', la.__version__])
    table.append(['la file', la.__file__])
    table.append(['NumPy', np.__version__])
    table.append(['Bottleneck', bn.__version__])
    table.append(['HDF5 archiving', io])  
    table.append(['listmap', listmap])
    table.append(['listmap_fill', listmap_fill])         
    print indent(table, hasHeader=False, delim='  ')          
