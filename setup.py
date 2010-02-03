#!/usr/bin/env python

import os
from distutils.core import setup

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "Intended Audience :: Developers",
               "License :: OSI Approved :: Simplified BSD",
               "Programming Language :: Python",
               "Topic :: Software Development",
               "Topic :: Scientific/Engineering",
               "Operating System :: POSIX",
               "Operating System :: Unix", 
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows"]

# Get la description
doc_file = os.path.join('doc', 'launchpad.txt')
fid = file(doc_file, 'r')
DOC = fid.read()
fid.close()
DOC = DOC.split("\n")

# Get la version
ver_file = os.path.join('la', 'version.py')
fid = file(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')
VER = [int(z) for z in VER]

NAME                = 'la'
MAINTAINER          = "Keith Goodman"
MAINTAINER_EMAIL    = "larry-discuss@lists.launchpad.net"
DESCRIPTION         = DOC[0]
LONG_DESCRIPTION    = "\n".join(DOC[2:])
URL                 = "http://larry.sourceforge.net"
DOWNLOAD_URL        = ""
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Archipel Asset Management AB"
AUTHOR_EMAIL        = "kwgoodman@gmail.com"
PLATFORMS           = ["Linux", "Solaris", "Mac OS-X", "Unix", "Windows"]
MAJOR               = VER[0]
MINOR               = VER[1]
MICRO               = VER[2]
ISRELEASED          = False
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
PACKAGES            = ["la", "la/tests", "la/util", "la/util/tests"]
PACKAGE_DATA        = {'la': ['LICENSE']}
REQUIRES            = ["numpy"]


setup(name=NAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      platforms=PLATFORMS,
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      requires=REQUIRES,
      zip_safe=False
     )

