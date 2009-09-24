"Export code, make tarball"

import tarfile
import os
from shutil import rmtree

try:
    import pysvn
except:
    print 'Cannot import pysvn'    


def svn_export(svnrepos='file:///ba/svn/la/trunk/',
               dirroot='/ba/send',
               packagename='la'):
    """
    Exports a clean directory tree from the source specified by
    svnrepos at HEAD, into dirroot and tar.bz2 it.
    
    Parameters
    ----------
    svnrepos : str
        Path to svn repository directory.
    dirroot : str
        Path where package is exported.
    packagename : str
         Name of package.
    """
        
    rmtree('%s/%s' % (dirroot, packagename), ignore_errors=True)

    # svn export      
    print 'Exporting code from repository'
    revision = export(svnrepos, dirroot + '/' + packagename)

    # Make version.py
    print 'Creating version file'
    vfile = open(dirroot + '/' + packagename + '/version.py', 'w')
    msg = '\"Package %s svn revision number\"\n\nversion = %s'
    vfile.write(msg % (packagename, str(revision.number)))
    vfile.close() 

    # tar.bz2
    print 'Making %s%d.tar.bz2' % (packagename, revision.number)
    archivename = packagename + str(revision.number) + '.tar.bz2'
    tarbz2(dirroot, archivename, packagename)

def export(path1, path2):
    """Export package
    
    Parameters
    ----------
    path1 : str
        Path to working copy directory
    path2 : str
        Path where package is exported    
    """
    client = pysvn.Client()
    revision = client.export(path1, path2, force=True)  
    # Workaround for windows
    if revision.number == -1:
        revision = client.info(path1).revision
    return revision 
  
def tarbz2(dirroot, archivename, packagename):
    """Archive directory
    
    Parameters
    ----------
    dirroot : str
        Directory to be archived
    archivename : str
        Name of archive
    packagename : str
         Name of package (e.g. am)       
    """
    cwd = os.getcwd()
    os.chdir(dirroot)
    tar = tarfile.open(archivename, 'w|bz2')
    tar.add(packagename, recursive=True)
    tar.close()
    os.chdir(cwd)
