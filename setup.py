#!/usr/bin/env python

"""PyFFTW: Python bindings to the FFTW3 C-library

PyFFTW provieds Python bindings to the FFTW3  "Fastest Fourier Transform in 
the West." C-library(http://www.fftw.org/) for computing discrete Fourier 
transforms. It uses numpy and ctypes and includes a somewhat pythonic interface
to the FFTW routines, but leaves the concept of creating plans and executing 
these plans intact.
"""

classifiers = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License (GPL)
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
"""



from distutils.core import setup
from distutils.log import warn
from distutils.command.build_py import build_py
import ctypes
from ctypes import util
import os
import platform
import sys

package_data = {}
if os.name=='nt' or platform.system()=='Windows':
    # Assuming that the dll content of 
    #  ftp://ftp.fftw.org/pub/fftw/fftw-3.2.2.pl1-dll32.zip
    # is copied to the current working directory.
    # FFTW_PATH should be either the final installation directory
    # of the dll files or empty.
    try:
        FFTW_PATH = os.environ['FFTW_PATH']
    except KeyError:
        FFTW_PATH = r''
    packages_library_names = {'fftw3': 'libfftw3-3.dll',
                              'fftw3f' : 'libfftw3f-3.dll',
                              'fftw3l': 'libfftw3l-3.dll'}
    for l, dll in packages_library_names.items():
        s = os.path.join (FFTW_PATH, dll)
        package_data[l] = [s]
elif platform.system()=='Darwin':
    try:
        FFTW_PATH = os.environ['FFTW_PATH']
    except KeyError:
        FFTW_PATH=r'/Users/ruediger/Downloads/gigi/lib/lib'
    packages_library_names = {'fftw3': 'libfftw3.dylib', 'fftw3f' : 'libfftw3f.dylib',
                              'fftw3l': 'libfftw3l.dylib'}
else:
    try:
        FFTW_PATH = os.environ['FFTW_PATH']
    except KeyError:
        FFTW_PATH = r'/usr/lib/'
    packages_library_names = {'fftw3': 'libfftw3.so', 'fftw3f' : 'libfftw3f.so',
                              'fftw3l': 'libfftw3l.so'}

_complex_typedict = {'fftw3':'complex', 'fftw3f': 'singlecomplex', 'fftw3l': 'longcomplex'}
_float_typedict = {'fftw3': 'double', 'fftw3f': 'single', 'fftw3l': 'longdouble'}
packages_0 = ['fftw3','fftw3f', 'fftw3l']
# To used threads, p+'_threads' library must exist in the system where p in packages_0.

def create_source_from_template(tmplfile, outfile, lib, libname, _complex,
                                _float, location):
    fp = open(tmplfile, 'r')
    tmpl = fp.read()
    fp.close()
    mod = tmpl.replace('$libname$',libname).replace('$complex$',_complex).\
            replace('$library$',packages_library_names[lib]).replace('$float$',_float).\
            replace('$libraryfullpath$', location)
    fp = open(outfile, 'w')
    fp.write(mod)
    fp.close()
    print "build %s from template %s" %(outfile, tmplfile)

def check_libs(packages):
    libs = {}
    for name in packages[:]:
        try:
            libpath = os.path.join(FFTW_PATH, packages_library_names[name])
            if libpath == None:
                raise OSError
            lib = ctypes.cdll.LoadLibrary(libpath)
            print "found %s at %s" %(name, libpath)
        except OSError, e:
            warn("%s is not located at %s, trying util.find_library(%s)"
                 %(name, libpath, name))
            try:
                libpath = util.find_library(name)
                if libpath == None:
                    raise OSError
                lib = ctypes.cdll.LoadLibrary(libpath)
                print "found %s at %s" %(name, libpath)
            except (TypeError,OSError), e:
                warn("Not installing bindings for %s, because could not load\
                     the library: %s\n if you know the library is installed\
                     you can specify the absolute path in setup.py" % (name, e))
                packages.remove(name)
        libs[name] = libpath
    return packages, libs

packages, liblocation = check_libs(packages_0)
if len(packages) < 1:
    packages = ['None']

class build_from_templates(build_py):
    def build_module(self, module, module_file, package):
        if package == 'None':
            raise ValueError('no FFTW files found')
        module, ext = os.path.splitext(module)
        if not ext == '.tmpl':
            outfile = self.get_module_outfile(self.build_lib, [package], module)
            dir = os.path.dirname(outfile)
            self.mkpath(dir)
            return self.copy_file(module_file, outfile, preserve_mode=0)
        else:
            outfile = self.get_module_outfile(self.build_lib, [package], module)
            dir = os.path.dirname(outfile)
            self.mkpath(dir)
            return create_source_from_template(module_file, outfile, package,
                                               package.replace('3',''),
                                               _complex_typedict[package],
                                               _float_typedict[package],
                                               liblocation[package])

doclines = __doc__.split("\n")

setup(name='PyFFTW3',
      version = '0.2.2',
      platforms = ["any"],
      description = doclines[0],
      classifiers = filter(None, classifiers.split("\n")),
      long_description = "\n".join(doclines[2:]),
      author = 'Jochen Schroeder',
      author_email = 'cycomanic@gmail.com',
      url = 'www.launchpad.net/pyfftw',
      packages = packages,
      package_dir = dict([(n, 'src/templates') for n in packages]),
      package_data = {},
      cmdclass = {"build_py": build_from_templates},
      license ='GPL v3'
     )
