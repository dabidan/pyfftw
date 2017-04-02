#   This file is part of PyFFTW.
#
#    Copyright (C) 2009 Jochen Schroeder
#    Email: jschrod@berlios.de
#
#    PyFFTW is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    PyFFTW is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with PyFFTW.  If not, see <http://www.gnu.org/licenses/>.

import ctypes
from ctypes import pythonapi, util, py_object
from numpy import ctypeslib, typeDict
from platform import system as psystem
from os.path import splitext, join, isfile, dirname, abspath, basename
from os.path import join as joinpath
from os import name as osname
from os import environ
from warnings import warn

try:
    fftw_path = environ['FFTW_PATH']
    libfullpath = joinpath(abspath(fftw_path),r'$library$')
    if not isfile(libfullpath):
        raise IOError
except KeyError:
    libfullpath = r'$libraryfullpath$'
except IOError:
    warn('could not find %s in FFTW_PATH using installtime path'
             %'$library$')
    libfullpath = r'$libraryfullpath$'

if not isfile(libfullpath) and (osname=='nt' or psystem=='Windows'):
    if isfile(joinpath(dirname(__file__), libfullpath)):
        libfullpath = joinpath(dirname(__file__), libfullpath)

# must use ctypes.RTLD_GLOBAL for threading support
ctypes._dlopen(libfullpath, ctypes.RTLD_GLOBAL)
lib = ctypes.cdll.LoadLibrary(libfullpath)
#check if library is actually loaded there doesn't seem to be a better way to
#do this in ctypes
if not hasattr(lib, '$libname$_plan_dft_1d'):
    raise OSError('Could not load $library$')

if osname == 'nt' or psystem() == 'Windows':
    lib_threads = lib
else:
    libbase, dot, ext = basename(libfullpath).partition('.')
    libdir = dirname(libfullpath)
    lib_threads = joinpath(libdir, libbase + '_threads.'+ ext)
    try:
        lib_threads = ctypes.cdll.LoadLibrary(lib_threads)
    except OSError, e:
        warn("Could not load threading library %s, threading support is disabled"
            %lib_threads)
        lib_threads = None

typedict = {'c':typeDict['$complex$'], 'r':typeDict['$float$']}


def _strfunc(func, *argtypes):
    func.argtypes = argtypes
    func.restype = ctypes.POINTER(ctypes.c_char)
    def call(*args):
        ptr = func(*args)
        result = ctypes.cast(ptr, ctypes.c_char_p).value
        lib.free(ptr)
        return result
    return call

def set_argtypes(val, types):
    args = [ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypeslib.ndpointer(dtype=typedict[types[0]], flags='contiguous, writeable, aligned'),
        ctypeslib.ndpointer(dtype=typedict[types[1]],flags='contiguous, writeable, aligned'),
        ctypes.c_uint]
    if types == 'cc':
        args.insert(-1, ctypes.c_int)
    elif types == 'rr':
        args.insert(-1, ctypes.POINTER(ctypes.c_int))
    val.argtypes = args
    val.restype = ctypes.c_void_p
    return val

plan_dft = set_argtypes(lib.$libname$_plan_dft, 'cc')
plan_dft_r2c = set_argtypes(lib.$libname$_plan_dft_r2c, 'rc')
plan_dft_c2r = set_argtypes(lib.$libname$_plan_dft_c2r, 'cr')
plan_r2r = set_argtypes(lib.$libname$_plan_r2r, 'rr')


def set_argtypes_adv(val, types):
    args = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypeslib.ndpointer(dtype=typedict[types[0]], flags='contiguous, writeable, aligned'),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypeslib.ndpointer(dtype=typedict[types[1]],flags='contiguous, writeable, aligned'),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
        ctypes.c_uint
    ]
    if types == 'cc':
        args.insert(-1, ctypes.c_int)
    elif types == 'rr':
        args.insert(-1, ctypes.POINTER(ctypes.c_int))
    val.argtypes = args
    val.restype = ctypes.c_void_p
    return val

plan_many_dft = set_argtypes_adv(lib.$libname$_plan_many_dft, 'cc')
plan_many_dft_r2c = set_argtypes_adv(lib.$libname$_plan_many_dft_r2c, 'rc')
plan_many_dft_c2r = set_argtypes_adv(lib.$libname$_plan_many_dft_c2r, 'cr')
plan_many_r2r = set_argtypes_adv(lib.$libname$_plan_many_r2r, 'rr')

#malloc and free
lib.$libname$_malloc.restype = ctypes.c_void_p
lib.$libname$_malloc.argtypes = [ctypes.c_int]
lib.$libname$_free.restype = None
lib.$libname$_free.argtypes = [ctypes.c_void_p]

#create a buffer from memory (necessary for array allocation)
PyBuffer_FromReadWriteMemory = pythonapi.PyBuffer_FromReadWriteMemory
PyBuffer_FromReadWriteMemory.restype = py_object
PyBuffer_FromReadWriteMemory.argtypes = [ctypes.c_void_p, ctypes.c_int]

#executing arrays
execute = lib.$libname$_execute
execute.restype = None
execute.argtypes = [ctypes.c_void_p]

#guru execution
execute_dft = lib.$libname$_execute_dft
execute_dft.restype = None
execute_dft.argtypes = [ctypes.c_void_p,
    ctypeslib.ndpointer(flags='aligned, contiguous, writeable'),
    ctypeslib.ndpointer(flags='aligned, contiguous, writeable')]

#destroy plans
destroy_plan = lib.$libname$_destroy_plan
destroy_plan.restype = None
destroy_plan.argtypes = [ctypes.c_void_p]

#enable threading for plans
def plan_with_nthreads(nthreads):
    if nthreads > 1:
        raise ValueError("Cannot use more than 1 thread for non-threaded $libname$: %s" % (nthreads, ))

if lib_threads is not None:
    lib_threads.$libname$_init_threads.restype = ctypes.c_int
    lib_threads.$libname$_init_threads.argtypes = []
    s = lib_threads.$libname$_init_threads()
    if not s:
        sys.stderr.write('$libname$_init_threads call failed, disabling threads support\n')
        lib_threads = None
    else:
        plan_with_nthreads = lib_threads.$libname$_plan_with_nthreads
        plan_with_nthreads.restype = None
        plan_with_nthreads.argtypes = [ctypes.c_int]
        lib_threads.$libname$_cleanup_threads.restype = None
        lib_threads.$libname$_cleanup_threads.argtypes = []

sprintf_plan = _strfunc(lib.$libname$_sprint_plan, ctypes.c_void_p)

#wisdom

# create c-file object from python
PyFile_AsFile = pythonapi.PyFile_AsFile
PyFile_AsFile.argtypes = [ctypes.py_object]
PyFile_AsFile.restype = ctypes.c_void_p

#export to string
export_wisdom_to_string = _strfunc(lib.$libname$_export_wisdom_to_string)

#import from string
import_wisdom_from_string = lib.$libname$_import_wisdom_from_string
import_wisdom_from_string.argtypes = [ctypes.c_char_p]
import_wisdom_from_string.restype = ctypes.c_int

#import system wisdom
import_system_wisdom = lib.$libname$_import_system_wisdom
import_system_wisdom.restype = ctypes.c_int
import_system_wisdom.argtypes = None

#forget wisdom
forget_wisdom = lib.$libname$_forget_wisdom
forget_wisdom.restype = None
forget_wisdom.argtype = None
