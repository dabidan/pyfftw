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

import numpy as np
import lib
from lib import typedict, plan_with_nthreads, sprintf_plan, plan_dft, plan_dft_r2c, plan_dft_c2r, plan_r2r
from ctypes import byref, c_double, c_int

fftw_flags = {'measure': 0,
              'destroy input': 1,
              'unaligned': 2,
              'conserve memory': 4,
              'exhaustive': 8,
              'preserve input': 16,
              'patient': 32,
              'estimate': 64}

realfft_type = {'halfcomplex r2c': 0,
                'halfcomplex c2r': 1,
                'discrete hartley': 2,
                'realeven 00': 3,
                'realeven 01': 4,
                'realeven 10': 5,
                'realeven 11': 6,
                'realodd 00': 7,
                'realodd 01': 8,
                'realodd 10': 9,
                'realodd 11': 10}


fft_direction = {'forward' : -1, 'backward': 1}

def create_aligned_array(shape, dtype=typedict["c"], boundary=16):
    """Create an array which is aligned in memory

    Parameters:
        shape       --  the shape of the array 
        dtype       --  the dtype of the array (default=typedict["c"])
        boundary    --  the byte boundary to align to (default=16)
    """
    N = np.prod(shape)*np.array(1,dtype).nbytes
    tmp = np.zeros(N+boundary, dtype=np.uint8)
    address = tmp.__array_interface__['data'][0]
    offset = (boundary - address % boundary)
    return tmp[offset:offset + N].view(dtype=dtype).reshape(shape)


def execute(plan):
    """Execute fftw-plan, i.e. perform Fourier transform on the arrays given
    when the plan was created"""
    lib.execute(plan)

def guru_execute_dft(plan, inarray, outarray):
    """Guru interface: perform Fourier transform on two arrays,
    outarray=fft(inarray) using the given plan. Important: This function
    does not perform any checks on the array shape and alignment for
    performance reasons. It is therefore crucial to only provide arrays
    with the same shape, dtype and alignment as the arrays used for planning,
    failure to do so can lead to unexpected behaviour and even python
    segfaulting.
    """
    lib.execute_dft(plan, inarray, outarray)

def destroy_plan(plan):
    """Delete the given plan"""
    lib.destroy_plan(plan)

def _create_complex2real_plan(inarray, outarray, flags):
    """Internal function to create complex fft plan given an input and output
    np array and the direction and flags integers"""
    if inarray.dtype == typedict["r"]:
        func = plan_dft_r2c
        shape = inarray.shape
    else:
        func = plan_dft_c2r
        shape = outarray.shape
    shape = (c_int*len(shape))(*shape)
    plan = func(len(shape), shape, inarray, outarray,  flags)
    if plan is None:
        raise Exception("Error creating plan %s for the given parameters" % func.__name__)
    return plan

def _create_complex_plan(inarray, outarray, direction, flags):
    """Internal function to create complex fft plan given an input and output
    np array and the direction and flags integers"""
    shape = (c_int*len(inarray.shape))(*inarray.shape)
    plan = plan_dft(len(shape), shape, inarray, outarray, direction, flags)
    if plan is None:
        raise Exception("Error creating plan %s for the given parameters" % plan_dft.__name__)
    return plan

def _create_real_plan(inarray, outarray, realtype, flags):
    """Internal function to create real fft plan given an input and output 
    np array and the realtype and flags integers"""
    if realtype == None:
        raise ValueError("Two real input arrays but no realtype list given")
    shape = (c_int*len(inarray.shape))(*inarray.shape)
    realtype = (c_int*len(realtype))(*realtype)
    plan = plan_r2r(len(shape), shape, inarray, outarray, realtype, flags)
    if plan is None:
        raise Exception("Error creating plan %s for the given parameters" % plan_r2r.__name__)
    return plan

def _create_plan(inarray, outarray, direction='forward', flags=['estimate'],
                realtypes=None, nthreads=1):
    """Internal function to create a complex fft plan given an input and output
    np array and the direction and flags integers"""
    plan_with_nthreads(nthreads)
    flags = _cal_flag_value(flags)
    if inarray.dtype == typedict["c"] and outarray.dtype == typedict["c"]:
        return _create_complex_plan(inarray,outarray, fft_direction[direction], flags)
    elif inarray.dtype == typedict["c"] or outarray.dtype == typedict["c"]:
        return _create_complex2real_plan(inarray,outarray, flags)
    elif inarray.dtype == typedict["r"] and outarray.dtype == typedict["r"]:
        return _create_real_plan(inarray,outarray, [realfft_type[r] for r in realtypes], flags)
    else:
        raise TypeError("The input or output array has a dtype which is not supported by %s: %r, %r"
            % (lib.lib.__name__, inarray.dtype, outarray.dtype))

def _cal_flag_value(flags):
    """Calculate the integer flag value from a list of string flags"""
    return sum(fftw_flags[f] for f in flags)

def print_plan(plan):
    """Print a nerd-readable version of the plan to stdout"""
    print sprintf_plan(plan)
 
def fprint_plan(plan, filename):
    """Print a nerd-readable version of the plan to the given filename"""
    with open(filename, 'w') as fp:
        fp.write(sprintf_plan(plan))

class Plan(object):
    """Object representing a fftw plan used to execute Fourier transforms in
    fftw
    
    Attributes:
        shape       --  the shape of the input and output arrays, i.e. the FFT
        flags       --  a list of the fft flags used in the planning
        direction   --  the direction of the FFT
        ndim        --  the dimensionality of the FFT
        inarray     --  the input array
        outarray    --  the output array
        """
    def __init__(self, inarray, outarray=None, direction='forward',
                 flags=['estimate'], realtypes=None,
                 nthreads = 1):
        """Initialize the fftw plan. 
        Parameters:
            inarray     --  array to be transformed (default=None)
            outarray    --  array to contain the Fourier transform
                            (default=None)
            If one of the arrays is None, the fft is considered to be
            an inplace transform.

            direction   --  direction of the Fourier transform, forward
                            or backward (default='forward')
            flags       --  list of fftw-flags to be used in planning
                            (default=['estimate'])
            realtypes   --  list of fft-types for real-to-real ffts, this
                            needs to be given if both input and output
                            arrays are real (default=None)
            nthreads    --  number of threads to be used by the plan,
                            available only for threaded libraries (default=1)
            """

        self.flags = flags
        self.direction = direction
        self.realtypes = realtypes
        self.nthreads = nthreads
        if outarray is None:
            outarray = inarray
        self.plan = _create_plan(inarray, outarray,
            direction=self.direction,
            flags=self.flags,
            realtypes=self.realtypes,
            nthreads=self.nthreads)
        self.inarray = inarray
        self.outarray = outarray

    @property
    def shape(self):
        return self.inarray.shape

    @property
    def ndim(self):
        return self.inarray.ndim

    def _get_parameter(self):
        return self.plan
    _as_parameter_ = property(_get_parameter)

    def __call__(self):
        """Perform the Fourier transform outarray = fft(inarray) for
        the arrays given at plan creation"""
        self.execute()

    def execute(self):
        """Execute the fftw plan, i.e. perform the FFT outarray = fft(inarray)
        for the arrays given at plan creation"""
        execute(self)

    def __del__(self):
        destroy_plan(self)

    def guru_execute_dft(self,inarray,outarray):
        """Guru interface: perform Fourier transform on two given arrays,
        outarray=fft(inarray). Important: This method does not perform any
        checks on the array shape and alignment for performance reasons. It is
        therefore crucial to only provide arrays with the same shape, dtype and
        alignment as the arrays used for planning, failure to do so can lead to
        unexpected behaviour and possibly python segfaulting
        """
        guru_execute_dft(self,inarray,outarray)

    def get_flops(self):
        """Return an exact count of the number of floating-point additions,
        multiplications, and fused multiply-add operations involved in
        the plan's execution. The total number of floating-point
        operations (flops) is add + mul + 2*fma, or add + mul + fma if
        the hardware supports fused multiply-add instructions
        (although the number of FMA operations is only approximate
        because of compiler voodoo).
        """
        add = c_double(0)
        mul = c_double(0)
        fma = c_double(0)
        lib.flops(self, byref (add), byref (mul), byref (fma))
        return add.value, mul.value, fma.value
        
    def __str__(self):
        return sprintf_plan(self)
