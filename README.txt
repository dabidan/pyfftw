PyFFTW-0.2

PyFFTW are python bindings for the FFTW3 (fastest Fourier transform in the West)
C-library written in python ctypes. 

Requirements:

a shared library version of libfftw3 and the python numpy package. 

Install:

To install the package type "python setup.py install" inside a terminal. 
The buildscript automatically detects if the single precision and longdouble
precision versions of the fftw3 library are installed and creates the 
appropriate python modules. The modules are called fftw3, fftw3f and
fftw3l for the double, single and longdouble precision versions of the 
library respectively. You can specify the location of the fftw3 libraries using
a FFTW_PATH environment variable, the location from this variable is however
overridden if the install script finds the location differently. However the
variable can also be used at runtime, when it will override the location of the
found library. Currently there's no support for having the ffw3l, fftw3f and
fftw3 C-libraries in different directories.

Usage:

In order to use PyFFTW you should have a basic understanding of the 
FFTW3 interface. For documentation about FFTW3 go to http://www.fftw.org
In order to achieve maximum performance FFTW3 requires a planning stage
where the actual FFT is created from an input and output array. 
To perform the FFT between the input and output array the plan is then
executed. This interface is therefore significantly different from the 
traditional A = fft(B) interface. 
In contrast to the C-library PyFFTW utilizes the Plan class for planning.
To create a fftw plan one creates a Plan object using an input and output
array, and possible parameters. PyFFTW determines from the input and output
arrays the correct plan to create. To perform the FFT one can either 
call the Plan directly or call the method execute() or pass the plan
to the execute function.

Example:

#create arrays
inputa = numpy.zeros((1024,3), dtype=complex)
outputa = numpy.zeros((1024,3), dtype=complex)

# create a forward and backward fft plan
fft = fftw3.Plan(inputa,outputa, direction='forward', flags=['measure'])
ifft = fftw3.Plan(outputa, inputa, direction='backward', flags=['measure'])

#initialize the input array
inputa[:] = 0
inputa += exp(-x**2/2)

#perform a forward transformation
fft() # alternatively fft.execute() or fftw.execute(fft)

# do some calculations with the output array
outputa *= D

#perform a backward transformation
ifft() 

The planning functions expect aligned, contiguous input arrays of any shape. 
Currently strides are not implemented. The dtype has to either be complex
or double. If you want to perform ffts on single or longdouble precision
arrays use the appropriate fftw3f or fftw3l module. FFTW overwrites the 
arrays in the planning process, thus, if you use planning strategies 
other than 'estimate' the arrays are going to be overwritten and have to 
be reinitialized. 

!IMPORTANT! 
Because the plan uses pointers to the data of the arrays you cannot perform
operations on the arrays that change the data pointer. Therefore

a = zeros(1024, dtype=complex)
p = plan(a,b)
a = a+10
p()

does not work, i.e. the a object references different memory, however the 
Fourier transform will be performed on the original memory (the plan actually 
contains a reference to the orgininal data (p.inarray), otherwise this 
operation could even result in a python segfault).

Aligned memory:

On many platforms using the SIMD units for part of the floating point
arithmetic significantly improves performance. FFTW can make use of the 
SIMD operations, however the arrays have to be specifically aligned in memory.
PyFFTW provides a function which creates an numpy array which is aligned to 
a specified boundary. In most circumstances the default alignment to 16 byte boundary is what you want. Note that the same precautions as above apply, 
i.e. creating an aligned array and then doing something like a=a+1 will
result in new memory allocated by python which might not be aligned.

PyFFTW interface naming conventions:
All exposed fftw-functions do have the same names as the C-functions with the
leading fftw_ striped from the name.

Direct access to the C-functions is available by importing lib.lib, the usual
precautions for using C-functions from Python apply. 

Advanced and Guru interface:
Currently only the execute_dft function from the fftw guru and advanced 
interface is exposed.
It is explicitly name guru_execute_dft. You should only use
these if you know what you're doing, as no checking is done on these functions.

Threads:
It is possible to specify the number of threads to use for the fft functions,
by supplying the keyword nthreads to the Plan-Class. Note that your arrays 
have to be sufficiently large to yield a speedup on multicore hardware. In
the case of small arrays the overhead for creating the threads is too large 
resulting in slower execution times. You should therefore run some test. 

Thanks:
Finally I would like to the Matteo Frigo and Stephen G. Johnson for creating 
the outstanding fftw3 library. I also would like to thank Pearu Peterson for
writing the threading support.
