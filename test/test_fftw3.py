import sys
sys.path.insert(0,'../build/lib')
sys.path.insert(0,'../build/lib.linux-x86_64-2.6')
import unittest
import time
import fftw3
import fftw3f
#import fftw3l
from numpy.fft import fft,ifft,fftshift
import numpy as np
import fftw3.lib
#import fftw3l.lib
import fftw3f.lib
import fftw3.planning
#import fftw3l.planning
import fftw3f.planning
import os

h = 0.01
epsilon = 1e-1
beta = 1
N = 512
libs = [fftw3, fftw3f] #, fftw3l]

_complex = ['complex','singlecomplex','longcomplex']
_float = ['double','single','longdouble']


def fftw_propagation_aligned(N,repeats, lib, dtype):
    t = np.linspace(-5,5,N)
    dt = t[1]-t[0]
    f = np.linspace(-1/dt/2.,1/dt/2.,N)
    f = fftshift(f)
    t = fftshift(t)
    farray = lib.create_aligned_array(f.shape,dtype=dtype)
    tarray = lib.create_aligned_array(t.shape,dtype=dtype)
    fftplan = lib.Plan(tarray,farray,'forward')
    ifftplan = lib.Plan(farray,tarray,'backward')
    farray[:] = 0
    tarray[:] = 0
    tarray += np.exp(-t**2/0.5)
    dispersion = np.exp(-1.j*h*beta*f)
    ti = time.time()
    for i in xrange(repeats):
        fftplan()
        farray *= dispersion/N
        ifftplan()
    to = time.time()-ti
    return fftshift(t),fftshift(tarray),fftshift(f),fftshift(farray), to

def fftw_propagation(N,repeats,lib, dtype):
    t = np.linspace(-5,5,N)
    dt = t[1]-t[0]
    f = np.linspace(-1/dt/2.,1/dt/2.,N)
    f = fftshift(f)
    t = fftshift(t)
    farray = np.zeros(f.shape,dtype=dtype)
    tarray = np.zeros(t.shape,dtype=dtype)
    fftplan = lib.Plan(tarray,farray,'forward')
    ifftplan = lib.Plan(farray,tarray,'backward')
    farray[:] = 0
    tarray[:] = 0
    tarray += np.exp(-t**2/0.5)
    dispersion = np.exp(-1.j*h*beta*f)
    ti = time.time()
    for i in xrange(repeats):
        fftplan()
        farray *= dispersion/N
        ifftplan()
    to = time.time()-ti
    return fftshift(t),fftshift(tarray),fftshift(f),fftshift(farray), to


def np_propagation(N,repeats,dtype):
    t = np.linspace(-5,5,N)
    dt = t[1]-t[0]
    f = np.linspace(-1/dt/2.,1/dt/2.,N)
    f = fftshift(f)
    t = fftshift(t)
    tarray = np.zeros(t.shape,dtype)
    tarray += np.exp(-t**2/0.5)
    farray = np.zeros(tarray.shape,dtype)
    dispersion = np.zeros(tarray.shape,dtype)
    dispersion += np.exp(-1.j*h*beta*f)
    ti = time.time()
    for i in xrange(repeats):
        farray = fft(tarray)
        farray *= dispersion
        tarray = ifft(farray)
    to = time.time()-ti
    return fftshift(t),fftshift(tarray),fftshift(f),fftshift(farray),to


def scipy_propagation(N,repeats, dtype):
    try:
        from scipy.fftpack import fft as sfft, ifft as sifft
    except:
        return None, None, None, None, np.NaN
    t = np.linspace(-5,5,N)
    dt = t[1]-t[0]
    f = np.linspace(-1/dt/2.,1/dt/2.,N)
    f = fftshift(f)
    t = fftshift(t)
    tarray = np.zeros(t.shape,dtype)
    tarray += np.exp(-t**2/0.5)
    farray = np.zeros(tarray.shape,dtype)
    dispersion = np.zeros(tarray.shape,dtype)
    dispersion += np.exp(-1.j*h*beta*f)
    ti = time.time()
    for i in xrange(repeats):
        farray = sfft(tarray)
        farray *= dispersion
        tarray = sifft(tarray)
    to = time.time()-ti
    return fftshift(t),fftshift(tarray),fftshift(f),fftshift(farray),to


class ProductTestCase(unittest.TestCase):
    def testWisdom(self):
        for i, lib in enumerate(libs):
            lib.forget_wisdom()
            inputa = lib.create_aligned_array(1024,np.typeDict[_complex[i]])
            outputa = lib.create_aligned_array(1024,np.typeDict[_complex[i]])
            plan = lib.Plan(inputa,outputa,flags=['patient'])
            soriginal = lib.export_wisdom_to_string()
            lib.import_wisdom_from_string(soriginal)
            lib.export_wisdom_to_file('test.wisdom')
            lib.forget_wisdom()
            lib.import_wisdom_from_file('test.wisdom')
            os.remove('test.wisdom')

    def testPropagation(self):
        #can only test for fftw3 because longdouble and single are not both implemented for scipy.fft and numpy.fft
        Ns = [2**i for i in range(10,15)]
        repeats = 2000
        epsilon = 1e-3
        times = []
        for Nn in Ns:
            t,A,f,B, ti = fftw_propagation(Nn,repeats, fftw3, _complex[0])
            ft,fA,ff,fB, fti = fftw_propagation_aligned(Nn,repeats, fftw3, _complex[0])
            nt,nA, nf, nB, nti = np_propagation(Nn,repeats, _complex[0])
            st,sA, sf, sB, sti = scipy_propagation(Nn,repeats, _complex[0])
            times.append((ti, fti,nti, sti))
            self.failUnless(sum(abs(A)**2-abs(nA)**2)< epsilon,
                "Propagation of fftw3 and numpy gives different results")
            self.failUnless(sum(abs(fA)**2-abs(nA)**2)< epsilon,
                "Propagation of aligned fftw3 and numpy gives different results")
        print     "Benchmark: %s" %fftw3
        print     "N   fftw3  fftw3_aligned   numpy   scipy"
        for n, t in zip(Ns, times):
            print "%5d  %5.2f    %5.2f      %5.2f   %5.2f" % (n, t[0], t[1], t[2],t [3])
                
    def test2D(self):
        try:
            from pylab import imread
        except:
            return 
        im = imread('Fourier2.png')
        im = im[:,:,1]
        a = np.zeros(im.shape, dtype=im.dtype)
        a[:] = im[:]
        b = np.zeros((im.shape[0],im.shape[1]/2+1),dtype=np.typeDict['singlecomplex'])
        p = fftw3f.Plan(a,b,'forward')
        ip = fftw3f.Plan(b,a,'backward')
        p()
        b/=np.prod(a.shape)
        ip()
        self.failUnless(a.sum()-im.sum() < epsilon, "2D fft and ifft did not "\
                                                  "reproduce the same image")



if __name__ == '__main__': unittest.main()
