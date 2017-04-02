import sys
sys.path.append('../build/lib')
sys.path.append('../build/lib.linux-x86_64-2.6')
import unittest
import time
import fftw3f
import numpy as np
import fftw3f.lib
import fftw3f.planning
import os

MAX_THREADS = 10

class ProductTestCase(unittest.TestCase):

    def test3D(self):
        repeats = 10
        shape = (256, 256, 16)
        lib = fftw3f
        tarray = np.random.random(shape).astype('f')
        farray = np.random.random(shape).astype('F')
        l = []
        for nthreads in range(1,MAX_THREADS):
            fftplan = lib.Plan(tarray,farray,'forward', nthreads=nthreads)
            ifftplan = lib.Plan(farray,tarray,'backward', nthreads=nthreads)        
            ti = time.time()
            for i in xrange(repeats):
                fftplan()
                ifftplan()
            to = time.time()-ti
            print '#threads:%s timing:%.3f speedup:%.3f'\
                % (nthreads, to, l[0]/to if l else 1)
            l.append(to)

if __name__ == '__main__':
    unittest.main()
