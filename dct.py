#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
"""
--------------------------------------------------------------------------------
This is a modified implementation of a discrete cosine transform, which is built
on top of the fast fourier transform. Unlike the other DCT implementations in 
numpy, this appraoch has the following useful properties: 

 - This implmementation can be used directy to evaluate convolution with 
   reflected boundary conditions via pointwise multiplication (convolution 
   theorem)
 - It's uses the FFT of real symmetric data, so the data packing and 
   interpretation of the coefficient matrix is the same as the FFT
 - Unlike the FFT, eigenvalues are real, so they can be used directly with 
   linear algebra routines that require real-valued input
"""

# Load a Matlab-like namespace
from pylab import *

def mirror(x):
    '''
    ----------------------------------------------------------------------------
    Mirror LxL data up to 2L+1 x 2L+1
    '''
    x = x.reshape(L,L)
    return block([[x,fliplr(x[:,1:])],[flipud(x[1:,:]),fliplr(flipud(x[1:,1:]))]])

def padout(kern):
    '''
    ----------------------------------------------------------------------------
    Zero-pad LxL kernel up to 2L+1 x 2L+1
    '''
    k2 = zeros((L*2-1,L*2-1))
    k2[L//2:L//2+L,L//2:L//2+L] = fftshift(kern)
    return fftshift(k2)

def dct2v(x):
    '''
    ----------------------------------------------------------------------------
    DCT Option 1: reflect data to create symmetry
    '''
    return real(fft2(mirror(x)))[:L,:L]/(L*2-1)

def dct2k(k):
    '''
    ----------------------------------------------------------------------------
    DCT Option 2: if kernel already symmetric, zero pad
    '''
    return real(fft2(padout(k)))[:L,:L]/(L*2-1)

def idct2(x):
    '''
    ----------------------------------------------------------------------------
    Inverse DCT
    '''
    return real(fft2(mirror(x)))[:L,:L]/(L*2-1)

def collapse(v):
    '''
    ----------------------------------------------------------------------------
    Return dct2v(v)[keep].ravel()
    '''
    if all(v==0): return zeros(M)
    v = v.reshape(L,L)
    for i in range(2):
        v = block([v,fliplr(v[:,1:])])
        v = real(fft(v)).T[:L][keep1d]
    return v.ravel()[keeprc]/(L*2-1)

def collapse2(v):
    '''
    ----------------------------------------------------------------------------
    Collapse vector to subspace
    '''
    return dct2v(v)[keep].ravel()

def expand(u):
    '''
    ----------------------------------------------------------------------------
    Expand vector from subspace
    '''
    return idct2(down.T@u.ravel()).ravel()

def collapseA(A):
    '''
    ----------------------------------------------------------------------------
    Collapse L²×L² matrix to subspace
    '''
    A = A.reshape(L*L,L,L)
    A = array([collapse(a) for a in A]).T
    A = A.reshape(M,L,L)
    A = array([collapse(a) for a in A]).T
    return A

def expandAleft(A):
    '''
    ----------------------------------------------------------------------------
    Expand matrix on left size from subspace
    Expand compressed representation on the left
    A is MxM, we return NxM, N=L*L
    '''
    return array([expand(a) for a in A.T]).T