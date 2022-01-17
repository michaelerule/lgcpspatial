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
----------------------------------------------------------------------------
Routines for Gaussian blurs, FFT-based convolutions and autocorrelation, 
constructing radially-symmetric kernels, and common types of linear operators.
"""

# Load a Matlab-like namespace
from pylab import *
#import numpy as np
#import pylab as pl

# We keep a separate handle specifically to numpy. This allows us to (if
# desired) replace numpy/pylab with the Jax namespace, but retain access to some
# true numpy functions as needed. This is mainly needed to create mutable
# arrays, which we don't need to differentiate. The ability to assign to array 
# locations creates more readable initialization code, but isn't supported by
# the Jax array type. 
import numpy as np0

# Used to locate peaks in the autocorrelogram
from scipy.signal import find_peaks

# Used to explicitly construct covariance matrix from a convolution kernel
# (Generally we shouldn't be doing this for large problems; it's only used to
# verify code on smaller problems as a sanity check)
from scipy.linalg import circulant 

# Used to make radially symmetric 2D kernel from 1D radial kernel
from scipy.interpolate import interp1d

# Allows us to work with linear operators without constructing them as matrices
# This is very useful for large covariance priors, which may not fit in memory,
# but can be calculated easily as convolution kernels using the FFT. 
from scipy.sparse.linalg import minres,LinearOperator

# Common helper functions
from util import *

def blurkernel(L,σ,normalize=False):
    '''
    ----------------------------------------------------------------------------
    1D Gaussian blur convolution kernel    
    
    Parameters
    ----------
    L: Size of L×L spatial domain
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    normalize: boolean; whether to make kernel sum to 1
    '''
    k = exp(-(arange(-L//2,L//2)/σ)**2)
    if normalize: 
        k /= sum(k)
    return fftshift(k)

def blurkernel2D(L,σ,normalize=False):
    '''
    ----------------------------------------------------------------------------
    2D Gaussian blur convolution kernel    
    
    Parameters
    ----------
    L: Size of L×L spatial domain
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    normalize: boolean; whether to make kernel sum to 1
    '''
    k = blurkernel(L,σ)
    k = outer(k,k)
    if normalize: 
        k /= sum(k)
    return k

def conv(x,K):
    '''
    ----------------------------------------------------------------------------
    Compute circular 2D convolution using FFT
    Kernel K should already be fourier-transformed
    
    Parameters
    ----------
    x: 2D array
    K: Fourier-transformed convolution kernel
    '''
    return ifft2(fft2(x.reshape(K.shape))*K).real

def blur(x,σ,**kwargs):
    '''
    ----------------------------------------------------------------------------
    2D Gaussian blur via fft
    
    Parameters
    ----------
    x: 2D array
    σ: kernel radius exp(-x²/σ) (standard deviation in x and y ×⎷2)
    '''
    kern = fft(blurkernel(x.shape[0],σ,**kwargs))
    return conv(x,outer(kern,kern))

def zeromean(x, mask):
    '''
    ----------------------------------------------------------------------------
    Mean-center data, accounting for masked-out regions
    
    Parameters
    ----------
    x: L×L 2D array
    mask: L×L binary mask of "in bounds" regions 
    '''
    x = x.reshape(mask.shape)
    return (x-mean(x[mask]))*mask

def fft_acorr(x,mask):
    '''
    ----------------------------------------------------------------------------
    Zero-lag normalized to match signal variance
    
    Parameters
    ----------
    x: L×L 2D array
    mask: L×L binary mask of "in bounds" regions 
    '''
    x   = zeromean(x,mask)
    # Window attenuates boundary artefacts
    L = x.shape[0]
    win = hanning(L)
    win = outer(win,win)
    # Calculate autocorrelation using FFT
    psd = (abs(fft2(x*win))/L)**2
    acr = fftshift(ifft2(psd).real)
    # Adjust peak for effects of mask, window
    return acr*var(x[mask])/acr[L//2,L//2]

def radial_average(y):
    '''
    ----------------------------------------------------------------------------
    Get radial autocorrelation by averaging 2D autocorrelogram
    '''
    L = y.shape[0]
    coords = zgrid(L)
    i = int32(abs(coords)) # Radial distance
    a = array([mean(y[i==j]) for j in range(L//2+1)])
    return concatenate([a[::-1],a[1:-1]])

def radial_acorr(y,mask):
    '''
    ----------------------------------------------------------------------------
    Autocorrelation as a function of distance
    '''
    return radial_average(fft_acorr(y,mask))

def fft_upsample_1D(x,factor=4):
    '''
    ----------------------------------------------------------------------------
    Upsample 1D array using the FFT
    '''
    n  = len(x)
    n2 = n*factor
    f  = fftshift(fft(x))*hanning(n)
    f2 = np0.complex128(np.zeros(n2))
    r0 = (n2+1)//2-(n+0)//2
    f2[r0:r0+n] = f
    return np.real(ifft(fftshift(f2)))*factor

def acorr_peak(r,F=6):
    '''
    ----------------------------------------------------------------------------
    sinc upsample at ×F resolution to get distance to first peak
    '''
    r2 = fft_upsample_1D(r,F)
    return min(find_peaks(r2[len(r2)//2:])[0])/F,r2

def kernel_to_covariance(kern):
    '''
    ----------------------------------------------------------------------------
    Explicitly construct covariance matrix from a convolution kernel (Generally 
    we shouldn't be doing this for large problems; it's only used to verify code
    on smaller problems as a sanity check)
    
    Covariance is a doubly block-circulant matrix Use np.circulant to build 
    blocks, then copy with shift to make 2D block-circulant matrix.
    '''
    assert(argmax(kern.ravel())==0)
    L = kern.shape[0]
    b = array([circulant(r) for r in kern])
    b = b.reshape(L**2,L).T
    s = array([roll(b,i*L,1) for i in range(L)])
    return s.reshape(L**2,L**2)

def repair_small_eigenvalues(kern,mineig=1e-6):
    '''
    ----------------------------------------------------------------------------
    Kernel must be positive; fix small eigenvalues
    '''
    assert(argmax(kern.ravel())==0)
    kfft = np0.array(fft2(kern))
    keig = abs(kfft)
    υmin = mineig*np.max(keig)
    zero = keig<υmin
    kfft[zero] = υmin
    kern = real(ifft2(maximum(υmin,kfft)))
    return kern

def solveGP(kern,y,τe,mask,tol=1e-4,reg=1e-5):
    '''
    ----------------------------------------------------------------------------
    Minimum residual solver is fast
    '''
    L = kern.shape[0]
    kern = repair_small_eigenvalues(kern,reg)
    knft = fft2(kern)
    τy   = τe*zeromean(y,mask).ravel()
    Στy  = conv(τy,knft).ravel()
    Hv   = lambda v:conv(τe*v,knft).ravel() + v
    ΣτεI = LinearOperator((L**2,L**2),Hv,Hv,dtype=np.float64)
    μ    = minres(ΣτεI,Στy,tol=tol)[0]
    return μ.reshape(L,L) + mean(y.ravel()[mask.ravel()])

def mirrorpad(y,pad):
    '''
    ----------------------------------------------------------------------------
    Reflected boundary for convolution
    '''
    y = np0.array(y)
    y[:pad, :]=flipud(y[ pad: pad*2,:])
    y[:, :pad]=fliplr(y[:, pad: pad*2])
    y[-pad:,:]=flipud(y[-pad*2:-pad,:])
    y[:,-pad:]=fliplr(y[:,-pad*2:-pad])
    return y

def radial_kernel(rk):
    '''
    ----------------------------------------------------------------------------
    Make radially symmetric 2D kernel from 1D radial kernel
    '''
    L = rk.shape[0]
    coords = zgrid(L)
    r    = abs(coords)
    kern = interp1d(arange(L//2),rk[L//2:],fill_value=0,bounds_error=0)(r)
    return fftshift(kern)

def zerolag(ac,r=3):
    '''
    ----------------------------------------------------------------------------
    Estimate true zero-lag variance via quadratic interpolation.
    '''
    L = ac.shape[0]
    z = array(ac[L//2-r:L//2+r+1])
    v = float32(arange(r*2+1))
    return polyfit(v[v!=r],z[v!=r],2)@array([r**2,r,1])

def op(M,Av):
    '''
    ----------------------------------------------------------------------------
    Construct a symmetric linear operator from a function which computes the
    product of said operator with a vector. 
    
    Parameters
    ----------
    M: dimension of operator
    Av: f:R^M→R^M: linear operator acting on length M vectors
    '''
    return LinearOperator((M,)*2,Av,Av,dtype=np.float64)

def cop(k):
    '''
    ----------------------------------------------------------------------------
    Construct a convolution operator
    
    Parameters
    ----------
    k: array; convolution kernel Fourier transform. 
    '''
    M = prod(k.shape)
    return op(M, lambda v:conv(v,k).ravel())

def diagop(d):
    '''
    ----------------------------------------------------------------------------
    Construct a diagonal operator
    
    Parameters
    ----------
    d: vector; diagonal of matrix operator
    '''
    d = d.ravel()
    M = d.shape[0]
    return op(M, lambda v:v.ravel()*d)

